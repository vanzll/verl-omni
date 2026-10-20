import ast
import asyncio
from datetime import timedelta
import json
import logging
import os
from pathlib import Path
import socket
import sys
import sysconfig
import time
import traceback
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


ROOT = Path(__file__).resolve().parents[3]
NATIVE = Path(sysconfig.get_paths()["purelib"]) / "vllm_omni"
LOGGER = logging.getLogger("full32_cpu_rpc_contract")
LOGGER.addHandler(logging.NullHandler())


def definitions(path, names, namespace, class_name=None, parent=None):
    tree = ast.parse(path.read_text())
    methods = []
    for name in names:
        node = next(node for node in ast.walk(tree)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name)
        node.decorator_list = [decorator for decorator in node.decorator_list
                               if isinstance(decorator, ast.Name) and decorator.id == "staticmethod"]
        methods.append(node)
    body = methods
    if class_name is not None:
        bases = [ast.Name(id="Parent", ctx=ast.Load())] if parent is not None else []
        namespace["Parent"] = parent
        body = [ast.ClassDef(name=class_name, bases=bases, keywords=[], body=methods, decorator_list=[])]
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                              *body], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[class_name] if class_name else namespace


def envelope_type():
    candidates = [NATIVE / "diffusion/ipc.py", NATIVE / "diffusion/ipc/__init__.py"]
    path = next(path for path in candidates if path.exists())
    tree = ast.parse(path.read_text())
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        if any(isinstance(target, ast.Name) and target.id == "DIFFUSION_RPC_RESULT_ENVELOPE" for target in targets):
            return ast.literal_eval(node.value)
    raise RuntimeError("Installed native RPC envelope identity was not found.")


def run_rank(rank, rendezvous):
    assert not torch.cuda.is_initialized()
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank,
                            world_size=2, timeout=timedelta(seconds=90))
    try:
        namespace = {"torch": torch, "dist": dist, "os": os, "socket": socket,
                     "logger": LOGGER, "time": time, "asyncio": asyncio, "traceback": traceback,
                     "get_world_group": lambda: SimpleNamespace(cpu_group=dist.group.WORLD),
                     "DIFFUSION_RPC_RESULT_ENVELOPE": envelope_type(),
                     "_MEMORY_RELEASING_METHODS": frozenset(),
                     "_cleanup_after_execution_error": lambda error: None,
                     "unpack_diffusion_output_shm": lambda response: None}
        native_worker_path = NATIVE / "diffusion/worker/diffusion_worker.py"
        definitions(native_worker_path, ["_all_gather_rank_values", "_run_and_gather_rank_values"], namespace)
        sys.modules["vllm_omni.diffusion.worker.diffusion_worker"] = SimpleNamespace(
            _run_and_gather_rank_values=namespace["_run_and_gather_rank_values"])
        extension_class = definitions(ROOT / "verl_omni/workers/rollout/vllm_rollout/utils.py",
                                      ["_full_local_policy_sync_state", "get_full_policy_sync_state"],
                                      namespace, "FullExtension")
        native_class = definitions(native_worker_path, ["_execute_rpc", "_gather_rpc_rank_statuses"],
                                   namespace, "NativeWorker")
        executor_class = definitions(NATIVE / "diffusion/executor/multiproc_executor.py",
                                     ["collective_rpc", "_handle_rpc_response", "_unwrap_rpc_result_envelope",
                                      "_raise_for_rpc_error_dict"], namespace, "MultiprocDiffusionExecutor")
        process_class = definitions(NATIVE / "diffusion/stage_diffusion_proc.py", ["_handle_collective_rpc"],
                                    namespace, "StageProcess")
        namespace["StagePool"] = SimpleNamespace(_CACHE_RESET_METHODS=frozenset())
        namespace["CollectiveRPCResultMessage"] = SimpleNamespace
        orchestrator_class = definitions(NATIVE / "engine/orchestrator.py", ["_handle_collective_rpc"],
                                         namespace, "Orchestrator")
        client_class = definitions(NATIVE / "entrypoints/async_omni.py", ["collective_rpc"], namespace, "Client")
        definitions(ROOT / "verl_omni/workers/rollout/vllm_rollout/full_weight_sync.py",
                    ["unpack_full_policy_sync_replies"], namespace)
        sync = SimpleNamespace(unpack_full_policy_sync_replies=namespace["unpack_full_policy_sync_replies"])
        sys.modules["verl_omni.workers.rollout.vllm_rollout.full_weight_sync"] = sync
        clock_names = ["full_policy_sync_tag", "full_policy_sync_clock", "validate_full_policy_sync_clock"]
        definitions(ROOT / "verl_omni/utils/full_policy_sync.py", clock_names, namespace)
        sys.modules["verl_omni.utils.full_policy_sync"] = SimpleNamespace(
            **{name: namespace[name] for name in clock_names})

        class Parent:
            async def set_global_steps(self, value):
                self.global_steps = value

        namespace["get_rollout_world_size"] = lambda config: 2
        server_class = definitions(ROOT / "verl_omni/workers/rollout/vllm_rollout/vllm_omni_async_server.py",
                                   ["set_global_steps"], namespace, "Server", Parent)
        os.environ["OMNI_FULL_FINETUNE"] = "1"
        os.environ["OMNI_VARIANT"] = "ART_FULL32"
        outcomes = []
        for scenario in ("old_rank_local_reply", "healthy_gathered_reply", "peer_not_ready", "peer_wrong_version",
                         "duplicate_physical_identity", "peer_local_exception"):
            os.environ["CUDA_VISIBLE_DEVICES"] = f"cpu-only-{rank}"
            if scenario == "duplicate_physical_identity":
                os.environ["CUDA_VISIBLE_DEVICES"] = "cpu-only-0"
            if scenario == "peer_local_exception" and rank == 1:
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
            extension = extension_class()
            extension.device = SimpleNamespace(index=0)
            extension._full_weight_sync_valid = not (scenario == "peer_not_ready" and rank == 1)
            version = 0 if scenario == "peer_wrong_version" and rank == 1 else 2
            extension._full_committed_state = {
                "policy_algorithm": "diffusion_art", "optimizer_updates_per_rollout": 2,
                "policy_version": version, "global_steps": version // 2, "post_load_checks": 4186,
                "buckets": 19, "source_tensors": {"cpu-fixture": {}}, "loaded_targets": ["cpu-fixture"],
                "checked_kinds": ["packed_q", "packed_k", "packed_v"]}
            extension._full_initial_probe = {}
            if scenario == "old_rank_local_reply":
                extension.get_full_policy_sync_state = extension._full_local_policy_sync_state
            native = native_class()
            native.gpu_id = rank
            native.result_mq = object()
            native.worker = SimpleNamespace(execute_method=lambda method, *args, **kwargs:
                                            getattr(extension, method)(*args, **kwargs))
            response, should_reply = native._execute_rpc({
                "method": "get_full_policy_sync_state", "args": (), "kwargs": {}, "wave_id": 1,
                "output_rank": 0, "exec_all_ranks": True, "collect_rank_status": True})
            assert should_reply == (rank == 0)
            if rank == 0:
                executor = executor_class()
                executor._ensure_open = lambda: None
                executor._rpc_wave_id = 0
                executor.od_config = SimpleNamespace(step_execution=True,
                                                      parallel_config=SimpleNamespace(data_parallel_size=1))
                requests = []
                executor._broadcast_mq = SimpleNamespace(enqueue=requests.append)
                executor._dequeue_one_with_failure_polling = lambda *args: response
                executor._validate_wave_id = lambda received, *args: received
                process = process_class()
                process._executor = None
                process._engine = executor

                async def stage_rpc(**kwargs):
                    return await process._handle_collective_rpc("get_full_policy_sync_state", 5, (), {})

                async def route(**kwargs):
                    pool = SimpleNamespace(stage_id=0, stage_type="diffusion", live_replica_ids=lambda: [0],
                                           collective_rpc=stage_rpc)
                    orchestrator = orchestrator_class()
                    orchestrator.stage_pools = [pool]
                    orchestrator.num_stages = 1
                    orchestrator.rpc_async_queue = asyncio.Queue()
                    message = SimpleNamespace(rpc_id="cpu-contract", method="get_full_policy_sync_state", timeout=5,
                                              args=(), kwargs={}, stage_ids=[0])
                    await orchestrator._handle_collective_rpc(message)
                    return orchestrator.rpc_async_queue.get_nowait().results

                client = client_class()
                client.engine = SimpleNamespace(collective_rpc_async=route)
                server = server_class()
                server.global_steps = 0
                server.config = SimpleNamespace()
                server._invalidate_lora_request_cache = lambda: None
                server.collective_rpc = client.collective_rpc
                success = False
                try:
                    asyncio.run(server.set_global_steps(1))
                    success = True
                except RuntimeError:
                    pass
                assert success == (scenario == "healthy_gathered_reply"), scenario
                assert server.global_steps == (1 if success else 0), scenario
                assert requests and requests[0]["collect_rank_status"] and requests[0]["exec_all_ranks"]
                assert requests[0]["output_rank"] == 0
                outcomes.append({"scenario": scenario, "committed": success})
            dist.barrier()
        assert not torch.cuda.is_initialized()
        print(json.dumps({"rank": rank, "status": "passed_native_cpu_rpc_contract", "outcomes": outcomes,
                          "cuda_initialized": torch.cuda.is_initialized(),
                          "scope": "Actual installed worker/executor/stage/orchestrator/AsyncOmni method bodies with real two-rank Gloo; in-process queue/SHM/GPU-cleanup substitutes, no Ray/GPU/native-engine E2E claim."}),
              flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    directory = Path(os.environ["FULL32_CPU_IPC_ROOT"])
    directory.mkdir(parents=True, exist_ok=True)
    rendezvous = directory / f"rpc-contract-{os.getpid()}-{time.time_ns()}"
    mp.spawn(run_rank, args=(str(rendezvous),), nprocs=2, join=True)
