import ast
import asyncio
import importlib.util
import logging
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[3]
AUDIT_SPEC = importlib.util.spec_from_file_location("tensor_audit_cpu_test", ROOT / "verl_omni/utils/tensor_audit.py")
AUDIT = importlib.util.module_from_spec(AUDIT_SPEC)
AUDIT_SPEC.loader.exec_module(AUDIT)
SPEC = importlib.util.spec_from_file_location(
    "full_sync_cpu_test", ROOT / "verl_omni/workers/rollout/vllm_rollout/full_weight_sync.py")
SYNC = importlib.util.module_from_spec(SPEC)
with patch.dict(sys.modules, {"verl_omni.utils.tensor_audit": AUDIT}):
    SPEC.loader.exec_module(SYNC)
FullWeightCoverage = SYNC.FullWeightCoverage


class TinyNativeTransformer(torch.nn.Module):
    stacked_params_mapping = (("qkv", "query", "q"), ("qkv", "key", "k"), ("qkv", "value", "v"))

    def __init__(self, acknowledge_only=False, skip_qkv=False, tp_rank=0, tp_size=1):
        super().__init__()
        self.qkv = torch.nn.Parameter(torch.zeros(6 // tp_size, 2))
        self.dense = torch.nn.Parameter(torch.zeros(2, 2 // tp_size))
        self.acknowledge_only = acknowledge_only
        self.skip_qkv = skip_qkv
        self.tp_rank = tp_rank
        self.tp_size = tp_size

    def _get_shard_offset_mapping(self, part):
        return {"q": 0, "k": 2, "v": 4}[part] // self.tp_size

    def _get_shard_size_mapping(self, part):
        return 2 // self.tp_size

    def load_weights(self, weights):
        loaded = set()
        with torch.no_grad():
            for name, tensor in weights:
                part = next((part for _, source, part in self.stacked_params_mapping if name == source), None)
                if part is not None:
                    offset = self._get_shard_offset_mapping(part)
                    size = self._get_shard_size_mapping(part)
                    if not self.acknowledge_only and not self.skip_qkv:
                        self.qkv[offset:offset + size].copy_(tensor.chunk(self.tp_size, dim=0)[self.tp_rank])
                    loaded.add("qkv")
                elif name == "dense":
                    if not self.acknowledge_only:
                        self.dense.copy_(tensor.chunk(self.tp_size, dim=1)[self.tp_rank])
                    loaded.add("dense")
        return loaded


class FullWeightSyncTests(unittest.TestCase):
    def setUp(self):
        self.weights = {f"transformer.{name}": torch.full((2, 2), float(index + 1))
                        for index, name in enumerate(("query", "key", "value", "dense"))}
        self.metadata = {"policy_version": 2, "global_steps": 1,
                         "policy_algorithm": "diffusion_art", "optimizer_updates_per_rollout": 2,
                         "tensors": {name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
                                     for name, tensor in self.weights.items()},
                         "sample_names": list(self.weights)}

    def test_native_complete_stream_maps_qkv_and_commits_exact_version(self):
        model = TinyNativeTransformer()
        audit = FullWeightCoverage(model, self.metadata)
        for item in self.weights.items():
            audit.load_bucket([item])
        state = audit.finish()
        self.assertEqual(state["policy_version"], 2)
        self.assertEqual(state["loaded_targets"], ["dense", "qkv"])
        self.assertEqual(state["qkv_parts"], {"qkv": ["k", "q", "v"]})
        self.assertEqual(state["maximum_bucket_bytes"], 16)
        torch.testing.assert_close(model.qkv, torch.cat(list(self.weights.values())[:3]))

    def test_duplicate_unknown_shape_and_dtype_fail_closed(self):
        first = next(iter(self.weights.items()))
        audit = FullWeightCoverage(TinyNativeTransformer(), self.metadata)
        audit.load_bucket([first])
        with self.assertRaises(ValueError):
            audit.load_bucket([first])
        for name, tensor in (("transformer.missing", first[1]),
                             (first[0], torch.ones(3, 2)), (first[0], first[1].to(torch.bfloat16))):
            audit = FullWeightCoverage(TinyNativeTransformer(), self.metadata)
            with self.assertRaises(ValueError):
                audit.load_bucket([(name, tensor)])

    def test_partial_qkv_and_missing_parameter_do_not_commit(self):
        audit = FullWeightCoverage(TinyNativeTransformer(), self.metadata)
        audit.load_bucket(list(self.weights.items())[:2])
        with self.assertRaises(RuntimeError):
            audit.finish()
        metadata = dict(self.metadata, tensors={name: value for name, value in self.metadata["tensors"].items()
                                               if name != "transformer.value"})
        audit = FullWeightCoverage(TinyNativeTransformer(), metadata)
        audit.load_bucket([(name, tensor) for name, tensor in self.weights.items() if name in metadata["tensors"]])
        with self.assertRaisesRegex(RuntimeError, "QKV"):
            audit.finish()

    def test_acknowledgement_without_copy_is_not_success(self):
        audit = FullWeightCoverage(TinyNativeTransformer(acknowledge_only=True), self.metadata)
        with self.assertRaisesRegex(RuntimeError, "did not copy"):
            audit.load_bucket(list(self.weights.items()))

    def test_qkv_acknowledgement_without_copy_is_not_success_even_with_dense_copy(self):
        audit = FullWeightCoverage(TinyNativeTransformer(skip_qkv=True), self.metadata)
        with self.assertRaisesRegex(RuntimeError, "did not copy"):
            audit.load_bucket(list(self.weights.items()))

    def test_tp2_qkv_and_row_slices_match_their_own_rank(self):
        weights = {name: torch.arange(4, dtype=torch.float32).reshape(2, 2) + 10 * index
                   for index, name in enumerate(self.weights)}
        for rank in (0, 1):
            model = TinyNativeTransformer(tp_rank=rank, tp_size=2)
            audit = FullWeightCoverage(model, self.metadata)
            audit.load_bucket(list(weights.items()))
            state = audit.finish()
            self.assertEqual(state["post_load_checks"], 4)
            self.assertEqual(state["checked_kinds"], ["packed_k", "packed_q", "packed_v", "tp_dimension_1"])
            torch.testing.assert_close(model.dense, weights["transformer.dense"][:, rank:rank + 1])

    def test_tp_norm_without_module_rank_uses_native_group_rank(self):
        class Norm(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.tp_size = 2
                self.weight = torch.nn.Parameter(torch.zeros(2))

        class Model(torch.nn.Module):
            stacked_params_mapping = ()

            def __init__(self):
                super().__init__()
                self.norm = Norm()

            def load_weights(self, weights):
                with torch.no_grad():
                    for _, tensor in weights:
                        self.norm.weight.copy_(tensor[2:])
                return {"norm.weight"}

        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
        name = "transformer.norm.weight"
        metadata = {"policy_version": 2, "global_steps": 1, "tensor_parallel_rank": 1, "tensor_parallel_size": 2,
                    "tensors": {name: {"shape": [4], "dtype": "torch.float32"}}, "sample_names": [name]}
        audit = FullWeightCoverage(Model(), metadata)
        audit.load_bucket([(name, tensor)])
        self.assertEqual(audit.finish()["post_load_samples"][name]["actual"]["values"], [3.0, 4.0])

    def test_server_clock_waits_for_every_tp_rank(self):
        path = ROOT / "verl_omni/workers/rollout/vllm_rollout/vllm_omni_async_server.py"
        tree = ast.parse(path.read_text())
        method = next(node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef)
                      and node.name == "set_global_steps")
        namespace = {"os": os, "get_rollout_world_size": lambda config: 2}

        class Parent:
            async def set_global_steps(self, value):
                self.global_steps = value

        wrapper = ast.ClassDef(name="Server", bases=[ast.Name(id="Parent", ctx=ast.Load())],
                               keywords=[], body=[method], decorator_list=[])
        namespace["Parent"] = Parent
        exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])), str(path), "exec"), namespace)
        server = namespace["Server"]()
        server.global_steps = 0
        server.config = SimpleNamespace()
        server._invalidate_lora_request_cache = lambda: None

        async def rpc(**kwargs):
            return server.replies

        server.collective_rpc = rpc
        good = {"ready": True, "hostname": "host", "physical_gpu": "0", "native_rank": 0,
                "native_world_size": 2, "state": {"policy_version": 2, "global_steps": 1,
                "policy_algorithm": "diffusion_art", "optimizer_updates_per_rollout": 2}}
        peer = dict(good, physical_gpu="1", native_rank=1)
        with patch.dict(os.environ, {"OMNI_FULL_FINETUNE": "1", "OMNI_VARIANT": "ART_FULL32"}), patch.dict(sys.modules, {
            "verl_omni.workers.rollout.vllm_rollout.full_weight_sync": SYNC,
        }):
            for replies in ([good], [[good]], [good, good], [[good, peer]], [[[good]]],
                            [[[good, good]]], [[[good, dict(peer, physical_gpu="0")]]],
                            [[[good, dict(peer, native_rank=2)]]],
                            [[[good, dict(peer, native_world_size=1)]]],
                            [[[good, dict(peer, ready=False)]]],
                            [[[good, dict(peer, ready="true")]]],
                            [[[good, dict(peer, supported=False)]]],
                            [[[good, dict(peer, state={"policy_version": 0, "global_steps": 0})]]],
                            [[[good, dict(peer, error="failed")]]]):
                server.replies = replies
                with self.assertRaises(RuntimeError):
                    asyncio.run(server.set_global_steps(1))
                self.assertEqual(server.global_steps, 0)
            server.replies = [[[good, peer]]]
            asyncio.run(server.set_global_steps(1))
            self.assertEqual(server.global_steps, 1)
            self.assertEqual(server._full_committed_policy_version, 2)
        with patch.dict(os.environ, {"OMNI_FULL_FINETUNE": "0"}):
            server.replies = None
            asyncio.run(server.set_global_steps(2))
            self.assertEqual(server.global_steps, 2)

    def test_full_gate_worker_normalizes_actual_rpc_layers_and_preserves_nonleader_none(self):
        path = ROOT / "verl_omni/workers/engine_workers.py"
        tree = ast.parse(path.read_text())
        method = next(node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef)
                      and node.name == "get_full32_policy_state")
        method.decorator_list = []
        namespace = {"get_rollout_world_size": lambda config: 2}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
        worker = SimpleNamespace(role=["actor"], actor=SimpleNamespace(engine=SimpleNamespace(
            full_policy_state=lambda: {"rank": 0})), config=SimpleNamespace(rollout=SimpleNamespace()))

        async def execute(name):
            self.assertEqual(name, "get_full_policy_sync_state")
            return worker.replies

        worker.rollout = SimpleNamespace(_execute_method=execute)
        ranks = [{"native_rank": rank, "native_world_size": 2, "hostname": "cpu-fixture",
                  "physical_gpu": str(rank)} for rank in range(2)]
        with patch.dict(sys.modules, {"verl_omni.workers.rollout.vllm_rollout.full_weight_sync": SYNC}):
            worker.replies = [[ranks]]
            result = asyncio.run(namespace[method.name](worker))
            self.assertEqual(result["rollout"], [ranks])
            worker.replies = None
            self.assertIsNone(asyncio.run(namespace[method.name](worker))["rollout"])
            worker.replies = [ranks]
            with self.assertRaises(RuntimeError):
                asyncio.run(namespace[method.name](worker))

    def test_actual_receiver_callback_and_single_use_metadata(self):
        path = ROOT / "verl_omni/workers/rollout/vllm_rollout/utils.py"
        tree = ast.parse(path.read_text())
        method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                      and node.name == "update_weights_from_ipc")
        namespace = {"os": os, "torch": torch, "logger": logging.getLogger(__name__)}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
        buckets = [list(self.weights.items())[:2], list(self.weights.items())[2:]]

        class Receiver:
            def __init__(self, **kwargs):
                pass

            def receive_weights(self, on_bucket_received):
                for index, bucket in enumerate(buckets):
                    on_bucket_received(bucket, index == len(buckets) - 1)

        committed = []
        worker = SimpleNamespace(_pending_lora_peft_config=None, device=torch.device("cpu"),
                                 _get_zmq_handle=lambda: "isolated-cpu-test",
                                 _get_standard_weight_model_and_config=lambda: None,
                                 model_runner=SimpleNamespace(pipeline=SimpleNamespace(transformer=TinyNativeTransformer())),
                                 _full_pending_metadata=dict(self.metadata, policy_version=4, global_steps=2),
                                 _full_output_change_observed=True,
                                 _commit_full_policy_sync=committed.append)
        replacement = SimpleNamespace(BucketedWeightReceiver=Receiver)
        with patch.dict(os.environ, {"OMNI_FULL_FINETUNE": "1"}), patch.dict(sys.modules, {
            "verl.workers.rollout.vllm_rollout.bucketed_weight_transfer": replacement,
            "verl_omni.workers.rollout.vllm_rollout.full_weight_sync": SYNC,
            "vllm.distributed": SimpleNamespace(get_tensor_model_parallel_rank=lambda: 0,
                                                  get_tensor_model_parallel_world_size=lambda: 1),
        }):
            namespace[method.name](worker)
            self.assertEqual(len(committed), 1)
            self.assertIsNone(worker._full_pending_metadata)
            with self.assertRaisesRegex(RuntimeError, "tagged"):
                namespace[method.name](worker)
            with self.assertRaisesRegex(ValueError, "adapter"):
                namespace[method.name](worker, peft_config={"unexpected": True})
            worker._pending_lora_peft_config = {"stale": True}
            with self.assertRaisesRegex(ValueError, "adapter"):
                namespace[method.name](worker)

    def test_fixed_input_probe_is_deterministic_and_rng_neutral(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj_in = torch.nn.Linear(2, 2)
                self.config = SimpleNamespace(in_channels=2, audio_in_channels=2,
                                              cross_attention_dim=3, audio_cross_attention_dim=5)

            def forward(self, hidden_states, audio_hidden_states, **kwargs):
                self.probe_shapes = (list(kwargs["encoder_hidden_states"].shape),
                                     list(kwargs["audio_encoder_hidden_states"].shape))
                return self.proj_in(hidden_states), self.proj_in(audio_hidden_states)

        model = Model()
        rng = torch.random.get_rng_state().clone()
        native = Path(sys.executable).parent.parent / "lib/python3.12/site-packages/vllm_omni/diffusion/models/ltx2/ltx2_guidance.py"
        tree = ast.parse(native.read_text())
        method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "timestep_kwargs")
        method.decorator_list = []
        namespace = {"torch": torch}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(native), "exec"), namespace)
        replacement = SimpleNamespace(LTXGuidanceExecutor=SimpleNamespace(timestep_kwargs=namespace[method.name]))
        with patch.dict(sys.modules, {"vllm_omni.diffusion.models.ltx2.ltx2_guidance": replacement}):
            first = SYNC.fixed_ltx2_output_probe(model)
            second = SYNC.fixed_ltx2_output_probe(model)
            self.assertEqual(first, second)
            self.assertEqual(model.probe_shapes, ([1, 2, 3], [1, 2, 5]))
            torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
            with torch.no_grad():
                model.proj_in.bias.add_(0.25)
            self.assertNotEqual(first, SYNC.fixed_ltx2_output_probe(model))


if __name__ == "__main__":
    unittest.main()
