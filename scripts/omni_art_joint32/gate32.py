# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Actual allocation readiness and 32-device collective gate, not a training proof."""

import datetime
import importlib.util
import json
import os
import socket
import sys
import time
from pathlib import Path


def validate_nodes(nodes, expected):
    """Reject stale pairs, extra nodes, and fractional or missing GPU capacity."""
    alive = [node for node in nodes if node.get("Alive")]
    if len(expected) != 4 or len(alive) != 4:
        raise RuntimeError("exactly four live nodes required")
    if {node["NodeManagerAddress"] for node in alive} != set(expected):
        raise RuntimeError("Ray allocation differs from manifest")
    if any(node["Resources"].get("GPU") != 8 for node in alive):
        raise RuntimeError("eight GPU resources per node required")
    return alive


def ordered_nodes(nodes, descriptors):
    """Use manifest order so rank zero and the TCPStore head are identical."""
    by_ip = {node["NodeManagerAddress"]: node for node in nodes}
    return [by_ip[descriptor["ip"]] for descriptor in descriptors]


def main():
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    output = Path(os.environ["OUTPUT_DIR"])
    manifest = json.loads((output / "manifest.json").read_text())
    expected = {node["ip"]: node["hostname"] for node in manifest["nodes"]}
    ray.init(address=os.environ["RAY_ADDRESS"], namespace=os.environ["OMNI_ATTEMPT"], logging_level="ERROR")
    nodes = validate_nodes(ray.nodes(), expected)

    @ray.remote(num_cpus=0.1)
    def location():
        return {
            "hostname": socket.gethostname().split(".")[0],
            "temp_dir": ray._private.worker._global_node.get_temp_dir_path(),
        }

    locations = ray.get(
        [
            location.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node["NodeID"], soft=False)).remote()
            for node in nodes
        ],
        timeout=15,
    )
    for node, row in zip(nodes, locations, strict=True):
        host = expected[node["NodeManagerAddress"]]
        required = Path(os.environ["OMNI_RUNTIME_ROOT"]) / "tmp/ray" / manifest["attempt"] / host
        if row["hostname"] != host or Path(row["temp_dir"]).resolve() != required.resolve():
            raise RuntimeError("wrong host or non-isolated Ray temp directory")
    if sys.argv[1] == "readiness":
        print(json.dumps({"scope": "Ray resources only", "nodes": locations, "gpus": 32}))
        ray.shutdown()
        return

    @ray.remote(num_cpus=0.1, num_gpus=1, max_restarts=0)
    class Rank:
        def run(self, rank, address):
            spec = importlib.util.spec_from_file_location(
                "attempt_transport", Path(os.environ["OMNI_CONTROL_DIR"]) / "transport.py"
            )
            checks = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(checks)
            environment = checks.rank_transport()
            log_path = output / "logs" / f"gate_nccl_{socket.gethostname()}.{os.getpid()}.log"
            if log_path.exists():
                raise RuntimeError("gate NCCL log already exists")
            os.environ["NCCL_DEBUG_FILE"] = str(log_path)
            import torch
            import torch.distributed as dist

            if str(torch.__version__) != "2.13.0+cu129" or tuple(torch.cuda.nccl.version()) != (2, 29, 7):
                raise RuntimeError("Gate requires the validated Torch/cu129/native NCCL environment")
            torch.cuda.set_device(0)
            dist.init_process_group(
                "nccl", init_method=address, rank=rank, world_size=32, timeout=datetime.timedelta(seconds=120)
            )
            try:
                for _ in range(2):
                    value = torch.full((262144,), float(rank + 1), device="cuda")
                    dist.all_reduce(value)
                    torch.cuda.synchronize()
                    if not bool(torch.isfinite(value).all()) or not bool((value == 528).all()):
                        raise RuntimeError("32-rank sum mismatch")
                value = torch.full((4 * 1024**2,), float(rank + 1), device="cuda", dtype=torch.float32)
                gathered = torch.empty((32 * value.numel(),), device="cuda", dtype=torch.float32)
                timings = []
                for cycle in range(5):
                    value.fill_(rank + 1)
                    dist.barrier()
                    torch.cuda.synchronize()
                    start = time.monotonic()
                    dist.all_gather_into_tensor(gathered, value)
                    dist.reduce_scatter_tensor(value, gathered)
                    torch.cuda.synchronize()
                    elapsed = time.monotonic() - start
                    if cycle >= 2:
                        timings.append(elapsed)
                    if not bool((value == 32 * (rank + 1)).all()):
                        raise RuntimeError("32-rank gather/reduce-scatter mismatch")
                local_groups = [
                    dist.new_group(list(range(first, first + 8)), backend="nccl") for first in range(0, 32, 8)
                ]
                value.fill_(rank + 1)
                dist.all_reduce(value, group=local_groups[rank // 8])
                torch.cuda.synchronize()
                if not bool((value == sum(range(8 * (rank // 8) + 1, 8 * (rank // 8) + 9))).all()):
                    raise RuntimeError("eight-GPU local collective mismatch")
                sample_checks = []
                if os.environ.get("OMNI_FULL_FINETUNE") == "1":
                    sample_spec = importlib.util.spec_from_file_location(
                        "gate_tensor_audit", Path(os.environ["SOURCE_CHECKOUT_DIR"]) / "verl_omni/utils/tensor_audit.py"
                    )
                    sampler = importlib.util.module_from_spec(sample_spec)
                    sample_spec.loader.exec_module(sampler)

                    for count in (2**25, 2**26, 2**27):
                        expanded = torch.full((1,), float(rank + 1), dtype=torch.bfloat16, device="cuda").expand(count)
                        sample = sampler.tensor_sample(expanded)
                        if sample["flat_indices"][-1] != count - 1 or sample["values"] != [float(rank + 1)] * 8:
                            raise RuntimeError("Full-size CUDA diagnostic sampler failed.")
                        sample_checks.append(
                            {
                                "numel": count,
                                "last_index": sample["flat_indices"][-1],
                                "storage_bytes": expanded.untyped_storage().nbytes(),
                            }
                        )
                import subprocess

                devices = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=10,
                ).stdout
                processes = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=10,
                ).stdout
                uuid_by_index = dict(line.strip().split(", ") for line in devices.strip().splitlines())
                own_uuids = {
                    line.split(", ")[1].strip()
                    for line in processes.splitlines()
                    if line.split(", ")[0].strip() == str(os.getpid())
                }
                gpu_ids = ray.get_gpu_ids()
                if len(gpu_ids) != 1 or own_uuids != {uuid_by_index[str(int(gpu_ids[0]))]}:
                    raise RuntimeError("Actual CUDA process/GPU UUID differs from the Ray assignment")
                result = {
                    "rank": rank,
                    "hostname": socket.gethostname().split(".")[0],
                    "gpu_ids": ray.get_gpu_ids(),
                    "finite": True,
                    "cycles": 2,
                    "pid": os.getpid(),
                    "gpu_uuid": next(iter(own_uuids)),
                    "transport": checks.validate_nccl_log(log_path.read_text()),
                    "transport_environment": environment,
                    "nccl_log": str(log_path),
                    "nccl_libraries": checks.loaded_libraries(),
                    "torch_version": str(torch.__version__),
                    "nccl_version": torch.cuda.nccl.version(),
                    "gather_input_bytes_per_rank": value.numel() * value.element_size(),
                    "gather_output_bytes_per_rank": gathered.numel() * gathered.element_size(),
                    "gather_reduce_scatter_seconds": timings,
                    "full_size_cuda_sample_checks": sample_checks,
                }
                (output / "logs" / f"gate_rank_{rank:02d}.json").write_text(json.dumps(result, indent=2))
                return result
            finally:
                dist.destroy_process_group()

    # The port is unique to this attempt; a bind conflict fails rather than attaches.
    address = f"tcp://{manifest['nodes'][0]['ip']}:{manifest['collective_port']}"
    actors = []
    try:
        for node in ordered_nodes(nodes, manifest["nodes"]):
            for _ in range(8):
                actors.append(
                    Rank.options(
                        scheduling_strategy=NodeAffinitySchedulingStrategy(node["NodeID"], soft=False)
                    ).remote()
                )
        rows = ray.get([actor.run.remote(rank, address) for rank, actor in enumerate(actors)], timeout=180)
        pairs = {(row["hostname"], str(row["gpu_ids"])) for row in rows}
        if len(pairs) != 32 or {row["hostname"] for row in rows} != set(expected.values()):
            raise RuntimeError("32 distinct physical node/GPU assignments not observed")
        if len({row["gpu_uuid"] for row in rows}) != 32:
            raise RuntimeError("32 distinct actual GPU UUIDs required")
        spec = importlib.util.spec_from_file_location("attempt_transport", Path(__file__).with_name("transport.py"))
        checks = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checks)
        throughput = checks.validate_throughput(rows)
        if not sum(row["transport"]["validated_rdma_edges"] for row in rows):
            raise RuntimeError("No actual cross-node external-plugin channel observed")
        result = {
            "status": "passed_actual_32gpu_collectives",
            "scope": "not optimizer/math/training acceptance",
            "attempt": manifest["attempt"],
            "source_manifest_sha256": manifest["source_manifest_sha256"],
            "throughput": throughput,
            "ranks": rows,
        }
        temporary = output / "gate32.json.tmp"
        with temporary.open("w") as stream:
            json.dump(result, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(output / "gate32.json")
        print(json.dumps(result))
    finally:
        for actor in actors:
            ray.kill(actor, no_restart=True)
        ray.shutdown()


if __name__ == "__main__":
    main()
