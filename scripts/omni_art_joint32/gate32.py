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
import json
import os
from pathlib import Path
import socket
import sys

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

    locations = ray.get([
        location.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node["NodeID"], soft=False)).remote()
        for node in nodes
    ], timeout=15)
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
            import torch
            import torch.distributed as dist
            torch.cuda.set_device(0)
            dist.init_process_group("nccl", init_method=address, rank=rank, world_size=32,
                                    timeout=datetime.timedelta(seconds=120))
            try:
                for _ in range(2):
                    value = torch.full((262144,), float(rank + 1), device="cuda")
                    dist.all_reduce(value)
                    torch.cuda.synchronize()
                    if not bool(torch.isfinite(value).all()) or not bool((value == 528).all()):
                        raise RuntimeError("32-rank sum mismatch")
                sample_checks = []
                if os.environ.get("OMNI_FULL_FINETUNE") == "1":
                    from verl_omni.utils.tensor_audit import tensor_sample

                    for count in (2**25, 2**26, 2**27):
                        expanded = torch.full((1,), float(rank + 1), dtype=torch.bfloat16, device="cuda").expand(count)
                        sample = tensor_sample(expanded)
                        if sample["flat_indices"][-1] != count - 1 or sample["values"] != [float(rank + 1)] * 8:
                            raise RuntimeError("Full-size CUDA diagnostic sampler failed.")
                        sample_checks.append({"numel": count, "last_index": sample["flat_indices"][-1],
                                              "storage_bytes": expanded.untyped_storage().nbytes()})
                return {"rank": rank, "hostname": socket.gethostname().split(".")[0],
                        "gpu_ids": ray.get_gpu_ids(), "finite": True, "cycles": 2,
                        "full_size_cuda_sample_checks": sample_checks}
            finally:
                dist.destroy_process_group()

    # The port is unique to this attempt; a bind conflict fails rather than attaches.
    address = f"tcp://{manifest['nodes'][0]['ip']}:{manifest['collective_port']}"
    actors = []
    try:
        for node in ordered_nodes(nodes, manifest["nodes"]):
            for _ in range(8):
                actors.append(Rank.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(node["NodeID"], soft=False)
                ).remote())
        rows = ray.get([actor.run.remote(rank, address) for rank, actor in enumerate(actors)], timeout=180)
        pairs = {(row["hostname"], str(row["gpu_ids"])) for row in rows}
        if len(pairs) != 32 or {row["hostname"] for row in rows} != set(expected.values()):
            raise RuntimeError("32 distinct physical node/GPU assignments not observed")
        result = {"status": "passed_actual_32gpu_collectives", "scope": "not optimizer/math/training acceptance",
                  "attempt": manifest["attempt"], "source_manifest_sha256": manifest["source_manifest_sha256"],
                  "ranks": rows}
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
