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

"""Validated native NCCL transport for the KML eight-GPU-per-node allocation."""

import hashlib
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
import sysconfig
from pathlib import Path

NETWORK = "NCCL RDMA Plugin v9"
PLUGIN = Path("/opt/hpcx/nccl_rdma_sharp_plugin/lib/libnccl-net.so")
CORE_SHA256 = "19d9851b65feef08fb78280ffc0d3e9067d23bc1c637e4c93e3a6dd6e20b06e6"
PLUGIN_SHA256 = "699d05ec3017a1562835f0b3a43e36de1d02a789c855a53c773dc444326c9ef9"
CLEARED_ENVIRONMENT = (
    "NCCL_CUMEM_ENABLE",
    "NCCL_CUMEM_HOST_ENABLE",
    "NCCL_P2P_LEVEL",
    "NCCL_P2P_DIRECT_DISABLE",
)
COMMON_ENVIRONMENT = {
    "VERL_OMNI_NCCL_UNSET_ENV_VARS": ",".join(CLEARED_ENVIRONMENT),
    "NCCL_NET": NETWORK,
    "NCCL_NET_PLUGIN": str(PLUGIN),
    "NCCL_ALGO": "Ring",
    "NCCL_PROTO": "Simple",
    "NCCL_IB_DISABLE": "0",
    "NCCL_IB_PCI_RELAXED_ORDERING": "0",
    "NCCL_P2P_DISABLE": "0",
    "NCCL_NVLS_ENABLE": "0",
    "NCCL_SHM_DISABLE": "0",
    "NCCL_IB_GID_INDEX": "3",
    "NCCL_IB_QPS_PER_CONNECTION": "4",
    "NCCL_IB_TIMEOUT": "20",
    "NCCL_NET_OVERHEAD": "1000",
    "NCCL_SOCKET_IFNAME": "bond0",
    "GLOO_SOCKET_IFNAME": "bond0",
    "NCCL_DEBUG": "INFO",
    "NCCL_DEBUG_SUBSYS": "INIT,NET,GRAPH,P2P",
}


def active_hca_list(root: Path = Path("/sys/class/infiniband")) -> str:
    """Derive exactly eight active 400G Ethernet/RoCEv2 GID3 ports locally."""
    ports = []
    for port in sorted(root.glob("*/ports/*")):
        if not (port / "state").read_text().startswith("4:") or not (port / "rate").read_text().startswith("400 "):
            continue
        if (port / "link_layer").read_text().strip() != "Ethernet":
            raise RuntimeError("Active 400G port is not Ethernet/RoCE")
        if (port / "gid_attrs/types/3").read_text().strip() != "RoCE v2":
            raise RuntimeError("Active 400G port lacks RoCEv2 GID3")
        if (port / "gids/3").read_text().strip() in ("", "::", "0:0:0:0:0:0:0:0"):
            raise RuntimeError("Active 400G port has an empty GID3")
        ports.append(port.parents[1].name + ":" + port.name)
    if len(ports) != 8:
        raise RuntimeError(f"Expected eight active local 400G RoCE ports, found {ports}")
    return "=" + ",".join(ports)


def library_identity(path: Path, expected: str) -> dict:
    """Check the actual immutable library contents without loading CUDA."""
    resolved = path.resolve(strict=True)
    checksum = hashlib.sha256()
    with resolved.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    actual = checksum.hexdigest()
    if actual != expected:
        raise RuntimeError(f"Unexpected NCCL library identity: {resolved}")
    return {"path": str(resolved), "sha256": actual}


def native_core_path() -> Path:
    """Resolve the core shipped with this Python environment, not the image."""
    return Path(sysconfig.get_paths()["purelib"]) / "nvidia/nccl/lib/libnccl.so.2"


def runtime_environment(output: Path) -> dict[str, str]:
    """Construct the measured recipe with this node's HCA and native core."""
    core = library_identity(native_core_path(), CORE_SHA256)
    library_identity(PLUGIN, PLUGIN_SHA256)
    return {
        **COMMON_ENVIRONMENT,
        "NCCL_IB_HCA": active_hca_list(),
        "NCCL_DEBUG_FILE": str(output / "logs/nccl.%h.%p.log"),
        "LD_PRELOAD": core["path"],
    }


def rank_transport() -> dict[str, str]:
    """Reject inherited overrides and master-broadcast HCA lists on each rank."""
    expected = {
        **COMMON_ENVIRONMENT,
        "NCCL_IB_HCA": active_hca_list(),
        "LD_PRELOAD": str(native_core_path().resolve(strict=True)),
    }
    wrong = [key for key, value in expected.items() if os.environ.get(key) != value]
    wrong.extend(key for key in CLEARED_ENVIRONMENT if key in os.environ)
    if wrong:
        raise RuntimeError(f"GPU worker transport environment differs from the validated recipe: {wrong}")
    return expected


def loaded_libraries() -> dict:
    """Verify mapped native core and external plugin, including their hashes."""
    paths = {
        Path(line.split()[-1]).resolve()
        for line in Path("/proc/self/maps").read_text().splitlines()
        if re.search(r"/libnccl(?:-net)?\.so(?:\.[0-9]+)*$", line)
    }
    core = library_identity(native_core_path(), CORE_SHA256)
    plugin = library_identity(PLUGIN, PLUGIN_SHA256)
    if paths != {Path(core["path"]), Path(plugin["path"])}:
        raise RuntimeError(f"Actual mapped NCCL core/plugin differ from the validated pair: {paths}")
    return {"core": core, "plugin": plugin}


def validate_topology(text: str) -> dict:
    """Require the allocation's complete eight-GPU NVLink topology."""
    rows = {}
    for line in text.splitlines():
        fields = line.split()
        if fields and re.fullmatch(r"GPU[0-7]", fields[0]):
            rows[int(fields[0][3:])] = fields[1:9]
    if set(rows) != set(range(8)) or any(
        len(links) != 8
        or links[rank] != "X"
        or any(not re.fullmatch(r"NV[1-9][0-9]*", link) for peer, link in enumerate(links) if peer != rank)
        for rank, links in rows.items()
    ):
        raise RuntimeError("All eight physical GPUs must have complete NVLink connectivity")
    return rows


def node_topology() -> dict:
    """Read hardware admission evidence without creating a CUDA context."""
    topology = subprocess.run(
        ["nvidia-smi", "topo", "-m"], check=True, capture_output=True, text=True, timeout=15
    ).stdout
    if not list(Path("/dev/infiniband").glob("uverbs*")):
        raise RuntimeError("Container-accessible RDMA devices required")
    return {"nvlink_pairs": validate_topology(topology), "local_hca": active_hca_list(), "raw_topology": topology}


def validate_nccl_log(text: str) -> dict:
    """Check actual plugin/P2P channels while permitting bootstrap TCP."""
    networks = re.findall(r"Using network ([^\n]+)", text)
    if not networks or any(network.strip() != NETWORK for network in networks):
        raise RuntimeError("Missing or conflicting external RDMA plugin selection")
    if re.search(r"via NET/Socket\b|via SHM\b", text):
        raise RuntimeError("Forbidden Socket/SHM tensor data path")
    channels = re.findall(
        r"Channel\s+\d+(?:/\d+)?\s*:\s*(\d+)\[[^\]]+\]\s*->\s*" r"(\d+)\[[^\]]+\][^\n]*?via\s+([^\n]+)",
        text,
    )
    if not channels:
        raise RuntimeError("Missing actual NCCL channel endpoints")
    local_edges = remote_edges = 0
    for sender, receiver, path in channels:
        if int(sender) // 8 == int(receiver) // 8:
            if not path.startswith("P2P/"):
                raise RuntimeError("Local NCCL channel did not select NVLink/P2P")
            local_edges += 1
        else:
            if not path.startswith("NET/" + NETWORK + "/"):
                raise RuntimeError("Cross-node NCCL channel did not select the validated external plugin")
            remote_edges += 1
    if not local_edges:
        raise RuntimeError("Missing actual local P2P channel evidence")
    return {
        "rdma_selected": True,
        "p2p_channels_observed": True,
        "socket_data_path_observed": False,
        "validated_channel_edges": len(channels),
        "validated_rdma_edges": remote_edges,
    }


def validate_throughput(rows: list[dict]) -> dict:
    """Enforce the allocation's warmed world32 16MiB AG/RS regression gate."""
    if len(rows) != 32 or {row["rank"] for row in rows} != set(range(32)):
        raise RuntimeError("Throughput evidence requires all 32 distinct ranks")
    timings = []
    for row in rows:
        values = row["gather_reduce_scatter_seconds"]
        if (
            row["gather_input_bytes_per_rank"] != 16 * 1024**2
            or row["gather_output_bytes_per_rank"] != 512 * 1024**2
            or len(values) != 3
            or any(not math.isfinite(value) or value <= 0 for value in values)
        ):
            raise RuntimeError("Invalid world32 throughput measurement contract")
        timings.append(values)
    maxima = [max(values[index] for values in timings) for index in range(3)]
    median = statistics.median(maxima)
    if median >= 0.2:
        raise RuntimeError(f"Warmed world32 AG/RS regressed: cross-rank-max median {median:.6f}s >= 0.2s")
    return {
        "warmup_cycles": 2,
        "measurement_cycles": 3,
        "cross_rank_max_seconds": maxima,
        "median_cross_rank_max_seconds": median,
        "maximum_allowed_seconds": 0.2,
    }


def main() -> None:
    """Apply the shell recipe or emit public-only Ray/Hydra overrides."""
    command = sys.argv[1] if len(sys.argv) > 1 else "topology"
    if command == "shell-env":
        environment = runtime_environment(Path(os.environ["OUTPUT_DIR"]))
        print("unset " + " ".join(CLEARED_ENVIRONMENT))
        for key, value in environment.items():
            print("export " + key + "=" + shlex.quote(value))
    elif command == "hydra":
        environment = rank_transport()
        environment["NCCL_DEBUG_FILE"] = os.environ["NCCL_DEBUG_FILE"]
        for key, value in environment.items():
            if key != "NCCL_IB_HCA":
                print("++ray_kwargs.ray_init.runtime_env.env_vars." + key + "=" + json.dumps(value))
        provenance = Path(__file__).resolve().parents[2] / "transport-publication.json"
        if provenance.is_file():
            metadata = json.loads(provenance.read_text())
            for key in ("publication_commit", "reference_run_id", "science_base_manifest_sha256"):
                value = metadata[key]
                if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{16}|[0-9a-f]{40}|[0-9a-f]{64}", value):
                    raise RuntimeError("Invalid frozen transport provenance: " + key)
                print("++trainer.campaign.transport_" + key + "=" + json.dumps(value))
    elif command == "topology":
        print(json.dumps(node_topology()))
    else:
        raise ValueError("Unsupported transport command")


if __name__ == "__main__":
    main()
