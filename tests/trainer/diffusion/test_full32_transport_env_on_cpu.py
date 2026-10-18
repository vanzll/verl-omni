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

"""Transport-only runtime regressions without CUDA contexts or real hardware."""

import ast
import asyncio
import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("transport", ROOT / "scripts/omni_art_joint32/transport.py")
transport = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(transport)


def test_active_hca_requires_local_eight_roce_gid3(tmp_path):
    for index in range(8):
        port = tmp_path / f"mlx5_{index}/ports/1"
        (port / "gid_attrs/types").mkdir(parents=True)
        (port / "gids").mkdir()
        for name, value in {
            "state": "4: ACTIVE",
            "rate": "400 Gb/sec",
            "link_layer": "Ethernet",
            "gid_attrs/types/3": "RoCE v2",
            "gids/3": "::ffff:10.0.0.1",
        }.items():
            (port / name).write_text(value)
    expected = "=" + ",".join(f"mlx5_{index}:1" for index in range(8))
    assert transport.active_hca_list(tmp_path) == expected
    (tmp_path / "mlx5_7/ports/1/gids/3").write_text("::")
    with pytest.raises(RuntimeError, match="empty GID3"):
        transport.active_hca_list(tmp_path)
    (tmp_path / "mlx5_7/ports/1/state").write_text("1: DOWN")
    with pytest.raises(RuntimeError, match="eight"):
        transport.active_hca_list(tmp_path)


def test_runtime_validates_library_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr(transport, "native_core_path", lambda: Path("/core"))
    identities = []

    def identity(path, expected):
        identities.append((path, expected))
        return {"path": str(path), "sha256": expected}

    monkeypatch.setattr(transport, "library_identity", identity)
    monkeypatch.setattr(transport, "active_hca_list", lambda: "=local:1")
    environment = transport.runtime_environment(tmp_path)
    assert identities == [(Path("/core"), transport.CORE_SHA256), (transport.PLUGIN, transport.PLUGIN_SHA256)]
    assert environment == {
        **transport.COMMON_ENVIRONMENT,
        "NCCL_IB_HCA": "=local:1",
        "LD_PRELOAD": "/core",
        "NCCL_DEBUG_FILE": str(tmp_path / "logs/nccl.%h.%p.log"),
    }


@pytest.mark.parametrize("key", transport.CLEARED_ENVIRONMENT)
def test_rank_rejects_even_empty_stale_flags(tmp_path, monkeypatch, key):
    core = tmp_path / "libnccl.so.2"
    core.write_bytes(b"test")
    monkeypatch.setattr(transport, "native_core_path", lambda: core)
    monkeypatch.setattr(transport, "active_hca_list", lambda: "=local:1")
    for flag in transport.CLEARED_ENVIRONMENT:
        monkeypatch.delenv(flag, raising=False)
    for flag, value in {**transport.COMMON_ENVIRONMENT, "LD_PRELOAD": str(core), "NCCL_IB_HCA": "=local:1"}.items():
        monkeypatch.setenv(flag, value)
    assert transport.rank_transport()["NCCL_IB_HCA"] == "=local:1"
    monkeypatch.setenv(key, "")
    with pytest.raises(RuntimeError, match=key):
        transport.rank_transport()


def test_hydra_never_broadcasts_master_hca(monkeypatch, capsys):
    monkeypatch.setattr(
        transport,
        "rank_transport",
        lambda: {**transport.COMMON_ENVIRONMENT, "NCCL_IB_HCA": "=master:1", "LD_PRELOAD": "/core"},
    )
    monkeypatch.setenv("NCCL_DEBUG_FILE", "/attempt/nccl.%h.%p.log")
    monkeypatch.setattr(sys, "argv", ["transport.py", "hydra"])
    transport.main()
    output = capsys.readouterr().out
    assert "NCCL_IB_HCA" not in output
    assert 'NCCL_NET="NCCL RDMA Plugin v9"' in output
    assert 'NCCL_ALGO="Ring"' in output
    assert 'NCCL_PROTO="Simple"' in output
    assert "VERL_OMNI_NCCL_UNSET_ENV_VARS=" in output


@pytest.mark.parametrize("policy", ["", ",".join(transport.CLEARED_ENVIRONMENT), "UNRELATED_SECRET"])
def test_native_server_applies_opt_in_unset_policy_before_spawning(monkeypatch, policy):
    path = ROOT / "verl_omni/workers/rollout/vllm_rollout/vllm_omni_async_server.py"
    tree = ast.parse(path.read_text())
    server = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "vLLMOmniHttpServer")
    method = next(node for node in server.body if isinstance(node, ast.FunctionDef) and node.name == "_post_init")
    probe = ast.ClassDef(
        name="Probe", bases=[ast.Name(id="Base", ctx=ast.Load())], keywords=[], body=[method], decorator_list=[]
    )

    class Base:
        def _post_init(self, devices):
            self.finished = True

    namespace = {
        "Base": Base,
        "os": os,
        "asyncio": asyncio,
        "LoRARequest": object,
        "_LORA_REQUEST_CACHE_MISS": object(),
        "get_rollout_sequence_parallel_size": lambda config: 1,
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[probe], type_ignores=[])), str(path), "exec"), namespace)
    instance = namespace["Probe"]()
    instance.config = SimpleNamespace(full_determinism=False)
    observed = []
    instance._generate_strategy = SimpleNamespace(post_init=lambda devices: observed.append(dict(os.environ)))
    for key in transport.CLEARED_ENVIRONMENT:
        monkeypatch.setenv(key, "0")
    monkeypatch.setenv("NCCL_ALGO", "Ring")
    monkeypatch.setenv("VERL_OMNI_NCCL_UNSET_ENV_VARS", policy)
    if policy == "UNRELATED_SECRET":
        with pytest.raises(ValueError, match="only accepts NCCL_"):
            instance._post_init("0,1")
        assert not observed
    else:
        instance._post_init("0,1")
        assert instance.finished and observed[0]["NCCL_ALGO"] == "Ring"
        assert all((key not in observed[0]) == bool(policy) for key in transport.CLEARED_ENVIRONMENT)


def test_library_hash_rejects_an_unvalidated_core(tmp_path):
    library = tmp_path / "libnccl.so.2"
    library.write_bytes(b"not the validated library")
    with pytest.raises(RuntimeError, match="identity"):
        transport.library_identity(library, transport.CORE_SHA256)


def test_mapped_libraries_require_exact_core_and_plugin(tmp_path, monkeypatch):
    core, plugin = tmp_path / "libnccl.so.2", tmp_path / "libnccl-net.so.0.0.0"
    core.touch()
    plugin.touch()
    monkeypatch.setattr(transport, "native_core_path", lambda: core)
    monkeypatch.setattr(transport, "PLUGIN", plugin)
    monkeypatch.setattr(transport, "library_identity", lambda path, expected: {"path": str(path), "sha256": expected})
    original = Path.read_text
    mapped = [core, plugin]
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, *args, **kwargs: "".join(f"100-200 r-xp 0 0 0 {library}\n" for library in mapped)
        if str(path) == "/proc/self/maps"
        else original(path, *args, **kwargs),
    )
    assert transport.loaded_libraries()["plugin"]["path"] == str(plugin)
    mapped.pop()
    with pytest.raises(RuntimeError, match="mapped"):
        transport.loaded_libraries()


GOOD_LOG = """Bootstrap : Using bond0
Using network NCCL RDMA Plugin v9
Channel 03/0 : 0[0] -> 1[1] via P2P/CUMEM
Channel 00/0 : 24[0] -> 0[0] [receive] via NET/NCCL RDMA Plugin v9/0/GDRDMA
"""


def test_log_accepts_exact_plugin_and_bootstrap_tcp():
    result = transport.validate_nccl_log(GOOD_LOG)
    assert result["validated_rdma_edges"] == 1
    assert result["p2p_channels_observed"]


@pytest.mark.parametrize(
    "bad",
    [
        GOOD_LOG.replace("Using network NCCL RDMA Plugin v9", "Using network IB"),
        GOOD_LOG + "Using network Socket\n",
        GOOD_LOG.replace("P2P/CUMEM", "SHM/direct/direct"),
        GOOD_LOG.replace("NET/NCCL RDMA Plugin v9/0/GDRDMA", "NET/IB/0"),
        GOOD_LOG.replace("P2P/CUMEM", "NET/NCCL RDMA Plugin v9/0"),
    ],
)
def test_log_rejects_data_path_fallback(bad):
    with pytest.raises(RuntimeError):
        transport.validate_nccl_log(bad)


def throughput_rows():
    return [
        {
            "rank": rank,
            "gather_input_bytes_per_rank": 16 * 1024**2,
            "gather_output_bytes_per_rank": 512 * 1024**2,
            "gather_reduce_scatter_seconds": [0.008, 0.009, 0.010],
        }
        for rank in range(32)
    ]


def test_throughput_uses_cross_rank_max_then_median():
    rows = throughput_rows()
    rows[31]["gather_reduce_scatter_seconds"] = [0.02, 0.03, 0.9]
    result = transport.validate_throughput(rows)
    assert result["median_cross_rank_max_seconds"] == 0.03
    assert result["cross_rank_max_seconds"] == [0.02, 0.03, 0.9]
    rows[0]["gather_reduce_scatter_seconds"] = [0.3, 0.3, 0.01]
    with pytest.raises(RuntimeError, match="regressed"):
        transport.validate_throughput(rows)


@pytest.mark.parametrize(
    "field,value",
    [
        ("rank", 1),
        ("gather_input_bytes_per_rank", 8),
        ("gather_output_bytes_per_rank", 8),
        ("gather_reduce_scatter_seconds", [0.01, float("nan"), 0.01]),
        ("gather_reduce_scatter_seconds", [0.01]),
        ("gather_reduce_scatter_seconds", [0, 0, 0]),
    ],
)
def test_throughput_rejects_incomplete_contract(field, value):
    rows = copy.deepcopy(throughput_rows())
    rows[0][field] = value
    with pytest.raises(RuntimeError):
        transport.validate_throughput(rows)


@pytest.mark.parametrize("full", ["0", "1"])
def test_shared_shell_applies_recipe_for_lora_and_full(tmp_path, full):
    scripts = tmp_path / "scripts"
    control = scripts / "omni_art_joint32"
    control.mkdir(parents=True)
    (control / "runtime_env.sh").write_text((ROOT / "scripts/omni_art_joint32/runtime_env.sh").read_text())
    (scripts / "nccl_transport_env.sh").write_text((ROOT / "scripts/nccl_transport_env.sh").read_text())
    environment = {
        **transport.COMMON_ENVIRONMENT,
        "NCCL_IB_HCA": "=this_node:1",
        "LD_PRELOAD": "",
        "NCCL_DEBUG_FILE": str(tmp_path / "logs/nccl.%h.%p.log"),
    }
    (control / "transport.py").write_text(
        "import json,shlex\n"
        + "print('unset "
        + " ".join(transport.CLEARED_ENVIRONMENT)
        + "')\n"
        + "for key,value in "
        + repr(environment)
        + ".items(): print('export '+key+'='+shlex.quote(value))\n"
    )
    (scripts / "runtime_env.sh").write_text(
        "export NCCL_NET=Socket NCCL_P2P_DISABLE=1 NCCL_NET_PLUGIN=old_plugin\n"
        "export NCCL_CUMEM_ENABLE=0 NCCL_P2P_LEVEL=NVL LD_LIBRARY_PATH=unchanged_cuda\n"
    )
    env = dict(
        os.environ,
        OMNI_CAMPAIGN_ROOT=str(tmp_path),
        OMNI_EXPECTED_IPS="10.0.0.1",
        OMNI_FULL_FINETUNE=full,
        OUTPUT_DIR=str(tmp_path),
    )
    result = subprocess.run(
        [
            "bash",
            "-ec",
            "source \"$1\"; python -c 'import json,os; print(json.dumps(dict(os.environ)))'",
            "runtime-test",
            str(control / "runtime_env.sh"),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    actual = json.loads(result.stdout)
    assert all(actual[key] == value for key, value in environment.items())
    assert all(key not in actual for key in transport.CLEARED_ENVIRONMENT)
    assert actual["LD_LIBRARY_PATH"] == "unchanged_cuda"
