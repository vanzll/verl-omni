import json
import os
from pathlib import Path
import subprocess

import pytest


RUNTIME = Path(__file__).resolve().parents[3] / "scripts/omni_art_joint32/runtime_env.sh"


@pytest.mark.parametrize("full", ["0", "1"])
def test_transport_override_is_limited_to_full_training(tmp_path, full):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "runtime_env.sh").write_text(
        "export NCCL_NET=Socket NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0\n"
        "export NCCL_IB_DISABLE=1 NCCL_NET_PLUGIN=old_plugin\n"
        "export NCCL_DEBUG=WARN\n"
        "export NCCL_IB_HCA='=mlx5_0:1' NCCL_IB_GID_INDEX=3\n"
        "export LD_LIBRARY_PATH=unchanged_cuda LD_PRELOAD=\n"
    )
    env = dict(os.environ, OMNI_CAMPAIGN_ROOT=str(tmp_path), OMNI_EXPECTED_IPS="10.0.0.1,10.0.0.2",
               OMNI_FULL_FINETUNE=full, OUTPUT_DIR=str(tmp_path), PYTHON=os.sys.executable)
    result = subprocess.run(
        ["bash", "-c", 'source "$1"; "$PYTHON" -c \'import json,os; print(json.dumps(dict(os.environ)))\'',
         "transport-test", str(RUNTIME)],
        env=env, text=True, capture_output=True, check=True, timeout=10,
    )
    actual = json.loads(result.stdout)
    assert actual["NCCL_NET"] == ("IB" if full == "1" else "Socket")
    assert actual["NCCL_P2P_DISABLE"] == ("0" if full == "1" else "1")
    assert actual["NCCL_IB_DISABLE"] == ("0" if full == "1" else "1")
    assert actual["NCCL_NET_PLUGIN"] == ("none" if full == "1" else "old_plugin")
    assert actual["NCCL_DEBUG"] == ("INFO" if full == "1" else "WARN")
    assert actual["NCCL_NVLS_ENABLE"] == "0"
    assert actual["NCCL_IB_HCA"] == "=mlx5_0:1"
    assert actual["NCCL_IB_GID_INDEX"] == "3"
    assert actual["LD_LIBRARY_PATH"] == "unchanged_cuda"
    if full == "1":
        assert actual["NCCL_DEBUG_SUBSYS"] == "INIT,NET"
        assert actual["NCCL_DEBUG_FILE"] == str(tmp_path / "logs/nccl.%h.%p.log")
