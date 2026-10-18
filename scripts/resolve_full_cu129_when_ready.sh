#!/usr/bin/env bash
set -euo pipefail
source /m2v_intern2/zhenglin/my_bashrc.sh
campaign=/m2v_intern/zhenglin/omni_art_ablation_20261016
export TMPDIR="$campaign/tmp" XDG_CACHE_HOME="$campaign/cache/xdg"
timeout 1500 /usr/bin/python -u - <<'PY'
import json
import time
from pathlib import Path

root = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')
deadline = time.monotonic() + 1450
while time.monotonic() < deadline:
    state = json.loads((root / 'logs/cu129_bootstrap_state.json').read_text())
    wheel = root / 'assets/wheelhouse/vllm-0.30.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl'
    if state['status'] == 'completed' and wheel.is_file() and len(state['verified']) == 1:
        print('Publisher-hash-verified cu129 wheel ready; resolving full pinned runtime', flush=True)
        break
    if state['status'] in ('failed', 'incomplete') or not Path(f"/proc/{state['pid']}").exists():
        raise RuntimeError('Bootstrap stopped before verifying candidate wheel')
    time.sleep(5)
else:
    raise TimeoutError('Candidate bootstrap readiness deadline exceeded')
PY
bash "$campaign/scripts/compile_runtime_cu129.sh"
