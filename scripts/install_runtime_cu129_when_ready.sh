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
    state = json.loads((root / 'logs/runtime_cu129_wheels_state.json').read_text())
    proc = Path(f"/proc/{state['pid']}")
    if state['status'] == 'completed' and not state['failed']:
        if not proc.exists() or b'runtime_cu129_wheels_state.json' not in (proc / 'cmdline').read_bytes():
            print('Full verified wheel queue completed and helper exited; starting offline installation', flush=True)
            break
    elif state['status'] in ('failed', 'incomplete') or not proc.exists():
        raise RuntimeError('Full runtime queue stopped before successful completion')
    time.sleep(5)
else:
    raise TimeoutError('Full runtime artifact readiness deadline exceeded')
PY
exec bash "$campaign/scripts/install_locked_runtime_cu129.sh"
