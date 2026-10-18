#!/usr/bin/env bash
# Download preparation only: leave environment installation to the runtime audit.
set -euo pipefail
source /m2v_intern2/zhenglin/my_bashrc.sh
campaign=/m2v_intern/zhenglin/omni_art_ablation_20261016
export TMPDIR="$campaign/tmp" XDG_CACHE_HOME="$campaign/cache/xdg"
timeout 1500 /usr/bin/python -u - <<'PY'
import json
import time
from pathlib import Path

root = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')
names = ('torch_cu129_bootstrap_state.json', 'torchcodec_cu129_bootstrap_state.json')
deadline = time.monotonic() + 1450
while time.monotonic() < deadline:
    ready = []
    for name in names:
        path = root / 'logs' / name
        if not path.exists():
            ready.append(False)
            continue
        state = json.loads(path.read_text())
        if state['status'] == 'completed' and not state['failed']:
            ready.append(True)
            continue
        if state['status'] in ('incomplete', 'failed') or not Path(f"/proc/{state['pid']}").exists():
            raise RuntimeError('Bootstrap stopped before completion: ' + name)
        ready.append(False)
    if all(ready):
        print('CUDA12 bootstrap artifacts verified; compiling complete runtime', flush=True)
        break
    time.sleep(5)
else:
    raise TimeoutError('CUDA12 bootstrap preparation deadline exceeded')
PY
bash "$campaign/scripts/compile_runtime_cu129.sh"
"$campaign/env/bin/python" -u - <<'PY'
import tomllib
from pathlib import Path

root = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')
lock = tomllib.loads((root / 'assets/pylock.runtime-cu129.toml').read_text())
versions = {package['name']: package.get('version') for package in lock['packages']}
expected = dict(torch='2.13.0+cu129', torchvision='0.28.0+cu129',
                torchaudio='2.11.0+cu129', torchcodec='0.16.0+cu129', vllm='0.30.0+cu129')
for name, version in expected.items():
    assert versions[name] == version, (name, versions[name], version)
assert 'cupy-cuda12x' in versions and 'cupy-cuda13x' not in versions
assert all('+cu130' not in version for version in versions.values() if version)
assert versions['kernels'] == '0.16.0'
print('Complete candidate lock passes CUDA variant checks:', expected, flush=True)
PY
timeout 180 "$campaign/env/bin/python" -u "$campaign/scripts/make_wheel_manifest.py" \
  "$campaign/assets/pylock.runtime-cu129.toml" "$campaign/assets/runtime-cu129-wheel-manifest.json"
timeout --kill-after=15 7200 /usr/bin/python -u "$campaign/scripts/fast_download.py" \
  "$campaign/assets/runtime-cu129-wheel-manifest.json" \
  --state "$campaign/logs/runtime_cu129_wheels_state.json" \
  --files 2 --connections 64 --min-free-gib 32 --resume-free-gib 48
printf '[%s] complete candidate wheelhouse verified; environment installation pending audit\n' "$(date -u +%FT%TZ)"
