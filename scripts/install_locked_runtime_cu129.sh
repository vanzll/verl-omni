#!/usr/bin/env bash
set -euo pipefail
source /m2v_intern2/zhenglin/my_bashrc.sh
campaign=/m2v_intern/zhenglin/omni_art_ablation_20261016
export OMNI_RUNTIME_CUDA_VARIANT=cu129 CUDA_VISIBLE_DEVICES=""
source "$campaign/scripts/runtime_env.sh"
export TMPDIR="$campaign/tmp" XDG_CACHE_HOME="$campaign/cache/xdg"
export UV_CACHE_DIR="$campaign/cache/uv" HF_HOME="$campaign/cache/hf"
export HF_HUB_CACHE="$HF_HOME/hub" TORCH_EXTENSIONS_DIR="$campaign/cache/torch_extensions"
export TRITON_CACHE_DIR="$campaign/cache/triton" VLLM_CACHE_ROOT="$campaign/cache/vllm"
export PYTHONPATH="$campaign/source" PYTHONDONTWRITEBYTECODE=1
exec 8>"$campaign/logs/install_runtime_cu129.lock"
flock -n 8
"$campaign/env/bin/python" -u - <<'PY'
import importlib.metadata
import json
import subprocess
import tomllib
from pathlib import Path

root = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')
state = json.loads((root / 'logs/runtime_cu129_wheels_state.json').read_text())
tasks = json.loads((root / 'assets/runtime-cu129-wheel-manifest.json').read_text())
assert state['status'] == 'completed' and not state['failed']
assert len(state['verified']) == state['tasks'] == len(tasks)
proc = Path(f"/proc/{state['pid']}")
if proc.exists():
    cmd = (proc / 'cmdline').read_bytes()
    assert b'runtime_cu129_wheels_state.json' not in cmd, 'Downloader is still finishing'
lock = tomllib.loads((root / 'assets/pylock.runtime-cu129.toml').read_text())
versions = {p['name']: p.get('version', '') for p in lock['packages']}
expected = dict(torch='2.13.0+cu129', torchaudio='2.11.0+cu129', torchvision='0.28.0+cu129',
                torchcodec='0.16.0+cu129', vllm='0.30.0+cu129')
assert all(versions[k] == v for k, v in expected.items())
assert not any('+cu130' in version for version in versions.values())
artifacts = {task['name']: task for task in tasks}
sources = json.loads((root / 'assets/runtime-cu129-wheel-manifest-sources.json').read_text())
local_dirs = []
for source in sources:
    if 'directory' in source:
        path = Path(source['directory']['path'])
        commit = {'verl': 'fefb080262e1c015a0ea05f958822a6a512dc795',
                  'vllm-omni': '12e9280010afc41ce5c6a1f46900fc329cf08377'}[source['name']]
        actual = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
        assert actual == commit
        local_dirs.append(str(path))
    else:
        assert source['kind'] == 'verified-local-artifact'
        name = source['name'] + '==' + source['version']
        artifacts[name] = dict(destination=source['path'], sha256=source['sha256'])
lines = []
for package in lock['packages']:
    if 'directory' in package:
        continue
    name = package['name'] + '==' + package['version']
    artifact = artifacts[name]
    assert Path(artifact['destination']).is_file()
    lines.append(name + ' --hash=sha256:' + artifact['sha256'])
assert len(lines) == len(lock['packages']) - len(local_dirs) and len(local_dirs) == 2
(root / 'assets/runtime-cu129-install.txt').write_text('\n'.join(sorted(lines)) + '\n')
(root / 'assets/runtime-cu129-source-dirs.json').write_text(json.dumps(local_dirs) + '\n')
before = sorted((p.metadata['Name'], p.version) for p in importlib.metadata.distributions())
before_path = root / 'assets/runtime-before-cu129.json'
if not before_path.exists():
    before_path.write_text(json.dumps(before, indent=2) + '\n')
print('Ready to install', len(lines), 'hash-pinned artifacts and', len(local_dirs), 'exact source commits', flush=True)
PY
printf '[%s] installing complete locked artifact set offline\n' "$(date -u +%FT%TZ)"
timeout --kill-after=15 1800 "$campaign/env/bin/uv" pip install --python "$campaign/env/bin/python" \
  --offline --no-index --no-deps --require-hashes --find-links "$campaign/assets/wheelhouse" \
  -r "$campaign/assets/runtime-cu129-install.txt"
printf '[%s] installing exact pinned source dependencies\n' "$(date -u +%FT%TZ)"
timeout --kill-after=15 600 "$campaign/env/bin/uv" pip install --python "$campaign/env/bin/python" \
  --offline --no-deps \
  "$campaign/cache/uv/git-v1/checkouts/120e4b336e9ccec4/fefb0802" \
  "$campaign/assets/vendor/vllm-omni-12e9280010afc41ce5c6a1f46900fc329cf08377"
printf '[%s] installing campaign source editable without changing the locked graph\n' "$(date -u +%FT%TZ)"
timeout --kill-after=15 300 "$campaign/env/bin/uv" pip install --python "$campaign/env/bin/python" \
  --offline --no-index --no-deps --no-build-isolation -e "$campaign/source"
printf '[%s] locked runtime installed; imports and CPU tests pending\n' "$(date -u +%FT%TZ)"
timeout 100 /usr/bin/python -u "$campaign/scripts/install_gl_libraries.py"
timeout --kill-after=15 240 "$campaign/env/bin/python" -u "$campaign/scripts/check_runtime_imports.py" \
  > "$campaign/logs/runtime_imports_cu129.log" 2>&1
set +e
timeout 45 "$campaign/env/bin/uv" pip check --python "$campaign/env/bin/python" \
  > "$campaign/logs/runtime_dependency_check.log" 2>&1
dependency_check_status=$?
set -e
printf '[%s] dependency check exit=%s; official override exceptions require review\n' \
  "$(date -u +%FT%TZ)" "$dependency_check_status"
bash "$campaign/scripts/run_cpu_regressions.sh" > "$campaign/logs/cpu_runtime_regressions.log" 2>&1
"$campaign/env/bin/python" - <<'PY'
import hashlib
import json
import time
from pathlib import Path
root = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')
record = dict(status='installed_and_cpu_tests_passed',
              utc=time.strftime('%FT%TZ', time.gmtime()),
              lock_sha256=hashlib.sha256((root / 'assets/pylock.runtime-cu129.toml').read_bytes()).hexdigest(),
              dependency_exception_review='pending', gpu_validation='pending')
(root / 'logs/runtime_cu129_cpu_ready.json').write_text(json.dumps(record, indent=2) + '\n')
PY
printf '[%s] complete CPU runtime checks passed; dependency review and GPU pilot pending\n' "$(date -u +%FT%TZ)"
