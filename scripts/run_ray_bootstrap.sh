#!/usr/bin/env bash
set -euo pipefail
source /m2v_intern2/zhenglin/my_bashrc.sh
campaign=/m2v_intern/zhenglin/omni_art_ablation_20261016
export TMPDIR="$campaign/tmp" XDG_CACHE_HOME="$campaign/cache/xdg"
export UV_CACHE_DIR="$campaign/cache/uv" CUDA_VISIBLE_DEVICES=""
timeout 40 "$campaign/env/bin/python" -u "$campaign/scripts/make_wheel_manifest.py" \
  "$campaign/assets/pylock.runtime-cu129.toml" "$campaign/assets/ray-bootstrap-manifest.json" --only ray --only msgpack
timeout --kill-after=10 300 /usr/bin/python -u "$campaign/scripts/fast_download.py" \
  "$campaign/assets/ray-bootstrap-manifest.json" --state "$campaign/logs/ray_bootstrap_state.json" \
  --files 1 --connections 64 --min-free-gib 32 --resume-free-gib 48
version=$("$campaign/env/bin/python" - <<'PY'
import json
from pathlib import Path
task = json.loads(Path('/m2v_intern/zhenglin/omni_art_ablation_20261016/assets/ray-bootstrap-manifest.json').read_text())
ray = [item for item in task if item['name'].startswith('ray==')]
assert len(ray) == 1
print(ray[0]['name'])
PY
)
timeout 60 "$campaign/env/bin/uv" pip install --offline --python "$campaign/env/bin/python" \
  --no-index --find-links "$campaign/assets/wheelhouse" "$version"
printf 'Pinned Ray installed for CPU containment preflight\n'
