#!/usr/bin/env bash
set -euo pipefail
source /m2v_intern2/zhenglin/my_bashrc.sh
campaign=/m2v_intern/zhenglin/omni_art_ablation_20261016
export PATH="$campaign/env/bin:$PATH"
export TMPDIR="$campaign/tmp"
export XDG_CACHE_HOME="$campaign/cache/xdg"
export UV_CACHE_DIR="$campaign/cache/uv"
export UV_HTTP_TIMEOUT=600
timeout --kill-after=15 21600 /usr/bin/python -u "$campaign/scripts/fast_download.py" \
  "$campaign/assets/vllm-wheel-manifest.json" --state "$campaign/logs/fast_wheels_state.json" \
  --files 2 --connections 64 --min-free-gib 32 --resume-free-gib 48
printf '[%s] installing verified vllm wheelhouse offline\n' "$(date -u +%FT%TZ)"
timeout 1800 uv pip install --offline --python "$campaign/env/bin/python" --no-index \
  --find-links "$campaign/assets/wheelhouse" 'vllm==0.30.0'
printf '[%s] vllm wheelhouse installation complete\n' "$(date -u +%FT%TZ)"
