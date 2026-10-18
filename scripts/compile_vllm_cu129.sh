#!/usr/bin/env bash
set -euo pipefail
source /m2v_intern2/zhenglin/my_bashrc.sh
campaign=/m2v_intern/zhenglin/omni_art_ablation_20261016
export UV_CACHE_DIR="$campaign/cache/uv" TMPDIR="$campaign/tmp"
export XDG_CACHE_HOME="$campaign/cache/xdg" UV_HTTP_TIMEOUT=60 UV_HTTP_RETRIES=3
timeout --kill-after=15 600 "$campaign/env/bin/uv" pip compile \
  --python "$campaign/env/bin/python" --torch-backend=cu129 \
  --find-links "$campaign/assets/wheelhouse" \
  --format pylock.toml --output-file "$campaign/assets/pylock.vllm-cu129.toml" \
  "$campaign/assets/vllm-cu129-requirements.in"
