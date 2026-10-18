#!/usr/bin/env bash
set -euo pipefail
source /m2v_intern2/zhenglin/my_bashrc.sh
campaign=/m2v_intern/zhenglin/omni_art_ablation_20261016
export UV_CACHE_DIR="$campaign/cache/uv" TMPDIR="$campaign/tmp"
export XDG_CACHE_HOME="$campaign/cache/xdg" UV_HTTP_TIMEOUT=600 UV_HTTP_RETRIES=10
timeout 600 "$campaign/env/bin/uv" pip compile --python "$campaign/env/bin/python" \
  --torch-backend=cu130 --extra gpu --extra omni --format pylock.toml \
  --output-file "$campaign/assets/pylock.runtime.toml" \
  "$campaign/source/pyproject.toml" "$campaign/assets/runtime-extras-requirements.in"
