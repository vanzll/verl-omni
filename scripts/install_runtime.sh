#!/usr/bin/env bash
set -euo pipefail
source /m2v_intern2/zhenglin/my_bashrc.sh
campaign=/m2v_intern/zhenglin/omni_art_ablation_20261016
export PATH="$campaign/env/bin:$PATH"
export UV_CACHE_DIR="$campaign/cache/uv"
export XDG_CACHE_HOME="$campaign/cache/xdg"
export TMPDIR="$campaign/tmp"
export HF_HOME="$campaign/cache/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export UV_HTTP_TIMEOUT=600
export UV_HTTP_RETRIES=10
mkdir -p "$UV_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR"

printf '[%s] installing vllm 0.30.0 cu130\n' "$(date -u +%FT%TZ)"
uv pip install --python "$campaign/env/bin/python" --torch-backend=cu130 'vllm==0.30.0'
printf '[%s] installing fixed checkout and omni extras\n' "$(date -u +%FT%TZ)"
cd "$campaign/source"
uv pip install --python "$campaign/env/bin/python" --torch-backend=cu130 -e '.[gpu,omni]' \
  pytest 'audiobox_aesthetics==0.0.4' 'qwen-vl-utils==0.0.14' 'timm==1.0.27'
printf '[%s] runtime install complete\n' "$(date -u +%FT%TZ)"
