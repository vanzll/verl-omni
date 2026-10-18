#!/usr/bin/env bash
set -euo pipefail
source /m2v_intern2/zhenglin/my_bashrc.sh
campaign=/m2v_intern/zhenglin/omni_art_ablation_20261016
export UV_CACHE_DIR="$campaign/cache/uv" TMPDIR="$campaign/tmp"
export XDG_CACHE_HOME="$campaign/cache/xdg" UV_HTTP_TIMEOUT=60 UV_HTTP_RETRIES=3
omni="$campaign/assets/vendor/vllm-omni-12e9280010afc41ce5c6a1f46900fc329cf08377"
verl="$campaign/cache/uv/git-v1/checkouts/120e4b336e9ccec4/fefb0802"
[[ "$(git -C "$omni" rev-parse HEAD)" == 12e9280010afc41ce5c6a1f46900fc329cf08377 ]]
[[ "$(git -C "$verl" rev-parse HEAD)" == fefb080262e1c015a0ea05f958822a6a512dc795 ]]
[[ -f "$omni/pyproject.toml" && -f "$verl/pyproject.toml" ]]
timeout --kill-after=15 600 "$campaign/env/bin/uv" pip compile \
  --python "$campaign/env/bin/python" --torch-backend=cu129 --extra omni \
  --find-links "$campaign/assets/wheelhouse" \
  --override "$campaign/assets/local-source-overrides.in" \
  --format pylock.toml --output-file "$campaign/assets/pylock.runtime-cu129.toml" \
  "$campaign/source/pyproject.toml" "$campaign/assets/runtime-cu129-requirements.in"
