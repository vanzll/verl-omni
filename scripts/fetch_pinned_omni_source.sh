#!/usr/bin/env bash
set -euo pipefail
source /m2v_intern2/zhenglin/my_bashrc.sh
campaign=/m2v_intern/zhenglin/omni_art_ablation_20261016
target="$campaign/assets/vendor/vllm-omni-12e9280010afc41ce5c6a1f46900fc329cf08377"
export TMPDIR="$campaign/tmp" XDG_CACHE_HOME="$campaign/cache/xdg"
export GIT_TERMINAL_PROMPT=0
mkdir -p "$target"
if [[ ! -d "$target/.git" ]]; then
  git init "$target"
  git -C "$target" remote add origin https://github.com/vllm-project/vllm-omni.git
fi
printf '[%s] fetching pinned commit metadata without history/blobs\n' "$(date -u +%FT%TZ)"
timeout --kill-after=10 180 git -C "$target" -c protocol.version=2 fetch \
  --depth=1 --filter=blob:none origin 12e9280010afc41ce5c6a1f46900fc329cf08377
git -C "$target" sparse-checkout init --cone
git -C "$target" sparse-checkout set vllm_omni requirements
printf '[%s] checking out pinned runtime files\n' "$(date -u +%FT%TZ)"
timeout --kill-after=10 300 git -C "$target" checkout --detach FETCH_HEAD
[[ "$(git -C "$target" rev-parse HEAD)" == 12e9280010afc41ce5c6a1f46900fc329cf08377 ]]
printf '[%s] pinned source ready\n' "$(date -u +%FT%TZ)"
