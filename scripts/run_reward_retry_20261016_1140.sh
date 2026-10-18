#!/usr/bin/env bash
set -euo pipefail
source /m2v_intern2/zhenglin/my_bashrc.sh
campaign=/m2v_intern/zhenglin/omni_art_ablation_20261016
export TMPDIR="$campaign/tmp" XDG_CACHE_HOME="$campaign/cache/xdg"
export HF_HOME="$campaign/cache/hf" HF_HUB_CACHE="$campaign/cache/hf/hub"
export HF_HUB_DISABLE_XET=1
timeout --kill-after=15s 1800s /usr/bin/python -u "$campaign/scripts/fast_download.py" \
  "$campaign/assets/reward-retry-20261016_1140-manifest.json" \
  --state "$campaign/logs/reward_retry_20261016_1140_state.json" \
  --files 1 --connections 64 --min-free-gib 250 --resume-free-gib 275
