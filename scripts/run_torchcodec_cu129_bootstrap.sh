#!/usr/bin/env bash
set -euo pipefail
source /m2v_intern2/zhenglin/my_bashrc.sh
campaign=/m2v_intern/zhenglin/omni_art_ablation_20261016
export TMPDIR="$campaign/tmp" XDG_CACHE_HOME="$campaign/cache/xdg"
timeout --kill-after=15 900 /usr/bin/python -u "$campaign/scripts/fast_download.py" \
  "$campaign/assets/torchcodec-cu129-bootstrap-manifest.json" \
  --state "$campaign/logs/torchcodec_cu129_bootstrap_state.json" \
  --files 1 --connections 64 --min-free-gib 32 --resume-free-gib 48
