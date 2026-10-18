#!/usr/bin/env bash
set -euo pipefail
source /m2v_intern2/zhenglin/my_bashrc.sh
campaign=/m2v_intern/zhenglin/omni_art_ablation_20261016
export TMPDIR="$campaign/tmp"
export XDG_CACHE_HOME="$campaign/cache/xdg"
timeout --kill-after=10 600 /usr/bin/python -u "$campaign/scripts/download_benchmark.py" \
  >> "$campaign/logs/download_benchmark_20261016.log" 2>&1
