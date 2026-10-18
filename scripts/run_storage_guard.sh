#!/usr/bin/env bash
set -euo pipefail
source /m2v_intern2/zhenglin/my_bashrc.sh
campaign=/m2v_intern/zhenglin/omni_art_ablation_20261016
export TMPDIR="$campaign/tmp"
export XDG_CACHE_HOME="$campaign/cache/xdg"
exec python "$campaign/scripts/guard_download_space.py" >> "$campaign/logs/storage_guard.log" 2>&1
