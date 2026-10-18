#!/usr/bin/env bash
set -euo pipefail
source /m2v_intern2/zhenglin/my_bashrc.sh
campaign=/m2v_intern/zhenglin/omni_art_ablation_20261016
manifest=${1:?pass one prepared pair manifest}
[[ "$(realpath "$manifest")" == "$campaign/attempts/"*/pair_manifest.json ]] || exit 2
export TMPDIR="$campaign/tmp" XDG_CACHE_HOME="$campaign/cache/xdg"
exec /usr/bin/python -u "$campaign/scripts/coordinate_pair.py" "$manifest"
