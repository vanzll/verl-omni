#!/usr/bin/env bash
set -euo pipefail
[[ "${OMNI_VARIANT:?}" == ART_FULL32_ROUTED ]] || exit 2
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_ltx2_t2av_art_joint32.sh" "$@"
