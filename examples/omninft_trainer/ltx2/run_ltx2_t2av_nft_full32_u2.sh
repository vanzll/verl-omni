#!/usr/bin/env bash
set -euo pipefail
[[ "${OMNI_VARIANT:?}" == NFT_FULL32_U2 ]] || exit 2
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_ltx2_t2av_nft_full32_u1.sh" "$@"
