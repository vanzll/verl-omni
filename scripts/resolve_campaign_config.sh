#!/usr/bin/env bash
set -eo pipefail
source /m2v_intern2/zhenglin/my_bashrc.sh
set -u
campaign=/m2v_intern/zhenglin/omni_art_ablation_20261016
variant=${1:?pass N0/A0/A1/A2}
case "$variant" in N0|A0|A1|A2) ;; *) exit 2 ;; esac
source "$campaign/assets/source-selected.env"
export CUDA_VISIBLE_DEVICES="" VLLM_LOGGING_LEVEL=ERROR OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export OUTPUT_DIR="$campaign/preflight/configs/$variant" RUN_NAME="config-check-$variant"
export RAY_ADDRESS=127.0.0.1:65530
if [[ "$variant" == N0 ]]; then
    exec bash "$campaign/scripts/run_n0.sh" --cfg job --resolve
else
    export VARIANT="$variant"
    exec bash "$campaign/scripts/run_art.sh" --cfg job --resolve
fi
