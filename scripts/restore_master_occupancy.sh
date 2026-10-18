#!/usr/bin/env bash
# Recovery of the initial failed runtime probe, using a new fenced generation.
set -eo pipefail
source /m2v_intern2/zhenglin/my_bashrc.sh
set -u
campaign=/m2v_intern/zhenglin/omni_art_ablation_20261016
[[ "$(hostname -s)" == klingai-wlf2-ge103-134 ]] || exit 2
export GPU_OCCUPY_SCRIPT=/m2v_intern2/zhenglin/gpu_occupy.py
export GPU_OCCUPY_WORKDIR="$campaign"
export GPU_OCCUPY_LOG="$campaign/logs/gpu_occupy_$(hostname -s).log"
export GPU_OCCUPY_SESSION="omni_hold_$(hostname -s)"
export GPU_OCCUPY_PYTHON=/usr/bin/python
export TMPDIR="$campaign/tmp" TMUX_TMPDIR="$campaign/tmp"
export FLOWART_GPU_LEASE_ROOT="/m2v_intern/zhenglin/paper_coordination/GPU_LEASES/$(hostname -s)"
unalias gpu_free gpu_hold 2>/dev/null || true
gpu_free() { "$campaign/bin/gpu_free"; }
gpu_hold() { LD_LIBRARY_PATH= "$campaign/bin/gpu_hold"; }
source "$campaign/scripts/gpu_lease_lib.sh"
flowart_gpu_lease_configure "$(hostname -s)" "$campaign" occupancy-recovery-20261016T1052 recovery-1052
flowart_gpu_lease_acquire
flowart_gpu_lease_release
