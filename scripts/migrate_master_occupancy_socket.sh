#!/usr/bin/env bash
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
gpu_free() {
    # Remove only the exact old controller created by the previous probe owner.
    python - <<'PY'
import pathlib, subprocess
sock='/m2v_intern/zhenglin/omni_art_ablation_20261016/tmp/tmux-0/omni_runtime_probes'
session='omni_hold_klingai-wlf2-ge103-134'
pid=subprocess.check_output(['tmux','-S',sock,'list-panes','-t',session,'-F','#{pane_pid}'],timeout=5,text=True).strip()
assert pid=='549476', pid
cmd=(pathlib.Path('/proc')/pid/'cmdline').read_bytes().decode()
assert 'gpu_occupy.py' in cmd and 'omni_art_ablation_20261016/logs/gpu_occupy_' in cmd
subprocess.run(['tmux','-S',sock,'kill-session','-t',session],check=True,timeout=5)
PY
    "$campaign/bin/gpu_free"
}
gpu_hold() { LD_LIBRARY_PATH= "$campaign/bin/gpu_hold"; }
source "$campaign/scripts/gpu_lease_lib.sh"
flowart_gpu_lease_configure "$(hostname -s)" "$campaign" occupancy-socket-migration-20261016T1056 socket-migration-1056
trap 'flowart_gpu_lease_release' EXIT
flowart_gpu_lease_acquire
flowart_gpu_lease_release
