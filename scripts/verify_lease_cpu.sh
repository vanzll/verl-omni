#!/usr/bin/env bash
# CPU-only checks: all occupancy calls are stubs; real lease roots are untouched.
set -euo pipefail
source /m2v_intern2/zhenglin/my_bashrc.sh
campaign=/m2v_intern/zhenglin/omni_art_ablation_20261016
export TMPDIR="$campaign/tmp"
export FLOWART_GPU_LEASE_ROOT
FLOWART_GPU_LEASE_ROOT=$(mktemp -d "$campaign/tmp/lease-cpu.XXXXXXXX")
unalias gpu_free gpu_hold 2>/dev/null || true
gpu_free() { printf 'free\n' >> "$FLOWART_GPU_LEASE_ROOT/stub_calls"; }
gpu_hold() { printf 'hold\n' >> "$FLOWART_GPU_LEASE_ROOT/stub_calls"; }
source "$campaign/scripts/gpu_lease_lib.sh"
flowart_gpu_lease_configure cpu-only-host "$campaign" cpu-smoke cpu-nonce
trap 'flowart_gpu_lease_cleanup_owned_group; flowart_gpu_lease_release' EXIT
flowart_gpu_lease_acquire
[[ "$FLOWART_GPU_LEASE_GENERATION" == 1 ]]
printf 'lease-root=%s\n' "$FLOWART_GPU_LEASE_ROOT"

# A second owner must be rejected before invoking even the mocked free.
if bash -c '
  set -euo pipefail
  source "$1/scripts/gpu_lease_lib.sh"
  gpu_free() { echo unexpected-free >> "$FLOWART_GPU_LEASE_ROOT/stub_calls"; }
  flowart_gpu_lease_configure cpu-only-host "$1" contender other-nonce
  flowart_gpu_lease_acquire
' bash "$campaign"; then
  echo 'ERROR: live owner was not fenced' >&2
  exit 1
fi
[[ "$(cat "$FLOWART_GPU_LEASE_ROOT/generation")" == 1 ]]

# Leader exits successfully with a live child: group must be drained and marked failure.
set +e
flowart_gpu_lease_run_owned_command orphan-smoke bash -c \
  'sleep 90 & echo $! > "$1/orphan.pid"; exit 0' bash "$FLOWART_GPU_LEASE_ROOT"
status=$?
set -e
[[ "$status" == 1 ]]
orphan=$(cat "$FLOWART_GPU_LEASE_ROOT/orphan.pid")
[[ ! -r "/proc/$orphan/stat" || "$(awk '{print $3}' "/proc/$orphan/stat")" == Z ]]
flowart_gpu_lease_release
[[ "$(cat "$FLOWART_GPU_LEASE_ROOT/stub_calls")" == $'free\nhold' ]]
printf 'PASS live-owner rejection, orphan cleanup, hold-after-drain\n'

# An obsolete generation must never restore occupancy over a successor.
flowart_gpu_lease_configure cpu-only-host "$campaign" stale-smoke stale-nonce
flowart_gpu_lease_acquire
[[ "$FLOWART_GPU_LEASE_GENERATION" == 2 ]]
sed -i 's/^GPU_LEASE_GENERATION=.*/GPU_LEASE_GENERATION=3/' "$FLOWART_GPU_LEASE_ROOT/owner.env"
flowart_gpu_lease_release
[[ "$(cat "$FLOWART_GPU_LEASE_ROOT/stub_calls")" == $'free\nhold\nfree' ]]
printf 'PASS stale-generation release does not restore occupancy\n'
for scenario in live-group quarantined; do
  FLOWART_GPU_LEASE_ROOT=$(mktemp -d "$campaign/tmp/lease-$scenario.XXXXXXXX")
  flowart_gpu_lease_configure cpu-only-host "$campaign" "$scenario" "$scenario-nonce"
  flowart_gpu_lease_acquire
  setsid sleep 90 &
  child_pid=$!
  FLOWART_GPU_LEASE_ACTIVE_PGID="$child_pid"
  flowart_gpu_lease_bind_process_group "$child_pid"
  if [[ "$scenario" == quarantined ]]; then
    flowart_gpu_lease_quarantine "$child_pid" cpu-injected-quarantine
    kill -TERM "$child_pid"
    wait "$child_pid" || true
  fi
  if [[ "$scenario" == live-group ]]; then
    if flowart_gpu_lease_release; then
      echo 'ERROR: live-group lease was released' >&2
      exit 1
    fi
  else
    # quarantine clears ownership, so release is a successful no-op.
    flowart_gpu_lease_release
  fi
  [[ -s "$FLOWART_GPU_LEASE_OWNER" ]]
  [[ "$(cat "$FLOWART_GPU_LEASE_ROOT/stub_calls")" == free ]]
  if [[ "$scenario" == live-group ]]; then
    kill -TERM "$child_pid"
    wait "$child_pid" || true
  fi
  FLOWART_GPU_LEASE_ACTIVE_PGID=""
  printf 'PASS %s prevents occupancy restoration\n' "$scenario"
done
printf 'CPU checks only; Ray process containment remains to be verified\n'
