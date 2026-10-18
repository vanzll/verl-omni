#!/usr/bin/env bash
# One persistent owner per allocation node; runtime-only probe uses no model assets.
set -eo pipefail
source /m2v_intern2/zhenglin/my_bashrc.sh
set -u

campaign=/m2v_intern/zhenglin/omni_art_ablation_20261016
env_file=${1:?pass the campaign-local node environment file}
[[ "$(realpath "$env_file")" == "$campaign"/* ]] || { echo "node env must be under campaign" >&2; exit 2; }
set -a
source "$env_file"
set +a
source "$campaign/scripts/gpu_lease_lib.sh"

: "${OMNI_EXPECTED_HOST:?}"
: "${OMNI_ATTEMPT:?}"
: "${OMNI_NONCE:?}"
: "${OMNI_HOLDER_PID:?}"
: "${OMNI_HOLDER_START_TICKS:?}"
: "${OMNI_NODE_IP:?}"
: "${OMNI_HEAD_IP:?}"
: "${OMNI_RAY_PORT:?}"
: "${OUTPUT_DIR:?}"
: "${SOURCE_SNAPSHOT_SHA256:?}"
source "$campaign/scripts/use_source_snapshot.sh"
operation=${OMNI_OPERATION:-training}
case "$operation" in
    training|runtime_probe) ;;
    *) echo 'unsupported owner operation' >&2; exit 2 ;;
esac
holder_kind=${OMNI_HOLDER_KIND:-platform}
[[ "$holder_kind" == platform || "$holder_kind" == campaign ]] || exit 2

[[ "$(hostname -s)" == "$OMNI_EXPECTED_HOST" ]] || { echo "wrong node" >&2; exit 2; }
[[ "$OMNI_HOLDER_PID" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ "$OMNI_HOLDER_START_TICKS" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ "$OMNI_RAY_PORT" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ "$OUTPUT_DIR" == "$campaign"/* ]] || { echo "output outside campaign" >&2; exit 2; }

export PATH="$campaign/env/bin:$PATH"
source "$campaign/scripts/runtime_env.sh"
export HF_HOME="$campaign/cache/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export TMPDIR="$campaign/tmp"
export TMUX_TMPDIR="$campaign/tmp"
export XDG_CACHE_HOME="$campaign/cache/xdg"
export TORCH_EXTENSIONS_DIR="$campaign/cache/torch_extensions"
export TRITON_CACHE_DIR="$campaign/cache/triton"
export VLLM_CACHE_ROOT="$campaign/cache/vllm"
export CUDA_CACHE_PATH="$campaign/cache/cuda"
export TORCHINDUCTOR_CACHE_DIR="$campaign/cache/torchinductor"
source "$campaign/scripts/ray_temp_env.sh"
export RAY_ADDRESS="$OMNI_HEAD_IP:$OMNI_RAY_PORT"
export GPU_OCCUPY_SCRIPT=/m2v_intern2/zhenglin/gpu_occupy.py
export GPU_OCCUPY_WORKDIR="$campaign"
export GPU_OCCUPY_LOG="$campaign/logs/gpu_occupy_$OMNI_EXPECTED_HOST.log"
export GPU_OCCUPY_SESSION="omni_hold_$OMNI_EXPECTED_HOST"
export GPU_OCCUPY_PYTHON=/usr/bin/python
[[ -x "$GPU_OCCUPY_PYTHON" ]] || { echo 'occupancy Python does not exist' >&2; exit 2; }
export FLOWART_GPU_LEASE_ROOT="/m2v_intern/zhenglin/paper_coordination/GPU_LEASES/$OMNI_EXPECTED_HOST"
mkdir -p "$TMPDIR" "$RAY_TMPDIR" "$OUTPUT_DIR/logs" "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" "$CUDA_CACHE_PATH" "$TORCHINDUCTOR_CACHE_DIR"

# The sourced bashrc has gpu_hold/gpu_free aliases; the lease must use our
# campaign-scoped controls, whose log and temporary paths stay on Intern.
unalias gpu_hold gpu_free 2>/dev/null || true
gpu_free() { "$campaign/bin/gpu_free"; }
gpu_hold() {
    LD_LIBRARY_PATH= "$campaign/bin/gpu_hold" || return 1
    local apps pid cmd
    for _ in {1..60}; do
        apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^$/d')"
        pid="$(printf '%s\n' "$apps" | sort -u)"
        if [[ "$(printf '%s\n' "$apps" | wc -l)" -eq 8 && "$pid" =~ ^[1-9][0-9]*$ && -r "/proc/$pid/cmdline" ]]; then
            cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
            if [[ "${cmd% }" == "$GPU_OCCUPY_PYTHON $GPU_OCCUPY_SCRIPT" && "$(readlink -f "/proc/$pid/cwd")" == "$campaign" ]]; then
                echo "verified campaign occupancy on all eight GPUs: pid=$pid"
                return 0
            fi
        fi
        sleep 1
    done
    echo 'campaign occupancy did not cover all eight GPUs within 60 seconds' >&2
    return 1
}

holder_is_exact() {
    [[ -r "/proc/$OMNI_HOLDER_PID/stat" ]] || return 1
    [[ "$(flowart_pid_start_ticks "$OMNI_HOLDER_PID")" == "$OMNI_HOLDER_START_TICKS" ]] || return 1
    local cmd
    cmd="$(tr '\0' ' ' < "/proc/$OMNI_HOLDER_PID/cmdline")"
    if [[ "$holder_kind" == platform ]]; then
        [[ "${cmd% }" == 'python /m2v_intern/yuanziyang/launch_scripts/run.py' ]]
    else
        [[ "${cmd% }" == "$GPU_OCCUPY_PYTHON $GPU_OCCUPY_SCRIPT" ]] || return 1
        [[ "$(readlink -f "/proc/$OMNI_HOLDER_PID/cwd")" == "$campaign" ]]
    fi
}

assert_only_authorized_holder_on_gpus() {
    local apps
    apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^$/d')"
    [[ "$(printf '%s\n' "$apps" | wc -l)" -eq 8 ]] || { echo "expected one holder on eight GPUs" >&2; return 1; }
    [[ "$(printf '%s\n' "$apps" | sort -u)" == "$OMNI_HOLDER_PID" ]] || {
        echo "another GPU compute process is present" >&2; return 1;
    }
}

stop_exact_authorized_holder() {
    holder_is_exact && assert_only_authorized_holder_on_gpus || return 1
    echo "[$(date -u +%FT%TZ)] stopping authorized holder pid=$OMNI_HOLDER_PID start=$OMNI_HOLDER_START_TICKS"
    kill -TERM "$OMNI_HOLDER_PID"
    for _ in {1..40}; do
        if ! holder_is_exact; then break; fi
        sleep 0.25
    done
    if holder_is_exact; then
        echo "authorized holder did not stop after TERM; sending KILL to same identity" >&2
        kill -KILL "$OMNI_HOLDER_PID"
    fi
    for _ in {1..40}; do
        if ! holder_is_exact; then break; fi
        sleep 0.25
    done
    if holder_is_exact; then echo "authorized holder remains live" >&2; return 1; fi
    for _ in {1..40}; do
        if [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^$/d')" ]]; then
            return 0
        fi
        sleep 0.25
    done
    {
        echo "GPU compute process remains after holder stop" >&2; return 1;
    }
}

cleanup_done=0
lease_acquired=0
trainer_pid=""
cleanup() {
    if [[ "$cleanup_done" == 0 ]]; then
        trap '' INT TERM HUP
        cleanup_done=1
        if flowart_gpu_lease_cleanup_owned_group; then
            if [[ "$lease_acquired" == 1 && "${FLOWART_GPU_LEASE_OWNED:-0}" != 1 ]]; then
                echo 'lease quarantined: retain reservation interval and skip occupancy' >&2
                exit 1
            fi
            if [[ "${FLOWART_GPU_LEASE_OWNED:-0}" == 1 ]]; then
                drained=0
                drain_deadline=$((SECONDS + 30))
                while (( SECONDS < drain_deadline )); do
                    if apps=$(timeout 5 nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits); then
                        if [[ -z "$apps" ]]; then drained=1; break; fi
                    fi
                    sleep 0.5
                done
                if [[ "$drained" != 1 ]]; then
                    echo 'GPU compute process remains after owned-group cleanup; retaining lease and skipping occupancy' >&2
                    owned_pgid=$(sed -n 's/^GPU_LEASE_PGID=//p' "$FLOWART_GPU_LEASE_OWNER")
                    if [[ "$owned_pgid" =~ ^[1-9][0-9]*$ ]]; then
                        flowart_gpu_lease_quarantine "$owned_pgid" gpu-compute-not-drained || true
                    fi
                    exit 1
                fi
            fi
            if [[ -e "$OUTPUT_DIR/logs/gpu_reservation_$OMNI_EXPECTED_HOST.json" ]]; then
                python "$campaign/scripts/record_gpu_reservation.py" stop || echo 'reservation stop log failed' >&2
            fi
            flowart_gpu_lease_release
        else
            echo "lease retained: owned Ray/trainer process group remains live" >&2
        fi
    fi
}
trap cleanup EXIT
trap 'flowart_gpu_lease_forward_signal INT 130' INT
trap 'flowart_gpu_lease_forward_signal TERM 143' TERM
trap 'flowart_gpu_lease_forward_signal TERM 129' HUP

holder_is_exact
assert_only_authorized_holder_on_gpus
if [[ "$operation" == training ]]; then
    : "${OMNI_NODE_DESCRIPTOR:?training owner descriptor is required}"
    for _ in {1..50}; do
        [[ -f "$OUTPUT_DIR/logs/owner_identity_$OMNI_EXPECTED_HOST.json" ]] && break
        sleep 0.1
    done
    python "$campaign/scripts/node_owner_control.py" verify-self "$OMNI_NODE_DESCRIPTOR" "$$"
    timeout 60 "$campaign/env/bin/python" "$campaign/scripts/check_assets.py"
else
    [[ -f "$campaign/logs/runtime_cu129_cpu_ready.json" ]] || exit 2
    [[ "$OMNI_RUNTIME_CUDA_VARIANT" == cu129 ]] || exit 2
    # Probe payloads and the working launch helpers must match the archived
    # files before any lease is acquired or occupancy is released.
    python - <<'PY'
import hashlib, json, os
from pathlib import Path
root = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')
code = Path(os.environ['OUTPUT_DIR']) / 'probe_code'
for name, expected in json.loads((code / 'sha256.json').read_text()).items():
    assert '/' not in name
    assert hashlib.sha256((code / name).read_bytes()).hexdigest() == expected
    assert hashlib.sha256((root / 'scripts' / name).read_bytes()).hexdigest() == expected
PY
fi
flowart_gpu_lease_configure "$OMNI_EXPECTED_HOST" "$campaign" "$OMNI_ATTEMPT" "$OMNI_NONCE"
flowart_gpu_lease_acquire
lease_acquired=1
if [[ "$holder_kind" == platform ]]; then
    stop_exact_authorized_holder
else
    # lease_acquire already called our campaign occupancy control. Never signal
    # a stale PID again after it has exited.
    ! holder_is_exact || { echo 'campaign holder remains live after gpu_free' >&2; exit 1; }
    clear=0
    for _ in {1..40}; do
        if [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^$/d')" ]]; then
            clear=1; break
        fi
        sleep 0.25
    done
    [[ "$clear" == 1 ]] || exit 1
fi
python "$campaign/scripts/record_gpu_reservation.py" start

set +e
if [[ "$operation" == runtime_probe ]]; then
    flowart_gpu_lease_run_owned_command omni-runtime-probe bash "$OUTPUT_DIR/probe_code/run_gpu_runtime_probe.sh"
else
    flowart_gpu_lease_run_owned_command omni-ray-node bash "$campaign/scripts/node_payload.sh"
fi
status=$?
set -e
exit "$status"
