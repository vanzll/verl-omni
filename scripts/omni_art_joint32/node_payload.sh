#!/usr/bin/env bash
# Runs inside the generation-fenced owner's one child process group.
set -eo pipefail
source /m2v_intern2/zhenglin/my_bashrc.sh
set -u

campaign=${OMNI_RUNTIME_ROOT:?}
control=${OMNI_CONTROL_DIR:?}
source "$control/use_source_snapshot.sh"
: "${OMNI_ROLE:?head or worker required}"
: "${OMNI_ATTEMPT:?}"
: "${OMNI_EXPECTED_IPS:?}"
: "${OMNI_NODE_IP:?}"
: "${OMNI_HEAD_IP:?}"
: "${OMNI_RAY_PORT:?}"
: "${OUTPUT_DIR:?}"
[[ "$OMNI_ROLE" == head || "$OMNI_ROLE" == worker ]] || exit 2

export PATH="${OMNI_CAMPAIGN_ROOT:?}/env/bin:$PATH"
source "$control/runtime_env.sh"
export RAY_ADDRESS="$OMNI_HEAD_IP:$OMNI_RAY_PORT"
: "${RAY_TMPDIR:?owner must supply a bounded Ray socket path}"
[[ "$(readlink -f "$RAY_TMPDIR")" == "$campaign"/* ]] || {
    echo 'Ray actual directory must stay under campaign' >&2
    exit 2
}
export TMPDIR="$RAY_TMPDIR"
# This shared mount reports pool-wide capacity and user-available free bytes.
# No capacity floor is enforced; Ray's percentage heuristic
# would reject spilling even with hundreds of GiB available to this campaign.
export RAY_local_fs_capacity_threshold=1.0
if [[ "${OMNI_VARIANT:-}" == NFT_FULL32_U* ]]; then
    source "$control/nft_cache_env.sh"
    export RAY_ROTATION_MAX_BYTES=67108864
    export RAY_ROTATION_BACKUP_COUNT=2
fi
mkdir -p "$OUTPUT_DIR/logs" "$RAY_TMPDIR" "$TMPDIR"
timeout --foreground 30 python "$control/transport.py" topology \
    > "$OUTPUT_DIR/logs/transport_topology_$(hostname -s).json"
status_file="$OUTPUT_DIR/allocation.status"
ray_log="$OUTPUT_DIR/logs/ray_$(hostname -s).log"
spill_dir="$campaign/tmp/ray_spill/$OMNI_ATTEMPT/$(hostname -s)"
mkdir -p "$spill_dir"
ray_cli_pid=""

finish() {
    local code=$?
    trap - EXIT
    if [[ "$OMNI_ROLE" == head && ! -e "$status_file" ]]; then
        printf 'exit_code=%s\nutc=%s\n' "$code" "$(date -u +%FT%TZ)" > "$status_file.tmp.$$"
        mv "$status_file.tmp.$$" "$status_file"
    fi
    if [[ -n "$ray_cli_pid" ]] && kill -0 "$ray_cli_pid" 2>/dev/null; then
        kill -TERM "$ray_cli_pid" 2>/dev/null || true
        wait "$ray_cli_pid" 2>/dev/null || true
    fi
    exit "$code"
}
trap finish EXIT

wait_for_head_port() {
    local deadline=$((SECONDS + 300))
    while (( SECONDS < deadline )); do
        if [[ -n "$ray_cli_pid" ]] && ! kill -0 "$ray_cli_pid" 2>/dev/null; then
            echo 'Ray CLI exited before its head port became ready' >&2
            return 1
        fi
        if timeout 3 python - <<'PY'
import os, socket
with socket.create_connection((os.environ['OMNI_HEAD_IP'], int(os.environ['OMNI_RAY_PORT'])), timeout=2):
    pass
PY
        then return 0; fi
        sleep 3
    done
    echo "Ray head port did not open within five minutes" >&2
    return 1
}

if [[ "$OMNI_ROLE" == head ]]; then
    [[ ! -e "$status_file" ]] || { echo "attempt status already exists" >&2; exit 2; }
    echo "[$(date -u +%FT%TZ)] starting allocation Ray head $RAY_ADDRESS"
    ray start --head --port="$OMNI_RAY_PORT" --node-ip-address="$OMNI_NODE_IP" \
        --num-gpus=8 --include-dashboard=false --disable-usage-stats \
        --min-worker-port=20000 --max-worker-port=29999 \
        --object-store-memory="${OMNI_RAY_OBJECT_STORE_BYTES:?}" --plasma-directory=/dev/shm \
        --object-spilling-directory="$spill_dir" \
        --temp-dir="$RAY_TMPDIR" --block > "$ray_log" 2>&1 &
    ray_cli_pid=$!
    wait_for_head_port
    ready=0
    ready_deadline=$((SECONDS + 300))
    while (( SECONDS < ready_deadline )); do
        if ! kill -0 "$ray_cli_pid" 2>/dev/null; then
            echo "Ray head exited before four-node readiness" >&2
            exit 1
        fi
        if timeout --foreground 30 python "$control/gate32.py" readiness >> "$OUTPUT_DIR/logs/ray_readiness.log" 2>&1
        then ready=1; break; fi
        sleep 10
    done
    [[ "$ready" == 1 ]] || { echo "Ray four-node cluster not ready after five minutes" >&2; exit 1; }
    export OMNI_GPU_RESERVATION_START_UNIX_MEAN
    OMNI_GPU_RESERVATION_START_UNIX_MEAN=$(python "$control/record_gpu_reservation.py" mean-start)
    echo "[$(date -u +%FT%TZ)] allocation ready; launching $OMNI_VARIANT"
    timeout --foreground 240 python "$control/gate32.py" collective
    case "$OMNI_VARIANT" in
        NFT_FULL32_U1)
            bash "$SOURCE_CHECKOUT_DIR/examples/omninft_trainer/ltx2/run_ltx2_t2av_nft_full32_u1.sh"
            ;;
        NFT_FULL32_U2)
            bash "$SOURCE_CHECKOUT_DIR/examples/omninft_trainer/ltx2/run_ltx2_t2av_nft_full32_u2.sh"
            ;;
        ART_FULL32_ROUTED)
            bash "$SOURCE_CHECKOUT_DIR/examples/diffusionart_trainer/ltx2/run_ltx2_t2av_art_routed_full32.sh"
            ;;
        ART_FULL32|A12JOINT32)
            bash "$SOURCE_CHECKOUT_DIR/examples/diffusionart_trainer/ltx2/run_ltx2_t2av_art_joint32.sh"
            ;;
        *) echo "Unsupported training variant: $OMNI_VARIANT" >&2; exit 2 ;;
    esac

else
    wait_for_head_port
    echo "[$(date -u +%FT%TZ)] joining Ray head $RAY_ADDRESS"
    ray start --address="$RAY_ADDRESS" --node-ip-address="$OMNI_NODE_IP" \
        --num-gpus=8 --disable-usage-stats --object-store-memory="${OMNI_RAY_OBJECT_STORE_BYTES:?}" \
        --min-worker-port=20000 --max-worker-port=29999 \
        --plasma-directory=/dev/shm --object-spilling-directory="$spill_dir" \
        --temp-dir="$RAY_TMPDIR" --block > "$ray_log" 2>&1 &
    ray_cli_pid=$!
    while [[ ! -e "$status_file" ]]; do
        if ! kill -0 "$ray_cli_pid" 2>/dev/null; then
            echo "Ray worker exited before head finished" >&2
            exit 1
        fi
        sleep 10
    done
    echo "[$(date -u +%FT%TZ)] head published allocation status"
fi
