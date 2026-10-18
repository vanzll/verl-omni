#!/usr/bin/env bash
# Runs inside the generation-fenced owner's one child process group.
set -euo pipefail
source /m2v_intern2/zhenglin/my_bashrc.sh

campaign=/m2v_intern/zhenglin/omni_art_ablation_20261016
source "$campaign/scripts/use_source_snapshot.sh"
: "${OMNI_ROLE:?head or worker required}"
: "${OMNI_ATTEMPT:?}"
: "${OMNI_EXPECTED_PEER_IP:?}"
: "${OMNI_NODE_IP:?}"
: "${OMNI_HEAD_IP:?}"
: "${OMNI_RAY_PORT:?}"
: "${OUTPUT_DIR:?}"
[[ "$OMNI_ROLE" == head || "$OMNI_ROLE" == worker ]] || exit 2

export PATH="$campaign/env/bin:$PATH"
source "$campaign/scripts/runtime_env.sh"
export RAY_ADDRESS="$OMNI_HEAD_IP:$OMNI_RAY_PORT"
: "${RAY_TMPDIR:?owner must supply a bounded Ray socket path}"
[[ "$(readlink -f "$RAY_TMPDIR")" == "$campaign"/* ]] || {
    echo 'Ray actual directory must stay under campaign' >&2
    exit 2
}
export TMPDIR="$campaign/tmp"
# This shared mount reports pool-wide capacity and user-available free bytes.
# The coordinator enforces an absolute 250GiB floor; Ray's percentage heuristic
# would reject spilling even with hundreds of GiB available to this campaign.
export RAY_local_fs_capacity_threshold=1.0
mkdir -p "$OUTPUT_DIR/logs" "$RAY_TMPDIR" "$TMPDIR"
status_file="$OUTPUT_DIR/node_pair.status"
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
    echo "[$(date -u +%FT%TZ)] starting pair Ray head $RAY_ADDRESS"
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
            echo "Ray head exited before two-node readiness" >&2
            exit 1
        fi
        if timeout 25 python - <<'PY' >> "$OUTPUT_DIR/logs/ray_pair_readiness.log" 2>&1
import os
import json
import socket
from pathlib import Path
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

expected = {os.environ['OMNI_NODE_IP'], os.environ['OMNI_EXPECTED_PEER_IP']}
ray.init(address=os.environ['RAY_ADDRESS'], namespace=os.environ['OMNI_ATTEMPT'], logging_level='ERROR')
nodes = [node for node in ray.nodes() if node.get('Alive')]
actual = {node['NodeManagerAddress'] for node in nodes}
if actual != expected or len(nodes) != 2:
    raise RuntimeError(f'pair nodes={actual}, expected={expected}')
for node in nodes:
    if node['Resources'].get('GPU', 0) != 8:
        raise RuntimeError(f"node {node['NodeManagerAddress']} has GPU={node['Resources'].get('GPU')}")
@ray.remote(num_cpus=0.1, num_gpus=0)
def node_location():
    return {'hostname': socket.gethostname().split('.')[0],
            'temp_dir': ray._private.worker._global_node.get_temp_dir_path()}

locations = ray.get([node_location.options(scheduling_strategy=NodeAffinitySchedulingStrategy(
    node_id=node['NodeID'], soft=False)).remote() for node in nodes], timeout=15)
manifest = json.loads((Path(os.environ['OUTPUT_DIR']) / 'pair_manifest.json').read_text())
expected_hosts = {node['ip']: node['hostname'] for node in manifest['nodes']}
for node, location in zip(nodes, locations, strict=True):
    host = expected_hosts[node['NodeManagerAddress']]
    required = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016/tmp/ray') / os.environ['OMNI_ATTEMPT'] / host
    if location['hostname'] != host or Path(location['temp_dir']).resolve() != required.resolve():
        raise RuntimeError(f'Ray node temp directory is not host-specific: {location}, required={required}')
print('verified per-node Ray directories:', json.dumps(locations))
print(f"ready Ray pair: {sorted(actual)}; GPUs=16")
ray.shutdown()
PY
        then ready=1; break; fi
        sleep 10
    done
    [[ "$ready" == 1 ]] || { echo "Ray pair not ready after five minutes" >&2; exit 1; }
    export OMNI_GPU_RESERVATION_START_UNIX_MEAN
    OMNI_GPU_RESERVATION_START_UNIX_MEAN=$(python "$campaign/scripts/record_gpu_reservation.py" mean-start)
    echo "[$(date -u +%FT%TZ)] pair ready; launching $OMNI_VARIANT"
    resume_args=()
    if [[ -n "${OMNI_RESUME_FROM_PATH:-}" ]]; then
        [[ "$(realpath "$OMNI_RESUME_FROM_PATH")" == "$campaign/attempts/"*/checkpoints/global_step_* ]] || exit 2
        resume_args=(trainer.resume_mode=resume_path "trainer.resume_from_path=$OMNI_RESUME_FROM_PATH"
            "++trainer.resume_extra_end_to_end_gpu_seconds=${OMNI_RESUME_EXTRA_GPU_SECONDS:-0}")
    fi
    case "${OMNI_VARIANT:?}" in
        N0) bash "$campaign/scripts/run_n0.sh" "${resume_args[@]}" ;;
        A0|A1|A2) VARIANT="$OMNI_VARIANT" bash "$campaign/scripts/run_art.sh" "${resume_args[@]}" ;;
        *) echo "invalid variant $OMNI_VARIANT" >&2; exit 2 ;;
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
    echo "[$(date -u +%FT%TZ)] head published pair status"
fi
