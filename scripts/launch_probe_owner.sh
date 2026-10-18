#!/usr/bin/env bash
set -eo pipefail
source /m2v_intern2/zhenglin/my_bashrc.sh
set -u
campaign=/m2v_intern/zhenglin/omni_art_ablation_20261016
env_file=${1:?}
[[ "$(realpath "$env_file")" == "$campaign/preflight/runtime_gpu/"* ]] || exit 2
source "$env_file"
[[ "$OUTPUT_DIR" == "$campaign/preflight/runtime_gpu/"* ]] || exit 2
set +e
if [[ -n "${OMNI_NODE_DESCRIPTOR:-}" ]]; then
    mkdir -p "$OUTPUT_DIR/logs"
    bash "$campaign/scripts/run_node_owner.sh" "$env_file" > "$OUTPUT_DIR/owner.log" 2>&1 &
    owner_pid=$!
    /usr/bin/python "$campaign/scripts/node_owner_control.py" identity "$OMNI_NODE_DESCRIPTOR" "$owner_pid"
    if [[ "$?" != 0 ]]; then
        # The unreaped child PID cannot be reused. Stop this child only.
        kill -TERM "$owner_pid" 2>/dev/null || true
    fi
    wait "$owner_pid"
    status=$?
    /usr/bin/python "$campaign/scripts/node_owner_control.py" exit "$OMNI_NODE_DESCRIPTOR" "$status"
else
    bash "$campaign/scripts/run_node_owner.sh" "$env_file" > "$OUTPUT_DIR/owner.log" 2>&1
    status=$?
fi
set -e
printf '%s\n' "$status" > "$OUTPUT_DIR/owner_exit_code"
exit "$status"
