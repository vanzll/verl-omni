#!/usr/bin/env bash
set -eo pipefail
source /m2v_intern2/zhenglin/my_bashrc.sh
set -u
runtime=${OMNI_RUNTIME_ROOT:?}
control=${OMNI_CONTROL_DIR:?}
env_file=${1:?pass node env}
[[ "$(realpath "$env_file")" == "$runtime/attempts/"* ]] || exit 2
source "$env_file"
: "${OMNI_NODE_DESCRIPTOR:?}"
set +e
bash "$control/run_node_owner.sh" "$env_file" > "$OUTPUT_DIR/logs/owner_$OMNI_EXPECTED_HOST.log" 2>&1 &
owner_pid=$!
/usr/bin/python "$control/node_owner_control.py" identity "$OMNI_NODE_DESCRIPTOR" "$owner_pid"
identity_status=$?
if [[ "$identity_status" != 0 ]]; then
    # The child has not passed verify-self, so it cannot have freed GPUs.
    echo 'owner identity publication failed' >&2
fi
wait "$owner_pid"
status=$?
set -e
/usr/bin/python "$control/node_owner_control.py" exit "$OMNI_NODE_DESCRIPTOR" "$status"
exit "$status"
