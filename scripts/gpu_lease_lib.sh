#!/usr/bin/env bash

flowart_pid_start_ticks() {
  # /proc/<pid>/stat field 2 is a parenthesized comm that may contain spaces.
  # Strip it before selecting starttime (field 22, now field 20).
  sed -E 's/^[0-9]+ \(.*\) //' "/proc/$1/stat" 2>/dev/null | awk '{print $20}' || true
}

flowart_gpu_lease_owner_is_live() {
  local pid="${1:-}"
  local start_ticks="${2:-}"
  [[ "${pid}" =~ ^[1-9][0-9]*$ && "${start_ticks}" =~ ^[0-9]+$ ]] || return 1
  local current_start
  current_start="$(flowart_pid_start_ticks "${pid}")"
  [[ -n "${current_start}" && "${current_start}" == "${start_ticks}" ]]
}

flowart_gpu_lease_process_group_has_live_members() {
  local expected_pgid="${1:-}"
  [[ "${expected_pgid}" =~ ^[1-9][0-9]*$ ]] || return 1
  local process_table
  if ! process_table="$(ps -eo pgid=,stat=)"; then
    return 0
  fi
  printf '%s\n' "${process_table}" | awk -v expected="${expected_pgid}" '
    $1 == expected && $2 !~ /^Z/ { found = 1 }
    END { exit(found ? 0 : 1) }
  '
}

flowart_gpu_lease_log() {
  local event="$1"
  local reason="${2:-}"
  local log_path="${FLOWART_GPU_LEASE_EVENTS:?missing FLOWART_GPU_LEASE_EVENTS}"
  mkdir -p "$(dirname "${log_path}")"
  printf 'ts=%(%Y-%m-%dT%H:%M:%SZ)T event=%q host=%q campaign=%q attempt=%q nonce=%q generation=%q pid=%q reason=%q\n' \
    -1 "${event}" "${FLOWART_GPU_LEASE_HOST:-}" "${FLOWART_GPU_LEASE_CAMPAIGN:-}" \
    "${FLOWART_GPU_LEASE_ATTEMPT:-}" "${FLOWART_GPU_LEASE_NONCE:-}" \
    "${FLOWART_GPU_LEASE_GENERATION:-}" "$$" "${reason}" >> "${log_path}"
}

flowart_gpu_lease_configure() {
  FLOWART_GPU_LEASE_HOST="${1:?missing host}"
  FLOWART_GPU_LEASE_CAMPAIGN="${2:?missing campaign}"
  FLOWART_GPU_LEASE_ATTEMPT="${3:?missing attempt}"
  FLOWART_GPU_LEASE_NONCE="${4:?missing nonce}"
  FLOWART_GPU_LEASE_ROOT="${FLOWART_GPU_LEASE_ROOT:-/m2v_intern2/zhenglin/paper_coordination/GPU_LEASES/${FLOWART_GPU_LEASE_HOST}}"
  FLOWART_GPU_LEASE_OWNER="${FLOWART_GPU_LEASE_ROOT}/owner.env"
  FLOWART_GPU_LEASE_GENERATION_FILE="${FLOWART_GPU_LEASE_ROOT}/generation"
  FLOWART_GPU_LEASE_LOCK="${FLOWART_GPU_LEASE_ROOT}/lock"
  FLOWART_GPU_LEASE_EVENTS="${FLOWART_GPU_LEASE_ROOT}/events.log"
  FLOWART_GPU_LEASE_OWNED=0
  FLOWART_GPU_LEASE_GENERATION=""
  FLOWART_GPU_LEASE_ACTIVE_PGID=""
  FLOWART_GPU_LEASE_SPAWNING=0
  FLOWART_GPU_LEASE_PENDING_SIGNAL=""
}

flowart_gpu_lease_acquire() {
  : "${FLOWART_GPU_LEASE_ROOT:?call flowart_gpu_lease_configure first}"
  mkdir -p "${FLOWART_GPU_LEASE_ROOT}"
  exec 9>"${FLOWART_GPU_LEASE_LOCK}"
  flock 9
  local previous_generation=0
  local acquire_reason="owner-missing"
  if [[ -s "${FLOWART_GPU_LEASE_GENERATION_FILE}" ]]; then
    previous_generation="$(< "${FLOWART_GPU_LEASE_GENERATION_FILE}")"
  fi
  if [[ -s "${FLOWART_GPU_LEASE_OWNER}" ]]; then
    unset GPU_LEASE_SCHEMA GPU_LEASE_CAMPAIGN GPU_LEASE_ATTEMPT
    unset GPU_LEASE_EXPERIMENT_ID GPU_LEASE_NONCE GPU_LEASE_PID
    unset GPU_LEASE_START_TICKS GPU_LEASE_GENERATION GPU_LEASE_HOST
    unset GPU_LEASE_STATUS GPU_LEASE_PGID GPU_LEASE_QUARANTINE_REASON
    # shellcheck disable=SC1090
    source "${FLOWART_GPU_LEASE_OWNER}"
    if [[ ! -s "${FLOWART_GPU_LEASE_GENERATION_FILE}" ]]; then
      previous_generation="${GPU_LEASE_GENERATION:-0}"
    fi
    if [[ ! "${GPU_LEASE_PID:-}" =~ ^[1-9][0-9]*$ || ! "${GPU_LEASE_START_TICKS:-}" =~ ^[0-9]+$ ]]; then
      flowart_gpu_lease_log acquire-rejected-invalid-owner "missing-pid-or-start-ticks"
      echo "GPU lease owner record is missing a valid PID/start tick identity" >&2
      flock -u 9
      return 1
    fi
    case "${GPU_LEASE_STATUS:-active}" in
      active)
        if flowart_gpu_lease_owner_is_live "${GPU_LEASE_PID:-}" "${GPU_LEASE_START_TICKS:-}"; then
          FLOWART_GPU_LEASE_GENERATION="${GPU_LEASE_GENERATION:-unknown}"
          flowart_gpu_lease_log acquire-rejected-live-owner "owner-pid-${GPU_LEASE_PID:-unknown}"
          echo "GPU lease already has live owner pid=${GPU_LEASE_PID:-unknown} on ${FLOWART_GPU_LEASE_HOST}" >&2
          flock -u 9
          return 1
        fi
        if flowart_gpu_lease_process_group_has_live_members "${GPU_LEASE_PGID:-}"; then
          FLOWART_GPU_LEASE_GENERATION="${GPU_LEASE_GENERATION:-unknown}"
          flowart_gpu_lease_log acquire-rejected-live-process-group "owner-pgid-${GPU_LEASE_PGID}"
          echo "GPU lease owner exited but process group ${GPU_LEASE_PGID} is still live" >&2
          flock -u 9
          return 1
        fi
        acquire_reason="dead-owner-stale-takeover"
        ;;
      quarantined)
        if [[ ! "${GPU_LEASE_PGID:-}" =~ ^[1-9][0-9]*$ ]]; then
          flowart_gpu_lease_log acquire-rejected-invalid-owner "quarantine-missing-pgid"
          echo "quarantined GPU lease is missing a valid process-group identity" >&2
          flock -u 9
          return 1
        fi
        if flowart_gpu_lease_process_group_has_live_members "${GPU_LEASE_PGID}"; then
          FLOWART_GPU_LEASE_GENERATION="${GPU_LEASE_GENERATION:-unknown}"
          flowart_gpu_lease_log acquire-rejected-quarantined "live-pgid-${GPU_LEASE_PGID}"
          echo "GPU lease is quarantined while process group ${GPU_LEASE_PGID} remains live" >&2
          flock -u 9
          return 1
        fi
        acquire_reason="quarantine-drained-takeover"
        ;;
      *)
        flowart_gpu_lease_log acquire-rejected-invalid-owner "unknown-status-${GPU_LEASE_STATUS:-missing}"
        echo "GPU lease owner record has an unknown status" >&2
        flock -u 9
        return 1
        ;;
    esac
  fi
  if [[ ! "${previous_generation}" =~ ^[0-9]+$ ]]; then
    previous_generation=0
  fi
  FLOWART_GPU_LEASE_GENERATION=$((previous_generation + 1))
  local start_ticks
  start_ticks="$(flowart_pid_start_ticks $$)"
  {
    printf 'GPU_LEASE_SCHEMA=%q\n' 3
    printf 'GPU_LEASE_STATUS=%q\n' active
    printf 'GPU_LEASE_CAMPAIGN=%q\n' "${FLOWART_GPU_LEASE_CAMPAIGN}"
    printf 'GPU_LEASE_ATTEMPT=%q\n' "${FLOWART_GPU_LEASE_ATTEMPT}"
    printf 'GPU_LEASE_EXPERIMENT_ID=%q\n' "${FLOWART_GPU_LEASE_ATTEMPT}"
    printf 'GPU_LEASE_NONCE=%q\n' "${FLOWART_GPU_LEASE_NONCE}"
    printf 'GPU_LEASE_PID=%q\n' "$$"
    printf 'GPU_LEASE_START_TICKS=%q\n' "${start_ticks}"
    printf 'GPU_LEASE_GENERATION=%q\n' "${FLOWART_GPU_LEASE_GENERATION}"
    printf 'GPU_LEASE_HOST=%q\n' "${FLOWART_GPU_LEASE_HOST}"
  } > "${FLOWART_GPU_LEASE_OWNER}.tmp.$$"
  printf '%s\n' "${FLOWART_GPU_LEASE_GENERATION}" > "${FLOWART_GPU_LEASE_GENERATION_FILE}.tmp.$$"
  mv "${FLOWART_GPU_LEASE_GENERATION_FILE}.tmp.$$" "${FLOWART_GPU_LEASE_GENERATION_FILE}"
  mv "${FLOWART_GPU_LEASE_OWNER}.tmp.$$" "${FLOWART_GPU_LEASE_OWNER}"
  FLOWART_GPU_LEASE_OWNED=1
  flowart_gpu_lease_log acquire "${acquire_reason}"
  flock -u 9
  gpu_free
}

flowart_gpu_lease_bind_process_group() {
  local pgid="${1:?missing process group}"
  [[ "${FLOWART_GPU_LEASE_OWNED:-0}" == 1 && "${pgid}" =~ ^[1-9][0-9]*$ ]] || return 1
  exec 9>"${FLOWART_GPU_LEASE_LOCK}"
  flock 9
  if [[ ! -s "${FLOWART_GPU_LEASE_OWNER}" ]]; then
    flowart_gpu_lease_log process-group-bind-rejected "owner-missing"
    flock -u 9
    return 1
  fi
  unset GPU_LEASE_PID GPU_LEASE_START_TICKS GPU_LEASE_NONCE
  unset GPU_LEASE_GENERATION GPU_LEASE_STATUS GPU_LEASE_PGID
  # shellcheck disable=SC1090
  source "${FLOWART_GPU_LEASE_OWNER}"
  local current_start
  current_start="$(flowart_pid_start_ticks $$)"
  if [[ "${GPU_LEASE_PID:-}" != "$$" \
    || "${GPU_LEASE_START_TICKS:-}" != "${current_start}" \
    || "${GPU_LEASE_NONCE:-}" != "${FLOWART_GPU_LEASE_NONCE}" \
    || "${GPU_LEASE_GENERATION:-}" != "${FLOWART_GPU_LEASE_GENERATION}" \
    || "${GPU_LEASE_STATUS:-active}" != active ]]; then
    flowart_gpu_lease_log process-group-bind-rejected "owner-generation-mismatch"
    flock -u 9
    return 1
  fi
  if ! sed -E "/^GPU_LEASE_PGID=/d" "${FLOWART_GPU_LEASE_OWNER}" \
    > "${FLOWART_GPU_LEASE_OWNER}.tmp.$$" \
    || ! printf 'GPU_LEASE_PGID=%q\n' "${pgid}" >> "${FLOWART_GPU_LEASE_OWNER}.tmp.$$" \
    || ! mv "${FLOWART_GPU_LEASE_OWNER}.tmp.$$" "${FLOWART_GPU_LEASE_OWNER}"; then
    rm -f "${FLOWART_GPU_LEASE_OWNER}.tmp.$$"
    flowart_gpu_lease_log process-group-bind-rejected "owner-publication-failed"
    flock -u 9
    return 1
  fi
  flowart_gpu_lease_log process-group-bound "pgid-${pgid}"
  flock -u 9
}

flowart_gpu_lease_quarantine() {
  local pgid="${1:?missing process group}"
  local quarantine_reason="${2:-owned-process-group-remained-live}"
  [[ "${FLOWART_GPU_LEASE_OWNED:-0}" == 1 && "${pgid}" =~ ^[1-9][0-9]*$ ]] || return 1
  exec 9>"${FLOWART_GPU_LEASE_LOCK}"
  flock 9
  if [[ ! -s "${FLOWART_GPU_LEASE_OWNER}" ]]; then
    flowart_gpu_lease_log quarantine-rejected "owner-missing"
    flock -u 9
    return 1
  fi
  unset GPU_LEASE_PID GPU_LEASE_START_TICKS GPU_LEASE_NONCE
  unset GPU_LEASE_GENERATION GPU_LEASE_STATUS GPU_LEASE_PGID
  # shellcheck disable=SC1090
  source "${FLOWART_GPU_LEASE_OWNER}"
  local current_start
  current_start="$(flowart_pid_start_ticks $$)"
  if [[ "${GPU_LEASE_PID:-}" != "$$" \
    || "${GPU_LEASE_START_TICKS:-}" != "${current_start}" \
    || "${GPU_LEASE_NONCE:-}" != "${FLOWART_GPU_LEASE_NONCE}" \
    || "${GPU_LEASE_GENERATION:-}" != "${FLOWART_GPU_LEASE_GENERATION}" ]]; then
    flowart_gpu_lease_log quarantine-rejected "owner-generation-mismatch"
    flock -u 9
    return 1
  fi
  if ! sed -E "/^GPU_LEASE_(STATUS|PGID|QUARANTINE_REASON)=/d" \
    "${FLOWART_GPU_LEASE_OWNER}" > "${FLOWART_GPU_LEASE_OWNER}.tmp.$$" \
    || ! printf 'GPU_LEASE_STATUS=%q\n' quarantined >> "${FLOWART_GPU_LEASE_OWNER}.tmp.$$" \
    || ! printf 'GPU_LEASE_PGID=%q\n' "${pgid}" >> "${FLOWART_GPU_LEASE_OWNER}.tmp.$$" \
    || ! printf 'GPU_LEASE_QUARANTINE_REASON=%q\n' "${quarantine_reason}" \
      >> "${FLOWART_GPU_LEASE_OWNER}.tmp.$$" \
    || ! mv "${FLOWART_GPU_LEASE_OWNER}.tmp.$$" "${FLOWART_GPU_LEASE_OWNER}"; then
    rm -f "${FLOWART_GPU_LEASE_OWNER}.tmp.$$"
    flowart_gpu_lease_log quarantine-rejected "owner-publication-failed"
    flock -u 9
    return 1
  fi
  FLOWART_GPU_LEASE_OWNED=0
  flowart_gpu_lease_log quarantine "${quarantine_reason}-pgid-${pgid}"
  flock -u 9
}

flowart_gpu_lease_wait_for_group_drain() {
  local pgid="$1" timeout_seconds="$2"
  local deadline=$((SECONDS + timeout_seconds))
  while flowart_gpu_lease_process_group_has_live_members "${pgid}" \
    && (( SECONDS < deadline )); do
    sleep 1
  done
  ! flowart_gpu_lease_process_group_has_live_members "${pgid}"
}

flowart_gpu_lease_fence_process_group() {
  local pgid="$1" signal="$2" context="$3"
  if ! flowart_gpu_lease_process_group_has_live_members "${pgid}"; then
    wait "${pgid}" 2>/dev/null || true
    return 0
  fi
  kill -"${signal}" -- "-${pgid}" 2>/dev/null || true
  if ! flowart_gpu_lease_wait_for_group_drain "${pgid}" 30; then
    echo "${context} process group did not stop after 30s; escalating to KILL" >&2
    kill -KILL -- "-${pgid}" 2>/dev/null || true
    if ! flowart_gpu_lease_wait_for_group_drain "${pgid}" 10; then
      if flowart_gpu_lease_quarantine "${pgid}" "${context// /-}-survived-kill"; then
        FLOWART_GPU_LEASE_ACTIVE_PGID=""
        trainer_pid=""
        return 0
      fi
      echo "${context} process group remains live and lease quarantine failed" >&2
      return 1
    fi
  fi
  wait "${pgid}" 2>/dev/null || true
  return 0
}

flowart_gpu_lease_cleanup_owned_group() {
  [[ -n "${FLOWART_GPU_LEASE_ACTIVE_PGID:-}" ]] || return 0
  if ! flowart_gpu_lease_fence_process_group \
    "${FLOWART_GPU_LEASE_ACTIVE_PGID}" TERM "wrapper-cleanup"; then
    return 1
  fi
  FLOWART_GPU_LEASE_ACTIVE_PGID=""
  trainer_pid=""
}

flowart_gpu_lease_forward_signal() {
  local signal="$1" status="$2"
  if [[ "${FLOWART_GPU_LEASE_SPAWNING:-0}" == 1 \
    && -z "${FLOWART_GPU_LEASE_ACTIVE_PGID:-}" ]]; then
    FLOWART_GPU_LEASE_PENDING_SIGNAL="${signal}:${status}"
    return 0
  fi
  if [[ -n "${FLOWART_GPU_LEASE_ACTIVE_PGID:-}" ]]; then
    if ! flowart_gpu_lease_fence_process_group \
      "${FLOWART_GPU_LEASE_ACTIVE_PGID}" "${signal}" "signal-${signal}"; then
      exit 1
    fi
    FLOWART_GPU_LEASE_ACTIVE_PGID=""
    trainer_pid=""
  fi
  exit "${status}"
}

flowart_gpu_lease_deliver_pending_signal() {
  [[ -n "${FLOWART_GPU_LEASE_PENDING_SIGNAL:-}" ]] || return 0
  local signal="${FLOWART_GPU_LEASE_PENDING_SIGNAL%%:*}"
  local status="${FLOWART_GPU_LEASE_PENDING_SIGNAL##*:}"
  FLOWART_GPU_LEASE_PENDING_SIGNAL=""
  flowart_gpu_lease_forward_signal "${signal}" "${status}"
}

flowart_gpu_lease_run_owned_command() {
  local context="${1:?missing child context}"
  shift
  local gate_dir="${FLOWART_GPU_LEASE_ROOT}/start-gates"
  local safe_context="${context//[^A-Za-z0-9_.-]/_}"
  local gate_path="${gate_dir}/${FLOWART_GPU_LEASE_NONCE}.${safe_context}.$$.gate"
  local child_status=0 pgid
  mkdir -p "${gate_dir}"
  rm -f "${gate_path}" "${gate_path}.tmp.$$"
  FLOWART_GPU_LEASE_SPAWNING=1
  setsid bash -c '
    gate_path="$1" parent_pid="$2"; shift 2
    deadline=$((SECONDS + 30))
    while [[ ! -s "${gate_path}" ]]; do
      kill -0 "${parent_pid}" 2>/dev/null || exit 125
      (( SECONDS < deadline )) || exit 125
      sleep 0.05
    done
    exec "$@"
  ' flowart-gated-child "${gate_path}" "$$" "$@" &
  pgid=$!
  FLOWART_GPU_LEASE_ACTIVE_PGID="${pgid}"
  trainer_pid="${pgid}"
  if ! flowart_gpu_lease_bind_process_group "${pgid}"; then
    FLOWART_GPU_LEASE_SPAWNING=0
    flowart_gpu_lease_fence_process_group "${pgid}" TERM "lease-bind-failure" || true
    return 2
  fi
  if ! printf 'start\n' > "${gate_path}.tmp.$$" \
    || ! mv "${gate_path}.tmp.$$" "${gate_path}"; then
    FLOWART_GPU_LEASE_SPAWNING=0
    flowart_gpu_lease_fence_process_group "${pgid}" TERM "gate-publication-failure" || true
    return 2
  fi
  FLOWART_GPU_LEASE_SPAWNING=0
  flowart_gpu_lease_deliver_pending_signal
  if wait "${pgid}"; then
    child_status=0
  else
    child_status=$?
  fi
  if flowart_gpu_lease_process_group_has_live_members "${pgid}"; then
    echo "${context} leader exited while its process group remained alive" >&2
    if ! flowart_gpu_lease_fence_process_group "${pgid}" TERM "${context}-orphan-cleanup"; then
      child_status=1
    elif [[ "${child_status}" == 0 ]]; then
      child_status=1
    fi
  fi
  FLOWART_GPU_LEASE_ACTIVE_PGID=""
  trainer_pid=""
  rm -f "${gate_path}" "${gate_path}.tmp.$$"
  return "${child_status}"
}

flowart_gpu_lease_release() {
  [[ "${FLOWART_GPU_LEASE_OWNED:-0}" == 1 ]] || return 0
  FLOWART_GPU_LEASE_OWNED=0
  exec 9>"${FLOWART_GPU_LEASE_LOCK}"
  flock 9
  local reason="owner-missing"
  if [[ -s "${FLOWART_GPU_LEASE_OWNER}" ]]; then
    # shellcheck disable=SC1090
    source "${FLOWART_GPU_LEASE_OWNER}"
    local current_start
    current_start="$(flowart_pid_start_ticks $$)"
    if [[ "${GPU_LEASE_PID:-}" == "$$" \
      && "${GPU_LEASE_START_TICKS:-}" == "${current_start}" \
      && "${GPU_LEASE_NONCE:-}" == "${FLOWART_GPU_LEASE_NONCE}" \
      && "${GPU_LEASE_GENERATION:-}" == "${FLOWART_GPU_LEASE_GENERATION}" ]]; then
      if [[ "${GPU_LEASE_STATUS:-active}" != active ]] \
        || flowart_gpu_lease_process_group_has_live_members "${GPU_LEASE_PGID:-}"; then
        flowart_gpu_lease_log release-rejected-not-drained "status-${GPU_LEASE_STATUS:-active}-pgid-${GPU_LEASE_PGID:-none}"
        echo 'lease retained: quarantined or live owned process group; skip gpu_hold' >&2
        flock -u 9
        return 1
      fi
      rm -f "${FLOWART_GPU_LEASE_OWNER}"
      flowart_gpu_lease_log release-gpu-hold "owner-generation-current"
      # Keep the generation lock until gpu_hold completes. A newer owner then
      # acquires after this call and its gpu_free is guaranteed to run last.
      if ! (unset LD_PRELOAD; exec 9>&-; gpu_hold); then
        flowart_gpu_lease_log release-gpu-hold-failed "occupancy-restoration-failed"
        flock -u 9
        return 1
      fi
      flock -u 9
      return 0
    fi
    reason="superseded-by-generation-${GPU_LEASE_GENERATION:-unknown}"
  fi
  flowart_gpu_lease_log stale-cleanup-noop "${reason}"
  echo "stale-cleanup-noop: skip gpu_hold: a newer GPU lease owns ${FLOWART_GPU_LEASE_HOST}" >&2
  flock -u 9
}
