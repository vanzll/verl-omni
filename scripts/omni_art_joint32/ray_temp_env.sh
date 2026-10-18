#!/usr/bin/env bash
# A short symlink keeps AF_UNIX socket names bounded; actual data remains in the campaign.
ray_runtime=${OMNI_RUNTIME_ROOT:?}
: "${OMNI_ATTEMPT:?}"
: "${OMNI_EXPECTED_HOST:?}"
: "${OMNI_NONCE:?}"
for ray_identifier in "$OMNI_ATTEMPT" "$OMNI_EXPECTED_HOST" "$OMNI_NONCE"; do
  [[ "$ray_identifier" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || {
    echo 'Ray attempt, host and nonce must be plain path identifiers' >&2
    return 1
  }
done
actual_ray_dir="$ray_runtime/tmp/ray/$OMNI_ATTEMPT/$OMNI_EXPECTED_HOST"
ray_path_identity=$(printf '%s\n' "$OMNI_ATTEMPT" "$OMNI_EXPECTED_HOST" "$OMNI_NONCE" | sha256sum)
ray_path_identity=${ray_path_identity:0:10}
export RAY_TMPDIR="${OMNI_SHORT_PATH_ROOT:?}/r$ray_path_identity"
mkdir -p "$actual_ray_dir"
if [[ -e "$RAY_TMPDIR" || -L "$RAY_TMPDIR" ]]; then
  [[ -L "$RAY_TMPDIR" && "$(readlink -f "$RAY_TMPDIR")" == "$actual_ray_dir" ]] || {
    echo 'Ray short-path identity collision' >&2
    return 1
  }
else
  ln -s "$actual_ray_dir" "$RAY_TMPDIR"
fi
export TMPDIR="$RAY_TMPDIR"
export VLLM_RPC_BASE_PATH="$RAY_TMPDIR"
