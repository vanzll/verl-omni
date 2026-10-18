#!/usr/bin/env bash
set -euo pipefail
source /m2v_intern2/zhenglin/my_bashrc.sh
campaign=/m2v_intern/zhenglin/omni_art_ablation_20261016
cache="$campaign/cache/conda"
base=cuda-compat-impl-610.43.02-h676940d_1.conda
partial="$cache/$base.partial"
final="$cache/$base"
url="https://conda.anaconda.org/conda-forge/linux-64/$base"
expected=109517566

for attempt in $(seq 1 20); do
  size=$(stat -c '%s' "$partial" 2>/dev/null || printf 0)
  printf '[%s] attempt=%s bytes=%s/%s\n' "$(date -u +%FT%TZ)" "$attempt" "$size" "$expected"
  if ((size == expected)); then
    mv "$partial" "$final"
    printf '[%s] CUDA compat package ready: %s\n' "$(date -u +%FT%TZ)" "$final"
    exit 0
  fi
  if ((size > expected)); then
    echo "CUDA compat partial exceeds expected size" >&2
    exit 1
  fi
  timeout 1800 curl --fail --location --silent --show-error \
    --connect-timeout 20 --max-time 1500 --continue-at - \
    --output "$partial" "$url" || true
  sleep 10
done
echo 'CUDA compat download exhausted retry budget' >&2
exit 1
