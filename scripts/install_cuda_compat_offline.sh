#!/usr/bin/env bash
set -euo pipefail
source /m2v_intern2/zhenglin/my_bashrc.sh

campaign=/m2v_intern/zhenglin/omni_art_ablation_20261016
package="$campaign/cache/conda/cuda-compat-impl-610.43.02-h676940d_1.conda"
staged="$campaign/cache/conda/cuda-compat-extracted/cuda-compat"
link="$campaign/env/cuda-compat"

[[ -f "$package" && "$(stat -c '%s' "$package")" == 109517566 ]] || {
  echo 'staged CUDA compatibility package is missing or truncated' >&2
  exit 1
}
for library in libcuda.so.1 libnvidia-ptxjitcompiler.so.1 libnvidia-nvvm.so.4; do
  [[ -f "$staged/$library" ]] || { echo "missing $library in staged package" >&2; exit 1; }
done
if [[ -e "$link" || -L "$link" ]]; then
  [[ "$(readlink -f "$link")" == "$staged" ]] || {
    echo 'env/cuda-compat already points elsewhere' >&2
    exit 1
  }
else
  ln -s ../cache/conda/cuda-compat-extracted/cuda-compat "$link"
fi
printf 'CUDA compatibility libraries linked into env: %s\n' "$(readlink -f "$link")"
sha256sum "$package"
