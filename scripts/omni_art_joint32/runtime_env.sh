#!/usr/bin/env bash
# Keep the existing CUDA/runtime/cache selection; never select an old source.
_a3_runtime_env() {
  local campaign
  source "${OMNI_CAMPAIGN_ROOT:?}/scripts/runtime_env.sh"
}
_a3_runtime_env
unset -f _a3_runtime_env
source "$(dirname "${BASH_SOURCE[0]}")/../nccl_transport_env.sh"
export no_proxy="${no_proxy:-localhost,127.0.0.1},${OMNI_EXPECTED_IPS:?}"
export NO_PROXY="$no_proxy"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_RESUME=never
