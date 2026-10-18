#!/usr/bin/env bash
# Keep the existing CUDA/runtime/cache selection; never select an old source.
_a3_runtime_env() {
  local campaign
  source "${OMNI_CAMPAIGN_ROOT:?}/scripts/runtime_env.sh"
}
_a3_runtime_env
unset -f _a3_runtime_env
if [[ "${OMNI_FULL_FINETUNE:-0}" == 1 ]]; then
  # Full sharding must not inherit the replicated-LoRA transport workaround.
  export NCCL_NET=IB NCCL_NET_PLUGIN=none NCCL_IB_DISABLE=0 NCCL_P2P_DISABLE=0
  export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
  export NCCL_DEBUG_FILE="${OUTPUT_DIR:?}/logs/nccl.%h.%p.log"
fi
export no_proxy="${no_proxy:-localhost,127.0.0.1},${OMNI_EXPECTED_IPS:?}"
export NO_PROXY="$no_proxy"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_RESUME=never
