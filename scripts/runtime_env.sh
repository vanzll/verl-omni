#!/usr/bin/env bash
# Source after my_bashrc.sh. Keep the campaign runtime isolated from image Torch/TRT libraries.
campaign=/m2v_intern/zhenglin/omni_art_ablation_20261016
if [[ -z "${OMNI_RUNTIME_CUDA_VARIANT:-}" ]]; then
  if [[ ! -f "$campaign/assets/runtime-selected.env" ]]; then
    echo 'No campaign CUDA runtime has been selected' >&2
    return 1
  fi
  source "$campaign/assets/runtime-selected.env"
fi
case "$OMNI_RUNTIME_CUDA_VARIANT" in
  cu129)
    export LD_LIBRARY_PATH="$campaign/env/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
    ;;
  cu130)
    export LD_LIBRARY_PATH="$campaign/env/cuda-compat:$campaign/env/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
    ;;
  *)
    echo "Unsupported campaign CUDA variant: $OMNI_RUNTIME_CUDA_VARIANT" >&2
    return 1
    ;;
esac
export LIBRARY_PATH="$LD_LIBRARY_PATH"
export PATH="$campaign/env/bin:$PATH"
# The native executor applies this finite request deadline even at DP=1.
# The first LTX/Inductor compile on shared storage exceeded the 600s default
# while kernel outputs were advancing. Preserve model/math and permit warmup.
export VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=3600
# All four arms share this measured transport workaround. With this allocation,
# Torch 2.13/cu129/NCCL 2.29.7 default large collectives were pathologically slow.
# Same 16-rank payload and exact correctness checks: disabling P2P and NVLS
# together restored both all-gather and all-reduce throughput. See the report
# in analysis/gpu_transport/. This does not change model or optimizer settings.
export NCCL_P2P_DISABLE=1
export NCCL_NVLS_ENABLE=0
# Actual world16 two-call GPU probe and same-input Adam state replay passed.
# Replicated FSDP2 and foreach=false are supplied identically by all four wrappers.
export NCCL_NET=Socket
# Distributed RPC must stay on the allocation network while external HTTP uses
# the selected proxy. Reapply after each bashrc source in nested launch shells.
if [[ -n "${OMNI_HEAD_IP:-}" && -n "${OMNI_NODE_IP:-}" && -n "${OMNI_EXPECTED_PEER_IP:-}" ]]; then
  export no_proxy="${no_proxy:-localhost,127.0.0.1},$OMNI_HEAD_IP,$OMNI_NODE_IP,$OMNI_EXPECTED_PEER_IP,${OMNI_EXPECTED_HOST:-localhost}"
  export NO_PROXY="$no_proxy"
fi
