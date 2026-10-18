#!/usr/bin/env bash
set -eo pipefail
source /m2v_intern2/zhenglin/my_bashrc.sh
set -u
campaign=/m2v_intern/zhenglin/omni_art_ablation_20261016
source "$campaign/scripts/runtime_env.sh"
source "$campaign/scripts/use_source_snapshot.sh"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
: "${OUTPUT_DIR:?}"
[[ "$OUTPUT_DIR" == "$campaign"/* ]] || exit 2
if [[ "${OMNI_PROBE_KIND:-native}" == actor_cycle ]]; then
  # This no-checkpoint diagnostic must not write process core dumps.
  ulimit -c 0
  [[ "${OMNI_PROBE_NNODES:-1}" == 2 ]] || exit 2
  : "${OMNI_CYCLE_CPU_PROOF:?}" "${OMNI_ACTOR_PROBE_CAPTURES:?}"
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_HUB_OFFLINE=1 NCCL_DEBUG=INFO
  export NCCL_DEBUG_FILE="$OUTPUT_DIR/nccl.%h.%p.log"
  export PYTHONPATH="$OUTPUT_DIR/probe_code:$PYTHONPATH"
  # Preserve the lease-owned process group; rank entry handles launcher death.
  timeout --foreground --signal=TERM --kill-after=30s 1200s torchrun --nnodes=2 --nproc-per-node=8 \
    --node-rank="$OMNI_PROBE_NODE_RANK" --master-addr="$OMNI_HEAD_IP" --master-port="$OMNI_RAY_PORT" \
    --max-restarts=0 "$OUTPUT_DIR/probe_code/fsdp_cycle_rank_entry.py" --captures "$OMNI_ACTOR_PROBE_CAPTURES"
  exit $?
fi
if [[ "${OMNI_PROBE_KIND:-native}" == transport ]]; then
  [[ "${OMNI_PROBE_NNODES:-1}" == 2 ]] || exit 2
  : "${OMNI_TRANSPORT_CPU_PROOF:?}"
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NCCL_DEBUG=INFO
  export NCCL_DEBUG_FILE="$OUTPUT_DIR/nccl.%h.%p.log"
  timeout --signal=TERM --kill-after=20s 300s torchrun --nnodes=2 --nproc-per-node=8 \
    --node-rank="$OMNI_PROBE_NODE_RANK" --master-addr="$OMNI_HEAD_IP" --master-port="$OMNI_RAY_PORT" \
    --max-restarts=0 "$OUTPUT_DIR/probe_code/transport_rank_entry.py"
  exit $?
fi
if [[ "${OMNI_PROBE_KIND:-native}" == checkpoint_readback ]]; then
  [[ "${OMNI_PROBE_NNODES:-1}" == 2 ]] || exit 2
  : "${OMNI_CHECKPOINT_READBACK_CONFIG:?}" "${OMNI_CHECKPOINT_READBACK_PATH:?}" "${OMNI_ACTOR_PROBE_CAPTURES:?}"
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_HUB_OFFLINE=1
  export PYTHONPATH="$OUTPUT_DIR/probe_code:$PYTHONPATH"
  timeout --signal=TERM --kill-after=30s 5400s torchrun --nnodes=2 --nproc-per-node=8 \
    --node-rank="$OMNI_PROBE_NODE_RANK" --master-addr="$OMNI_HEAD_IP" --master-port="$OMNI_RAY_PORT" \
    --max-restarts=0 "$OUTPUT_DIR/probe_code/checkpoint_rank_entry.py" \
    --config "$OMNI_CHECKPOINT_READBACK_CONFIG" --checkpoint "$OMNI_CHECKPOINT_READBACK_PATH" \
    --captures "$OMNI_ACTOR_PROBE_CAPTURES"
  exit $?
fi
if [[ "${OMNI_PROBE_KIND:-native}" == actor || "${OMNI_PROBE_KIND:-native}" == nft_actor || "${OMNI_PROBE_KIND:-native}" == art_precision ]]; then
  [[ "${OMNI_PROBE_NNODES:-1}" == 1 ]] || exit 2
  : "${OMNI_ACTOR_PROBE_CAPTURES:?}"
  export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1
  probe_script=verify_actor_gpu.py
  if [[ "$OMNI_PROBE_KIND" == nft_actor ]]; then probe_script=verify_nft_gpu.py; fi
  if [[ "$OMNI_PROBE_KIND" == art_precision ]]; then probe_script=verify_art_precision_gpu.py; fi
  echo "[$(date -u +%FT%TZ)] isolated world1 actual FSDP2 $OMNI_PROBE_KIND diagnostic on $(hostname -s); allocated 8 GPUs"
  timeout --signal=TERM --kill-after=30s 2400s torchrun --standalone --nnodes=1 --nproc-per-node=1 --max-restarts=0 \
    "$OUTPUT_DIR/probe_code/$probe_script" --captures "$OMNI_ACTOR_PROBE_CAPTURES"
  exit $?
fi
if [[ "${OMNI_PROBE_KIND:-native}" == rewards ]]; then
  [[ "${OMNI_PROBE_NNODES:-1}" == 1 ]] || exit 2
  : "${OMNI_REWARD_PROBE_INPUT:?}"
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_HUB_OFFLINE=1
  echo "[$(date -u +%FT%TZ)] actual five native GPU reward scorers on $(hostname -s); allocated 8 GPUs"
  timeout --signal=TERM --kill-after=30s 1200s python "$OUTPUT_DIR/probe_code/verify_reward_gpu.py" --input "$OMNI_REWARD_PROBE_INPUT"
  exit $?
fi
if [[ "${OMNI_PROBE_KIND:-native}" == ltx_rollout ]]; then
  [[ "${OMNI_PROBE_NNODES:-1}" == 1 ]] || exit 2
  export CUDA_VISIBLE_DEVICES=0,1 OMNI_ISOLATED_VELOCITY_PROBE=1
  export PYTHONPATH="$OUTPUT_DIR/probe_code:$PYTHONPATH"
  echo "[$(date -u +%FT%TZ)] isolated TP2 full-LTX generation probe on $(hostname -s); allocated node has 8 GPUs"
  timeout --signal=TERM --kill-after=30s 4500s python "$OUTPUT_DIR/probe_code/verify_ltx_rollout_gpu.py"
  exit $?
fi
[[ "${OMNI_PROBE_KIND:-native}" == native ]] || exit 2
echo "[$(date -u +%FT%TZ)] starting bounded 8-GPU native CUDA12.9/NCCL probe on $(hostname -s)"
nodes=${OMNI_PROBE_NNODES:-1}
node_rank=${OMNI_PROBE_NODE_RANK:-0}
case "$nodes:$node_rank" in
  1:0) rendezvous=(--standalone --nnodes=1) ;;
  2:0|2:1) rendezvous=(--nnodes=2 --node-rank="$node_rank" --master-addr="$OMNI_HEAD_IP" --master-port="$OMNI_RAY_PORT") ;;
  *) echo 'invalid fixed runtime probe topology' >&2; exit 2 ;;
esac
timeout --signal=TERM --kill-after=20s 420s torchrun "${rendezvous[@]}" \
  --nproc_per_node=8 --max-restarts=0 "$OUTPUT_DIR/probe_code/verify_gpu_runtime.py"
