#!/usr/bin/env bash
set -euo pipefail

source /m2v_intern2/zhenglin/my_bashrc.sh
campaign=${OMNI_CAMPAIGN_ROOT:?}
control=${OMNI_CONTROL_DIR:?}
[[ $# == 0 ]] || { echo "A3 protocol is fixed; use OMNI_UPDATE_MICROBATCH=1,2,4 at prepare" >&2; exit 2; }
case "${OMNI_UPDATE_MICROBATCH:-4}" in 1|2|4) ;; *) exit 2 ;; esac
export PATH="$campaign/env/bin:$PATH"
source "$control/runtime_env.sh"
export HF_HOME="$campaign/cache/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export TMPDIR="${RAY_TMPDIR:-${OMNI_RUNTIME_ROOT:?}/tmp}"
export XDG_CACHE_HOME="$campaign/cache/xdg"
export TORCH_EXTENSIONS_DIR="$campaign/cache/torch_extensions"
export TRITON_CACHE_DIR="$campaign/cache/triton"
export VLLM_CACHE_ROOT="$campaign/cache/vllm"
source "$control/use_source_snapshot.sh"
repo="$SOURCE_CHECKOUT_DIR"
output=${OUTPUT_DIR:?OUTPUT_DIR is required}
run_name=${RUN_NAME:?RUN_NAME is required}
source_sha=${SOURCE_SNAPSHOT_SHA256:?SOURCE_SNAPSHOT_SHA256 is required}
ray_address=${RAY_ADDRESS:?}
[[ "$run_name" == *A12JOINT32* ]] || exit 2
[[ "${WANDB_RESUME:-never}" == never ]] || exit 2
export WANDB_RESUME=never
: "${WANDB_RUN_ID:?fresh run ID required}"

export OUTPUT_DIR="$output"
export WANDB_MODE=online
export MODEL_PATH="$campaign/cache/hf/hub/models--diffusers--LTX-2.3-Diffusers/snapshots/8eee8edcf067e838b843f926ec4d4cc9b2be1aaf"
export REWARD_ROOT="$campaign/assets/rewards"
export DESYNC_SOURCE_ROOT="$REWARD_ROOT/OmniNFT-reference"
export TRAIN_FILE="$campaign/assets/vggsound/parquet/train.parquet"
export VAL_FILE="$campaign/assets/vggsound/parquet/test8.parquet"
export NUM_GPUS=8
export ROLLOUT_TP=2
export ROLLOUT_N=8
export TOTAL_TRAINING_STEPS=null

cd "$repo"
if [[ "${OMNI_CONFIG_ONLY:-0}" == 1 ]]; then
  : "${OMNI_CONFIG_OUTPUT:?}"
  python3() { command python3 "$OMNI_CONTROL_DIR/resolve_config.py" "$@"; }
  export -f python3
fi

exec bash "$repo/examples/diffusionart_trainer/ltx2/run_ltx2_t2av_art_lora.sh" \
  data.val_max_samples=8 \
  data.val_batch_size=8 \
  data.validation_shuffle=false \
  actor_rollout_ref.model.pipeline.height=256 \
  actor_rollout_ref.model.pipeline.width=384 \
  actor_rollout_ref.model.pipeline.num_frames=121 \
  actor_rollout_ref.model.pipeline.frame_rate=24.0 \
  actor_rollout_ref.model.pipeline.num_inference_steps=20 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${OMNI_UPDATE_MICROBATCH:-4}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.shuffle=false \
  actor_rollout_ref.actor.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=1 \
  actor_rollout_ref.actor.optim.override_optimizer_config='{foreach: false}' \
  actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0 \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
  actor_rollout_ref.actor.checkpoint.save_lora_only=true \
  actor_rollout_ref.rollout.agent.num_workers=16 \
  actor_rollout_ref.rollout.pipeline.guidance_scale=null \
  ++actor_rollout_ref.rollout.pipeline.video_cfg_scale=1.5 \
  ++actor_rollout_ref.rollout.pipeline.audio_cfg_scale=3.0 \
  ++actor_rollout_ref.rollout.val_kwargs.pipeline.video_cfg_scale=3.0 \
  ++actor_rollout_ref.rollout.val_kwargs.pipeline.audio_cfg_scale=7.0 \
  actor_rollout_ref.rollout.val_kwargs.pipeline.height=512 \
  actor_rollout_ref.rollout.val_kwargs.pipeline.width=768 \
  algorithm.art_prompt_balanced_split=true \
  algorithm.art_joint_reward_routing=true \
  algorithm.art_joint_latent=true \
  algorithm.art_cached_conditional_velocity=true \
  algorithm.norm_adv_by_std_in_grpo=false \
  ++algorithm.art_prompt_group_std=true \
  ++algorithm.art_first_update_mask_exempt=true \
  ++actor_rollout_ref.actor.diffusion_loss.art_first_update_mask_exempt=true \
  ++actor_rollout_ref.actor.diffusion_loss.art_joint_latent=true \
  ++actor_rollout_ref.actor.diffusion_loss.art_cached_conditional_velocity=true \
  ++actor_rollout_ref.rollout.algo.art_cached_conditional_velocity=true \
  ++actor_rollout_ref.rollout.algo.art_joint_latent=true \
  actor_rollout_ref.model.pipeline.guidance_scale=1.0 \
  ++actor_rollout_ref.model.pipeline.video_cfg_scale=1.0 \
  ++actor_rollout_ref.model.pipeline.audio_cfg_scale=1.0 \
  trainer.project_name=diffusionart-omni-ablation \
  trainer.experiment_name="$run_name" \
  trainer.nnodes=4 \
  trainer.max_actor_ckpt_to_keep=2 \
  trainer.rollout_data_max_samples=8 \
  trainer.save_freq=-1 \
  trainer.total_training_steps=null \
  trainer.run_until_stopped=true \
  ++trainer.gpu_reservation_start_unix_mean="${OMNI_GPU_RESERVATION_START_UNIX_MEAN:-0}" \
  ++trainer.launch_files_manifest_sha256="${OMNI_LAUNCH_MANIFEST_SHA256:-preflight}" \
  ++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="$PYTHONPATH" \
  ++ray_kwargs.ray_init.address="$ray_address" \
  ++trainer.campaign.base_commit=93c8a747252d27e4549fba4e4b23abc19d6114bb \
  ++trainer.campaign.source_snapshot_sha256="$source_sha" \
  ++trainer.campaign.source_files_manifest_sha256="$SOURCE_FILES_MANIFEST_SHA256" \
  ++trainer.campaign.source_checkout_dir="$SOURCE_CHECKOUT_DIR" \
  ++trainer.campaign.model_revision=8eee8edcf067e838b843f926ec4d4cc9b2be1aaf \
  ++trainer.campaign.checkpoint_format=disabled \
  ++trainer.campaign.train_parquet_sha256=fb92dc5f97e920c00067ad2129655682166322c7e9897c53c0d75ccdf7c9a7b5 \
  ++trainer.campaign.val_parquet_sha256=3c40c064352eb8e2f0c85431eaf47bbe41e0f9c51450b6462afc186dd88d03b7 \
  ++trainer.campaign.validation_seed_rule=42_plus_row_index \
  ++trainer.expected_optimizer_updates_per_rollout=2 \
  ++trainer.expected_generated_samples_per_rollout=256 \
  ++reward.models.video_align.dispatch_batch_size=2 \
  ++reward.models.hpsv3.dispatch_batch_size=2 \
  ++reward.models.audiobox.dispatch_batch_size=2 \
  ++reward.models.clap.dispatch_batch_size=2 \
  ++reward.models.desync.dispatch_batch_size=2 \
  reward.num_workers=32 \
  reward.models.video_align.placement.devices='[0,1,8,9,16,17,24,25]' \
  reward.models.hpsv3.placement.devices='[2,3,10,11,18,19,26,27]' \
  reward.models.audiobox.placement.devices='[4,12,20,28]' \
  reward.models.clap.placement.devices='[5,13,21,29]' \
  reward.models.desync.placement.devices='[6,7,14,15,22,23,30,31]' \
  ++reward.rpc_excluded_tensor_keys='[all_latents,all_next_latents,all_conditional_velocities,all_log_probs,prompt_embeds,audio_prompt_embeds,prompt_embeds_mask,negative_prompt_embeds,negative_audio_prompt_embeds,negative_prompt_embeds_mask]' \
  ++reward.profile_dispatch=true \
  ++reward.share_rpc_batches=true \
  ++reward.shared_rpc_cache_batches=16 \
  trainer.resume_mode=disable
