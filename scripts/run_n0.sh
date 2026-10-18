#!/usr/bin/env bash
set -euo pipefail

# Public PR #640 OmniNFT recipe, with the PR #744 author's mini=16 clarification.
# Paths, pair identity, Ray address and GPU lease are supplied by the attempt owner.
source /m2v_intern2/zhenglin/my_bashrc.sh
campaign=/m2v_intern/zhenglin/omni_art_ablation_20261016
checkpoint_policy_sha=$(sha256sum "$campaign/assets/compact_checkpoint_storage_policy.json" | cut -d' ' -f1)
[[ "$checkpoint_policy_sha" == "${OMNI_CHECKPOINT_STORAGE_POLICY_SHA256:-$checkpoint_policy_sha}" ]] || exit 2
export PATH="$campaign/env/bin:$PATH"
source "$campaign/scripts/runtime_env.sh"
export HF_HOME="$campaign/cache/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export TMPDIR="$campaign/tmp"
export XDG_CACHE_HOME="$campaign/cache/xdg"
export TORCH_EXTENSIONS_DIR="$campaign/cache/torch_extensions"
export TRITON_CACHE_DIR="$campaign/cache/triton"
export VLLM_CACHE_ROOT="$campaign/cache/vllm"
source "$campaign/scripts/use_source_snapshot.sh"
repo="$SOURCE_CHECKOUT_DIR"
reward="$campaign/assets/rewards"
model="$campaign/cache/hf/hub/models--diffusers--LTX-2.3-Diffusers/snapshots/8eee8edcf067e838b843f926ec4d4cc9b2be1aaf"
output=${OUTPUT_DIR:?OUTPUT_DIR is required}
run_name=${RUN_NAME:?RUN_NAME is required}
source_sha=${SOURCE_SNAPSHOT_SHA256:?SOURCE_SNAPSHOT_SHA256 is required}
ray_address=${RAY_ADDRESS:?RAY_ADDRESS is required for pair-isolated Ray}
export WANDB_MODE=online
export WANDB_DIR="$output/wandb"
mkdir -p "$WANDB_DIR" "$output/checkpoints" "$output/validation" "$output/train_samples"

cd "$repo"

exec python3 -m verl_omni.trainer.main_diffusion \
  --config-path "$repo/examples/omninft_trainer/ltx2" \
  --config-name ltx2_omninft \
  trainer.device=cuda \
  data.train_files="$campaign/assets/vggsound/parquet/train.parquet" \
  data.val_files="$campaign/assets/vggsound/parquet/test8.parquet" \
  data.return_multi_modal_inputs=false \
  data.train_batch_size=32 \
  data.val_max_samples=8 \
  data.val_batch_size=8 \
  data.validation_shuffle=false \
  data.max_prompt_length=1024 \
  data.truncation=error \
  data.seed=42 \
  actor_rollout_ref.model.path="$model" \
  actor_rollout_ref.model.pipeline.height=256 \
  actor_rollout_ref.model.pipeline.width=384 \
  actor_rollout_ref.model.pipeline.num_frames=121 \
  actor_rollout_ref.model.pipeline.frame_rate=24.0 \
  actor_rollout_ref.model.pipeline.num_inference_steps=20 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.shuffle=false \
  actor_rollout_ref.actor.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=1 \
  actor_rollout_ref.actor.optim.override_optimizer_config='{foreach: false}' \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
  actor_rollout_ref.actor.checkpoint.save_lora_only=true \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.agent.num_workers=8 \
  ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.vae_use_tiling=true \
  ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.vae_patch_parallel_size=2 \
  actor_rollout_ref.rollout.val_kwargs.pipeline.height=512 \
  actor_rollout_ref.rollout.val_kwargs.pipeline.width=768 \
  ++reward.reward_functions.video_align.fps=24.0 \
  ++reward.reward_functions.desync.fps=24.0 \
  +reward.models.video_align.model_path="$reward/VideoReward/checkpoint-11352/model.pth" \
  '+reward.models.video_align.placement.devices=[0,1]' \
  +reward.models.video_align.executor.kwargs.base_model_path="$reward/Qwen2-VL-2B-Instruct" \
  +reward.models.hpsv3.model_path="$reward/HPSv3/HPSv3.safetensors" \
  '+reward.models.hpsv3.placement.devices=[2,3]' \
  +reward.models.hpsv3.executor.kwargs.base_model_path="$reward/Qwen2-VL-7B-Instruct" \
  +reward.models.audiobox.model_path="$reward/audiobox-aesthetics" \
  '+reward.models.audiobox.placement.devices=[4]' \
  +reward.models.clap.model_path="$reward/checkpoints/clap-htsat-unfused" \
  '+reward.models.clap.placement.devices=[5]' \
  +reward.models.desync.model_path="$reward/synchformer/synchformer_state_dict.pth" \
  '+reward.models.desync.placement.devices=[6,7]' \
  +reward.models.desync.executor.kwargs.source_root="$reward/OmniNFT-reference" \
  ++ray_kwargs.ray_init.runtime_env.env_vars.WANDB_DIR="$WANDB_DIR" \
  ++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="$PYTHONPATH" \
  ++ray_kwargs.ray_init.address="$ray_address" \
  trainer.project_name=diffusionart-omni-ablation \
  trainer.experiment_name="$run_name" \
  'trainer.logger=[console,tensorboard,wandb]' \
  trainer.default_local_dir="$output/checkpoints" \
  trainer.rollout_data_dir="$output/rollout_media" \
  trainer.rollout_data_save_freq=10 \
  trainer.rollout_data_max_samples=8 \
  trainer.validation_data_dir="$output/validation" \
  trainer.validation_data_max_samples=8 \
  trainer.val_before_train=true \
  trainer.resume_mode=disable \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=2 \
  trainer.max_actor_ckpt_to_keep=2 \
  trainer.save_freq=-1 \
  trainer.test_freq=10 \
  trainer.total_epochs=100 \
  trainer.total_training_steps=75 \
  ++trainer.gpu_reservation_start_unix_mean="${OMNI_GPU_RESERVATION_START_UNIX_MEAN:-0}" \
  ++trainer.launch_files_manifest_sha256="${OMNI_LAUNCH_MANIFEST_SHA256:-preflight}" \
  ++trainer.campaign.base_commit=93c8a747252d27e4549fba4e4b23abc19d6114bb \
  ++trainer.campaign.source_snapshot_sha256="$source_sha" \
  ++trainer.campaign.source_files_manifest_sha256="$SOURCE_FILES_MANIFEST_SHA256" \
  ++trainer.campaign.source_checkout_dir="$SOURCE_CHECKOUT_DIR" \
  ++trainer.campaign.runtime_lock_sha256="$RUNTIME_LOCK_SHA256" \
  ++trainer.campaign.native_gl_manifest_sha256="$NATIVE_GL_MANIFEST_SHA256" \
  ++trainer.campaign.model_revision=8eee8edcf067e838b843f926ec4d4cc9b2be1aaf \
  ++trainer.campaign.checkpoint_format=disabled \
  ++trainer.campaign.checkpoint_storage_policy_sha256="$checkpoint_policy_sha" \
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
  "$@"
