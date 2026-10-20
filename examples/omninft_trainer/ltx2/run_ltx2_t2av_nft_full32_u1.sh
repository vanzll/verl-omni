#!/usr/bin/env bash
set -eo pipefail
source /m2v_intern2/zhenglin/my_bashrc.sh
set -u

[[ $# == 0 ]] || { echo 'NFT Full32 U1 uses a fixed prepared configuration' >&2; exit 2; }
[[ "${OMNI_FULL_FINETUNE:?}" == 1 ]] || exit 2
case "${OMNI_VARIANT:?}" in
  NFT_FULL32_U1) updates=1; mini=32 ;;
  NFT_FULL32_U2) updates=2; mini=16 ;;
  *) exit 2 ;;
esac
case "${OMNI_UPDATE_MICROBATCH:-4}" in 1|2|4) ;; 8) [[ "$updates" == 1 ]] || exit 2 ;; *) exit 2 ;; esac
campaign=${OMNI_CAMPAIGN_ROOT:?}
control=${OMNI_CONTROL_DIR:?}
export PATH="$campaign/env/bin:$PATH"
source "$control/runtime_env.sh"
transport_overrides=$(python "$control/transport.py" hydra)
mapfile -t transport_args <<< "$transport_overrides"
export HF_HOME="$campaign/cache/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export TMPDIR="${RAY_TMPDIR:-${OMNI_RUNTIME_ROOT:?}/tmp}"
export XDG_CACHE_HOME="$campaign/cache/xdg"
export TORCH_EXTENSIONS_DIR="$campaign/cache/torch_extensions"
export TRITON_CACHE_DIR="$campaign/cache/triton"
export VLLM_CACHE_ROOT="$campaign/cache/vllm"
source "$control/use_source_snapshot.sh"
repo=${SOURCE_CHECKOUT_DIR:?}
output=${OUTPUT_DIR:?}
run_name=${RUN_NAME:?}
[[ "$run_name" == LTX23-OMNINFT-NOKL-FULL32-U"${updates}"-PROMPTSTD-P32K8-S42-* ]] || exit 2
[[ "${WANDB_RESUME:-never}" == never ]] || exit 2
: "${WANDB_RUN_ID:?fresh run ID required}"
export WANDB_RESUME=never WANDB_MODE=online WANDB_DIR="$output"
reward="$campaign/assets/rewards"
model="$campaign/cache/hf/hub/models--diffusers--LTX-2.3-Diffusers/snapshots/8eee8edcf067e838b843f926ec4d4cc9b2be1aaf"

training_args=(
  "hydra.searchpath=[file://$repo/verl_omni/trainer/config]"
  trainer.device=cuda
  data.train_files="$campaign/assets/vggsound/parquet/train.parquet"
  data.val_files="$campaign/assets/vggsound/parquet/test8.parquet"
  data.return_multi_modal_inputs=false data.train_batch_size=32 data.val_max_samples=8
  data.val_batch_size=8 data.validation_shuffle=false data.max_prompt_length=1024
  data.truncation=error data.seed=42
  algorithm.global_std=false algorithm.norm_adv_by_std_in_grpo=true
  algorithm.old_policy_decay_schedule=linear_to_0_5 algorithm.old_policy_decay=null
  algorithm.old_policy_update_interval=1 algorithm.timestep_fraction=1.0
  algorithm.train_timestep_count=null algorithm.train_timestep_range=null algorithm.adv_mode=continuous
  algorithm.art_pairwise_reward=false algorithm.art_prompt_balanced_split=false
  algorithm.art_joint_reward_routing=false algorithm.art_joint_latent=false
  algorithm.art_cached_conditional_velocity=false algorithm.art_exclude_final_step=false
  ++algorithm.art_prompt_group_std=false ++algorithm.art_first_update_mask_exempt=false
  actor_rollout_ref.model.path="$model" ++actor_rollout_ref.model.full_finetune=true
  actor_rollout_ref.model.lora_rank=0 actor_rollout_ref.model.lora_adapter_path=null
  actor_rollout_ref.model.pipeline.height=256 actor_rollout_ref.model.pipeline.width=384
  actor_rollout_ref.model.pipeline.num_frames=121 actor_rollout_ref.model.pipeline.frame_rate=24.0
  actor_rollout_ref.model.pipeline.num_inference_steps=20 actor_rollout_ref.model.pipeline.guidance_scale=null
  actor_rollout_ref.actor.strategy=fsdp2 actor_rollout_ref.actor.ppo_mini_batch_size="$mini"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${OMNI_UPDATE_MICROBATCH:-4}"
  actor_rollout_ref.actor.ppo_epochs=1 actor_rollout_ref.actor.shuffle=false
  actor_rollout_ref.actor.use_kl_loss=false actor_rollout_ref.actor.kl_loss_coef=0.0
  actor_rollout_ref.actor.diffusion_loss.mix_beta=1.0 actor_rollout_ref.actor.diffusion_loss.adv_clip_max=5.0
  actor_rollout_ref.actor.diffusion_loss.ref_kl_coef=0.0
  actor_rollout_ref.actor.diffusion_loss.video_ref_kl_coef=0.0
  actor_rollout_ref.actor.diffusion_loss.audio_ref_kl_coef=0.0
  ++actor_rollout_ref.actor.diffusion_loss.prompt_group_std=true
  actor_rollout_ref.actor.diffusion_loss.art_corrective_mask_enabled=false
  ++actor_rollout_ref.actor.diffusion_loss.art_first_update_mask_exempt=false
  ++actor_rollout_ref.actor.diffusion_loss.art_joint_latent=false
  ++actor_rollout_ref.actor.diffusion_loss.art_cached_conditional_velocity=false
  actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 actor_rollout_ref.actor.fsdp_config.fsdp_size=32
  actor_rollout_ref.actor.fsdp_config.reshard_after_forward=true
  actor_rollout_ref.actor.fsdp_config.param_offload=false actor_rollout_ref.actor.fsdp_config.optimizer_offload=false
  actor_rollout_ref.actor.use_no_sync_for_gradient_accumulation=false
  actor_rollout_ref.actor.optim.lr=3e-5 actor_rollout_ref.actor.optim.weight_decay=1e-4
  'actor_rollout_ref.actor.optim.betas=[0.9,0.999]' actor_rollout_ref.actor.optim.clip_grad=1.0
  actor_rollout_ref.actor.grad_clip=1.0 'actor_rollout_ref.actor.optim.override_optimizer_config={foreach:false}'
  actor_rollout_ref.actor.optim.lr_scheduler_type=constant actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0
  actor_rollout_ref.actor.checkpoint.save_lora_only=false
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 actor_rollout_ref.rollout.n=8
  actor_rollout_ref.rollout.agent.num_workers=16 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2
  actor_rollout_ref.rollout.pipeline.guidance_scale=null actor_rollout_ref.rollout.algo.noise_level=0.0
  ++actor_rollout_ref.rollout.algo.art_joint_latent=false
  ++actor_rollout_ref.rollout.algo.art_cached_conditional_velocity=false
  ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.vae_use_tiling=true
  ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.vae_patch_parallel_size=2
  actor_rollout_ref.rollout.val_kwargs.pipeline.height=512 actor_rollout_ref.rollout.val_kwargs.pipeline.width=768
  ++reward.reward_functions.video_align.fps=24.0 ++reward.reward_functions.desync.fps=24.0
  +reward.models.video_align.model_path="$reward/VideoReward/checkpoint-11352/model.pth"
  '+reward.models.video_align.placement.devices=[0,1,8,9,16,17,24,25]'
  +reward.models.video_align.executor.kwargs.base_model_path="$reward/Qwen2-VL-2B-Instruct"
  +reward.models.hpsv3.model_path="$reward/HPSv3/HPSv3.safetensors"
  '+reward.models.hpsv3.placement.devices=[2,3,10,11,18,19,26,27]'
  +reward.models.hpsv3.executor.kwargs.base_model_path="$reward/Qwen2-VL-7B-Instruct"
  +reward.models.audiobox.model_path="$reward/audiobox-aesthetics" '+reward.models.audiobox.placement.devices=[4,12,20,28]'
  +reward.models.clap.model_path="$reward/checkpoints/clap-htsat-unfused" '+reward.models.clap.placement.devices=[5,13,21,29]'
  +reward.models.desync.model_path="$reward/synchformer/synchformer_state_dict.pth"
  '+reward.models.desync.placement.devices=[6,7,14,15,22,23,30,31]'
  +reward.models.desync.executor.kwargs.source_root="$reward/OmniNFT-reference"
  ++reward.models.video_align.dispatch_batch_size=2 ++reward.models.hpsv3.dispatch_batch_size=2
  ++reward.models.audiobox.dispatch_batch_size=2 ++reward.models.clap.dispatch_batch_size=2
  ++reward.models.desync.dispatch_batch_size=2 reward.num_workers=32
  ++reward.profile_dispatch=true ++reward.share_rpc_batches=true ++reward.shared_rpc_cache_batches=16
  '++reward.rpc_excluded_tensor_keys=[all_latents,all_next_latents,all_conditional_velocities,all_log_probs,prompt_embeds,audio_prompt_embeds,prompt_embeds_mask,negative_prompt_embeds,negative_audio_prompt_embeds,negative_prompt_embeds_mask]'
  ++ray_kwargs.ray_init.runtime_env.env_vars.WANDB_DIR="$WANDB_DIR"
  ++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="$PYTHONPATH"
  ++ray_kwargs.ray_init.runtime_env.env_vars.OMNI_VARIANT="\"$OMNI_VARIANT\""
  '++ray_kwargs.ray_init.runtime_env.env_vars.OMNI_FULL_FINETUNE="1"'
  ++ray_kwargs.ray_init.address="${RAY_ADDRESS:?}"
  trainer.project_name=diffusionart-omni-ablation trainer.experiment_name="$run_name"
  'trainer.logger=[console,tensorboard,wandb]' trainer.default_local_dir="$output/checkpoints"
  trainer.rollout_data_dir="$output/rollout_media" trainer.rollout_data_save_freq=10 trainer.rollout_data_max_samples=8
  trainer.validation_data_dir="$output/validation" trainer.validation_data_max_samples=8
  trainer.val_before_train=true trainer.resume_mode=disable trainer.n_gpus_per_node=8 trainer.nnodes=4
  trainer.max_actor_ckpt_to_keep=2 trainer.save_freq=-1 trainer.test_freq=10 trainer.total_epochs=100
  trainer.total_training_steps=null trainer.run_until_stopped=true
  ++trainer.gpu_reservation_start_unix_mean="${OMNI_GPU_RESERVATION_START_UNIX_MEAN:-0}"
  ++trainer.launch_files_manifest_sha256="${OMNI_LAUNCH_MANIFEST_SHA256:-preflight}"
  ++trainer.campaign.base_commit=93c8a747252d27e4549fba4e4b23abc19d6114bb
  ++trainer.campaign.nft_config_anchor_run_id=f07cd5db ++trainer.campaign.infra_anchor_run_id=266a1574205eda99
  ++trainer.campaign.source_snapshot_sha256="${SOURCE_SNAPSHOT_SHA256:?}"
  ++trainer.campaign.source_files_manifest_sha256="${SOURCE_FILES_MANIFEST_SHA256:?}"
  ++trainer.campaign.source_checkout_dir="$repo"
  ++trainer.campaign.model_revision=8eee8edcf067e838b843f926ec4d4cc9b2be1aaf
  ++trainer.campaign.checkpoint_format=disabled
  ++trainer.campaign.train_parquet_sha256=fb92dc5f97e920c00067ad2129655682166322c7e9897c53c0d75ccdf7c9a7b5
  ++trainer.campaign.val_parquet_sha256=3c40c064352eb8e2f0c85431eaf47bbe41e0f9c51450b6462afc186dd88d03b7
  ++trainer.campaign.validation_seed_rule=42_plus_row_index
  ++trainer.campaign.reward_std_correction=1 ++trainer.campaign.reward_std_epsilon=0.0001
  ++trainer.expected_optimizer_updates_per_rollout="$updates" ++trainer.expected_generated_samples_per_rollout=256
)

cd "$repo"
if [[ "${OMNI_CONFIG_ONLY:-0}" == 1 ]]; then
  python - "$repo" "${OMNI_CONFIG_OUTPUT:?}" "${training_args[@]}" "${transport_args[@]}" <<'PY'
from pathlib import Path
import sys
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

with initialize_config_dir(config_dir=str(Path(sys.argv[1]) / "examples/omninft_trainer/ltx2"), version_base=None):
    config = compose(config_name="ltx2_omninft", overrides=sys.argv[3:])
    OmegaConf.resolve(config)
    OmegaConf.save(config, sys.argv[2], resolve=True)
PY
else
  exec python "$control/bounded_console.py" \
    --log "$output/logs/trainer.console.log" --max-bytes 67108864 --backup-count 3 \
    --io-timeout 10 --drain-timeout 1 --output-root "$output" -- python3 -m verl_omni.trainer.main_diffusion \
    --config-path "$repo/examples/omninft_trainer/ltx2" --config-name ltx2_omninft \
    "${training_args[@]}" "${transport_args[@]}"
fi
