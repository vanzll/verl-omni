# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validate the resolved new-source A3 configuration before any lease release."""
import sys
from pathlib import Path

def expected(micro):
    values = {
        "data.train_batch_size": 32, "data.seed": 42, "data.val_max_samples": 8,
        "data.val_batch_size": 8, "data.validation_shuffle": False,
        "algorithm.art_joint_latent": True, "algorithm.art_cached_conditional_velocity": True,
        "algorithm.art_joint_reward_routing": True, "algorithm.art_prompt_balanced_split": True,
        "algorithm.norm_adv_by_std_in_grpo": False, "algorithm.art_exclude_final_step": True,
        "algorithm.old_policy_decay_schedule": "copy", "algorithm.old_policy_update_interval": 1,
        "actor_rollout_ref.actor.use_kl_loss": False,
        "actor_rollout_ref.actor.diffusion_loss.art_mask_threshold": 1e-5,
        "actor_rollout_ref.actor.diffusion_loss.art_reference_noise_level": 0.8,
        "actor_rollout_ref.actor.diffusion_loss.art_guidance_gain": 1.0,
        "actor_rollout_ref.actor.diffusion_loss.art_corrective_mask_enabled": True,
        "actor_rollout_ref.actor.diffusion_loss.art_loss_space": "velocity_linear_sigma",
        "actor_rollout_ref.actor.diffusion_loss.art_mask_metric": "sde_kl",
        "actor_rollout_ref.actor.ppo_mini_batch_size": 16,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": micro,
        "actor_rollout_ref.actor.ppo_epochs": 1, "actor_rollout_ref.actor.shuffle": False,
        "actor_rollout_ref.actor.fsdp_config.fsdp_size": 1,
        "actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size": 1,
        "actor_rollout_ref.actor.fsdp_config.param_offload": False,
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": False,
        "actor_rollout_ref.actor.optim.override_optimizer_config.foreach": False,
        "actor_rollout_ref.actor.optim.lr": 3e-5, "actor_rollout_ref.actor.optim.weight_decay": 1e-4,
        "actor_rollout_ref.model.lora_rank": 32, "actor_rollout_ref.model.lora_alpha": 64,
        "actor_rollout_ref.rollout.tensor_model_parallel_size": 2,
        "actor_rollout_ref.rollout.n": 8, "actor_rollout_ref.rollout.agent.num_workers": 16,
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 2,
        "actor_rollout_ref.rollout.pipeline.guidance_scale": None,
        "actor_rollout_ref.model.pipeline.guidance_scale": 1.0,
        "actor_rollout_ref.model.pipeline.video_cfg_scale": 1.0,
        "actor_rollout_ref.model.pipeline.audio_cfg_scale": 1.0,
        "trainer.nnodes": 4, "trainer.n_gpus_per_node": 8, "trainer.total_training_steps": None,
        "trainer.run_until_stopped": True,
        "actor_rollout_ref.actor.optim.lr_scheduler_type": "constant",
        "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio": 0,
        "trainer.save_freq": -1, "trainer.resume_mode": "disable", "trainer.test_freq": 10,
        "trainer.val_before_train": True, "trainer.expected_optimizer_updates_per_rollout": 2,
        "trainer.expected_generated_samples_per_rollout": 256,
        "trainer.project_name": "diffusionart-omni-ablation", "trainer.campaign.checkpoint_format": "disabled",
    }
    for branch in ("actor.diffusion_loss", "rollout.algo"):
        for key in ("art_joint_latent", "art_cached_conditional_velocity"):
            values[f"actor_rollout_ref.{branch}.{key}"] = True
    for branch, height, width, steps, video, audio in (
        ("model.pipeline", 256, 384, 20, 1.0, 1.0),
        ("rollout.pipeline", 256, 384, 20, 1.5, 3.0),
        ("rollout.val_kwargs.pipeline", 512, 768, 40, 3.0, 7.0),
    ):
        for key, value in dict(height=height, width=width, num_inference_steps=steps,
                               num_frames=121, frame_rate=24.0, video_cfg_scale=video, audio_cfg_scale=audio).items():
            values[f"actor_rollout_ref.{branch}.{key}"] = value
    for name in ("video_align", "hpsv3", "audiobox", "clap", "desync"):
        values[f"reward.models.{name}.dispatch_batch_size"] = 2
    values["reward.num_workers"] = 32
    values["reward.profile_dispatch"] = True
    values["reward.share_rpc_batches"] = True
    values["reward.shared_rpc_cache_batches"] = 16
    values["reward.rpc_excluded_tensor_keys"] = [
        "all_latents", "all_next_latents", "all_conditional_velocities", "all_log_probs",
        "prompt_embeds", "audio_prompt_embeds", "prompt_embeds_mask",
        "negative_prompt_embeds", "negative_audio_prompt_embeds", "negative_prompt_embeds_mask",
    ]
    for name, devices in {
        "video_align": [0, 1, 8, 9, 16, 17, 24, 25],
        "hpsv3": [2, 3, 10, 11, 18, 19, 26, 27],
        "audiobox": [4, 12, 20, 28], "clap": [5, 13, 21, 29],
        "desync": [6, 7, 14, 15, 22, 23, 30, 31],
    }.items():
        values[f"reward.models.{name}.placement.devices"] = devices
    for name, video, audio in (("video_align", 1.0, 0.0), ("hpsv3", 1.5, 0.0),
                               ("audiobox", 0.0, 0.5), ("clap", 0.0, 1.0), ("desync", 1.0, 1.0)):
        values[f"reward.reward_functions.{name}.routing_weights"] = {"video": video, "audio": audio}
    for branch, modality, rescale in (("rollout.pipeline", 1.0, 0.0), ("rollout.val_kwargs.pipeline", 3.0, 0.7)):
        for kind in ("video", "audio"):
            values[f"actor_rollout_ref.{branch}.{kind}_modality_scale"] = modality
            values[f"actor_rollout_ref.{branch}.{kind}_rescale_scale"] = rescale
    return values

def validate(config, micro):
    for key, wanted in expected(micro).items():
        actual = config
        for component in key.split("."):
            actual = actual[component]
        if actual != wanted:
            raise ValueError(f"{key}: actual={actual!r}, required={wanted!r}")
    if "A3JOINT32" not in config["trainer"]["experiment_name"]:
        raise ValueError("wrong experiment identity")

if __name__ == "__main__":
    from omegaconf import OmegaConf
    validate(OmegaConf.to_container(OmegaConf.load(Path(sys.argv[1])), resolve=True), int(sys.argv[2]))
    print("A3 resolved CPU configuration passed; actual GPU gate and training acceptance remain pending.")
