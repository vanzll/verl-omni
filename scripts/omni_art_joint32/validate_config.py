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
import argparse
import json
import sys
from pathlib import Path

def expected(micro, full=False, variant="art"):
    if variant not in ("art", "nft_full32_u1", "nft_full32_u2", "art_full32_routed"):
        raise ValueError("unsupported configuration variant")
    nft = variant in ("nft_full32_u1", "nft_full32_u2")
    full = full or nft or variant == "art_full32_routed"
    if micro not in ((1, 2, 4, 8) if variant == "nft_full32_u1" else (1, 2, 4)):
        raise ValueError("unsupported update microbatch")
    values = {
        "data.train_batch_size": 32, "data.seed": 42, "data.val_max_samples": 8,
        "data.val_batch_size": 8, "data.validation_shuffle": False,
        "algorithm.art_joint_latent": True, "algorithm.art_cached_conditional_velocity": True,
        "algorithm.art_joint_reward_routing": True, "algorithm.art_prompt_balanced_split": True,
        "algorithm.norm_adv_by_std_in_grpo": False, "algorithm.art_exclude_final_step": True,
        "algorithm.art_prompt_group_std": True, "algorithm.art_first_update_mask_exempt": True,
        "actor_rollout_ref.actor.diffusion_loss.art_first_update_mask_exempt": True,
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
    if full:
        values.update({"actor_rollout_ref.model.full_finetune": True,
                       "actor_rollout_ref.model.lora_rank": 0,
                       "actor_rollout_ref.actor.strategy": "fsdp2",
                       "actor_rollout_ref.actor.fsdp_config.strategy": "fsdp2",
                       "actor_rollout_ref.actor.fsdp_config.fsdp_size": 32,
                       "actor_rollout_ref.actor.fsdp_config.reshard_after_forward": True,
                       "actor_rollout_ref.actor.use_no_sync_for_gradient_accumulation": False,
                       "ray_kwargs.ray_init.runtime_env.env_vars.OMNI_FULL_FINETUNE": "1",
                       "actor_rollout_ref.actor.checkpoint.save_lora_only": False})
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
    if nft:
        values = {key: value for key, value in values.items()
                  if not any(component.startswith("art_") for component in key.split("."))}
        values.update({
            "algorithm.trainer_type": "direct_preference", "algorithm.sample_source": "online",
            "algorithm.adv_estimator": "omni_nft", "algorithm.paired_preference": False,
            "algorithm.global_std": False, "algorithm.norm_adv_by_std_in_grpo": True,
            "algorithm.old_policy_decay_schedule": "linear_to_0_5", "algorithm.old_policy_decay": None,
            "algorithm.timestep_fraction": 1.0, "algorithm.train_timestep_count": None,
            "algorithm.train_timestep_range": None, "algorithm.adv_mode": "continuous",
            "actor_rollout_ref.model.algorithm": "omni_nft",
            "actor_rollout_ref.model.model_type": "omni_nft_model",
            "actor_rollout_ref.model.lora_adapter_path": None,
            "actor_rollout_ref.model.policy_state_adapters": ["default", "old"],
            "actor_rollout_ref.actor.ppo_mini_batch_size": 32,
            "actor_rollout_ref.actor.data_loader_seed": 42,
            "actor_rollout_ref.actor.use_distill_loss": False,
            "actor_rollout_ref.actor.kl_loss_coef": 0.0,
            "actor_rollout_ref.actor.diffusion_loss._target_": "verl_omni.pipelines.ltx2_omni_nft.config.OmniNFTLossConfig",
            "actor_rollout_ref.actor.diffusion_loss.loss_mode": "omni_nft",
            "actor_rollout_ref.actor.diffusion_loss.prompt_group_std": True,
            "actor_rollout_ref.actor.diffusion_loss.mix_beta": 1.0,
            "actor_rollout_ref.actor.diffusion_loss.adv_clip_max": 5.0,
            "actor_rollout_ref.actor.diffusion_loss.video_weight": 1.0,
            "actor_rollout_ref.actor.diffusion_loss.audio_weight": 1.0,
            "actor_rollout_ref.actor.diffusion_loss.ref_kl_coef": 0.0,
            "actor_rollout_ref.actor.diffusion_loss.video_ref_kl_coef": 0.0,
            "actor_rollout_ref.actor.diffusion_loss.audio_ref_kl_coef": 0.0,
            "actor_rollout_ref.actor.optim.optimizer": "AdamW",
            "actor_rollout_ref.actor.optim.betas": [0.9, 0.999],
            "actor_rollout_ref.actor.optim.clip_grad": 1.0,
            "actor_rollout_ref.actor.grad_clip": 1.0,
            "actor_rollout_ref.actor.fsdp_config.model_dtype": "bfloat16",
            "actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap": ["LTX2VideoTransformerBlock"],
            "actor_rollout_ref.actor.fsdp_config.offload_policy": False,
            "actor_rollout_ref.model.pipeline.guidance_scale": None,
            "actor_rollout_ref.rollout.rollout_adapter": "old",
            "actor_rollout_ref.rollout.algo.noise_level": 0.0,
            "actor_rollout_ref.rollout.val_kwargs.algo.noise_level": 0.0,
            "actor_rollout_ref.rollout.seed": 42,
            "actor_rollout_ref.rollout.load_format": "safetensors",
            "actor_rollout_ref.rollout.layered_summon": True,
            "actor_rollout_ref.rollout.calculate_log_probs": False,
            "reward.aggregation": "preserve_components",
            "trainer.expected_optimizer_updates_per_rollout": 1,
            "trainer.campaign.nft_config_anchor_run_id": "f07cd5db",
            "trainer.campaign.infra_anchor_run_id": "266a1574205eda99",
            "trainer.campaign.reward_std_correction": 1,
            "trainer.campaign.reward_std_epsilon": 1e-4,
            "trainer.campaign.model_revision": "8eee8edcf067e838b843f926ec4d4cc9b2be1aaf",
            "trainer.campaign.train_parquet_sha256": "fb92dc5f97e920c00067ad2129655682166322c7e9897c53c0d75ccdf7c9a7b5",
            "trainer.campaign.val_parquet_sha256": "3c40c064352eb8e2f0c85431eaf47bbe41e0f9c51450b6462afc186dd88d03b7",
            "data.shuffle": True,
        })
        for key in ("art_pairwise_reward", "art_joint_latent", "art_cached_conditional_velocity",
                    "art_joint_reward_routing", "art_prompt_balanced_split", "art_exclude_final_step",
                    "art_prompt_group_std", "art_first_update_mask_exempt", "art_log_rollout_velocity_gap"):
            values[f"algorithm.{key}"] = False
        for branch in ("actor.diffusion_loss", "rollout.algo"):
            for key in ("art_joint_latent", "art_cached_conditional_velocity"):
                values[f"actor_rollout_ref.{branch}.{key}"] = False
        values["actor_rollout_ref.actor.diffusion_loss.art_first_update_mask_exempt"] = False
        values["actor_rollout_ref.actor.diffusion_loss.art_corrective_mask_enabled"] = False
        for branch in ("model.pipeline", "rollout.pipeline", "rollout.val_kwargs.pipeline"):
            values[f"actor_rollout_ref.{branch}._target_"] = "verl_omni.pipelines.ltx2_omni_nft.config.LTXDiffusionPipelineConfig"
            values[f"actor_rollout_ref.{branch}.video_flow_shift"] = 12.0
    if variant == "nft_full32_u2":
        values["actor_rollout_ref.actor.ppo_mini_batch_size"] = 16
        values["trainer.expected_optimizer_updates_per_rollout"] = 2
    if variant == "art_full32_routed":
        values["algorithm.art_joint_latent"] = False
        values["algorithm.art_joint_reward_routing"] = False
        values["algorithm.art_exclude_final_step"] = False
        for branch in ("actor.diffusion_loss", "rollout.algo"):
            values[f"actor_rollout_ref.{branch}.art_joint_latent"] = False
    return values

def leaves(value, prefix=""):
    if isinstance(value, dict) and value:
        return {key: child for name, item in value.items()
                for key, child in leaves(item, f"{prefix}.{name}" if prefix else name).items()}
    return {prefix: value}

def compare_nft_anchors(config, baseline, infra, micro, variant="nft_full32_u1"):
    required = expected(micro, variant=variant)
    science = {
        "actor_rollout_ref.model.full_finetune", "actor_rollout_ref.model.lora_rank",
        "actor_rollout_ref.actor.checkpoint.save_lora_only", "actor_rollout_ref.actor.fsdp_config.fsdp_size",
        "actor_rollout_ref.actor.fsdp_config.strategy", "actor_rollout_ref.actor.ppo_mini_batch_size",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu", "actor_rollout_ref.actor.kl_loss_coef",
        "actor_rollout_ref.actor.diffusion_loss.prompt_group_std",
        "actor_rollout_ref.actor.diffusion_loss.video_ref_kl_coef",
        "actor_rollout_ref.actor.diffusion_loss.audio_ref_kl_coef", "algorithm.global_std",
        "trainer.expected_optimizer_updates_per_rollout",
    }
    execution = {
        "actor_rollout_ref.rollout.agent.num_workers", "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu",
        "reward.num_workers", "reward.profile_dispatch", "reward.share_rpc_batches",
        "reward.shared_rpc_cache_batches", "reward.rpc_excluded_tensor_keys", "trainer.nnodes",
    }
    execution.update(f"reward.models.{name}.placement.devices"
                     for name in ("video_align", "hpsv3", "audiobox", "clap", "desync"))
    compatibility = {key for key in required if "art_" in key and required[key] is False}
    compatibility.update({"data.apply_chat_template_kwargs", "actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs",
                          "reward.reward_model.rollout.engine_kwargs"})
    bookkeeping = {
        "actor_rollout_ref.actor.optim.total_training_steps": -1,
        "actor_rollout_ref.model.tokenizer_path": None,
        "reward.reward_model.nnodes": 0,
        "trainer.total_training_steps": None,
        "trainer.run_until_stopped": True,
    }
    ode = {"actor_rollout_ref.rollout.algo.noise_level", "actor_rollout_ref.model.algo.noise_level"}
    trainer_keys = {"trainer." + key for key in (
        "device", "nnodes", "n_gpus_per_node", "total_training_steps", "total_epochs", "run_until_stopped",
        "save_freq", "test_freq", "val_before_train", "val_only", "resume_mode", "resume_from_path",
        "rollout_data_save_freq", "rollout_data_max_samples", "validation_data_max_samples", "video_fps",
        "expected_optimizer_updates_per_rollout", "expected_generated_samples_per_rollout",
    )}
    actual, anchor, transport_anchor = leaves(config), leaves(baseline), leaves(infra)
    missing = "<absent>"
    changes = {}
    unexpected = {}
    for key in sorted(set(actual) | set(anchor)):
        if not key.startswith(("algorithm.", "data.", "actor_rollout_ref.model.", "actor_rollout_ref.actor.",
                               "actor_rollout_ref.rollout.", "reward.")) and key not in trainer_keys:
            continue
        before, after = anchor.get(key, missing), actual.get(key, missing)
        if before == after:
            continue
        category = None
        wanted = missing
        if key in science:
            category, wanted = "four_scientific_changes", required[key]
        elif key in execution:
            category, wanted = "art266_execution", transport_anchor.get(key, missing)
        elif key in compatibility:
            category, wanted = "disabled_art_or_empty_schema_default", required.get(key, {})
        elif key in bookkeeping:
            category, wanted = "runtime_bookkeeping", bookkeeping[key]
        elif key in ode:
            category, wanted = "explicit_ode_zero_noise_requirement", 0.0
        difference = {"f07cd5db": before, "prepared": after}
        if category is None or after != wanted or after == missing:
            unexpected[key] = difference
        else:
            changes[key] = dict(difference, category=category)
    report = {"status": "passed" if not unexpected else "failed", "changes": changes, "unexpected": unexpected}
    if unexpected:
        raise ValueError("unexpected NFT scientific/config differences: " + json.dumps(unexpected, sort_keys=True))
    return report

def validate(config, micro, variant=None):
    full = config["actor_rollout_ref"]["model"].get("full_finetune", False)
    if variant is None:
        variant = "nft_full32_u1" if config["actor_rollout_ref"]["model"].get("algorithm") == "omni_nft" else "art"
    for key, wanted in expected(micro, full=full, variant=variant).items():
        actual = config
        for component in key.split("."):
            actual = actual[component]
        if actual != wanted:
            raise ValueError(f"{key}: actual={actual!r}, required={wanted!r}")
    nft = variant in ("nft_full32_u1", "nft_full32_u2")
    updates = 2 if variant == "nft_full32_u2" else 1
    prefix = f"LTX23-OMNINFT-NOKL-FULL32-U{updates}-PROMPTSTD-P32K8-S42-" if nft else (
        "LTX23-ART-A12-FULL32-P32K8-U2-S42-" if full else "LTX23-A12JOINT32-")
    if variant == "art_full32_routed":
        prefix = "LTX23-ART-ROUTED20-FULL32-P32K8-U2-S42-"
    name = config["trainer"]["experiment_name"]
    if ((full or nft) and not name.startswith(prefix)) or (not full and not nft and "A12JOINT32" not in name):
        raise ValueError("wrong experiment identity")

if __name__ == "__main__":
    from omegaconf import OmegaConf
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("micro", type=int)
    parser.add_argument("variant", nargs="?", choices=("art", "nft_full32_u1", "nft_full32_u2", "art_full32_routed"))
    parser.add_argument("--nft-baseline", type=Path)
    parser.add_argument("--infra-baseline", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    validate(config, args.micro, args.variant)
    if args.variant in ("nft_full32_u1", "nft_full32_u2"):
        if not args.nft_baseline or not args.infra_baseline or not args.report:
            parser.error("NFT prepare validation requires both frozen anchor configs and a report path")
        report = compare_nft_anchors(config, json.loads(args.nft_baseline.read_text()),
                                     json.loads(args.infra_baseline.read_text()), args.micro, variant=args.variant)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print("Resolved CPU configuration passed; actual GPU gate and training acceptance remain pending.")
