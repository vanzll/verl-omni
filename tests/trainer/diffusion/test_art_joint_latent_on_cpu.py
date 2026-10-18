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
"""Joint ART's full-coordinate math, raw rewards, balanced dispatch and replay contract."""

from unittest.mock import patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from verl import DataProto
from verl.utils.config import omega_conf_to_dataclass

from verl_omni.trainer.config.algorithm import DiffusionAlgoConfig
from verl_omni.trainer.diffusion.diffusion_algos import DiffusionARTLoss
from verl_omni.trainer.diffusion.diffusion_art import art_guidance, art_loss, art_prompt_balanced_indices
from verl_omni.workers.config import DiffusionLossConfig, DiffusionRolloutAlgoConfig


def make_config(dp=32, **algorithm):
    """P32K8 with two optimizer windows and independent opt-in flags at every boundary."""
    flags = {"art_joint_latent": True, "art_cached_conditional_velocity": True}
    config = OmegaConf.create(
        {
            "algorithm": {
                **flags,
                "art_joint_reward_routing": True,
                "art_prompt_balanced_split": True,
                "norm_adv_by_std_in_grpo": False,
                "global_std": True,
                "timestep_fraction": 0.5,
            },
            "actor_rollout_ref": {
                "model": {
                    "lora_rank": 32,
                    "policy_state_adapters": ["default", "old"],
                    "pipeline": {"video_cfg_scale": 1.0, "audio_cfg_scale": 1.0},
                },
                "actor": {
                    "data_loader_seed": 42,
                    "shuffle": False,
                    "ppo_mini_batch_size": 16,
                    "ppo_micro_batch_size_per_gpu": 2,
                    "fsdp_config": {"ulysses_sequence_parallel_size": 1},
                    "diffusion_loss": {
                        **flags,
                        "art_video_weight": 1.0,
                        "art_audio_weight": 1.0,
                        "art_guidance_gain": 1.0,
                        "art_loss_space": "velocity_linear_sigma",
                        "art_mask_metric": "sde_kl",
                        "art_mask_threshold": 1e-5,
                        "art_reference_noise_level": 0.8,
                        "art_corrective_mask_enabled": True,
                    },
                },
                "rollout": {
                    "n": 8,
                    "rollout_adapter": "old",
                    "algo": {**flags, "noise_level": 0.0},
                    "pipeline": {"video_cfg_scale": 3.0, "audio_cfg_scale": 1.5},
                },
            },
            "reward": {
                "reward_functions": {
                    "video_align": {"routing_weights": {"video": 1.0}},
                    "hps": {"routing_weights": {"video": 1.5}},
                    "aesthetic": {"routing_weights": {"video": 0.5}},
                    "clap": {"routing_weights": {"audio": 1.0}},
                    "desync": {"routing_weights": {"video": 1.0, "audio": 1.0}},
                }
            },
            "data": {"train_batch_size": 32},
            "trainer": {"n_gpus_per_node": 8, "nnodes": dp // 8},
        }
    )
    config.algorithm.update(algorithm)
    return config


def make_batch(prompts=32, video=12, audio=1):
    """Distinct cached predictions cannot accidentally match CFG or a state difference."""
    generator = torch.Generator().manual_seed(29)
    batch, steps, width, channels = prompts * 8, 5, video + audio, 2
    uid = np.repeat(np.arange(prompts).astype(str), 8).astype(object)
    scores = torch.randn(batch, 5, generator=generator) * torch.tensor([1.0, 10.0, 0.1, 2.0, 0.5])
    times = torch.linspace(1000, 200, steps).expand(batch, -1).clone()
    return DataProto.from_dict(
        tensors={
            "all_latents": torch.randn(batch, steps, width, channels, generator=generator),
            "all_next_latents": torch.randn(batch, steps, width, channels, generator=generator),
            "all_timesteps": times,
            "all_segment_timesteps": times.unsqueeze(-1).expand(-1, -1, 2).clone(),
            "all_conditional_velocities": torch.randn(batch, steps, width, channels, generator=generator) + 100,
            "video_seq_len": torch.full((batch,), video, dtype=torch.long),
            "sample_id": torch.arange(batch),
        },
        non_tensors={
            "uid": uid,
            **{
                f"reward/{name}": scores[:, i].numpy()
                for i, name in enumerate(["video_align", "hps", "aesthetic", "clap", "desync"])
            },
        },
        meta_info={"global_steps": 2},
    )


@pytest.mark.parametrize("global_std", [False, True])
def test_raw_components_keep_scale_and_shared_reward_enters_once(global_std):
    batch = make_batch(prompts=2)
    config = make_config(global_std=global_std)
    routed, (centered, names) = DiffusionARTLoss._modality_rewards(batch, torch.zeros(16), config, ["video", "audio"])
    raw = torch.stack([torch.as_tensor(batch.non_tensor_batch[f"reward/{name}"]) for name in names], dim=1)
    expected_centered = (raw.reshape(2, 8, -1) - raw.reshape(2, 8, -1).mean(1, keepdim=True)).reshape(16, -1)
    torch.testing.assert_close(centered, expected_centered)
    weights = {"video_align": 1.0, "hps": 1.5, "aesthetic": 0.5, "clap": 1.0, "desync": 1.0}
    expected = expected_centered @ torch.tensor([weights[name] for name in names])
    torch.testing.assert_close(routed["video"], expected)
    torch.testing.assert_close(routed["audio"], expected)
    doubled = expected + expected_centered[:, names.index("desync")]
    assert not torch.allclose(expected, doubled)


@pytest.mark.parametrize("dp", [16, 32])
def test_full_k_guidance_balanced_coverage_cache_gather_and_raw_logs(dp):
    batch = make_batch()
    source = batch.batch.clone()
    raw = torch.stack(
        [
            torch.as_tensor(batch.non_tensor_batch[f"reward/{name}"])
            for name in ["video_align", "hps", "aesthetic", "clap", "desync"]
        ],
        dim=1,
    )
    reward = raw @ torch.tensor([1.0, 1.5, 0.5, 1.0, 1.0])
    group = torch.arange(32).repeat_interleave(8)
    expected, _ = art_guidance(source["all_next_latents"][:, -1], reward, group, 32)
    config = make_config(dp=dp)
    DiffusionARTLoss.validate_trainer_config(config)
    with patch("verl_omni.trainer.diffusion.diffusion_algos.art_guidance", wraps=art_guidance) as guidance_call:
        prepared = DiffusionARTLoss.prepare_actor_batch(batch, reward, config)
    assert guidance_call.call_count == 1
    ids = prepared.batch["sample_id"]
    torch.testing.assert_close(prepared.batch["art_guidance"], expected[group[ids]])
    assert sorted(ids.tolist()) == list(range(256))
    order = art_prompt_balanced_indices(np.repeat(np.arange(32), 8), 8, 44, dp_size=dp)
    np.testing.assert_array_equal(ids.numpy(), order)
    rank_rows = ids.reshape(dp, -1)
    first, second = rank_rows.chunk(2, dim=1)
    rank_batches = prepared.chunk(dp)
    for update, expected_ids in enumerate((first, second)):
        dispatched = torch.cat([rank_batch.chunk(2)[update].batch["sample_id"] for rank_batch in rank_batches])
        torch.testing.assert_close(dispatched, expected_ids.flatten())
    for window in (first, second):
        counts = torch.bincount((window.flatten() // 8), minlength=32)
        torch.testing.assert_close(counts, torch.full((32,), 4))
    assert not set(first.flatten().tolist()) & set(second.flatten().tolist())
    for row, sample_id in enumerate(ids.tolist()):
        times = prepared.batch["train_timesteps"][row]
        indices = (source["all_timesteps"][sample_id, None] == times[:, None]).nonzero()[:, 1]
        torch.testing.assert_close(prepared.batch["train_latents"][row], source["all_latents"][sample_id, indices])
        torch.testing.assert_close(
            prepared.batch["train_conditional_velocities"][row],
            source["all_conditional_velocities"][sample_id, indices],
        )
        assert source["all_timesteps"][sample_id, -1] not in times
    assert "all_conditional_velocities" not in prepared.batch.keys()
    assert prepared.batch["train_segment_timesteps"].shape[-1] == 1
    metrics = DiffusionARTLoss.collect_batch_metrics(prepared.batch)
    assert metrics["actor/reward/sum/mean"] == pytest.approx(raw.sum(1).mean().item())
    assert metrics["actor/reward/hps/std"] == pytest.approx(raw[:, 1].std(correction=0).item())
    assert "actor/newjoint/covariance_rms" in metrics
    torch.testing.assert_close(prepared.batch["sample_level_rewards"][:, 0], reward[ids])


@pytest.mark.parametrize("step", [1, 10, 2])
def test_joint_cross_covariance_probe_preserves_modal_segments_and_training_tensors(step):
    config = make_config(art_prompt_balanced_split=False)
    batch = make_batch(prompts=2)
    batch.meta_info["global_steps"] = step
    rewards = torch.linspace(-1, 1, 16)
    _, components = DiffusionARTLoss._modality_rewards(batch, rewards, config, ["video", "audio"])
    expected = DiffusionARTLoss._cross_covariance_probe(
        batch.batch["all_next_latents"][:, -1],
        [("video", slice(0, 12)), ("audio", slice(12, 13))],
        torch.arange(2).repeat_interleave(8),
        components,
        config.reward.reward_functions,
    )
    assert len(expected) == 10
    for modality in ("video", "audio"):
        assert expected[f"cov_probe/{modality}/same_rms"] > 0
        assert expected[f"cov_probe/{modality}/cross_rms"] > 0

    baseline = make_batch(prompts=2)
    baseline.meta_info["global_steps"] = step
    with patch.object(DiffusionARTLoss, "_cross_covariance_probe", return_value={}):
        without_probe = DiffusionARTLoss.prepare_actor_batch(baseline, rewards, config)
    with patch("verl_omni.trainer.diffusion.diffusion_algos.art_guidance", wraps=art_guidance) as guidance_call:
        prepared = DiffusionARTLoss.prepare_actor_batch(batch, rewards, config)
    assert guidance_call.call_count == 1
    metrics = DiffusionARTLoss.collect_batch_metrics(prepared.batch)
    observed = {key: value for key, value in metrics.items() if key.startswith("actor/cov_probe/")}
    if step in (1, 10):
        assert observed == pytest.approx({f"actor/{key}": value.item() for key, value in expected.items()})
    else:
        assert observed == {}
    # Suppressing diagnostics must not change guidance, cache selection, raw logs or any other tensor.
    for key in without_probe.batch.keys():
        torch.testing.assert_close(prepared.batch[key], without_probe.batch[key], rtol=0, atol=0)
    np.testing.assert_array_equal(prepared.non_tensor_batch["uid"], without_probe.non_tensor_batch["uid"])


@pytest.mark.parametrize("mask_metric", ["sde_kl", "fixed_mse"])
@pytest.mark.parametrize("loss_space", ["velocity_linear_sigma", "velocity_uniform"])
def test_joint_loss_mask_and_gradient_match_explicit_concat(mask_metric, loss_space):
    generator = torch.Generator().manual_seed(17)
    # Exact 92.4% / 7.6% coordinate proportions; modality weights must be irrelevant.
    video, audio = 231, 19
    old = torch.randn(4, video + audio, 2, generator=generator)
    endpoints = torch.randn(8, video + audio, 2, generator=generator)
    guidance, _ = art_guidance(endpoints, torch.arange(8).float(), torch.zeros(8, dtype=torch.long), 1)
    guidance = guidance.expand(4, -1, -1)
    drift = torch.tensor([-0.3, 0.3, -0.001, 0.001])[:, None, None] * guidance
    current = (old + drift).requires_grad_()
    sigma = torch.tensor([0.9, 0.7, 0.5, 0.3])
    next_sigma = sigma - 0.1
    config = make_config().actor_rollout_ref.actor
    config.diffusion_loss.update(
        {
            "art_mask_metric": mask_metric,
            "art_loss_space": loss_space,
            "art_video_weight": 100,
            "art_audio_weight": 0.01,
        }
    )
    kwargs = {
        key.removeprefix("art_"): value
        for key, value in config.diffusion_loss.items()
        if key
        in {
            "art_guidance_gain",
            "art_loss_space",
            "art_mask_metric",
            "art_mask_threshold",
            "art_reference_noise_level",
            "art_corrective_mask_enabled",
        }
    }
    concatenated = torch.cat((current[:, :video], current[:, video:]), dim=1)
    expected = art_loss(concatenated, old, guidance, sigma, next_sigma, **kwargs)
    with patch("verl_omni.trainer.diffusion.diffusion_algos.art_loss", wraps=art_loss) as loss_call:
        actual, metrics = DiffusionARTLoss.compute_loss(
            forward_prediction=current,
            old_prediction=old,
            art_guidance=guidance,
            t_expanded=sigma[:, None, None],
            sigma_next=next_sigma,
            video_seq_len=torch.full((4,), video),
            config=config,
            segment_sigma=sigma[:, None].expand(-1, 2),
            segment_sigma_next=next_sigma[:, None].expand(-1, 2),
        )
    assert loss_call.call_count == 1
    torch.testing.assert_close(actual, expected["per_item_loss"].mean())
    actual_grad = torch.autograd.grad(actual, current, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected["per_item_loss"].mean(), current)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    assert metrics["actor/newjoint/mask_frac"] == pytest.approx(expected["masked"].float().mean().item())
    assert not any(key.startswith("actor/video/") or key.startswith("actor/audio/") for key in metrics)


def test_constant_components_produce_zero_guidance_loss_and_gradient():
    batch = make_batch(prompts=2)
    for key in batch.non_tensor_batch:
        if key.startswith("reward/"):
            batch.non_tensor_batch[key][:] = 3.0
    config = make_config(art_prompt_balanced_split=False)
    prepared = DiffusionARTLoss.prepare_actor_batch(batch, torch.full((16,), 99.0), config)
    guidance = prepared.batch["art_guidance"]
    assert torch.count_nonzero(guidance) == 0
    current = torch.ones_like(guidance, requires_grad=True)
    loss, _ = DiffusionARTLoss.compute_loss(
        forward_prediction=current,
        old_prediction=torch.zeros_like(current),
        art_guidance=guidance,
        t_expanded=torch.full((16, 1, 1), 0.5),
        sigma_next=torch.full((16,), 0.4),
        video_seq_len=prepared.batch["video_seq_len"],
        config=config.actor_rollout_ref.actor,
    )
    loss.backward()
    assert loss.item() == 0 and torch.count_nonzero(current.grad) == 0


@pytest.mark.parametrize("bad_cache", ["missing", "shape", "nan", "inf", "integer"])
def test_invalid_conditional_cache_fails_closed(bad_cache):
    batch = make_batch(prompts=1)
    cache = batch.batch["all_conditional_velocities"]
    if bad_cache == "missing":
        del batch.batch["all_conditional_velocities"]
    elif bad_cache == "shape":
        batch.batch["all_conditional_velocities"] = cache[:, :-1]
    elif bad_cache == "integer":
        batch.batch["all_conditional_velocities"] = cache.long()
    else:
        cache[0, -1, 0, 0] = float(bad_cache)  # Even excluded timesteps must be finite.
    with pytest.raises(ValueError, match="all_conditional_velocities"):
        DiffusionARTLoss.prepare_actor_batch(batch, torch.zeros(8), make_config(art_prompt_balanced_split=False))


def test_async_segment_times_rejected_by_prepare_and_loss():
    batch = make_batch(prompts=1)
    batch.batch["all_segment_timesteps"][0, -1, 1] += 1
    with pytest.raises(ValueError, match="synchronous"):
        DiffusionARTLoss.prepare_actor_batch(batch, torch.zeros(8), make_config(art_prompt_balanced_split=False))
    for which in ("segment_sigma", "segment_sigma_next"):
        times = {"segment_sigma": torch.full((2, 2), 0.5), "segment_sigma_next": torch.full((2, 2), 0.4)}
        times[which][0, 1] += 0.1
        with pytest.raises(ValueError, match="synchronous"):
            DiffusionARTLoss.compute_loss(
                forward_prediction=torch.zeros(2, 13, 2),
                old_prediction=torch.zeros(2, 13, 2),
                art_guidance=torch.ones(2, 13, 2),
                t_expanded=torch.full((2, 1, 1), 0.5),
                sigma_next=torch.full((2,), 0.4),
                video_seq_len=torch.full((2,), 12),
                config=make_config().actor_rollout_ref.actor,
                **times,
            )


@pytest.mark.parametrize("key", ["art_joint_latent", "art_cached_conditional_velocity"])
@pytest.mark.parametrize(
    "section", ["algorithm", "actor_rollout_ref.actor.diffusion_loss", "actor_rollout_ref.rollout.algo"]
)
def test_three_layer_flags_must_agree(key, section):
    config = make_config()
    OmegaConf.update(config, f"{section}.{key}", False)
    with pytest.raises(ValueError, match=key):
        DiffusionARTLoss.validate_trainer_config(config)


@pytest.mark.parametrize(
    "key,wrong",
    [
        ("guidance_scale", 2),
        ("true_cfg_scale", 2),
        ("video_cfg_scale", 3),
        ("audio_cfg_scale", 1.5),
        ("video_stg_scale", 1),
        ("audio_stg_scale", 1),
        ("video_modality_scale", 2),
        ("audio_modality_scale", 2),
        ("video_rescale_scale", 0.2),
        ("audio_rescale_scale", 0.2),
    ],
)
def test_cached_model_guidance_must_be_neutral(key, wrong):
    config = make_config()
    OmegaConf.update(config, f"actor_rollout_ref.model.pipeline.{key}", wrong)
    with pytest.raises(ValueError, match=key):
        DiffusionARTLoss.validate_trainer_config(config)


def test_legacy_flags_and_cfg_constraints_are_preserved():
    assert not DiffusionAlgoConfig().art_joint_latent
    assert not DiffusionAlgoConfig().art_cached_conditional_velocity
    config = make_config()
    for section in (
        config.algorithm,
        config.actor_rollout_ref.actor.diffusion_loss,
        config.actor_rollout_ref.rollout.algo,
    ):
        section.art_joint_latent = False
        section.art_cached_conditional_velocity = False
    with pytest.raises(ValueError, match="separate ablations"):
        DiffusionARTLoss.validate_trainer_config(config)
    config.algorithm.art_prompt_balanced_split = False
    with pytest.raises(ValueError, match="must equal"):
        DiffusionARTLoss.validate_trainer_config(config)
    config.actor_rollout_ref.rollout.pipeline = config.actor_rollout_ref.model.pipeline.copy()
    DiffusionARTLoss.validate_trainer_config(config)


def test_joint_rejects_std_normalization():
    with pytest.raises(ValueError, match="norm_adv_by_std_in_grpo"):
        DiffusionARTLoss.validate_trainer_config(make_config(norm_adv_by_std_in_grpo=True))


@pytest.mark.parametrize("config_cls", [DiffusionAlgoConfig, DiffusionLossConfig, DiffusionRolloutAlgoConfig])
def test_flags_survive_dataclass_conversion_and_default_off(config_cls):
    defaults = config_cls()
    assert defaults.art_joint_latent is False
    assert defaults.art_cached_conditional_velocity is False
    config = OmegaConf.create(
        {
            "_target_": f"{config_cls.__module__}.{config_cls.__name__}",
            "art_joint_latent": True,
            "art_cached_conditional_velocity": True,
        }
    )
    if config_cls is DiffusionAlgoConfig:
        config.norm_adv_by_std_in_grpo = False
    converted = omega_conf_to_dataclass(config)
    assert isinstance(converted, config_cls)
    assert converted.art_joint_latent is True
    assert converted.art_cached_conditional_velocity is True
