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
"""Routed ART oracle: full-K standardization, modality mean and all 20 positions."""

import pytest
import torch

from verl_omni.trainer.diffusion.diffusion_algos import DiffusionARTLoss
from verl_omni.trainer.diffusion.diffusion_art import art_guidance, art_loss

from test_art_full32_on_cpu import TinyFullEngine
from test_art_joint_latent_on_cpu import make_batch, make_config


def routed_config():
    """Keep the Full baseline while selecting separate routes and the final position."""
    config = make_config(
        art_joint_latent=False,
        art_joint_reward_routing=False,
        art_prompt_group_std=True,
        art_first_update_mask_exempt=True,
        art_exclude_final_step=False,
        timestep_fraction=1.0,
    )
    config.actor_rollout_ref.actor.diffusion_loss.art_joint_latent = False
    config.actor_rollout_ref.actor.diffusion_loss.art_first_update_mask_exempt = True
    config.actor_rollout_ref.rollout.algo.art_joint_latent = False
    config.actor_rollout_ref.model.update(full_finetune=True, lora_rank=0)
    config.actor_rollout_ref.actor.strategy = "fsdp2"
    config.actor_rollout_ref.actor.fsdp_config.update(fsdp_size=32, reshard_after_forward=True)
    config.reward.reward_functions.aesthetic.routing_weights = {"audio": 0.5}
    return config


def full20_batch():
    """Distinct states and conditional predictions make final-position replay observable."""
    batch = make_batch()
    generator = torch.Generator().manual_seed(12)
    for name in ("all_latents", "all_next_latents", "all_conditional_velocities"):
        source = batch.batch[name]
        batch.batch[name] = torch.randn(256, 20, *source.shape[2:], generator=generator)
    times = torch.linspace(1000, 50, 20).expand(256, -1).clone()
    batch.batch["all_timesteps"] = times
    batch.batch["all_segment_timesteps"] = times.unsqueeze(-1).expand(-1, -1, 2).clone()
    batch.batch["art_policy_version"] = torch.zeros(256, dtype=torch.int64)
    return batch


def test_routed_full20_matches_component_covariance_oracle_and_replays_final_position():
    config = routed_config()
    DiffusionARTLoss.validate_trainer_config(config)
    batch = full20_batch()
    original = batch.batch.clone()
    names = ["video_align", "hps", "aesthetic", "clap", "desync"]
    raw = torch.stack([torch.as_tensor(batch.non_tensor_batch[f"reward/{name}"]) for name in names], dim=1)
    grouped = raw.reshape(32, 8, 5)
    standardized = ((grouped - grouped.mean(1, keepdim=True)) / (grouped.std(1, keepdim=True) + 1e-4)).reshape(256, 5)
    video_reward = standardized @ torch.tensor([1.0, 1.5, 0.0, 0.0, 1.0])
    audio_reward = standardized @ torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0])
    groups = torch.arange(32).repeat_interleave(8)
    endpoint = original["all_next_latents"][:, -1]
    video_guidance, _ = art_guidance(endpoint[:, :12], video_reward, groups, 32)
    audio_guidance, _ = art_guidance(endpoint[:, 12:], audio_reward, groups, 32)
    expected = torch.cat([video_guidance[groups], audio_guidance[groups]], dim=1)
    prepared = DiffusionARTLoss.prepare_actor_batch(batch, torch.zeros(256), config)
    ids = prepared.batch["sample_id"]
    torch.testing.assert_close(prepared.batch["art_guidance"], expected[ids])
    assert prepared.batch["train_segment_timesteps"].shape == (256, 20, 2)
    assert prepared.batch["art_train_step_count"].tolist() == [20] * 256
    for row, sample in enumerate(ids.tolist()):
        times = prepared.batch["train_timesteps"][row]
        torch.testing.assert_close(times.sort().values, original["all_timesteps"][sample].sort().values)
        indices = (original["all_timesteps"][sample, None] == times[:, None]).nonzero()[:, 1]
        torch.testing.assert_close(prepared.batch["train_conditional_velocities"][row],
                                   original["all_conditional_velocities"][sample, indices])
    for rank in range(32):
        local = prepared.batch[rank * 8:(rank + 1) * 8]
        assert local["art_update_window"].tolist() == [0] * 4 + [1] * 4
        assert local["art_mask_exempt"].tolist() == [True] * 4 + [False] * 4
    for window in (0, 1):
        rows = prepared.batch["art_update_window"] == window
        assert torch.bincount(ids[rows] // 8).tolist() == [4] * 32

    engine = TinyFullEngine()
    engine.copy_adapter()
    for window in (0, 1):
        engine._full_policy_version = window
        engine._validate_full_cached_window(prepared.batch[window * 4:(window + 1) * 4], "train_timesteps")
    assert engine.events[-1][3]["training_timesteps"] == 20
    corrupt = prepared.batch[:4].clone()
    engine._full_policy_version = 0
    corrupt["train_timesteps"][:, -1] = corrupt["train_timesteps"][:, 0]
    with pytest.raises(ValueError, match="rollout time grid"):
        engine._validate_full_cached_window(corrupt, "train_timesteps")


def test_changing_audio_only_reward_leaves_video_guidance_unchanged():
    config = routed_config()
    baseline = full20_batch()
    changed = full20_batch()
    changed.non_tensor_batch["reward/clap"] = changed.non_tensor_batch["reward/clap"][::-1].copy()
    first = DiffusionARTLoss.prepare_actor_batch(baseline, torch.zeros(256), config)
    second = DiffusionARTLoss.prepare_actor_batch(changed, torch.zeros(256), config)
    torch.testing.assert_close(first.batch["art_guidance"][:, :12], second.batch["art_guidance"][:, :12])
    assert not torch.allclose(first.batch["art_guidance"][:, 12:], second.batch["art_guidance"][:, 12:])


@pytest.mark.parametrize("window", [0, 1])
def test_separate_masks_and_equal_modality_mean_have_oracle_gradient(window):
    config = routed_config().actor_rollout_ref.actor
    old = torch.zeros(4, 13, 2)
    current = old.clone()
    current[:, :12] = -0.1
    current.requires_grad_()
    guidance = torch.ones_like(old)
    sigma, sigma_next = torch.full((4,), 0.9), torch.full((4,), 0.8)
    exemption = torch.full((4,), window == 0, dtype=torch.bool)
    actual, metrics = DiffusionARTLoss.compute_loss(
        forward_prediction=current, old_prediction=old, art_guidance=guidance,
        t_expanded=sigma[:, None, None], sigma_next=sigma_next, config=config,
        video_seq_len=torch.full((4,), 12), art_mask_exempt=exemption,
        art_update_window=torch.full((4,), window, dtype=torch.int64),
    )
    video = art_loss(current[:, :12], old[:, :12], guidance[:, :12], sigma, sigma_next, mask_exempt=exemption)
    audio = art_loss(current[:, 12:], old[:, 12:], guidance[:, 12:], sigma, sigma_next, mask_exempt=exemption)
    expected = (video["per_item_loss"].mean() + audio["per_item_loss"].mean()) / 2
    torch.testing.assert_close(actual, expected)
    expected_gradient = torch.autograd.grad(expected, current, retain_graph=True)[0]
    actual_gradient = torch.autograd.grad(actual, current)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)
    assert metrics["actor/video/mask_frac"] == float(window)
    assert metrics["actor/audio/mask_frac"] == 0.0


def test_routed_group_standardization_rejects_incomplete_prompt_group():
    config = routed_config()
    batch = make_batch(prompts=2)[:15]
    with pytest.raises(ValueError, match="complete prompt-K8"):
        DiffusionARTLoss._modality_rewards(batch, torch.zeros(15), config, ["video", "audio"])
