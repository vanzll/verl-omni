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
"""Combined first-window mask exemption and prompt-component standardization."""

import ast
from pathlib import Path

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from verl_omni.trainer.diffusion.diffusion_algos import DiffusionARTLoss
from verl_omni.trainer.diffusion.diffusion_art import art_loss

from test_art_joint_latent_on_cpu import make_batch, make_config


def test_first_window_preserves_target_and_second_window_preserves_gradient():
    old = torch.zeros(4, 3)
    guidance = torch.ones_like(old)
    current = torch.full_like(old, -0.1, requires_grad=True)
    t = torch.full((4,), 0.9)
    baseline = art_loss(current, old, guidance, t, t - 0.05)
    exempt = torch.tensor([True, True, False, False])
    result = art_loss(current, old, guidance, t, t - 0.05, mask_exempt=exempt)
    assert baseline["masked"].all()
    assert result["would_mask"].all()
    assert result["masked"].tolist() == [False, False, True, True]
    torch.testing.assert_close(result["target"], baseline["target"], rtol=0, atol=0)
    expected = t[:2] * (current[:2] - baseline["target"][:2]).square().mean(1)
    torch.testing.assert_close(result["per_item_loss"][:2], expected)
    g = torch.autograd.grad(result["per_item_loss"].mean(), current, retain_graph=True)[0]
    bg = torch.autograd.grad(baseline["per_item_loss"].mean(), current)[0]
    torch.testing.assert_close(g[2:], bg[2:], rtol=0, atol=0)
    assert not result["target"].requires_grad


@pytest.mark.parametrize("micro", [1, 2, 4])
@pytest.mark.parametrize("rollout", [1, 2])
def test_window_resets_every_rollout_and_survives_dp_microbatch(micro, rollout):
    batch = make_batch()
    batch.meta_info["global_steps"] = rollout
    config = make_config(art_first_update_mask_exempt=True, art_prompt_group_std=True)
    prepared = DiffusionARTLoss.prepare_actor_batch(batch, torch.zeros(256), config)
    for rank in range(32):
        local = prepared.batch[rank * 8:(rank + 1) * 8]
        for window in range(2):
            half = local[window * 4:(window + 1) * 4]
            assert half["art_update_window"].tolist() == [window] * 4
            assert half["art_mask_exempt"].tolist() == [window == 0] * 4
            for start in range(0, 4, micro):
                assert half[start:start + micro]["art_mask_exempt"].tolist() == [window == 0] * micro
    for window in range(2):
        rows = prepared.batch["art_update_window"].numpy() == window
        counts = np.unique(prepared.non_tensor_batch["uid"][rows], return_counts=True)[1]
        assert counts.tolist() == [4] * 32


def test_component_std_uses_full_k_and_shared_component_once():
    batch = make_batch(prompts=2)
    config = make_config(art_prompt_group_std=True)
    routed, (normalized, names) = DiffusionARTLoss._modality_rewards(
        batch, torch.zeros(16), config, ["video", "audio"]
    )
    raw = torch.stack([torch.as_tensor(batch.non_tensor_batch[f"reward/{name}"]) for name in names], dim=1)
    groups = raw.reshape(2, 8, -1)
    expected = ((groups - groups.mean(1, keepdim=True)) / (groups.std(1, keepdim=True) + 1e-4)).reshape(16, -1)
    torch.testing.assert_close(normalized, expected)
    weights = {"video_align": 1.0, "hps": 1.5, "aesthetic": 0.5, "clap": 1.0, "desync": 1.0}
    weighted = expected @ torch.tensor([weights[name] for name in names])
    torch.testing.assert_close(routed["video"], weighted)
    torch.testing.assert_close(routed["audio"], weighted)
    batch.non_tensor_batch["reward/hps"][:8] = 17.0
    _, (normalized, names) = DiffusionARTLoss._modality_rewards(batch, torch.zeros(16), config, ["video", "audio"])
    assert torch.equal(normalized[:8, names.index("hps")], torch.zeros(8))
    torch.testing.assert_close(normalized[8:], expected[8:])


def test_exemption_keeps_inactive_rows_in_original_denominator():
    old = torch.zeros(4, 3)
    current = torch.full_like(old, -0.1, requires_grad=True)
    guidance = torch.ones_like(old)
    guidance[2:] = 0
    t = torch.full((4,), 0.9)
    terms = art_loss(current, old, guidance, t, t - 0.05, mask_exempt=torch.ones(4, dtype=torch.bool))
    assert terms["per_item_loss"][2:].eq(0).all()
    torch.testing.assert_close(terms["per_item_loss"].mean(), terms["per_item_loss"][:2].sum() / 4)


def test_exemption_requires_two_balanced_updates():
    config = make_config(art_first_update_mask_exempt=True, art_prompt_balanced_split=False)
    config.actor_rollout_ref.actor.diffusion_loss.art_first_update_mask_exempt = True
    with pytest.raises(ValueError, match="requires art_prompt_balanced_split"):
        DiffusionARTLoss.validate_trainer_config(config)
    config = make_config(art_first_update_mask_exempt=True)
    config.actor_rollout_ref.actor.diffusion_loss.art_first_update_mask_exempt = True
    config.actor_rollout_ref.actor.ppo_mini_batch_size = 8
    with pytest.raises(ValueError, match="exactly two disjoint"):
        DiffusionARTLoss.validate_trainer_config(config)


@pytest.mark.parametrize("missing", ["art_mask_exempt", "art_update_window"])
def test_a1_rejects_missing_markers(missing):
    config = make_config()
    config.actor_rollout_ref.actor.diffusion_loss.art_first_update_mask_exempt = True
    inputs = dict(
        forward_prediction=torch.full((4, 3, 1), -0.1, requires_grad=True),
        old_prediction=torch.zeros(4, 3, 1),
        art_guidance=torch.ones(4, 3, 1),
        t_expanded=torch.full((4, 1, 1), 0.9),
        sigma_next=torch.full((4,), 0.8),
        art_update_window=torch.zeros(4, dtype=torch.long),
        art_mask_exempt=torch.ones(4, dtype=torch.bool),
        config=config.actor_rollout_ref.actor,
    )
    del inputs[missing]
    with pytest.raises(ValueError, match="A1 requires both"):
        DiffusionARTLoss.compute_loss(**inputs)


def test_a2_rejects_scalar_fallback():
    config = make_config(art_prompt_group_std=True)
    with pytest.raises(ValueError, match="requires named component"):
        DiffusionARTLoss._modality_rewards(make_batch(prompts=2), torch.zeros(16), config, [""])


def test_engine_passes_window_markers_into_loss():
    path = Path(__file__).resolve().parents[3] / "verl_omni/workers/engine/fsdp/diffusers_impl.py"
    tree = ast.parse(path.read_text())
    engine = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ARTDiffusersFSDPEngine")
    function = next(node for node in engine.body if isinstance(node, ast.FunctionDef) and node.name == "art_loss_inputs")
    function.decorator_list = []
    namespace = {"torch": torch, "TensorDict": TensorDict}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    data = TensorDict({
        "train_timesteps_next": torch.full((4, 1), 800.0),
        "art_guidance": torch.ones(4, 3, 1),
        "art_update_window": torch.zeros(4, dtype=torch.long),
        "art_mask_exempt": torch.ones(4, dtype=torch.bool),
    }, batch_size=[4])
    result = namespace["art_loss_inputs"](data, 0, torch.zeros(4, 3, 1), torch.full((4, 1, 1), 0.9), 1000)
    assert result["art_update_window"] is data["art_update_window"]
    assert result["art_mask_exempt"] is data["art_mask_exempt"]


def test_second_window_rejects_exemption_and_first_window_logs_hypothetical_mask():
    config = make_config()
    actor = config.actor_rollout_ref.actor
    actor.diffusion_loss.art_first_update_mask_exempt = True
    inputs = dict(
        forward_prediction=torch.full((4, 3, 1), -0.1, requires_grad=True),
        old_prediction=torch.zeros(4, 3, 1), art_guidance=torch.ones(4, 3, 1),
        t_expanded=torch.full((4, 1, 1), 0.9), sigma_next=torch.full((4,), 0.8),
        art_update_window=torch.ones(4, dtype=torch.long),
        art_mask_exempt=torch.ones(4, dtype=torch.bool), config=actor,
    )
    with pytest.raises(ValueError, match="exemption must equal"):
        DiffusionARTLoss.compute_loss(**inputs)
    inputs["art_update_window"].zero_()
    _, metrics = DiffusionARTLoss.compute_loss(**inputs)
    assert metrics["actor/newjoint/update1/masked_state_fraction"] == 0.0
    assert metrics["actor/newjoint/update1/would_mask_state_fraction"] == 1.0
