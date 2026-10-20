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

import ast
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]


def load_production_losses():
    path = ROOT / "verl_omni/trainer/diffusion/diffusion_algos.py"
    tree = ast.parse(path.read_text())
    names = {"DiffusionLossResult", "_format_available_keys", "DiffusionLossFn", "DiffusionNFTLoss", "OmniNFTLoss"}
    selected = [node for node in tree.body if getattr(node, "name", None) in names]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    namespace = {
        "ABC": ABC,
        "abstractmethod": abstractmethod,
        "dataclass": dataclass,
        "defaultdict": defaultdict,
        "torch": torch,
        "register_diffusion_loss": lambda name: lambda cls: cls,
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["OmniNFTLoss"], namespace["DiffusionNFTLoss"]


OmniNFTLoss, DiffusionNFTLoss = load_production_losses()


class ConfigNamespace(SimpleNamespace):
    def get(self, name, default=None):
        return getattr(self, name, default)


def production_wrapper(config, model_output, reward_prob):
    path = ROOT / "verl_omni/workers/utils/losses.py"
    tree = ast.parse(path.read_text())
    selected = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "diffusion_loss")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, selected], type_ignores=[]))
    namespace = {
        "get_diffusion_loss_fn": lambda name: OmniNFTLoss(),
        "Metric": SimpleNamespace(from_dict=lambda values, aggregation: values),
        "AggregationType": SimpleNamespace(MEAN="mean"),
        "tu": SimpleNamespace(get_non_tensor_data=lambda data, name, default=None: data.get(name, default)),
    }
    exec(compile(module, str(path), "exec"), namespace)
    data = {"reward_prob": reward_prob, "gradient_accumulation_steps": 1, "sp_size": 1}
    return namespace["diffusion_loss"](config=config, model_output=model_output, data=data)


def make_config(prompt_group_std=True, **overrides):
    loss = ConfigNamespace(
        loss_mode="omni_nft",
        prompt_group_std=prompt_group_std,
        mix_beta=1.0,
        adaptive_weight_min=1e-5,
        adv_clip_max=5.0,
        video_weight=1.0,
        audio_weight=1.0,
        ref_kl_coef=0.0,
        video_ref_kl_coef=0.0,
        audio_ref_kl_coef=0.0,
    )
    for name, value in overrides.items():
        setattr(loss, name, value)
    actor = SimpleNamespace(
        diffusion_loss=loss, data_loader_seed=42, use_kl_loss=False, use_distill_loss=False,
        global_batch_info={}, loss_scale_factor=1.0,
    )
    routes = {
        "video_align": (1.0, 0.0),
        "hpsv3": (1.5, 0.0),
        "audiobox": (0.0, 0.5),
        "clap": (0.0, 1.0),
        "desync": (1.0, 1.0),
    }
    return SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(actor=actor, rollout=SimpleNamespace(n=8)),
        algorithm=SimpleNamespace(
            norm_adv_by_std_in_grpo=True, global_std=False, timestep_fraction=1.0, adv_mode="continuous"
        ),
        reward=SimpleNamespace(
            reward_functions={
                name: {"routing_weights": {"video": weights[0], "audio": weights[1]}}
                for name, weights in routes.items()
            }
        ),
    )


def make_batch(prompts=2):
    generator = torch.Generator().manual_seed(42)
    scores = torch.randn(prompts * 8, 5, generator=generator)
    for prompt in range(prompts):
        scores[prompt * 8 : (prompt + 1) * 8] = scores[prompt * 8 : (prompt + 1) * 8] * (prompt + 1) ** 2 + prompt * 10
    scores[:8, 2] = 7.0
    return SimpleNamespace(
        batch={
            "rm_scores": scores,
            "video_latents_clean": torch.zeros(prompts * 8, 3, 2),
            "audio_latents_clean": torch.zeros(prompts * 8, 2, 2),
            "train_timesteps": torch.linspace(1000, 50, 20).expand(prompts * 8, -1).clone(),
        },
        non_tensor_batch={"uid": np.repeat(np.arange(prompts), 8)},
        meta_info={"reward_names": ["video_align", "hpsv3", "audiobox", "clap", "desync"]},
    )


def expected_signals(batch):
    scores = batch.batch["rm_scores"]
    grouped = scores.reshape(-1, 8, 5)
    normalized = (
        (grouped - grouped.mean(1, keepdim=True)) / (grouped.std(1, correction=1, keepdim=True) + 1e-4)
    ).reshape(-1, 5)
    video = normalized[:, 0] + 1.5 * normalized[:, 1] + normalized[:, 4]
    audio = 0.5 * normalized[:, 2] + normalized[:, 3] + normalized[:, 4]
    return normalized, torch.stack((video, audio), dim=1)


def test_full_k8_normalization_matches_art_and_keeps_original_routes_goodness():
    batch = make_batch()
    raw = batch.batch["rm_scores"].clone()
    expected, routed = expected_signals(batch)
    for component in range(raw.shape[1]):
        art = DiffusionNFTLoss._compute_group_advantages(
            raw[:, component], batch.non_tensor_batch["uid"], norm_by_std=True, global_std=False
        )
        torch.testing.assert_close(expected[:, component], art)
    normalized = OmniNFTLoss._compute_component_advantages(
        raw, batch.non_tensor_batch["uid"], norm_by_std=True, global_std=False, correction=1
    )
    torch.testing.assert_close(normalized, expected)
    assert torch.equal(normalized[:8, 2], torch.zeros(8))
    prepared = OmniNFTLoss.prepare_actor_batch(batch, raw, make_config())
    torch.testing.assert_close(
        prepared.batch["reward_prob"], (0.5 + routed.clamp(-5, 5)[:, None, :] / 10).expand(-1, 20, -1)
    )
    torch.testing.assert_close(prepared.batch["advantages"], routed.mean(1, keepdim=True).expand(-1, 20))
    torch.testing.assert_close(prepared.batch["sample_level_rewards"], raw)
    torch.testing.assert_close(prepared.batch["rm_scores"], raw)
    assert prepared.batch["train_timesteps"].shape == (16, 20)


def test_uid_grouping_is_permutation_invariant_and_uses_fp32():
    batch = make_batch()
    scores = batch.batch["rm_scores"].double()
    permutation = torch.randperm(16, generator=torch.Generator().manual_seed(7))
    original = OmniNFTLoss._compute_component_advantages(
        scores, batch.non_tensor_batch["uid"], norm_by_std=True, global_std=False, correction=1
    )
    permuted = OmniNFTLoss._compute_component_advantages(
        scores[permutation], batch.non_tensor_batch["uid"][permutation.numpy()],
        norm_by_std=True, global_std=False, correction=1,
    )
    assert permuted.dtype == torch.float32
    torch.testing.assert_close(permuted, original[permutation])
    shuffled_batch = SimpleNamespace(
        batch={name: value[permutation] for name, value in batch.batch.items()},
        non_tensor_batch={"uid": batch.non_tensor_batch["uid"][permutation.numpy()]},
        meta_info=batch.meta_info,
    )
    prepared = OmniNFTLoss.prepare_actor_batch(batch, batch.batch["rm_scores"], make_config())
    shuffled = OmniNFTLoss.prepare_actor_batch(shuffled_batch, shuffled_batch.batch["rm_scores"], make_config())
    torch.testing.assert_close(shuffled.batch["reward_prob"], prepared.batch["reward_prob"][permutation])


def test_constant_components_keep_all_256_samples_and_neutral_goodness():
    batch = make_batch(prompts=32)
    batch.batch["rm_scores"] = torch.full((256, 5), 17.0)
    prepared = OmniNFTLoss.prepare_actor_batch(batch, batch.batch["rm_scores"], make_config())
    torch.testing.assert_close(prepared.batch["reward_prob"], torch.full((256, 20, 2), 0.5))
    assert torch.equal(prepared.batch["advantages"], torch.zeros(256, 20))
    assert len(prepared.non_tensor_batch["uid"]) == 256


@pytest.mark.parametrize(
    "problem", ["local_four", "missing_uid_member", "wrong_k", "global_std", "no_std", "nonfinite"]
)
def test_opt_in_preparation_rejects_incomplete_or_inconsistent_normalization(problem):
    batch = make_batch()
    config = make_config()
    if problem == "local_four":
        batch.batch = {name: value[:4] for name, value in batch.batch.items()}
        batch.non_tensor_batch["uid"] = batch.non_tensor_batch["uid"][:4]
    elif problem == "missing_uid_member":
        batch.non_tensor_batch["uid"][-1] = 2
    elif problem == "wrong_k":
        config.actor_rollout_ref.rollout.n = 4
    elif problem == "global_std":
        config.algorithm.global_std = True
    elif problem == "no_std":
        config.algorithm.norm_adv_by_std_in_grpo = False
    else:
        batch.batch["rm_scores"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="prompt_group_std"):
        OmniNFTLoss.prepare_actor_batch(batch, batch.batch["rm_scores"], config)


def test_legacy_population_default_is_unchanged():
    batch = make_batch()
    scores = batch.batch["rm_scores"]
    grouped = scores.reshape(2, 8, 5)
    expected = (
        (grouped - grouped.mean(1, keepdim=True)) / (grouped.std(1, correction=0, keepdim=True) + 1e-4)
    ).reshape(16, 5)
    actual = OmniNFTLoss._compute_component_advantages(
        scores, batch.non_tensor_batch["uid"], norm_by_std=True, global_std=False
    )
    torch.testing.assert_close(actual, expected)
    prepared = OmniNFTLoss.prepare_actor_batch(batch, scores, make_config(prompt_group_std=False))
    routing = torch.tensor([[1.0, 0.0], [1.5, 0.0], [0.0, 0.5], [0.0, 1.0], [1.0, 1.0]])
    torch.testing.assert_close(
        prepared.batch["reward_prob"][:, 0], 0.5 + (expected @ routing).clamp(-5, 5) / 10
    )


def loss_inputs(sample_count=16):
    generator = torch.Generator().manual_seed(9)
    values = {}
    for modality, length in (("video", 3), ("audio", 2)):
        shape = (sample_count, length, 2)
        for field in ("forward_prediction", "old_prediction", "ref_forward_prediction", "x0", "xt"):
            value = torch.randn(shape, generator=generator)
            values[f"{modality}_{field}"] = value.requires_grad_(field.endswith("prediction"))
        values[f"{modality}_t_expanded"] = torch.full((sample_count, 1, 1), 0.4)
    return values


def test_no_reference_loss_and_gradients_equal_legacy_zero_coefficient_path():
    batch = make_batch()
    prepared = OmniNFTLoss.prepare_actor_batch(batch, batch.batch["rm_scores"], make_config())
    goodness = prepared.batch["reward_prob"][:, 0]
    actor_config = make_config().actor_rollout_ref.actor
    supplied = loss_inputs()
    absent = {name: value.detach().clone().requires_grad_(value.requires_grad) for name, value in supplied.items()}
    absent["video_ref_forward_prediction"] = None
    absent["audio_ref_forward_prediction"] = None
    reference_loss, reference_metrics = OmniNFTLoss.compute_loss(
        **supplied, video_reward_prob=goodness[:, 0], audio_reward_prob=goodness[:, 1], config=actor_config
    )
    actual_loss, actual_metrics = OmniNFTLoss.compute_loss(
        **absent, video_reward_prob=goodness[:, 0], audio_reward_prob=goodness[:, 1], config=actor_config
    )
    torch.testing.assert_close(actual_loss, reference_loss, rtol=0, atol=0)
    reference_loss.backward()
    actual_loss.backward()
    for modality in ("video", "audio"):
        torch.testing.assert_close(
            absent[f"{modality}_forward_prediction"].grad,
            supplied[f"{modality}_forward_prediction"].grad, rtol=0, atol=0,
        )
        assert supplied[f"{modality}_old_prediction"].grad is None
        assert supplied[f"{modality}_ref_forward_prediction"].grad is None
        assert f"actor/{modality}/ref_kl_loss" not in actual_metrics
        assert f"actor/{modality}/ref_kl_loss" in reference_metrics


@pytest.mark.parametrize("coefficient", ["ref_kl_coef", "video_ref_kl_coef", "audio_ref_kl_coef", "use_kl_loss"])
def test_missing_reference_rejected_for_every_nonzero_reference_control(coefficient):
    actor_config = make_config().actor_rollout_ref.actor
    if coefficient == "use_kl_loss":
        actor_config.use_kl_loss = True
    else:
        setattr(actor_config.diffusion_loss, coefficient, 1e-4)
    assert OmniNFTLoss.requires_reference(actor_config)
    model_output = loss_inputs()
    model_output.pop("video_ref_forward_prediction")
    model_output.pop("audio_ref_forward_prediction")
    data = {"reward_prob": torch.full((16, 2), 0.7)}
    loss = OmniNFTLoss()
    loss.validate_inputs(loss_name="omni_nft", model_output=model_output, data=data)
    with pytest.raises(ValueError, match="reference predictions are required"):
        loss(config=actor_config, model_output=model_output, data=data)
    with pytest.raises(ValueError, match="reference predictions are required"):
        production_wrapper(actor_config, model_output, data["reward_prob"])


@pytest.mark.parametrize("explicit_none", [True, False])
def test_zero_reference_callable_accepts_none_or_missing_optional_keys(explicit_none):
    actor_config = make_config().actor_rollout_ref.actor
    assert not OmniNFTLoss.requires_reference(actor_config)
    model_output = loss_inputs()
    for modality in ("video", "audio"):
        if explicit_none:
            model_output[f"{modality}_ref_forward_prediction"] = None
        else:
            model_output.pop(f"{modality}_ref_forward_prediction")
    reward_prob = torch.full((16, 2), 0.7)
    result = OmniNFTLoss()(config=actor_config, model_output=model_output, data={"reward_prob": reward_prob})
    assert torch.isfinite(result.loss)
    assert not any("ref_kl_loss" in name for name in result.metrics)
    wrapped, metrics = production_wrapper(actor_config, model_output, reward_prob)
    torch.testing.assert_close(wrapped, result.loss)
    assert not any("ref_kl_loss" in name for name in metrics)


def test_positive_legacy_reference_loss_keeps_exact_weighted_mse_and_gradients():
    config = make_config(
        ref_kl_coef=1e-4, video_ref_kl_coef=2e-4, audio_ref_kl_coef=3e-4
    ).actor_rollout_ref.actor
    inputs = loss_inputs()
    goodness = torch.full((16,), 0.7)
    total, metrics = OmniNFTLoss.compute_loss(
        **inputs, video_reward_prob=goodness, audio_reward_prob=goodness, config=config
    )
    expected = total.new_zeros(())
    for modality in ("video", "audio"):
        distance = (
            inputs[f"{modality}_forward_prediction"] - inputs[f"{modality}_ref_forward_prediction"]
        ).square().mean()
        torch.testing.assert_close(distance.detach(), torch.tensor(metrics[f"actor/{modality}/ref_kl_loss"]))
        expected = (
            expected + metrics[f"actor/{modality}/policy_loss"]
            + getattr(config.diffusion_loss, f"{modality}_ref_kl_coef") * distance
        )
    torch.testing.assert_close(total, expected / 2)
    total.backward()
    for modality in ("video", "audio"):
        assert torch.isfinite(inputs[f"{modality}_forward_prediction"].grad).all()
        assert inputs[f"{modality}_ref_forward_prediction"].grad is None
        assert inputs[f"{modality}_old_prediction"].grad is None


@pytest.mark.parametrize("microbatch", [1, 2, 4, 8])
def test_full_k8_goodness_loss_gradient_equivalence_under_sample_accumulation(microbatch):
    batch = make_batch(prompts=32)
    config = make_config()
    prepared = OmniNFTLoss.prepare_actor_batch(batch, batch.batch["rm_scores"], config)
    goodness = prepared.batch["reward_prob"][:, 0]
    inputs = loss_inputs(256)
    for modality in ("video", "audio"):
        inputs[f"{modality}_ref_forward_prediction"] = None
    full_parameter = torch.tensor(0.3, requires_grad=True)
    micro_parameter = full_parameter.detach().clone().requires_grad_(True)

    def evaluate(parameter, start, stop):
        selected = {name: value[start:stop] if value is not None else None for name, value in inputs.items()}
        for modality in ("video", "audio"):
            selected[f"{modality}_forward_prediction"] = (
                selected[f"{modality}_forward_prediction"].detach() * parameter
            )
        return OmniNFTLoss.compute_loss(
            **selected, video_reward_prob=goodness[start:stop, 0], audio_reward_prob=goodness[start:stop, 1],
            config=config.actor_rollout_ref.actor,
        )[0]

    evaluate(full_parameter, 0, 256).backward()
    for start in range(0, 256, microbatch):
        (evaluate(micro_parameter, start, start + microbatch) * microbatch / 256).backward()
    torch.testing.assert_close(micro_parameter.grad, full_parameter.grad, rtol=2e-5, atol=1e-6)
