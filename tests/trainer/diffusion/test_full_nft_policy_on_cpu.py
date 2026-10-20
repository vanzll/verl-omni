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
import copy
import importlib.util
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu


ROOT = Path(__file__).resolve().parents[3]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_module("nft_base_policy_test", ROOT / "verl_omni/workers/engine/full_policy.py")
namespace = {"torch": torch, "math": __import__("math"), "FullPolicyMixin": BASE.FullPolicyMixin,
             "local_tensor": BASE.local_tensor}
tree = ast.parse((ROOT / "verl_omni/workers/engine/full_nft_policy.py").read_text())
node = next(item for item in tree.body if isinstance(item, ast.ClassDef))
exec(compile(ast.Module(body=[node], type_ignores=[]), "full_nft_policy.py", "exec"), namespace)


class TinyNFT(namespace["FullNFTPolicyMixin"]):
    def __init__(self):
        self.model_config = SimpleNamespace(full_finetune=True, algorithm="omni_nft", lora_rank=0,
                                            lora_adapter_path=None)
        self.module = torch.nn.Linear(3, 2)
        self.events = []
        self._initialize_full_policy()
        self._nft_old_module = copy.deepcopy(self.module).requires_grad_(False)

    def _full_event(self, kind, **values):
        self.events.append((kind, values))


def test_distinct_frozen_ema_and_original_linear_decay():
    engine = TinyNFT()
    engine.copy_adapter()
    for rollout in range(1, 7):
        previous = copy.deepcopy(engine._nft_old_module.state_dict())
        with torch.no_grad():
            for parameter in engine.module.parameters():
                parameter.add_(0.1)
        for name, target in engine._nft_old_module.named_parameters():
            torch.testing.assert_close(target, previous[name], rtol=0, atol=0)
        engine._full_policy_version += 1
        with pytest.raises(ValueError):
            engine._full_export_version("old")
        decay = min(rollout * 0.001, 0.5)
        engine.ema_update_adapter(decay=decay)
        for name, target in engine._nft_old_module.named_parameters():
            expected = torch.lerp(previous[name], engine.module.state_dict()[name], 1 - decay)
            torch.testing.assert_close(target, expected, rtol=0, atol=0)
            assert not target.requires_grad and target.grad is None
        assert engine._full_export_version("old") == rollout
        with pytest.raises(RuntimeError):
            engine.ema_update_adapter(decay=decay)
    assert engine._nft_ema_updates == 6


def test_alias_and_art_window_are_rejected():
    engine = TinyNFT()
    engine.copy_adapter()
    batch = TensorDict({"train_timesteps": torch.ones(8, 20)}, batch_size=[8])
    engine._validate_full_cached_window(batch, "train_timesteps")
    batch["art_update_window"] = torch.zeros(8)
    with pytest.raises(ValueError):
        engine._validate_full_cached_window(batch, "train_timesteps")
    engine._nft_old_module = engine.module
    with pytest.raises(RuntimeError):
        engine._nft_parameter_pairs()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_ema_audit_matches_original_lerp_in_parameter_dtype(dtype):
    engine = TinyNFT()
    engine.module.to(dtype)
    engine._nft_old_module.to(dtype)
    engine.copy_adapter()
    with torch.no_grad():
        for parameter in engine.module.parameters():
            parameter.add_(0.1)
    engine._full_policy_version = 1
    engine.ema_update_adapter(decay=0.001)
    assert engine.events[-1][1]["expected_sample_max_error"] == 0


def test_one_update_full_twenty_timestep_sample_denominator():
    torch.manual_seed(42)
    original = torch.nn.Linear(3, 2)
    inputs = torch.randn(8, 3)
    target = torch.randn(8, 2)
    outputs = []
    for microbatch in (8, 4, 2, 1):
        module = copy.deepcopy(original)
        optimizer = torch.optim.AdamW(module.parameters(), lr=3e-5, betas=(0.9, 0.999), weight_decay=1e-4,
                                      foreach=False)
        optimizer.zero_grad()
        visited = []
        for start in range(0, 8, microbatch):
            for timestep in range(20):
                prediction = module(inputs[start:start + microbatch] + timestep * 0.001)
                loss = (prediction - target[start:start + microbatch]).square().mean()
                (loss / ((8 // microbatch) * 20)).backward()
                visited.extend((sample, timestep) for sample in range(start, start + microbatch))
        assert len(visited) == len(set(visited)) == 160
        norm = torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
        optimizer.step()
        outputs.append((module.state_dict(), norm))
    for state, norm in outputs[1:]:
        torch.testing.assert_close(norm, outputs[0][1])
        for name, value in state.items():
            torch.testing.assert_close(value, outputs[0][0][name], rtol=1e-5, atol=1e-7)


def test_production_forward_uses_distinct_old_and_skips_reference():
    path = ROOT / "verl_omni/workers/engine/fsdp/diffusers_impl.py"
    engine_class = next(item for item in ast.parse(path.read_text()).body
                        if isinstance(item, ast.ClassDef) and item.name == "OmniNFTDiffusersFSDPEngine")
    selected = [item for item in engine_class.body if isinstance(item, ast.FunctionDef)
                and item.name in ("forward_step", "prepare_model_outputs")]
    scope = {"torch": torch, "TensorDict": TensorDict, "tu": tu, "nullcontext": nullcontext,
             "forward": lambda module, **kwargs: module(**kwargs["model_inputs"])}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), scope)

    class PairedPolicy(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(value))
            self.inputs = []

        def forward(self, hidden_states, audio_hidden_states):
            self.inputs.append((hidden_states, audio_hidden_states))
            return hidden_states * self.weight, audio_hidden_states * self.weight

    current = PairedPolicy(2.0)
    old = PairedPolicy(1.0).requires_grad_(False)
    state = (torch.ones(4, 3, 2), torch.ones(4, 2, 2))
    inputs = {"hidden_states": state[0], "audio_hidden_states": state[1]}
    batch = TensorDict({"train_timesteps": torch.ones(4, 20), "reward_prob": torch.ones(4, 20, 2)}, batch_size=[4])
    tu.assign_non_tensor(batch, gradient_accumulation_steps=40, sp_size=1)
    config = SimpleNamespace(use_kl_loss=False, diffusion_loss={"ref_kl_coef": 0, "video_ref_kl_coef": 0,
                                                              "audio_ref_kl_coef": 0})

    def objective(config, model_output, **kwargs):
        assert model_output["video_ref_forward_prediction"] is None
        assert not model_output["video_old_prediction"].requires_grad
        return model_output["video_forward_prediction"].square().mean(), {}

    engine = SimpleNamespace(full_nft=True, _full_policy_version=7, module=current, _nft_old_module=old,
                             model_config=SimpleNamespace(),
                             prepare_model_inputs=lambda *args: (inputs, None, state, state, (state[0], state[1])),
                             prepare_model_outputs=scope["prepare_model_outputs"],
                             get_data_parallel_group=lambda: None)
    loss, _ = scope["forward_step"](engine, batch, partial(objective, config=config), False, 0)
    loss.backward()
    assert current.weight.grad.item() > 0 and old.weight.grad is None
    assert len(current.inputs) == len(old.inputs) == 1
    assert current.inputs[0][0] is old.inputs[0][0]
    assert current.inputs[0][1] is old.inputs[0][1]


def test_two_updates_freeze_ema_until_rollout_end(monkeypatch):
    monkeypatch.setenv("OMNI_VARIANT", "NFT_FULL32_U2")
    engine = TinyNFT()
    engine.copy_adapter()
    optimizer = torch.optim.AdamW(engine.module.parameters(), lr=3e-5, foreach=False)
    batch = TensorDict({"train_timesteps": torch.ones(4, 20)}, batch_size=[4])
    for rollout in range(1, 4):
        previous = copy.deepcopy(engine._nft_old_module.state_dict())
        for window in range(2):
            engine._validate_full_cached_window(batch, "train_timesteps")
            optimizer.zero_grad()
            for timestep in range(20):
                inputs = torch.full((4, 3), rollout + window + timestep * 0.01)
                with torch.no_grad():
                    old = engine._nft_old_module(inputs)
                (engine.module(inputs).sub(old + 0.5).square().mean() / 20).backward()
            optimizer.step()
            engine._full_policy_version += 1
            for name, target in engine._nft_old_module.named_parameters():
                torch.testing.assert_close(target, previous[name], rtol=0, atol=0)
            with pytest.raises(ValueError):
                engine._full_export_version("old")
            if window == 0:
                with pytest.raises(RuntimeError, match="all successful updates"):
                    engine.ema_update_adapter(decay=rollout * 0.001)
        with pytest.raises(ValueError):
            engine._validate_full_cached_window(batch, "train_timesteps")
        engine.ema_update_adapter(decay=rollout * 0.001)
        assert engine._nft_ema_rollouts == rollout
        assert engine._full_export_version("old") == 2 * rollout
        for name, target in engine._nft_old_module.named_parameters():
            expected = torch.lerp(previous[name], engine.module.state_dict()[name], 1 - rollout * 0.001)
            torch.testing.assert_close(target, expected, rtol=0, atol=0)
        with pytest.raises(RuntimeError):
            engine.ema_update_adapter(decay=rollout * 0.001)
    assert len(optimizer.state) == len(list(engine.module.parameters()))
    assert {int(state["step"]) for state in optimizer.state.values()} == {6}
