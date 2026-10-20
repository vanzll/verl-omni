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
"""CPU boundary tests executing production methods without importing the GPU dependency stack.

The native dependency is a tiny two-pass transformer harness, not an LTX numerical
oracle. Every cache used by the engine tests is produced by the real capture hook.
The dependency-backed LTX tests separately exercise the pinned native guidance executor.
"""

import ast
import hashlib
import os
from collections.abc import Mapping
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

ROOT = Path(__file__).parents[2]


def load_source(path, name, namespace, methods=None):
    tree = ast.parse((ROOT / path).read_text())
    node = next(item for item in tree.body if getattr(item, "name", None) == name)
    if isinstance(node, ast.ClassDef):
        node.decorator_list = []
        if methods is not None:
            node.body = [item for item in node.body if getattr(item, "name", None) in methods]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(ROOT / path), "exec"), namespace)
    return namespace[name]


class TinyTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25))
        self.calls = []

    def forward(self, hidden_states, audio_hidden_states, encoder_hidden_states, **kwargs):
        video = hidden_states * self.weight + encoder_hidden_states
        audio = audio_hidden_states * self.weight - encoder_hidden_states
        self.calls.append((video.detach().clone(), audio.detach().clone()))
        return video, audio


class NativeHarness:
    def _predict_noise_for_step(self, index, timestep, state, forward_ctx, denoise_ctx):
        b = state.video.shape[0]
        context = state.video.new_tensor([2.0, -3.0]).repeat_interleave(b).view(-1, 1, 1)
        video, audio = self.transformer(state.video.repeat(2, 1, 1), state.audio.repeat(2, 1, 1), context)
        # Deliberately different CFG scales, so caching CFG cannot pass these tests.
        return video[:b] * 1.5 - video[b:] * 0.5, audio[:b] * 3 - audio[b:] * 2

    @torch.no_grad()
    def run_phase(self, req, request_inputs, **kwargs):
        state = self.initial_state
        for i in range(request_inputs.num_inference_steps):
            state = self._denoise_step(i, torch.tensor(900.0 - 200 * i), state, self.context, self.denoise)
        return state

    def _synchronize_guidance_parallel_step_output(self, state, **kwargs):
        return state


@pytest.fixture
def rollout():
    def clear_padding(audio, length):
        result = audio.clone()
        result[:, length:] = 0
        return result

    ns = {"torch": torch, "LTX2Pipeline": NativeHarness, "LTXAVState": NS, "clear_audio_padding": clear_padding}
    load_source(
        "verl_omni/pipelines/ltx2_flow_grpo/vllm_omni_rollout_adapter.py",
        "LTX23PipelineWithLogProb",
        ns,
        {"run_phase", "_denoise_step", "_video_policy_mask"},
    )
    common = {}
    load_source("verl_omni/pipelines/ltx2_flow_grpo/common.py", "normalize_ltx_output_type", common)
    ns.update(common)
    cls = load_source(
        "verl_omni/pipelines/ltx2_diffusion_art/vllm_omni_rollout_adapter.py", "LTX23DiffusionARTPipeline", ns
    )
    pipeline = cls()
    pipeline.device = torch.device("cpu")
    pipeline.transformer = TinyTransformer()
    pipeline.initial_state = NS(video=torch.ones(2, 4, 3), audio=torch.ones(2, 3, 3) * 2)
    pipeline.context = NS(
        guidance_parallel_ready=False,
        original_audio_num_frames=2,
        latent_num_frames=2,
        request_inputs=NS(generator=None),
    )
    pipeline.denoise = NS(conditioning_mask=None)
    pipeline._guidance_plan = NS(
        names=("cond", "uncond"),
        passes=(
            NS(negative_video_context=False, negative_audio_context=False),
            NS(negative_video_context=True, negative_audio_context=True),
        ),
    )
    pipeline.scheduler = NS(step=lambda pred, t, sample, **kw: (sample - 0.2 * pred, None, None, None))
    pipeline.request = NS(
        sampling_params=NS(
            output_type="pt",
            guidance_scale=1.5,
            extra_args={
                "art_cached_conditional_velocity": True,
                "art_joint_latent": True,
                "audio_cfg_scale": 3.0,
            },
        )
    )
    pipeline._configure_flow_grpo(pipeline.request)
    pipeline.run = lambda: pipeline.run_phase(
        None,
        NS(num_inference_steps=3),
        noise_scale=0.0,
        sigmas=None,
        timesteps=None,
        attention_kwargs=None,
        phase_recipe=NS(sampler="euler", adapter_slot=None),
    )
    return pipeline


def test_capture_actual_positive_output_and_preserve_baseline(rollout):
    rollout.run()
    cached = rollout._flow_grpo_trajectory["all_conditional_velocities"]
    assert cached.shape == (2, 3, 6, 3) and cached.dtype == torch.float32
    assert not cached.requires_grad
    for step, (video, audio) in enumerate(rollout.transformer.calls):
        torch.testing.assert_close(cached[:, step], torch.cat((video[:2], audio[:2, :2]), dim=1))
        cfg = (rollout._next_latents[step] - rollout._current_latents[step]) / -0.2
        assert not torch.allclose(cached[:, step], cfg)
    original_states = rollout._flow_grpo_trajectory["all_next_latents"].clone()
    rollout.request.sampling_params.extra_args = {}
    rollout._configure_flow_grpo(rollout.request)
    rollout.run()
    assert "all_conditional_velocities" not in rollout._flow_grpo_trajectory
    torch.testing.assert_close(original_states, rollout._flow_grpo_trajectory["all_next_latents"])
    assert not rollout.transformer._forward_hooks


@pytest.mark.parametrize("fault", ["parallel", "negative_cond", "missing_output", "shape", "exception"])
def test_capture_fails_closed_and_removes_hook(rollout, fault):
    if fault == "parallel":
        rollout.context.guidance_parallel_ready = True
    elif fault == "negative_cond":
        rollout._guidance_plan.passes[0].negative_video_context = True
    elif fault == "missing_output":
        rollout.transformer = NS(
            register_forward_hook=lambda hook: NS(remove=lambda: None), __call__=lambda *args: None
        )
    elif fault == "shape":
        rollout.transformer.register_forward_hook(lambda module, args, output: (output[0][:1], output[1]))
    else:
        rollout.scheduler.step = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("scheduler failure"))
    initial_hooks = len(getattr(rollout.transformer, "_forward_hooks", {}))
    with pytest.raises((RuntimeError, ValueError, NotImplementedError, TypeError)):
        rollout.run()
    assert len(getattr(rollout.transformer, "_forward_hooks", {})) == initial_hooks
    assert rollout._art_conditional_velocities == []


@pytest.fixture
def engine_case(rollout):
    rollout.run()
    trajectory = rollout._flow_grpo_trajectory
    config = NS(diffusion_loss=NS(art_cached_conditional_velocity=True))
    ns = {
        "torch": torch,
        "partial": partial,
        "NFTDiffusersFSDPEngine": object,
        "tu": NS(
            get_non_tensor_data=lambda batch, key, default=None: batch.get(key, default),
            get_tensordict=dict,
            assign_non_tensor=lambda data, **kwargs: data.update(kwargs),
        ),
    }
    cls = load_source("verl_omni/workers/engine/fsdp/diffusers_impl.py", "ARTDiffusersFSDPEngine", ns)
    engine = cls()
    engine.module = TinyTransformer()
    engine.model_config = NS()
    engine.full_finetune = False
    engine.get_data_parallel_group = lambda: None
    engine._adapter_predictions = lambda *a, **kw: pytest.fail("old adapter/forward must not run")
    engine.use_adapter = lambda *a, **kw: pytest.fail("must not switch adapter")
    engine.prepare_model_inputs = lambda micro_batch, step: (
        {
            "hidden_states": micro_batch["train_latents"][:, step, :4],
            "audio_hidden_states": micro_batch["train_latents"][:, step, 4:],
            "encoder_hidden_states": torch.full((2, 1, 1), 2.0),
        },
        None,
        None,
        micro_batch["train_latents"][:, step],
        torch.full((2, 1, 1), 0.5),
    )
    ns["forward"] = lambda module, model_inputs, **kw: torch.cat(module(**model_inputs), dim=1).float()

    def loss_fn(config, model_output, data, dp_group):
        assert "rollout_velocity" not in data
        assert data["art_guidance"] is batch["art_guidance"]
        assert data["art_anchor"] is batch["art_anchor"]
        return (model_output["forward_prediction"] - model_output["old_prediction"]).square().mean(), {}

    batch = {
        "train_conditional_velocities": trajectory["all_conditional_velocities"],
        "train_latents": trajectory["all_latents"],
        "train_latents_next": trajectory["all_next_latents"],
        "train_timesteps": trajectory["all_timesteps"],
        "train_timesteps_next": trajectory["all_timesteps"] - 200,
        "art_guidance": torch.ones(2, 6, 3),
        "art_anchor": torch.tensor([True, False]),
        "video_seq_len": torch.tensor([4, 4]),
        "num_train_timesteps": 1000,
    }
    return engine, batch, partial(loss_fn, config=config), ns


def test_engine_reads_selected_cache_without_old_forward(engine_case):
    engine, batch, loss_fn, _ = engine_case
    loss, output = engine.forward_step(batch, loss_fn, False, 1)
    torch.testing.assert_close(output["model_output"]["old_prediction"], batch["train_conditional_velocities"][:, 1])
    loss.backward()
    assert len(engine.module.calls) == 1 and engine.module.weight.grad is not None
    assert not output["model_output"]["old_prediction"].requires_grad


def test_cache_acceptance_audit_preserves_loss_and_gradient(engine_case, monkeypatch, tmp_path):
    import json

    engine, batch, loss_fn, _ = engine_case
    batch["uid"] = ["prompt-first", "prompt-second"]
    batch["_rollout_seed_global_idx"] = torch.tensor([3, 7])
    monkeypatch.delenv("OMNI_ART_CACHE_ACCEPTANCE", raising=False)
    baseline, _ = engine.forward_step(batch, loss_fn, False, 1)
    baseline.backward()
    gradient = engine.module.weight.grad.clone()
    engine.module.zero_grad()
    monkeypatch.setenv("OMNI_ART_CACHE_ACCEPTANCE", "1")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    audited, _ = engine.forward_step(batch, loss_fn, False, 1)
    audited.backward()
    torch.testing.assert_close(audited, baseline, rtol=0, atol=0)
    torch.testing.assert_close(engine.module.weight.grad, gradient, rtol=0, atol=0)
    path = tmp_path / "logs/cache_acceptance_rank_0.jsonl"
    record = json.loads(path.read_text())
    assert record["uid"] == batch["uid"] and record["sample_ids"] == [3, 7]
    assert record["current_old_mse"] == 0 and record["current_finite"] and record["old_finite"]
    assert record["timesteps"] == batch["train_timesteps"][:, 1].tolist()
    engine._art_cache_audit_count = 38
    engine.forward_step(batch, loss_fn, False, 1)
    assert len(path.read_text().splitlines()) == 1


def test_engine_default_retains_old_adapter_path(engine_case):
    engine, batch, _, _ = engine_case
    calls = []

    def old_forward(*args, **kwargs):
        calls.append(kwargs)
        return torch.zeros(2, 6, 3), torch.ones(2, 6, 3), None

    engine._adapter_predictions = old_forward
    engine.forward_step(batch, None, True, 0)
    assert calls == [{"reference": False}]


@pytest.mark.parametrize("fault", ["absent", "shape", "dtype", "nonfinite", "cfg"])
def test_engine_rejects_bad_cache_before_forward(engine_case, fault):
    engine, batch, loss_fn, _ = engine_case
    if fault == "absent":
        del batch["train_conditional_velocities"]
    elif fault == "shape":
        batch["train_conditional_velocities"] = batch["train_conditional_velocities"][:, :1]
    elif fault == "dtype":
        batch["train_conditional_velocities"] = batch["train_conditional_velocities"].bfloat16()
    elif fault == "nonfinite":
        batch["train_conditional_velocities"][0, 1, 0, 0] = float("nan")
    else:
        prepare = engine.prepare_model_inputs
        engine.prepare_model_inputs = lambda **kw: (prepare(**kw)[0], {}, *prepare(**kw)[2:])
    with pytest.raises(ValueError):
        engine.forward_step(batch, loss_fn, False, 1)
    assert engine.module.calls == []


def test_projection_and_staging_keep_selected_cache(engine_case):
    engine, batch, _, ns = engine_case
    # Exercise the actual driver-to-worker projection, with only DataProto conversion stubbed.
    project = load_source(
        "verl_omni/trainer/diffusion/diffusion_trainer_utils.py", "_to_diffusion_worker_tensordict", {}
    )
    projected = project(NS(to_tensordict=lambda: {**batch, "responses": torch.zeros(2, 3, 4, 4)}))
    assert "responses" not in projected
    assert projected["train_conditional_velocities"] is batch["train_conditional_velocities"]
    step_fields, shared = engine._prepare_timestep_staging(projected, "train_timesteps")
    assert "train_conditional_velocities" in step_fields and "art_anchor" in shared
    for step in range(3):
        staged = {key: projected[key] for key in shared if key in projected}
        staged.update({key: projected[key][:, step : step + width] for key, width in step_fields.items()})
        torch.testing.assert_close(
            staged["train_conditional_velocities"][:, 0], batch["train_conditional_velocities"][:, step]
        )


def test_transport_preserves_actual_cache(rollout):
    rollout.run()
    ns = {"torch": torch, "np": np, "Mapping": Mapping, "os": os}
    path = "verl_omni/workers/rollout/vllm_rollout/vllm_omni_diffusion_strategy.py"
    for name in ("_unbatch_training_field", "_rollout_metadata_groups"):
        load_source(path, name, ns)
    tree = ast.parse((ROOT / path).read_text())
    cls = next(
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and any(getattr(item, "name", None) == "_training_output_fields" for item in node.body)
    )
    ns["OmniStrategyBase"] = object
    strategy = load_source(path, cls, ns, {"_training_output_fields"})()
    strategy.server = NS(global_steps=1)
    trajectory = rollout._flow_grpo_trajectory
    single = {key: value[:1] for key, value in trajectory.items() if isinstance(value, torch.Tensor)}
    fields = strategy._training_output_fields(
        NS(
            trajectory_latents=single.pop("all_latents"),
            trajectory_timesteps=single.pop("all_timesteps"),
            multimodal_output={"metadata": {"rl": single}},
        ),
        context="test",
    )
    torch.testing.assert_close(fields["all_conditional_velocities"], trajectory["all_conditional_velocities"][0])


@pytest.mark.parametrize("stage_inputs", [False, True])
def test_temporal_loop_preserves_late_injected_metadata(engine_case, stage_inputs):
    """Run the production temporal loop with metadata absent when staging is planned."""
    engine, batch, loss_fn, ns = engine_case

    class CPUData(dict):
        def select(self, *keys, strict=False):
            return CPUData({key: self[key] for key in keys if key in self})

        def to(self, device):
            assert device == "cpu"
            return self

        def clone(self, recurse=False):
            return CPUData(self)

    def micro_batches(data, **kwargs):
        return [CPUData(data), CPUData(data)], None

    ns.update(
        nullcontext=nullcontext,
        get_device_id=lambda: "cpu",
        prepare_micro_batches=micro_batches,
        LoRAAdapterMixin=type("LoRAAdapterMixin", (), {}),
        FullNFTPolicyMixin=type("FullNFTPolicyMixin", (), {}),
        BaseEngine=type("BaseEngine", (), {}),
        ABC=type("ABC", (), {}),
    )
    base = load_source(
        "verl_omni/workers/engine/fsdp/diffusers_impl.py", "DiffusersFSDPEngine", ns, {"_run_forward_backward_batch"}
    )
    engine.ulysses_sequence_parallel_size = 2
    engine.postprocess_batch_func = lambda **kwargs: kwargs
    expected_accumulation = 2 * batch["train_timesteps"].shape[1]
    visited = []
    original_loss = loss_fn.func

    def check_loss(config, model_output, data, dp_group):
        assert data["sp_size"] == 2
        assert data["gradient_accumulation_steps"] == expected_accumulation
        visited.append(model_output["old_prediction"].clone())
        loss, metrics = original_loss(config, model_output, data, dp_group)
        return loss / data["gradient_accumulation_steps"], metrics

    data = CPUData(batch, enable_timestep_staging=stage_inputs)
    assert "sp_size" not in data and "gradient_accumulation_steps" not in data
    base._run_forward_backward_batch(
        engine, data, partial(check_loss, config=loss_fn.keywords["config"]), False, timesteps_key="train_timesteps"
    )
    assert len(visited) == expected_accumulation
    for i, old in enumerate(visited):
        torch.testing.assert_close(old, batch["train_conditional_velocities"][:, i % 3])


@pytest.mark.skipif(
    not os.environ.get("ART_NATIVE_SOURCE_DIR"), reason="Set ART_NATIVE_SOURCE_DIR to pinned source evidence"
)
@pytest.mark.parametrize("reverse_passes", [False, True])
def test_hook_against_fixed_native_source(rollout, monkeypatch, reverse_passes):
    """Execute SHA256-checked upstream runtime dispatch and guidance code, not the harness's batching."""
    source_dir = Path(os.environ["ART_NATIVE_SOURCE_DIR"])
    hashes = {
        "ltx2_guidance.py": "c43f49b21163605bef3cc2cbb8ebf6700d0b3bb7121142fc8a4bc380f6ec935b",
        "ltx2_runtime.py": "51b491277afeae2e5da17f7d96e8ad8de8f1e8b95467a644d3469b81dcb235f7",
    }
    trees = {}
    for name, digest in hashes.items():
        source = (source_dir / name).read_bytes()
        assert hashlib.sha256(source).hexdigest() == digest
        trees[name] = ast.parse(source)
    # Only remove GPU dependency imports and their logger initialization; preserve all native math and classes.
    tree = trees["ltx2_guidance.py"]
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.ImportFrom)
            and node.module.startswith(("vllm.", "vllm_omni."))
            or isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "logger" for target in node.targets)
        )
    ]
    ns = {"__name__": __name__}
    exec(compile(tree, str(source_dir / "ltx2_guidance.py"), "exec"), ns)
    runtime = next(
        node for node in trees["ltx2_runtime.py"].body if isinstance(node, ast.ClassDef) and node.name == "LTXRuntime"
    )
    dispatch = next(node for node in runtime.body if getattr(node, "name", None) == "_predict_noise_for_step")
    future = next(node for node in tree.body if isinstance(node, ast.ImportFrom) and node.module == "__future__")
    exec(compile(ast.Module(body=[future, dispatch], type_ignores=[]), str(source_dir / "ltx2_runtime.py"), "exec"), ns)
    monkeypatch.setattr(NativeHarness, "_predict_noise_for_step", ns["_predict_noise_for_step"])
    rollout.guidance_executor = ns["LTXGuidanceExecutor"]()
    spec = ns["LTXGuidanceSpec"](
        video=ns["LTXModalityGuidance"](cfg_scale=1.5), audio=ns["LTXModalityGuidance"](cfg_scale=3.0)
    )
    plan = ns["LTXGuidancePlan"].build(spec)
    if reverse_passes:
        plan = ns["LTXGuidancePlan"](spec=spec, passes=tuple(reversed(plan.passes)))
    rollout._guidance_plan = plan
    positive = torch.tensor([2.0, 7.0]).view(2, 1, 1)
    negative = torch.tensor([-3.0, -8.0]).view(2, 1, 1)
    rollout.context.prompt_context = NS(
        positive_connector_prompt_embeds=positive,
        positive_connector_audio_prompt_embeds=positive,
        negative_connector_prompt_embeds=negative,
        negative_connector_audio_prompt_embeds=negative,
    )
    rollout.context.attention_kwargs = None
    rollout.context.sampler = "euler"
    rollout.scheduler.sigmas = torch.tensor([0.9, 0.7, 0.5, 0.0])
    rollout.context.audio_scheduler = rollout.scheduler
    rollout._build_transformer_kwargs = lambda *args, **kwargs: kwargs
    rollout._transformer_cache_context = lambda *args: nullcontext()
    rollout._video_guidance_model_sigma = lambda sigma, denoise_ctx: sigma
    rollout.run()
    cache = rollout._flow_grpo_trajectory["all_conditional_velocities"]
    for i, states in enumerate(rollout._current_latents):
        expected = torch.cat((states[:, :4] * 0.25 + positive, states[:, 4:] * 0.25 - positive), dim=1)
        torch.testing.assert_close(cache[:, i], expected)
        cfg_velocity = (rollout._next_latents[i] - states) / -0.2
        assert not torch.allclose(cache[:, i], cfg_velocity)
    assert len(rollout.transformer.calls) == 3 and not rollout.transformer._forward_hooks
