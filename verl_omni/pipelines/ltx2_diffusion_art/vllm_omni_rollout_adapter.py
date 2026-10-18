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
"""LTX-2.3 joint audio-video rollout for DiffusionART: a pure ODE whose every visited state is kept."""

from __future__ import annotations

import torch
from vllm_omni.diffusion.request import OmniDiffusionRequest

from verl_omni.pipelines.ltx2_flow_grpo.common import normalize_ltx_output_type
from verl_omni.pipelines.ltx2_flow_grpo.vllm_omni_rollout_adapter import LTX23PipelineWithLogProb
from verl_omni.pipelines.model_base import VllmOmniPipelineBase

__all__ = ["LTX23DiffusionARTPipeline"]


@VllmOmniPipelineBase.register("LTX2Pipeline", algorithm="diffusion_art")
class LTX23DiffusionARTPipeline(LTX23PipelineWithLogProb):
    """Deterministic LTX-2.3 rollout that keeps every visited joint video/audio state.

    Reuses the FlowGRPO trajectory capture (``all_latents`` / ``all_next_latents`` /
    ``all_timesteps`` / ``video_seq_len``) and the base adapter's named media artifacts
    (``video_preview`` with the resolved frame rate, ``audio`` with the vocoder sample rate,
    ``video_latent`` / ``audio_latent``; see ``docs/advanced/diffusion_media_artifacts.md``), which the
    audio-video scorers select by name. This class only changes the sampler: noise level 0 at every
    step, all ``T`` transitions recorded, no log-probabilities. Per-modality CFG (``video_cfg_scale`` /
    ``audio_cfg_scale``) is allowed because the DiffusionART training adapter applies the same
    clean-space guidance when it recomputes ``v_old``. STG, modality and rescale guidance are passed
    through to the native LTX sampler untouched: validation rollouts (``val_kwargs``) may use them,
    while ``DiffusionARTLoss.validate_trainer_config`` keeps them at their neutral values for the
    training rollout, where the recomputed ``v_old`` has to match the sampled velocity.
    Opt-in conditional caching captures the native positive branch before CFG instead, without
    changing the rollout's guided trajectory; the trainer then uses conditional-only predictions.
    """

    supports_request_batch = False

    def _configure_flow_grpo(self, req: OmniDiffusionRequest) -> None:
        req.sampling_params.output_type = normalize_ltx_output_type(req.sampling_params.output_type)
        extra_args = dict(req.sampling_params.extra_args or {})
        self._art_cached_conditional_velocity = extra_args.get("art_cached_conditional_velocity", False)
        if not isinstance(self._art_cached_conditional_velocity, bool):
            raise ValueError("art_cached_conditional_velocity must be a boolean.")
        guidance_scale = float(getattr(req.sampling_params, "guidance_scale", None) or 1.0)
        for key in ("video_cfg_scale", "audio_cfg_scale"):
            configured = extra_args.get(key, None)
            extra_args[key] = float(configured) if configured is not None else guidance_scale
        task = extra_args.get("task")
        if task not in (None, "t2va"):
            raise NotImplementedError(f"LTX-2.3 DiffusionART supports text-to-audio-video only, got task={task!r}.")
        req.sampling_params.extra_args = extra_args
        self._flow_grpo_noise_level = 0.0
        self._flow_grpo_sde_type = "sde"
        self._flow_grpo_window_size = None
        self._flow_grpo_window_range = None
        self._flow_grpo_sde_contiguous = True
        self._flow_grpo_logprobs = False
        self._flow_grpo_task = task
        self._flow_grpo_seed = 0

    def _select_sde_steps(self, num_steps: int, device: torch.device) -> list[int]:
        del device
        return list(range(num_steps))

    def run_phase(self, *args, **kwargs):
        """Publish conditional predictions only for an explicitly requested cache."""
        if not getattr(self, "_art_cached_conditional_velocity", False):
            return super().run_phase(*args, **kwargs)
        self._art_conditional_velocities = []
        try:
            result = super().run_phase(*args, **kwargs)
            if len(self._art_conditional_velocities) != len(self._current_latents):
                raise RuntimeError("ART conditional cache must contain exactly one prediction per rollout state.")
            cache = torch.stack(self._art_conditional_velocities, dim=1)
            if cache.shape != self._flow_grpo_trajectory["all_latents"].shape:
                raise ValueError("ART conditional cache shape does not match the joint rollout trajectory.")
            self._flow_grpo_trajectory["all_conditional_velocities"] = cache
            return result
        finally:
            self._art_conditional_velocities = []

    def _predict_noise_for_step(self, index, timestep, state, forward_ctx, denoise_ctx):
        if not getattr(self, "_art_cached_conditional_velocity", False):
            return super()._predict_noise_for_step(index, timestep, state, forward_ctx, denoise_ctx)
        if forward_ctx.guidance_parallel_ready:
            raise NotImplementedError("ART conditional caching requires guidance_parallel_size=1.")
        plan = self._guidance_plan
        names = plan.names
        if names.count("cond") != 1 or len(set(names)) != len(names):
            raise ValueError("ART conditional caching requires one unambiguous native cond pass.")
        cond_index = names.index("cond")
        cond_pass = plan.passes[cond_index]
        if cond_pass.negative_video_context or cond_pass.negative_audio_context:
            raise ValueError("ART cond pass must use positive video and audio context.")
        video_mask = self._video_policy_mask(state, denoise_ctx)
        if not bool(video_mask.all()):
            raise NotImplementedError("ART conditional caching supports text-to-audio-video only.")
        batch_size = state.video.shape[0]
        audio_length = int(forward_ctx.original_audio_num_frames)
        if not 0 < audio_length <= state.audio.shape[1]:
            raise ValueError("ART conditional caching requires a valid logical audio length.")
        captured = []

        def capture(_module, _args, output):
            # vllm-omni 12e9280010af: ltx2_guidance.py:502-540 cats contexts in plan order,
            # then maps transformer output.chunk(pass_count) to plan.names before CFG.
            if captured or not isinstance(output, tuple) or len(output) != 2:
                raise RuntimeError("ART cache requires exactly one joint transformer output per step.")
            video, audio = output
            for value, source in ((video, state.video), (audio, state.audio)):
                expected = (batch_size * len(names), *source.shape[1:])
                if not isinstance(value, torch.Tensor) or value.shape != expected:
                    raise ValueError("ART transformer output shape does not match the native guidance plan.")
            rows = slice(cond_index * batch_size, (cond_index + 1) * batch_size)
            captured.append(torch.cat((video[rows], audio[rows, :audio_length]), dim=1).detach().float())

        handle = self.transformer.register_forward_hook(capture)
        try:
            prediction = super()._predict_noise_for_step(index, timestep, state, forward_ctx, denoise_ctx)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError("ART conditional cache did not observe the native transformer output.")
        if index != len(self._art_conditional_velocities):
            raise RuntimeError("ART conditional cache requires consecutive rollout steps starting at zero.")
        self._art_conditional_velocities.append(captured[0])
        return prediction
