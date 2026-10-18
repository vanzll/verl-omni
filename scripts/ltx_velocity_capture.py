"""Capture actual scheduler inputs in an isolated LTX ART probe, never in training.

This is preparation for the GPU diagnostic, not evidence of cache equivalence.
The recorder delegates every model/scheduler call and retains the native BF16
state rounding. Actor recomputation and parameter-gradient comparisons are a
separate probe stage using the saved fixed inputs.
"""
from contextlib import AbstractContextManager
import hashlib
import json
import os
from pathlib import Path
import time
from types import MethodType

import torch

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')


def _cpu(value):
    return value.detach().to(device='cpu', copy=True).contiguous()


def _restore_attribute(instance, name, previous):
    if previous[0]:
        setattr(instance, name, previous[1])
    else:
        delattr(instance, name)


class LTXVelocityCapture(AbstractContextManager):
    """Temporarily observe selected native denoise steps in one private process."""

    def __init__(self, pipeline, identity, indices=(0, 9, 18)):
        if os.environ.get('OMNI_ISOLATED_VELOCITY_PROBE') != '1':
            raise RuntimeError('Recorder is restricted to an explicitly isolated probe')
        required = {'source_sha256', 'model_revision', 'adapter_fingerprint',
                    'seed_identity', 'attention_backend', 'video_cfg', 'audio_cfg'}
        if required - identity.keys():
            raise ValueError('Missing probe input identity: ' + str(sorted(required - identity.keys())))
        if not indices or min(indices) < 0 or len(set(indices)) != len(indices):
            raise ValueError('Use a nonempty set of distinct denoise indices')
        self.pipeline = pipeline
        self.identity = dict(identity)
        self.indices = frozenset(indices)
        self.records = []
        self.condition = None
        self.active = False

    def __enter__(self):
        p = self.pipeline
        if self.active or getattr(p, '_isolated_velocity_capture_active', False):
            raise RuntimeError('Nested/repeated active capture is not supported')
        self.original_denoise = p._denoise_step
        self.previous_denoise = ('_denoise_step' in vars(p), vars(p).get('_denoise_step'))

        def denoise(instance, index, timestep, state, forward_ctx, denoise_ctx):
            if index not in self.indices:
                return self.original_denoise(index, timestep, state, forward_ctx, denoise_ctx)
            if forward_ctx.sampler != 'euler' or forward_ctx.guidance_parallel_ready:
                raise ValueError('Probe currently supports Euler without guidance parallelism only')
            if float(instance._flow_grpo_noise_level) != 0.0:
                raise ValueError('Euler reconstruction requires noise_level=0')
            mask = instance._video_policy_mask(state, denoise_ctx)
            if not bool(mask.all()):
                raise ValueError('This fixed probe supports t2va, without conditioned image rows')
            audio_length = int(forward_ctx.original_audio_num_frames)
            video_length = int(state.video.shape[1])
            if self.condition is None:
                c = forward_ctx.prompt_context
                names = {
                    'prompt_embeds': 'positive_connector_prompt_embeds',
                    'audio_prompt_embeds': 'positive_connector_audio_prompt_embeds',
                    'negative_prompt_embeds': 'negative_connector_prompt_embeds',
                    'negative_audio_prompt_embeds': 'negative_connector_audio_prompt_embeds',
                }
                self.condition = {key: _cpu(getattr(c, name)) if getattr(c, name) is not None else None
                                  for key, name in names.items()}
            row = {'index': int(index), 'video_seq_len': video_length,
                   'audio_seq_len': audio_length, 'video_state_dtype': str(state.video.dtype),
                   'audio_state_dtype': str(state.audio.dtype), 'scheduler_calls': 0}
            scheduler = instance.scheduler
            original_step = scheduler.step
            previous_step = ('step' in vars(scheduler), vars(scheduler).get('step'))
            started = time.perf_counter()

            def observe_step(model_output, timestep, sample, *args, **kwargs):
                if kwargs.get('noise_level') != 0.0 or kwargs.get('per_token_timesteps') is not None:
                    raise ValueError('Only the original zero-noise scalar-sigma transition is supported')
                row['scheduler_calls'] += 1
                if row['scheduler_calls'] != 1:
                    raise RuntimeError('Expected exactly one joint scheduler step per denoise step')
                position = scheduler.step_index
                if position is None:
                    position = scheduler.index_for_timestep(timestep)
                row.update(sigma=_cpu(scheduler.sigmas[position]),
                           sigma_next=_cpu(scheduler.sigmas[position + 1]),
                           timestep=_cpu(torch.as_tensor(timestep)),
                           actual_sampler_velocity=_cpu(model_output),
                           state=_cpu(sample))
                result = original_step(model_output, timestep, sample, *args, **kwargs)
                # The ART adapter requests return_dict=False and subsequently casts
                # the resulting video/audio states back to their original dtype.
                if kwargs.get('return_dict') is not False:
                    raise ValueError('Expected the original ART scheduler tuple interface')
                row['scheduler_next_unrounded'] = _cpu(result[0])
                return result

            scheduler.step = observe_step
            try:
                output = self.original_denoise(index, timestep, state, forward_ctx, denoise_ctx)
            finally:
                _restore_attribute(scheduler, 'step', previous_step)
            if row['scheduler_calls'] != 1:
                raise RuntimeError('No actual scheduler prediction was captured')
            row['actual_next_rounded'] = _cpu(torch.cat(
                [output.video, output.audio[:, :audio_length]], dim=1).float())
            row['capture_and_step_wall_seconds'] = time.perf_counter() - started
            self.records.append(row)
            return output

        p._denoise_step = MethodType(denoise, p)
        p._isolated_velocity_capture_active = True
        self.active = True
        return self

    def __exit__(self, exc_type, exc, traceback):
        _restore_attribute(self.pipeline, '_denoise_step', self.previous_denoise)
        delattr(self.pipeline, '_isolated_velocity_capture_active')
        self.active = False
        return False

    def publish(self, destination):
        destination = Path(destination).resolve()
        if self.active or not self.records or not destination.is_relative_to(ROOT):
            raise ValueError('Publish a finished nonempty capture under the campaign root')
        if destination.exists():
            raise FileExistsError(destination)
        indices = [r['index'] for r in self.records]
        if len(set(indices)) != len(indices):
            raise ValueError('Capture spans repeated trajectories; use one recorder per fixed request')
        destination.parent.mkdir(parents=True, exist_ok=True)
        tensors = [v for r in self.records for v in r.values() if isinstance(v, torch.Tensor)]
        tensors += [v for v in self.condition.values() if isinstance(v, torch.Tensor)]
        payload = {'format_version': 1, 'scope': 'native sampler inputs; actor recompute not yet attached',
                   'identity': self.identity, 'condition': self.condition, 'records': self.records,
                   'stored_tensor_bytes': sum(t.numel() * t.element_size() for t in tensors)}
        temporary = destination.with_name(destination.name + '.tmp')
        torch.save(payload, temporary)
        with temporary.open('rb') as f:
            os.fsync(f.fileno())
            digest = hashlib.file_digest(f, 'sha256').hexdigest()
        temporary.replace(destination)
        metadata = {'sha256': digest, 'bytes': destination.stat().st_size,
                    'indices': indices, 'identity': self.identity,
                    'stored_tensor_bytes': payload['stored_tensor_bytes'],
                    'scope': payload['scope']}
        destination.with_suffix(destination.suffix + '.json').write_text(json.dumps(metadata, indent=2) + '\n')
        return metadata
