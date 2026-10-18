"""Isolated TP2 full-model generation preflight and actual velocity capture.

No trainer, optimizer update, W&B scientific curve, or FSDP actor is created.
All allocation/cleanup is owned by the existing generation-fenced node owner.
"""
import asyncio
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from verl.utils.config import omega_conf_to_dataclass
from vllm_omni.entrypoints import AsyncOmni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.config.omni_config import normalize_and_validate_diffusion_engine_ingress_kwargs

from verl_omni.pipelines.rollout_request import OmniRolloutRequest
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniHttpServer
from verl_omni.utils.net_utils import get_non_ephemeral_free_port

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')
OUTPUT = Path(os.environ['OUTPUT_DIR']).resolve()
assert OUTPUT.is_relative_to(ROOT / 'preflight/runtime_gpu')
assert os.environ['OMNI_ISOLATED_VELOCITY_PROBE'] == '1'
assert os.environ['VLLM_OMNI_DLO_DP_WAVE_TIMEOUT'] == '3600'
parser = argparse.ArgumentParser()
parser.add_argument('--cpu-config-only', action='store_true')
ARGS = parser.parse_args()


def publish(value):
    path = OUTPUT / 'gpu_generation_state.json'
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str) + '\n')
    tmp.replace(path)


def tensor_summary(value):
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError('Nonfinite actual GPU generation output')
        return {'shape': list(value.shape), 'dtype': str(value.dtype), 'numel': value.numel()}
    return None


async def main():
    config = OmegaConf.load(ROOT / 'preflight/configs/A0/resolved.yaml')
    rollout = omega_conf_to_dataclass(config.actor_rollout_ref.rollout)
    server = object.__new__(vLLMOmniHttpServer)
    server.config = server._init_config(rollout)
    server.model_config = server._init_model_config(config.actor_rollout_ref.model)
    server.global_steps = 0
    preflight = json.loads((ROOT / 'logs/four_variant_rollout_config_cpu.json').read_text())
    engine_args = dict(next(row['engine_args'] for row in preflight['variants'] if row['variant'] == 'A0'))
    assert engine_args['tensor_parallel_size'] == 2
    # The production strategy filters the LLM LoRA CLI flags before ingress;
    # native diffusion handles adapter updates through its own extension API.
    engine_args.update(log_stats=True, seed=42,
                       custom_pipeline_args={'pipeline_class': 'ltx_probe_pipeline.LTXProbePipeline'})
    if rollout.rollout_attn_backend is not None:
        engine_args['diffusion_attention_config'] = rollout.to_vllm_omni_attention_config()
    normalize_and_validate_diffusion_engine_ingress_kwargs(engine_args, stage_id=0)
    os.environ['OMNI_PROBE_ATTENTION_BACKEND'] = str(rollout.rollout_attn_backend)
    requests = [
        ('train_seed42', config.actor_rollout_ref.rollout.pipeline, 42, True),
        ('train_seed43', config.actor_rollout_ref.rollout.pipeline, 43, True),
        ('validation_seed42', OmegaConf.merge(config.actor_rollout_ref.rollout.pipeline,
                                             config.actor_rollout_ref.rollout.val_kwargs.pipeline), 42, False),
    ]

    def prepare(pipeline, seed, capture):
        params = OmegaConf.to_container(pipeline, resolve=True)
        params.pop('_target_', None)
        params['seed'] = seed
        params['extra_args'] = dict(params.get('extra_args') or {})
        if capture:
            params['extra_args']['_isolated_velocity_capture'] = True
        return params

    # Exercise all prompt/tokenizer setup in CPU preflight as well, before
    # constructing an expensive engine. This diagnostic has fixed raw text;
    # it is not a held-out validation score or a training-data substitution.
    tokenizer = AutoTokenizer.from_pretrained(Path(server.model_config.local_path) / 'tokenizer', local_files_only=True)
    prompt = 'A dog barks repeatedly while running across a grassy field.'
    negative = ''
    request = OmniRolloutRequest.from_generate_kwargs(
        prompt_ids=tokenizer(prompt, add_special_tokens=True)['input_ids'],
        negative_prompt_ids=tokenizer(negative, add_special_tokens=True)['input_ids'])
    if ARGS.cpu_config_only:
        from ltx_probe_pipeline import LTXProbePipeline
        assert LTXProbePipeline.__module__ == 'ltx_probe_pipeline'
        server.engine = SimpleNamespace(default_sampling_params_list=[OmniDiffusionSamplingParams()],
                                        engine=SimpleNamespace(get_stage_metadata=lambda _: SimpleNamespace(stage_type='diffusion')))
        rows = []
        for name, pipeline, seed, capture in requests:
            params = prepare(pipeline, seed, capture)
            payload, native = server._generate_strategy.preprocess_input(request, params, None)
            assert native[-1].seed == seed
            assert native[-1].height == pipeline.height and native[-1].width == pipeline.width
            assert native[-1].num_inference_steps == pipeline.num_inference_steps
            assert bool(native[-1].extra_args.get('_isolated_velocity_capture')) == capture
            rows.append({'name': name, 'height': pipeline.height, 'width': pipeline.width,
                         'num_frames': pipeline.num_frames, 'num_inference_steps': pipeline.num_inference_steps,
                         'seed': seed, 'capture': capture, 'video_cfg': params['video_cfg_scale'],
                         'audio_cfg': params['audio_cfg_scale']})
        assert [(x['height'], x['width'], x['num_inference_steps']) for x in rows] == [(256,384,20),(256,384,20),(512,768,40)]
        OUTPUT.mkdir(parents=True, exist_ok=True)
        publish({'stage': 'cpu_config_passed', 'scope': 'actual native request preparation; no GPU engine',
                 'requests': rows, 'engine_args': engine_args})
        print(json.dumps({'stage': 'cpu_config_passed', 'requests': rows}), flush=True)
        return
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(get_non_ephemeral_free_port('127.0.0.1'))
    state = {'scope': 'isolated TP2 native full-model generation, base weights without a loaded adapter',
             'source_sha256': os.environ['SOURCE_SNAPSHOT_SHA256'],
             'allocated_gpus': 8, 'active_generation_tp': 2, 'scientific_optimizer_updates': 0,
             'rpc_timeout_seconds': 3600, 'started_unix': time.time(), 'stage': 'engine_initializing',
             'requests': [], 'attention_backend': str(rollout.rollout_attn_backend)}
    publish(state)
    print('PROBE_STAGE engine_initializing', flush=True)
    server.engine = AsyncOmni(**engine_args)
    print('PROBE_STAGE engine_ready', flush=True)
    state['fixed_prompt'] = prompt
    state['prompt_sha256'] = hashlib.sha256(prompt.encode()).hexdigest()
    state['negative_prompt_sha256'] = hashlib.sha256(negative.encode()).hexdigest()
    state['stage'] = 'engine_ready'
    publish(state)
    try:
        for name, pipeline, seed, capture in requests:
            params = prepare(pipeline, seed, capture)
            state['stage'] = name
            publish(state)
            print('PROBE_STAGE', name, 'started', flush=True)
            began = time.monotonic()
            prompt_payload, sampling = server._generate_strategy.preprocess_input(request, params, None)
            final = await server._generate_strategy.run_generation(prompt_payload, sampling, name, None, 0)
            parsed = server._generate_strategy.process_output(final, sampling, params)
            tensors = {key: getattr(parsed, key) for key in type(parsed).model_fields
                       if isinstance(getattr(parsed, key), torch.Tensor)}
            summary = {key: tensor_summary(value) for key, value in tensors.items()}
            artifacts = parsed.artifacts or {}
            artifact_tensors = {}
            for key, artifact in artifacts.items():
                data = artifact.data
                if isinstance(data, torch.Tensor):
                    summary['artifact/' + key] = tensor_summary(data)
                    artifact_tensors[key] = data.detach().cpu()
            if not summary:
                raise ValueError('Actual generation returned no tensor artifacts')
            payload_path = OUTPUT / (name + '_output.pt')
            torch.save({'fields': {k: v.detach().cpu() for k, v in tensors.items()},
                        'artifacts': artifact_tensors}, payload_path)
            with payload_path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            entry = {'name': name, 'seed': seed, 'seconds': time.monotonic() - began,
                     'height': pipeline.height, 'width': pipeline.width,
                     'num_frames': pipeline.num_frames, 'num_inference_steps': pipeline.num_inference_steps,
                     'tensors': summary, 'artifact': str(payload_path), 'sha256': digest}
            state['requests'].append(entry)
            publish(state)
            print('PROBE_REQUEST_FINISHED', json.dumps(entry), flush=True)
        state.update(stage='passed_generation', finished_unix=time.time())
        publish(state)
        print('PROBE_STAGE passed_generation', flush=True)
    finally:
        server.engine.shutdown(timeout=30)


if __name__ == '__main__':
    asyncio.run(main())
