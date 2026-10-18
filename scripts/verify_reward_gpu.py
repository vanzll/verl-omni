"""Actual five-scorer preflight on fixed generated AV; never creates a trainer."""
import argparse
import asyncio
import dataclasses
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from omegaconf import OmegaConf
import torch
from verl import DataProto
from verl_omni.pipelines.rollout_media import MediaArtifact, MediaSpec, PREVIEW_ARTIFACT
from verl_omni.reward_loop.reward_model_executor import NativeRewardExecutor
from verl_omni.workers.config.reward import RewardModelSpec

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')
NAMES = ('video_align', 'hpsv3', 'audiobox', 'clap', 'desync')
parser = argparse.ArgumentParser()
parser.add_argument('--cpu-config-only', action='store_true')
parser.add_argument('--worker', choices=NAMES)
parser.add_argument('--input', type=Path, required=True)
args = parser.parse_args()
OUTPUT = Path(os.environ['OUTPUT_DIR']).resolve()
assert OUTPUT.is_relative_to(ROOT / 'preflight/runtime_gpu')
assert args.input.resolve().is_relative_to(ROOT / 'preflight/runtime_gpu')


def publish(name, value):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    p = OUTPUT / name
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str) + '\n')
    tmp.replace(p)


def fixed_inputs():
    state = json.loads((args.input.parent / 'gpu_generation_state.json').read_text())
    entry = next(x for x in state['requests'] if Path(x['artifact']).resolve() == args.input.resolve())
    with args.input.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == entry['sha256']
    data = torch.load(args.input, map_location='cpu', weights_only=True)
    video, audio = data['artifacts']['video_preview'], data['artifacts']['audio']
    assert list(video.shape) == [121, 3, entry['height'], entry['width']] and video.dtype == torch.uint8
    assert audio.ndim == 2 and audio.shape[0] == 2 and torch.isfinite(audio).all()
    config = OmegaConf.load(ROOT / 'preflight/configs/A0/resolved.yaml')
    vocoder = Path(config.actor_rollout_ref.model.path) / 'vocoder/config.json'
    sample_rate = int(json.loads(vocoder.read_text())['output_sampling_rate'])
    assert sample_rate == 48000
    fps = float(config.actor_rollout_ref.rollout.pipeline.frame_rate)
    assert fps == 24.0
    artifacts = {
        'video_preview': MediaArtifact(MediaSpec('video', 'decoded', 'TCHW', fps=fps), video),
        'audio': MediaArtifact(MediaSpec('audio', 'decoded', 'CT', sample_rate=sample_rate), audio),
    }
    for key, value in artifacts.items():
        value.validate(context='actual reward GPU probe', name=key)
    prompt = state['fixed_prompt']
    assert hashlib.sha256(prompt.encode()).hexdigest() == state['prompt_sha256']
    extra = {'media_artifacts': artifacts, PREVIEW_ARTIFACT: 'video_preview',
             'media_kind': 'video', 'fps': fps, 'audio_sample_rate': sample_rate, 'audio_layout': 'CT'}
    batch = DataProto.from_dict(tensors={'responses': video.unsqueeze(0), 'audio': audio.unsqueeze(0),
                                        'audio_sample_rate': torch.tensor([sample_rate])},
                               non_tensors={'reward_inputs': np.array([{'text': {'video': prompt, 'audio': prompt}}], dtype=object)})
    return config, video, audio, prompt, extra, batch, entry['sha256']


def contracts(config, name, device):
    model = OmegaConf.to_container(config.reward.models[name], resolve=True)
    term = OmegaConf.to_container(config.reward.reward_functions[name], resolve=True)
    class_module, class_name = model['executor']['model'].split(':')
    cls = getattr(importlib.import_module(class_module), class_name)
    kwargs = dict(model['executor'].get('kwargs') or {})
    kwargs.update(model_path=model['model_path'], device=device)
    inspect.signature(cls).bind(**kwargs)
    module = importlib.import_module(term['path'].removeprefix('pkg://'))
    score_fn = getattr(module, term['name'])
    term_kwargs = {k: v for k, v in term.items()
                   if k not in ('path', 'name', 'model', 'weight', 'required', 'routing_weights')}
    spec = RewardModelSpec(name=name, backend=model['backend'], model_path=model['model_path'],
                           executor_config=model['executor'])
    return spec, score_fn, term_kwargs, cls


def numeric_summary(value):
    if isinstance(value, torch.Tensor):
        x = value.detach().float().cpu()
        if not torch.isfinite(x).all():
            raise ValueError('Nonfinite native scorer output')
        return {'shape': list(x.shape), 'dtype': str(value.dtype),
                'values': x.tolist() if x.numel() <= 128 else None,
                'mean': float(x.mean()) if x.numel() else None}
    if isinstance(value, dict):
        return {str(k): numeric_summary(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [numeric_summary(v) for v in value]
    if isinstance(value, (float, int)):
        if not math.isfinite(value):
            raise ValueError('Nonfinite native scorer scalar')
        return value
    return str(value)


async def worker():
    config, video, audio, prompt, extra, batch, digest = fixed_inputs()
    name = args.worker
    assert torch.cuda.device_count() == 1
    spec, score_fn, kwargs, cls = contracts(config, name, torch.device('cuda', 0))
    executor = NativeRewardExecutor(spec)
    report = {'model': name, 'stage': 'constructing', 'input_sha256': digest,
              'source_sha256': os.environ['SOURCE_SNAPSHOT_SHA256'], 'raw_inference_outputs': [],
              'class': spec.executor_config['model'], 'started_unix': time.time()}
    publish(name + '.json', report)
    original_infer = executor.infer

    async def capture(*a, **kw):
        result = await original_infer(*a, **kw)
        report['raw_inference_outputs'].append(numeric_summary(result))
        return result

    executor.infer = capture
    try:
        torch.cuda.reset_peak_memory_stats()
        began = time.monotonic()
        await executor.wake_up()
        torch.cuda.synchronize()
        report.update(stage='scoring', construct_seconds=time.monotonic() - began)
        publish(name + '.json', report)
        began = time.monotonic()
        result = await score_fn(data_source='isolated_native_gpu_probe', solution_image=video,
                                ground_truth=prompt, extra_info=extra, reward_model=executor,
                                batch=batch, **kwargs)
        torch.cuda.synchronize()
        score = float(result['score'])
        assert math.isfinite(score)
        report.update(stage='score_passed', result=numeric_summary(result),
                      score_seconds=time.monotonic() - began,
                      max_memory_allocated_bytes=torch.cuda.max_memory_allocated())
        await executor.sleep()
        await executor.close()
        report.update(stage='passed_and_closed', finished_unix=time.time())
        publish(name + '.json', report)
        print(json.dumps({'model': name, 'stage': report['stage'], 'result': report['result']}), flush=True)
    except BaseException as error:
        report.update(stage='failed', error_type=type(error).__name__, error_message=str(error),
                      finished_unix=time.time())
        publish(name + '.json', report)
        raise
    finally:
        await executor.close()


async def main():
    config, video, audio, prompt, extra, batch, digest = fixed_inputs()
    if args.cpu_config_only:
        rows = []
        for name in NAMES:
            spec, fn, kw, cls = contracts(config, name, torch.device('cpu'))
            inspect.signature(fn).bind(data_source='isolated_native_gpu_probe', solution_image=video,
                                       ground_truth=prompt, extra_info=extra,
                                       reward_model=NativeRewardExecutor(spec), batch=batch, **kw)
            rows.append({'name': name, 'model_class': spec.executor_config['model'],
                         'model_path': spec.model_path, 'scorer_kwargs': kw})
        publish('reward_gpu_state.json', {'stage': 'cpu_contract_passed', 'input_sha256': digest,
                                         'scope': 'real input/hash/media/batch/native model and scorer signatures; no GPU construction',
                                         'models': rows})
        print('CPU_REWARD_CONTRACT_PASSED', json.dumps([x['name'] for x in rows]), flush=True)
        return
    if args.worker:
        await worker()
        return
    state = {'stage': 'workers_running', 'input_sha256': digest, 'started_unix': time.time(),
             'scope': 'single actual generated sample, five native GPU scorer interfaces; not a scientific training/evaluation run',
             'allocated_gpus': 8, 'active_scorer_gpus': 5, 'models': {}}
    publish('reward_gpu_state.json', state)
    processes = []
    for index, name in enumerate(NAMES):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(index)
        logfile = (OUTPUT / (name + '.log')).open('w')
        process = subprocess.Popen(['timeout', '--signal=TERM', '--kill-after=15s', '900s', sys.executable,
                                    str(Path(__file__).resolve()), '--worker', name, '--input', str(args.input)],
                                   env=env, stdout=logfile, stderr=subprocess.STDOUT)
        logfile.close()
        processes.append((name, process))
    while any(p.poll() is None for _, p in processes):
        for name, process in processes:
            state['models'][name] = {'pid': process.pid, 'exit_code': process.poll()}
        publish('reward_gpu_state.json', state)
        await asyncio.sleep(2)
    for name, process in processes:
        result_path = OUTPUT / (name + '.json')
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        state['models'][name] = {'exit_code': process.returncode, 'result': result}
    passed = all(x['exit_code'] == 0 and x['result'].get('stage') == 'passed_and_closed'
                 for x in state['models'].values())
    state.update(stage='passed' if passed else 'failed', finished_unix=time.time())
    if passed:
        scores = {n: float(x['result']['result']['score']) for n, x in state['models'].items()}
        state['once_counted_weighted_total'] = (scores['video_align'] + 1.5 * scores['hpsv3']
                                              + 0.5 * scores['audiobox'] + scores['clap'] + scores['desync'])
    publish('reward_gpu_state.json', state)
    print('REWARD_GPU_STAGE', state['stage'], flush=True)
    if not passed:
        raise RuntimeError('One or more actual native GPU scorers failed; preserve individual logs')


if __name__ == '__main__':
    asyncio.run(main())
