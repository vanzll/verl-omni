#!/usr/bin/env python3
"""Audit the installed candidate versions, native libraries, and actual package imports."""
import ctypes
import importlib
import importlib.metadata
import json
import os
import time
import tomllib
import traceback
import warnings
from pathlib import Path

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')
lock = tomllib.loads((ROOT / 'assets/pylock.runtime-cu129.toml').read_text())
report = dict(utc=time.strftime('%FT%TZ', time.gmtime()), versions={}, imports={}, errors=[])
for package in lock['packages']:
    name = package['name']
    try:
        actual = importlib.metadata.version(name)
        report['versions'][name] = actual
        if package.get('version') and actual != package['version']:
            report['errors'].append(f"Version mismatch {name}: {actual} != {package['version']}")
    except importlib.metadata.PackageNotFoundError:
        report['errors'].append('Missing installed distribution: ' + name)
modules = ('torch', 'torchaudio', 'torchvision', 'torchcodec', 'diffusers', 'transformers',
           'ray', 'tensordict', 'hydra', 'peft', 'kernels', 'cupy', 'verl',
           'vllm._C_stable_libtorch', 'vllm_omni', 'verl_omni', 'cv2', 'av', 'torchcodec.decoders',
           'verl_omni.utils.reward_score.videoalign',
           'verl_omni.utils.reward_score.hpsv3_reward',
           'verl_omni.utils.reward_score.audiobox',
           'verl_omni.utils.reward_score.clap', 'verl_omni.utils.reward_score.desync')
for name in modules:
    start = time.monotonic()
    try:
        module = importlib.import_module(name)
        report['imports'][name] = dict(status='ok', path=getattr(module, '__file__', None),
                                       elapsed=time.monotonic() - start)
        print('IMPORT OK', name, flush=True)
    except Exception as error:
        report['imports'][name] = dict(status='failed', error=type(error).__name__,
                                       detail=str(error), traceback=traceback.format_exc())
        report['errors'].append('Import failed: ' + name)
        print('IMPORT FAILED', name, type(error).__name__, str(error)[:350], flush=True)
if report['imports']['torch']['status'] == 'ok':
    import torch
    assert torch.version.cuda == '12.9', torch.version.cuda
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter('always')
        x = torch.tensor([1.0], requires_grad=True)
        x.square().sum().backward()
    report['cpu_backward'] = x.grad.item()
    report['cpu_backward_warnings'] = [str(item.message) for item in observed]
    if any('CUDA initialization' in str(item.message) for item in observed):
        report['errors'].append('CUDA initialization warning during CPU autograd')
    library = ctypes.CDLL('libcuda.so.1')
    version = ctypes.c_int()
    report['driver_version_api_status'] = library.cuDriverGetVersion(ctypes.byref(version))
    report['driver_version_api'] = version.value
    paths = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                    if any(name in line for name in ('libcuda', 'libcudart', 'libcudnn', 'libtorch', 'libcublas'))})
    report['loaded_native_libraries'] = paths
    if any('/cu13/' in path or 'libcudart.so.13' in path for path in paths):
        report['errors'].append('CUDA13 native library loaded in the CUDA12 candidate')
report['status'] = 'failed' if report['errors'] else 'passed'
(ROOT / 'logs/runtime_imports_cu129.json').write_text(json.dumps(report, indent=2) + '\n')
print('Runtime import audit:', report['status'], report['errors'], flush=True)
print('GPU kernels remain pending the guarded GPU pilot', flush=True)
raise SystemExit(bool(report['errors']))
