"""Exercise actual server config conversion and ingress preparation, without GPU engines."""
import argparse
from argparse import Namespace
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')
parser = argparse.ArgumentParser()
parser.add_argument('--source-dir', type=Path, default=ROOT / 'source')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
source = args.source_dir.resolve()
assert source == ROOT / 'source' or source.is_relative_to(ROOT / 'snapshots')
assert args.output.resolve().is_relative_to(ROOT)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, str(source))

from omegaconf import OmegaConf
from verl.utils.config import omega_conf_to_dataclass
from verl_omni.workers.rollout.vllm_rollout import vllm_omni_diffusion_strategy as strategy_module
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniHttpServer

assert Path(strategy_module.__file__).resolve().is_relative_to(source)
report = {'scope': 'Actual CPU server schema conversion and engine ingress preparation; no GPU engine/generation',
          'source_dir': str(source), 'unix': time.time(),
          'strategy_sha256': hashlib.sha256(Path(strategy_module.__file__).read_bytes()).hexdigest(),
          'variants': []}
for variant in ('N0', 'A0', 'A1', 'A2'):
    config = OmegaConf.load(ROOT / 'preflight/configs' / variant / 'resolved.yaml')
    rollout = omega_conf_to_dataclass(config.actor_rollout_ref.rollout)
    server = object.__new__(vLLMOmniHttpServer)
    server.config = server._init_config(rollout)
    server.model_config = server._init_model_config(config.actor_rollout_ref.model)
    model = server.model_config
    for key in ('video_cfg_scale', 'audio_cfg_scale'):
        assert getattr(model.pipeline, key) == config.actor_rollout_ref.model.pipeline[key]
    engine = {'model': model.local_path, 'max_num_seqs': rollout.max_num_seqs,
              'tensor_parallel_size': rollout.tensor_model_parallel_size}
    engine_options = dict(rollout.engine_kwargs.get('vllm_omni', {}))
    server._generate_strategy.prepare_engine_args(engine, Namespace(**engine_options))
    assert engine['tensor_parallel_size'] == 2
    assert engine['custom_pipeline_args']['pipeline_class']
    row = {'variant': variant, 'status': 'passed',
           'model_pipeline_class': type(model.pipeline).__name__,
           'rollout_pipeline_class': type(rollout.pipeline).__name__,
           'model_video_cfg': model.pipeline.video_cfg_scale,
           'model_audio_cfg': model.pipeline.audio_cfg_scale,
           'rollout_video_cfg': rollout.pipeline.video_cfg_scale,
           'rollout_audio_cfg': rollout.pipeline.audio_cfg_scale,
           'validation_height': rollout.val_kwargs.pipeline.height,
           'validation_width': rollout.val_kwargs.pipeline.width,
           'engine_args': engine}
    report['variants'].append(row)
    print(json.dumps(row), flush=True)
report['status'] = 'passed'
args.output.write_text(json.dumps(report, indent=2, default=str) + '\n')
