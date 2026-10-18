"""Measure existing static/dynamic reward RPC batching with exact pickle roundtrips."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import pickle
import time
from types import SimpleNamespace

import numpy as np
import torch
from verl.protocol import DataProto
from verl_omni.reward_loop.replica_dispatch import dispatch_reward_groups
from verl_omni.workers.config.reward import NativeRewardModelConfig, parse_reward_model_config

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')
config_file = ROOT / 'attempts/LTX23-A0-ART-P32K8-U150-S42-A009/analysis/wandb/config.json'
config = json.loads(config_file.read_text())
models = config['reward']['models']
for name, model in models.items():
    candidate = dict(model, dispatch_batch_size=2)
    parsed = parse_reward_model_config(name, candidate)
    assert isinstance(parsed, NativeRewardModelConfig) and parsed.dispatch_batch_size == 2
    assert {k: v for k, v in candidate.items() if k != 'dispatch_batch_size'} == model

generator = torch.Generator().manual_seed(42)
data = DataProto.from_dict(tensors={
    'sample_id': torch.arange(256),
    'rendered_video': torch.randn((256, 4, 3, 24, 32), generator=generator),
    'train_latents': torch.randn((256, 3, 1024), generator=generator),
    'prompt_embeds': torch.randn((256, 128, 16), generator=generator),
}, non_tensors={'uid': np.asarray([f'prompt-{i // 8}' for i in range(256)], dtype=object)})


async def mode(dynamic):
    sizes, calls = [], {name: [] for name in models}
    def worker(name):
        async def compute(chunk):
            payload = pickle.dumps(chunk, protocol=5)
            sizes.append(len(payload))
            restored = pickle.loads(payload)
            assert set(restored.batch.keys()) == set(chunk.batch.keys())
            assert all(torch.equal(restored.batch[key], chunk.batch[key]) for key in chunk.batch.keys())
            assert np.array_equal(restored.non_tensor_batch['uid'], chunk.non_tensor_batch['uid'])
            ids = restored.batch['sample_id'].tolist()
            calls[name].append(ids)
            result = [{'id': value, 'score': float(restored.batch['rendered_video'][index].mean())}
                      for index, value in enumerate(ids)]
            await asyncio.sleep(0)
            return result
        return SimpleNamespace(compute_score_batch=SimpleNamespace(remote=compute))
    groups = {name: [worker(name) for _ in model['placement']['devices']] for name, model in models.items()}
    began = time.monotonic()
    outputs = await dispatch_reward_groups(data, groups, {name: 2 for name in models} if dynamic else {})
    for name, values in outputs.items():
        assert [value['id'] for value in values] == list(range(256))
        assert sorted(value for chunk in calls[name] for value in chunk) == list(range(256))
    return outputs, {'rpc_calls': len(sizes), 'maximum_pickle_bytes_per_rpc': max(sizes),
                     'sum_pickle_bytes': sum(sizes), 'elapsed_cpu_seconds': time.monotonic() - began,
                     'maximum_samples_per_rpc': max(len(chunk) for group in calls.values() for chunk in group)}


before, static = asyncio.run(mode(False))
after, dynamic = asyncio.run(mode(True))
assert before == after
assert dynamic['maximum_samples_per_rpc'] == 2
assert dynamic['maximum_pickle_bytes_per_rpc'] < static['maximum_pickle_bytes_per_rpc'] / 50
report = {'status': 'passed', 'unix': time.time(), 'source_sha256': os.environ['SOURCE_SNAPSHOT_SHA256'],
          'actual_model_config_sha256': hashlib.sha256(config_file.read_bytes()).hexdigest(),
          'all_five_native_configs_accept_dispatch_batch_size_2': True,
          'all_tensor_and_uid_fields_preserved_exactly_after_pickle': True,
          'per_sample_fake_scores_and_order_exactly_equal': True,
          'static': static, 'dynamic2': dynamic,
          'maximum_rpc_size_reduction_factor': static['maximum_pickle_bytes_per_rpc'] / dynamic['maximum_pickle_bytes_per_rpc'],
          'scope': 'CPU real DataProto/dispatch/native-config/pickle contract with scaled synthetic tensors; not GPU scorer equivalence or measured production throughput.'}
(ROOT / 'logs/reward_dispatch_payload_cpu.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2), flush=True)
