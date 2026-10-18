"""Four CPU ranks exercise the pinned native HSDP LoRA checkpoint manager.

This small model verifies serialization and state restoration, not LTX GPU
checkpoint correctness or controller/data-cursor restoration.
"""
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from transformers import PretrainedConfig
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
import verl.utils.checkpoint.fsdp_checkpoint_manager as checkpoint_module

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')
OUTPUT = ROOT / 'preflight/runtime_cpu/hsdp_native_checkpoint_20261017'
assert os.environ['CUDA_VISIBLE_DEVICES'] == '' and not torch.cuda.is_initialized()
dist.init_process_group('gloo', timeout=timedelta(seconds=60))
rank, world = dist.get_rank(), dist.get_world_size()
assert world == 4
OUTPUT.mkdir(parents=True, exist_ok=True)


class Adapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Parameter(torch.randn(8, 8), requires_grad=False)
        self.lora_A = torch.nn.ParameterDict({
            'default': torch.nn.Parameter(torch.randn(2, 8) * .03),
            'old': torch.nn.Parameter(torch.randn(2, 8) * .03, requires_grad=False)})
        self.lora_B = torch.nn.ParameterDict({
            'default': torch.nn.Parameter(torch.zeros(8, 2)),
            'old': torch.nn.Parameter(torch.randn(8, 2) * .03, requires_grad=False)})
        self.config = PretrainedConfig()
        self.peft_config = {'default': type('Meta', (), {'r': 2, 'lora_alpha': 4, 'task_type': None})()}

    def can_generate(self):
        return False

    def forward(self, x):
        return x @ self.base.T + (x @ self.lora_A['default'].T) @ self.lora_B['default'].T


def local(value):
    return value.to_local() if hasattr(value, 'to_local') else value


def copied(value):
    if isinstance(value, torch.Tensor):
        return local(value).detach().clone()
    if isinstance(value, dict):
        return {k: copied(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(copied(v) for v in value)
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(local(a), local(b))
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, type(a)) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


torch.manual_seed(916)
model = Adapter()
mesh = init_device_mesh('cpu', (world, 1), mesh_dim_names=('ddp', 'fsdp'))
fully_shard(model, mesh=mesh, reshard_after_forward=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4, foreach=False)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1. / (step + 1))
for step in range(2):
    optimizer.zero_grad(set_to_none=True)
    x = torch.arange(16, dtype=torch.float32).reshape(2, 8) / (10 + rank + step)
    (model(x) - .3).square().mean().backward()
    optimizer.step()
    scheduler.step()

manager = FSDPCheckpointManager(model, optimizer, scheduler,
    checkpoint_config={'save_lora_only': True, 'save_contents': ['model', 'optimizer', 'extra'],
                       'load_contents': ['model', 'optimizer', 'extra']})
directory = OUTPUT / 'native_checkpoint'
assert not directory.exists() or not any(directory.iterdir()), 'Do not overwrite an earlier checkpoint probe'
dist.barrier()
torch.manual_seed(428 + rank)
np.random.seed(428 + rank)
random.seed(428 + rank)
before_model = copied(model.state_dict())
before_optimizer = copied(optimizer.state_dict())
before_scheduler = copied(scheduler.state_dict())
before_rng = copied(manager.get_rng_state())
manager.save_checkpoint(str(directory), global_step=2, max_ckpt_to_keep=None)
assert equal(before_model, model.state_dict())
assert equal(before_optimizer, optimizer.state_dict())
assert equal(before_rng, manager.get_rng_state())

model_file = directory / f'model_world_size_{world}_rank_{rank}.pt'
saved = torch.load(model_file, weights_only=False)
assert set(saved) == {key for key in before_model if 'lora_' in key}
assert all(local(v).numel() == v.numel() for v in saved.values()), 'HSDP shard1 should contain full local adapter tensors'
saved_tensor_bytes = sum(local(v).numel() * local(v).element_size() for v in saved.values())
with torch.no_grad():
    for name, parameter in model.named_parameters():
        if 'lora_' in name:
            parameter.add_(.125)
    for state in optimizer.state.values():
        for value in state.values():
            value.add_(1)
assert not equal(before_model, model.state_dict())
assert not equal(before_optimizer, optimizer.state_dict())
scheduler.step()
torch.rand(13)
np.random.rand(13)
random.random()
manager.load_checkpoint(str(directory), del_local_after_load=False)
checks = dict(all_named_current_old_and_base_restored=equal(before_model, model.state_dict()),
              adam_moments_and_step_restored=equal(before_optimizer, optimizer.state_dict()),
              scheduler_and_lr_restored=equal(before_scheduler, scheduler.state_dict()),
              cpu_python_numpy_rng_restored=equal(before_rng, manager.get_rng_state()),
              save_did_not_initialize_cuda=not torch.cuda.is_initialized())
assert all(checks.values()), checks
files = {p.name: p.stat().st_size for p in directory.iterdir() if p.is_file() and f'_rank_{rank}.' in p.name}
report = dict(status='passed_four_rank_CPU_native_HSDP_LoRA_checkpoint_roundtrip', rank=rank,
              world_size=world, mesh_shape=list(mesh.shape), checks=checks,
              saved_adapter_local_tensor_bytes=saved_tensor_bytes, files=files,
              model_keys=sorted(saved), native_manager_sha256=hashlib.sha256(Path(checkpoint_module.__file__).read_bytes()).hexdigest(),
              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              torch_version=torch.__version__, observed_unix=time.time(),
              gpu_verified=False, controller_cursor_verified=False, scientific_optimizer_updates=0,
              scope='Actual CPU/Gloo HSDP(4,1) tiny model uses pinned native checkpoint manager; per-rank complete local adapter serialization, Adam/scheduler and CPU RNG restoration. No LTX/BF16/CUDA/world16 or formal checkpoint claim.')
(OUTPUT / f'rank_{rank}.json').write_text(json.dumps(report, indent=2) + '\n')
dist.barrier()
dist.destroy_process_group()
print(json.dumps({'rank': rank, 'status': report['status']}), flush=True)
