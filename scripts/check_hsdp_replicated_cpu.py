"""Actual four-rank CPU/Gloo HSDP vs global-batch reference; no GPU claim."""
from copy import deepcopy
from unittest.mock import patch
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from verl.utils.fsdp_utils import fsdp2_clip_grad_norm_
from verl.utils import fsdp_utils

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')
OUTPUT = ROOT / 'preflight/runtime_gpu/fsdp_cycle_cpu_contract_v9_replicated/hsdp_gloo_oracle'
OUTPUT.mkdir(exist_ok=True)
assert not torch.cuda.is_initialized() and os.environ['CUDA_VISIBLE_DEVICES'] == ''
dist.init_process_group('gloo', timeout=timedelta(seconds=90))
rank, world = dist.get_rank(), dist.get_world_size()
assert world == 4

class ToyAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Parameter(torch.randn(4, 4) * .1, requires_grad=False)
        self.a = torch.nn.Parameter(torch.randn(2, 4) * .03)
        self.b = torch.nn.Parameter(torch.zeros(4, 2))
        self.old_a = torch.nn.Parameter(self.a.detach().clone(), requires_grad=False)
        self.old_b = torch.nn.Parameter(self.b.detach().clone(), requires_grad=False)
    def forward(self, x, old=False):
        a, b = (self.old_a, self.old_b) if old else (self.a, self.b)
        return x @ self.base.T + (x @ a.T) @ b.T

def full(value):
    return value.full_tensor() if hasattr(value, 'full_tensor') else value

torch.manual_seed(84931)
model = ToyAdapter(); reference = deepcopy(model)
mesh = init_device_mesh('cpu', (world, 1), mesh_dim_names=('ddp', 'fsdp'))
fully_shard(model, mesh=mesh, reshard_after_forward=True)
kwargs = dict(lr=3e-5, betas=(.9, .999), eps=1e-8, weight_decay=1e-4, foreach=False)
optimizer = torch.optim.AdamW(model.parameters(), **kwargs)
reference_optimizer = torch.optim.AdamW(reference.parameters(), **kwargs)
generator = torch.Generator().manual_seed(4827)
results = []
for step in range(3):
    x = torch.randn(world * 2, 4, generator=generator)
    guidance = torch.randn(world * 2, 4, generator=generator) * .1
    local = slice(rank * 2, (rank + 1) * 2)
    optimizer.zero_grad(set_to_none=True); reference_optimizer.zero_grad(set_to_none=True)
    with torch.no_grad(): target = model(x[local], old=True) + guidance[local]
    ((model(x[local]) - target).square().mean()).backward()
    with torch.no_grad(): reference_target = reference(x, old=True) + guidance
    ((reference(x) - reference_target).square().mean()).backward()
    gradient_error = 0.
    ref_params = dict(reference.named_parameters())
    for name, param in model.named_parameters():
        if param.grad is not None:
            actual, expected = full(param.grad), ref_params[name].grad
            error = (actual - expected).abs().max().item(); gradient_error = max(gradient_error, error)
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-8)
    # CPU test device routing only; use the unchanged native norm/clip math.
    with patch.object(fsdp_utils, 'get_device_id', return_value='cpu'):
        actual_norm = full(fsdp2_clip_grad_norm_(model.parameters(), max_norm=1e-4))
    expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), max_norm=1e-4)
    torch.testing.assert_close(actual_norm, expected_norm, rtol=2e-5, atol=2e-8)
    optimizer.step(); reference_optimizer.step()
    parameter_error = state_error = 0.
    for name, param in model.named_parameters():
        expected = ref_params[name]
        actual_full = full(param.detach())
        parameter_error = max(parameter_error, (actual_full - expected).abs().max().item())
        torch.testing.assert_close(actual_full, expected, rtol=2e-5, atol=2e-8)
        if param in optimizer.state:
            for key in ('exp_avg', 'exp_avg_sq', 'step'):
                actual_value = full(optimizer.state[param][key])
                expected_value = reference_optimizer.state[expected][key]
                state_error = max(state_error, (actual_value - expected_value).abs().max().item())
                torch.testing.assert_close(actual_value, expected_value, rtol=2e-5, atol=2e-8)
    results.append(dict(step=step+1, gradient_max_abs_error=gradient_error,
                        native_clipping_norm=actual_norm.item(), reference_norm=expected_norm.item(),
                        parameter_max_abs_error=parameter_error, adam_state_max_abs_error=state_error))
assert not torch.cuda.is_initialized()
report = dict(status='passed_actual_cpu_gloo_hsdp_global_gradient_clip_adam_oracle', rank=rank,
              world_size=world, mesh_shape=list(mesh.shape), shard_world_size=dist.get_world_size(mesh.get_group('fsdp')),
              replicate_world_size=dist.get_world_size(mesh.get_group('ddp')), steps=results,
              cuda_initialized=False, gpu_verified=False, dtype='float32',
              cpu_only_native_clip_device_route_override='cpu',
              torch_version=torch.__version__, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              scope='Real CPU/Gloo FSDP2 HSDP(4,1), rank-distinct inputs, frozen old adapter/no-grad forward, global-mean gradients, native clip helper with CPU-only device routing override, AdamW parameters/moments vs unsharded global-batch reference. Not LTX/BF16/CUDA/world16 production equivalence.')
(OUTPUT / f'rank_{rank}.json').write_text(json.dumps(report, indent=2)+'\n')
dist.barrier(); dist.destroy_process_group()
print(json.dumps({'rank':rank,'status':report['status']}), flush=True)
