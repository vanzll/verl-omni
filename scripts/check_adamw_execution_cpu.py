"""Synthetic CPU numerical screen only, never a training/GPU equivalence proof."""
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from omegaconf import OmegaConf
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config.optimizer import build_optimizer

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')
assert not torch.cuda.is_initialized()
raw = OmegaConf.load(ROOT / 'preflight/configs/A0/resolved.yaml').actor_rollout_ref.actor.optim
results = []
for dtype in (torch.float32, torch.bfloat16):
    g = torch.Generator().manual_seed(48752)
    # LoRA-like matrix slices, zero-initialized B, and an unused/frozen weight.
    initial = [torch.randn((16, 128), generator=g).mul_(.02).to(dtype),
               torch.zeros((128, 16), dtype=dtype), torch.zeros(4, dtype=dtype)]
    branches = []
    for foreach in (True, False):
        params = [torch.nn.Parameter(x.clone()) for x in initial]
        params[-1].requires_grad_(False)
        cfg = omega_conf_to_dataclass(OmegaConf.merge(raw, {'override_optimizer_config': {'foreach': foreach}}))
        opt = build_optimizer(params, cfg)
        assert type(opt) is torch.optim.AdamW
        assert opt.defaults['foreach'] is foreach
        assert opt.defaults['lr'] == 3e-5 and opt.defaults['weight_decay'] == 1e-4
        assert opt.defaults['betas'] == (0.9, .999) and opt.defaults['eps'] == 1e-8
        branches.append((params, opt))
    timings = [0., 0.]
    for step in range(20):
        grads = [torch.randn(x.shape, generator=g).mul_(.0001).to(dtype) for x in initial[:-1]]
        if step in (5, 13): grads[0].zero_()
        for i, (params, opt) in enumerate(branches):
            opt.zero_grad(set_to_none=True)
            for p, grad in zip(params[:-1], grads, strict=True): p.grad = grad.clone()
            t = time.monotonic(); opt.step(); timings[i] += time.monotonic() - t
            assert params[-1] not in opt.state and torch.equal(params[-1], initial[-1])
    comparisons = []
    for i in range(2):
        p, q = branches[0][0][i], branches[1][0][i]
        for name, a, b in [('parameter', p, q)] + [
                (key, branches[0][1].state[p][key], branches[1][1].state[q][key])
                for key in ('exp_avg', 'exp_avg_sq', 'step')]:
            assert a.shape == b.shape and a.dtype == b.dtype
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            diff = a.double() - b.double()
            comparisons.append(dict(parameter_index=i, field=name, dtype=str(a.dtype),
                                    bitwise_equal=torch.equal(a, b), max_abs=diff.abs().max().item(),
                                    relative_l2=(diff.norm() / a.double().norm().clamp_min(1e-30)).item()))
    results.append(dict(dtype=str(dtype), steps=20, comparisons=comparisons,
                        cpu_optimizer_seconds={'foreach': timings[0], 'single_tensor': timings[1]}))
assert not torch.cuda.is_initialized()
report = dict(status='completed_cpu_synthetic_numerical_screen', cuda_initialized=False,
              gpu_verified=False, distributed_verified=False, config_modified=False,
              source_sha256=os.environ['SOURCE_SNAPSHOT_SHA256'], torch_version=torch.__version__,
              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              config_sha256=hashlib.sha256((ROOT / 'preflight/configs/A0/resolved.yaml').read_bytes()).hexdigest(),
              results=results, scope='Actual pinned build_optimizer and same AdamW hyperparameters, synthetic CPU matrices only. No DTensor/FSDP/GPU equivalence or speedup claim; no change to active GPU probe or scientific runs.')
(ROOT / 'logs/adamw_foreach_vs_single_cpu_numerical_screen.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
