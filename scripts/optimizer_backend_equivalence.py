"""Replay one native AdamW step on local tensors; no model/state/file mutation.

Capture must run after gradient clipping and before the native step. Compare
runs after that native step. The reference uses the same pointwise optimizer
state on the same device; distributed gradient reduction is outside its scope.
"""
import hashlib
import json
from pathlib import Path
import time
import torch

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')

def local(value):
    return value.to_local() if hasattr(value, 'to_local') else value

def capture(optimizer, reference_foreach=True):
    assert type(optimizer) is torch.optim.AdamW
    assert reference_foreach is True
    groups, mapping = [], []
    for group_index, group in enumerate(optimizer.param_groups):
        assert group['foreach'] is False and group['fused'] is None
        assert not group['amsgrad'] and not group['capturable'] and not group['differentiable']
        copied = {key: value for key, value in group.items() if key != 'params'}
        copied.update(params=[], foreach=True)
        for parameter_index, parameter in enumerate(group['params']):
            if parameter.grad is None: continue
            grad = parameter.grad
            if hasattr(parameter, 'placements'):
                assert parameter.placements == grad.placements
                assert parameter.device_mesh == grad.device_mesh
            source, gradient = local(parameter).detach(), local(grad).detach()
            assert source.device == gradient.device and source.shape == gradient.shape
            candidate = torch.nn.Parameter(source.clone())
            candidate.grad = gradient.clone()
            copied['params'].append(candidate)
            state = optimizer.state.get(parameter, {})
            assert set(state) <= {'step', 'exp_avg', 'exp_avg_sq'}
            state_copy = {key: local(value).detach().clone() for key, value in state.items()}
            mapping.append((group_index, parameter_index, parameter, candidate, source.clone(), state_copy))
        groups.append(copied)
    assert mapping
    reference = torch.optim.AdamW(groups)
    for _, _, _, candidate, _, state in mapping:
        if state: reference.state[candidate] = state
    return reference, mapping

def compare_after(optimizer, bundle):
    reference, mapping = bundle
    reference.step()
    fields = ('parameter', 'parameter_update', 'exp_avg', 'exp_avg_sq', 'step')
    totals = {key: dict(bitwise_equal=True, max_abs=0., native_sum_squares=0., difference_sum_squares=0., elements=0) for key in fields}
    devices, dtypes = set(), set()
    for group_index, parameter_index, native, candidate, before, _ in mapping:
        actual, expected = local(native).detach(), candidate.detach()
        devices.add(actual.device.type); dtypes.add(str(actual.dtype))
        pairs = [('parameter', actual, expected),
                 ('parameter_update', actual.float()-before.float(), expected.float()-before.float())]
        for key in ('exp_avg', 'exp_avg_sq', 'step'):
            pairs.append((key, local(optimizer.state[native][key]).detach(), local(reference.state[candidate][key]).detach()))
        for key, a, b in pairs:
            assert a.shape == b.shape and a.dtype == b.dtype and a.device == b.device
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            row = totals[key]; row['elements'] += a.numel()
            same = torch.equal(a, b); row['bitwise_equal'] &= same
            # Exact comparisons avoid expensive norm work unless a difference exists.
            if not same:
                delta = a.double() - b.double()
                row['max_abs'] = max(row['max_abs'], delta.abs().max().item())
                row['difference_sum_squares'] += delta.square().sum().item()
            row['native_sum_squares'] += a.double().square().sum().item()
    for row in totals.values():
        row['relative_l2'] = (row['difference_sum_squares'] / max(row['native_sum_squares'], 1e-30)) ** .5
    return dict(status='completed_same_input_optimizer_backend_replay', parameter_objects=len(mapping),
                actual_foreach=False, reference_foreach=True, devices=sorted(devices), parameter_dtypes=sorted(dtypes),
                comparisons=totals, all_fields_bitwise_equal=all(v['bitwise_equal'] for v in totals.values()),
                scope='Pointwise native AdamW vs local-tensor foreach replay from exact post-clip inputs/moments. Does not verify distributed gradient averaging or change native model/state.')

def cpu_contract():
    assert not torch.cuda.is_initialized()
    reports=[]
    for dtype in (torch.float32, torch.bfloat16):
        generator=torch.Generator().manual_seed(2947)
        params=[torch.nn.Parameter(torch.randn(8,8,generator=generator).to(dtype)),
                torch.nn.Parameter(torch.zeros(8,8,dtype=dtype)),
                torch.nn.Parameter(torch.zeros(2,dtype=dtype),requires_grad=False)]
        opt=torch.optim.AdamW(params,lr=3e-5,weight_decay=1e-4,betas=(.9,.999),foreach=False)
        for p in params[:2]: p.grad=torch.randn(p.shape,generator=generator).to(dtype)*.001
        opt.step(); opt.zero_grad(set_to_none=True)
        for p in params[:2]: p.grad=torch.randn(p.shape,generator=generator).to(dtype)*.001
        bundle=capture(opt); opt.step(); result=compare_after(opt,bundle)
        assert result['all_fields_bitwise_equal'] and result['devices']==['cpu']
        assert result['parameter_objects']==2 and params[-1] not in opt.state
        assert all(opt.state[p]['step'].item()==2 for p in params[:2])
        reports.append(result)
    assert not torch.cuda.is_initialized()
    proof=dict(status='passed_cpu_optimizer_replay_contract',gpu_verified=False,cuda_initialized=False,
               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),torch_version=torch.__version__,
               observed_unix=time.time(),reports=reports,scope='CPU capture/replay/aggregate logic only; no GPU numerical result.')
    return proof

if __name__ == '__main__':
    proof=cpu_contract()
    (ROOT/'logs/optimizer_backend_replay_cpu_contract.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps({'status':proof['status'],'dtypes':len(proof['reports'])}))
