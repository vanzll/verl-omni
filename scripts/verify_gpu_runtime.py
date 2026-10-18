"""Small real kernels and eight-rank NCCL, independent of campaign model/RNG."""
from datetime import timedelta
import json
import os
from pathlib import Path
import socket
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

rank = int(os.environ['LOCAL_RANK'])
global_rank = int(os.environ['RANK'])
report = {'host': socket.gethostname(), 'rank': rank, 'torch': torch.__version__,
          'cuda_runtime': torch.version.cuda, 'status': 'running', 'stages': {}}
output = Path(os.environ['OUTPUT_DIR']) / f'gpu_rank_{rank}.json'


def save(stage, values):
    report['stages'][stage] = values
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(output)
    print(f'rank={rank} stage={stage} {values}', flush=True)


def finite(tensor):
    assert bool(torch.isfinite(tensor).all()), 'nonfinite GPU tensor'


try:
    started = time.monotonic()
    torch.cuda.set_device(rank)
    torch.manual_seed(20261016 + rank)
    report['gpu'] = torch.cuda.get_device_name(rank)
    report['capability'] = torch.cuda.get_device_capability(rank)
    assert torch.cuda.device_count() == 8
    save('cuda_init', {'seconds': time.monotonic() - started})
    a = torch.randn(128, 128, device='cuda', dtype=torch.float32)
    b = torch.randn_like(a)
    reference = a.cpu().double() @ b.cpu().double()
    for dtype in [torch.float32, torch.bfloat16]:
        result = a.to(dtype) @ b.to(dtype)
        finite(result)
        err = (result.cpu().double() - reference).norm() / reference.norm()
        assert float(err) < (0.02 if dtype == torch.bfloat16 else 0.002), err
        save(f'matmul_{dtype}', {'relative_error': float(err)})
    q, k, v = [torch.randn(1, 4, 128, 64, device='cuda', dtype=torch.bfloat16,
                          requires_grad=True) for _ in range(3)]
    with sdpa_kernel(SDPBackend.MATH):
        reference_attn = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        attention = F.scaled_dot_product_attention(q, k, v)
    finite(attention)
    err = ((attention.float() - reference_attn).norm() / reference_attn.norm()).detach()
    assert float(err) < 0.03, err
    attention.float().square().mean().backward()
    for value in [q, k, v]:
        finite(value.grad)
    save('flash_sdpa_forward_backward', {'relative_error': float(err)})
    image = torch.randn(1, 4, 3, 16, 16, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(8, 4, 3, 3, 3, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    convolution = F.conv3d(image, weight, padding=1)
    convolution.float().square().mean().backward()
    finite(convolution)
    finite(image.grad)
    finite(weight.grad)
    audio = torch.randn(1, 4096, device='cuda')
    finite(torch.fft.rfft(audio).abs())
    parameter = torch.nn.Parameter(torch.ones(64, device='cuda', dtype=torch.float32))
    optimizer = torch.optim.AdamW([parameter], lr=3e-5, weight_decay=1e-4)
    parameter.float().square().mean().backward()
    optimizer.step()
    finite(parameter)
    assert bool((parameter < 1.0).all()), 'AdamW did not change FP32 parameters'
    save('conv3d_fft_adamw', {'finite': True, 'updated_parameter': float(parameter[0].detach())})
    dist.init_process_group('nccl', timeout=timedelta(seconds=90), device_id=torch.device('cuda', rank))
    world = dist.get_world_size()
    expected = world * (world + 1) / 2
    value = torch.full((4096,), float(global_rank + 1), device='cuda')
    start = time.monotonic()
    for _ in range(3):
        value.fill_(global_rank + 1)
        dist.all_reduce(value)
        assert bool((value == expected).all()), 'NCCL sum mismatch'
    dist.barrier()
    torch.cuda.synchronize()
    save('nccl_all_reduce', {'world_size': dist.get_world_size(), 'sum': float(value[0]),
                             'seconds': time.monotonic() - start})
    dist.destroy_process_group()
    report['status'] = 'passed'
    save('complete', {'seconds': time.monotonic() - started})
except BaseException as error:
    report['status'] = 'failed'
    save('exception', {'type': type(error).__name__, 'message': str(error)[:2000]})
    raise
