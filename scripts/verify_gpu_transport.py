"""Independent CUDA launch and world16 NCCL measurements; no training/optimizer state."""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--cpu-contract-only', action='store_true')
args = parser.parse_args()
output = Path(os.environ['OUTPUT_DIR']).resolve()
assert output.is_relative_to(ROOT / 'preflight')
output.mkdir(parents=True, exist_ok=True)
if args.cpu_contract_only:
    assert not torch.cuda.is_initialized()
    inputs = [torch.full((16,), float(rank + 1)) for rank in range(16)]
    assert torch.equal(torch.stack(inputs).sum(0), torch.full((16,), 136.))
    expected = torch.arange(1, 17, dtype=torch.float32).repeat_interleave(16)
    assert torch.equal(torch.cat(inputs), expected)
    proof = {'status': 'passed_cpu_vector_contract', 'world_size': 16,
             'gpu_verified': False, 'scientific_optimizer_updates': 0,
             'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'source_sha256': os.environ['SOURCE_SNAPSHOT_SHA256'],
             'torch_version': torch.__version__, 'cuda_initialized': torch.cuda.is_initialized(),
             'scope': 'CPU checks only. Actual CUDA launch and NCCL bandwidth not yet measured.'}
    (output / 'transport_cpu_contract.json').write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps(proof), flush=True)
    raise SystemExit(0)

rank, local_rank = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
assert int(os.environ['WORLD_SIZE']) == 16 and 0 <= local_rank < 8
launcher_local_rank = int(os.environ['OMNI_TRANSPORT_LAUNCHER_LOCAL_RANK'])
assert local_rank == 0 and 0 <= launcher_local_rank < 8
assert os.environ['CUDA_VISIBLE_DEVICES'] == str(launcher_local_rank)
proof = json.loads(Path(os.environ['OMNI_TRANSPORT_CPU_PROOF']).read_text())
assert proof['status'] == 'passed_cpu_vector_contract' and proof['world_size'] == 16
assert proof['gpu_verified'] is False and proof['cuda_initialized'] is False
assert proof['script_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
assert proof['source_sha256'] == os.environ['SOURCE_SNAPSHOT_SHA256']
assert proof['torch_version'] == torch.__version__
torch.cuda.set_device(local_rank)
device = torch.device('cuda', local_rank)
record = {'rank': rank, 'local_rank': local_rank, 'world_size': 16,
          'launcher_local_rank': launcher_local_rank, 'physical_visible_device': os.environ['CUDA_VISIBLE_DEVICES'],
          'attempt': os.environ['OMNI_ATTEMPT'], 'nonce': os.environ['OMNI_NONCE'],
          'source_sha256': os.environ['SOURCE_SNAPSHOT_SHA256'], 'started_unix': time.time(),
          'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
          'scientific_optimizer_updates': 0, 'optimizer_step_calls': 0,
          'stage': 'initializing_cuda', 'collectives': []}
def publish():
    target = output / f'transport_rank_{rank}.json'
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(target)
publish()
try:
    x = torch.ones(1024 * 1024, device=device)
    a = torch.ones((2048, 2048), dtype=torch.bfloat16, device=device)
    torch.cuda.synchronize()
    start = time.monotonic()
    for _ in range(100):
        x.add_(1.)
    torch.cuda.synchronize()
    record['pointwise_100_launches_seconds'] = time.monotonic() - start
    assert bool((x == 101).all())
    start = time.monotonic()
    for _ in range(10):
        result = a @ a
    torch.cuda.synchronize()
    record['bf16_matmul_2048_ten_seconds'] = time.monotonic() - start
    assert bool((result == 2048).all())
    record['stage'] = 'initializing_nccl'
    publish()
    dist.init_process_group('nccl', timeout=timedelta(seconds=60), device_id=device)
    record['nccl_version'] = torch.cuda.nccl.version()
    for kind, sizes in [('all_reduce', [16 * 1024, 1024**2, 32 * 1024**2, 128 * 1024**2]),
                        ('all_gather_into_tensor', [1024**2, 4 * 1024**2, 16 * 1024**2])]:
        for size in sizes:
            value = torch.full((size // 4,), float(rank + 1), device=device)
            gathered = torch.empty(16 * len(value), device=device) if kind == 'all_gather_into_tensor' else None
            dist.barrier()
            torch.cuda.synchronize()
            elapsed = []
            for _ in range(3):
                value.fill_(rank + 1)
                torch.cuda.synchronize()
                start = time.monotonic()
                if kind == 'all_reduce':
                    dist.all_reduce(value)
                else:
                    dist.all_gather_into_tensor(gathered, value)
                torch.cuda.synchronize()
                elapsed.append(time.monotonic() - start)
                if gathered is None:
                    assert bool((value == 136.).all())
                else:
                    assert all(bool((chunk == peer + 1).all()) for peer, chunk in enumerate(gathered.chunk(16)))
            record['collectives'].append({'kind': kind, 'input_bytes_per_rank': size,
                                           'seconds_each': elapsed, 'correct': True})
            record['stage'] = 'collectives_in_progress'
            publish()
            if rank == 0:
                print(json.dumps(record['collectives'][-1]), flush=True)
            del value, gathered
    record.update(stage='passed_correctness_measurements_recorded', finished_unix=time.time(),
                  peak_gpu_memory_bytes=torch.cuda.max_memory_allocated())
    publish()
    dist.destroy_process_group()
except BaseException as error:
    record.update(stage='failed', error_type=type(error).__name__, finished_unix=time.time())
    publish()
    raise
