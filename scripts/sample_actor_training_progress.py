"""Read exact-owned actor loop positions without retaining arbitrary Python locals."""
import argparse
import json
from pathlib import Path
import re
import socket
import subprocess
import time

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('descriptor', type=Path)
parser.add_argument('pid', type=int)
args = parser.parse_args()
assert args.descriptor.resolve().is_relative_to(ROOT / 'attempts')
node = json.loads(args.descriptor.read_text())
assert node['hostname'] == socket.gethostname().split('.')[0]
output = Path(node['output'])
owner = json.loads((output / 'logs' / f'owner_identity_{node["hostname"]}.json').read_text())
assert owner['attempt'] == node['attempt'] and owner['nonce'] == node['nonce']
owner_stat = Path('/proc') / str(owner['pid']) / 'stat'
assert owner_stat.read_text().rsplit(') ', 1)[1].split()[19] == owner['start_ticks']
proc = Path('/proc') / str(args.pid)
before = (proc / 'stat').read_text().rsplit(') ', 1)[1].split()
assert before[0] != 'Z' and (proc / 'cmdline').read_bytes().startswith(b'ray::WorkerDict')
environment = dict(row.split(b'=', 1) for row in (proc / 'environ').read_bytes().split(b'\0') if b'=' in row)
assert environment.get(b'OMNI_ATTEMPT') == node['attempt'].encode()
assert environment.get(b'OMNI_NONCE') == node['nonce'].encode()
sample = subprocess.run([str(ROOT / 'env/bin/py-spy'), 'dump', '--pid', str(args.pid), '--json', '--locals'],
                        capture_output=True, text=True, timeout=18)
after = (proc / 'stat').read_text().rsplit(') ', 1)[1].split()
assert after[19] == before[19]
assert owner_stat.read_text().rsplit(') ', 1)[1].split()[19] == owner['start_ticks']
frames = []
if sample.returncode == 0:
    for thread in json.loads(sample.stdout):
        for frame in thread['frames']:
            keys = set()
            if frame['filename'].endswith('/fsdp/diffusers_impl.py') and frame['name'] == '_run_forward_backward_batch':
                keys = {'micro_idx', 'n_micro', 'step', 'num_timesteps', 'gradient_accumulation_steps'}
            if frame['filename'].endswith('/workers/engine_workers.py') and 300 <= frame['line'] <= 415:
                keys = {'batch_idx', 'total_num_iterations', 'mini_batch_size_per_gpu'}
            if frame['filename'].endswith('/diffusers/models/transformers/transformer_ltx2.py') and frame['name'] == 'forward' and frame['line'] > 1500:
                keys = {'index', 'idx', 'i', 'block_idx'}
            if not keys:
                continue
            values = {row['name']: int(row['repr']) for row in frame.get('locals') or []
                      if row.get('name') in keys and re.fullmatch(r'\d{1,6}', row.get('repr', ''))}
            frames.append({'tid': thread['os_thread_id'], 'name': frame['name'],
                           'file': frame['filename'], 'line': frame['line'], 'positions': values})
result = {'unix': time.time(), 'attempt': node['attempt'], 'host': node['hostname'],
          'pid': args.pid, 'start_ticks': before[19], 'state': after[0],
          'cpu_ticks': int(after[11]) + int(after[12]), 'sample_returncode': sample.returncode,
          'frames': frames, 'scope': 'Rank-local execution positions only; not a global successful optimizer-update counter.'}
with (output / 'logs/actor_training_positions.jsonl').open('a') as stream:
    stream.write(json.dumps(result) + '\n')
print(json.dumps(result))
