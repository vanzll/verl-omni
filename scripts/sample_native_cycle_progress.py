"""Read a nonce-owned diagnostic rank's frames without retaining raw locals."""
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
parser.add_argument('--native', action='store_true')
args = parser.parse_args()
assert args.descriptor.resolve().is_relative_to(ROOT / 'preflight/runtime_gpu')
node = json.loads(args.descriptor.read_text())
assert node['hostname'] == socket.gethostname().split('.')[0]
output = Path(node['output'])
owner = json.loads((output / 'logs' / f'owner_identity_{node["hostname"]}.json').read_text())
assert owner['attempt'] == node['attempt'] and owner['nonce'] == node['nonce']


def stat(path):
    return path.joinpath('stat').read_text().rsplit(') ', 1)[1].split()


owner_proc = Path('/proc', str(owner['pid']))
assert stat(owner_proc)[19] == owner['start_ticks']
ranks = []
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        fields = stat(proc)
        if fields[0] == 'Z' or b'fsdp_cycle_rank_entry.py' not in proc.joinpath('cmdline').read_bytes():
            continue
        env = dict(row.split(b'=', 1) for row in proc.joinpath('environ').read_bytes().split(b'\0') if b'=' in row)
        if env.get(b'OMNI_ATTEMPT') != node['attempt'].encode() or env.get(b'OMNI_NONCE') != node['nonce'].encode():
            continue
        rank = env.get(b'RANK', b'')
        if rank.isdigit():
            ranks.append({'pid': int(proc.name), 'rank': int(rank), 'start_ticks': fields[19],
                          'cpu_ticks': int(fields[11]) + int(fields[12]), 'state': fields[0]})
    except (OSError, ProcessLookupError):
        pass
assert len({row['rank'] for row in ranks}) == len(ranks)
threads = []
target = next((row for row in ranks if row['rank'] % 8 == 0), None)
if target:
    command = [str(ROOT / 'env/bin/py-spy'), 'dump', '--json', '--pid', str(target['pid'])]
    command.append('--native' if args.native else '--locals')
    result = subprocess.run(command, text=True, capture_output=True, timeout=18)
    assert result.returncode == 0
    assert stat(Path('/proc', str(target['pid'])))[19] == target['start_ticks']
    for thread in json.loads(result.stdout):
        if thread['os_thread_id'] != target['pid']:
            continue
        frames = []
        for frame_index, frame in enumerate(thread['frames']):
            names = set()
            if frame['name'] == '_run_forward_backward_batch' and frame['filename'].endswith('/fsdp/diffusers_impl.py'):
                names = {'micro_idx', 'n_micro', 'step', 'num_timesteps', 'gradient_accumulation_steps'}
            if frame['name'] == 'forward' and frame['filename'].endswith('/transformer_ltx2.py') and frame['line'] > 1500:
                names = {'block_idx'}
            if frame_index >= 24 and not names:
                continue
            values = {item['name']: int(item['repr']) for item in frame.get('locals') or []
                      if item.get('name') in names and re.fullmatch(r'\d{1,6}', item.get('repr', ''))}
            frames.append({'file': frame['filename'], 'line': frame['line'], 'name': frame['name'], 'positions': values})
        threads.append({'tid': thread['os_thread_id'], 'frames': frames})
assert stat(owner_proc)[19] == owner['start_ticks']
record = {'unix': time.time(), 'attempt': node['attempt'], 'host': node['hostname'],
          'ranks': sorted(ranks, key=lambda row: row['rank']), 'target': target,
          'threads': threads, 'native': args.native,
          'scope': 'Read-only diagnostic process/rank positions; not successful optimizer counters.'}
with (output / 'logs/native_cycle_observations.jsonl').open('a') as stream:
    stream.write(json.dumps(record) + '\n')
print(json.dumps(record))
