"""Read integer reward dispatch positions from an exact live campaign driver."""
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
owner_stat = Path('/proc', str(owner['pid']), 'stat')
assert owner_stat.read_text().rsplit(') ', 1)[1].split()[19] == owner['start_ticks']
proc = Path('/proc', str(args.pid))
before = (proc / 'stat').read_text().rsplit(') ', 1)[1].split()
assert before[0] != 'Z' and (proc / 'cmdline').read_bytes().startswith(b'ray::TaskRunner')
environment = dict(x.split(b'=', 1) for x in (proc / 'environ').read_bytes().split(b'\0') if b'=' in x)
assert environment.get(b'OMNI_ATTEMPT') == node['attempt'].encode()
assert environment.get(b'OMNI_NONCE') == node['nonce'].encode()
sample = subprocess.run([str(ROOT / 'env/bin/py-spy'), 'dump', '--pid', str(args.pid), '--json', '--locals'],
                        capture_output=True, text=True, timeout=18)
assert sample.returncode == 0
frames = []
for thread in json.loads(sample.stdout):
    for frame in thread['frames']:
        if frame['name'] != 'run_replica' or not frame['filename'].endswith('/reward_loop/replica_dispatch.py'):
            continue
        positions = {}
        for value in frame.get('locals') or []:
            key, representation = value.get('name'), value.get('repr', '')
            if key in {'start', 'stop', 'replica_index'} and re.fullmatch(r'\d{1,6}', representation):
                positions[key] = int(representation)
            if key == 'name' and representation.strip('"\'') in {'video_align', 'hpsv3', 'audiobox', 'clap', 'desync'}:
                positions[key] = representation.strip('"\'')
        frames.append({'tid': thread['os_thread_id'], 'line': frame['line'], 'positions': positions})
assert (proc / 'stat').read_text().rsplit(') ', 1)[1].split()[19] == before[19]
assert owner_stat.read_text().rsplit(') ', 1)[1].split()[19] == owner['start_ticks']
result = {'unix': time.time(), 'attempt': node['attempt'], 'pid': args.pid, 'start_ticks': before[19],
          'frames': frames,
          'scope': 'Currently executing reward dispatch coroutine only; not all-group completion or successful global optimizer updates.'}
with (output / 'logs/reward_dispatch_positions.jsonl').open('a') as stream:
    stream.write(json.dumps(result) + '\n')
print(json.dumps(result))
