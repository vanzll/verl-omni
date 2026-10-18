"""Read whitelisted frame metadata from an exact live, nonce-fenced campaign process."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import time

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('descriptor', type=Path)
parser.add_argument('pid', type=int)
parser.add_argument('--native', action='store_true')
parser.add_argument('--tid', type=int, action='append', default=[])
parser.add_argument('--prefix', required=True, choices=['ray::WorkerDict', 'ray::TaskRunner',
                    'ray::OmniRewardLoopWorker', 'ray::vLLMOmniHttpServer',
                    'ray::DiffusionAgentLoopWorker'])
args = parser.parse_args()
assert args.descriptor.resolve().is_relative_to(ROOT / 'attempts')
node = json.loads(args.descriptor.read_text())
assert node['hostname'] == socket.gethostname().split('.')[0]
output = Path(node['output'])
owner = json.loads((output / 'logs' / f'owner_identity_{node["hostname"]}.json').read_text())
assert owner['nonce'] == node['nonce'] and owner['attempt'] == node['attempt']
owner_stat = Path('/proc') / str(owner['pid']) / 'stat'
assert owner_stat.read_text().rsplit(') ', 1)[1].split()[19] == owner['start_ticks']
proc = Path('/proc') / str(args.pid)
before = proc.joinpath('stat').read_text().rsplit(') ', 1)[1].split()
assert before[0] != 'Z' and proc.joinpath('cmdline').read_bytes().startswith(args.prefix.encode())
environment = dict(value.split(b'=', 1) for value in proc.joinpath('environ').read_bytes().split(b'\0') if b'=' in value)
assert environment.get(b'OMNI_ATTEMPT') == node['attempt'].encode()
assert environment.get(b'OMNI_NONCE') == node['nonce'].encode()
command = [str(ROOT / 'env/bin/py-spy'), 'dump', '--pid', str(args.pid), '--json']
if args.native:
    command.append('--native')
sample = subprocess.run(command,
                        capture_output=True, text=True, timeout=18)
assert sample.returncode == 0
after = proc.joinpath('stat').read_text().rsplit(') ', 1)[1].split()
assert after[19] == before[19]
assert owner_stat.read_text().rsplit(') ', 1)[1].split()[19] == owner['start_ticks']
threads = []
for thread in json.loads(sample.stdout):
    if args.tid and thread['os_thread_id'] not in args.tid:
        continue
    if not args.tid and thread['os_thread_id'] != args.pid and not any(
            'verl_omni' in frame['filename'] for frame in thread['frames']):
        continue
    threads.append({'tid': thread['os_thread_id'], 'active': thread.get('active'),
                    'frames': [{'name': frame['name'], 'file': frame['filename'], 'line': frame['line']}
                               for frame in thread['frames'][:24 if args.native else 32]]})
report = {'unix': time.time(), 'attempt': node['attempt'], 'host': node['hostname'],
          'pid': args.pid, 'start_ticks': before[19], 'cpu_ticks': int(after[11]) + int(after[12]),
          'threads': threads}
report['native_stack_sample'] = args.native
suffix = '_native' if args.native else ''
(output / 'logs' / f'process_stage_{args.pid}{suffix}.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
