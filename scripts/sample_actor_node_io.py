"""Bounded read-only CPU-thread and host-wide network counters for one owned actor."""
import argparse
import json
from pathlib import Path
import socket
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
env = dict(row.split(b'=', 1) for row in (proc / 'environ').read_bytes().split(b'\0') if b'=' in row)
assert env.get(b'OMNI_ATTEMPT') == node['attempt'].encode() and env.get(b'OMNI_NONCE') == node['nonce'].encode()
keys = ['NCCL_SOCKET_IFNAME', 'NCCL_IB_HCA', 'NCCL_IB_DISABLE', 'NCCL_NET_GDR_LEVEL',
        'NCCL_P2P_DISABLE', 'NCCL_NET', 'NCCL_DEBUG', 'NCCL_IB_GID_INDEX', 'NCCL_IB_TC',
        'NCCL_ALGO', 'NCCL_PROTO', 'NCCL_MIN_NCHANNELS', 'NCCL_MAX_NCHANNELS']
def snapshot():
    counters, threads = {}, {}
    paths = list(Path('/sys/class/infiniband').glob('*/ports/*/counters/port_*_data'))
    paths += list(Path('/sys/class/net').glob('*/statistics/*_bytes'))
    for path in paths:
        try:
            counters[str(path)] = int(path.read_text()) * (4 if '/infiniband/' in str(path) else 1)
        except (OSError, ValueError):
            pass
    for task in (proc / 'task').iterdir():
        try:
            stat = (task / 'stat').read_text().rsplit(') ', 1)[1].split()
            threads[task.name] = {'ticks': int(stat[11]) + int(stat[12]),
                                  'start_ticks': stat[19], 'name': (task / 'comm').read_text().strip()}
        except OSError:
            pass
    return time.time(), counters, threads
a, ca, ta = snapshot()
time.sleep(20)
b, cb, tb = snapshot()
assert (proc / 'stat').read_text().rsplit(') ', 1)[1].split()[19] == before[19]
assert owner_stat.read_text().rsplit(') ', 1)[1].split()[19] == owner['start_ticks']
threads = [{**value, 'tid': tid, 'delta_cpu_ticks': value['ticks'] - ta[tid]['ticks']}
           for tid, value in tb.items() if tid in ta and value['start_ticks'] == ta[tid]['start_ticks']]
record = {'attempt': node['attempt'], 'host': node['hostname'], 'pid': args.pid,
          'start_ticks': before[19], 'started_unix': a, 'finished_unix': b, 'elapsed_seconds': b - a,
          'whitelisted_nccl_env': {key: env.get(key.encode(), b'').decode() for key in keys},
          'network_bytes_per_second_host_wide': {path: (cb[path] - value) / (b - a)
                                                for path, value in ca.items() if path in cb},
          'top_cpu_threads': sorted(threads, key=lambda value: -value['delta_cpu_ticks'])[:12],
          'scope': 'Host-wide net/IB counters include all processes; not isolated NCCL bandwidth. CPU ticks and loop positions are not optimizer-update counts.'}
with (output / 'logs/actor_node_io_samples.jsonl').open('a') as stream:
    stream.write(json.dumps(record) + '\n')
print(json.dumps(record))
