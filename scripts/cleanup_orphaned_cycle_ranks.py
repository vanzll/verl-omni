"""Remove only recorded, nonce-verified ranks after their diagnostic owner exited."""
import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import time

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('descriptor', type=Path)
args = parser.parse_args()
assert args.descriptor.resolve().is_relative_to(ROOT / 'preflight/runtime_gpu')
node = json.loads(args.descriptor.read_text())
assert node['hostname'] == socket.gethostname().split('.')[0]
output = Path(node['output'])
terminal = json.loads((output / 'logs' / f'owner_exit_{node["hostname"]}.json').read_text())
assert terminal['nonce'] == node['nonce'] and terminal['attempt'] == node['attempt']
owner = terminal['identity']
owner_stat = Path('/proc', str(owner['pid']), 'stat')
if owner_stat.exists():
    fields = owner_stat.read_text().rsplit(') ', 1)[1].split()
    assert fields[19] != owner['start_ticks'] or fields[0] == 'Z'
lease = {}
for line in (Path('/m2v_intern/zhenglin/paper_coordination/GPU_LEASES') / node['hostname'] / 'owner.env').read_text().splitlines():
    if line.startswith('GPU_LEASE_') and '=' in line:
        key, value = line.split('=', 1)
        parsed = shlex.split(value)
        lease[key] = parsed[0] if parsed else ''
assert lease['GPU_LEASE_ATTEMPT'] == node['attempt'] and lease['GPU_LEASE_NONCE'] == node['nonce']
assert lease['GPU_LEASE_STATUS'] == 'quarantined'
observation = json.loads((output / 'logs/native_cycle_observations.jsonl').read_text().splitlines()[-1])
assert observation['attempt'] == node['attempt'] and observation['host'] == node['hostname']
assert 0 <= time.time() - observation['unix'] < 900
ranks = observation['ranks']
assert len(ranks) == 8 and {row['rank'] for row in ranks} == set(range(node['node_rank'] * 8, (node['node_rank'] + 1) * 8))
handles = []
for row in ranks:
    proc = Path('/proc', str(row['pid']))
    if not proc.exists():
        continue
    handle = os.pidfd_open(row['pid'])
    fields = proc.joinpath('stat').read_text().rsplit(') ', 1)[1].split()
    assert fields[19] == row['start_ticks'] and fields[0] != 'Z'
    argv = proc.joinpath('cmdline').read_bytes().split(b'\0')
    assert str(output / 'probe_code/fsdp_cycle_rank_entry.py').encode() in argv
    env = dict(value.split(b'=', 1) for value in proc.joinpath('environ').read_bytes().split(b'\0') if b'=' in value)
    assert env.get(b'OMNI_ATTEMPT') == node['attempt'].encode() and env.get(b'OMNI_NONCE') == node['nonce'].encode()
    assert env.get(b'RANK') == str(row['rank']).encode()
    handles.append((handle, row))
plan = {'unix': time.time(), 'attempt': node['attempt'], 'nonce': node['nonce'],
        'verified_terminal_owner': owner, 'ranks': [row for _, row in handles],
        'method': 'PIDFD TERM, then PIDFD KILL for the same still-live identities only.'}
(output / 'logs/orphan_rank_cleanup_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
for handle, row in handles:
    try:
        signal.pidfd_send_signal(handle, signal.SIGTERM)
    except ProcessLookupError:
        pass
time.sleep(3)
killed = []
for handle, row in handles:
    proc = Path('/proc', str(row['pid']), 'stat')
    if proc.exists():
        fields = proc.read_text().rsplit(') ', 1)[1].split()
        assert fields[19] == row['start_ticks']
        if fields[0] != 'Z':
            signal.pidfd_send_signal(handle, signal.SIGKILL)
            killed.append(row['pid'])
    os.close(handle)
report = {**plan, 'kill_sent': killed, 'finished_unix': time.time(),
          'occupancy_restored': False, 'scope': 'Exact old diagnostic ranks only; no lease mutation or GPU occupancy launch.'}
(output / 'logs/orphan_rank_cleanup_execution.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'host': node['hostname'], 'exact_ranks_signaled': len(handles), 'kill_sent': killed}))
