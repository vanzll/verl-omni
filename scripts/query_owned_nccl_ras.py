"""Read STATUS from a loopback RAS listener owned only by this diagnostic's ranks."""
import argparse
from datetime import datetime, UTC
import json
import os
from pathlib import Path
import socket
import time

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('descriptor', type=Path)
args = parser.parse_args()
assert args.descriptor.resolve().is_relative_to(ROOT / 'preflight/runtime_gpu')
node = json.loads(args.descriptor.read_text())
assert node['hostname'] == socket.gethostname().split('.')[0]
output = Path(node['output'])
owner = json.loads((output / 'logs' / f'owner_identity_{node["hostname"]}.json').read_text())
assert owner['nonce'] == node['nonce'] and owner['attempt'] == node['attempt']


def fields(pid):
    return Path('/proc', str(pid), 'stat').read_text().rsplit(') ', 1)[1].split()


assert fields(owner['pid'])[19] == owner['start_ticks']
ranks, owned_sockets = [], set()
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        stat = fields(proc.name)
        if stat[0] == 'Z':
            continue
        argv = proc.joinpath('cmdline').read_bytes().split(b'\0')
        if str(output / 'probe_code/fsdp_cycle_rank_entry.py').encode() not in argv:
            continue
        env = dict(row.split(b'=', 1) for row in proc.joinpath('environ').read_bytes().split(b'\0') if b'=' in row)
        if env.get(b'OMNI_ATTEMPT') != node['attempt'].encode() or env.get(b'OMNI_NONCE') != node['nonce'].encode():
            continue
        rank_value = env.get(b'RANK', b'')
        if not rank_value.isdigit():
            continue
        rank = int(rank_value)
        assert rank // 8 == node['node_rank']
        ranks.append({'rank': rank, 'pid': int(proc.name), 'start_ticks': stat[19]})
        for fd in proc.joinpath('fd').iterdir():
            try:
                target = os.readlink(fd)
            except OSError:
                continue
            if target.startswith('socket:['):
                owned_sockets.add(target[8:-1])
    except (OSError, ProcessLookupError):
        continue
assert len(ranks) == 8 and len({row['rank'] for row in ranks}) == 8
listeners, connect_host = set(), None
for table, address, host in [('tcp', '0100007F', '127.0.0.1'),
                             ('tcp6', '00000000000000000000000001000000', '::1')]:
    candidates = set()
    for line in Path('/proc/net', table).read_text().splitlines()[1:]:
        row = line.split()
        if row[1] == f'{address}:{28028:04X}' and row[3] == '0A':
            candidates.add(row[9])
    if candidates:
        # Refuse this address if SO_REUSEPORT can dispatch to another job.
        assert candidates <= owned_sockets, 'RAS listener ownership is not exclusive'
        listeners, connect_host = candidates, host
        break
assert listeners and connect_host is not None, 'owned RAS listener absent'
record = {'unix': time.time(), 'attempt': node['attempt'], 'nonce': node['nonce'],
          'host': node['hostname'], 'verified_ranks': ranks,
          'verified_listener_inodes': sorted(listeners), 'loopback_host': connect_host, 'commands': ['TIMEOUT 5', 'SET FORMAT json', 'VERBOSE STATUS'],
          'scope': 'Read-only RAS status; counts are submitted collectives, not GPU completion or successful updates.'}
chunks = []
deadline = time.monotonic() + 8
with socket.create_connection((connect_host, 28028), timeout=2) as client:
    client.sendall(b'TIMEOUT 5\nSET FORMAT json\nVERBOSE STATUS\n')
    client.shutdown(socket.SHUT_WR)
    while time.monotonic() < deadline:
        client.settimeout(min(2, deadline - time.monotonic()))
        try:
            data = client.recv(65536)
        except socket.timeout:
            continue
        if not data:
            break
        chunks.append(data)
raw = b''.join(chunks).decode(errors='replace')
record['raw_status'] = raw
record['received_bytes'] = len(b''.join(chunks))
parsed = None
for offset, char in enumerate(raw):
    if char != '{':
        continue
    try:
        candidate, _ = json.JSONDecoder().raw_decode(raw[offset:])
    except json.JSONDecodeError:
        continue
    if isinstance(candidate, dict) and 'communicators' in candidate:
        parsed = candidate
        break
record['parsed_status'] = parsed
for row in ranks:
    assert fields(row['pid'])[19] == row['start_ticks']
assert fields(owner['pid'])[19] == owner['start_ticks']
stamp = datetime.now(UTC).strftime('%Y%m%dT%H%M%S.%fZ')
path = output / 'logs' / f'nccl_ras_{stamp}.json'
path.write_text(json.dumps(record, indent=2) + '\n')
summary = {'unix': record['unix'], 'host': node['hostname'], 'evidence': str(path),
           'received_bytes': record['received_bytes'], 'parsed_json': parsed is not None}
if parsed:
    summary['communicators'] = [{key: comm.get(key) for key in ('hash', 'size', 'ranks_count', 'missing_ranks_count')}
                                for comm in parsed['communicators']]
else:
    summary['text_status'] = raw[:3000]
print(json.dumps(summary))
