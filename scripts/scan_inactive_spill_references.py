"""Read-only process/FD/maps admission evidence for an explicit inactive spill plan."""
import argparse
import json
import os
from pathlib import Path
import socket
import time

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('plan', type=Path)
args = parser.parse_args()
assert args.plan.resolve().is_relative_to(ROOT / 'logs')
plan = json.loads(args.plan.read_text())
assert plan['status'] == 'read_only_candidates_no_deletion' and not plan['deletion_performed']
names = {row['attempt'] for row in plan['attempts']}
prefixes = [row['spill'] for row in plan['attempts']]
for name, prefix in zip([row['attempt'] for row in plan['attempts']], prefixes):
    assert Path(prefix) == ROOT / 'tmp/ray_spill' / name
host = socket.gethostname().split('.')[0]
assert host in {node['hostname'] for row in plan['attempts'] for node in row['nodes']}
processes, references, denied = [], [], set()
scanned = 0
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        before = proc.joinpath('stat').read_text().rsplit(') ', 1)[1].split()
        if before[0] == 'Z':
            continue
        environment = dict(row.split(b'=', 1) for row in proc.joinpath('environ').read_bytes().split(b'\0') if b'=' in row)
        argv = proc.joinpath('cmdline').read_bytes()
        matches = [name for name in names if environment.get(b'OMNI_ATTEMPT') == name.encode() or name.encode() in argv]
        if matches:
            processes.append({'pid': int(proc.name), 'start_ticks': before[19], 'attempts': matches})
        targets = []
        try:
            targets.append(('cwd', os.readlink(proc / 'cwd')))
        except FileNotFoundError:
            pass
        for fd in (proc / 'fd').iterdir():
            try:
                targets.append(('fd:' + fd.name, os.readlink(fd)))
            except FileNotFoundError:
                continue
        for line in proc.joinpath('maps').read_text().splitlines():
            if any(prefix in line for prefix in prefixes):
                targets.append(('maps', line.split(maxsplit=5)[-1]))
        after = proc.joinpath('stat').read_text().rsplit(') ', 1)[1].split()
        if before[19] != after[19]:
            continue
        for kind, target in targets:
            if any(target.startswith(prefix + '/') or target == prefix for prefix in prefixes):
                references.append({'pid': int(proc.name), 'start_ticks': before[19], 'kind': kind, 'path': target})
        scanned += 1
    except PermissionError:
        denied.add(int(proc.name))
    except (FileNotFoundError, ProcessLookupError):
        continue
report = {'unix': time.time(), 'host': host, 'plan': str(args.plan.resolve()),
          'scanned_processes': scanned, 'old_attempt_processes': processes,
          'open_old_spill_references': references, 'permission_denied_processes': len(denied),
          'deletion_performed': False, 'scope': 'Only identity matches and planned-spill path references retained; no raw environment/other-process maps.'}
path = ROOT / 'logs' / f'inactive_spill_reference_scan_{host}_20261016T2304.json'
path.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
