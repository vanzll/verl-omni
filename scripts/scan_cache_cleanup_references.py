"""Read-only admission for the explicit campaign cache eviction plan."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import time

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('mode', choices=['environment', 'processes'])
args = parser.parse_args()
plan_path = ROOT / 'logs/cache_cleanup_20261017_plan.json'
plan = json.loads(plan_path.read_text())
assert plan['status'] == 'read_only_candidates_no_deletion' and not plan['deletion_performed']
targets = [row['path'] for row in plan['targets']]
anchors = [str(ROOT / 'cache/uv/archive-v0'), str(ROOT / 'cache/uv/.tmp'),
           str(ROOT / 'assets/wheelhouse'), str(ROOT / 'cache/conda')]


def matches(value):
    if not any(anchor in value for anchor in anchors):
        return []
    return [target for target in targets if target in value]


start = time.monotonic()
references, denied, mutations = [], [], []
scanned = 0
if args.mode == 'environment':
    for base, dirs, files in os.walk(ROOT / 'env', followlinks=False):
        assert time.monotonic() - start < 70, 'environment scan budget exceeded'
        for name in dirs + files:
            path = Path(base, name)
            if path.is_symlink():
                found = matches(str(path.resolve(strict=False)))
                if found:
                    references.append(dict(kind='environment_symlink', path=str(path), targets=found))
            elif name.endswith(('.pth', '.egg-link')) or name.startswith('__editable__') and name.endswith('.py'):
                found = matches(path.read_text(errors='replace'))
                if found:
                    references.append(dict(kind='environment_import_path', path=str(path), targets=found))
            scanned += 1
else:
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            before = (proc / 'stat').read_text().rsplit(') ', 1)[1].split()
            if before[0] == 'Z':
                continue
            argv = [x.decode(errors='replace') for x in (proc / 'cmdline').read_bytes().split(b'\0') if x]
            environment = (proc / 'environ').read_bytes().replace(b'\0', b'\n').decode(errors='replace')
            values = [('argv', '\n'.join(argv)), ('environment', environment)]
            try:
                values.append(('cwd', os.readlink(proc / 'cwd')))
            except FileNotFoundError:
                pass
            for fd in (proc / 'fd').iterdir():
                try:
                    values.append(('fd', os.readlink(fd)))
                except FileNotFoundError:
                    pass
            values.append(('maps', (proc / 'maps').read_text()))
            after = (proc / 'stat').read_text().rsplit(') ', 1)[1].split()
            if before[19] != after[19]:
                continue
            for kind, value in values:
                found = matches(value)
                if found:
                    references.append(dict(pid=int(proc.name), start_ticks=before[19], kind=kind, targets=found))
            executable = Path(argv[0]).name if argv else ''
            package_tool = executable in ('uv', 'pip', 'pip3', 'conda', 'mamba') or '-m' in argv and any(x in argv for x in ('pip', 'conda'))
            if package_tool and any(x in argv for x in ('install', 'download', 'sync', 'clean', 'remove', 'uninstall')):
                if str(ROOT) in '\n'.join(argv) + environment:
                    mutations.append(dict(pid=int(proc.name), start_ticks=before[19], executable=executable))
            scanned += 1
        except PermissionError:
            denied.append(int(proc.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
host = socket.gethostname().split('.')[0]
report = dict(observed_unix=time.time(), mode=args.mode, host=host, scanned=scanned,
              wall_seconds=time.monotonic() - start, references=references,
              permission_denied_count=len(denied), package_mutations=mutations,
              plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
              passed=not references and not denied and not mutations,
              scope='Only planned-target matches retained; no raw process environment or unrelated mapped paths.')
output = ROOT / 'logs' / f'cache_cleanup_reference_{args.mode}_{host}.json'
output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
