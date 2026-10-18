"""Evict only the explicitly planned and reference-checked package caches."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import time

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')
plan_path = ROOT / 'logs/cache_cleanup_20261017_plan.json'
plan_bytes = plan_path.read_bytes()
plan = json.loads(plan_bytes)
plan_sha = hashlib.sha256(plan_bytes).hexdigest()
assert plan['status'] == 'read_only_candidates_no_deletion' and not plan['deletion_performed']
assert shutil.rmtree.avoids_symlink_attacks
lock = (ROOT / 'logs/cache_cleanup_20261017.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
hosts = ['klingai-wlf2-ge103-134', 'klingai-wlf2-ge111-95',
         'klingai-wlf2-ge111-71', 'klingai-wlf2-ge111-36']
reports = [ROOT / 'logs' / f'cache_cleanup_reference_processes_{host}.json' for host in hosts]
reports.append(ROOT / 'logs/cache_cleanup_reference_environment_klingai-wlf2-ge103-134.json')
for path in reports:
    report = json.loads(path.read_text())
    assert report['passed'] and report['plan_sha256'] == plan_sha
    assert time.time() - report['observed_unix'] < 240, 'refresh reference admission'


def identity(path):
    value = path.lstat()
    return value, (value.st_dev, value.st_ino, value.st_mtime_ns)


def permitted(row):
    path = Path(row['path'])
    kind = row['kind']
    assert path.resolve() == path and not path.is_symlink()
    if kind == 'uv_unpacked_archive':
        assert path.parent == ROOT / 'cache/uv/archive-v0'
    elif kind == 'uv_abandoned_temporary':
        assert path.parent == ROOT / 'cache/uv' and path.name.startswith('.tmp')
    elif kind == 'downloaded_wheel':
        assert path.parent == ROOT / 'assets/wheelhouse' and path.suffix == '.whl'
        assert row['reproducible_manifest'] and len(row['sha256']) == 64
    elif kind == 'conda_package_archive':
        assert path.parent == ROOT / 'cache/conda' and path.suffix == '.conda'
    else:
        raise AssertionError(kind)
    value, found = identity(path)
    assert found == (row['device'], row['inode'], row['mtime_ns']), path
    assert stat.S_ISDIR(value.st_mode) if row['is_directory'] else stat.S_ISREG(value.st_mode)


for row in plan['targets']:
    permitted(row)
started = time.time()
space = os.statvfs(ROOT)
result = dict(status='running', started_unix=started, plan_sha256=plan_sha,
              free_bytes_before=space.f_bavail * space.f_frsize,
              completed_targets=0, logical_bytes_unlinked=0,
              single_link_allocated_bytes_before_unlink=0, deleted_files=0, targets=[])
state_path = ROOT / 'logs/cache_cleanup_20261017_execution.json'
progress = (ROOT / 'logs/cache_cleanup_20261017_progress.jsonl').open('a', buffering=1)


def publish():
    temporary = state_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(state_path)


publish()
try:
    for row in plan['targets']:
        assert time.time() - started < 210, 'bounded deletion budget reached; inspect progress before continuation'
        permitted(row)
        path = Path(row['path'])
        logical = single = count = 0
        if row['is_directory']:
            for base, dirs, files in os.walk(path, followlinks=False):
                for name in files:
                    value = Path(base, name).lstat()
                    if stat.S_ISREG(value.st_mode):
                        logical += value.st_size
                        single += value.st_blocks * 512 if value.st_nlink == 1 else 0
                        count += 1
            shutil.rmtree(path)
        else:
            value = path.lstat()
            logical, single, count = value.st_size, value.st_blocks * 512 if value.st_nlink == 1 else 0, 1
            path.unlink()
        assert not path.exists() and not path.is_symlink()
        item = dict(path=str(path), kind=row['kind'], logical_bytes=logical,
                    single_link_allocated_bytes_before_unlink=single, files=count, finished_unix=time.time())
        result['targets'].append(item)
        result['completed_targets'] += 1
        result['logical_bytes_unlinked'] += logical
        result['single_link_allocated_bytes_before_unlink'] += single
        result['deleted_files'] += count
        progress.write(json.dumps(item) + '\n')
        if result['completed_targets'] % 10 == 0:
            publish()
    result['status'] = 'completed_explicit_reference_checked_cache_eviction'
except BaseException as error:
    result.update(status='interrupted_inspect_progress_before_retry', error_type=type(error).__name__,
                  error=str(error)[:300])
    raise
finally:
    result['finished_unix'] = time.time()
    space = os.statvfs(ROOT)
    result['free_bytes_after'] = space.f_bavail * space.f_frsize
    result['observed_shared_free_delta_bytes'] = result['free_bytes_after'] - result['free_bytes_before']
    result['physical_reclamation_guaranteed'] = False
    publish()
    print(json.dumps({key: value for key, value in result.items() if key != 'targets'}), flush=True)
