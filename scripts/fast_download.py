#!/usr/bin/env python3
"""Resumable verified HTTP ranges, with a bounded disk-space pause."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import logging
import math
import os
import signal
import threading
import time
from pathlib import Path

import requests

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')
logging.disable(logging.CRITICAL)
LOCAL = threading.local()


class RangeDownloadError(RuntimeError):
    def __init__(self, cause_type, http_status):
        super().__init__(cause_type)
        self.cause_type = cause_type
        self.http_status = http_status


def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except OSError:
        pass


def atomic(path, value):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def digest(path, git_blob=False):
    h = hashlib.sha1() if git_blob else hashlib.sha256()
    if git_blob:
        h.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as f:
        while data := f.read(4 * 1024**2):
            h.update(data)
    return h.hexdigest()


def session():
    if not hasattr(LOCAL, 'session'):
        LOCAL.session = requests.Session()
    return LOCAL.session


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest')
    parser.add_argument('--state', required=True)
    parser.add_argument('--files', type=int, default=2)
    parser.add_argument('--connections', type=int, default=64)
    parser.add_argument('--request-mib', type=int, default=1)
    parser.add_argument('--min-free-gib', type=int, default=250)
    parser.add_argument('--resume-free-gib', type=int, default=275)
    args = parser.parse_args()
    assert args.resume_free_gib > args.min_free_gib >= 16
    assert 1 <= args.request_mib <= 32
    tasks = json.loads(Path(args.manifest).read_text())
    state_path = Path(args.state)
    assert state_path.resolve().is_relative_to(ROOT)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = state_path.with_suffix('.lock').open('w')
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stop, space_ok = threading.Event(), threading.Event()
    guard_lock = threading.Lock()
    state = dict(status='running', pid=os.getpid(), min_free_gib=args.min_free_gib,
                 resume_free_gib=args.resume_free_gib, connections_per_large_file=args.connections,
                 concurrent_files=args.files, tasks=len(tasks), verified=[], failed=[], active={},
                 request_max_bytes=args.request_mib * 1024**2,
                 started_utc=time.strftime('%FT%TZ', time.gmtime()), transfer_bytes=0,
                 retry_events=0, retry_causes={})
    started = time.monotonic()

    def interrupted(signum, frame):
        stop.set()

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def monitor():
        while not stop.is_set():
            try:
                disk = os.statvfs(ROOT)
                free = disk.f_bavail * disk.f_frsize
                if free < args.min_free_gib * 1024**3 and space_ok.is_set():
                    space_ok.clear()
                    safe_print('SPACE PAUSE free_GiB=' + str(free // 1024**3), flush=True)
                elif free >= args.resume_free_gib * 1024**3 and not space_ok.is_set():
                    space_ok.set()
                    safe_print('SPACE RESUME free_GiB=' + str(free // 1024**3), flush=True)
                with guard_lock:
                    state['free_bytes'] = free
                    state['space_paused'] = not space_ok.is_set()
                    state['elapsed_seconds'] = time.monotonic() - started
                    state.pop('storage_errno', None)
                    atomic(state_path, state)
            except OSError as error:
                space_ok.clear()
                with guard_lock:
                    state['storage_errno'] = error.errno
                    state['space_paused'] = True
            stop.wait(5)

    def gate():
        deadline = time.monotonic() + 1800
        while not space_ok.wait(1):
            if stop.is_set() or time.monotonic() > deadline:
                raise InterruptedError('space wait ended')
        if stop.is_set():
            raise InterruptedError('download stopping')

    def download(task):
        name, size = task['name'], int(task['size'])
        git_blob = 'git_blob_sha1' in task
        expected = task['git_blob_sha1'] if git_blob else task['sha256']
        target = Path(task['destination'])
        assert target.resolve().is_relative_to(ROOT)
        gate()
        target.parent.mkdir(parents=True, exist_ok=True)
        parts = ROOT / 'cache/http_ranges' / expected
        parts.mkdir(parents=True, exist_ok=True)

        def legacy_safe():
            owner = task.get('legacy_owner')
            if not owner:
                return False
            try:
                raw = Path(f"/proc/{owner['pid']}/stat").read_text()
                ticks = int(raw[raw.rfind(')') + 2:].split()[19])
                return ticks != owner['start_ticks']
            except FileNotFoundError:
                return True

        def publish_metadata():
            if task.get('hf_metadata'):
                p = Path(task['hf_metadata'])
                assert p.resolve().is_relative_to(ROOT)
                p.parent.mkdir(parents=True, exist_ok=True)
                tmp = p.with_name(p.name + '.fast-tmp')
                tmp.write_text(f"{task['hf']['revision']}\n{expected}\n{time.time()}\n")
                tmp.replace(p)
            if legacy_safe():
                for old in task.get('legacy_partials', []):
                    old = Path(old)
                    assert old.resolve().is_relative_to(ROOT) and old.name.endswith('.incomplete')
                    old.unlink(missing_ok=True)

        with (parts / 'lock').open('w') as file_lock:
            fcntl.flock(file_lock, fcntl.LOCK_EX)
            if target.is_file() and target.stat().st_size == size and digest(target, git_blob) == expected:
                publish_metadata()
                return dict(name=name, size=size, status='verified', reused=True)
            gate()
            if legacy_safe():
                for old in task.get('legacy_partials', []):
                    old = Path(old)
                    assert old.resolve().is_relative_to(ROOT)
                    if old.is_file() and old.stat().st_size == size and digest(old, git_blob) == expected:
                        old.replace(target)
                        publish_metadata()
                        return dict(name=name, size=size, status='verified', reused=True)
            stamp = dict(size=size, sha256=expected)
            if git_blob:
                stamp['hash_kind'] = 'git_blob_sha1'
            meta = parts / 'identity.json'
            if meta.exists():
                assert json.loads(meta.read_text()) == stamp
            else:
                atomic(meta, stamp)
            connections = 1 if size < 1024**2 else min(args.connections, 16 if size < 16 * 1024**2 else 64)
            chunk_size = max(64 * 1024, min(32 * 1024**2, math.ceil(size / connections)))
            count = math.ceil(size / chunk_size)
            signed = {'url': task['url'], 'at': 0}
            signed_lock = threading.Lock()

            def url(force=False):
                if 'hf' not in task:
                    return task['url']
                with signed_lock:
                    if force or time.monotonic() - signed['at'] > 900:
                        from huggingface_hub import get_hf_file_metadata
                        m = get_hf_file_metadata(task['url'], timeout=20)
                        assert m.size == size and m.etag == expected
                        signed.update(url=m.location, at=time.monotonic())
                    return signed['url']

            prior = sum(p.stat().st_size for p in parts.glob('chunk-*.done'))
            prior += sum(p.stat().st_size for p in parts.glob('chunk-*.partial'))
            with guard_lock:
                state['active'][name] = dict(size=size, downloaded_bytes=prior, connections=connections)
            safe_print(f'START {name} bytes={size} resume={prior} connections={connections}', flush=True)

            def chunk(index):
                first, last = index * chunk_size, min(size, (index + 1) * chunk_size) - 1
                done = parts / f'chunk-{index:06d}.done'
                partial = parts / f'chunk-{index:06d}.partial'
                expected_size = last - first + 1
                if done.exists():
                    assert done.stat().st_size == expected_size
                    return done
                failures = 0
                while True:
                    gate()
                    have = partial.stat().st_size if partial.exists() else 0
                    assert have <= expected_size
                    if have == expected_size:
                        partial.replace(done)
                        return done
                    request_last = min(last, first + have + args.request_mib * 1024**2 - 1)
                    request_expected_end = have + request_last - (first + have) + 1
                    try:
                        response = None
                        with session().get(url(), headers={'Range': f'bytes={first + have}-{request_last}',
                            'Accept-Encoding': 'identity'}, stream=True, timeout=(10, 30)) as response:
                            if response.status_code == 403 and 'hf' in task:
                                url(force=True)
                            response.raise_for_status()
                            content_range = f'bytes {first + have}-{request_last}/{size}'
                            if response.status_code != 206 or response.headers.get('Content-Range') != content_range:
                                raise ValueError('server did not honor byte range')
                            with partial.open('ab') as output:
                                for data in response.iter_content(128 * 1024):
                                    gate()
                                    if output.tell() + len(data) > request_expected_end:
                                        raise ValueError('range exceeded expected size')
                                    output.write(data)
                                    with guard_lock:
                                        state['transfer_bytes'] += len(data)
                                        state['active'][name]['downloaded_bytes'] += len(data)
                                output.flush()
                                os.fsync(output.fileno())
                        if partial.stat().st_size != request_expected_end:
                            raise IOError('truncated range')
                        failures = 0
                    except InterruptedError:
                        raise
                    except Exception as error:
                        failed_response = getattr(error, 'response', None)
                        if failed_response is None:
                            failed_response = response
                        status = getattr(failed_response, 'status_code', None)
                        cause = f'{type(error).__name__}:{status}'
                        with guard_lock:
                            state['retry_events'] += 1
                            state['retry_causes'][cause] = state['retry_causes'].get(cause, 0) + 1
                        failures += 1
                        if failures >= 8:
                            raise RangeDownloadError(type(error).__name__, status) from None
                        if stop.wait(min(20, 2**(failures - 1))):
                            raise InterruptedError('download stopping')

            with concurrent.futures.ThreadPoolExecutor(max_workers=connections) as pool:
                completed = list(pool.map(chunk, range(count)))
            gate()
            temporary = target.with_name(target.name + '.fast-download-partial')
            h = hashlib.sha1() if git_blob else hashlib.sha256()
            if git_blob:
                h.update(f'blob {size}\0'.encode())
            with temporary.open('wb') as output:
                for piece in completed:
                    with piece.open('rb') as source:
                        while data := source.read(4 * 1024**2):
                            gate()
                            h.update(data)
                            output.write(data)
                output.flush()
                os.fsync(output.fileno())
            assert temporary.stat().st_size == size and h.hexdigest() == expected, 'checksum mismatch'
            temporary.replace(target)
            publish_metadata()
            for piece in completed:
                piece.unlink()
            safe_print(f'VERIFIED {name} bytes={size} {"git_blob_sha1" if git_blob else "sha256"}={expected}', flush=True)
            return dict(name=name, size=size, status='verified', destination=str(target))

    def one(task):
        if stop.is_set():
            return
        try:
            result = download(task)
            with guard_lock:
                state['verified'].append(result)
        except Exception as error:
            with guard_lock:
                state['failed'].append(dict(name=task['name'], error_type=type(error).__name__,
                    cause_type=getattr(error, 'cause_type', None), http_status=getattr(error, 'http_status', None)))
            safe_print(f'ERROR {task["name"]} type={type(error).__name__}', flush=True)
        finally:
            with guard_lock:
                state['active'].pop(task['name'], None)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.files) as pool:
            list(pool.map(one, tasks))
    finally:
        stop.set()
        monitor_thread.join(timeout=6)
        state['status'] = 'incomplete' if state['failed'] or len(state['verified']) != len(tasks) else 'completed'
        try:
            atomic(state_path, state)
        except OSError as error:
            safe_print('final state storage errno=' + str(error.errno), flush=True)
    safe_print('DONE verified=' + str(len(state['verified'])) + ' failed=' + str(len(state['failed'])), flush=True)
    sys_exit = 1 if state['failed'] else 0
    raise SystemExit(sys_exit)


if __name__ == '__main__':
    main()
