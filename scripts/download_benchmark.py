#!/usr/bin/env python3
"""Bounded, isolated HF transport comparison; never uses training asset caches."""
import concurrent.futures
import hashlib
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')
BENCH_NAME = os.environ.get('OMNI_BENCH_NAME', 'download_benchmark_20261016')
assert re.fullmatch(r'[A-Za-z0-9_.-]+', BENCH_NAME)
OUT = ROOT / 'logs' / BENCH_NAME
START_FREE_GIB = int(os.environ.get('OMNI_BENCH_START_FREE_GIB', '350'))
RUN_FREE_GIB = int(os.environ.get('OMNI_BENCH_RUN_FREE_GIB', '300'))
assert START_FREE_GIB >= RUN_FREE_GIB >= 250
SMALL = dict(repo='sshleifer/tiny-gpt2', revision='5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be',
             filename='pytorch_model.bin', size=2514146,
             sha256='b706b24034032bdfe765ded5ab6403d201d295a995b790cb24c74becca5c04e6')
LARGE = dict(repo='diffusers/LTX-2.3-Diffusers', revision='8eee8edcf067e838b843f926ec4d4cc9b2be1aaf',
             filename='audio_vae/diffusion_pytorch_model.safetensors', size=106507972,
             sha256='45dcb1c3cffc8c8b7710a04dd19500e481daff31f5ea8e8529f2cf697c439d12')


def atomic(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def identity(pid):
    p = Path(f'/proc/{pid}')
    raw = (p / 'stat').read_text()
    fields = raw[raw.rfind(')') + 2:].split()
    return dict(pid=pid, ticks=int(fields[19]), state=fields[0],
                cmd=(p / 'cmdline').read_bytes().replace(b'\0', b' ').decode().strip())


def free_bytes():
    s = os.statvfs(ROOT)
    return s.f_bavail * s.f_frsize


def worker(cfg):
    import requests
    from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url
    logging.disable(logging.CRITICAL)
    dest = Path(cfg['directory'])
    asset = cfg['asset']
    start = time.monotonic()
    if cfg['mode'].startswith('xet') or cfg['concurrency'] == 1:
        path = Path(hf_hub_download(asset['repo'], asset['filename'], revision=asset['revision'],
                                  cache_dir=str(dest / 'hub'), local_dir=str(dest / 'output')))
    else:
        n = cfg['concurrency']
        meta = get_hf_file_metadata(hf_hub_url(asset['repo'], asset['filename'],
                                             revision=asset['revision']), timeout=15)
        assert meta.size == asset['size'] and meta.etag == asset['sha256']

        def part(i):
            begin, end = asset['size'] * i // n, asset['size'] * (i + 1) // n - 1
            with requests.get(meta.location, headers={'Range': f'bytes={begin}-{end}',
                              'Accept-Encoding': 'identity'}, stream=True, timeout=(8, 20)) as r:
                assert r.status_code == 206
                assert r.headers.get('Content-Range') == f'bytes {begin}-{end}/{asset["size"]}'
                target = dest / f'part-{i:02d}'
                with target.open('wb') as f:
                    for chunk in r.iter_content(64 * 1024):
                        f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
                assert target.stat().st_size == end - begin + 1
                return target

        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
            pieces = list(pool.map(part, range(n)))
        path = dest / 'payload.bin'
        with path.open('wb') as f:
            for piece in pieces:
                with piece.open('rb') as source:
                    while chunk := source.read(1024 * 1024):
                        f.write(chunk)
            f.flush()
            os.fsync(f.fileno())
    received = time.monotonic() - start
    h = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    actual = h.hexdigest()
    assert path.stat().st_size == asset['size'] and actual == asset['sha256']
    atomic(dest / 'completed.json', dict(status='verified', download_seconds=received,
           total_seconds=time.monotonic() - start, bytes=asset['size'], sha256=actual,
           payload_MBps=asset['size'] / received / 1e6))


def sockets(pid):
    lines = subprocess.check_output(['ss', '-tinp'], text=True, timeout=4).splitlines()
    found = {}
    for i, line in enumerate(lines[:-1]):
        if not re.search(rf'pid={pid}(?:,|\))', line):
            continue
        match = re.search(r'bytes_received:(\d+)', lines[i + 1])
        if match:
            fields = line.split()
            found[(fields[3], fields[4])] = int(match.group(1))
    return found


def run_case(name, mode, concurrency, asset, limit, state):
    directory = OUT / name
    directory.mkdir()
    cfg = dict(directory=str(directory), mode=mode, concurrency=concurrency, asset=asset)
    atomic(directory / 'config.json', cfg)
    env = os.environ.copy()
    env.update(HF_HOME=str(directory / 'hf'), HF_HUB_CACHE=str(directory / 'hub'),
               HF_XET_CACHE=str(directory / 'xet'), HF_HUB_DISABLE_PROGRESS_BARS='1',
               HF_HUB_DISABLE_XET='0' if mode.startswith('xet') else '1',
               HF_XET_CHUNK_CACHE_SIZE_BYTES='0', HF_HUB_DOWNLOAD_TIMEOUT='30',
               HF_HUB_ETAG_TIMEOUT='15', RUST_LOG='off',
               XDG_CACHE_HOME=str(directory / 'xdg'), TMPDIR=str(ROOT / 'tmp'))
    for key in ['HF_XET_HIGH_PERFORMANCE', 'HF_XET_NUM_CONCURRENT_RANGE_GETS',
                'HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY']:
        env.pop(key, None)
    if mode.startswith('xet') and concurrency is not None:
        env['HF_XET_NUM_CONCURRENT_RANGE_GETS'] = str(concurrency)
    if mode == 'xet_hp':
        env['HF_XET_HIGH_PERFORMANCE'] = '1'
    if mode == 'xet_sequential':
        env['HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY'] = '1'
    print(f'START {name} mode={mode} concurrency={concurrency} limit={limit}s', flush=True)
    start = time.monotonic()
    proc = subprocess.Popen(['/usr/bin/python', __file__, '--worker', str(directory / 'config.json')],
                            env=env, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    state['current'] = dict(name=name, pid=proc.pid, start_utc=time.strftime('%FT%TZ', time.gmtime()))
    atomic(OUT / 'state.json', state)
    previous, network_bytes, timeout = {}, 0, False
    try:
        while proc.poll() is None:
            if free_bytes() < RUN_FREE_GIB * 1024**3:
                raise RuntimeError('space fell below diagnostic reserve')
            current = sockets(proc.pid)
            network_bytes += sum(max(0, value - previous.get(key, 0)) for key, value in current.items())
            previous = current
            if time.monotonic() - start > limit:
                timeout = True
                break
            time.sleep(.5)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=3)
    elapsed = time.monotonic() - start
    result = dict(name=name, mode=mode, concurrency=concurrency, elapsed_seconds=elapsed,
                  status='timeout' if timeout else 'error', exit_code=proc.returncode,
                  observed_tcp_bytes=network_bytes, observed_tcp_MBps=network_bytes / elapsed / 1e6,
                  note='TCP count is a lower bound when sockets close between polls; partial Xet file size is not progress')
    if (directory / 'completed.json').exists():
        result.update(json.loads((directory / 'completed.json').read_text()))
    atomic(directory / 'result.json', result)
    state['results'].append(result)
    atomic(OUT / 'state.json', state)
    print('RESULT ' + json.dumps(result), flush=True)
    return result


def main():
    OUT.mkdir(exist_ok=False)
    expected = {184791: 'hf download diffusers/LTX-2.3-Diffusers',
                297729: 'hf download KlingTeam/VideoReward',
                316592: 'uv pip install --python ' + str(ROOT / 'env/bin/python')}
    originals = [identity(pid) for pid in expected if Path(f'/proc/{pid}').exists()]
    assert originals
    for p in originals:
        assert expected[p['pid']] in p['cmd'] and p['state'] != 'T'
    guard = identity(327308)
    assert 'scripts/guard_download_space.py' in guard['cmd']
    assert free_bytes() > START_FREE_GIB * 1024**3
    pause_path = ROOT / 'logs/storage_guard_pause.json'
    pause_stamp = pause_path.stat().st_mtime_ns
    state = dict(status='running', originals=originals, guard=guard, results=[],
                 cache_policy='fresh local cache per case; remote CDN cache uncontrolled',
                 small_asset=SMALL, large_asset=LARGE,
                 start_free_gib=START_FREE_GIB, run_free_gib=RUN_FREE_GIB)
    stopped = []
    deadline = time.monotonic() + 540

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    atomic(OUT / 'state.json', state)
    try:
        for p in originals:
            assert identity(p['pid'])['ticks'] == p['ticks']
            os.kill(p['pid'], signal.SIGSTOP)
            stopped.append(p)
        print('Temporarily paused existing transfers; bounded benchmark deadline 540s', flush=True)
        time.sleep(2)
        extras = json.loads(os.environ.get('OMNI_BENCH_EXTRA_CASES', '[]'))
        cases = [] if extras else [('xet_default', 'xet', None), ('http_1', 'http', 1), ('xet_4', 'xet', 4),
                                  ('http_4', 'http', 4), ('xet_1', 'xet', 1), ('http_16', 'http', 16)]
        small = []
        for name, mode, n in cases:
            limit = min(55, deadline - time.monotonic())
            if limit < 5:
                break
            small.append(run_case('small_' + name, mode, n, SMALL, limit, state))
        xets = [x for x in small if x['mode'] == 'xet']
        if extras:
            large_cases = extras
        else:
            best = max(xets, key=lambda x: x.get('payload_MBps', x['observed_tcp_MBps']))
            large_cases = [('xet_best', 'xet', best['concurrency'], 45), ('http_1', 'http', 1, 45),
                           ('http_4', 'http', 4, 45), ('http_16', 'http', 16, 45)]
        for name, mode, n, cap in large_cases:
            limit = min(cap, deadline - time.monotonic())
            if limit < 5:
                break
            run_case('large_' + name, mode, n, LARGE, limit, state)
        state['status'] = 'completed'
    except BaseException as error:
        state['status'] = 'interrupted'
        state['error_type'] = type(error).__name__
        print('BENCHMARK STOPPED ' + type(error).__name__, flush=True)
    finally:
        restore = pause_path.stat().st_mtime_ns == pause_stamp and free_bytes() >= 250 * 1024**3
        try:
            restore = restore and identity(guard['pid'])['ticks'] == guard['ticks']
        except FileNotFoundError:
            restore = False
        state['restored'], state['left_paused'] = [], []
        for p in stopped:
            try:
                current = identity(p['pid'])
                if restore and current['ticks'] == p['ticks'] and current['cmd'] == p['cmd']:
                    os.kill(p['pid'], signal.SIGCONT)
                    state['restored'].append(p['pid'])
                else:
                    state['left_paused'].append(p['pid'])
            except FileNotFoundError:
                pass
        atomic(OUT / 'state.json', state)
        print('FINISHED ' + json.dumps(dict(status=state['status'], restored=state['restored'],
                                           left_paused=state['left_paused'])), flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        try:
            worker(json.loads(Path(sys.argv[2]).read_text()))
        except BaseException as error:
            cfg = json.loads(Path(sys.argv[2]).read_text())
            response = getattr(error, 'response', None)
            atomic(Path(cfg['directory']) / 'error.json', dict(
                error_type=type(error).__name__,
                http_status=getattr(response, 'status_code', None)))
            sys.exit(2)
    else:
        main()
