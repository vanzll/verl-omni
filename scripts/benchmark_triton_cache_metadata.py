"""CPU-only metadata contention diagnostic; uses a fresh, tiny private test tree."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time

from triton import knobs
from triton.runtime.cache import FileCacheManager

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')
parser = argparse.ArgumentParser()
parser.add_argument('--start-unix', type=float)
parser.add_argument('--control-file', type=Path)
parser.add_argument('--label', required=True)
args = parser.parse_args()
assert args.label.replace('_', '').isalnum()
assert (args.start_unix is None) != (args.control_file is None)
host = os.uname().nodename.split('.')[0]
base = ROOT / 'tmp' / ('triton_metadata_' + args.label)
knobs.cache.dir = str(base)
results = {'host': host, 'unix': time.time(), 'writes_per_thread': 3,
           'threads_per_node': 4, 'native_method': 'triton.runtime.cache.FileCacheManager.put',
           'modes': {}}
if args.control_file is not None:
    assert args.control_file.resolve().is_relative_to(ROOT / 'tmp')
    (ROOT / 'logs' / f'triton_metadata_{args.label}_{host}.ready').write_text(str(time.time()) + '\n')
    deadline = time.monotonic() + 65
    while not args.control_file.is_file():
        if time.monotonic() > deadline:
            raise TimeoutError('No shared probe start signal')
        time.sleep(0.25)
    args.start_unix = json.loads(args.control_file.read_text())['start_unix']
for mode_index, mode in enumerate(('shared', 'private')):
    managers = [FileCacheManager('shared_kernel' if mode == 'shared' else f'{host}_worker_{i}')
                for i in range(4)]
    start = args.start_unix + mode_index * 17
    if time.time() > start + 2:
        raise RuntimeError('Late metadata probe start; do not report unpaired timing')
    time.sleep(max(0, start - time.time()))

    def write(index):
        manager = managers[index]
        durations = []
        for i in range(3):
            began = time.monotonic()
            manager.put(json.dumps({'host': host, 'thread': index, 'write': i, 'payload': 'x' * 1024}),
                        'probe.json', binary=False)
            durations.append(time.monotonic() - began)
        return {'thread': index, 'seconds': durations}

    began = time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as executor:
        rows = list(executor.map(write, range(4)))
    results['modes'][mode] = {'wall_seconds': time.monotonic() - began, 'threads': rows}
    print(json.dumps({'host': host, 'mode': mode, 'wall_seconds': results['modes'][mode]['wall_seconds']}), flush=True)
    path = ROOT / 'logs' / f'triton_metadata_{args.label}_{host}.json'
    path.write_text(json.dumps(results, indent=2) + '\n')
print(json.dumps({'host': host, 'status': 'completed', 'output': str(path)}), flush=True)
