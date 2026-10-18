"""Sample an owned diffusion child; discard all non-whitelisted Python locals."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')
parser = argparse.ArgumentParser()
parser.add_argument('pid', type=int)
parser.add_argument('parent', type=int)
parser.add_argument('attempt')
parser.add_argument('--probe-output', type=Path)
args = parser.parse_args()
assert args.pid > 1 and args.parent > 1 and '/' not in args.attempt
proc = Path('/proc') / str(args.pid)
stat = (proc / 'stat').read_text().rsplit(') ', 1)[1].split()
assert stat[1] == str(args.parent) and stat[0] != 'Z'
environment = dict(row.split('=', 1) for row in (proc / 'environ').read_bytes().decode(
    errors='replace').split('\0') if '=' in row)
assert environment.get('OMNI_ATTEMPT') == args.attempt
parent = Path('/proc') / str(args.parent)
if args.probe_output is not None:
    probe_output = args.probe_output.resolve()
    assert probe_output.is_relative_to(ROOT / 'preflight/runtime_gpu')
    # /proc/environ describes the initial environment, not later Python
    # os.environ updates. Fence with the inherited nonce and exact live owner.
    descriptor = json.loads((ROOT / 'logs/current_ltx_rollout_probe.json').read_text())
    assert descriptor['attempt'] == args.attempt and descriptor['output'] == str(probe_output)
    assert environment.get('OMNI_NONCE') == descriptor['nonce']
    owner = Path('/proc') / str(descriptor['owner_pid'])
    assert (owner / 'stat').read_text().rsplit(') ', 1)[1].split()[19] == descriptor['owner_start_ticks']
    assert str(probe_output / 'probe_code/verify_ltx_rollout_gpu.py').encode() in (parent / 'cmdline').read_bytes()
    output_root = probe_output
else:
    assert (parent / 'cmdline').read_bytes().startswith(b'ray::vLLMOmniHttpServer')
    output_root = ROOT / 'attempts' / args.attempt
parent_ticks = (parent / 'stat').read_text().rsplit(') ', 1)[1].split()[19]
result = subprocess.run([str(ROOT / 'env/bin/py-spy'), 'dump', '--pid', str(args.pid),
                         '--json', '--locals'], capture_output=True, text=True, timeout=18)
after = (proc / 'stat').read_text().rsplit(') ', 1)[1].split()
assert after[19] == stat[19] and after[1] == stat[1]
assert (parent / 'stat').read_text().rsplit(') ', 1)[1].split()[19] == parent_ticks
report = {'unix': time.time(), 'pid': args.pid, 'parent': args.parent,
          'start_ticks': stat[19], 'parent_start_ticks': parent_ticks,
          'attempt': args.attempt, 'returncode': result.returncode,
          'state': after[0], 'user_ticks': int(after[11]), 'system_ticks': int(after[12]),
          'clock_ticks_per_second': os.sysconf('SC_CLK_TCK'), 'frames': []}
if result.returncode == 0:
    for thread in json.loads(result.stdout):
        if thread['os_thread_id'] != args.pid:
            continue
        for index, frame in enumerate(thread['frames']):
            native = '/ltx2/ltx2_runtime.py' in frame['filename'] and frame['name'] in (
                '_predict_noise_for_step', '_denoise_step')
            cache = frame['filename'].endswith('/triton/runtime/cache.py') and frame['name'] == 'put'
            block = (frame['filename'].endswith('/ltx2/ltx2_transformer.py')
                     and frame['name'] == 'forward' and frame['line'] > 2000)
            if index >= 3 and not native and not cache and not block:
                continue
            item = {'name': frame['name'], 'file': frame['filename'], 'line': frame['line']}
            for value in frame.get('locals') or []:
                name, representation = value.get('name'), value.get('repr', '')
                if native and name == 'index' and re.fullmatch(r'\d{1,4}', representation):
                    item['denoise_index'] = int(representation)
                if block and name == 'block_idx' and re.fullmatch(r'\d{1,3}', representation):
                    item['transformer_block_index'] = int(representation)
                if cache and name == 'filename':
                    filename = representation.strip('"\'')
                    if re.fullmatch(r'[A-Za-z0-9_.-]{1,200}', filename):
                        item['compiler_output_filename'] = filename
                if cache and name == 'filepath':
                    filename = representation.strip('"\'')
                    if filename.startswith(str(ROOT / 'cache') + '/'):
                        item['compiler_output_path'] = filename
            report['frames'].append(item)
try:
    report['process_io'] = {key: int(value) for key, value in
                            (line.split(':', 1) for line in (proc / 'io').read_text().splitlines())}
except FileNotFoundError:
    report['process_io'] = None
output = output_root / 'logs/rollout_initialization_progress.jsonl'
with output.open('a') as file:
    file.write(json.dumps(report) + '\n')
print(json.dumps(report))
