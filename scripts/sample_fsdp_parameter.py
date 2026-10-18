"""Sample one owned actor's FSDP initialization position without logging other locals."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')

parser = argparse.ArgumentParser()
parser.add_argument('pid', type=int)
parser.add_argument('start_ticks')
parser.add_argument('attempt')
args = parser.parse_args()
assert args.pid > 1 and '/' not in args.attempt
proc = Path('/proc') / str(args.pid)
fields = (proc / 'stat').read_text().rsplit(') ', 1)[1].split()
assert fields[19] == args.start_ticks and fields[0] != 'Z'
environment = dict(row.split('=', 1) for row in (proc / 'environ').read_bytes().decode(
    errors='replace').split('\0') if '=' in row)
assert environment.get('OMNI_ATTEMPT') == args.attempt
assert (proc / 'cmdline').read_bytes().startswith(b'ray::WorkerDict')
result = subprocess.run([str(ROOT / 'env/bin/py-spy'), 'dump', '--pid', str(args.pid),
                         '--json', '--locals'], capture_output=True, text=True, timeout=18)
report = {'unix': time.time(), 'pid': args.pid, 'start_ticks': args.start_ticks,
          'attempt': args.attempt, 'returncode': result.returncode, 'frames': []}
if result.returncode == 0:
    for thread in json.loads(result.stdout):
        for frame in thread['frames']:
            if frame['name'] == 'apply_fsdp2' and frame['filename'].endswith('/verl_omni/utils/fsdp_utils.py'):
                values = [row for row in frame.get('locals') or []
                          if isinstance(row, dict) and row.get('name') == 'idx'
                          and re.fullmatch(r'\d{1,6}', row.get('repr', ''))]
                report['frames'].append({'tid': thread['os_thread_id'], 'line': frame['line'],
                                         'wrap_module_index': int(values[0]['repr']) if values else None})
            if frame['name'] == '_broadcast_tensors':
                # Only parameter names in this exact PyTorch function are retained.
                values = [row for row in frame.get('locals') or []
                          if isinstance(row, dict) and row.get('name') == 'key']
                report['frames'].append({'tid': thread['os_thread_id'], 'line': frame['line'],
                                         'parameter': values[0].get('repr') if values else None})
        if thread['active'] and thread['frames']:
            report.setdefault('active_functions', []).append(thread['frames'][0]['name'])
# No arbitrary local variables, HTTP responses, environment, or raw dump is written.
output = ROOT / 'attempts' / args.attempt / 'logs/fsdp_parameter_progress.jsonl'
with output.open('a') as file:
    file.write(json.dumps(report) + '\n')
print(json.dumps(report))
