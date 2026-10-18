#!/usr/bin/env python3
"""Gate only the exact existing LTX downloader on shared-disk headroom."""
import json
import os
import signal
import time
from pathlib import Path

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')
PID, TICKS = 184791, 2416459204
EXPECTED = ('/usr/bin/python /usr/local/bin/hf download diffusers/LTX-2.3-Diffusers '
            '--revision 8eee8edcf067e838b843f926ec4d4cc9b2be1aaf --cache-dir ' + str(ROOT / 'cache/hf/hub'))
stopping = False


def process():
    p = Path(f'/proc/{PID}')
    raw = (p / 'stat').read_text()
    fields = raw[raw.rfind(')') + 2:].split()
    cmd = (p / 'cmdline').read_bytes().replace(b'\0', b' ').decode().strip()
    assert int(fields[19]) == TICKS and cmd == EXPECTED, 'LTX downloader identity changed'
    return fields[0]


def stop(signum, frame):
    global stopping
    stopping = True


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
path = ROOT / 'logs/ltx_space_gate.json'
previous = None


def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except OSError:
        pass


def pause_exact():
    if process() != 'T':
        os.kill(PID, signal.SIGSTOP)

try:
    while not stopping:
        state = process()
        try:
            disk = os.statvfs(ROOT)
            free = disk.f_bavail * disk.f_frsize
            if free < 250 * 1024**3 and state != 'T':
                os.kill(PID, signal.SIGSTOP)
                state = 'T'
            elif free >= 300 * 1024**3 and state == 'T':
                os.kill(PID, signal.SIGCONT)
                state = 'R'
            record = dict(pid=PID, start_ticks=TICKS, state=state, free_bytes=free,
                          pause_below_gib=250, resume_at_gib=300,
                          utc=time.strftime('%FT%TZ', time.gmtime()))
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(record, indent=2) + '\n')
            temporary.replace(path)
            if (state == 'T') != previous:
                safe_print(record['utc'], 'LTX paused' if state == 'T' else 'LTX resumed',
                      'free_GiB', free // 1024**3, flush=True)
                previous = state == 'T'
        except OSError:
            # A full/unavailable shared disk must not kill the gate or allow writes.
            pause_exact()
        time.sleep(5)
except FileNotFoundError:
    safe_print('existing LTX downloader exited', flush=True)
finally:
    if stopping:
        try:
            if process() != 'T':
                os.kill(PID, signal.SIGSTOP)
            safe_print('gate stopped; left exact LTX downloader paused', flush=True)
        except FileNotFoundError:
            pass
