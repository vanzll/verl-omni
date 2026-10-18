"""Bounded task-local sync/mmap/rename and actual cross-process lock probe."""
import fcntl
import hashlib
import json
import mmap
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')
if len(sys.argv) == 2:
    path = Path(sys.argv[1]).resolve()
    assert path.is_relative_to(ROOT / 'tmp')
    with path.open('rb') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit(0)
    sys.exit(9)

host = socket.gethostname().split('.')[0]
directory = ROOT / 'tmp' / ('fs_integrity_' + host + '_' + uuid.uuid4().hex[:8])
directory.mkdir()
results = []
for trial in range(3):
    start = time.monotonic()
    path = directory / ('data_' + str(trial))
    payload = os.urandom(1024 * 1024)
    with path.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fdatasync(stream.fileno())
    with path.open('rb') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        child = subprocess.run([sys.executable, __file__, str(path)], capture_output=True, text=True, timeout=5)
        assert child.returncode == 0, (child.returncode, child.stderr)
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            assert hashlib.sha256(mapped).digest() == hashlib.sha256(payload).digest()
    renamed = path.with_suffix('.renamed')
    path.rename(renamed)
    assert renamed.read_bytes() == payload
    renamed.unlink()
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    os.fsync(descriptor)
    os.close(descriptor)
    results.append({'trial': trial, 'seconds': time.monotonic() - start, 'all_operations_passed': True})
    print(json.dumps({'host': host, 'trial': trial, 'stage': 'fs_passed'}), flush=True)
    time.sleep(2)
for path in sorted((ROOT / 'env/lib/python3.12/site-packages/onnxruntime/capi').glob('*.so*')):
    with path.open('rb') as stream:
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            digest = hashlib.sha256(mapped).hexdigest()
    print(json.dumps({'host': host, 'stage': 'native_so_mmap_passed', 'file': path.name, 'sha256': digest}), flush=True)
directory.rmdir()
print(json.dumps({'host': host, 'stage': 'passed', 'trials': results,
                  'scope': 'short bounded integrity/lock/mmap observation; not a sustained health guarantee'}), flush=True)
