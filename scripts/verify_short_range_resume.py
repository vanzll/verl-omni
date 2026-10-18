"""Real HTTP range/resume/whole-SHA check with old larger cached chunks."""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import time

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')
work = Path(tempfile.mkdtemp(prefix='short-range-check-', dir=ROOT / 'tmp'))
data = hashlib.sha256(str(time.time_ns()).encode()).digest() * (1024**2)
sha = hashlib.sha256(data).hexdigest()
parts = ROOT / 'cache/http_ranges' / sha
parts.mkdir()
(parts / 'identity.json').write_text(json.dumps({'size': len(data), 'sha256': sha}))
(parts / 'chunk-000000.done').write_bytes(data[:8 * 1024**2])
prior = 5 * 1024**2 // 2
(parts / 'chunk-000001.partial').write_bytes(data[8 * 1024**2:8 * 1024**2 + prior])
requests_seen = []
injected = threading.Event()
guard = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        match = re.fullmatch(r'bytes=([0-9]+)-([0-9]+)', self.headers['Range'])
        first, last = map(int, match.groups())
        with guard:
            requests_seen.append((first, last))
            truncate = not injected.is_set()
            injected.set()
        self.send_response(206)
        self.send_header('Content-Range', f'bytes {first}-{last}/{len(data)}')
        self.send_header('Content-Length', str(last - first + 1))
        self.end_headers()
        if truncate:
            self.wfile.write(data[first:first + (last - first + 1) // 2])
            self.wfile.flush()
            self.close_connection = True
        else:
            self.wfile.write(data[first:last + 1])


server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
target = work / 'verified.bin'
manifest = work / 'manifest.json'
manifest.write_text(json.dumps([{'name': 'local-resume-check', 'size': len(data), 'sha256': sha,
                               'destination': str(target), 'url': f'http://127.0.0.1:{server.server_port}/data'}]))
state_path = work / 'state.json'
try:
    result = subprocess.run(['/usr/bin/python', str(ROOT / 'scripts/fast_download.py'), str(manifest),
                             '--state', str(state_path), '--files', '1', '--connections', '4',
                             '--request-mib', '1', '--min-free-gib', '16', '--resume-free-gib', '17'],
                            timeout=75, capture_output=True, text=True)
    (work / 'helper.log').write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.returncode
    state = json.loads(state_path.read_text())
    assert state['status'] == 'completed' and not state['failed'] and len(state['verified']) == 1
    assert hashlib.sha256(target.read_bytes()).hexdigest() == sha
    assert state['transfer_bytes'] == len(data) - 8 * 1024**2 - prior
    assert state['retry_events'] >= 1
    assert max(last - first + 1 for first, last in requests_seen) <= 1024**2
    assert min(first for first, last in requests_seen) >= 8 * 1024**2
    report = {'status': 'passed', 'requests': len(requests_seen), 'retry_causes': state['retry_causes'],
              'verified_bytes': len(data), 'preserved_prior_bytes': 8 * 1024**2 + prior,
              'max_http_range_bytes': max(last - first + 1 for first, last in requests_seen), 'work': str(work)}
    (ROOT / 'logs/short_range_resume_check.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
finally:
    server.shutdown()
    server.server_close()
