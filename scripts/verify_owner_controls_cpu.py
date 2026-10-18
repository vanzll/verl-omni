"""Adversarial identity checks and real pidfd signal to an owned CPU child only."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from unittest.mock import patch

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')
work = Path(tempfile.mkdtemp(prefix='owner-control-cpu-', dir=ROOT / 'tmp'))
(work / 'logs').mkdir()
descriptor = work / 'node.json'
node = {'hostname': socket.gethostname().split('.')[0], 'attempt': 'cpu-identity-check',
        'nonce': 'cpu-identity-nonce', 'output': str(work), 'env_file': str(work / 'node.env'),
        'tmux_socket': str(work / 'nonexistent.sock'), 'tmux_session': 'cpu-only',
        'launch_manifest': str(work / 'launch.json')}
(work / 'launch.json').write_text('{}')
import hashlib
(work / 'node.env').write_text('# CPU fixture only\n')
node['env_sha256'] = hashlib.sha256((work / 'node.env').read_bytes()).hexdigest()
node['launch_manifest_sha256'] = hashlib.sha256(b'{}').hexdigest()
descriptor.write_text(json.dumps(node))
spec = importlib.util.spec_from_file_location('owner_control', ROOT / 'scripts/node_owner_control.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
real_process = module.process
child = subprocess.Popen(['sleep', '60'])


def labeled_process(pid):
    record = real_process(pid)
    if record and int(pid) == child.pid:
        # Mock only command labels. PID, start ticks and delivered signal are real.
        record['argv'] = ['bash', str(ROOT / 'scripts/run_node_owner.sh'), node['env_file']]
    return record


def call(action, *args):
    output = io.StringIO()
    with patch.object(sys, 'argv', ['control', action, str(descriptor), *map(str, args)]):
        with contextlib.redirect_stdout(output):
            module.main()
    return json.loads(output.getvalue()) if output.getvalue() else None


checks = []
try:
    with patch.object(module, 'process', labeled_process):
        call('identity', child.pid)
        call('verify-self', child.pid)
        identity_file = work / 'logs' / f'owner_identity_{node["hostname"]}.json'
        identity = json.loads(identity_file.read_text())
        corrupt = dict(identity, nonce='different-attempt')
        identity_file.write_text(json.dumps(corrupt))
        try:
            call('stop')
            raise RuntimeError('wrong nonce accepted')
        except AssertionError:
            pass
        assert child.poll() is None
        checks.append('wrong nonce rejected without signal')
        corrupt = dict(identity, start_ticks=str(int(identity['start_ticks']) + 1))
        identity_file.write_text(json.dumps(corrupt))
        assert not call('stop')['signal_sent'] and child.poll() is None
        checks.append('stale PID start rejected without signal')
        identity_file.write_text(json.dumps(identity))
        with patch.object(module, 'process', real_process):
            try:
                call('stop')
                raise RuntimeError('changed command accepted')
            except AssertionError:
                pass
        assert child.poll() is None
        checks.append('changed command rejected without signal')
        (work / 'launch.json').write_text('{"tampered": true}')
        try:
            call('verify-self', child.pid)
            raise RuntimeError('changed launch manifest accepted')
        except AssertionError:
            pass
        checks.append('changed launch manifest rejected')
        (work / 'launch.json').write_text('{}')
        (work / 'node.env').write_text('# changed environment\n')
        try:
            call('verify-self', child.pid)
            raise RuntimeError('changed node environment accepted')
        except AssertionError:
            pass
        checks.append('changed node environment rejected')
        assert call('stop')['signal_sent']
        assert child.wait(timeout=5) == -15
        checks.append('matching CPU child receives real pidfd SIGTERM')
    result = {'status': 'passed', 'scope': 'CPU child; command labels mocked; no GPU or real lease mutation',
              'checks': checks, 'work': str(work)}
    (ROOT / 'logs/owner_controls_cpu_result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
finally:
    if child.poll() is None:
        child.terminate()
        child.wait(timeout=5)
