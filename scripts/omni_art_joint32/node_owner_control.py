# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Local node identity/status/signal operations for one explicit attempt."""
import json
import hashlib
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(os.environ['OMNI_RUNTIME_ROOT'])
CONTROL = Path(__file__).resolve().parent
CAMPAIGN = Path(os.environ['OMNI_CAMPAIGN_ROOT'])


def atomic(path, value):
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def process(pid):
    try:
        directory = Path('/proc') / str(pid)
        fields = (directory / 'stat').read_text().rsplit(') ', 1)[1].split()
        return {'pid': int(pid), 'ticks': fields[19], 'state': fields[0],
                'argv': [v.decode() for v in (directory / 'cmdline').read_bytes().split(b'\0') if v]}
    except (OSError, ProcessLookupError):
        return None


def main():
    action, descriptor = sys.argv[1:3]
    node_file = Path(descriptor)
    assert node_file.resolve().is_relative_to(ROOT)
    node = json.loads(node_file.read_text())
    assert socket.gethostname().split('.')[0] == node['hostname'], 'wrong node'
    output = Path(node['output'])
    assert output.resolve().is_relative_to(ROOT)
    identity_file = output / 'logs' / f'owner_identity_{node["hostname"]}.json'
    exit_file = output / 'logs' / f'owner_exit_{node["hostname"]}.json'
    if action == 'identity':
        # The launcher records its child immediately, before slow source checks.
        owner = process(int(sys.argv[3]))
        assert owner and owner['state'] != 'Z'
        assert Path(owner['argv'][0]).name == 'bash'
        assert owner['argv'][1:] == [str(CONTROL / 'run_node_owner.sh'), node['env_file']]
        record = {'pid': owner['pid'], 'start_ticks': owner['ticks'], 'attempt': node['attempt'],
                  'nonce': node['nonce'], 'host': node['hostname'], 'created_unix': time.time()}
        temporary = identity_file.with_name(identity_file.name + f'.tmp.{os.getpid()}')
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        os.link(temporary, identity_file)  # refuses a duplicate owner identity
        temporary.unlink()
        return
    if action == 'verify-self':
        assert hashlib.sha256(Path(node['env_file']).read_bytes()).hexdigest() == node['env_sha256']
        identity = json.loads(identity_file.read_text())
        assert identity['pid'] == int(sys.argv[3])
        assert identity['attempt'] == node['attempt'] and identity['nonce'] == node['nonce']
        current = process(identity['pid'])
        assert current and current['ticks'] == identity['start_ticks']
        launch = Path(node['launch_manifest'])
        assert hashlib.sha256(launch.read_bytes()).hexdigest() == node['launch_manifest_sha256']
        for name, digest in json.loads(launch.read_text()).items():
            file = Path(name)
            assert file.is_absolute()
            assert hashlib.sha256(file.read_bytes()).hexdigest() == digest, name
        return
    if action == 'exit':
        identity = json.loads(identity_file.read_text()) if identity_file.exists() else None
        atomic(exit_file, {'attempt': node['attempt'], 'nonce': node['nonce'],
                          'exit_code': int(sys.argv[3]), 'finished_unix': time.time(), 'identity': identity})
        return
    identity = json.loads(identity_file.read_text()) if identity_file.exists() else None
    if identity:
        assert identity['attempt'] == node['attempt'] and identity['nonce'] == node['nonce']
    current = process(identity['pid']) if identity else None
    live = bool(current and current['ticks'] == identity['start_ticks'] and current['state'] != 'Z')
    if live:
        assert Path(current['argv'][0]).name == 'bash'
        assert current['argv'][1:] == [str(CONTROL / 'run_node_owner.sh'), node['env_file']]
    if action == 'stop':
        if live:
            handle = os.pidfd_open(identity['pid'])
            try:
                current = process(identity['pid'])
                assert current and current['ticks'] == identity['start_ticks']
                signal.pidfd_send_signal(handle, signal.SIGTERM)
            finally:
                os.close(handle)
        print(json.dumps({'host': node['hostname'], 'signal_sent': live}))
    elif action == 'status':
        terminal = json.loads(exit_file.read_text()) if exit_file.exists() else None
        if terminal:
            assert terminal['attempt'] == node['attempt'] and terminal['nonce'] == node['nonce']
        result = {'host': node['hostname'], 'live': live, 'identity': identity, 'terminal': terminal}
        native_env = os.environ.copy()
        native_env['LD_LIBRARY_PATH'] = ''
        pane = subprocess.run(['tmux', '-S', node['tmux_socket'], 'list-panes', '-t', node['tmux_session'],
                               '-F', '#{pane_pid}'], timeout=5, env=native_env, capture_output=True, text=True)
        result['launcher_session_present'] = pane.returncode == 0
        if terminal and not live:
            apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                           '--format=csv,noheader,nounits'], timeout=5, text=True).split()
            holders = sorted(set(apps))
            valid = len(apps) == 8 and len(holders) == 1
            holder = process(int(holders[0])) if valid else None
            valid = bool(holder and holder['argv'] == ['/usr/bin/python', '/m2v_intern2/zhenglin/gpu_occupy.py']
                         and (Path('/proc') / str(holder['pid']) / 'cwd').resolve() == CAMPAIGN)
            result['occupancy_verified'] = valid
        print(json.dumps(result))
    else:
        raise ValueError('invalid local owner operation')


if __name__ == '__main__':
    main()
