"""Own one prepared pair's startup and failure cleanup; no experiment queue decisions."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from shared_storage_health import MODE, validate_manifest, check_write_health

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')


def write_state(path, value, nodes=()):
    temporary = path.with_suffix('.tmp')
    try:
        temporary.write_text(json.dumps(value, indent=2) + '\n')
        temporary.replace(path)
    except OSError:
        # A full/unwritable shared disk must not orphan an already launched pair.
        for node in nodes:
            try:
                control(node, 'stop')
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
                pass
        raise


def remote(node, command, timeout=30):
    authorized = {row.split()[0] for row in Path('/etc/mpi/hostfile').read_text().splitlines() if row.strip()}
    if node['ip'] not in authorized:
        raise RuntimeError('node no longer in current allocation; no remote command sent')
    args = ['ssh', '-p', str(node['ssh_port']), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=6',
            '-o', 'StrictHostKeyChecking=yes', '-o', f'UserKnownHostsFile={ROOT}/assets/ssh_known_hosts',
            f'root@{node["ip"]}', command]
    return subprocess.run(args, text=True, capture_output=True, timeout=timeout)


def control(node, action):
    descriptor = str(Path(node['env_file']).with_suffix('.json'))
    command = 'source /m2v_intern2/zhenglin/my_bashrc.sh; ' + shlex.join(
        ['timeout', '20', '/usr/bin/python', str(ROOT / 'scripts/node_owner_control.py'), action, descriptor])
    result = remote(node, command)
    if result.returncode:
        raise RuntimeError(f'{action} failed for {node["hostname"]}:exit{result.returncode}')
    return json.loads(result.stdout)


def main():
    manifest_path = Path(sys.argv[1])
    assert manifest_path.resolve().is_relative_to(ROOT / 'attempts')
    manifest = json.loads(manifest_path.read_text())
    output = Path(manifest['output'])
    lock = (output / 'coordinator.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_file = output / 'coordinator_state.json'
    if state_file.exists():
        raise RuntimeError('attempt already started; inspect its owners rather than restarting')
    launch_file = Path(manifest['launch_manifest'])
    assert hashlib.sha256(launch_file.read_bytes()).hexdigest() == manifest['launch_manifest_sha256']
    for name, digest in json.loads(launch_file.read_text()).items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    no_reserve = manifest.get('storage_admission_mode') == MODE
    if no_reserve:
        validate_manifest(manifest, ROOT / 'assets/compact_checkpoint_storage_policy.json')
        check_write_health()
    else:
        stat = os.statvfs(ROOT)
        assert stat.f_bavail * stat.f_frsize >= manifest['minimum_free_gib'] * 1024**3
    if manifest.get('storage_admission_mode') == 'single_pair_allocation_idle':
        assert manifest['minimum_free_gib'] == 250 and manifest['dynamic_stop_free_gib'] == 128
        # Keep this nonblocking lock for the coordinator lifetime. A second
        # reduced-budget start cannot race; a normal two-pair start may join.
        single_storage_lock = (ROOT / 'tmp/single_pair_storage_admission.lock').open('a')
        fcntl.flock(single_storage_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ports = {'10.83.2.231': 21010, '10.83.1.141': 21010,
                 '10.83.2.97': 21013, '10.83.1.23': 21007}
        authorized = {line.split()[0] for line in Path('/etc/mpi/hostfile').read_text().splitlines() if line.strip()}
        assert authorized == set(ports), 'reduced-budget admission requires a fresh four-node allocation audit'

        def check_allocation_idle(ip):
            command = 'source /m2v_intern2/zhenglin/my_bashrc.sh\n' + shlex.join([
                'timeout', '25', '/usr/bin/python', str(ROOT / 'scripts/inspect_gpu_node.py')])
            result = remote({'ip': ip, 'ssh_port': ports[ip]}, command)
            assert result.returncode == 0
            audit = json.loads(result.stdout)
            assert len(audit['gpus']) == len(audit['compute_apps']) == 8
            assert len(audit['processes']) == 1 and audit['lease_owner_process'] is None
            assert not audit['lease_live_group_members']
            assert audit['processes'][0]['cmd'] == '/usr/bin/python /m2v_intern2/zhenglin/gpu_occupy.py'
            (output / 'logs' / f'single_pair_storage_admission_{audit["hostname"]}.json').write_text(result.stdout)

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(check_allocation_idle, sorted(authorized)))
        stat = os.statvfs(ROOT)
        assert stat.f_bavail * stat.f_frsize >= manifest['minimum_free_gib'] * 1024**3
    snapshot = Path(manifest['source']['SOURCE_SNAPSHOT_DIR'])
    schema_proof = json.loads((snapshot / 'provenance/four_variant_rollout_config_cpu.json').read_text())
    assert schema_proof['status'] == 'passed'
    assert any(row['variant'] == manifest['variant'] and row['status'] == 'passed'
               for row in schema_proof['variants'])
    strategy = snapshot / 'runtime/source/verl_omni/workers/rollout/vllm_rollout/vllm_omni_diffusion_strategy.py'
    assert hashlib.sha256(strategy.read_bytes()).hexdigest() == schema_proof['strategy_sha256']
    # Asset readiness is mandatory before any node owner can release its holder.
    subprocess.run([str(ROOT / 'env/bin/python'), str(ROOT / 'scripts/check_assets.py')], check=True, timeout=60)
    for node in manifest['nodes']:
        assert hashlib.sha256(Path(node['env_file']).read_bytes()).hexdigest() == node['env_sha256']
        command = 'source /m2v_intern2/zhenglin/my_bashrc.sh; ' + shlex.join([
            'timeout', '25', '/usr/bin/python', str(ROOT / 'scripts/inspect_gpu_node.py')])
        result = remote(node, command)
        assert result.returncode == 0
        audit = json.loads(result.stdout)
        assert audit['hostname'] == node['hostname'] and len(audit['compute_apps']) == 8
        assert len(audit['gpus']) == 8 and all('H800' in row and '560.35.03' in row for row in audit['gpus'])
        assert len(audit['processes']) == 1 and audit['lease_owner_process'] is None
        assert not audit['lease_live_group_members']
        assert audit['shared_memory']['free_bytes'] >= manifest.get('ray_object_store_bytes', 6 * 1024**3) + 2 * 1024**3
        assert audit['processes'][0]['pid'] == node['planned_holder']['pid']
        assert audit['processes'][0]['start_ticks'] == node['planned_holder']['start_ticks']
        (output / 'logs' / f'prelaunch_audit_{node["hostname"]}.json').write_text(result.stdout)
    state = {'status': 'starting', 'attempt': manifest['attempt'], 'pid': os.getpid(),
             'pid_start_ticks': Path('/proc/self/stat').read_text().rsplit(') ', 1)[1].split()[19],
             'host': socket.gethostname(), 'started_unix': time.time(), 'nodes': {}, 'stop_sent': []}
    write_state(state_file, state, manifest['nodes'])
    for node in manifest['nodes']:
        command = 'source /m2v_intern2/zhenglin/my_bashrc.sh; ' + shlex.join([
            'env', 'LD_LIBRARY_PATH=', 'tmux', '-S', node['tmux_socket'], 'new-session', '-d',
            '-s', node['tmux_session'], '-c', str(ROOT), 'bash',
            str(ROOT / 'scripts/launch_training_owner.sh'), node['env_file']])
        try:
            result = remote(node, command)
            state['nodes'][node['hostname']] = {'launch_returncode': result.returncode}
        except subprocess.TimeoutExpired:
            # SSH timeout is not proof that the remote startup failed. Poll the same identity.
            state['nodes'][node['hostname']] = {'launch_observation': 'timeout'}
        write_state(state_file, state, manifest['nodes'])
    state['status'] = 'running'
    normal_finish_seen = None
    dead_since = {}
    while True:
        rows = {}
        for node in manifest['nodes']:
            try:
                rows[node['hostname']] = control(node, 'status')
            except (OSError, subprocess.SubprocessError, ValueError, RuntimeError) as error:
                rows[node['hostname']] = {'observation_error': type(error).__name__}
        state.update(nodes=rows, observed_unix=time.time())
        stop_floor_gib = manifest.get('dynamic_stop_free_gib', 250)
        storage_sample = {'unix': time.time(), 'stop_floor_gib': stop_floor_gib}
        storage_failed = False
        try:
            storage = os.statvfs(ROOT)
            state['free_bytes'] = storage.f_bavail * storage.f_frsize
            state['minimum_observed_free_bytes'] = min(state.get('minimum_observed_free_bytes', state['free_bytes']),
                                                     state['free_bytes'])
            storage_sample.update(available_bytes=state['free_bytes'], free_bytes=storage.f_bfree * storage.f_frsize,
                                  total_bytes=storage.f_blocks * storage.f_frsize)
            if no_reserve:
                check_write_health()
            with (output / 'logs/free_space_samples.jsonl').open('a') as stream:
                stream.write(json.dumps(storage_sample) + '\n')
        except OSError as error:
            state['storage_history_write_error'] = type(error).__name__
            storage_failed = True
        low_space = storage_failed or (not no_reserve and state['free_bytes'] < stop_floor_gib * 1024**3)
        if low_space:
            state.setdefault('first_low_space_sample', storage_sample)
            state['failure_reason'] = 'actual shared-storage I/O failure' if storage_failed else f'shared Intern free space below{stop_floor_gib}GiB'
        terminals = [row.get('terminal') for row in rows.values()]
        if all(terminals):
            healthy_exit = all(row['terminal']['exit_code'] == 0 and row.get('occupancy_verified')
                               for row in rows.values())
            state['status'] = 'finished_pending_scientific_verification' if healthy_exit else 'failed'
            write_state(state_file, state, manifest['nodes'])
            return 0 if healthy_exit else 1
        failed = any(value and value['exit_code'] != 0 for value in terminals)
        if any(terminals) and normal_finish_seen is None:
            normal_finish_seen = time.time()
        grace_expired = normal_finish_seen is not None and time.time() - normal_finish_seen > 180
        startup_expired = time.time() - state['started_unix'] > 180
        missing = startup_expired and any(row.get('identity') is None and not row.get('launcher_session_present')
                                         and 'observation_error' not in row for row in rows.values())
        for host, row in rows.items():
            if row.get('identity') and not row.get('live') and not row.get('terminal'):
                dead_since.setdefault(host, time.time())
            else:
                dead_since.pop(host, None)
        dead = any(time.time() - since > 30 for since in dead_since.values())
        if failed or grace_expired or missing or dead or low_space:
            state['status'] = 'stopping_peer_after_failure'
            for node in manifest['nodes']:
                row = rows[node['hostname']]
                if row.get('live') and node['hostname'] not in state['stop_sent']:
                    try:
                        result = control(node, 'stop')
                        state['stop_sent'].append(node['hostname'])
                        state.setdefault('stop_results', {})[node['hostname']] = result
                    except (OSError, subprocess.SubprocessError, ValueError, RuntimeError) as error:
                        state.setdefault('stop_errors', {})[node['hostname']] = type(error).__name__
            if all('observation_error' not in row and not row.get('live')
                   and not row.get('launcher_session_present') for row in rows.values()):
                state['status'] = 'failed_requires_review'
                write_state(state_file, state, manifest['nodes'])
                return 1
        write_state(state_file, state, manifest['nodes'])
        time.sleep(15)


if __name__ == '__main__':
    raise SystemExit(main())
