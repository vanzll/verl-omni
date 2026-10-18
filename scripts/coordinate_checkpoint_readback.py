"""Own one prepared world16 GPU readback and stop only its exact peer on failure."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from coordinate_pair import ROOT, control, remote, write_state
from shared_storage_health import MODE, validate_manifest, check_write_health

REQUIRED_CHECKS = {
    'first_load_all_named_adapters_exact', 'first_load_adam_state_exact',
    'first_load_scheduler_exact', 'first_load_all_worker_rng_exact',
    'reload_all_named_adapters_exact', 'reload_adam_state_exact',
    'reload_scheduler_exact', 'reload_all_worker_rng_exact',
    'fixed_input_old_prediction_bitwise', 'fixed_input_current_prediction_bitwise',
}


def check_reports(manifest):
    reports = {}
    for node in manifest['nodes']:
        for file in Path(node['output']).glob('checkpoint_gpu_rank_*.json'):
            report = json.loads(file.read_text())
            rank = report['rank']
            assert rank not in reports and 0 <= rank < 16
            assert report['world_size'] == 16
            assert rank // 8 == node['node_rank']
            assert report['probe_attempt'] == manifest['attempt'] and report['probe_nonce'] == node['nonce']
            assert report['source_sha256'] == manifest['source_sha256']
            assert report['config_sha256'] == manifest['config_sha256']
            assert report['checkpoint'] == manifest['checkpoint']
            assert report['physical_visible_device'] == str(rank % 8)
            assert report['launcher_local_rank'] == str(rank % 8)
            assert report['optimizer_step_calls'] == report['scientific_optimizer_updates'] == 0
            assert report['stage'] == 'passed' and REQUIRED_CHECKS <= report['checks'].keys()
            assert all(value is True for value in report['checks'].values())
            reports[rank] = str(file)
    assert set(reports) == set(range(16)), 'all16 actual GPU rank reports required'
    return reports


def check_checkpoint_admission(manifest, proof):
    assert proof['checkpoint_integrity_verified'] is True and proof['gpu_readback_verified'] is False
    assert proof['world_size'] == 16
    assert proof['source_sha256'] == manifest['source_sha256']
    assert proof['config_sha256'] == manifest['config_sha256']
    assert proof['checkpoint'] == manifest['checkpoint']
    assert hashlib.sha256(Path(proof['config']).read_bytes()).hexdigest() == manifest['config_sha256']
    assert hashlib.sha256((Path(manifest['checkpoint']) / 'checkpoint_manifest.json').read_bytes()).hexdigest() == manifest['checkpoint_manifest_sha256']


def main(path, admission=check_checkpoint_admission, reports=check_reports,
         finished_status='finished_verified_world16_gpu_readback'):
    path = path.resolve()
    assert path.is_relative_to(ROOT / 'preflight/runtime_gpu')
    manifest = json.loads(path.read_text())
    output = Path(manifest['output'])
    assert output == path.parent and manifest['world_size'] == 16 and len(manifest['nodes']) == 2
    lock = (output / 'coordinator.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = output / 'coordinator_state.json'
    assert not state_path.exists(), 'probe already started; inspect the same owners'
    proof_path = Path(manifest['cpu_proof'])
    assert hashlib.sha256(proof_path.read_bytes()).hexdigest() == manifest['cpu_proof_sha256']
    proof = json.loads(proof_path.read_text())
    admission(manifest, proof)
    no_reserve = manifest.get('storage_admission_mode') == MODE
    if no_reserve:
        validate_manifest(manifest, manifest['storage_policy'], 'minimum_start_free_gib')
        check_write_health()
    else:
        stat = os.statvfs(ROOT)
        assert stat.f_bavail * stat.f_frsize >= manifest['minimum_start_free_gib'] * 2**30
    for node in manifest['nodes']:
        assert hashlib.sha256(Path(node['env_file']).read_bytes()).hexdigest() == node['env_sha256']
        launch = Path(node['launch_manifest'])
        assert hashlib.sha256(launch.read_bytes()).hexdigest() == node['launch_manifest_sha256']
        for name, expected in json.loads(launch.read_text()).items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
        command = 'source /m2v_intern2/zhenglin/my_bashrc.sh\n' + shlex.join([
            'timeout', '25', '/usr/bin/python', str(ROOT / 'scripts/inspect_gpu_node.py')])
        result = remote(node, command)
        assert result.returncode == 0
        audit = json.loads(result.stdout)
        assert audit['hostname'] == node['hostname'] and len(audit['gpus']) == len(audit['compute_apps']) == 8
        assert len(audit['processes']) == 1
        assert all(audit['processes'][0][key] == node['planned_holder'][key]
                   for key in ('pid', 'start_ticks', 'pgid', 'cmd'))
        assert audit['lease_owner_process'] is None and not audit['lease_live_group_members']
        (Path(node['output']) / 'prelaunch_audit.json').write_text(result.stdout)
    state = {'attempt': manifest['attempt'], 'pid': os.getpid(),
             'pid_start_ticks': Path('/proc/self/stat').read_text().rsplit(') ', 1)[1].split()[19],
             'status': 'starting', 'started_unix': time.time(), 'stop_sent': [], 'nodes': {}}
    write_state(state_path, state, manifest['nodes'])
    for node in manifest['nodes']:
        command = 'source /m2v_intern2/zhenglin/my_bashrc.sh\n' + shlex.join([
            'env', 'LD_LIBRARY_PATH=', 'tmux', '-S', node['tmux_socket'], 'new-session', '-d',
            '-s', node['tmux_session'], '-c', str(ROOT), 'bash', str(ROOT / 'scripts/launch_probe_owner.sh'), node['env_file']])
        try:
            result = remote(node, command)
            state.setdefault('launch_observations', {})[node['hostname']] = result.returncode
        except (OSError, subprocess.SubprocessError):
            state.setdefault('launch_observations', {})[node['hostname']] = 'observation_timeout'
        write_state(state_path, state, manifest['nodes'])
    first_terminal = None
    while True:
        rows = {}
        def observe(node):
            try:
                return node['hostname'], control(node, 'status')
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                return node['hostname'], {'observation_error': type(error).__name__}
        with ThreadPoolExecutor(max_workers=2) as pool:
            rows.update(pool.map(observe, manifest['nodes']))
        state.update(status='running', nodes=rows, observed_unix=time.time())
        terminals = [row.get('terminal') for row in rows.values()]
        if all(terminals):
            if all(row['terminal']['exit_code'] == 0 and row.get('occupancy_verified') for row in rows.values()):
                try:
                    state['rank_reports'] = reports(manifest)
                except (AssertionError, KeyError, TypeError, ValueError, OSError) as error:
                    state['status'] = 'failed_report_verification'
                    state['report_verification_error'] = type(error).__name__
                    write_state(state_path, state, manifest['nodes'])
                    return 1
                state['status'] = finished_status
                write_state(state_path, state, manifest['nodes'])
                return 0
            state['status'] = 'failed_or_requires_occupancy_recovery'
            write_state(state_path, state, manifest['nodes'])
            return 1
        if any(terminals) and first_terminal is None:
            first_terminal = time.time()
        failure = any(item and item['exit_code'] != 0 for item in terminals)
        expired = time.time() - state['started_unix'] > manifest['timeout_seconds']
        peer_grace = first_terminal is not None and time.time() - first_terminal > 180
        try:
            stat = os.statvfs(ROOT)
            state['free_bytes'] = stat.f_bavail * stat.f_frsize
            if no_reserve:
                check_write_health()
            low_space = (not no_reserve and
                         state['free_bytes'] < manifest['dynamic_stop_free_gib'] * 2**30)
        except OSError as error:
            state['storage_observation_error'] = type(error).__name__
            low_space = True  # Actual storage observation/write failure; stop exact owners.
        if failure or expired or peer_grace or low_space:
            state['status'] = 'stopping_owned_peer'
            state['stop_reason'] = {'peer_failed': failure, 'probe_execution_budget_expired': expired,
                                    'peer_completion_grace_expired': peer_grace, 'low_space': low_space}
            for node in manifest['nodes']:
                if rows[node['hostname']].get('live') and node['hostname'] not in state['stop_sent']:
                    try:
                        control(node, 'stop')
                        state['stop_sent'].append(node['hostname'])
                    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                        state.setdefault('stop_errors', {})[node['hostname']] = type(error).__name__
        write_state(state_path, state, manifest['nodes'])
        time.sleep(15)


if __name__ == '__main__':
    raise SystemExit(main(Path(sys.argv[1])))
