"""Own a bounded native two-update diagnostic on a pair whose parent exited."""
import argparse
from datetime import datetime, UTC
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

from coordinate_pair import ROOT
from shared_storage_health import validate_policy, MODE
from coordinate_checkpoint_readback import main as run_pair


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replicated_cpu_oracle():
    directory = ROOT / 'preflight/runtime_gpu/fsdp_cycle_cpu_contract_v9_replicated/hsdp_gloo_oracle'
    files = sorted(directory.glob('rank_*.json'))
    assert len(files) == 4
    found = set()
    for file in files:
        row = json.loads(file.read_text())
        assert row['status'] == 'passed_actual_cpu_gloo_hsdp_global_gradient_clip_adam_oracle'
        assert row['world_size'] == row['replicate_world_size'] == 4
        assert row['mesh_shape'] == [4, 1] and row['shard_world_size'] == 1
        assert row['cuda_initialized'] is False and row['gpu_verified'] is False
        assert row['cpu_only_native_clip_device_route_override'] == 'cpu'
        assert row['script_sha256'] == digest(ROOT / 'scripts/check_hsdp_replicated_cpu.py')
        assert len(row['steps']) == 3 and [x['step'] for x in row['steps']] == [1, 2, 3]
        for step in row['steps']:
            assert all(math.isfinite(value) for value in step.values())
            assert step['gradient_max_abs_error'] <= 2e-8
            assert step['parameter_max_abs_error'] <= 2e-8
            assert step['adam_state_max_abs_error'] <= 2e-8
            assert math.isclose(step['native_clipping_norm'], step['reference_norm'], rel_tol=2e-5, abs_tol=2e-8)
        assert row['rank'] not in found
        found.add(row['rank'])
    assert found == set(range(4))
    return {str(file): digest(file) for file in files}


def admission(manifest, proof):
    assert manifest['probe_kind'] == 'actor_cycle' and manifest['world_size'] == 16
    assert proof['status'] == 'passed_cpu_fsdp_cycle_contract'
    assert proof['world_size'] == 16 and proof['gpu_verified'] is False
    assert proof['cuda_initialized'] is False and proof['gpu_checkpoint_writes'] is False
    assert proof['scientific_optimizer_updates'] == 0
    assert proof['cpu_native_metric_postprocess_verified'] is True
    assert proof['cpu_native_worker_body_executed'] is True
    assert proof['cpu_boundary_event_contract_verified'] is True
    assert proof['cpu_loss_avg_sum_contract_verified'] is True
    assert proof['cpu_native_postprocess_device_restore_verified'] is True
    assert proof['cpu_local_postprocess_no_collectives_restore_verified'] is True
    assert proof['cpu_adamw_backend_switch_verified'] is True
    assert proof['cpu_optimizer_backend_replay_contract_verified'] is True
    assert proof['optimizer_replay_helper_sha256'] == digest(ROOT / 'scripts/optimizer_backend_equivalence.py')
    assert proof['cpu_native_mesh_factory_and_actor_dp_verified'] is True
    assert proof['cpu_ltx_train_mode_hook_is_inherited_noop'] is True
    assert proof['requested_process_group_backend'] == 'cpu:gloo,cuda:nccl'
    assert proof['diagnostic_local_samples'] == 2 and proof['diagnostic_timesteps'] == 3
    assert proof['source_sha256'] == manifest['source_sha256']
    assert proof['script_sha256'] == digest(ROOT / 'scripts/verify_fsdp_cycle_gpu.py')
    assert proof['shared_helper_sha256'] == digest(ROOT / 'scripts/verify_actor_gpu.py')
    assert proof['config_sha256'] == digest(ROOT / 'preflight/configs/A0/resolved.yaml')
    storage_path = Path(manifest['storage_policy'])
    assert storage_path == ROOT / 'assets/actor_cycle_diagnostic_storage_policy.json'
    assert digest(storage_path) == manifest['storage_policy_sha256']
    storage = json.loads(storage_path.read_text())
    assert storage['kind'] == 'actor_cycle_no_checkpoint_diagnostic_only'
    assert storage['gpu_checkpoint_writes'] is False and storage['model_downloads_permitted'] is False
    validate_policy(storage)
    assert manifest['storage_admission_mode'] == MODE
    assert manifest['minimum_start_free_gib'] == storage['minimum_start_free_gib'] == 0
    assert manifest['dynamic_stop_free_gib'] == storage['dynamic_stop_free_gib'] == 0
    assert storage['conservative_new_output_budget_bytes'] <= 2**30
    env = manifest['diagnostic_environment']
    assert 'NCCL_NET' not in env or env['NCCL_NET'] == 'Socket'
    assert env['OMNI_CYCLE_OPTIMIZER_REPLAY'] in ('0','1')
    base_env = {key: value for key, value in env.items() if key not in ('NCCL_NET','OMNI_CYCLE_OPTIMIZER_REPLAY')}
    assert base_env in [dict(NCCL_P2P_DISABLE='1', NCCL_NVLS_ENABLE='0',
                        NCCL_LAUNCH_ORDER_IMPLICIT=order, OMNI_CYCLE_BOUNDARY_DEVICE_SYNC=sync,
                        OMNI_CYCLE_LOSS_AVG_AS_SUM=loss_sum, OMNI_CYCLE_LOSS_REDUCTION_CPU=cpu,
                        OMNI_CYCLE_LOCAL_POSTPROCESS=local, OMNI_CYCLE_ADAMW_FOREACH_OFF=foreach,
                        OMNI_CYCLE_REPLICATED_FSDP2=replicated)
                   for order, sync, loss_sum, cpu, local, foreach, replicated in [
                       ('0','0','0','0','0','0','0'), ('1','0','0','0','0','0','0'),
                       ('0','1','0','0','0','0','0'), ('0','0','1','0','0','0','0'),
                       ('0','0','1','1','0','0','0'), ('0','0','0','0','1','0','0'),
                       ('0','0','0','0','0','1','0'), ('0','0','0','0','0','1','1')]]
    assert proof['boundary_device_sync_between_workers'] is (env['OMNI_CYCLE_BOUNDARY_DEVICE_SYNC'] == '1')
    assert proof['loss_avg_as_sum'] is (env['OMNI_CYCLE_LOSS_AVG_AS_SUM'] == '1')
    assert proof['loss_reduction_on_cpu'] is (env['OMNI_CYCLE_LOSS_REDUCTION_CPU'] == '1')
    assert proof['local_only_postprocess'] is (env['OMNI_CYCLE_LOCAL_POSTPROCESS'] == '1')
    assert proof['adamw_foreach_disabled'] is (env['OMNI_CYCLE_ADAMW_FOREACH_OFF'] == '1')
    replicated = env['OMNI_CYCLE_REPLICATED_FSDP2'] == '1'
    assert proof['replicated_fsdp2'] is replicated
    socket_network = env.get('NCCL_NET') == 'Socket'
    assert proof['requested_nccl_network'] == ('Socket' if socket_network else 'auto')
    optimizer_replay = env['OMNI_CYCLE_OPTIMIZER_REPLAY'] == '1'
    assert proof['gpu_optimizer_backend_replay'] is optimizer_replay
    assert not optimizer_replay or (socket_network and proof['adamw_foreach_disabled'])
    assert not socket_network or proof['adamw_foreach_disabled']
    assert proof['private_engine_config_overrides'] == ({'fsdp_size': 1} if replicated else {})
    assert proof['expected_actor_dp_size'] == 16
    assert proof['expected_mesh_shape'] == ([16, 1] if replicated else [16])
    assert manifest.get('replicated_cpu_global_gradient_oracle') == (replicated_cpu_oracle() if replicated else None)
    expected_foreach = False if proof['adamw_foreach_disabled'] else None
    assert proof['cpu_adamw_execution']['foreach_defaults'] is expected_foreach
    assert proof['cpu_adamw_execution']['foreach_groups'] == [expected_foreach]
    for node in manifest['nodes']:
        lines = Path(node['env_file']).read_text().splitlines()
        for key, value in env.items():
            assert [line for line in lines if line.startswith(f'export {key}=')] == [f'export {key}={value}']
    for name, expected in manifest['coordinator_files_sha256'].items():
        assert digest(ROOT / name) == expected


def reports(manifest):
    found = {}
    proof = json.loads(Path(manifest['cpu_proof']).read_text())
    for node in manifest['nodes']:
        for file in Path(node['output']).glob('fsdp_cycle_rank_*.json'):
            row = json.loads(file.read_text())
            rank = row['rank']
            assert rank not in found and 0 <= rank < 16 and rank // 8 == node['node_rank']
            assert row['world_size'] == 16 and row['local_rank'] == 0
            assert row['launcher_local_rank'] == rank % 8 and row['physical_visible_device'] == str(rank % 8)
            assert row['attempt'] == manifest['attempt'] and row['nonce'] == node['nonce']
            assert row['source_sha256'] == manifest['source_sha256']
            for key in ('script_sha256', 'shared_helper_sha256', 'optimizer_replay_helper_sha256', 'config_sha256', 'capture_hashes', 'torch_version'):
                assert row[key] == proof[key]
            assert row['requested_process_group_backend'] == proof['requested_process_group_backend']
            assert row['actual_process_group_backend'] == 'cpu:gloo,cuda:nccl'
            assert row['object_collective_device'] == 'cpu'
            assert row['requested_nccl_network'] == proof['requested_nccl_network']
            assert row['gpu_optimizer_backend_replay'] is proof['gpu_optimizer_backend_replay']
            if proof['gpu_optimizer_backend_replay']:
                replay = row['optimizer_backend_replay']
                assert replay['status'] == 'completed_same_input_optimizer_backend_replay'
                assert replay['native_optimizer_call'] == 2 and replay['parameter_objects'] > 0
                assert replay['devices'] == ['cuda']
                assert replay['actual_foreach'] is False and replay['reference_foreach'] is True
                assert replay['all_fields_bitwise_equal'] is True
                assert set(replay['comparisons']) == {'parameter','parameter_update','exp_avg','exp_avg_sq','step'}
                for comparison in replay['comparisons'].values():
                    assert comparison['bitwise_equal'] is True and comparison['elements'] > 0
                    assert comparison['max_abs'] == comparison['relative_l2'] == comparison['difference_sum_squares'] == 0
                    assert math.isfinite(comparison['native_sum_squares']) and comparison['native_sum_squares'] >= 0

            if proof['requested_nccl_network'] == 'Socket':
                assert type(row['pid']) is int and row['pid'] > 0
                logs = list(Path(node['output']).glob(f'nccl.*.{row["pid"]}.log'))
                assert len(logs) == 1, 'actual per-rank NCCL network log required'
                network_log = logs[0].read_text()
                assert 'Using network Socket' in network_log, 'requested Socket backend was not verified'
                assert 'Using network NCCL RDMA Plugin' not in network_log
            replicated = proof['replicated_fsdp2']
            assert row['replicated_fsdp2'] is replicated
            assert row['private_engine_config_overrides'] == proof['private_engine_config_overrides']
            assert row['expected_mesh_shape'] == proof['expected_mesh_shape']
            assert row['actual_mesh_shape'] == proof['expected_mesh_shape']
            assert row['actual_mesh_dim_names'] == (['ddp', 'fsdp'] if replicated else ['fsdp'])
            assert row['actor_dp_size'] == row['expected_actor_dp_size'] == 16
            assert row['fsdp_shard_world_size'] == (1 if replicated else 16)
            assert row['fsdp_replicate_world_size'] == (16 if replicated else 1)
            assert type(row['gpu_memory_after_engine_init_bytes']) is int and row['gpu_memory_after_engine_init_bytes'] > 0
            assert type(row['peak_gpu_memory_bytes']) is int
            assert row['peak_gpu_memory_bytes'] >= row['gpu_memory_after_engine_init_bytes']
            assert row['fsdp_group_backend'] in ('nccl', 'cpu:gloo,cuda:nccl')
            assert row['metric_and_fsdp_group_are_same_object'] is (not replicated)
            assert row['metric_and_replicate_group_are_same_object'] is replicated
            if replicated:
                assert row['replicate_group_backend'] == 'cpu:gloo,cuda:nccl'
            else:
                assert row['replicate_group_backend'] is None
            assert row['boundary_device_sync_between_workers'] is proof['boundary_device_sync_between_workers']
            assert row['loss_avg_as_sum'] is proof['loss_avg_as_sum']
            assert row['loss_avg_replacements'] == (2 if proof['loss_avg_as_sum'] else 0)
            assert row['loss_reduction_on_cpu'] is proof['loss_reduction_on_cpu']
            assert row['cpu_loss_postprocess_calls'] == (2 if proof['loss_reduction_on_cpu'] else 0)
            assert row['local_only_postprocess'] is proof['local_only_postprocess']
            assert row['local_logging_postprocess_calls'] == (2 if proof['local_only_postprocess'] else 0)
            assert row['adamw_foreach_disabled'] is proof['adamw_foreach_disabled']
            backend = row['actual_adamw_execution']
            assert backend['optimizer_class'] == 'torch.optim.AdamW'
            assert backend['foreach_disabled'] is proof['adamw_foreach_disabled']
            expected_foreach = False if proof['adamw_foreach_disabled'] else None
            assert backend['foreach_defaults'] is expected_foreach
            assert backend['foreach_groups'] == [expected_foreach]
            assert backend['unchanged_hyperparameters'] == proof['cpu_adamw_execution']['unchanged_hyperparameters']
            assert backend['parameter_count'] > 0 and backend['parameter_dtypes']
            expected_devices = [('cpu' if proof['loss_reduction_on_cpu'] else 'cuda')] * 2 if proof['loss_avg_as_sum'] else []
            assert row['loss_reduce_tensor_devices'] == expected_devices
            boundary = row['boundary_synchronization']
            if proof['boundary_device_sync_between_workers']:
                assert isinstance(boundary, dict), 'completed boundary sync record required'
                assert boundary['stage'] == 'returned'
                assert math.isfinite(boundary['seconds']) and boundary['seconds'] >= 0
            else:
                assert boundary is None
            assert row['stage'] == 'passed_two_native_worker_cycles'
            assert row['scientific_optimizer_updates'] == 0 and row['gpu_checkpoint_writes'] is False
            assert row['diagnostic_optimizer_steps_returned'] == 2
            assert row['environment'] == manifest['diagnostic_environment']
            assert len(row['cycles']) == 2 and [x['cycle'] for x in row['cycles']] == [0, 1]
            for cycle in row['cycles']:
                assert all(math.isfinite(cycle[key]) for key in ('loss', 'finite_grad_norm', 'native_worker_seconds'))
                assert cycle['finite_grad_norm'] >= 0 and cycle['native_worker_seconds'] > 0
                # Unsupported LTX FLOPs can omit the numeric mfu key even
                # though the real counter/metadata collectives ran.
                assert {'loss', 'grad_norm'} <= set(cycle['metric_keys'])
            found[rank] = str(file)
    assert set(found) == set(range(16)), 'all16 native cycle reports required'
    return found


def prepare(parent_path, proof_path, implicit_order=False, boundary_sync=False, loss_sum=False, loss_cpu=False,
            local_postprocess=False, foreach_off=False, replicated=False, socket_network=False, optimizer_replay=False):
    loss_sum = loss_sum or loss_cpu
    assert sum([implicit_order, boundary_sync, loss_sum, local_postprocess, foreach_off]) <= 1
    assert not replicated or foreach_off
    assert not socket_network or foreach_off
    assert not optimizer_replay or (socket_network and foreach_off)
    parent_path, proof_path = parent_path.resolve(), proof_path.resolve()
    assert parent_path.is_relative_to(ROOT / 'attempts')
    assert proof_path.is_relative_to(ROOT / 'preflight/runtime_gpu')
    parent = json.loads(parent_path.read_text())
    state = json.loads((parent_path.parent / 'coordinator_state.json').read_text())
    proc = Path('/proc', str(state['pid']), 'stat')
    if proc.exists():
        fields = proc.read_text().rsplit(') ', 1)[1].split()
        assert fields[19] != str(state['pid_start_ticks']) or fields[0] == 'Z', 'parent coordinator still live'
    assert state['status'] in ('failed', 'finished_pending_scientific_verification')
    proof = json.loads(proof_path.read_text())
    assert proof['status'] == 'passed_cpu_fsdp_cycle_contract' and proof['gpu_verified'] is False
    assert proof['source_sha256'] == parent['source']['SOURCE_SNAPSHOT_SHA256']
    assert proof['boundary_device_sync_between_workers'] is boundary_sync
    assert proof['loss_avg_as_sum'] is loss_sum
    assert proof['loss_reduction_on_cpu'] is loss_cpu
    assert proof['local_only_postprocess'] is local_postprocess
    assert proof['adamw_foreach_disabled'] is foreach_off
    assert proof['replicated_fsdp2'] is replicated
    assert proof['requested_nccl_network'] == ('Socket' if socket_network else 'auto')
    assert proof['gpu_optimizer_backend_replay'] is optimizer_replay
    authorized = {line.split()[0] for line in Path('/etc/mpi/hostfile').read_text().splitlines() if line.strip()}
    for node in parent['nodes']:
        assert node['ip'] in authorized
        audit = json.loads((ROOT / 'logs' / f'node_gpu_audit_{node["ip"]}_latest.json').read_text())
        assert time.time() - audit['audit_unix'] < 300 and audit['hostname'] == node['hostname']
        assert len(audit['gpus']) == len(audit['compute_apps']) == 8 and len(audit['processes']) == 1
        assert audit['processes'][0]['cmd'] == '/usr/bin/python /m2v_intern2/zhenglin/gpu_occupy.py'
        assert audit['lease_owner_process'] is None and not audit['lease_live_group_members']
        terminal = json.loads((parent_path.parent / 'logs' / f'owner_exit_{node["hostname"]}.json').read_text())
        assert terminal['attempt'] == parent['attempt'] and terminal['nonce'] == node['nonce']
    capture_dirs = {str(Path(name).parent) for name in proof['capture_hashes']}
    assert len(capture_dirs) == 1
    captures = next(iter(capture_dirs))
    attempt = 'fsdp-cycle-' + datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')
    env = {'NCCL_P2P_DISABLE': '1', 'NCCL_NVLS_ENABLE': '0',
           'NCCL_LAUNCH_ORDER_IMPLICIT': '1' if implicit_order else '0',
           'OMNI_CYCLE_BOUNDARY_DEVICE_SYNC': '1' if boundary_sync else '0',
           'OMNI_CYCLE_LOSS_AVG_AS_SUM': '1' if loss_sum else '0',
           'OMNI_CYCLE_LOSS_REDUCTION_CPU': '1' if loss_cpu else '0',
           'OMNI_CYCLE_LOCAL_POSTPROCESS': '1' if local_postprocess else '0',
           'OMNI_CYCLE_ADAMW_FOREACH_OFF': '1' if foreach_off else '0',
           'OMNI_CYCLE_REPLICATED_FSDP2': '1' if replicated else '0',
           'OMNI_CYCLE_OPTIMIZER_REPLAY': '1' if optimizer_replay else '0'}
    if socket_network:
        env['NCCL_NET'] = 'Socket'
    nodes = []
    for rank, original in enumerate(parent['nodes']):
        command = ['/usr/bin/python', str(ROOT / 'scripts/prepare_runtime_probe.py'), original['ip'],
                   '--probe-kind', 'actor_cycle', '--pair-head', parent['nodes'][0]['ip'], '--node-rank', str(rank),
                   '--port', '18267', '--attempt-id', attempt, '--cpu-proof', str(proof_path), '--input', captures]
        result = subprocess.run(command, text=True, capture_output=True, timeout=45)
        if result.returncode:
            raise RuntimeError(f'probe preparation failed for {original["hostname"]}:\n{result.stderr[-4000:]}')
        path = Path(result.stdout.strip()).parent / 'node.json'
        node = json.loads(path.read_text())
        with Path(node['env_file']).open('a') as stream:
            for key, value in env.items():
                stream.write(f'export {key}={value}\n')
        node['env_sha256'] = digest(node['env_file'])
        path.write_text(json.dumps(node, indent=2) + '\n')
        nodes.append(node)
    output = ROOT / 'preflight/runtime_gpu' / attempt
    manifest = {'attempt': attempt, 'probe_kind': 'actor_cycle', 'output': str(output), 'nodes': nodes,
                'world_size': 16, 'source_sha256': proof['source_sha256'], 'parent_attempt': parent['attempt'],
                'cpu_proof': str(proof_path), 'cpu_proof_sha256': digest(proof_path),
                'storage_admission_mode': MODE,
                'minimum_start_free_gib': 0, 'dynamic_stop_free_gib': 0, 'timeout_seconds': 1380,
                'storage_policy': str(ROOT / 'assets/actor_cycle_diagnostic_storage_policy.json'),
                'storage_policy_sha256': digest(ROOT / 'assets/actor_cycle_diagnostic_storage_policy.json'),
                'payload_timeout_seconds': 1200, 'scientific_optimizer_updates': 0,
                'diagnostic_environment': env,
                'coordinator_files_sha256': {name: digest(ROOT / name) for name in (
                    'scripts/coordinate_fsdp_cycle_probe.py', 'scripts/coordinate_checkpoint_readback.py',
                    'scripts/coordinate_pair.py', 'scripts/shared_storage_health.py',
                    'assets/user_no_checkpoint_storage_amendment_20261017.json',
                    'assets/actor_cycle_diagnostic_storage_policy.json')}}
    manifest['replicated_cpu_global_gradient_oracle'] = replicated_cpu_oracle() if replicated else None
    admission(manifest, proof)
    archive = output / 'coordinator_code'
    archive.mkdir()
    for name in manifest['coordinator_files_sha256']:
        shutil.copy2(ROOT / name, archive / Path(name).name)
    path = output / 'pair_manifest.json'
    assert not path.exists()
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', type=Path)
    group.add_argument('--run', type=Path)
    parser.add_argument('--cpu-proof', type=Path)
    experiment = parser.add_mutually_exclusive_group()
    experiment.add_argument('--implicit-order', action='store_true')
    experiment.add_argument('--boundary-sync', action='store_true')
    experiment.add_argument('--loss-avg-as-sum', action='store_true')
    experiment.add_argument('--loss-on-cpu', action='store_true')
    experiment.add_argument('--local-postprocess', action='store_true')
    experiment.add_argument('--adamw-single-tensor', action='store_true')
    experiment.add_argument('--replicated-fsdp2', action='store_true')
    experiment.add_argument('--socket-network', action='store_true')
    experiment.add_argument('--socket-full-shard', action='store_true')
    experiment.add_argument('--optimizer-replay', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        assert args.cpu_proof is not None
        replicated = args.replicated_fsdp2 or args.socket_network or args.optimizer_replay
        foreach_off = args.adamw_single_tensor or replicated or args.socket_full_shard or args.optimizer_replay
        prepare(args.prepare, args.cpu_proof, args.implicit_order, args.boundary_sync, args.loss_avg_as_sum,
                args.loss_on_cpu, args.local_postprocess, foreach_off, replicated, args.socket_network or args.socket_full_shard or args.optimizer_replay, args.optimizer_replay)
    else:
        raise SystemExit(run_pair(args.run, admission, reports, 'finished_verified_two_native_worker_cycles'))
