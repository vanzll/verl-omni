"""Prepare one fixed-protocol pair attempt without starting services or freeing GPUs."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import secrets
import math
import shlex
import tarfile
import time
from shared_storage_health import MODE, validate_policy, check_write_health

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')
PAIRS = [[('10.83.2.231', 21010), ('10.83.1.141', 21010)],
         [('10.83.2.97', 21013), ('10.83.1.23', 21007)]]
OBJECT_STORE_BYTES = 4 * 1024**3
NAMES = {'N0': 'N0-OMNINFT', 'A0': 'A0-ART', 'A1': 'A1-PROMPTBAL', 'A2': 'A2-JOINTROUTE'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('variant', choices=NAMES)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--resume-extra-gpu-seconds', type=float, default=0.0)
    parser.add_argument('--single-pair-start', action='store_true',
                        help='lower projected peak only when coordinator verifies the entire allocation is idle')
    args = parser.parse_args()
    pair_index = 0 if args.variant in {'N0', 'A1'} else 1
    pair = PAIRS[pair_index]
    authorized = {line.split()[0] for line in Path('/etc/mpi/hostfile').read_text().splitlines() if line.strip()}
    assert all(ip in authorized for ip, _ in pair), 'pair is outside current allocation'
    source = {}
    for line in (ROOT / 'assets/source-selected.env').read_text().splitlines():
        key, value = line.removeprefix('export ').split('=', 1)
        source[key] = shlex.split(value)[0]
    snapshot = Path(source['SOURCE_SNAPSHOT_DIR'])
    assert snapshot.resolve().is_relative_to(ROOT / 'snapshots')
    assert hashlib.sha256((snapshot / 'source_full.tar.gz').read_bytes()).hexdigest() == source['SOURCE_SNAPSHOT_SHA256']
    policy_path = ROOT / 'assets/compact_checkpoint_storage_policy.json'
    policy_bytes = policy_path.read_bytes()
    policy_sha = hashlib.sha256(policy_bytes).hexdigest()
    policy = json.loads(policy_bytes)
    validate_policy(policy)
    assert policy['schema'] == 3
    assert policy['minimum_free_gib'] == policy['dynamic_stop_floor_gib'] == 0
    assert args.resume is None, 'No-checkpoint campaign cannot resume; start a fresh attempt'
    check_write_health()
    startup_budget = minimum_free_gib = 0
    numerical_path = ROOT / 'logs/fsdp_cycle_gpu_optimizer_replay_success.json'
    numerical = json.loads(numerical_path.read_text())
    assert numerical['status'] == 'passed_actual_world16_gpu_optimizer_backend_replay'
    assert numerical['all_five_fields_bitwise_equal_all_ranks'] is True
    numerical_state = json.loads(Path(numerical['coordinator_state']).read_text())
    assert numerical_state['status'] == 'finished_verified_two_native_worker_cycles'
    assert len(numerical_state['rank_reports']) == 16
    for rank_path in numerical_state['rank_reports'].values():
        rank = json.loads(Path(rank_path).read_text())
        assert rank['optimizer_backend_replay']['all_fields_bitwise_equal'] is True
        assert rank['actual_mesh_shape'] == [16, 1] and rank['actor_dp_size'] == 16
    config_audit = json.loads((ROOT / 'logs/resolved_config_audit.json').read_text())
    assert config_audit['status'] == 'passed'
    assert config_audit['checkpoint_storage_policy_sha256'] == policy_sha
    semantic = json.loads((ROOT / 'logs/compact_checkpoint_semantic_config_diff.json').read_text())
    assert semantic['status'] == 'passed' and semantic['checkpoint_storage_policy_sha256'] == policy_sha
    assert config_audit.get('native_reward_dispatch_batch_size') == 2
    assert semantic.get('native_reward_dispatch_batch_size') == 2
    transport_proof_path = ROOT / 'logs/reward_dispatch_payload_cpu.json'
    transport_proof = json.loads(transport_proof_path.read_text())
    assert transport_proof['status'] == 'passed' and transport_proof['source_sha256'] == source['SOURCE_SNAPSHOT_SHA256']
    assert transport_proof['all_five_native_configs_accept_dispatch_batch_size_2']
    assert transport_proof['all_tensor_and_uid_fields_preserved_exactly_after_pickle']
    assert transport_proof['per_sample_fake_scores_and_order_exactly_equal']
    assert transport_proof['maximum_rpc_size_reduction_factor'] > 50
    for variant, item in semantic['variants'].items():
        assert item['scientific_recipe_unchanged'] is True
        assert hashlib.sha256((ROOT / 'preflight/configs' / variant / 'resolved.yaml').read_bytes()).hexdigest() == item['resolved_sha256']
    if args.resume:
        args.resume = args.resume.resolve()
        assert args.resume.is_relative_to(ROOT / 'attempts')
        checkpoint = json.loads((args.resume / 'checkpoint_manifest.json').read_text())
        assert checkpoint['world_size'] == 16 and 0 < checkpoint['rollout'] < 75
        parent = json.loads((args.resume.parent.parent / 'pair_manifest.json').read_text())
        assert parent['variant'] == args.variant and parent['source'] == source
    assert args.resume_extra_gpu_seconds >= 0
    access = json.loads((ROOT / 'logs/wandb_api_preflight.json').read_text())
    assert access['status'] == 'passed_readonly_api_access'
    audits = []
    for ip, _ in pair:
        audit = json.loads((ROOT / 'logs' / f'node_gpu_audit_{ip}_latest.json').read_text())
        assert time.time() - audit['audit_unix'] < 300, 'refresh node audit'
        assert len(audit['gpus']) == 8 and len(audit['compute_apps']) == 8 and len(audit['processes']) == 1
        assert all('H800' in row and '560.35.03' in row for row in audit['gpus'])
        assert audit['processes'][0]['cmd'] == '/usr/bin/python /m2v_intern2/zhenglin/gpu_occupy.py'
        assert audit['lease_owner_process'] is None and not audit['lease_live_group_members']
        assert audit['shared_memory']['free_bytes'] >= OBJECT_STORE_BYTES + 2 * 1024**3, 'insufficient /dev/shm'
        audits.append(audit)
    attempts = ROOT / 'attempts'
    attempts.mkdir(exist_ok=True)
    with (attempts / 'prepare.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        number = 1
        while (attempts / f'LTX23-{NAMES[args.variant]}-P32K8-U150-S42-A{number:03d}').exists():
            number += 1
        run_name = f'LTX23-{NAMES[args.variant]}-P32K8-U150-S42-A{number:03d}'
        output = attempts / run_name
        (output / 'logs').mkdir(parents=True)
        (output / 'nodes').mkdir()
    provenance = output / 'provenance'
    provenance.mkdir()
    names = ['run_node_owner.sh', 'node_payload.sh', 'gpu_lease_lib.sh', 'runtime_env.sh',
             'use_source_snapshot.sh', 'ray_temp_env.sh', 'run_n0.sh', 'run_art.sh',
             'record_gpu_reservation.py', 'launch_training_owner.sh', 'node_owner_control.py',
             'prepare_pair_attempt.py', 'coordinate_pair.py', 'run_pair_coordinator.sh', 'check_assets.py', 'shared_storage_health.py', 'audit_running_driver.py']
    files = sorted([ROOT / 'scripts' / name for name in names] + [ROOT / 'bin/gpu_hold', ROOT / 'bin/gpu_free', policy_path, ROOT / 'assets/user_no_checkpoint_storage_amendment_20261017.json', ROOT / 'assets/common_operational_runtime_20261017.json'])
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    launch_manifest = provenance / 'launch_files_sha256.json'
    launch_manifest.write_text(json.dumps(hashes, indent=2, sort_keys=True) + '\n')
    launch_sha = hashlib.sha256(launch_manifest.read_bytes()).hexdigest()
    (provenance / 'compact_checkpoint_storage_policy.json').write_bytes(policy_bytes)
    (provenance / 'reward_dispatch_payload_cpu.json').write_bytes(transport_proof_path.read_bytes())
    (provenance / 'gpu_optimizer_backend_replay.json').write_bytes(numerical_path.read_bytes())
    with tarfile.open(provenance / 'launch_code.tar.gz', 'w:gz') as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
    run_id = secrets.token_hex(4)
    nodes = []
    for rank, ((ip, port), audit) in enumerate(zip(pair, audits, strict=True)):
        host = audit['hostname']
        descriptor = output / 'nodes' / f'{host}.json'
        env_file = descriptor.with_suffix('.env')
        nonce = secrets.token_hex(8)
        node = {'hostname': host, 'ip': ip, 'ssh_port': port, 'role': 'head' if rank == 0 else 'worker',
                'attempt': run_name, 'nonce': nonce, 'output': str(output), 'env_file': str(env_file),
                'launch_manifest': str(launch_manifest), 'launch_manifest_sha256': launch_sha,
                'tmux_socket': str(ROOT / 'tmp' / f'pair_{host}.sock'), 'tmux_session': f'omni_{args.variant}_{number:03d}'}
        node['planned_holder'] = audit['processes'][0]
        values = {'OMNI_OPERATION': 'training', 'OMNI_HOLDER_KIND': 'campaign', 'OMNI_EXPECTED_HOST': host,
                  'OMNI_ATTEMPT': run_name, 'OMNI_NONCE': nonce, 'OMNI_ROLE': node['role'], 'OMNI_NODE_IP': ip,
                  'OMNI_HEAD_IP': pair[0][0], 'OMNI_EXPECTED_PEER_IP': pair[1 - rank][0],
                  'OMNI_HOLDER_PID': audit['processes'][0]['pid'],
                  'OMNI_HOLDER_START_TICKS': audit['processes'][0]['start_ticks'],
                  'OMNI_RAY_PORT': 17500 + pair_index * 100 + number,
                  'OMNI_RAY_OBJECT_STORE_BYTES': OBJECT_STORE_BYTES, 'OMNI_NODE_DESCRIPTOR': str(descriptor),
                  'OMNI_LAUNCH_MANIFEST_SHA256': launch_sha, 'OMNI_VARIANT': args.variant,
                  'RUN_NAME': run_name, 'OUTPUT_DIR': str(output), 'WANDB_RUN_ID': run_id,
                  'OMNI_CHECKPOINT_STORAGE_POLICY_SHA256': policy_sha,
                  'WANDB_ENTITY': access['default_entity'], 'WANDB_MODE': 'online'}
        values.update(XDG_CONFIG_HOME=str(ROOT / 'cache/config'),
                      WANDB_CONFIG_DIR=str(ROOT / 'cache/wandb_config'),
                      WANDB_CACHE_DIR=str(ROOT / 'cache/wandb_cache'),
                      WANDB_DATA_DIR=str(ROOT / 'cache/wandb_data'),
                      CUPY_CACHE_DIR=str(ROOT / 'cache/cupy'), NUMBA_CACHE_DIR=str(ROOT / 'cache/numba'),
                      MPLCONFIGDIR=str(ROOT / 'cache/matplotlib'), TORCH_HOME=str(ROOT / 'cache/torch'))
        if args.resume:
            values.update(OMNI_RESUME_FROM_PATH=str(args.resume),
                          OMNI_RESUME_EXTRA_GPU_SECONDS=args.resume_extra_gpu_seconds)
        env_file.write_text('source ' + shlex.quote(str(snapshot / 'source_snapshot.env')) + '\n' +
                            ''.join(f'export {key}={shlex.quote(str(value))}\n' for key, value in values.items()))
        node['env_sha256'] = hashlib.sha256(env_file.read_bytes()).hexdigest()
        descriptor.write_text(json.dumps(node, indent=2) + '\n')
        nodes.append(node)
    manifest = {'schema': 1, 'attempt': run_name, 'variant': args.variant, 'created_unix': time.time(),
                'source': source, 'output': str(output), 'nodes': nodes, 'wandb_run_id': run_id,
                'wandb_entity': access['default_entity'], 'progress_kind': 'optimizer_updates',
                'target_updates': 150, 'target_rollouts': 75, 'gpus': 16,
                'minimum_free_gib': minimum_free_gib,
                'dynamic_stop_free_gib': policy['dynamic_stop_floor_gib'],
                'storage_admission_mode': MODE, 'checkpoint_writes': False, 'exact_resume_available': False,
                'projected_storage_start_gib': startup_budget,
                'checkpoint_storage_policy_sha256': policy_sha,
                'native_reward_dispatch_batch_size': 2,
                'ray_object_store_bytes': OBJECT_STORE_BYTES,
                'reward_dispatch_cpu_proof_sha256': hashlib.sha256(transport_proof_path.read_bytes()).hexdigest(),
                'launch_manifest': str(launch_manifest), 'launch_manifest_sha256': launch_sha,
                'runtime_exception': '/dev/shm object store:4GiB/node; capacities in node audits'}
    manifest['resume'] = str(args.resume) if args.resume else None
    manifest['resume_extra_gpu_seconds'] = args.resume_extra_gpu_seconds
    path = output / 'pair_manifest.json'
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(path)


if __name__ == '__main__':
    main()
