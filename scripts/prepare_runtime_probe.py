"""Build an owner env only from current allocation and fresh node audit evidence."""
import argparse
from datetime import datetime, UTC
import json
import hashlib
from pathlib import Path
import shlex
import shutil
import time
import uuid
import re
from shared_storage_health import validate_policy, check_write_health

parser = argparse.ArgumentParser()
parser.add_argument('ip')
parser.add_argument('--pair-head')
parser.add_argument('--node-rank', type=int, choices=[0, 1], default=0)
parser.add_argument('--port', type=int, default=18183)
parser.add_argument('--probe-kind', choices=['native', 'ltx_rollout', 'rewards', 'actor', 'nft_actor', 'art_precision', 'checkpoint_readback', 'transport', 'actor_cycle'], default='native')
parser.add_argument('--input', type=Path)
parser.add_argument('--cpu-proof', type=Path)
parser.add_argument('--attempt-id')
args = parser.parse_args()
c = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')
authorized = {row.split()[0] for row in Path('/etc/mpi/hostfile').read_text().splitlines() if row.strip()}
assert args.ip in authorized
assert args.pair_head is None or args.pair_head in authorized
assert 1024 <= args.port <= 65535
assert args.probe_kind in ('native', 'checkpoint_readback', 'transport', 'actor_cycle') or (args.pair_head is None and args.node_rank == 0)
assert args.attempt_id is None or (
    args.probe_kind in ('checkpoint_readback', 'transport', 'actor_cycle') and re.fullmatch(
        {'checkpoint_readback': 'checkpoint-readback-', 'transport': 'transport-',
         'actor_cycle': 'fsdp-cycle-'}[args.probe_kind] + r'\d{8}T\d{6}Z', args.attempt_id))
selected = {}
for line in (c / 'assets/source-selected.env').read_text().splitlines():
    if line.startswith('export ') and '=' in line:
        key, raw = line[7:].split('=', 1)
        selected[key] = shlex.split(raw)[0]
source_env = Path(selected['SOURCE_SNAPSHOT_DIR']) / 'source_snapshot.env'
assert source_env.resolve().is_relative_to(c / 'snapshots') and source_env.is_file()
checkpoint_proof = None
transport_proof = None
cycle_proof = None
if args.probe_kind == 'actor_cycle':
    assert args.pair_head is not None and (args.ip == args.pair_head) == (args.node_rank == 0)
    assert args.attempt_id is not None and args.cpu_proof is not None and args.input is not None
    assert args.cpu_proof.resolve().is_relative_to(c / 'preflight/runtime_gpu')
    assert args.input.resolve().is_relative_to(c / 'preflight/runtime_gpu')
    cycle_proof = json.loads(args.cpu_proof.read_text())
    assert cycle_proof['status'] == 'passed_cpu_fsdp_cycle_contract'
    assert cycle_proof['world_size'] == 16 and cycle_proof['gpu_verified'] is False
    assert cycle_proof['cuda_initialized'] is False and cycle_proof['gpu_checkpoint_writes'] is False
    assert cycle_proof['scientific_optimizer_updates'] == 0
    assert cycle_proof['cpu_native_worker_body_executed'] is True
    assert cycle_proof['cpu_boundary_event_contract_verified'] is True
    assert cycle_proof['cpu_loss_avg_sum_contract_verified'] is True
    assert cycle_proof['cpu_native_postprocess_device_restore_verified'] is True
    assert cycle_proof['cpu_local_postprocess_no_collectives_restore_verified'] is True
    assert type(cycle_proof['local_only_postprocess']) is bool
    assert cycle_proof['cpu_adamw_backend_switch_verified'] is True
    assert cycle_proof['cpu_optimizer_backend_replay_contract_verified'] is True
    assert type(cycle_proof['gpu_optimizer_backend_replay']) is bool
    assert cycle_proof['optimizer_replay_helper_sha256'] == hashlib.sha256((c / 'scripts/optimizer_backend_equivalence.py').read_bytes()).hexdigest()
    assert not cycle_proof['gpu_optimizer_backend_replay'] or (cycle_proof['requested_nccl_network'] == 'Socket' and cycle_proof['adamw_foreach_disabled'])
    assert type(cycle_proof['adamw_foreach_disabled']) is bool
    assert cycle_proof['cpu_native_mesh_factory_and_actor_dp_verified'] is True
    assert cycle_proof['cpu_ltx_train_mode_hook_is_inherited_noop'] is True
    assert type(cycle_proof['replicated_fsdp2']) is bool
    assert cycle_proof['requested_nccl_network'] in ('auto', 'Socket')
    assert cycle_proof['requested_nccl_network'] != 'Socket' or cycle_proof['adamw_foreach_disabled']
    assert not cycle_proof['replicated_fsdp2'] or cycle_proof['adamw_foreach_disabled']
    assert cycle_proof['expected_actor_dp_size'] == 16
    assert cycle_proof['expected_mesh_shape'] == ([16, 1] if cycle_proof['replicated_fsdp2'] else [16])
    assert type(cycle_proof['loss_reduction_on_cpu']) is bool
    assert type(cycle_proof['loss_avg_as_sum']) is bool
    assert type(cycle_proof['boundary_device_sync_between_workers']) is bool
    assert cycle_proof['requested_process_group_backend'] == 'cpu:gloo,cuda:nccl'
    assert cycle_proof['source_sha256'] == selected['SOURCE_SNAPSHOT_SHA256']
    for field, filename in [('script_sha256', 'verify_fsdp_cycle_gpu.py'),
                            ('shared_helper_sha256', 'verify_actor_gpu.py')]:
        assert cycle_proof[field] == hashlib.sha256((c / 'scripts' / filename).read_bytes()).hexdigest()
    assert cycle_proof['config_sha256'] == hashlib.sha256((c / 'preflight/configs/A0/resolved.yaml').read_bytes()).hexdigest()
    for filename, expected in cycle_proof['capture_hashes'].items():
        file = Path(filename)
        assert file.parent.resolve() == args.input.resolve()
        with file.open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected
    storage = json.loads((c / 'assets/actor_cycle_diagnostic_storage_policy.json').read_text())
    assert storage['kind'] == 'actor_cycle_no_checkpoint_diagnostic_only'
    assert storage['gpu_checkpoint_writes'] is False and storage['model_downloads_permitted'] is False
    validate_policy(storage)
    assert storage['minimum_start_free_gib'] == storage['dynamic_stop_free_gib'] == 0
    check_write_health()
    assert storage['conservative_new_output_budget_bytes'] <= 2**30
    assert storage['world_size'] == 16 and storage['diagnostic_local_samples'] == 2
    assert storage['diagnostic_timesteps'] == 3 and storage['diagnostic_worker_calls'] == 2
    available = __import__('os').statvfs(c)
    assert available.f_bavail * available.f_frsize >= storage['minimum_start_free_gib'] * 2**30
if args.probe_kind == 'transport':
    assert args.pair_head is not None and (args.ip == args.pair_head) == (args.node_rank == 0)
    assert args.attempt_id is not None and args.cpu_proof is not None
    assert args.cpu_proof.resolve().is_relative_to(c / 'preflight')
    transport_proof = json.loads(args.cpu_proof.read_text())
    assert transport_proof['status'] == 'passed_cpu_vector_contract'
    assert transport_proof['world_size'] == 16 and transport_proof['gpu_verified'] is False
    assert transport_proof['cuda_initialized'] is False
    assert transport_proof['source_sha256'] == selected['SOURCE_SNAPSHOT_SHA256']
    assert transport_proof['script_sha256'] == hashlib.sha256((c / 'scripts/verify_gpu_transport.py').read_bytes()).hexdigest()
    available = __import__('os').statvfs(c)
    assert available.f_bavail * available.f_frsize >= 300 * 2**30
if args.probe_kind == 'checkpoint_readback':
    assert args.pair_head is not None and (args.ip == args.pair_head) == (args.node_rank == 0)
    assert args.attempt_id is not None and args.cpu_proof is not None and args.input is not None
    assert args.cpu_proof.resolve().is_relative_to(c / 'preflight/runtime_gpu')
    assert args.input.resolve().is_relative_to(c / 'attempts')
    checkpoint_proof = json.loads(args.cpu_proof.read_text())
    assert checkpoint_proof['stage'] == 'config_and_native_contract_checked'
    assert checkpoint_proof['checkpoint_integrity_verified'] is True
    assert checkpoint_proof['gpu_readback_verified'] is False and checkpoint_proof['world_size'] == 16
    assert checkpoint_proof['source_sha256'] == selected['SOURCE_SNAPSHOT_SHA256']
    assert checkpoint_proof['checkpoint'] == str(args.input.resolve())
    checkpoint_config = Path(checkpoint_proof['config'])
    assert checkpoint_config.resolve().is_relative_to(c / 'attempts')
    assert hashlib.sha256(checkpoint_config.read_bytes()).hexdigest() == checkpoint_proof['config_sha256']
    assert hashlib.sha256((args.input / 'checkpoint_manifest.json').read_bytes()).hexdigest() == checkpoint_proof['checkpoint_manifest_sha256']
    capture_directories = set()
    for filename, expected in checkpoint_proof['capture_hashes'].items():
        file = Path(filename)
        assert file.resolve().is_relative_to(c / 'preflight/runtime_gpu')
        with file.open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected
        capture_directories.add(file.parent)
    assert len(capture_directories) == 1
    available = __import__('os').statvfs(c)
    assert available.f_bavail * available.f_frsize >= 300 * 2**30
if args.probe_kind in ('ltx_rollout', 'rewards', 'actor', 'nft_actor', 'art_precision'):
    proof_name = {'ltx_rollout': 'ltx_isolated_probe_cpu_preflight_provenance.json',
                  'rewards': 'reward_gpu_cpu_preflight_provenance.json',
                  'actor': 'actor_gpu_cpu_preflight_provenance.json',
                  'nft_actor': 'nft_gpu_cpu_preflight_provenance.json',
                  'art_precision': 'art_precision_cpu_preflight_provenance.json'}[args.probe_kind]
    proof = json.loads((c / 'logs' / proof_name).read_text())
    assert proof['status'] == 'passed'
    for name, digest in proof['files'].items():
        assert hashlib.sha256((c / 'scripts' / name).read_bytes()).hexdigest() == digest
    available = __import__('os').statvfs(c)
    minimum_gib = 275 if args.probe_kind == 'art_precision' else 300
    assert available.f_bavail * available.f_frsize >= minimum_gib * 2**30, f'isolated probe needs {minimum_gib}GiB headroom'
    if args.probe_kind == 'art_precision':
        assert proof['gpu_checkpoint_writes'] is False
        assert proof['minimum_start_free_gib'] == 275 and proof['dynamic_stop_free_gib'] == 250
        assert proof['conservative_max_probe_output_bytes'] <= 2**30
    if args.probe_kind == 'rewards':
        assert args.input is not None and args.input.resolve().is_relative_to(c / 'preflight/runtime_gpu')
        assert str(args.input.resolve()) == proof['input']
        with args.input.open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == proof['input_sha256']
    if args.probe_kind in ('actor', 'nft_actor', 'art_precision'):
        assert args.input is not None and str(args.input.resolve()) == proof['captures']
        for name, expected in proof['capture_hashes'].items():
            with Path(name).open('rb') as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected
        assert available.f_bavail * available.f_frsize >= proof['conservative_max_probe_output_bytes'] + 100 * 2**30
audit_file = c / 'logs' / f'node_gpu_audit_{args.ip}_latest.json'
assert time.time() - audit_file.stat().st_mtime < 300, 'refresh audit first'
a = json.loads(audit_file.read_text())
assert len(a['gpus']) == 8 and len(a['compute_apps']) == 8 and len(a['processes']) == 1
assert a['lease_owner_process'] is None and not a['lease_live_group_members']
holder = a['processes'][0]
if holder['cmd'] == 'python /m2v_intern/yuanziyang/launch_scripts/run.py':
    kind = 'platform'
else:
    assert holder['cmd'] == '/usr/bin/python /m2v_intern2/zhenglin/gpu_occupy.py'
    kind = 'campaign'
prefix = {'ltx_rollout': 'ltx-rollout-', 'rewards': 'reward-gpu-', 'actor': 'actor-gpu-',
          'checkpoint_readback': 'checkpoint-readback-', 'transport': 'transport-', 'actor_cycle': 'fsdp-cycle-',
          'nft_actor': 'nft-gpu-', 'art_precision': 'art-precision-', 'native': 'runtime-cu129-'}[args.probe_kind]
attempt = args.attempt_id or prefix + datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')
out = c / 'preflight/runtime_gpu' / attempt / a['hostname']
out.mkdir(parents=True, exist_ok=False)
values = {'OMNI_OPERATION': 'runtime_probe', 'OMNI_HOLDER_KIND': kind,
          'OMNI_EXPECTED_HOST': a['hostname'], 'OMNI_ATTEMPT': attempt,
          'OMNI_NONCE': uuid.uuid4().hex[:12], 'OMNI_HOLDER_PID': holder['pid'],
          'OMNI_HOLDER_START_TICKS': holder['start_ticks'], 'OMNI_NODE_IP': args.ip,
          'OMNI_HEAD_IP': args.pair_head or args.ip, 'OMNI_RAY_PORT': args.port,
          'OMNI_PROBE_NNODES': 2 if args.pair_head else 1,
          'OMNI_PROBE_NODE_RANK': args.node_rank, 'OUTPUT_DIR': str(out)}
values['OMNI_PROBE_KIND'] = args.probe_kind
if args.probe_kind == 'rewards':
    values['OMNI_REWARD_PROBE_INPUT'] = str(args.input.resolve())
if args.probe_kind in ('actor', 'nft_actor', 'art_precision'):
    values['OMNI_ACTOR_PROBE_CAPTURES'] = str(args.input.resolve())
if checkpoint_proof is not None:
    values.update(OMNI_CHECKPOINT_READBACK_CONFIG=str(checkpoint_config),
                  OMNI_CHECKPOINT_READBACK_PATH=str(args.input.resolve()),
                  OMNI_CHECKPOINT_READBACK_PROOF=str(args.cpu_proof.resolve()),
                  OMNI_ACTOR_PROBE_CAPTURES=str(next(iter(capture_directories))),
                  OMNI_NODE_DESCRIPTOR=str(out / 'node.json'))
if transport_proof is not None:
    proof_copy = out / 'transport_cpu_contract.json'
    shutil.copy2(args.cpu_proof, proof_copy)
    values.update(OMNI_TRANSPORT_CPU_PROOF=str(proof_copy), OMNI_NODE_DESCRIPTOR=str(out / 'node.json'))
if cycle_proof is not None:
    proof_copy = out / 'fsdp_cycle_cpu_contract.json'
    shutil.copy2(args.cpu_proof, proof_copy)
    values.update(OMNI_CYCLE_CPU_PROOF=str(proof_copy), OMNI_ACTOR_PROBE_CAPTURES=str(args.input.resolve()),
                  OMNI_NODE_DESCRIPTOR=str(out / 'node.json'))
code = out / 'probe_code'
code.mkdir()
hashes = {}
names = ['run_gpu_runtime_probe.sh', 'run_node_owner.sh', 'gpu_lease_lib.sh', 'runtime_env.sh',
         'use_source_snapshot.sh', 'record_gpu_reservation.py', 'ray_temp_env.sh']
names += {'ltx_rollout': ['verify_ltx_rollout_gpu.py', 'ltx_probe_pipeline.py', 'ltx_velocity_capture.py'],
          'rewards': ['verify_reward_gpu.py'], 'actor': ['verify_actor_gpu.py'],
          'nft_actor': ['verify_nft_gpu.py', 'verify_actor_gpu.py'],
          'art_precision': ['verify_art_precision_gpu.py', 'verify_actor_gpu.py', 'gradient_partition_details.py'],
          'checkpoint_readback': ['verify_formal_checkpoint_gpu.py', 'verify_actor_gpu.py', 'verify_nft_gpu.py',
                                  'checkpoint_rank_entry.py', 'launch_probe_owner.sh', 'node_owner_control.py'],
          'transport': ['verify_gpu_transport.py', 'transport_rank_entry.py', 'launch_probe_owner.sh', 'node_owner_control.py'],
          'actor_cycle': ['verify_fsdp_cycle_gpu.py', 'fsdp_cycle_rank_entry.py', 'verify_actor_gpu.py', 'optimizer_backend_equivalence.py',
                          'launch_probe_owner.sh', 'node_owner_control.py'],
          'native': ['verify_gpu_runtime.py']}[args.probe_kind]
for name in names:
    shutil.copy2(c / 'scripts' / name, code / name)
    hashes[name] = hashlib.sha256((code / name).read_bytes()).hexdigest()
(code / 'sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
env_file = out / 'node.env'
env_file.write_text('source ' + shlex.quote(str(source_env)) + '\n' +
                    ''.join(f'export {key}={shlex.quote(str(value))}\n' for key, value in values.items()))
if checkpoint_proof is not None or transport_proof is not None or cycle_proof is not None:
    (out / 'logs').mkdir(exist_ok=True)
    launch_files = {f'scripts/{name}': expected for name, expected in hashes.items()}
    launch = out / 'owner_control_launch_sha256.json'
    launch.write_text(json.dumps(launch_files, indent=2, sort_keys=True) + '\n')
    node = {'hostname': a['hostname'], 'ip': args.ip,
            'ssh_port': {'10.83.2.231': 21010, '10.83.1.141': 21010, '10.83.2.97': 21013, '10.83.1.23': 21007}[args.ip],
            'attempt': attempt, 'nonce': values['OMNI_NONCE'], 'output': str(out), 'env_file': str(env_file),
            'env_sha256': hashlib.sha256(env_file.read_bytes()).hexdigest(),
            'launch_manifest': str(launch), 'launch_manifest_sha256': hashlib.sha256(launch.read_bytes()).hexdigest(),
            'tmux_socket': str(c / 'tmp' / f'cp_{values["OMNI_NONCE"]}.sock'),
            'tmux_session': f'{args.probe_kind}_rank{args.node_rank}', 'planned_holder': holder,
            'node_rank': args.node_rank, 'checkpoint': str(args.input.resolve()) if checkpoint_proof is not None else None,
            'cpu_proof': str(args.cpu_proof.resolve()),
            'cpu_proof_sha256': hashlib.sha256(args.cpu_proof.read_bytes()).hexdigest()}
    (out / 'node.json').write_text(json.dumps(node, indent=2) + '\n')
(c / 'preflight/runtime_gpu' / f'current_{args.ip}.envpath').write_text(str(env_file) + '\n')
print(env_file)
