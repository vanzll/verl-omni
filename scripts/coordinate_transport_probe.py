"""Prepare and own an isolated two-node transport measurement after the parent exits."""
import argparse
from datetime import datetime, UTC
import hashlib
import json
import math
from pathlib import Path
import subprocess
import shutil
import time

from coordinate_pair import ROOT
from coordinate_checkpoint_readback import main as run_pair


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def admission(manifest, proof):
    assert manifest['probe_kind'] == 'transport' and manifest['world_size'] == 16
    assert proof['status'] == 'passed_cpu_vector_contract' and proof['world_size'] == 16
    assert proof['gpu_verified'] is False and proof['cuda_initialized'] is False
    assert proof['source_sha256'] == manifest['source_sha256']
    assert proof['script_sha256'] == digest(ROOT / 'scripts/verify_gpu_transport.py')
    overrides = manifest.get('transport_environment', {})
    assert overrides in ({}, {'NCCL_NET_GDR_LEVEL': 'LOC'}, {'NCCL_P2P_DISABLE': '1'},
                         {'NCCL_P2P_DISABLE': '1', 'NCCL_NVLS_ENABLE': '0'})
    for node in manifest['nodes']:
        lines = Path(node['env_file']).read_text().splitlines()
        for key in ('NCCL_NET_GDR_LEVEL', 'NCCL_P2P_DISABLE', 'NCCL_NVLS_ENABLE'):
            entries = [line for line in lines if line.startswith(f'export {key}=')]
            assert entries == ([f'export {key}={overrides[key]}'] if key in overrides else [])
    for name, expected in manifest['coordinator_files_sha256'].items():
        assert digest(ROOT / name) == expected


def reports(manifest):
    expected = {('all_reduce', size) for size in (16384, 1024**2, 32 * 1024**2, 128 * 1024**2)}
    expected |= {('all_gather_into_tensor', size) for size in (1024**2, 4 * 1024**2, 16 * 1024**2)}
    found = {}
    for node in manifest['nodes']:
        for path in Path(node['output']).glob('transport_rank_*.json'):
            row = json.loads(path.read_text())
            rank = row['rank']
            assert 0 <= rank < 16 and rank not in found and rank // 8 == node['node_rank']
            assert row['world_size'] == 16 and row['local_rank'] == 0
            assert row['launcher_local_rank'] == rank % 8 and row['physical_visible_device'] == str(rank % 8)
            assert row['attempt'] == manifest['attempt'] and row['nonce'] == node['nonce']
            assert row['source_sha256'] == manifest['source_sha256']
            assert row['stage'] == 'passed_correctness_measurements_recorded'
            assert row['scientific_optimizer_updates'] == row['optimizer_step_calls'] == 0
            assert len(row['collectives']) == 7
            assert {(entry['kind'], entry['input_bytes_per_rank']) for entry in row['collectives']} == expected
            for entry in row['collectives']:
                assert entry['correct'] is True and len(entry['seconds_each']) == 3
                assert all(math.isfinite(value) and value > 0 for value in entry['seconds_each'])
            assert all(math.isfinite(row[key]) and row[key] > 0 for key in (
                'pointwise_100_launches_seconds', 'bf16_matmul_2048_ten_seconds'))
            found[rank] = str(path)
    assert set(found) == set(range(16)), 'all16 actual transport rank reports required'
    return found


def prepare(parent_path, proof_path, gdr_off=False, p2p_off=False, nvls_off=False):
    assert not (gdr_off and p2p_off), 'one diagnostic transport change per run'
    overrides = {'NCCL_NET_GDR_LEVEL': 'LOC'} if gdr_off else ({'NCCL_P2P_DISABLE': '1'} if p2p_off else {})
    assert not nvls_off or (p2p_off and not gdr_off)
    if nvls_off:
        overrides['NCCL_NVLS_ENABLE'] = '0'
    parent_path, proof_path = parent_path.resolve(), proof_path.resolve()
    assert parent_path.is_relative_to(ROOT / 'attempts') and proof_path.is_relative_to(ROOT / 'preflight')
    original = json.loads(parent_path.read_text())
    state = json.loads((parent_path.parent / 'coordinator_state.json').read_text())
    proc = Path('/proc') / str(state['pid']) / 'stat'
    if proc.exists():
        fields = proc.read_text().rsplit(') ', 1)[1].split()
        assert fields[19] != str(state['pid_start_ticks']) or fields[0] == 'Z', 'parent coordinator still live'
    assert state['status'] in ('failed', 'finished_pending_scientific_verification')
    proof = json.loads(proof_path.read_text())
    assert proof['status'] == 'passed_cpu_vector_contract' and proof['gpu_verified'] is False
    assert proof['script_sha256'] == digest(ROOT / 'scripts/verify_gpu_transport.py')
    assert proof['source_sha256'] == original['source']['SOURCE_SNAPSHOT_SHA256']
    authorized = {line.split()[0] for line in Path('/etc/mpi/hostfile').read_text().splitlines() if line.strip()}
    for node in original['nodes']:
        assert node['ip'] in authorized
        audit = json.loads((ROOT / 'logs' / f'node_gpu_audit_{node["ip"]}_latest.json').read_text())
        assert time.time() - audit['audit_unix'] < 300 and audit['hostname'] == node['hostname']
        assert len(audit['gpus']) == len(audit['compute_apps']) == 8 and len(audit['processes']) == 1
        assert audit['lease_owner_process'] is None and not audit['lease_live_group_members']
        assert audit['processes'][0]['cmd'] == '/usr/bin/python /m2v_intern2/zhenglin/gpu_occupy.py'
        terminal = json.loads((parent_path.parent / 'logs' / f'owner_exit_{node["hostname"]}.json').read_text())
        assert terminal['attempt'] == original['attempt'] and terminal['nonce'] == node['nonce']
    attempt = 'transport-' + datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')
    nodes = []
    for rank, node in enumerate(original['nodes']):
        command = ['/usr/bin/python', str(ROOT / 'scripts/prepare_runtime_probe.py'), node['ip'],
                   '--probe-kind', 'transport', '--pair-head', original['nodes'][0]['ip'], '--node-rank', str(rank),
                   '--port', '18265', '--attempt-id', attempt, '--cpu-proof', str(proof_path)]
        result = subprocess.run(command, check=True, text=True, capture_output=True, timeout=45)
        descriptor = Path(result.stdout.strip()).parent / 'node.json'
        prepared = json.loads(descriptor.read_text())
        if overrides:
            env_file = Path(prepared['env_file'])
            with env_file.open('a') as stream:
                for key, value in overrides.items():
                    stream.write(f'export {key}={value}\n')
            prepared['env_sha256'] = digest(env_file)
            descriptor.write_text(json.dumps(prepared, indent=2) + '\n')
        nodes.append(prepared)
    output = ROOT / 'preflight/runtime_gpu' / attempt
    manifest = {'attempt': attempt, 'probe_kind': 'transport', 'output': str(output), 'nodes': nodes,
                'world_size': 16, 'source_sha256': proof['source_sha256'], 'parent_attempt': original['attempt'],
                'cpu_proof': str(proof_path), 'cpu_proof_sha256': digest(proof_path),
                'minimum_start_free_gib': 300, 'dynamic_stop_free_gib': 128, 'timeout_seconds': 480,
                'scientific_optimizer_updates': 0, 'payload_timeout_seconds': 300,
                'transport_environment': overrides,
                'coordinator_files_sha256': {name: digest(ROOT / name) for name in (
                    'scripts/coordinate_transport_probe.py', 'scripts/coordinate_checkpoint_readback.py',
                    'scripts/coordinate_pair.py')}}
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
    parser.add_argument('--gdr-off', action='store_true')
    parser.add_argument('--p2p-off', action='store_true')
    parser.add_argument('--nvls-off', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        assert args.cpu_proof is not None
        prepare(args.prepare, args.cpu_proof, args.gdr_off, args.p2p_off, args.nvls_off)
    else:
        raise SystemExit(run_pair(args.run, admission, reports, 'finished_verified_transport_measurements'))
