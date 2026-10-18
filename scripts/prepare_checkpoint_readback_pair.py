"""Prepare a read-only two-node checkpoint probe from verified formal CPU evidence."""
import argparse
from datetime import datetime, UTC
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')


def prepare(proof_path):
    proof_path = proof_path.resolve()
    assert proof_path.is_relative_to(ROOT / 'preflight/runtime_gpu')
    proof = json.loads(proof_path.read_text())
    assert proof['checkpoint_integrity_verified'] is True, 'actual complete formal checkpoint CPU proof required'
    assert proof['gpu_readback_verified'] is False and proof['world_size'] == 16
    assert proof['stage'] == 'config_and_native_contract_checked'
    config = Path(proof['config']).resolve()
    checkpoint = Path(proof['checkpoint']).resolve()
    assert config.is_relative_to(ROOT / 'attempts') and checkpoint.is_relative_to(ROOT / 'attempts')
    assert hashlib.sha256(config.read_bytes()).hexdigest() == proof['config_sha256']
    manifest_sha = hashlib.sha256((checkpoint / 'checkpoint_manifest.json').read_bytes()).hexdigest()
    assert manifest_sha == proof['checkpoint_manifest_sha256']
    parent = next(p for p in config.parents if (p / 'pair_manifest.json').is_file())
    original = json.loads((parent / 'pair_manifest.json').read_text())
    assert original['source']['SOURCE_SNAPSHOT_SHA256'] == proof['source_sha256']
    authorized = {line.split()[0] for line in Path('/etc/mpi/hostfile').read_text().splitlines() if line.strip()}
    assert all(node['ip'] in authorized for node in original['nodes'])
    # No SSH, CUDA call or occupancy mutation occurs here. Each node must have
    # a fresh, separately collected sole-holder audit before preparation.
    for node in original['nodes']:
        audit = json.loads((ROOT / 'logs' / f'node_gpu_audit_{node["ip"]}_latest.json').read_text())
        assert time.time() - audit['audit_unix'] < 300
        assert audit['hostname'] == node['hostname']
        assert len(audit['gpus']) == len(audit['compute_apps']) == 8 and len(audit['processes']) == 1
        assert audit['lease_owner_process'] is None and not audit['lease_live_group_members']
        assert audit['processes'][0]['cmd'] == '/usr/bin/python /m2v_intern2/zhenglin/gpu_occupy.py'
    attempt = 'checkpoint-readback-' + datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')
    head = original['nodes'][0]['ip']
    nodes = []
    for rank, node in enumerate(original['nodes']):
        command = ['/usr/bin/python', str(ROOT / 'scripts/prepare_runtime_probe.py'), node['ip'],
                   '--probe-kind', 'checkpoint_readback', '--pair-head', head,
                   '--node-rank', str(rank), '--port', '18241', '--attempt-id', attempt,
                   '--cpu-proof', str(proof_path), '--input', str(checkpoint)]
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=45)
        env_file = Path(result.stdout.strip())
        descriptor = env_file.parent / 'node.json'
        prepared = json.loads(descriptor.read_text())
        assert prepared['hostname'] == node['hostname'] and prepared['attempt'] == attempt
        nodes.append(prepared)
    output = ROOT / 'preflight/runtime_gpu' / attempt
    manifest = {'schema': 1, 'attempt': attempt, 'output': str(output), 'nodes': nodes,
                'source_sha256': proof['source_sha256'], 'parent_attempt': original['attempt'],
                'checkpoint': str(checkpoint), 'checkpoint_manifest_sha256': manifest_sha,
                'config_sha256': proof['config_sha256'], 'cpu_proof': str(proof_path),
                'cpu_proof_sha256': hashlib.sha256(proof_path.read_bytes()).hexdigest(),
                'world_size': 16, 'minimum_start_free_gib': 300, 'dynamic_stop_free_gib': 128,
                'timeout_seconds': 6000, 'created_unix': time.time(), 'scientific_optimizer_updates': 0,
                'payload_timeout_seconds': 5400, 'gpu_checkpoint_writes': False}
    path = output / 'pair_manifest.json'
    assert not path.exists()
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-proof', required=True, type=Path)
    prepare(parser.parse_args().cpu_proof)
