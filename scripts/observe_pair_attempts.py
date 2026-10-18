"""Read bounded attempt evidence; never control a process or infer successful updates."""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')
now = time.time()
rows = []
for name in sys.argv[1:]:
    attempt = ROOT / 'attempts' / name
    assert attempt.resolve().parent == ROOT / 'attempts'
    manifest = json.loads((attempt / 'pair_manifest.json').read_text())
    state_path = attempt / 'coordinator_state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    row = {'attempt': name, 'status': state.get('status', 'pre-admission'),
           'observed_unix': now,
           'coordinator_pid': state.get('pid'),
           'coordinator_start_ticks': state.get('pid_start_ticks'),
           'coordinator_observation_age_seconds': now - state.get('observed_unix', now),
           'source_sha256': manifest['source']['SOURCE_SNAPSHOT_SHA256'],
           'launch_sha256': manifest['launch_manifest_sha256'],
           'storage_admission_mode': manifest.get('storage_admission_mode'),
           'minimum_free_gib': manifest['minimum_free_gib'],
           'dynamic_stop_free_gib': manifest.get('dynamic_stop_free_gib'),
           'checkpoint_storage_policy_sha256': manifest.get('checkpoint_storage_policy_sha256'),
           'free_gib': state.get('free_bytes', 0) / 2**30,
           'minimum_observed_free_gib': state.get('minimum_observed_free_bytes', 0) / 2**30,
           'wandb_run_id': manifest['wandb_run_id'],
           'wandb_identity_verified': False,
           'nodes': state.get('nodes', {}),
           'failure_reason': state.get('failure_reason'),
           'first_low_space_sample': state.get('first_low_space_sample'),
           'tails': {}}
    for filename in [f'owner_{manifest["nodes"][0]["hostname"]}.log',
                     'ray_pair_readiness.log', 'train.log', 'training.log']:
        path = attempt / 'logs' / filename
        if not path.exists():
            continue
        with path.open('rb') as stream:
            stream.seek(max(0, os.fstat(stream.fileno()).st_size - 1800))
            row['tails'][filename] = stream.read().decode(errors='replace')
    rows.append(row)
print(json.dumps(rows, indent=2))
