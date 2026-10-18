"""Fenced live engine-config update through Ray's standard actor closure API.

No source/method patch, restart, optimizer step, model save, or RNG operation.
Actor closures queue behind the full native update_actor RPC (both windows).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import time
import uuid

import ray

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')


def actor_snapshot(outer, expected):
    import os
    import socket
    from pathlib import Path
    import json
    import time
    import torch

    host = socket.gethostname().split('.')[0]
    node = expected['nodes'][host]
    assert os.environ['OMNI_ATTEMPT'] == expected['attempt']
    assert os.environ['OMNI_NONCE'] == node['nonce']
    assert os.environ['SOURCE_SNAPSHOT_SHA256'] == expected['source_sha256']
    identity = json.loads(Path(node['owner_identity']).read_text())
    assert identity['attempt'] == expected['attempt'] and identity['nonce'] == node['nonce']
    assert Path(f"/proc/{identity['pid']}/stat").read_text().rsplit(') ', 1)[1].split()[19] == identity['start_ticks']
    policies = [w for w in outer.worker_dict.values() if getattr(w, 'actor', None) is not None]
    assert len(policies) == 1
    worker = policies[0]
    actor = worker.actor
    assert actor.engine.get_data_parallel_size() == 16
    cfgs = [actor.engine_config, actor.engine.engine_config]
    assert all(c.micro_batch_size_per_gpu == expected['old_micro'] for c in cfgs)
    state = next(v for v in actor.engine.optimizer.state.values() if 'step' in v)
    step = state['step']
    if hasattr(step, 'to_local'):
        step = step.to_local()
    step = int(step.item())
    return dict(rank=int(worker.rank), hostname=host, pid=os.getpid(),
                pid_start_ticks=Path('/proc/self/stat').read_text().rsplit(') ', 1)[1].split()[19],
                micro_batch_size_per_gpu=cfgs[0].micro_batch_size_per_gpu,
                infer_micro_batch_size_per_gpu=cfgs[0].infer_micro_batch_size_per_gpu,
                optimizer_updates_applied=step, optimizer_id=id(actor.engine.optimizer),
                scheduler_last_epoch=actor.engine.lr_scheduler.last_epoch,
                learning_rates=[g['lr'] for g in actor.engine.optimizer.param_groups],
                cuda_allocated_bytes=torch.cuda.memory_allocated(),
                cuda_reserved_bytes=torch.cuda.memory_reserved(),
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(), unix=time.time())


def actor_commit(outer, expected):
    import os
    import time
    from pathlib import Path
    import json

    events = getattr(outer, '_codex_microbatch_events', {})
    if expected['event_id'] in events:
        return events[expected['event_id']]
    # All checks run again at the actor's serial call boundary.
    before = actor_snapshot(outer, expected)
    assert before['optimizer_updates_applied'] == expected['boundary_updates']
    worker = next(w for w in outer.worker_dict.values() if getattr(w, 'actor', None) is not None)
    actor = worker.actor
    actor.engine_config.micro_batch_size_per_gpu = expected['new_micro']
    actor.engine.engine_config.micro_batch_size_per_gpu = expected['new_micro']
    post_expected = dict(expected, old_micro=expected['new_micro'])
    after = actor_snapshot(outer, post_expected)
    for key in ('optimizer_updates_applied', 'optimizer_id', 'scheduler_last_epoch', 'learning_rates',
                'infer_micro_batch_size_per_gpu', 'pid_start_ticks'):
        assert before[key] == after[key]
    receipt = dict(event_id=expected['event_id'], before=before, after=after,
                   status='applied_at_serial_actor_boundary', applied_unix=time.time())
    events[expected['event_id']] = receipt
    outer._codex_microbatch_events = events
    output = Path(expected['event_output']) / f"rank_{before['rank']}.json"
    output.write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def publish(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def gather(refs, label, deadline):
    result = []
    while refs:
        assert time.time() < deadline, f'{label} deadline; inspect same event, do not blindly retry'
        ready, refs = ray.wait(refs, num_returns=1, timeout=15)
        if ready:
            result.extend(ray.get(ready))
        print(json.dumps(dict(stage=label, acknowledged=len(result), pending=len(refs))), flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('manifest', type=Path)
    p.add_argument('--old-micro', type=int, default=2)
    p.add_argument('--new-micro', type=int, choices=(4, 8), required=True)
    p.add_argument('--event-id', required=True)
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_text())
    assert args.manifest.resolve().is_relative_to(ROOT / 'attempts')
    allocation = json.loads((ROOT / 'logs/microbatch_live_allocation.json').read_text())
    assert time.time() - allocation['captured_unix'] < 300
    assert {n['ip'] for n in manifest['nodes']} <= set(allocation['authorized_ips'])
    proof = json.loads((ROOT / 'logs/microbatch_2_4_8_cpu_contract.json').read_text())
    assert proof['status'] == 'passed_native_temporal_loop_actual_ART_NFT_loss_microbatch_contract'
    assert args.old_micro in (2, 4) and args.new_micro > args.old_micro
    event_output = Path(manifest['output']) / 'logs/runtime_microbatch' / args.event_id
    event_output.mkdir(parents=True, exist_ok=False)
    expected = dict(attempt=manifest['attempt'], source_sha256=manifest['source']['SOURCE_SNAPSHOT_SHA256'],
                    nodes={n['hostname']:dict(nonce=n['nonce'], owner_identity=str(Path(n['output']) / 'logs' / f"owner_identity_{n['hostname']}.json")) for n in manifest['nodes']},
                    old_micro=args.old_micro, new_micro=args.new_micro, event_id=args.event_id,
                    event_output=str(event_output))
    state = dict(status='preparing_readonly_actor_snapshots', expected=expected,
                 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 cpu_proof_sha256=hashlib.sha256((ROOT/'logs/microbatch_2_4_8_cpu_contract.json').read_bytes()).hexdigest(),
                 scientific_optimizer_batch=128, actor_dp=16, trajectory_accumulation=8//args.new_micro,
                 initial_launch_sha256=manifest['launch_manifest_sha256'], started_unix=time.time())
    publish(event_output / 'state.json', state)
    values = {}
    for line in Path(manifest['nodes'][0]['env_file']).read_text().splitlines():
        if line.startswith('export '):
            key, raw = line[7:].split('=', 1)
            values[key] = shlex.split(raw)[0]
    ray.init(address=f"{manifest['nodes'][0]['ip']}:{values['OMNI_RAY_PORT']}",
             namespace='codex-microbatch-control', logging_level='ERROR')
    handles = []
    for entry in ray.util.list_named_actors(all_namespaces=True):
        handle = ray.get_actor(entry['name'], namespace=entry['namespace'])
        if handle._ray_actor_creation_function_descriptor.class_name.endswith('WorkerDict'):
            handles.append(handle)
    assert len(handles) == 16
    before = gather([h.__ray_call__.remote(actor_snapshot, expected) for h in handles], 'readonly_snapshots', time.time()+1200)
    assert {x['rank'] for x in before} == set(range(16))
    boundaries = {x['optimizer_updates_applied'] for x in before}
    assert len(boundaries) == 1 and next(iter(boundaries)) % 2 == 0
    expected['boundary_updates'] = next(iter(boundaries))
    state.update(status='committing_at_verified_rollout_boundary', before=before, expected=expected)
    publish(event_output / 'state.json', state)
    receipts = gather([h.__ray_call__.remote(actor_commit, expected) for h in handles], 'commit', time.time()+90)
    assert {x['after']['rank'] for x in receipts} == set(range(16))
    state.update(status='applied_verified_all16', receipts=receipts, finished_unix=time.time())
    publish(event_output / 'state.json', state)
    print(json.dumps(dict(status=state['status'], event_id=args.event_id,
                         optimizer_updates_before_change=expected['boundary_updates'],
                         new_micro=args.new_micro, output=str(event_output))), flush=True)
    ray.shutdown()


if __name__ == '__main__':
    main()
