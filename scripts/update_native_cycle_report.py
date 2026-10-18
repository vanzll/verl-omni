"""Refresh completed native cycle evidence and real GPU reservation intervals."""
import html
import json
import os
from pathlib import Path
import re
import time

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')
now = time.time()
ledger_path = ROOT / 'logs/gpu_reservation_ledger.json'
intervals = []
paths = list((ROOT / 'attempts').glob('*/logs/gpu_reservation_*.json'))
paths += list((ROOT / 'preflight/runtime_gpu').glob('*/*/logs/gpu_reservation_*.json'))
for file in paths:
    row = json.loads(file.read_text())
    row.update(path=str(file.relative_to(ROOT)), provisional='stop_unix' not in row)
    row['allocated_gpu_hours'] = row.get('reserved_gpu_seconds', row['gpus'] * max(0, now - row['start_unix'])) / 3600
    intervals.append(row)
intervals.sort(key=lambda row: (row['start_unix'], row['host']))
ledger = dict(observed_unix=now, intervals=intervals,
              closed_allocated_gpu_hours=sum(r['allocated_gpu_hours'] for r in intervals if not r['provisional']),
              open_provisional_allocated_gpu_hours=sum(r['allocated_gpu_hours'] for r in intervals if r['provisional']),
              open_intervals=sum(r['provisional'] for r in intervals),
              scope='Actual reserved GPU intervals include initialization/eval/failures/diagnostics/waiting; pure occupancy excluded. Open intervals provisional, scientific completion separate.')
ledger_path.write_text(json.dumps(ledger, indent=2) + '\n')
report = ROOT / 'analysis/native_fsdp_cycle'; report.mkdir(exist_ok=True)
existing = json.loads((report / 'measurements.json').read_text())
cases = []
for output in sorted((ROOT / 'preflight/runtime_gpu').glob('fsdp-cycle-*')):
    statefile, manifestfile = output / 'coordinator_state.json', output / 'pair_manifest.json'
    if not statefile.exists() or not manifestfile.exists(): continue
    state, manifest = json.loads(statefile.read_text()), json.loads(manifestfile.read_text())
    if state['status'] in ('starting', 'running', 'stopping_owned_peer'): continue
    ranks = sorted([json.loads(f.read_text()) for f in output.glob('*/fsdp_cycle_rank_*.json')], key=lambda row: row['rank'])
    if len(ranks) != 16 or not all(row['cycles'] for row in ranks): continue
    env = manifest['diagnostic_environment']; label = 'Native baseline'
    for key, candidate in [('NCCL_LAUNCH_ORDER_IMPLICIT','NCCL launch order1'),
            ('OMNI_CYCLE_BOUNDARY_DEVICE_SYNC','Boundary device synchronization'),
            ('OMNI_CYCLE_LOSS_AVG_AS_SUM','Logged-loss AVG via GPU SUM'),
            ('OMNI_CYCLE_LOSS_REDUCTION_CPU','Logged-loss mean via CPU/Gloo SUM'),
            ('OMNI_CYCLE_LOCAL_POSTPROCESS','Local-only metric postprocess'),
            ('OMNI_CYCLE_ADAMW_FOREACH_OFF','AdamW single-tensor execution'),
            ('OMNI_CYCLE_REPLICATED_FSDP2','Replicated FSDP2, DP16/shard1, single-tensor AdamW')]:
        if env.get(key) == '1': label = candidate
    if env.get('NCCL_NET') == 'Socket': label += ', Socket network'
    stop = output / 'actual_failure_and_stop_observation.json'
    if not stop.exists(): stop = output / 'intentional_diagnostic_stop.json'
    cases.append(dict(attempt=manifest['attempt'], label=label, manifest=str(manifestfile),
        status=state['status'], environment=env,
        first_native_call_seconds_minmax=[min(r['cycles'][0]['native_worker_seconds'] for r in ranks), max(r['cycles'][0]['native_worker_seconds'] for r in ranks)],
        first_loss_values=sorted({r['cycles'][0]['loss'] for r in ranks}),
        first_grad_norm_values=sorted({r['cycles'][0]['finite_grad_norm'] for r in ranks}),
        second_call_returned_rank_count=sum(len(r['cycles']) >= 2 for r in ranks),
        scientific_optimizer_updates=0, ranks=ranks, stop_record=str(stop) if stop.exists() else None))
current = json.loads((ROOT / 'logs/current_fsdp_cycle_probe.json').read_text())
storage_stop_path = ROOT / 'logs/fsdp_cycle_socket_fullshard_storage_stop.json'
storage_stop = json.loads(storage_stop_path.read_text()) if storage_stop_path.exists() else None
storage_stat = os.statvfs(ROOT)
capacity = dict(free_bytes=storage_stat.f_bavail * storage_stat.f_frsize,
                diagnostic_start_gib=225, diagnostic_stop_gib=200,
                formal_start_gib=300, evidence=str(storage_stop_path) if storage_stop else None)
candidate = ('Socket with originalfullshard16' if current['diagnostic_environment'].get('OMNI_CYCLE_REPLICATED_FSDP2') == '0' else 'Socket with replicatedDP16/shard1') if current['diagnostic_environment'].get('NCCL_NET') == 'Socket' else ('Replicated FSDP2 DP16/shard1' if current['diagnostic_environment'].get('OMNI_CYCLE_REPLICATED_FSDP2') == '1' else 'AdamW single-tensor execution')
existing.update(observed_unix=now, cases=cases,
    scope='World16 native ART diagnostics: fixed2 recorded samples/rank x3 timesteps, synthetic guidance. Local-only outputs are diagnostic metrics. CPU/Gloo reported loss differs1.19e-7 from native GPU logging mean. Zero scientific updates/checkpoints.',
    conclusion='Launch-order, boundary device sync, logged-loss reduction variants and local-only postprocess failed to remove second-call slowdown. AdamW single-tensor execution improves firstcall~177s to~63s, but secondcall remains slow. ReplicatedFSDP2 removes parameterAllGather and completes firstcall~47s, but second-call gradientAllReduce6422528elements watchdogs after180s. Socket on the same replicated/foreachFalse setting passed bothnativecalls onall16 (second19.7–19.8s) andnormalcleanup; this is a conditional workingbypass, underlyingRDMAcauseunproven. No common formalruntimepromotion.',
    cleanup='Successful Socket replicated probe owners exited0. Subsequent full-shard Socket probe was stopped by the200GiB shared-space floor, both owners exited143 and restored occupancy. Fresh all4-node audit confirms only one8GPU holder/node and no live lease-owner groups.',
    all4_holders_evidence=str(ROOT / 'logs/fsdp_cycle_fullshard_storage_stop_cleanup.json'),
    latest_attempt_status=current['status'], storage_capacity=capacity,
    incomplete_storage_comparison=storage_stop,
    next='Await sufficient shared storage (225GiB diagnostic /300GiB formal), then run staged replicated Socket optimizer GPU replay. CPU contract and78 rejection guards passed; no GPU arithmetic or formal5-update acceptance claimed.',
    formal_scientific_acceptance_passed=False,
    closed_allocated_gpu_hours=ledger['closed_allocated_gpu_hours'],open_gpu_reservations=ledger['open_intervals'])
(report / 'measurements.json').write_text(json.dumps(existing, indent=2) + '\n')
def link(path,label): return '<a href="'+html.escape(os.path.relpath(path,report),quote=True)+'">'+html.escape(label)+'</a>'
parts=['<!doctype html><meta charset="utf-8"><title>Native FSDP cycle diagnostics</title><style>body{font:16px system-ui;max-width:1100px;margin:30px auto;line-height:1.5}td,th{border:1px solid #aaa;padding:8px}table{border-collapse:collapse}</style><h1>Native FSDP cycle diagnostics</h1>','<p>'+html.escape(existing['scope'])+'</p>','<table><tr><th>Setting</th><th>First call, range across16 ranks</th><th>Second calls returned</th><th>Evidence</th></tr>']
for c in cases:
    a,b = c['first_native_call_seconds_minmax']
    parts.append('<tr><td>'+html.escape(c['label'])+f'</td><td>{a:.3f}–{b:.3f}s</td><td>{c["second_call_returned_rank_count"]}/16</td><td>'+link(c['manifest'],'manifest')+(' / '+link(c['stop_record'],'stop/failure') if c['stop_record'] else '')+'</td></tr>')
parts.append('</table>')
for key in ['conclusion','debugger_limit','cleanup','next']: parts.append('<p>'+html.escape(existing[key])+'</p>')
if storage_stop:
    parts.append('<p>Original full-shard Socket comparison: no first native call returned before the storage guard stopped both owners. This does not establish a network failure. '+link(storage_stop_path,'Storage-stop evidence')+'</p>')
parts.append('<p>'+' · '.join([link(report/'measurements.json','All raw rank reports'),link(ROOT/'logs/fsdp_cycle_socket_two_call_success.json','Socket two-call success'),link(ROOT/'logs/fsdp_cycle_socket_full_chain_gpu_cleanup.json','Verified cleanup'),link(ledger_path,'Allocation ledger')])+'</p>')
page='\n'.join(parts)+'\n'; (report/'index.html').write_text(page)
links=re.findall(r'href="([^"]+)"',page)
for value in links: assert (report/html.unescape(value)).resolve().exists(),value
print(json.dumps(dict(cases=len(cases),relative_links_verified=len(links),closed_intervals=sum(not r['provisional'] for r in intervals),closed_gpu_hours=ledger['closed_allocated_gpu_hours'],open_intervals=ledger['open_intervals'],browser_verified=False)))
