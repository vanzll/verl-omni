"""Observe one applied microbatch change through its next full native update RPC."""
from pathlib import Path
import argparse,json,runpy,shlex,time
import ray
R=Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')
p=argparse.ArgumentParser();p.add_argument('event_state',type=Path);a=p.parse_args()
s=json.loads(a.event_state.read_text());assert s['status']=='applied_verified_all16'
expected=dict(s['expected'],old_micro=s['expected']['new_micro']);target=s['expected']['boundary_updates']+2
manifest=Path(s['expected']['event_output']).parents[2]/'pair_manifest.json'
# event_output = attempt/logs/runtime_microbatch/event; parents[2] is attempt.
m=json.loads(manifest.read_text());values={}
for line in Path(m['nodes'][0]['env_file']).read_text().splitlines():
 if line.startswith('export '):
  k,v=line[7:].split('=',1);values[k]=shlex.split(v)[0]
ray.init(address=f"{m['nodes'][0]['ip']}:{values['OMNI_RAY_PORT']}",namespace='codex-microbatch-observer',logging_level='ERROR')
fn=runpy.run_path(str(R/'scripts/update_live_actor_microbatch.py'))['actor_snapshot']
handles=[]
for entry in ray.util.list_named_actors(all_namespaces=True):
 handle=ray.get_actor(entry['name'],namespace=entry['namespace'])
 if handle._ray_actor_creation_function_descriptor.class_name.endswith('WorkerDict'):handles.append(handle)
assert len(handles)==16
end=time.time()+2400
while time.time()<end:
 refs=[h.__ray_call__.remote(fn,expected) for h in handles]
 rows=ray.get(refs,timeout=1300);assert {x['rank'] for x in rows}==set(range(16))
 record=dict(observed_unix=time.time(),target_optimizer_updates=target,ranks=rows)
 with (a.event_state.parent/'postchange_samples.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
 print(json.dumps(dict(minimum_optimizer_updates=min(x['optimizer_updates_applied'] for x in rows),target=target)),flush=True)
 if all(x['optimizer_updates_applied']>=target for x in rows):
  record['status']='verified_all16_next_full_update_RPC_completed_with_requested_microbatch'
  q=a.event_state.parent/'postchange_verification.json';tmp=q.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(q);break
 time.sleep(45)
else:raise TimeoutError('No next full update observed within bounded observer budget')
ray.shutdown()
