"""Record an applied engine microbatch schedule without rewriting initial config."""
from pathlib import Path
import json,time,sys
import wandb
R=Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')
api=wandb.Api(timeout=20)
for plan in json.loads(Path(sys.argv[1] if len(sys.argv)>1 else R/'logs/current_microbatch_tuning.json').read_text()):
 state_path=Path(plan['event_state'])
 if not state_path.exists():continue
 s=json.loads(state_path.read_text())
 if s['status']!='applied_verified_all16':continue
 m=json.loads(Path(plan['manifest']).read_text());run=api.run(f"{m['wandb_entity']}/diffusionart-omni-ablation/{m['wandb_run_id']}")
 schedule=list(run.config.get('runtime_actor_microbatch_schedule',[]))
 if any(x['event_id']==plan['event_id'] for x in schedule):continue
 event=dict(event_id=plan['event_id'],old_micro=s['expected']['old_micro'],new_micro=s['expected']['new_micro'],after_successful_optimizer_updates=s['expected']['boundary_updates'],actor_dp=16,trajectory_accumulation=s['trajectory_accumulation'],optimizer_batch=128,all16_verified=True,script_sha256=s['script_sha256'],state_path=str(state_path),applied_unix=s['finished_unix'])
 schedule.append(event);run.config['runtime_actor_microbatch_schedule']=schedule;run.config['runtime_current_actor_micro_batch_size_per_gpu']=event['new_micro'];run.update()
 print(json.dumps(dict(variant=m['variant'],event_id=event['event_id'],metadata_schedule_recorded=True)),flush=True)
