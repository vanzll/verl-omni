"""Read names and exact owned actor config through Ray's documented closure API."""
import argparse,json,os,time
from pathlib import Path
import ray
R=Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')
p=argparse.ArgumentParser();p.add_argument('manifest',type=Path);a=p.parse_args();m=json.loads(a.manifest.read_text())
evidence=json.loads((R/'logs/microbatch_live_allocation.json').read_text());assert time.time()-evidence['captured_unix']<300
authorized=set(evidence['authorized_ips']);assert {n['ip'] for n in m['nodes']}<=authorized
vals={}
import shlex
for l in Path(m['nodes'][0]['env_file']).read_text().splitlines():
 if l.startswith('export '):
  k,v=l[7:].split('=',1);vals[k]=shlex.split(v)[0]
address=f"{m['nodes'][0]['ip']}:{vals['OMNI_RAY_PORT']}"
ray.init(address=address,namespace='codex-microbatch-inspect',logging_level='ERROR')
names=ray.util.list_named_actors(all_namespaces=True)
rows=[]
for x in names:
 h=ray.get_actor(x['name'],namespace=x['namespace']);desc=h._ray_actor_creation_function_descriptor
 if desc.class_name.endswith('WorkerDict'):rows.append(dict(name=x['name'],namespace=x['namespace'],class_name=desc.class_name))
print(json.dumps(dict(attempt=m['attempt'],address=address,actors=rows)),flush=True)
(R/'logs'/f'live_actor_names_{m["variant"]}.json').write_text(json.dumps(rows,indent=2)+'\n')
ray.shutdown()
