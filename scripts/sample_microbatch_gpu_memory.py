"""Bounded read-only GPU telemetry, fenced against the master's current allocation."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import sys,json,time,shlex
from coordinate_pair import remote,ROOT
end=time.time()+2400
attempts=json.loads((ROOT/'logs/current_scientific_attempts.json').read_text())
nodes=[n for row in attempts for n in json.loads(Path(row['manifest']).read_text())['nodes']]
log=ROOT/'logs/microbatch_memory_trace_20261017.jsonl'
def sample(node):
 command='source /m2v_intern2/zhenglin/my_bashrc.sh\n'+shlex.join(['timeout','15','nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw','--format=csv,noheader,nounits'])
 try:
  z=remote(node,command,timeout=22)
  return dict(hostname=node['hostname'],unix=time.time(),returncode=z.returncode,gpus=z.stdout if z.returncode==0 else None)
 except Exception as e:return dict(hostname=node['hostname'],unix=time.time(),error_type=type(e).__name__)
while time.time()<end:
 with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(sample,nodes))
 with log.open('a') as f:
  for row in rows:f.write(json.dumps(row)+'\n')
 time.sleep(15)
