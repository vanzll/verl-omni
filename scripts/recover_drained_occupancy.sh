#!/usr/bin/env bash
set -eo pipefail
source /m2v_intern2/zhenglin/my_bashrc.sh
set -u
campaign=/m2v_intern/zhenglin/omni_art_ablation_20261016
descriptor=${1:?pass exact terminal node descriptor}
case "$(realpath "$descriptor")" in
  "$campaign/attempts/"*/nodes/*.json|"$campaign/preflight/runtime_gpu/"*/node.json) ;;
  *) exit 2 ;;
esac
export TMPDIR="$campaign/tmp" TMUX_TMPDIR="$campaign/tmp"
audit="$campaign/logs/recovery_empty_$(hostname -s)_$(date -u +%H%M%S).json"
timeout 25 /usr/bin/python "$campaign/scripts/inspect_gpu_node.py" > "$audit"
/usr/bin/python - "$descriptor" "$audit" <<'PY'
from pathlib import Path
import json,sys,socket,time,os
node=json.loads(Path(sys.argv[1]).read_text()); audit=json.loads(Path(sys.argv[2]).read_text())
assert node['hostname']==audit['hostname']==socket.gethostname().split('.')[0]
lease=audit['lease']
assert lease['GPU_LEASE_STATUS']=='quarantined' and lease['GPU_LEASE_NONCE']==node['nonce']
assert lease['GPU_LEASE_ATTEMPT']==node['attempt']
assert audit['lease_owner_process'] is None and not audit['lease_live_group_members']
assert len(audit['gpus'])==8 and not audit['compute_apps']
path=Path(node['output'])/'logs'/f'gpu_reservation_{node["hostname"]}.json'
record=json.loads(path.read_text());assert record['nonce']==node['nonce'] and record['attempt']==node['attempt']
if 'stop_unix' not in record:
    record.update(stop_unix=time.time(),stop_reason='verified_empty_after_quarantined_cleanup',stop_evidence=sys.argv[2])
    record['reserved_gpu_seconds']=8*max(0,record['stop_unix']-record['start_unix'])
    tmp=path.with_suffix('.recovery.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(path)
PY
export GPU_OCCUPY_SCRIPT=/m2v_intern2/zhenglin/gpu_occupy.py
export GPU_OCCUPY_WORKDIR="$campaign" GPU_OCCUPY_PYTHON=/usr/bin/python
export GPU_OCCUPY_LOG="$campaign/logs/gpu_occupy_$(hostname -s).log"
export GPU_OCCUPY_SESSION="omni_hold_$(hostname -s)"
export FLOWART_GPU_LEASE_ROOT="/m2v_intern/zhenglin/paper_coordination/GPU_LEASES/$(hostname -s)"
unalias gpu_free gpu_hold 2>/dev/null || true
gpu_free() { "$campaign/bin/gpu_free"; }
gpu_hold() { LD_LIBRARY_PATH= "$campaign/bin/gpu_hold"; }
source "$campaign/scripts/gpu_lease_lib.sh"
recovery_nonce="recovery-$(date -u +%Y%m%dT%H%M%SZ)-$$"
flowart_gpu_lease_configure "$(hostname -s)" "$campaign" "$recovery_nonce" "$recovery_nonce"
flowart_gpu_lease_acquire
flowart_gpu_lease_release
