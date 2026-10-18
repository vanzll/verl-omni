#!/usr/bin/env python3
"""Exercise real Ray cleanup under the lease with zero GPUs and stub occupancy."""
import json
import os
import signal
import socket
import subprocess
import time
import uuid
from pathlib import Path

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')


def identity(pid):
    try:
        raw = Path(f'/proc/{pid}/stat').read_text()
        fields = raw[raw.rfind(')') + 2:].split()
        return dict(pid=pid, state=fields[0], ppid=int(fields[1]), pgid=int(fields[2]),
                    sid=int(fields[3]), ticks=int(fields[19]))
    except FileNotFoundError:
        return None


def live(record):
    current = identity(record['pid'])
    return current is not None and current['ticks'] == record['ticks'] and current['state'] != 'Z'


def descendants(roots):
    records = {}
    for item in Path('/proc').iterdir():
        if item.name.isdecimal():
            record = identity(int(item.name))
            if record:
                records[record['pid']] = record
    selected = set(roots)
    while True:
        expanded = selected | {pid for pid, row in records.items() if row['ppid'] in selected}
        if expanded == selected:
            return {pid: records[pid] for pid in selected if pid in records}
        selected = expanded


def main():
    nonce = uuid.uuid4().hex[:6]
    work = ROOT / 'logs' / ('ray_cpu_' + nonce)
    work.mkdir()
    # All payloads stay under the campaign; only the short alias lives at the user mount root.
    alias = Path('/m2v_intern/zhenglin') / ('rc' + nonce)
    (work / 'ray').mkdir()
    (work / 'spill').mkdir()
    alias.symlink_to(work / 'ray')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', RAY_USAGE_STATS_ENABLED='0',
               RAY_local_fs_capacity_threshold='1.0',
               TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache/xdg'),
               CPU_WORK=str(work), CPU_NONCE=nonce, CPU_RAY_ALIAS=str(alias), CPU_RAY_PORT=str(port),
               PATH=str(ROOT / 'env/bin') + ':' + os.environ['PATH'])
    env.pop('RAY_ADDRESS', None)
    owner_script = work / 'owner.sh'
    owner_script.write_text('''#!/usr/bin/env bash
set -euo pipefail
source /m2v_intern2/zhenglin/my_bashrc.sh
campaign=/m2v_intern/zhenglin/omni_art_ablation_20261016
export PATH="$campaign/env/bin:$PATH" CUDA_VISIBLE_DEVICES="" RAY_USAGE_STATS_ENABLED=0
export TMPDIR="$campaign/tmp" RAY_TMPDIR="$CPU_RAY_ALIAS"
export FLOWART_GPU_LEASE_ROOT="$CPU_WORK/lease"
unalias gpu_hold gpu_free 2>/dev/null || true
gpu_free() { echo free >> "$CPU_WORK/stub_calls"; }
gpu_hold() {
  echo hold >> "$CPU_WORK/stub_calls"
  "$campaign/env/bin/python" - <<'CHECK'
import json,os
from pathlib import Path
root=Path(os.environ['CPU_WORK'])
registry=root/'pre_stop_registry.json'
remaining=[]
if registry.exists():
    saved=json.loads(registry.read_text())
    for row in saved['processes']:
        if row['pid'] in saved['ignore']: continue
        try:
            raw=Path(f"/proc/{row['pid']}/stat").read_text()
            fields=raw[raw.rfind(')')+2:].split()
            if int(fields[19])==row['ticks'] and fields[0]!='Z': remaining.append(row)
        except FileNotFoundError: pass
(root/'hold_survivors.json').write_text(json.dumps(remaining))
CHECK
}
source "$campaign/scripts/gpu_lease_lib.sh"
flowart_gpu_lease_configure cpu-ray-host "$campaign" cpu-ray-test "$CPU_NONCE"
trap 'flowart_gpu_lease_cleanup_owned_group; flowart_gpu_lease_release' EXIT
trap 'flowart_gpu_lease_forward_signal TERM 143' TERM
flowart_gpu_lease_acquire
flowart_gpu_lease_run_owned_command real-ray-cpu ray start --head \
  --node-ip-address=127.0.0.1 --port="$CPU_RAY_PORT" --num-cpus=1 --num-gpus=0 --object-store-memory=100000000 \
  --min-worker-port=30000 --max-worker-port=31999 --plasma-directory=/dev/shm \
  --object-spilling-directory="$CPU_WORK/spill" \
  --include-dashboard=false --disable-usage-stats --temp-dir="$CPU_RAY_ALIAS" --block
''')
    driver_script = work / 'driver.py'
    driver_script.write_text('''import json,os,time
from pathlib import Path
import ray
ray.init(address='127.0.0.1:'+os.environ['CPU_RAY_PORT'], namespace='lease-cpu-probe', logging_level='ERROR')
@ray.remote(num_cpus=1, num_gpus=0)
class CPUProbe:
    def pid(self): return os.getpid()
actor=CPUProbe.remote()
pid=ray.get(actor.pid.remote(),timeout=40)
first=ray.put(b'0'*80000000)
second=ray.put(b'1'*80000000)
assert len(ray.get(first,timeout=40))==80000000
from ray._private.internal_api import memory_summary
stats=memory_summary(stats_only=True)
Path(os.environ['CPU_WORK'],'spill_stats.txt').write_text(stats)
assert 'Spilled' in stats, stats
Path(os.environ['CPU_WORK'],'ready.json').write_text(json.dumps(dict(actor_pid=pid)))
while True: time.sleep(1)
''')
    owner = driver = None
    tracked = {}
    result = dict(work=str(work), short_alias=str(alias), port=port, status='starting')
    print('CPU Ray probe work:', work, 'short alias:', alias, 'port:', port, flush=True)
    try:
        with (work / 'owner.log').open('w') as out:
            owner = subprocess.Popen(['bash', str(owner_script)], env=env, stdout=out,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        driver = None
        deadline = time.monotonic() + 130
        while time.monotonic() < deadline:
            tracked.update(descendants([owner.pid] + ([driver.pid] if driver else [])))
            if owner.poll() is not None:
                raise RuntimeError('Ray owner exited during startup')
            if driver is None and (alias / 'session_latest/sockets/raylet').exists():
                with (work / 'driver.log').open('w') as out:
                    driver = subprocess.Popen([str(ROOT / 'env/bin/python'), str(driver_script)],
                                              env=env, stdout=out, stderr=subprocess.STDOUT,
                                              start_new_session=True)
            if driver and driver.poll() is not None:
                raise RuntimeError('CPU Ray driver exited before actor readiness')
            if (work / 'ready.json').exists():
                actor_pid = json.loads((work / 'ready.json').read_text())['actor_pid']
                tracked.update(descendants([owner.pid, driver.pid, actor_pid]))
                result['actor_pid'] = actor_pid
                (work / 'pre_stop_registry.json').write_text(json.dumps(dict(
                    processes=list(tracked.values()), ignore=[owner.pid, driver.pid])))
                print('CPU actor ready:', actor_pid, 'tracked processes:', len(tracked), flush=True)
                break
            time.sleep(1)
        else:
            raise TimeoutError('CPU Ray actor startup deadline exceeded')
        os.kill(owner.pid, signal.SIGTERM)
        owner.wait(timeout=55)
        print('Owner exited:', owner.returncode, flush=True)
        driver.terminate()
        driver.wait(timeout=10)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and any(live(row) for row in tracked.values()):
            time.sleep(1)
        remaining = [row for row in tracked.values() if live(row)]
        result.update(status='failed' if remaining else 'passed', survivors_before_repair=remaining,
                      tracked=list(tracked.values()), owner_exit_code=owner.returncode,
                      stub_calls=(work / 'stub_calls').read_text().splitlines())
        hold_path = work / 'hold_survivors.json'
        result['survivors_at_hold'] = json.loads(hold_path.read_text()) if hold_path.exists() else None
        if result['survivors_at_hold'] != []:
            result['status'] = 'failed'
    except Exception as error:
        result.update(status='failed', error=type(error).__name__ + ': ' + str(error),
                      tracked=list(tracked.values()))
    finally:
        # Repair only exact CPU-test process identities, even if they changed process groups.
        for sig in (signal.SIGTERM, signal.SIGKILL):
            for row in tracked.values():
                if live(row):
                    try:
                        os.kill(row['pid'], sig)
                    except ProcessLookupError:
                        pass
            time.sleep(1)
        for process in (driver, owner):
            if process:
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    result['cleanup_incomplete'] = True
        result['survivors_after_repair'] = [row for row in tracked.values() if live(row)]
        path = work / 'result.json'
        path.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
    return 0 if result['status'] == 'passed' and not result['survivors_after_repair'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
