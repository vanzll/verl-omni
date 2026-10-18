"""Read-only current-allocation identity audit. No environment or secrets dumped."""
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess


def proc_identity(pid):
    try:
        root = Path('/proc') / str(pid)
        fields = (root / 'stat').read_text().rsplit(') ', 1)[1].split()
        return {'pid': int(pid), 'start_ticks': fields[19], 'pgid': int(fields[2]),
                'state': fields[0], 'cmd': (root / 'cmdline').read_bytes().replace(b'\0', b' ').decode().strip()}
    except (FileNotFoundError, ProcessLookupError):
        return None


def query(fields, group='gpu'):
    return subprocess.check_output(['nvidia-smi', f'--query-{group}={fields}',
                                    '--format=csv,noheader,nounits'], timeout=15, text=True).strip().splitlines()


host = socket.gethostname().split('.')[0]
apps = query('pid,gpu_uuid', 'compute-apps')
pids = sorted({int(row.split(',')[0]) for row in apps if row.strip()})
lease_root = Path('/m2v_intern/zhenglin/paper_coordination/GPU_LEASES') / host
lease = {}
owner = lease_root / 'owner.env'
if owner.exists():
    for line in owner.read_text().splitlines():
        if line.startswith('GPU_LEASE_') and '=' in line:
            k, v = line.split('=', 1)
            vals = shlex.split(v)
            lease[k] = vals[0] if vals else ''
lease_pid = lease.get('GPU_LEASE_PID')
lease_proc = proc_identity(lease_pid) if lease_pid and lease_pid.isdigit() else None
lease_pgid = lease.get('GPU_LEASE_PGID', '')
group_members = []
if lease_pgid.isdigit():
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            try:
                f = (entry / 'stat').read_text().rsplit(') ', 1)[1].split()
                if f[2] == lease_pgid and f[0] != 'Z':
                    group_members.append(int(entry.name))
            except (OSError, ProcessLookupError):
                pass
print(json.dumps({'hostname': host, 'gpus': query('index,name,memory.total,driver_version'),
                  'audit_unix': __import__('time').time(),
                  'shared_memory': {'path': '/dev/shm',
                      'total_bytes': os.statvfs('/dev/shm').f_blocks * os.statvfs('/dev/shm').f_frsize,
                      'free_bytes': os.statvfs('/dev/shm').f_bavail * os.statvfs('/dev/shm').f_frsize},
                  'compute_apps': apps, 'processes': [proc_identity(pid) for pid in pids],
                  'lease': lease, 'lease_owner_process': lease_proc,
                  'lease_live_group_members': group_members}, indent=2))
