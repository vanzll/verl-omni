"""Give each torchrun rank the same single-visible-GPU namespace as Ray."""
import ctypes
import os
import signal
from pathlib import Path
import runpy
import sys

def install_parent_death_signal():
    # torchrun gives each worker its own process group. Kill this rank if its
    # launcher dies, so terminating the owner cannot leave detached GPU ranks.
    parent = os.getppid()
    assert parent > 1, 'refuse an already orphaned rank'
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'PR_SET_PDEATHSIG failed')
    if os.getppid() != parent:
        os.kill(os.getpid(), signal.SIGKILL)
    return parent


install_parent_death_signal()
assert 'torch' not in sys.modules
devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
rank = int(os.environ['LOCAL_RANK'])
assert len(devices) == len(set(devices)) == 8 and 0 <= rank < 8
os.environ['OMNI_CYCLE_LAUNCHER_LOCAL_RANK'] = str(rank)
os.environ['CUDA_VISIBLE_DEVICES'] = devices[rank]
os.environ['LOCAL_RANK'] = '0'
target = Path(__file__).with_name('verify_fsdp_cycle_gpu.py')
sys.argv[0] = str(target)
runpy.run_path(str(target), run_name='__main__')
