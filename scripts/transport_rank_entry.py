"""Match the formal Ray actor GPU namespace before importing Torch."""
import os
from pathlib import Path
import runpy
import sys

assert 'torch' not in sys.modules
devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
assert len(devices) == len(set(devices)) == 8
local_rank = int(os.environ['LOCAL_RANK'])
assert 0 <= local_rank < 8
os.environ['OMNI_TRANSPORT_LAUNCHER_LOCAL_RANK'] = str(local_rank)
os.environ['CUDA_VISIBLE_DEVICES'] = devices[local_rank]
os.environ['LOCAL_RANK'] = '0'
target = Path(__file__).with_name('verify_gpu_transport.py')
sys.argv[0] = str(target)
runpy.run_path(str(target), run_name='__main__')
