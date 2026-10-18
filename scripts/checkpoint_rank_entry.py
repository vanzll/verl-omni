"""Match Ray's one-visible-GPU worker namespace before any Torch import."""
import os
from pathlib import Path
import runpy
import sys

assert 'torch' not in sys.modules, 'GPU visibility must be set before importing Torch'
devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
assert len(devices) == len(set(devices)) == 8
local_rank = int(os.environ['LOCAL_RANK'])
assert 0 <= local_rank < len(devices)
os.environ['OMNI_CHECKPOINT_LAUNCHER_LOCAL_RANK'] = str(local_rank)
os.environ['CUDA_VISIBLE_DEVICES'] = devices[local_rank]
os.environ['LOCAL_RANK'] = '0'
target = Path(__file__).with_name('verify_formal_checkpoint_gpu.py')
sys.argv[0] = str(target)
runpy.run_path(str(target), run_name='__main__')
