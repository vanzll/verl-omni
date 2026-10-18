"""Verify native complete-state integrity/controller/cursor; GPU readback is separate."""
import argparse
import gc
import hashlib
import json
import math
import random
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf
from verl.utils import hf_tokenizer
from verl_omni.trainer.diffusion.diffusion_checkpoint import restore_controller_state, validate_complete_rollout
from verl_omni.trainer.diffusion.ray_diffusion_trainer import BaseRayDiffusionTrainer

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')


def canonical(value):
    if isinstance(value, torch.Tensor):
        return {'tensor_dtype': str(value.dtype), 'shape': list(value.shape), 'values': canonical(value.cpu().tolist())}
    if isinstance(value, np.ndarray):
        return {'array_dtype': str(value.dtype), 'shape': list(value.shape), 'values': canonical(value.tolist())}
    if isinstance(value, np.generic):
        return canonical(value.item())
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError(f'Unsupported canonical batch value: {type(value).__name__}')


def batch_digest(batch):
    return hashlib.sha256(json.dumps(canonical(batch), sort_keys=True, allow_nan=False).encode()).hexdigest()


def loaders(config):
    # Match TaskRunner's local model/tokenizer fallback before constructing data.
    if config.actor_rollout_ref.model.tokenizer_path is None:
        base = Path(config.actor_rollout_ref.model.path)
        candidate = base / 'tokenizer'
        config.actor_rollout_ref.model.tokenizer_path = str(candidate if candidate.is_dir() else base)
    tokenizer = hf_tokenizer(config.actor_rollout_ref.model.tokenizer_path,
                             trust_remote_code=config.data.get('trust_remote_code', False))
    holder = SimpleNamespace(config=config, tokenizer=tokenizer, processor=None)
    BaseRayDiffusionTrainer._create_dataloader(holder, None, None, None, None)
    return holder


def close_iterator(iterator):
    if hasattr(iterator, '_shutdown_workers'):
        iterator._shutdown_workers()
    del iterator
    gc.collect()


def cursor_check(config, saved_state):
    fingerprints = []
    # A separate loader each time exercises actual sampler/worker reconstruction.
    for _ in range(2):
        holder = loaders(OmegaConf.create(OmegaConf.to_container(config, resolve=True)))
        holder.train_dataloader.load_state_dict(saved_state)
        iterator = iter(holder.train_dataloader)
        try:
            fingerprints.append(batch_digest(next(iterator)))
        finally:
            close_iterator(iterator)
            del holder
            gc.collect()
    assert fingerprints[0] == fingerprints[1], 'Restored next batch differs'
    return {'native_next_batch_equal_on_two_independent_restores': True,
            'next_batch_sha256': fingerprints[0],
            'num_workers': config.data.dataloader_num_workers,
            'train_batch_size': config.data.train_batch_size}


def controller_check(state):
    holder = SimpleNamespace(_campaign_attempt_startup_gpu_seconds=0.0, _campaign_resume_extra_gpu_seconds=0.0)
    draws = []
    for _ in range(2):
        torch.rand(7); np.random.random(7); random.random()
        restore_controller_state(holder, state)
        assert torch.equal(torch.get_rng_state(), state['controller_rng']['torch'])
        actual_numpy = np.random.get_state()
        expected_numpy = state['controller_rng']['numpy']
        assert actual_numpy[0] == expected_numpy[0] and np.array_equal(actual_numpy[1], expected_numpy[1])
        assert actual_numpy[2:] == expected_numpy[2:]
        assert random.getstate() == state['controller_rng']['random']
        assert holder._campaign_training_gpu_seconds == state['training_gpu_seconds']
        assert holder._campaign_prior_end_to_end_gpu_seconds == state['end_to_end_gpu_seconds']
        assert holder._campaign_parent_wandb == state['wandb']
        draws.append(batch_digest({'torch': torch.rand(11), 'numpy': np.random.random(11), 'python': random.random()}))
    assert draws[0] == draws[1]
    return {'controller_all_cpu_rng_exact': True, 'controller_cost_and_parent_wandb_exact': True,
            'next_random_draws_equal_after_two_restores': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--cursor-self-check', action='store_true')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    assert args.output.resolve().is_relative_to(ROOT)
    assert args.config.resolve().is_relative_to(ROOT)
    config = OmegaConf.load(args.config)
    result = {'started_unix': time.time(), 'scope': 'CPU native file/controller/cursor verification; NOT world16 GPU readback',
              'config': str(args.config.resolve()), 'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
              'scientific_optimizer_updates': 0}
    if args.cursor_self_check:
        assert args.checkpoint is None
        holder = loaders(config)
        iterator = iter(holder.train_dataloader)
        try:
            for _ in range(3):
                next(iterator)
            saved_state = holder.train_dataloader.state_dict()
            expected_next = batch_digest(next(iterator))
        finally:
            close_iterator(iterator)
            del holder
            gc.collect()
        result['cursor'] = cursor_check(config, saved_state)
        assert result['cursor']['next_batch_sha256'] == expected_next
        result['cursor']['matches_uninterrupted_fourth_batch'] = True
        result['mode'] = 'actual_dataset_loader_self_check_not_formal_checkpoint'
    else:
        assert args.checkpoint is not None
        checkpoint = args.checkpoint.resolve()
        assert checkpoint.is_relative_to(ROOT / 'attempts')
        assert config.trainer.nnodes * config.trainer.n_gpus_per_node == 16
        state = validate_complete_rollout(str(checkpoint), config)
        result.update(checkpoint=str(checkpoint), world_size=16, rollout=state['rollout'],
                      saved_optimizer_updates=state['optimizer_updates'],
                      checkpoint_manifest_sha256=hashlib.sha256((checkpoint / 'checkpoint_manifest.json').read_bytes()).hexdigest())
        result['controller'] = controller_check(state)
        result['cursor'] = cursor_check(config, torch.load(checkpoint / 'data.pt', map_location='cpu', weights_only=False))
        result['mode'] = 'actual_formal_complete_checkpoint_cpu_verification'
    result.update(status='passed', finished_unix=time.time())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
