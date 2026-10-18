"""Independent world16 native checkpoint GPU reload; never performs an optimizer update."""
import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from verl.utils.config import omega_conf_to_dataclass
from verl_omni.trainer.diffusion.diffusion_checkpoint import validate_complete_rollout
from verl_omni.workers.engine.fsdp.diffusers_impl import ARTDiffusersFSDPEngine, OmniNFTDiffusersFSDPEngine
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniHttpServer

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def fingerprint(value):
    def encode(item):
        if isinstance(item, torch.Tensor):
            local = shared.local_tensor(item).detach().cpu().contiguous()
            return {'dtype': str(local.dtype), 'shape': list(local.shape),
                    'sha256': hashlib.sha256(local.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
        if isinstance(item, np.ndarray):
            return {'dtype': str(item.dtype), 'shape': list(item.shape),
                    'sha256': hashlib.sha256(item.tobytes()).hexdigest()}
        if isinstance(item, dict):
            return {str(k): encode(v) for k, v in item.items()}
        if isinstance(item, (tuple, list)):
            return [encode(v) for v in item]
        return item
    return hashlib.sha256(json.dumps(encode(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def configuration(raw):
    server = object.__new__(vLLMOmniHttpServer)
    server.config = server._init_config(omega_conf_to_dataclass(raw.actor_rollout_ref.rollout))
    model = server._init_model_config(raw.actor_rollout_ref.model)
    engine = omega_conf_to_dataclass(OmegaConf.merge(raw.actor_rollout_ref.actor.fsdp_config,
                                                   {'strategy': raw.actor_rollout_ref.actor.strategy}))
    optim = omega_conf_to_dataclass(raw.actor_rollout_ref.actor.optim)
    checkpoint = omega_conf_to_dataclass(raw.actor_rollout_ref.actor.checkpoint)
    cls = OmniNFTDiffusersFSDPEngine if model.algorithm == 'omni_nft' else ARTDiffusersFSDPEngine
    assert engine.strategy == 'fsdp2' and model.lora_rank == 32 and model.lora_alpha == 64
    assert not engine.param_offload and not engine.optimizer_offload
    assert raw.trainer.nnodes * raw.trainer.n_gpus_per_node == 16
    assert raw.actor_rollout_ref.actor.checkpoint.save_lora_only is True
    assert raw.trainer.campaign.source_snapshot_sha256 == os.environ['SOURCE_SNAPSHOT_SHA256']
    assert raw.trainer.expected_optimizer_updates_per_rollout == 2
    return cls, (model, engine, optim, checkpoint)


def fixed_batch(is_nft):
    if is_nft:
        import verify_nft_gpu
        return verify_nft_gpu.inputs()[0]
    return shared.fixed_inputs()[0]


def predictions(engine, batch, adapter):
    from verl_omni.pipelines.utils import forward
    data = batch.to(torch.device('cuda', torch.cuda.current_device()))
    positive, negative, *_ = engine.prepare_model_inputs(data, 0)
    with engine.use_adapter(adapter), torch.no_grad():
        output = forward(module=engine.module, model_config=engine.model_config,
                         model_inputs=positive, negative_model_inputs=negative)
    torch.cuda.synchronize()
    values = output if isinstance(output, (tuple, list)) else [output]
    assert all(torch.isfinite(value).all() for value in values)
    return shared.tree_copy(output)


def publish(output, rank, result):
    output.mkdir(parents=True, exist_ok=True)
    path = output / f'checkpoint_gpu_rank_{rank}.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(path)


def gpu_check(args, raw, cls, configs):
    rank, local_rank = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    output = Path(os.environ['OUTPUT_DIR']).resolve()
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group('nccl', timeout=datetime.timedelta(seconds=3600))
    assert torch.distributed.get_world_size() == 16
    result = {'stage': 'initializing_actual_native_engine', 'started_unix': time.time(),
              'rank': rank, 'world_size': 16, 'checkpoint': str(args.checkpoint),
              'physical_visible_device': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'launcher_local_rank': os.environ.get('OMNI_CHECKPOINT_LAUNCHER_LOCAL_RANK'),
              'scientific_optimizer_updates': 0, 'optimizer_step_calls': 0,
              'source_sha256': os.environ['SOURCE_SNAPSHOT_SHA256'],
              'probe_attempt': os.environ['OMNI_ATTEMPT'], 'probe_nonce': os.environ['OMNI_NONCE'],
              'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest()}
    publish(output, rank, result)
    if rank == 0:
        validate_complete_rollout(str(args.checkpoint), raw)
    torch.distributed.barrier()
    engine = cls(*configs)
    engine.initialize()
    directory = args.checkpoint / 'actor'
    result['stage'] = 'loading_actual_formal_checkpoint'
    publish(output, rank, result)
    engine.load_checkpoint(str(directory), del_local_after_load=False)
    payload = torch.load(directory / f'model_world_size_16_rank_{rank}.pt', weights_only=False)
    assert payload and all('lora_' in key or '.adapter_' in key for key in payload)
    assert any('.default.' in key for key in payload) and any('.old.' in key for key in payload)
    expected_model = shared.tree_copy(payload)
    del payload
    expected_optim = shared.tree_copy(torch.load(directory / f'optim_world_size_16_rank_{rank}.pt', weights_only=False))
    extra = torch.load(directory / f'extra_state_world_size_16_rank_{rank}.pt', map_location='cpu', weights_only=False)
    checks = {'first_load_all_named_adapters_exact': equal(expected_model, shared.adapter_copy(engine)),
              'first_load_adam_state_exact': equal(expected_optim, shared.tree_copy(engine.optimizer.state_dict())),
              'first_load_scheduler_exact': equal(extra['lr_scheduler'], shared.tree_copy(engine.lr_scheduler.state_dict())),
              'first_load_all_worker_rng_exact': equal(extra['rng'], engine.checkpoint_manager.get_rng_state())}
    result.update(stage='first_native_load_compared', checks=checks)
    publish(output, rank, result)
    assert all(checks.values()), checks
    batch = fixed_batch(cls is OmniNFTDiffusersFSDPEngine)
    before = {adapter: predictions(engine, batch, adapter) for adapter in ('old', 'default')}
    with torch.no_grad():
        for name, parameter in engine.module.named_parameters():
            if name in expected_model:
                shared.local_tensor(parameter).add_(.125)
        for state in engine.optimizer.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    shared.local_tensor(value).add_(.25)
    engine.lr_scheduler.step()
    torch.rand(11); torch.rand(11, device=torch.device('cuda', local_rank))
    np.random.random(11); random.random()
    assert not equal(expected_model, shared.adapter_copy(engine))
    result['stage'] = 'perturbed_state_reloading_same_native_checkpoint'
    publish(output, rank, result)
    engine.load_checkpoint(str(directory), del_local_after_load=False)
    checks.update(reload_all_named_adapters_exact=equal(expected_model, shared.adapter_copy(engine)),
                  reload_adam_state_exact=equal(expected_optim, shared.tree_copy(engine.optimizer.state_dict())),
                  reload_scheduler_exact=equal(extra['lr_scheduler'], shared.tree_copy(engine.lr_scheduler.state_dict())),
                  reload_all_worker_rng_exact=equal(extra['rng'], engine.checkpoint_manager.get_rng_state()))
    assert all(checks.values()), checks
    after = {adapter: predictions(engine, batch, adapter) for adapter in ('old', 'default')}
    checks.update(fixed_input_old_prediction_bitwise=equal(before['old'], after['old']),
                  fixed_input_current_prediction_bitwise=equal(before['default'], after['default']))
    result.update(stage='gpu_readback_checks_finished', checks=checks,
                  fingerprints={'adapters': fingerprint(expected_model), 'adam': fingerprint(expected_optim),
                                'scheduler': fingerprint(extra['lr_scheduler']), 'rng': fingerprint(extra['rng']),
                                'prediction_before': fingerprint(before), 'prediction_after': fingerprint(after)},
                  peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(), finished_unix=time.time())
    publish(output, rank, result)
    passed = torch.tensor(int(all(checks.values())), device=torch.device('cuda', local_rank))
    torch.distributed.all_reduce(passed, op=torch.distributed.ReduceOp.MIN)
    assert passed.item() == 1, 'At least one rank failed exact native GPU readback'
    result['stage'] = 'passed'
    publish(output, rank, result)
    torch.distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--captures', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--cpu-config-only', action='store_true')
    args = parser.parse_args()
    assert args.config.resolve().is_relative_to(ROOT)
    assert args.captures.resolve().is_relative_to(ROOT / 'preflight/runtime_gpu')
    output = Path(os.environ['OUTPUT_DIR']).resolve()
    assert output.is_relative_to(ROOT / 'preflight/runtime_gpu')
    # The existing diagnostic input helpers parse only their own capture option.
    original_argv = sys.argv
    sys.argv = [sys.argv[0], '--captures', str(args.captures)]
    global shared
    import verify_actor_gpu as shared
    sys.argv = original_argv
    raw = OmegaConf.load(args.config)
    cls, configs = configuration(raw)
    _, _, capture_hashes = shared.fixed_inputs()
    if args.checkpoint is not None:
        args.checkpoint = args.checkpoint.resolve()
        assert args.checkpoint.is_relative_to(ROOT / 'attempts')
    if args.cpu_config_only:
        import inspect
        inspect.signature(cls).bind(*configs)
        result = {'stage': 'config_and_native_contract_checked', 'source_sha256': os.environ['SOURCE_SNAPSHOT_SHA256'],
                  'config': str(args.config.resolve()), 'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
                  'capture_hashes': capture_hashes, 'world_size': 16, 'engine_class': cls.__name__,
                  'checkpoint_integrity_verified': False, 'gpu_readback_verified': False,
                  'scientific_optimizer_updates': 0, 'gpu_checkpoint_writes': False}
        if args.checkpoint is not None:
            state = validate_complete_rollout(str(args.checkpoint), raw)
            result.update(checkpoint_integrity_verified=True, checkpoint=str(args.checkpoint),
                          checkpoint_manifest_sha256=hashlib.sha256((args.checkpoint / 'checkpoint_manifest.json').read_bytes()).hexdigest(),
                          saved_rollout=state['rollout'], saved_optimizer_updates=state['optimizer_updates'])
        (output / 'checkpoint_gpu_cpu_contract.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2), flush=True)
    else:
        assert args.checkpoint is not None, 'GPU readback requires an actual complete formal checkpoint'
        proof_path = Path(os.environ['OMNI_CHECKPOINT_READBACK_PROOF']).resolve()
        assert proof_path.is_relative_to(ROOT / 'preflight/runtime_gpu')
        proof = json.loads(proof_path.read_text())
        assert proof['checkpoint_integrity_verified'] is True and proof['world_size'] == 16
        assert proof['gpu_readback_verified'] is False
        assert proof['source_sha256'] == os.environ['SOURCE_SNAPSHOT_SHA256']
        assert proof['config_sha256'] == hashlib.sha256(args.config.read_bytes()).hexdigest()
        assert proof['checkpoint'] == str(args.checkpoint) and proof['capture_hashes'] == capture_hashes
        assert proof['checkpoint_manifest_sha256'] == hashlib.sha256(
            (args.checkpoint / 'checkpoint_manifest.json').read_bytes()).hexdigest()
        gpu_check(args, raw, cls, configs)


if __name__ == '__main__':
    main()
