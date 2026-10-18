"""Isolated actual ART actor, microbatch and checkpoint diagnostic.

Uses saved native inputs and the selected frozen engine; never a scientific run.
CPU mode validates contracts and a conservative output budget before GPU use.
"""
import argparse
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import signal
import threading
import time
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf
from verl.utils.config import omega_conf_to_dataclass
from verl.utils import tensordict_utils as tu
from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.workers.engine.fsdp.diffusers_impl import ARTDiffusersFSDPEngine
from verl_omni.workers.engine.lora_adapter_mixin import LoRAAdapterMixin
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniHttpServer

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')
parser = argparse.ArgumentParser()
parser.add_argument('--cpu-config-only', action='store_true')
parser.add_argument('--captures', type=Path, required=True)
args = parser.parse_args()
OUTPUT = Path(os.environ['OUTPUT_DIR']).resolve()
assert OUTPUT.is_relative_to(ROOT / 'preflight/runtime_gpu')
assert args.captures.resolve().is_relative_to(ROOT / 'preflight/runtime_gpu')


def publish(name, value):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / name
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str) + '\n')
    tmp.replace(path)


def configs():
    raw = OmegaConf.load(ROOT / 'preflight/configs/A0/resolved.yaml')
    server = object.__new__(vLLMOmniHttpServer)
    server.config = server._init_config(omega_conf_to_dataclass(raw.actor_rollout_ref.rollout))
    model = server._init_model_config(raw.actor_rollout_ref.model)
    engine = omega_conf_to_dataclass(OmegaConf.merge(raw.actor_rollout_ref.actor.fsdp_config,
                                                    {'strategy': raw.actor_rollout_ref.actor.strategy}))
    optim = omega_conf_to_dataclass(OmegaConf.merge(raw.actor_rollout_ref.actor.optim,
                                                   {'total_training_steps': 75}))
    # Diagnostic only. No canonical launch config is modified by this script.
    checkpoint = omega_conf_to_dataclass(OmegaConf.merge(raw.actor_rollout_ref.actor.checkpoint,
                                                        {'save_lora_only': True}))
    assert engine.strategy == 'fsdp2' and model.lora_rank == 32
    assert model.pipeline.video_cfg_scale == 1.5 and model.pipeline.audio_cfg_scale == 3.0
    assert not engine.param_offload and not engine.optimizer_offload
    inspect.signature(ARTDiffusersFSDPEngine).bind(model, engine, optim, checkpoint)
    return model, engine, optim, checkpoint


def fixed_inputs():
    captures, hashes = [], {}
    for seed in (42, 43):
        path = args.captures / f'seed_{seed}_rank_0.pt'
        meta = json.loads(path.with_suffix('.pt.json').read_text())
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        assert digest == meta['sha256']
        hashes[str(path)] = digest
        value = torch.load(path, weights_only=True, map_location='cpu')
        assert value['identity']['source_sha256'] == os.environ['SOURCE_SNAPSHOT_SHA256']
        assert [r['index'] for r in value['records']] == [0, 9, 18]
        captures.append(value)
    states = torch.cat([torch.stack([r['state'] for r in c['records']], dim=1) for c in captures])
    nexts = torch.cat([torch.stack([r['actual_next_rounded'] for r in c['records']], dim=1) for c in captures])
    sigmas = torch.tensor([[float(r['sigma']) for r in c['records']] for c in captures])
    next_sigmas = torch.tensor([[float(r['sigma_next']) for r in c['records']] for c in captures])
    split = captures[0]['records'][0]['video_seq_len']
    generator = torch.Generator().manual_seed(91742)
    guidance = torch.randn(states[:, 0].shape, generator=generator)
    for section in (slice(None, split), slice(split, None)):
        x = guidance[:, section]
        guidance[:, section] = x / x.square().mean(dim=(1, 2), keepdim=True).sqrt()
    values = dict(train_latents=states.to(torch.bfloat16), train_latents_next=nexts.to(torch.bfloat16),
                  train_timesteps=sigmas * 1000, train_timesteps_next=next_sigmas * 1000,
                  video_seq_len=torch.full((2,), split), art_guidance=guidance)
    for key in captures[0]['condition']:
        values[key] = torch.cat([c['condition'][key] for c in captures])
    for key in ('prompt_embeds', 'negative_prompt_embeds'):
        values[key + '_mask'] = torch.ones(values[key].shape[:2], dtype=torch.bool)
    batch = tu.get_tensordict(values)
    sampler = torch.cat([torch.stack([r['actual_sampler_velocity'] for r in c['records']], dim=1)
                         for c in captures])
    return batch, sampler, hashes


def cpu_contract(model, engine, optim, checkpoint, batch, hashes):
    from accelerate import init_empty_weights
    from diffusers import LTX2VideoTransformer3DModel
    config = json.loads((Path(model.local_path) / model.transformer_subfolder / 'config.json').read_text())
    with init_empty_weights():
        module = LTX2VideoTransformer3DModel.from_config(config)
        module = LoRAAdapterMixin._build_lora_module(SimpleNamespace(model_config=model), module)
    adapter_numel = {a: sum(p.numel() for n, p in module.named_parameters() if f'.{a}.' in n)
                     for a in ('default', 'old')}
    assert adapter_numel['default'] == adapter_numel['old'] > 0
    # Float32 ceiling per adapter; Adam's two moments plus parameter/gradient
    # copies and two checkpoint generations, inflated 2x plus fixed 8 GiB.
    bound = sum(adapter_numel.values()) * 4 * 12 * 2 + 8 * 2**30
    cls = DiffusionModelBase.get_class(model)
    for step in range(3):
        positive, negative = cls.prepare_model_inputs(module, model, batch['train_latents'][:, step],
            batch['train_timesteps'][:, step], batch['prompt_embeds'], batch['prompt_embeds_mask'],
            batch['negative_prompt_embeds'], batch['negative_prompt_embeds_mask'], batch, step)
        assert positive['hidden_states'].shape[0] == 2 and negative is not None
    from verl.workers.engine.utils import prepare_micro_batches
    from verl_omni.workers.utils.losses import diffusion_loss
    actor = omega_conf_to_dataclass(OmegaConf.load(ROOT / 'preflight/configs/A0/resolved.yaml').actor_rollout_ref.actor)
    for microbatch in (1, 2):
        data = batch.clone(recurse=True)
        tu.assign_non_tensor(data, use_dynamic_bsz=False, micro_batch_size_per_gpu=microbatch)
        pieces, _ = prepare_micro_batches(data)
        assert len(pieces) == 2 // microbatch
        for piece in pieces:
            assert len(piece) == microbatch
            tu.assign_non_tensor(piece, gradient_accumulation_steps=len(pieces) * 3, sp_size=1)
            sigma = piece['train_timesteps'][:, 0] / 1000
            inputs = ARTDiffusersFSDPEngine.art_loss_inputs(piece, 0, piece['train_latents'][:, 0],
                                                          sigma[:, None, None], 1000)
            loss_data = tu.get_tensordict(inputs)
            tu.assign_non_tensor(loss_data, gradient_accumulation_steps=len(pieces) * 3, sp_size=1)
            velocity = torch.zeros_like(piece['train_latents'][:, 0])
            loss, _ = diffusion_loss(actor, {'forward_prediction': velocity, 'old_prediction': velocity,
                                            't_expanded': sigma[:, None, None]}, loss_data)
            assert torch.isfinite(loss)
    report = {'stage': 'cpu_contract_passed', 'capture_hashes': hashes,
              'adapter_numel': adapter_numel, 'conservative_max_probe_output_bytes': bound,
              'scope': 'meta model and exact named LoRA shapes, actual conditioned-input contracts; no GPU forwards',
              'diagnostic_dp': 1, 'scientific_dp': 16, 'selected_source': os.environ['SOURCE_SNAPSHOT_SHA256'],
              'guidance': 'fixed synthetic per-modality unit RMS; not a K8 covariance estimate',
              'checkpoint_save_lora_only': True, 'canonical_checkpoint_config_modified': False}
    publish('actor_gpu_state.json', report)
    print(json.dumps(report), flush=True)
    return report


def local_tensor(value):
    return value.to_local() if hasattr(value, 'to_local') else value


def adapter_copy(engine):
    return {n: local_tensor(p).detach().cpu().clone() for n, p in engine.module.named_parameters()
            if 'lora_' in n or '.adapter_' in n}


def tree_copy(value):
    if isinstance(value, torch.Tensor):
        return local_tensor(value).detach().cpu().clone()
    if isinstance(value, dict):
        return {k: tree_copy(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(tree_copy(v) for v in value)
    return copy.deepcopy(value)


def tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(tree_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(tree_equal(x, y) for x, y in zip(a, b))
    return a == b


def difference(first, second):
    assert first.keys() == second.keys()
    aa = bb = ab = dd = 0.0
    maximum = 0.0
    for name in first:
        a, b = first[name].float().reshape(-1), second[name].float().reshape(-1)
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        aa += a.double().square().sum().item()
        bb += b.double().square().sum().item()
        ab += (a.double() * b.double()).sum().item()
        dd += (a.double() - b.double()).square().sum().item()
        maximum = max(maximum, (a - b).abs().max().item())
    return {'relative_l2': (dd / max(aa, 1e-30)) ** .5,
            'norm_ratio_second_over_first': (bb / max(aa, 1e-30)) ** .5,
            'cosine': ab / max((aa * bb) ** .5, 1e-30), 'max_abs': maximum,
            'first_norm': aa ** .5, 'second_norm': bb ** .5}


def reset_initial(engine, initial, scheduler, rng):
    with torch.no_grad():
        for name, p in engine.module.named_parameters():
            if name in initial:
                local_tensor(p).copy_(initial[name].to(p.device))
    engine.optimizer.state.clear()
    engine.optimizer_zero_grad()
    engine.lr_scheduler.load_state_dict(copy.deepcopy(scheduler))
    engine.checkpoint_manager.load_rng_state(rng)
    engine._set_adapter('default')


def run_branch(engine, batch, sampler, actor, microbatch, use_cache, label):
    from functools import partial
    from verl_omni.workers.utils.losses import diffusion_loss
    native_loss = partial(diffusion_loss, actor)
    split = int(batch['video_seq_len'][0])
    counter, observations = 0, []

    def audited_loss(*, model_output, data, dp_group):
        nonlocal counter
        step = counter % 3
        start = (counter // 3) * microbatch
        cached = sampler[start:start + microbatch, step].to('cuda')
        current, old = model_output['forward_prediction'], model_output['old_prediction']
        row = {'step_index': (0, 9, 18)[step], 'sample_start': start}
        for name, section in (('video', slice(None, split)), ('audio', slice(split, None))):
            v, o, c = current[:, section].detach(), old[:, section], cached[:, section]
            row[name] = {'initial_restoration_mse': float((v - o).square().mean()),
                         'actor_sampler_relative_rms': float((o - c).square().mean().sqrt() /
                                                               c.square().mean().sqrt().clamp_min(1e-12))}
        if use_cache:
            model_output = dict(model_output, old_prediction=cached.detach())
        loss, metrics = native_loss(model_output=model_output, data=data, dp_group=dp_group)
        assert torch.isfinite(loss)
        row['scaled_loss'] = float(loss.detach())
        observations.append(row)
        counter += 1
        print('ACTOR_DIAGNOSTIC', label, 'forward_backward', counter, flush=True)
        return loss, metrics

    data = batch.clone(recurse=True)
    tu.assign_non_tensor(data, micro_batch_size_per_gpu=microbatch, use_dynamic_bsz=False,
                         use_no_sync_for_gradient_accumulation=False, return_model_output=False)
    engine.optimizer_zero_grad()
    torch.cuda.synchronize()
    started = time.monotonic()
    with engine.train_mode():
        engine.forward_backward_batch(data, audited_loss, forward_only=False)
        torch.cuda.synchronize()
        grads = {n: local_tensor(p.grad).detach().cpu().clone() for n, p in engine.module.named_parameters()
                 if p.grad is not None}
        assert grads and all('lora_' in name and '.default.' in name for name in grads)
        assert all(torch.isfinite(g).all() for g in grads.values())
        elapsed = time.monotonic() - started
        grad_norm = engine.optimizer_step()
        lr = engine.lr_scheduler_step()
    report = {'stage': 'finite_diagnostic_update', 'label': label, 'microbatch': microbatch,
              'accumulation': 2 // microbatch * 3, 'backward_calls': counter,
              'summed_scaled_loss': sum(r['scaled_loss'] for r in observations),
              'forward_backward_seconds': elapsed, 'grad_norm_before_clip': grad_norm,
              'lr': lr, 'observations': observations,
              'cache_branch_still_recomputes_old_for_control': use_cache}
    publish(label + '.json', report)
    return report, grads, adapter_copy(engine)


def old_predictions(engine, batch, adapter='old'):
    from verl_omni.pipelines.utils import forward
    data = batch.to('cuda')
    predictions, timings = [], []
    for step in range(3):
        positive, negative, _, _, _ = engine.prepare_model_inputs(data, step)
        torch.cuda.synchronize()
        began = time.monotonic()
        with engine.use_adapter(adapter), torch.no_grad():
            prediction = forward(module=engine.module, model_config=engine.model_config,
                                 model_inputs=positive, negative_model_inputs=negative)
        torch.cuda.synchronize()
        timings.append(time.monotonic() - began)
        assert torch.isfinite(prediction).all()
        predictions.append(prediction.detach().cpu())
    return torch.stack(predictions, dim=1), timings


def checkpoint_roundtrip(engine, batch):
    ensure_probe_space(64 * 2**30)
    directory = OUTPUT / 'checkpoint_lora_complete'
    assert not directory.exists()
    directory.mkdir()
    expected_model = adapter_copy(engine)
    expected_optim = tree_copy(engine.optimizer.state_dict())
    expected_scheduler = tree_copy(engine.lr_scheduler.state_dict())
    before_prediction, before_times = old_predictions(engine, batch)
    before_current, _ = old_predictions(engine, batch, 'default')
    engine.save_checkpoint(str(directory), global_step=1)
    files = []
    for path in sorted(directory.rglob('*')):
        if path.is_file():
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            files.append({'file': str(path.relative_to(directory)), 'bytes': path.stat().st_size,
                          'sha256': digest})
    payload = torch.load(directory / 'model_world_size_1_rank_0.pt', map_location='cpu', weights_only=False)
    assert payload and all('lora_' in key or '.adapter_' in key for key in payload)
    assert any('.default.' in key for key in payload) and any('.old.' in key for key in payload)
    assert set(payload) == set(expected_model), 'Checkpoint omitted an adapter state'
    del payload
    extra = torch.load(directory / 'extra_state_world_size_1_rank_0.pt', map_location='cpu', weights_only=False)
    # Perturb both policy adapters and Adam moments, scheduler and all RNGs.
    with torch.no_grad():
        for n, p in engine.module.named_parameters():
            if n in expected_model:
                local_tensor(p).add_(.125)
        for state in engine.optimizer.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    local_tensor(value).add_(.25)
    engine.lr_scheduler.step()
    torch.rand(11); torch.rand(11, device='cuda'); random.random(); np.random.random(11)
    assert not tree_equal(expected_model, adapter_copy(engine))
    engine.load_checkpoint(str(directory), del_local_after_load=False)
    actual_rng = engine.checkpoint_manager.get_rng_state()
    checks = {'all_named_adapters_bitwise': tree_equal(expected_model, adapter_copy(engine)),
              'adam_state_bitwise': tree_equal(expected_optim, tree_copy(engine.optimizer.state_dict())),
              'scheduler_exact': tree_equal(expected_scheduler, tree_copy(engine.lr_scheduler.state_dict())),
              'all_rng_exact': tree_equal(extra['rng'], actual_rng)}
    assert all(checks.values()), checks
    after_prediction, after_times = old_predictions(engine, batch)
    after_current, _ = old_predictions(engine, batch, 'default')
    checks['fixed_input_old_prediction_bitwise'] = torch.equal(before_prediction, after_prediction)
    checks['fixed_input_current_prediction_bitwise'] = torch.equal(before_current, after_current)
    assert all(checks.values()), checks
    result = {'stage': 'passed', 'scope': 'actual world1 FSDP2 engine named adapters + Adam + scheduler + CPU/CUDA/Python/NumPy RNG; not trainer cursor/controller or world16 readback',
              'checks': checks, 'files': files, 'total_bytes': sum(x['bytes'] for x in files),
              'old_forward_before_seconds': before_times, 'old_forward_after_seconds': after_times}
    publish('checkpoint_roundtrip.json', result)
    return result


def ensure_probe_space(additional_bytes=0):
    stat = os.statvfs(ROOT)
    free = stat.f_bavail * stat.f_frsize
    if free < 100 * 2**30 + additional_bytes:
        raise RuntimeError(f'Isolated diagnostic space guard: {free} available bytes')


def monitor_probe_space():
    while True:
        time.sleep(10)
        try:
            ensure_probe_space()
        except BaseException as error:
            try:
                publish('storage_guard_stop.json', {'reason': str(error), 'unix': time.time(),
                                                    'pid': os.getpid(), 'signal_target': 'self only'})
            finally:
                os.kill(os.getpid(), signal.SIGTERM)
            return


def gpu_main(configuration, batch, sampler, hashes):
    import datetime
    ensure_probe_space()
    threading.Thread(target=monitor_probe_space, daemon=True).start()
    torch.cuda.set_device(0)
    torch.distributed.init_process_group('nccl', timeout=datetime.timedelta(seconds=600))
    assert torch.distributed.get_world_size() == 1
    state = {'stage': 'actual_actor_initializing', 'scientific_optimizer_updates': 0,
             'diagnostic_dp': 1, 'production_dp': 16, 'capture_hashes': hashes,
             'selected_source': os.environ['SOURCE_SNAPSHOT_SHA256'],
             'guidance': 'fixed synthetic per-modality unit RMS; not a K8 reward covariance',
             'started_unix': time.time()}
    publish('actor_gpu_state.json', state)
    engine = ARTDiffusersFSDPEngine(*configuration)
    engine.initialize()
    engine.copy_adapter('default', 'old')
    initial = adapter_copy(engine)
    assert all(torch.count_nonzero(v) == 0 for n, v in initial.items() if 'lora_B' in n)
    assert all('lora_' in n and '.default.' in n for n, p in engine.module.named_parameters() if p.requires_grad)
    scheduler = copy.deepcopy(engine.lr_scheduler.state_dict())
    rng = engine.checkpoint_manager.get_rng_state()
    actor = omega_conf_to_dataclass(OmegaConf.load(ROOT / 'preflight/configs/A0/resolved.yaml').actor_rollout_ref.actor)
    branches = {}
    for microbatch in (2, 1):
        label = f'recompute_mb{microbatch}'
        state['stage'] = label
        publish('actor_gpu_state.json', state)
        reset_initial(engine, initial, scheduler, rng)
        branches[microbatch] = run_branch(engine, batch, sampler, actor, microbatch, False, label)
    r2, g2, p2 = branches[2]
    r1, g1, p1 = branches[1]
    grad_comparison = difference(g2, g1)
    updates2 = {n: p2[n] - initial[n] for n in p2}
    updates1 = {n: p1[n] - initial[n] for n in p1}
    state['microbatch'] = {'loss_mb2': r2['summed_scaled_loss'], 'loss_mb1': r1['summed_scaled_loss'],
                           'gradient': grad_comparison, 'parameter_update': difference(updates2, updates1)}
    state['stage'] = 'checkpoint_roundtrip'
    publish('actor_gpu_state.json', state)
    state['checkpoint'] = checkpoint_roundtrip(engine, batch)
    reset_initial(engine, initial, scheduler, rng)
    state['stage'] = 'cache_target_and_parameter_gradient_control'
    publish('actor_gpu_state.json', state)
    cache, cache_grads, _ = run_branch(engine, batch, sampler, actor, 1, True, 'actual_sampler_cached_target_mb1')
    state['cache_target'] = {'recompute_loss': r1['summed_scaled_loss'], 'actual_sampler_target_loss': cache['summed_scaled_loss'],
                            'parameter_gradient_difference': difference(g1, cache_grads),
                            'actual_sampler_cache_tensor_bytes': sampler.numel() * sampler.element_size(),
                            'target_difference_equals_old_minus_sampler_with_fixed_guidance': True,
                            'production_art_keeps_old_recompute': True}
    state.update(stage='diagnostics_completed', finished_unix=time.time(), diagnostic_optimizer_calls=3,
                 peak_gpu_memory_bytes=torch.cuda.max_memory_allocated())
    publish('actor_gpu_state.json', state)
    torch.distributed.destroy_process_group()
    print(json.dumps(state), flush=True)


if __name__ == '__main__':
    configuration = configs()
    batch, sampler, hashes = fixed_inputs()
    if args.cpu_config_only:
        cpu_contract(*configuration, batch, hashes)
    else:
        gpu_main(configuration, batch, sampler, hashes)
