"""Actual OmniNFT engine on fixed diagnostic clean estimates and forward noise.

Clean estimates come from a recorded native state minus sigma times its actual
sampler velocity, not from final generated clean endpoints. Reward probabilities
are explicitly synthetic. This checks the native GPU path, never science scores.
"""
import copy
import datetime
import inspect
import json
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.engine.utils import prepare_micro_batches
from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.workers.engine.fsdp.diffusers_impl import OmniNFTDiffusersFSDPEngine
from verl_omni.workers.utils.losses import diffusion_loss
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniHttpServer
import verify_actor_gpu as shared

ROOT, OUTPUT, args = shared.ROOT, shared.OUTPUT, shared.args


def configuration():
    raw = OmegaConf.load(ROOT / 'preflight/configs/N0/resolved.yaml')
    server = object.__new__(vLLMOmniHttpServer)
    server.config = server._init_config(omega_conf_to_dataclass(raw.actor_rollout_ref.rollout))
    model = server._init_model_config(raw.actor_rollout_ref.model)
    engine = omega_conf_to_dataclass(OmegaConf.merge(raw.actor_rollout_ref.actor.fsdp_config,
                                                   {'strategy': raw.actor_rollout_ref.actor.strategy}))
    optim = omega_conf_to_dataclass(OmegaConf.merge(raw.actor_rollout_ref.actor.optim,
                                                  {'total_training_steps': 75}))
    checkpoint = omega_conf_to_dataclass(OmegaConf.merge(raw.actor_rollout_ref.actor.checkpoint,
                                                       {'save_lora_only': True}))
    actor = omega_conf_to_dataclass(raw.actor_rollout_ref.actor)
    assert model.algorithm == 'omni_nft' and actor.diffusion_loss.loss_mode == 'omni_nft'
    assert model.pipeline.video_cfg_scale == model.pipeline.audio_cfg_scale == 1.0
    assert model.lora_rank == 32 and model.lora_alpha == 64
    assert optim.lr == 3e-5 and optim.weight_decay == 1e-4 and actor.diffusion_loss.mix_beta == 1.0
    inspect.signature(OmniNFTDiffusersFSDPEngine).bind(model, engine, optim, checkpoint)
    return model, engine, optim, checkpoint, actor


def inputs():
    original, sampler, hashes = shared.fixed_inputs()
    split = int(original['video_seq_len'][0])
    sigma = original['train_timesteps'][:, -1, None, None] / 1000
    estimate = original['train_latents'][:, -1].float() - sigma * sampler[:, -1].float()
    generator = torch.Generator().manual_seed(91743)
    values = {k: original[k] for k in original.keys() if 'prompt_embeds' in k}
    values.update(video_latents_clean=estimate[:, :split], audio_latents_clean=estimate[:, split:],
                  train_timesteps=original['train_timesteps'],
                  video_forward_noise=torch.randn((2, 3, split, estimate.shape[-1]), generator=generator),
                  audio_forward_noise=torch.randn((2, 3, estimate.shape[1] - split, estimate.shape[-1]), generator=generator),
                  reward_prob=torch.tensor([[[.7, .3]] * 3, [[.3, .7]] * 3]))
    return tu.get_tensordict(values), hashes


def cpu_contract(config, batch, hashes):
    model, engine_config, optim, checkpoint, actor = config
    # Exercise the real prepare path with only CPU batch tensors; the chosen
    # architecture's prepare_model_inputs does not access module weights.
    engine = object.__new__(OmniNFTDiffusersFSDPEngine)
    engine.model_config = model
    engine.module = SimpleNamespace()
    engine.scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))
    engine.use_ulysses_sp = False
    engine.ulysses_sequence_parallel_size = 1
    for microbatch in (1, 2):
        data = batch.clone(recurse=True)
        tu.assign_non_tensor(data, use_dynamic_bsz=False, micro_batch_size_per_gpu=microbatch)
        pieces, _ = prepare_micro_batches(data)
        assert len(pieces) == 2 // microbatch
        for piece in pieces:
            for step in range(3):
                positive, negative, x0, xt, times = engine.prepare_model_inputs(piece, step)
                assert negative is None and positive['hidden_states'].shape[0] == microbatch
                zeros = tuple(torch.zeros_like(value) for value in x0)
                output = engine.prepare_model_outputs((zeros, zeros, zeros, x0, xt, times), piece)
                loss_data = tu.get_tensordict({'reward_prob': piece['reward_prob'][:, step]})
                tu.assign_non_tensor(loss_data, gradient_accumulation_steps=len(pieces) * 3, sp_size=1)
                loss, _ = diffusion_loss(actor, output, loss_data)
                assert torch.isfinite(loss)
    report = {'stage': 'cpu_contract_passed', 'capture_hashes': hashes,
              'selected_source': os.environ['SOURCE_SNAPSHOT_SHA256'],
              'scope': 'native OmniNFT prepare/forward-noise/output/loss/microbatch contracts, no GPU model',
              'clean_estimates': 'recorded x_t minus sigma times actual sampler velocity at denoise index18; not final endpoints',
              'reward_probabilities': 'fixed synthetic [.7,.3] and [.3,.7], no measured reward substitution',
              'forward_noise': 'precomputed independent CPU generator91743, same values across partitions',
              'diagnostic_dp': 1, 'production_dp': 16, 'conservative_max_probe_output_bytes': 64 * 2**30}
    shared.publish('nft_gpu_state.json', report)
    print(json.dumps(report), flush=True)
    return report


def branch(engine, actor, batch, microbatch):
    observations = []

    def loss_function(*, model_output, data, dp_group):
        loss, metrics = diffusion_loss(actor, model_output, data, dp_group=dp_group)
        assert torch.isfinite(loss)
        row = {'scaled_loss': float(loss.detach())}
        for modality in ('video', 'audio'):
            p = model_output[modality + '_forward_prediction'].detach()
            old = model_output[modality + '_old_prediction']
            ref = model_output[modality + '_ref_forward_prediction']
            row[modality + '_old_mse'] = float((p - old).square().mean())
            row[modality + '_reference_mse'] = float((p - ref).square().mean())
        row['metrics'] = {k: getattr(v, 'value', v) for k, v in metrics.items()}
        observations.append(row)
        print('NFT_DIAGNOSTIC', microbatch, 'forward_backward', len(observations), flush=True)
        return loss, metrics

    data = batch.clone(recurse=True)
    tu.assign_non_tensor(data, micro_batch_size_per_gpu=microbatch, use_dynamic_bsz=False,
                         return_model_output=False, use_no_sync_for_gradient_accumulation=False)
    engine.optimizer_zero_grad()
    torch.cuda.synchronize()
    began = time.monotonic()
    with engine.train_mode():
        engine.forward_backward_batch(data, loss_function, forward_only=False)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - began
        gradients = {n: shared.local_tensor(p.grad).detach().cpu().clone()
                     for n, p in engine.module.named_parameters() if p.grad is not None}
        assert gradients and all('lora_' in n and '.default.' in n for n in gradients)
        assert all(torch.isfinite(g).all() for g in gradients.values())
        began = time.monotonic()
        norm = engine.optimizer_step()
        engine.lr_scheduler_step()
        torch.cuda.synchronize()
        optimizer_seconds = time.monotonic() - began
    result = {'stage': 'finite_diagnostic_update', 'microbatch': microbatch,
              'backward_calls': len(observations), 'accumulation': 2 // microbatch * 3,
              'scaled_loss_sum': sum(r['scaled_loss'] for r in observations),
              'gradient_norm_before_clip': norm, 'forward_backward_seconds': elapsed,
              'optimizer_seconds': optimizer_seconds, 'observations': observations}
    shared.publish(f'nft_mb{microbatch}.json', result)
    return result, gradients, shared.adapter_copy(engine)


def gpu_main(config, batch, hashes):
    shared.ensure_probe_space()
    threading.Thread(target=shared.monitor_probe_space, daemon=True).start()
    torch.cuda.set_device(0)
    torch.distributed.init_process_group('nccl', timeout=datetime.timedelta(seconds=600))
    assert torch.distributed.get_world_size() == 1
    state = {'stage': 'actual_nft_actor_initializing', 'source_sha256': os.environ['SOURCE_SNAPSHOT_SHA256'],
             'capture_hashes': hashes, 'scientific_optimizer_updates': 0, 'diagnostic_dp': 1,
             'production_dp': 16, 'started_unix': time.time(),
             'inputs': 'native one-step clean estimates plus synthetic probabilities and fixed forward noise; not final generated endpoints or science scores'}
    shared.publish('nft_gpu_state.json', state)
    engine = OmniNFTDiffusersFSDPEngine(*config[:4])
    engine.initialize()
    engine.copy_adapter('default', 'old')
    initial = shared.adapter_copy(engine)
    rng = engine.checkpoint_manager.get_rng_state()
    scheduler = copy.deepcopy(engine.lr_scheduler.state_dict())
    results = {}
    for size in (2, 1):
        shared.reset_initial(engine, initial, scheduler, rng)
        state['stage'] = f'nft_mb{size}'
        shared.publish('nft_gpu_state.json', state)
        results[size] = branch(engine, config[4], batch, size)
    r2, g2, p2 = results[2]
    r1, g1, p1 = results[1]
    state['comparison'] = {'mb2_loss': r2['scaled_loss_sum'], 'mb1_loss': r1['scaled_loss_sum'],
                           'gradient': shared.difference(g2, g1),
                           'parameter_update': shared.difference({n: p2[n] - initial[n] for n in p2},
                                                                 {n: p1[n] - initial[n] for n in p1})}
    state.update(stage='diagnostics_completed', finished_unix=time.time(), diagnostic_optimizer_calls=2,
                 peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),
                 microbatch_equivalence='report numerical differences; do not infer a pass from finite updates')
    shared.publish('nft_gpu_state.json', state)
    torch.distributed.destroy_process_group()
    print(json.dumps(state), flush=True)


if __name__ == '__main__':
    config = configuration()
    batch, hashes = inputs()
    if args.cpu_config_only:
        cpu_contract(config, batch, hashes)
    else:
        gpu_main(config, batch, hashes)
