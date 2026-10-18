"""Finite native ART repeatability/gradient-sign diagnostic; no checkpoint writes."""
import copy
import datetime
import hashlib
import json
import os
import random
import signal
import threading
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from verl.utils.config import omega_conf_to_dataclass
from verl_omni.workers.engine.fsdp.diffusers_impl import ARTDiffusersFSDPEngine
import verify_actor_gpu as shared
from gradient_partition_details import partition_details

ROOT, OUTPUT, args = shared.ROOT, shared.OUTPUT, shared.args


def check_precision_space():
    stat = os.statvfs(ROOT)
    free = stat.f_bavail * stat.f_frsize
    if free < 250 * 2**30:
        raise RuntimeError(f'Precision-only diagnostic free space below250GiB: {free}')
    return free


def monitor_precision_space():
    while True:
        time.sleep(10)
        try:
            check_precision_space()
        except BaseException as error:
            try:
                shared.publish('storage_guard_stop.json', {'reason': str(error), 'unix': time.time(),
                                                           'signal_target': 'self only'})
            finally:
                os.kill(os.getpid(), signal.SIGTERM)
            return


def fixture():
    a = {'lora_B': torch.tensor([1., -.001, 0.])}
    b = {'lora_B': torch.tensor([1., .001, .001])}
    u = {'lora_B': torch.tensor([-1., 1., 0.])}
    v = {'lora_B': torch.tensor([-1., -1., -1.])}
    result = partition_details(a, b, u, v, lr=1., eps=1e-8)
    assert result['sign_flip_count'] == result['zero_transition_count'] == 1
    assert abs(result['update_difference_energy_fraction_at_sign_flips'] - .8) < 1e-6
    assert abs(result['update_difference_energy_fraction_at_zero_transitions'] - .2) < 1e-6
    assert result['gradient_energy_fraction_at_sign_flips'] < .00001
    return result


def main():
    config = shared.configs()
    batch, sampler, hashes = shared.fixed_inputs()
    if args.cpu_config_only:
        proof = shared.cpu_contract(*config, batch, hashes)
        proof.update(scope='native ART input/loss/microbatch/meta-LoRA contracts plus independent sign/zero-transition energy fixture',
                     sign_fixture=fixture(), conservative_max_probe_output_bytes=2**30,
                     gpu_checkpoint_writes=False, minimum_start_free_gib=275,
                     dynamic_stop_free_gib=250, current_available_bytes=check_precision_space())
        shared.publish('precision_gpu_state.json', proof)
        print('CPU_PRECISION_CONTRACT_PASSED', flush=True)
        return
    check_precision_space()
    threading.Thread(target=monitor_precision_space, daemon=True).start()
    torch.cuda.set_device(0)
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.distributed.init_process_group('nccl', timeout=datetime.timedelta(seconds=600))
    assert torch.distributed.get_world_size() == 1
    state = {'stage': 'actual_art_actor_initializing', 'source_sha256': os.environ['SOURCE_SNAPSHOT_SHA256'],
             'capture_hashes': hashes, 'diagnostic_dp': 1, 'production_dp': 16,
             'scientific_optimizer_updates': 0, 'started_unix': time.time(), 'explicit_init_seed': 42,
             'scope': 'fixed native states/conditions; synthetic per-modality unit-RMS guidance; not science scores',
             'gpu_checkpoint_writes': False}
    shared.publish('precision_gpu_state.json', state)
    engine = ARTDiffusersFSDPEngine(*config)
    engine.initialize()
    engine.copy_adapter('default', 'old')
    initial = shared.adapter_copy(engine)
    digest = hashlib.sha256()
    for name, value in sorted(initial.items()):
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode()); digest.update(value.view(torch.uint8).numpy().tobytes())
    state['initial_all_named_adapter_sha256'] = digest.hexdigest()
    rng = engine.checkpoint_manager.get_rng_state()
    scheduler = copy.deepcopy(engine.lr_scheduler.state_dict())
    actor = omega_conf_to_dataclass(OmegaConf.load(ROOT / 'preflight/configs/A0/resolved.yaml').actor_rollout_ref.actor)
    results = {}
    for label, size in [('precision_mb2', 2), ('precision_mb2_repeat', 2), ('precision_mb1', 1)]:
        state['stage'] = label
        shared.publish('precision_gpu_state.json', state)
        shared.reset_initial(engine, initial, scheduler, rng)
        results[label] = shared.run_branch(engine, batch, sampler, actor, size, False, label)
    r2, g2, p2 = results['precision_mb2']
    rr, gr, pr = results['precision_mb2_repeat']
    r1, g1, p1 = results['precision_mb1']
    updates = lambda parameters: {n: parameters[n].float() - initial[n].float() for n in g2}
    u2, ur, u1 = updates(p2), updates(pr), updates(p1)
    state['same_microbatch_repeat'] = {
        'first_loss': r2['summed_scaled_loss'], 'repeat_loss': rr['summed_scaled_loss'],
        'gradient': shared.difference(g2, gr), 'gradient_bitwise_equal': shared.tree_equal(g2, gr),
        'parameter_update': shared.difference(u2, ur)}
    state['microbatch_partition'] = {
        'loss_mb2': r2['summed_scaled_loss'], 'loss_mb1': r1['summed_scaled_loss'],
        'gradient': shared.difference(g2, g1), 'parameter_update': shared.difference(u2, u1),
        'sign_and_energy': partition_details(g2, g1, u2, u1, lr=config[2].lr,
                                            eps=engine.optimizer.param_groups[0]['eps'])}
    state['parameter_dtypes'] = sorted({str(t.dtype) for t in initial.values()})
    state['gradient_dtypes'] = sorted({str(t.dtype) for t in g2.values()})
    state['Adam_moment_dtypes'] = sorted({str(t.dtype) for values in engine.optimizer.state.values()
                                        for k, t in values.items() if k in ('exp_avg', 'exp_avg_sq')})
    state.update(stage='diagnostics_completed', finished_unix=time.time(), diagnostic_optimizer_calls=3,
                 peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),
                 interpretation='measured repeatability and sign/energy decomposition; no automatic microbatch-equivalence pass',
                 production_microbatch=2, production_math_unchanged=True)
    shared.publish('precision_gpu_state.json', state)
    torch.distributed.destroy_process_group()
    print(json.dumps(state), flush=True)


if __name__ == '__main__':
    main()
