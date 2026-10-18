"""Two native ART worker updates, including both metric and MFU collectives.

Uses the existing two recorded inputs and three captured timesteps. Synthetic
fixed guidance is diagnostic only; there are zero scientific budget updates.
Default behavior adds no synchronization; an explicit diagnostic option tests
one device synchronization between the two native worker calls.
"""
import argparse
from datetime import timedelta
from dataclasses import asdict, replace
from contextlib import nullcontext, contextmanager
from functools import partial
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from tensordict import NonTensorData
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.distributed import initialize_global_process_group_ray
from verl_omni.utils.mfu import DiffusionFlopsCounter
from verl_omni.workers.engine_workers import TrainingWorker
from verl_omni.workers.utils.losses import diffusion_loss

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(output, record):
    path = output / f'fsdp_cycle_rank_{record["rank"]}.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def synchronize_worker_boundary(record, synchronize, publish_record):
    record['stage'] = 'boundary_device_synchronization_running'
    record['boundary_synchronization'] = {'stage': 'running', 'started_unix': time.time()}
    publish_record(record)
    start = time.monotonic()
    synchronize()
    elapsed = time.monotonic() - start
    record['boundary_synchronization'].update(stage='returned', seconds=elapsed, finished_unix=time.time())
    record['stage'] = 'boundary_device_synchronization_returned'
    publish_record(record)


@contextmanager
def replace_scalar_avg_with_sum(record, enabled):
    if not enabled:
        yield
        return
    original = dist.all_reduce
    def all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False, **kwargs):
        if op != dist.ReduceOp.AVG:
            return original(tensor, op=op, group=group, async_op=async_op, **kwargs)
        assert not async_op and tensor.numel() == 1 and tensor.dtype == torch.float32
        result = original(tensor, op=dist.ReduceOp.SUM, group=group, async_op=False, **kwargs)
        tensor.div_(dist.get_world_size(group))
        record['loss_avg_replacements'] += 1
        record.setdefault('loss_reduce_tensor_devices', []).append(tensor.device.type)
        return result
    with patch.object(dist, 'all_reduce', all_reduce):
        yield


def install_logging_reduction(worker, record, via_sum, on_cpu):
    assert not on_cpu or via_sum
    if not via_sum:
        return
    native_postprocess = worker._postprocess_output
    def logging_postprocess(*values, **options):
        original_device = worker.device_name
        try:
            if on_cpu:
                worker.device_name = 'cpu'
            with replace_scalar_avg_with_sum(record, True):
                result = native_postprocess(*values, **options)
            if on_cpu:
                record['cpu_loss_postprocess_calls'] += 1
            return result
        finally:
            worker.device_name = original_device
    worker._postprocess_output = logging_postprocess


def install_local_postprocess(worker, record):
    """Diagnostic only: native local metrics without auxiliary collectives/MFU.

    The override exists only after the engine has returned; it cannot change
    its forward/backward/optimizer or any FSDP process group during training.
    """
    native_postprocess = worker._postprocess_output
    def local_postprocess(*values, **options):
        original_device = worker.device_name
        original_group = worker.engine.get_data_parallel_group()
        try:
            worker.device_name = 'cpu'
            with patch.object(worker.engine, 'get_data_parallel_group', return_value=None), \
                    patch.object(worker, 'flops_counter', None):
                result = native_postprocess(*values, **options)
            record['local_logging_postprocess_calls'] += 1
            return result
        finally:
            worker.device_name = original_device
            assert worker.engine.get_data_parallel_group() is original_group
    worker._postprocess_output = local_postprocess


def configure_adamw_execution(optimizer, disable_foreach):
    """Select AdamW execution backend only; parameters/state/hypers stay put."""
    assert type(optimizer) is torch.optim.AdamW
    assert not optimizer.state, 'backend selection must precede the first update'
    hypers = ('lr', 'betas', 'eps', 'weight_decay', 'amsgrad', 'maximize',
              'capturable', 'differentiable', 'fused', 'decoupled_weight_decay')
    before = [{key: group.get(key) for key in hypers} for group in optimizer.param_groups]
    parameters = [tuple(id(p) for p in group['params']) for group in optimizer.param_groups]
    if disable_foreach:
        assert optimizer.defaults['foreach'] is None
        assert all(group['foreach'] is None for group in optimizer.param_groups)
        optimizer.defaults['foreach'] = False
        for group in optimizer.param_groups:
            group['foreach'] = False
    assert [{key: group.get(key) for key in hypers} for group in optimizer.param_groups] == before
    assert [tuple(id(p) for p in group['params']) for group in optimizer.param_groups] == parameters
    assert not optimizer.state
    return {'optimizer_class': 'torch.optim.AdamW', 'foreach_disabled': disable_foreach,
            'foreach_defaults': optimizer.defaults['foreach'],
            'foreach_groups': [group['foreach'] for group in optimizer.param_groups],
            'unchanged_hyperparameters': before,
            'parameter_count': sum(len(group['params']) for group in optimizer.param_groups),
            'parameter_dtypes': sorted({str(p.dtype) for group in optimizer.param_groups for p in group['params']})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures', type=Path, required=True)
    parser.add_argument('--cpu-contract-only', action='store_true')
    args = parser.parse_args()
    output = Path(os.environ['OUTPUT_DIR']).resolve()
    assert output.is_relative_to(ROOT / 'preflight/runtime_gpu')
    output.mkdir(parents=True, exist_ok=True)
    assert args.captures.resolve().is_relative_to(ROOT / 'preflight/runtime_gpu')
    original_argv = sys.argv
    sys.argv = [sys.argv[0], '--captures', str(args.captures)]
    import verify_actor_gpu as shared
    sys.argv = original_argv
    configs = shared.configs()
    batch, _, captures = shared.fixed_inputs()
    assert len(batch) == 2 and batch['train_timesteps'].shape[1] == 3
    raw = OmegaConf.load(ROOT / 'preflight/configs/A0/resolved.yaml')
    assert raw.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu == 2
    assert raw.trainer.nnodes * raw.trainer.n_gpus_per_node == 16
    config_path = ROOT / 'preflight/configs/A0/resolved.yaml'
    native_train_batch = inspect.unwrap(TrainingWorker.train_batch)
    boundary_sync = os.environ.get('OMNI_CYCLE_BOUNDARY_DEVICE_SYNC', '0')
    assert boundary_sync in ('0', '1')
    loss_sum = os.environ.get('OMNI_CYCLE_LOSS_AVG_AS_SUM', '0')
    assert loss_sum in ('0', '1')
    loss_cpu = os.environ.get('OMNI_CYCLE_LOSS_REDUCTION_CPU', '0')
    assert loss_cpu in ('0', '1') and not (loss_cpu == '1' and loss_sum != '1')
    local_postprocess = os.environ.get('OMNI_CYCLE_LOCAL_POSTPROCESS', '0')
    assert local_postprocess in ('0', '1')
    foreach_off = os.environ.get('OMNI_CYCLE_ADAMW_FOREACH_OFF', '0')
    assert foreach_off in ('0', '1')
    replicated = os.environ.get('OMNI_CYCLE_REPLICATED_FSDP2', '0')
    assert replicated in ('0', '1')
    assert not (replicated == '1' and foreach_off != '1')
    requested_network = os.environ.get('NCCL_NET')
    assert requested_network in (None, 'Socket')
    socket_network = requested_network == 'Socket'
    assert not socket_network or foreach_off == '1'
    optimizer_replay = os.environ.get('OMNI_CYCLE_OPTIMIZER_REPLAY', '0')
    assert optimizer_replay in ('0','1')
    assert optimizer_replay != '1' or (socket_network and foreach_off == '1')
    import optimizer_backend_equivalence as optimizer_check
    assert sum(value == '1' for value in (boundary_sync, loss_sum, local_postprocess, foreach_off)) <= 1
    original_engine_config = asdict(configs[1])
    if replicated == '1':
        configs = (configs[0], replace(configs[1], fsdp_size=1), configs[2], configs[3])
    actual_engine_config = asdict(configs[1])
    config_changes = {key: actual_engine_config[key] for key in actual_engine_config
                      if actual_engine_config[key] != original_engine_config[key]}
    assert config_changes == ({'fsdp_size': 1} if replicated == '1' else {})
    identity = {'source_sha256': os.environ['SOURCE_SNAPSHOT_SHA256'],
                'script_sha256': digest(__file__), 'shared_helper_sha256': digest(shared.__file__),
                'config_sha256': digest(config_path), 'capture_hashes': captures,
                'torch_version': torch.__version__, 'world_size': 16,
                'requested_process_group_backend': 'cpu:gloo,cuda:nccl',
                'boundary_device_sync_between_workers': boundary_sync == '1',
                'loss_avg_as_sum': loss_sum == '1',
                'loss_reduction_on_cpu': loss_cpu == '1',
                'local_only_postprocess': local_postprocess == '1',
                'adamw_foreach_disabled': foreach_off == '1',
                'replicated_fsdp2': replicated == '1',
                'requested_nccl_network': 'Socket' if socket_network else 'auto',
                'gpu_optimizer_backend_replay': optimizer_replay == '1',
                'optimizer_replay_helper_sha256': digest(optimizer_check.__file__),
                'private_engine_config_overrides': config_changes,
                'expected_actor_dp_size': 16,
                'expected_mesh_shape': [16, 1] if replicated == '1' else [16],
                'diagnostic_local_samples': 2, 'diagnostic_timesteps': 3,
                'scientific_optimizer_updates': 0, 'gpu_checkpoint_writes': False,
                'worker_call': 'native train_batch implementation body; excludes Ray/transferqueue/profiler decorators'}
    if args.cpu_contract_only:
        assert not torch.cuda.is_initialized()
        worker = SimpleNamespace(device_name='cpu', flops_counter=None,
                                 engine=SimpleNamespace(get_data_parallel_group=lambda: None))
        result = TrainingWorker._postprocess_output(
            worker, {'loss': [[.2, .3]], 'metrics': {'grad_norm': .01, 'aux': [1., 2.]},
                     'model_output': {}}, global_token_num=None, delta_time=1.,
            forward_only=False, images_seqlens=None)
        metrics = tu.get(result, 'metrics')
        assert math.isclose(metrics['loss'], .5) and metrics['grad_norm'] == .01
        assert metrics['aux'] == [1., 2.]
        # Exercise the actual implementation body with CPU results from a fake
        # engine. This checks constructor attributes and native postprocessing,
        # without pretending to exercise CUDA, FSDP or the distributed wrappers.
        holder = object.__new__(TrainingWorker)
        holder.device_name, holder.flops_counter = 'cpu', None
        holder.model_config, holder.engine_config = configs[0], configs[1]
        holder.engine_config.use_remove_padding = False
        holder.engine_config.use_fused_kernels = False
        holder.loss_fn = object()
        holder.engine = SimpleNamespace(
            get_data_parallel_group=lambda: None, train_mode=lambda **kwargs: nullcontext(),
            is_mp_src_rank_with_outputs=lambda: True, lr_scheduler_step=lambda: 3e-5,
            train_batch=lambda data, loss_function: {'loss': [[.2, .3]],
                'metrics': {'grad_norm': .01, 'aux': [1., 2.]}, 'model_output': {}})
        cpu_data = batch.clone(recurse=True)
        tu.assign_non_tensor(cpu_data, global_token_num=NonTensorData(None),
                             micro_batch_size_per_gpu=2, disable_auto_offload=True,
                             update_lr_scheduler=False)
        cpu_reduction_record = {'loss_avg_replacements': 0, 'cpu_loss_postprocess_calls': 0,
                                'local_logging_postprocess_calls': 0}
        if loss_cpu == '1' or local_postprocess == '1':
            holder.device_name = 'cuda'
        saved_device = holder.device_name
        install_logging_reduction(holder, cpu_reduction_record, loss_sum == '1', loss_cpu == '1')
        if local_postprocess == '1':
            install_local_postprocess(holder, cpu_reduction_record)
        integrated = native_train_batch(holder, cpu_data)
        assert holder.device_name == saved_device
        assert cpu_reduction_record['cpu_loss_postprocess_calls'] == int(loss_cpu == '1')
        assert cpu_reduction_record['local_logging_postprocess_calls'] == int(local_postprocess == '1')
        assert math.isclose(tu.get(integrated, 'metrics')['loss'], .5)
        assert tu.get(integrated, 'metrics')['grad_norm'] == .01
        events, sync_calls = [], []
        boundary_record = {'diagnostic_optimizer_steps_returned': 1}
        synchronize_worker_boundary(boundary_record, lambda: sync_calls.append(True),
                                    lambda row: events.append(json.loads(json.dumps(row))))
        assert sync_calls == [True] and len(events) == 2
        assert events[0]['stage'] == 'boundary_device_synchronization_running'
        assert events[1]['stage'] == 'boundary_device_synchronization_returned'
        assert events[1]['boundary_synchronization']['seconds'] >= 0
        assert boundary_record['diagnostic_optimizer_steps_returned'] == 1
        reduction_calls = []
        def fake_sum(tensor, op, **kwargs):
            assert op == dist.ReduceOp.SUM
            reduction_calls.append(op)
            tensor.mul_(16)
        avg_record = {'loss_avg_replacements': 0}
        scalar = torch.tensor(.5)
        with patch.object(dist, 'all_reduce', fake_sum), patch.object(dist, 'get_world_size', lambda group=None: 16):
            with replace_scalar_avg_with_sum(avg_record, True):
                dist.all_reduce(scalar, op=dist.ReduceOp.AVG)
        assert scalar.item() == .5 and avg_record['loss_avg_replacements'] == 1
        assert reduction_calls == [dist.ReduceOp.SUM]
        # Verify the local-only wrapper with a non-null group and an MFU
        # counter. Any distributed call or counter access is a hard failure.
        group_sentinel = object()
        local_holder = object.__new__(TrainingWorker)
        local_holder.device_name = 'cuda'
        local_holder.engine = SimpleNamespace(get_data_parallel_group=lambda: group_sentinel)
        counter_sentinel = object()
        local_holder.flops_counter = counter_sentinel
        local_record = {'local_logging_postprocess_calls': 0}
        install_local_postprocess(local_holder, local_record)
        def forbidden_collective(*args, **kwargs):
            raise AssertionError('local-only postprocess called a collective')
        with patch.object(dist, 'all_reduce', forbidden_collective), \
                patch.object(dist, 'all_gather_object', forbidden_collective):
            local_output = local_holder._postprocess_output(
                {'loss': [[.2, .3]], 'metrics': {'grad_norm': .01, 'aux': [1., 2.]}, 'model_output': {}},
                global_token_num=None, delta_time=1., forward_only=False, images_seqlens=None,
                diffusion_flops_meta={'latent_seqlens': [1], 'num_timesteps': 3})
        assert local_record['local_logging_postprocess_calls'] == 1
        assert math.isclose(tu.get(local_output, 'metrics')['loss'], .5)
        assert tu.get(local_output, 'metrics')['aux'] == [1., 2.]
        assert local_holder.device_name == 'cuda' and local_holder.flops_counter is counter_sentinel
        assert local_holder.engine.get_data_parallel_group() is group_sentinel
        # A malformed metric output must also restore device/group/counter.
        try:
            local_holder._postprocess_output({}, global_token_num=None, delta_time=1.,
                forward_only=False, images_seqlens=None)
            raise AssertionError('malformed output unexpectedly accepted')
        except KeyError:
            pass
        assert local_holder.device_name == 'cuda' and local_holder.flops_counter is counter_sentinel
        assert local_holder.engine.get_data_parallel_group() is group_sentinel
        assert local_record['local_logging_postprocess_calls'] == 1
        from verl.workers.config.optimizer import build_optimizer
        optimizer_parameter = torch.nn.Parameter(torch.zeros(4))
        cpu_optimizer = build_optimizer([optimizer_parameter], configs[2])
        cpu_backend = configure_adamw_execution(cpu_optimizer, foreach_off == '1')
        assert cpu_backend['foreach_groups'] == ([False] if foreach_off == '1' else [None])
        assert cpu_backend['unchanged_hyperparameters'][0]['lr'] == 3e-5
        assert cpu_backend['unchanged_hyperparameters'][0]['weight_decay'] == 1e-4
        optimizer_parameter.grad = torch.ones_like(optimizer_parameter) * .01
        cpu_optimizer.step()
        assert torch.isfinite(optimizer_parameter).all()
        assert cpu_optimizer.state[optimizer_parameter]['step'].item() == 1
        # Exercise the exact native mesh factory with a mocked constructor;
        # this verifies configuration routing, never distributed GPU math.
        from verl.workers.engine.fsdp import utils as mesh_utils
        expected_shape = (16, 1) if replicated == '1' else (16,)
        expected_names = ('ddp', 'fsdp') if replicated == '1' else ('fsdp',)
        mesh_calls = []
        def cpu_mesh_constructor(device, mesh_shape, mesh_dim_names):
            mesh_calls.append((tuple(mesh_shape), tuple(mesh_dim_names)))
            return SimpleNamespace(shape=tuple(mesh_shape))
        with patch.object(mesh_utils, 'init_device_mesh', cpu_mesh_constructor):
            mesh = mesh_utils.create_device_mesh(16, configs[1].fsdp_size)
        assert mesh.shape == expected_shape and mesh_calls == [(expected_shape, expected_names)]
        with patch.object(dist, 'get_world_size', lambda: 16):
            assert shared.ARTDiffusersFSDPEngine.get_data_parallel_size(
                SimpleNamespace(ulysses_sequence_parallel_size=1)) == 16
        from verl_omni.pipelines.model_base import DiffusionModelBase
        actual_model_class = DiffusionModelBase.get_class(configs[0])
        assert actual_model_class.configure_train_mode.__func__ is DiffusionModelBase.configure_train_mode.__func__
        optimizer_replay_contract = optimizer_check.cpu_contract()
        assert optimizer_replay_contract['gpu_verified'] is False
        assert all(x['all_fields_bitwise_equal'] for x in optimizer_replay_contract['reports'])
        proof = {**identity, 'status': 'passed_cpu_fsdp_cycle_contract',
                 'cuda_initialized': torch.cuda.is_initialized(), 'gpu_verified': False,
                 'cpu_native_metric_postprocess_verified': True,
                 'cpu_native_worker_body_executed': True,
                 'cpu_boundary_event_contract_verified': True,
                 'cpu_loss_avg_sum_contract_verified': True,
                 'cpu_native_postprocess_device_restore_verified': True,
                 'cpu_local_postprocess_no_collectives_restore_verified': True,
                 'cpu_adamw_backend_switch_verified': True,
                 'cpu_optimizer_backend_replay_contract_verified': True,
                 'cpu_native_mesh_factory_and_actor_dp_verified': True,
                 'cpu_ltx_train_mode_hook_is_inherited_noop': True,
                 'cpu_adamw_execution': cpu_backend,
                 'scope': 'CPU input and actual worker metric contracts; no GPU engine or resource admission.'}
        (output / 'fsdp_cycle_cpu_contract.json').write_text(json.dumps(proof, indent=2) + '\n')
        print(json.dumps(proof), flush=True)
        return
    proof = json.loads(Path(os.environ['OMNI_CYCLE_CPU_PROOF']).read_text())
    assert proof['status'] == 'passed_cpu_fsdp_cycle_contract' and proof['gpu_verified'] is False
    assert proof['cuda_initialized'] is False
    for key, value in identity.items():
        assert proof[key] == value, key
    rank, local_rank = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    launcher_rank = int(os.environ['OMNI_CYCLE_LAUNCHER_LOCAL_RANK'])
    assert int(os.environ['WORLD_SIZE']) == 16 and local_rank == 0
    assert os.environ['CUDA_VISIBLE_DEVICES'] == str(launcher_rank) and launcher_rank == rank % 8
    torch.cuda.set_device(0)
    initialize_global_process_group_ray(timeout_second=180)
    assert dist.get_backend() == 'cpu:gloo,cuda:nccl'
    assert dist.distributed_c10d._get_object_coll_device() == 'cpu'
    torch.manual_seed(13579)
    record = {**identity, 'pid': os.getpid(), 'rank': rank, 'local_rank': 0, 'launcher_local_rank': launcher_rank,
              'physical_visible_device': os.environ['CUDA_VISIBLE_DEVICES'],
              'attempt': os.environ['OMNI_ATTEMPT'], 'nonce': os.environ['OMNI_NONCE'],
              'started_unix': time.time(), 'stage': 'initializing_native_art_engine',
              'actual_process_group_backend': dist.get_backend(),
              'object_collective_device': dist.distributed_c10d._get_object_coll_device(),
              'diagnostic_optimizer_steps_returned': 0, 'cycles': [],
              'boundary_synchronization': None, 'loss_avg_replacements': 0,
              'loss_reduce_tensor_devices': [], 'cpu_loss_postprocess_calls': 0,
              'local_logging_postprocess_calls': 0,
              'environment': {key: os.environ.get(key) for key in (
                  'NCCL_P2P_DISABLE', 'NCCL_NVLS_ENABLE', 'NCCL_LAUNCH_ORDER_IMPLICIT',
                  'OMNI_CYCLE_BOUNDARY_DEVICE_SYNC', 'OMNI_CYCLE_LOSS_AVG_AS_SUM',
                  'OMNI_CYCLE_LOSS_REDUCTION_CPU', 'OMNI_CYCLE_LOCAL_POSTPROCESS',
                  'OMNI_CYCLE_ADAMW_FOREACH_OFF', 'OMNI_CYCLE_REPLICATED_FSDP2',
                  'OMNI_CYCLE_OPTIMIZER_REPLAY')
                  + (('NCCL_NET',) if socket_network else ())}}
    publish(output, record)
    engine = shared.ARTDiffusersFSDPEngine(*configs)
    engine.initialize()
    record['actual_adamw_execution'] = configure_adamw_execution(engine.optimizer, foreach_off == '1')
    engine.copy_adapter('default', 'old')
    mesh = engine.device_mesh
    shard_group = mesh.get_group('fsdp')
    replicate_group = mesh.get_group('ddp') if replicated == '1' else None
    record['actual_mesh_shape'] = list(mesh.shape)
    record['actual_mesh_dim_names'] = list(mesh.mesh_dim_names)
    record['actor_dp_size'] = engine.get_data_parallel_size()
    record['fsdp_shard_world_size'] = dist.get_world_size(shard_group)
    record['fsdp_replicate_world_size'] = dist.get_world_size(replicate_group) if replicate_group is not None else 1
    record['metric_and_fsdp_group_are_same_object'] = engine.get_data_parallel_group() is shard_group
    record['metric_and_replicate_group_are_same_object'] = engine.get_data_parallel_group() is replicate_group
    record['fsdp_group_backend'] = dist.get_backend(shard_group)
    record['replicate_group_backend'] = dist.get_backend(replicate_group) if replicate_group is not None else None
    assert record['actual_mesh_shape'] == identity['expected_mesh_shape']
    assert record['actor_dp_size'] == 16
    assert record['fsdp_shard_world_size'] == (1 if replicated == '1' else 16)
    assert record['fsdp_replicate_world_size'] == (16 if replicated == '1' else 1)
    record['gpu_memory_after_engine_init_bytes'] = torch.cuda.memory_allocated()
    record['peak_gpu_memory_bytes'] = torch.cuda.max_memory_allocated()
    worker = object.__new__(TrainingWorker)
    worker.engine, worker.model_config, worker.engine_config = engine, configs[0], configs[1]
    worker.device_name = 'cuda'
    worker.engine_config.use_remove_padding = worker.model_config.get('use_remove_padding', False)
    worker.engine_config.use_fused_kernels = worker.model_config.get('use_fused_kernels', False)
    worker.flops_counter = DiffusionFlopsCounter(
        architecture=worker.model_config.architecture,
        transformer_config=getattr(worker.model_config, 'transformer_config', None))
    actor = omega_conf_to_dataclass(raw.actor_rollout_ref.actor)
    worker.loss_fn = partial(diffusion_loss, actor)
    install_logging_reduction(worker, record, loss_sum == '1', loss_cpu == '1')
    if local_postprocess == '1':
        install_local_postprocess(worker, record)
    if optimizer_replay == '1':
        original_optimizer_step = engine.optimizer.step
        optimizer_step_calls = 0
        def step_with_replay(*step_args, **step_options):
            nonlocal optimizer_step_calls
            optimizer_step_calls += 1
            # Only capture the second real native step, after clipping. The
            # known second-call communication failure occurs before this point.
            inputs = optimizer_check.capture(engine.optimizer) if optimizer_step_calls == 2 else None
            result = original_optimizer_step(*step_args, **step_options)
            if inputs is not None:
                record['optimizer_backend_replay'] = optimizer_check.compare_after(engine.optimizer, inputs)
                record['optimizer_backend_replay']['native_optimizer_call'] = optimizer_step_calls
            return result
        step_with_replay._wrapped_by_lr_sched = getattr(original_optimizer_step, '_wrapped_by_lr_sched', False)
        engine.optimizer.step = step_with_replay
    datasets = [batch.clone(recurse=True) for _ in range(2)]
    for cycle, data in enumerate(datasets):
        tu.assign_non_tensor(data, global_token_num=NonTensorData(None), disable_auto_offload=True,
                             micro_batch_size_per_gpu=2, use_dynamic_bsz=False,
                             use_no_sync_for_gradient_accumulation=False, return_model_output=False,
                             update_lr_scheduler=cycle == 1)
    # Preserve the formal outer train_mode across both disjoint-update calls.
    # The repeated diagnostic inputs do not constitute scientific data/windows.
    with engine.train_mode(disable_auto_offload=False):
        for cycle, data in enumerate(datasets):
            if cycle == 1 and boundary_sync == '1':
                synchronize_worker_boundary(record, torch.cuda.synchronize,
                                            lambda row: publish(output, row))
            record.update(stage=f'native_worker_cycle_{cycle}_running', active_cycle=cycle)
            publish(output, record)
            start = time.monotonic()
            result = native_train_batch(worker, data)
            elapsed = time.monotonic() - start
            metrics = tu.get(result, 'metrics')
            loss, norm = float(metrics['loss']), float(metrics['grad_norm'])
            assert math.isfinite(loss) and math.isfinite(norm) and norm >= 0
            record['cycles'].append({'cycle': cycle, 'native_worker_seconds': elapsed,
                                     'loss': loss, 'finite_grad_norm': norm,
                                     'metric_keys': sorted(metrics)})
            record['peak_gpu_memory_bytes'] = torch.cuda.max_memory_allocated()
            record['diagnostic_optimizer_steps_returned'] += 1
            record['stage'] = f'native_worker_cycle_{cycle}_returned'
            publish(output, record)
    assert record['loss_avg_replacements'] == (2 if loss_sum == '1' else 0)
    assert record['cpu_loss_postprocess_calls'] == (2 if loss_cpu == '1' else 0)
    assert record['loss_reduce_tensor_devices'] == ([('cpu' if loss_cpu == '1' else 'cuda')] * 2 if loss_sum == '1' else [])
    assert record['local_logging_postprocess_calls'] == (2 if local_postprocess == '1' else 0)
    if optimizer_replay == '1':
        assert record['optimizer_backend_replay']['native_optimizer_call'] == 2
        assert record['optimizer_backend_replay']['devices'] == ['cuda']
    assert worker.device_name == 'cuda'
    # All extra verification is after both timed native worker calls.
    assert all(torch.isfinite(shared.local_tensor(p)).all() for n, p in engine.module.named_parameters()
               if 'lora_' in n or '.adapter_' in n)
    record.update(stage='passed_two_native_worker_cycles', finished_unix=time.time(),
                  peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),
                  scope='Two diagnostic ART updates with 2 fixed samples/rank and 3 captured timesteps; metric/MFU collectives bypassed only when local_only_postprocess=True. Optimizer replay adds GPU clone/comparison work only when gpu_optimizer_backend_replay=True; timedcallthen includes thiswork. Not 128-sample/19-timestep production equivalence or scientific updates.')
    publish(output, record)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
