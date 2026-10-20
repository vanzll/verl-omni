# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Serialized Full-policy producer/native receiver clock contracts on CPU."""

import ast
import asyncio
import copy
import importlib.util
import json
import logging
import os
from pathlib import Path
import pickle
import socket
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.distributed.tensor import DTensor


ROOT = Path(__file__).resolve().parents[3]
ENGINE = ROOT / 'verl_omni/workers/engine/fsdp/diffusers_impl.py'
NATIVE = ROOT / 'verl_omni/workers/rollout/vllm_rollout/utils.py'
SERVER = ROOT / 'verl_omni/workers/rollout/vllm_rollout/vllm_omni_async_server.py'


def load_file(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_methods(path, owner, names, namespace, parent=object):
    source = ast.parse(path.read_text())
    original = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == owner)
    methods = [node for node in original.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
               and node.name in names]
    if len(methods) != len(names):
        raise AssertionError('Missing production method')
    for method in methods:
        method.decorator_list = []
    namespace = dict(namespace, Parent=parent)
    wrapper = ast.ClassDef(name='Extracted', bases=[ast.Name(id='Parent', ctx=ast.Load())],
                           keywords=[], body=methods, decorator_list=[])
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, wrapper], type_ignores=[]))
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace['Extracted']


BASE = load_file('sync_clock_base', 'verl_omni/workers/engine/full_policy.py')
CLOCK = load_file('sync_clock_contract', 'verl_omni/utils/full_policy_sync.py')
AUDIT = load_file('sync_clock_tensor_audit', 'verl_omni/utils/tensor_audit.py')
with patch.dict(sys.modules, {'verl_omni.workers.engine.full_policy': BASE,
                              'verl_omni.utils.tensor_audit': AUDIT}):
    NFT = load_file('sync_clock_nft', 'verl_omni/workers/engine/full_nft_policy.py')
    COVERAGE = load_file('sync_clock_coverage', 'verl_omni/workers/rollout/vllm_rollout/full_weight_sync.py')


class TinyProducer(NFT.FullNFTPolicyMixin):
    def __init__(self, nft):
        self.model_config = SimpleNamespace(full_finetune=True, lora_rank=0, lora_adapter_path=None,
                                            algorithm='omni_nft' if nft else 'diffusion_art')
        self.module = TinyNative()
        self._initialize_full_policy()
        if nft:
            self._nft_old_module = copy.deepcopy(self.module).requires_grad_(False)
        self.copy_adapter()

    def _full_event(self, *args, **kwargs):
        pass


class TinyNative(torch.nn.Module):
    stacked_params_mapping = ()

    def __init__(self, tp_rank=0, tp_size=1):
        super().__init__()
        self.proj_in = torch.nn.Linear(2, 2 // tp_size)
        self.proj_in.tp_rank = tp_rank
        self.proj_in.tp_size = tp_size
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.config = SimpleNamespace(in_channels=2, audio_in_channels=2,
                                      cross_attention_dim=3, audio_cross_attention_dim=5)

    def forward(self, hidden_states, audio_hidden_states, **kwargs):
        return self.proj_in(hidden_states), self.proj_in(audio_hidden_states)

    def load_weights(self, weights):
        with torch.no_grad():
            for name, tensor in weights:
                dict(self.named_parameters())[name].copy_(tensor.chunk(self.tp_size, dim=0)[self.tp_rank])
        return set(dict(self.named_parameters()))


class ActualRoundtrip:
    def __init__(self, test, nft, updates=None):
        self.updates = updates or (1 if nft else 2)
        self.test = test
        self.nft = nft
        self.engine = test.producer_class(nft)
        self.metadata_fault = None
        self.ack_fault = None
        self.truncate = False
        self.calls = []
        self.payloads = []
        self.buckets = []
        self.rank = 0
        self.step = 0
        temporary = tempfile.TemporaryDirectory(prefix='nft-sync-clock-', dir=os.environ['TMPDIR'])
        test.addCleanup(temporary.cleanup)
        output = Path(temporary.name) / 'attempt'
        (output / 'logs').mkdir(parents=True)
        environment = patch.dict(os.environ, OMNI_RUNTIME_ROOT=temporary.name, OUTPUT_DIR=str(output),
                                 OMNI_VARIANT=f'NFT_FULL32_U{self.updates}' if nft else 'ART_FULL32')
        environment.start()
        test.addCleanup(environment.stop)
        distributed = patch.multiple(torch.distributed, get_rank=lambda: self.rank, get_world_size=lambda: 32)
        distributed.start()
        test.addCleanup(distributed.stop)
        native_class = load_methods(
            NATIVE, 'vLLMOmniColocateWorkerExtension',
            ['set_full_policy_sync_metadata', 'update_weights_from_ipc',
             '_full_local_policy_sync_state', '_commit_full_policy_sync'],
            {'os': os, 'torch': torch, 'socket': socket, 'Path': Path, 'json': json,
             'time': time, 'logger': logging.getLogger(__name__)},
        )
        self.natives = []
        for rank in range(32):
            native = native_class()
            native.device = torch.device('cpu', rank)
            native._pending_lora_peft_config = None
            native._get_zmq_handle = lambda: 'inproc://cpu-only'
            native._get_standard_weight_model_and_config = lambda: None
            native.model_runner = SimpleNamespace(pipeline=SimpleNamespace(transformer=TinyNative(rank % 2, 2)))
            self.natives.append(native)

        class ParentServer:
            async def set_global_steps(self, value):
                self.global_steps = value

        namespace = {'os': os, 'get_rollout_world_size': lambda config: 32}
        validator = next(node for node in ast.parse(SERVER.read_text()).body
                         if isinstance(node, ast.FunctionDef) and node.name == '_validate_collective_rpc_reply')
        future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future, validator], type_ignores=[])),
                     str(SERVER), 'exec'), namespace)
        server_class = load_methods(
            SERVER, 'vLLMOmniHttpServer',
            ['collective_rpc', '_collective_rpc_with_ack', 'set_global_steps'], namespace, ParentServer,
        )
        self.server = server_class()
        self.server.global_steps = 0
        self.server.config = SimpleNamespace()
        self.server._invalidate_lora_request_cache = lambda: None
        self.server._generate_strategy = SimpleNamespace(collective_rpc_stage_ids=lambda method: [0])
        self.server.engine = SimpleNamespace(num_stages=1, collective_rpc=self.native_rpc)
        adapter_class = load_methods(SERVER, 'vLLMOmniServerAdapter', ['_execute_method', 'update_weights'],
                                     {'os': os, 'asyncio': asyncio})
        self.adapter = adapter_class()
        self.adapter._ensure_server_handle = lambda: True
        self.adapter._has_server = True
        self.adapter.use_shm = False
        self.adapter.zmq_handle = 'inproc://cpu-only'
        self.adapter.config = SimpleNamespace(checkpoint_engine=SimpleNamespace(update_weights_bucket_megabytes=1))
        self.adapter.server_handle = SimpleNamespace(
            collective_rpc=SimpleNamespace(remote=self.remote_rpc),
            clear_kv_cache=SimpleNamespace(remote=self.clear),
            set_global_steps=SimpleNamespace(remote=self.server.set_global_steps),
        )
        self.worker = SimpleNamespace(actor=SimpleNamespace(engine=self.engine), rollout=self.adapter)
        harness = self

        class Sender:
            def __init__(self, **kwargs):
                harness.calls.append(('sender', kwargs['policy_version']))

            async def async_send_weights(self, values):
                items = [(name, tensor.clone()) for name, tensor in values]
                harness.buckets = [[item] for item in (items[:-1] if harness.truncate else items)]
                harness.ready.set()

        class Receiver:
            def __init__(self, **kwargs):
                harness.calls.append(('receiver', harness.rank))

            def receive_weights(self, on_bucket_received):
                for index, bucket in enumerate(harness.buckets):
                    on_bucket_received(bucket, index == len(harness.buckets) - 1)

        guidance = Path(sys.executable).parent.parent / 'lib/python3.12/site-packages/vllm_omni/diffusion/models/ltx2/ltx2_guidance.py'
        timestep = next(node for node in ast.walk(ast.parse(guidance.read_text()))
                        if isinstance(node, ast.FunctionDef) and node.name == 'timestep_kwargs')
        timestep.decorator_list = []
        scope = {'torch': torch}
        exec(compile(ast.Module(body=[timestep], type_ignores=[]), str(guidance), 'exec'), scope)
        modules = {
            'verl_omni.utils.full_weight_transfer': SimpleNamespace(FullBoundedWeightSender=Sender),
            'verl.workers.rollout.vllm_rollout.bucketed_weight_transfer':
                SimpleNamespace(BucketedWeightReceiver=Receiver),
            'vllm.distributed': SimpleNamespace(get_tensor_model_parallel_rank=lambda: self.rank % 2,
                                               get_tensor_model_parallel_world_size=lambda: 2),
            'vllm_omni.diffusion.models.ltx2.ltx2_guidance':
                SimpleNamespace(LTXGuidanceExecutor=SimpleNamespace(timestep_kwargs=scope['timestep_kwargs'])),
        }
        module_patch = patch.dict(sys.modules, modules)
        module_patch.start()
        test.addCleanup(module_patch.stop)

    async def clear(self):
        self.calls.append(('clear', self.step))

    def remote_rpc(self, method, **kwargs):
        self.calls.append((method, kwargs.get('require_ack')))
        payload = pickle.dumps(kwargs)
        if method == 'set_full_policy_sync_metadata':
            self.payloads.append(payload)
        return asyncio.create_task(self.server.collective_rpc(method, **pickle.loads(payload)))

    async def native_rpc(self, method, **kwargs):
        if method == 'update_weights_from_ipc':
            await self.ready.wait()
        replies = []
        for rank, native in enumerate(self.natives):
            self.rank = rank
            arguments = pickle.loads(pickle.dumps(kwargs.get('kwargs') or {}))
            try:
                if method == 'set_full_policy_sync_metadata':
                    if self.metadata_fault is not None:
                        self.metadata_fault(rank, arguments['metadata'])
                    result = native.set_full_policy_sync_metadata(**arguments)
                elif method == 'update_weights_from_ipc':
                    result = native.update_weights_from_ipc(**arguments)
                elif method == 'get_full_policy_sync_state':
                    result = native._full_local_policy_sync_state()
                    if self.ack_fault is not None:
                        self.ack_fault(rank, result)
                else:
                    raise AssertionError(method)
            except Exception as error:
                result = {'error': str(error), 'rank': rank}
            replies.append(result)
        return [[replies]]

    def advance(self):
        self.step += 1
        with torch.no_grad():
            for parameter in self.engine.module.parameters():
                parameter.add_(0.25)
        self.engine._full_policy_version += self.updates
        if self.nft:
            self.engine.ema_update_adapter(decay=0.001)
        else:
            self.engine.copy_adapter()

    async def sync(self):
        self.ready = asyncio.Event()
        weights, _ = self.engine.get_per_tensor_param(adapter_name='old')
        await self.test.metadata_rpc(self.worker, self.step)
        await self.adapter.update_weights(weights, global_steps=self.step)

    def assert_received_ema(self):
        version = self.updates * self.step
        self.test.assertEqual(self.server.global_steps, self.step)
        self.test.assertEqual(self.server._full_committed_policy_version, version)
        rpc = pickle.loads(self.payloads[-1])
        self.test.assertIs(rpc['require_ack'], True)
        self.test.assertGreater(rpc['timeout'], 0)
        wire = rpc['kwargs']['metadata']
        self.test.assertEqual(wire['policy_version'], version)
        self.test.assertEqual(wire['global_steps'], self.step)
        self.test.assertEqual(wire['policy_algorithm'], 'omni_nft' if self.nft else 'diffusion_art')
        self.test.assertEqual(wire['optimizer_updates_per_rollout'], self.updates)
        source = self.engine._nft_old_module if self.nft else self.engine.module
        self.test.assertEqual(self.engine._full_policy_version, version)
        self.test.assertEqual(self.engine._full_old_version, version)
        if self.nft and self.step:
            self.test.assertTrue(any(not torch.equal(parameter, self.engine.module.state_dict()[name])
                                     for name, parameter in source.state_dict().items()))
        for rank, native in enumerate(self.natives):
            state = native._full_committed_state
            self.test.assertTrue(hasattr(native, '_full_initial_probe'))
            self.test.assertEqual(state['policy_version'], version)
            self.test.assertEqual(state['policy_algorithm'], wire['policy_algorithm'])
            self.test.assertEqual(state['optimizer_updates_per_rollout'], wire['optimizer_updates_per_rollout'])
            self.test.assertEqual(state['post_load_checks'], 2)
            self.test.assertIn('tp_dimension_0', state['checked_kinds'])
            if self.step == 1:
                self.test.assertTrue(any(branch['changed_elements'] > 0
                                         for branch in state['fixed_input_output_change'].values()))
            model = native.model_runner.pipeline.transformer
            for name, parameter in model.named_parameters():
                expected = source.state_dict()[name].chunk(2, dim=0)[rank % 2]
                torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
        self.test.assertIn(('set_full_policy_sync_metadata', True), self.calls)
        self.test.assertIn(('update_weights_from_ipc', True), self.calls)
        self.test.assertIn(('sender', version), self.calls)


class FullNFTSyncClockTests(unittest.TestCase):
    def setUp(self):
        self.assertFalse(torch.cuda.is_initialized())
        modules = {'verl_omni.utils.full_policy_sync': CLOCK,
                   'verl_omni.utils.tensor_audit': AUDIT,
                   'verl_omni.workers.rollout.vllm_rollout.full_weight_sync': COVERAGE}
        module_patch = patch.dict(sys.modules, modules)
        module_patch.start()
        self.addCleanup(module_patch.stop)
        environment_patch = patch.dict(os.environ, OMNI_FULL_FINETUNE='1', OMNI_VARIANT='NFT_FULL32_U1',
                                        CUDA_VISIBLE_DEVICES=','.join(f'cpu-{rank}' for rank in range(32)))
        environment_patch.start()
        self.addCleanup(environment_patch.stop)
        self.producer_class = load_methods(
            ENGINE, 'DiffusersFSDPEngine', ['get_per_tensor_param'],
            {'torch': torch, 'DTensor': DTensor, 'tensor_sample': AUDIT.tensor_sample,
             'convert_weight_keys': lambda values, module: values}, TinyProducer,
        )
        self.rank_context = SimpleNamespace(rank=0)
        native_torch = SimpleNamespace(distributed=SimpleNamespace(
            get_rank=lambda: self.rank_context.rank, get_world_size=lambda: 32),
            cuda=SimpleNamespace(current_device=lambda: self.fail('Unexpected CUDA operation')))
        self.native_class = load_methods(
            NATIVE, 'vLLMOmniColocateWorkerExtension',
            ['set_full_policy_sync_metadata', '_full_local_policy_sync_state'],
            {'os': os, 'torch': native_torch, 'socket': socket},
        )
        self.metadata_rpc = self.load_metadata_rpc()

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def load_metadata_rpc(self):
        path = ROOT / 'verl_omni/workers/engine_workers.py'
        tree = ast.parse(path.read_text())
        matches = [node for node in ast.walk(tree) if isinstance(node, ast.If)
                   and any(isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
                           and child.func.attr == '_execute_method'
                           and child.args and isinstance(child.args[0], ast.Constant)
                           and child.args[0].value == 'set_full_policy_sync_metadata'
                           for statement in node.body for child in ast.walk(statement))
                   and len(node.body) == 2]
        self.assertEqual(len(matches), 1)
        wrapper = ast.parse('async def send_metadata(self, global_steps):\n    pass\n')
        wrapper.body[0].body = [matches[0]]
        namespace = {}
        exec(compile(ast.fix_missing_locations(wrapper), str(path), 'exec'), namespace)
        return namespace['send_metadata']

    def native(self):
        native = self.native_class()
        native.device = SimpleNamespace(index=0)
        return native

    def prepare_export(self, nft, step):
        engine = self.producer_class(nft)
        if step:
            with torch.no_grad():
                for parameter in engine.module.parameters():
                    parameter.add_(0.25)
            engine._full_policy_version = step if nft else 2 * step
            if nft:
                engine._nft_ema_updates = step - 1
                engine.ema_update_adapter(decay=0.001)
            else:
                engine.copy_adapter()
        weights, peft = engine.get_per_tensor_param(adapter_name='old')
        self.assertIsNone(peft)
        return engine, list(weights)

    def test_serialized_producer_native_32rank_commit_u0_u1_u2_and_art_u2(self):
        for nft in (True, False):
            for step in (0, 1, 2):
                with self.subTest(nft=nft, step=step):
                    os.environ['OMNI_VARIANT'] = 'NFT_FULL32_U1' if nft else 'ART_FULL32'
                    engine, weights = self.prepare_export(nft, step)
                    version = step if nft else 2 * step
                    natives = [self.native() for _ in range(32)]
                    payloads = []

                    async def metadata_rpc(method, *, kwargs):
                        self.assertEqual(method, 'set_full_policy_sync_metadata')
                        payload = pickle.dumps(kwargs['metadata'])
                        payloads.append(payload)
                        for native in natives:
                            native.set_full_policy_sync_metadata(pickle.loads(payload))

                    worker = SimpleNamespace(actor=SimpleNamespace(engine=engine),
                                             rollout=SimpleNamespace(_execute_method=metadata_rpc))
                    asyncio.run(self.metadata_rpc(worker, step))
                    self.assertEqual(len(payloads), 1)
                    wire = pickle.loads(payloads[0])
                    self.assertEqual(wire['policy_version'], version)
                    self.assertEqual(wire['global_steps'], step)
                    self.assertEqual(wire['policy_algorithm'], 'omni_nft' if nft else 'diffusion_art')
                    self.assertEqual(wire['optimizer_updates_per_rollout'], 1 if nft else 2)
                    states = []
                    expected_weights = engine._nft_old_module if nft else engine.module
                    for rank, native in enumerate(natives):
                        model = TinyNative()
                        audit = COVERAGE.FullWeightCoverage(model, native._full_pending_metadata)
                        audit.load_bucket(weights)
                        native._full_pending_metadata = None
                        native._full_committed_state = audit.finish()
                        native._full_weight_sync_valid = True
                        native._full_initial_probe = object()
                        native.device.index = rank
                        self.rank_context.rank = rank
                        states.append(native._full_local_policy_sync_state())
                        for name, parameter in model.named_parameters():
                            torch.testing.assert_close(parameter, expected_weights.state_dict()[name], rtol=0, atol=0)
                    self.assertEqual(engine._full_policy_version, version)
                    self.assertEqual(engine._full_old_version, version)
                    self.check_server_and_sender(states, weights, step, version)

    def check_server_and_sender(self, states, weights, step, version):
        class ParentServer:
            async def set_global_steps(self, value):
                self.global_steps = value

        server_class = load_methods(SERVER, 'vLLMOmniHttpServer', ['set_global_steps'],
                                    {'os': os, 'get_rollout_world_size': lambda config: 32}, ParentServer)
        server = server_class()
        server.config = SimpleNamespace()
        server.global_steps = 0
        server._invalidate_lora_request_cache = lambda: None

        async def collective_rpc(**kwargs):
            return [[states]]

        server.collective_rpc = collective_rpc
        observed = []

        class Sender:
            def __init__(self, **kwargs):
                observed.append(('sender_version', kwargs['policy_version']))

            async def async_send_weights(self, values):
                self_weights = list(values)
                if len(self_weights) != len(weights):
                    raise AssertionError('Unexpected source stream')

        class ParentAdapter:
            pass

        adapter_class = load_methods(SERVER, 'vLLMOmniServerAdapter', ['update_weights'],
                                     {'os': os, 'asyncio': asyncio}, ParentAdapter)
        adapter = adapter_class()
        adapter.use_shm = False
        adapter.zmq_handle = 'inproc://cpu-only'
        adapter._has_server = True
        adapter.config = SimpleNamespace(checkpoint_engine=SimpleNamespace(update_weights_bucket_megabytes=1))

        async def receive(*args, **kwargs):
            return None

        async def clear():
            observed.append(('clear', step))

        adapter._execute_method = receive
        adapter.server_handle = SimpleNamespace(clear_kv_cache=SimpleNamespace(remote=clear),
                                                set_global_steps=SimpleNamespace(remote=server.set_global_steps))
        with patch.dict(sys.modules, {'verl_omni.utils.full_weight_transfer':
                                      SimpleNamespace(FullBoundedWeightSender=Sender)}):
            asyncio.run(adapter.update_weights(iter(weights), global_steps=step))
        self.assertEqual(observed, [('sender_version', version), ('clear', step)])
        self.assertEqual(server.global_steps, step)
        self.assertEqual(server._full_committed_policy_version, version)
        bad = copy.deepcopy(states)
        bad[-1]['state']['policy_version'] += 1
        states[:] = bad
        with self.assertRaises(RuntimeError):
            asyncio.run(server.set_global_steps(step))
        self.assertEqual(server._full_committed_policy_version, version)

    def test_native_missing_or_forged_nft_tags_and_wrong_clock_fail_closed(self):
        engine, _ = self.prepare_export(True, 1)
        good = dict(engine._full_sync_metadata, global_steps=1)
        cases = []
        for key in ('policy_algorithm', 'optimizer_updates_per_rollout'):
            missing = dict(good)
            del missing[key]
            cases.append(missing)
        cases.extend(dict(good, **updates) for updates in (
            {'policy_algorithm': 'diffusion_art'}, {'optimizer_updates_per_rollout': 2},
            {'optimizer_updates_per_rollout': True}, {'policy_version': 2}, {'global_steps': 2},
            {'global_steps': True}, {'policy_version': True}, {'policy_algorithm': 'typo'},
        ))
        for metadata in cases:
            with self.subTest(metadata={key: metadata.get(key) for key in good if key != 'tensors'}):
                native = self.native()
                with self.assertRaises(ValueError):
                    native.set_full_policy_sync_metadata(pickle.loads(pickle.dumps(metadata)))
                self.assertIsNone(getattr(native, '_full_pending_metadata', None))

    def test_art_keeps_even_clock_and_cannot_adopt_nft_tags(self):
        os.environ['OMNI_VARIANT'] = 'ART_FULL32'
        engine, _ = self.prepare_export(False, 1)
        good = dict(engine._full_sync_metadata, global_steps=1)
        for metadata in (dict(good, policy_version=1),
                         dict(good, policy_algorithm='omni_nft', optimizer_updates_per_rollout=1)):
            with self.assertRaises(ValueError):
                self.native().set_full_policy_sync_metadata(metadata)
        legacy = {key: value for key, value in good.items()
                  if key not in ('policy_algorithm', 'optimizer_updates_per_rollout')}
        with self.assertRaises(ValueError):
            self.native().set_full_policy_sync_metadata(legacy)
        with self.assertRaises(ValueError):
            self.native().set_full_policy_sync_metadata(dict(legacy, policy_version=1))

    def test_native_pending_transaction_and_rollback_guards_remain(self):
        engine, _ = self.prepare_export(True, 1)
        metadata = dict(engine._full_sync_metadata, global_steps=1)
        native = self.native()
        native.set_full_policy_sync_metadata(metadata)
        with self.assertRaises(RuntimeError):
            native.set_full_policy_sync_metadata(metadata)
        native._full_pending_metadata = None
        native._full_committed_state = dict(metadata, policy_version=2)
        with self.assertRaises(ValueError):
            native.set_full_policy_sync_metadata(metadata)

    def test_actual_setter_receiver_tp2_probe_and_aggregated_ack_continuous_versions(self):
        for nft in (True, False):
            with self.subTest(nft=nft):
                harness = ActualRoundtrip(self, nft)
                for step in (0, 1, 2):
                    if step:
                        harness.advance()
                    asyncio.run(harness.sync())
                    harness.assert_received_ema()

    def test_nft_u2_ema_export_native_tp2_ack_and_next_rollout(self):
        with patch.dict(os.environ, OMNI_VARIANT='NFT_FULL32_U2'):
            harness = ActualRoundtrip(self, nft=True, updates=2)
            for rollout in range(4):
                if rollout:
                    harness.advance()
                asyncio.run(harness.sync())
                harness.assert_received_ema()
            forged = CLOCK.full_policy_sync_clock(3, nft=True, updates_per_rollout=1)
            with self.assertRaises(ValueError):
                CLOCK.validate_full_policy_sync_clock(forged, nft=True)

    def test_checked_setter_rejection_never_starts_ipc_or_advances_clock(self):
        for fault in ('missing_tag', 'wrong_tag', 'wrong_clock', 'single_rank_rejection'):
            with self.subTest(fault=fault):
                harness = ActualRoundtrip(self, True)
                asyncio.run(harness.sync())
                harness.advance()
                harness.calls.clear()

                def corrupt(rank, metadata):
                    if fault == 'missing_tag':
                        metadata.pop('policy_algorithm')
                    elif fault == 'wrong_tag':
                        metadata['optimizer_updates_per_rollout'] = 2
                    elif fault == 'wrong_clock':
                        metadata['policy_version'] = 2
                    elif rank == 31:
                        metadata['policy_algorithm'] = 'diffusion_art'

                harness.metadata_fault = corrupt
                with self.assertRaises(RuntimeError):
                    asyncio.run(harness.sync())
                self.assertEqual(harness.calls, [('set_full_policy_sync_metadata', True)])
                self.assertEqual(harness.server.global_steps, 0)
                self.assertEqual(harness.server._full_committed_policy_version, 0)

    def test_missing_metadata_adapter_delta_and_partial_stream_do_not_commit(self):
        harness = ActualRoundtrip(self, True)
        native = harness.natives[0]
        with self.assertRaisesRegex(RuntimeError, 'tagged'):
            native.update_weights_from_ipc()
        for arguments in ({'peft_config': {}}, {'delta_flush': True}):
            with self.subTest(arguments=arguments), self.assertRaisesRegex(ValueError, 'adapter or delta'):
                native.update_weights_from_ipc(**arguments)
        asyncio.run(harness.sync())
        harness.advance()
        harness.truncate = True
        with self.assertRaises(RuntimeError):
            asyncio.run(harness.sync())
        self.assertEqual(harness.server.global_steps, 0)
        self.assertEqual(harness.server._full_committed_policy_version, 0)
        self.assertTrue(all(not native._full_weight_sync_valid for native in harness.natives))

    def test_single_rank_ack_losing_tag_does_not_advance_server_clock(self):
        harness = ActualRoundtrip(self, True)
        asyncio.run(harness.sync())
        harness.advance()

        def corrupt(rank, reply):
            if rank == 31:
                reply['state'].pop('policy_algorithm')

        harness.ack_fault = corrupt
        with self.assertRaises(RuntimeError):
            asyncio.run(harness.sync())
        self.assertEqual(harness.server.global_steps, 0)
        self.assertEqual(harness.server._full_committed_policy_version, 0)


if __name__ == '__main__':
    unittest.main()
