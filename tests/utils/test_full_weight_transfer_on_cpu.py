import ast
import asyncio
from concurrent.futures import Future
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import zmq


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "verl_omni/utils/full_weight_transfer.py"
TREE = ast.parse(PATH.read_text())
SOCKET = next(node for node in TREE.body if isinstance(node, ast.ClassDef) and node.name == "FullSyncSocket")
NAMESPACE = {"time": time, "zmq": zmq}
exec(compile(ast.Module(body=[SOCKET], type_ignores=[]), str(PATH), "exec"), NAMESPACE)
FullSyncSocket = NAMESPACE["FullSyncSocket"]


class FullTransferTests(unittest.TestCase):
    def setUp(self):
        self.context = zmq.Context()
        self.client = self.context.socket(zmq.REQ)
        self.client.setsockopt(zmq.LINGER, 0)
        self.client.setsockopt(zmq.SNDTIMEO, 100)
        self.endpoint = "inproc://full32-" + uuid.uuid4().hex
        self.future = Future()
        self.socket = FullSyncSocket(self.client, 0.2, self.future, 2)
        self.thread = None

    def tearDown(self):
        self.client.close(linger=0)
        if self.thread is not None:
            self.thread.join(timeout=2)
            self.assertFalse(self.thread.is_alive())
        self.context.term()

    def peer(self, mode):
        ready = threading.Event()

        def worker():
            socket = self.context.socket(zmq.REP)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.RCVTIMEO, 500)
            socket.bind(self.endpoint)
            ready.set()
            try:
                while True:
                    payload = socket.recv_pyobj()
                    bucket = isinstance(payload, dict) and "bucket_meta" in payload
                    if bucket and mode == "callback_error":
                        self.future.set_exception(ValueError("native callback failed"))
                        return
                    if bucket and payload["is_last"] and mode == "missing_terminal_ack":
                        return
                    socket.send(b"ack")
                    if bucket and payload["is_last"]:
                        self.future.set_result(None)
                        return
            finally:
                socket.close(linger=0)

        self.thread = threading.Thread(target=worker)
        self.thread.start()
        self.assertTrue(ready.wait(timeout=1))
        self.client.connect(self.endpoint)

    @staticmethod
    def bucket(last=False, large=False):
        return {"bucket_meta": {"weight": {"handle": ("fake", 1) if large else None}}, "is_last": last}

    def test_healthy_handshake_multiple_buckets_and_terminal_ack(self):
        self.peer("healthy")
        for payload in (("cpu-handle",), self.bucket(), self.bucket(), self.bucket(last=True)):
            self.socket.send_pyobj(payload)
            self.assertEqual(self.socket.recv(), b"ack")
        self.assertEqual(self.socket.phase, "terminal_3")

    def test_handshake_absent_is_bounded(self):
        self.client.connect(self.endpoint)
        self.socket.send_pyobj(("cpu-handle",))
        started = time.monotonic()
        with self.assertRaisesRegex(TimeoutError, "policy_version=2, phase=handshake"):
            self.socket.recv()
        self.assertLess(time.monotonic() - started, 1)

    def test_native_callback_error_interrupts_missing_ack(self):
        self.peer("callback_error")
        self.socket.send_pyobj(("cpu-handle",))
        self.socket.recv()
        self.socket.send_pyobj(self.bucket())
        started = time.monotonic()
        with self.assertRaisesRegex(ValueError, "native callback failed"):
            self.socket.recv()
        self.assertLess(time.monotonic() - started, 1)

    def test_missing_terminal_ack_is_bounded(self):
        self.peer("missing_terminal_ack")
        self.socket.send_pyobj(("cpu-handle",))
        self.socket.recv()
        self.socket.send_pyobj(self.bucket(last=True))
        with self.assertRaisesRegex(TimeoutError, "phase=terminal_1"):
            self.socket.recv()

    def test_large_tensor_ack_uses_the_same_guard(self):
        self.peer("healthy")
        self.socket.send_pyobj(self.bucket(large=True))
        self.socket.recv()
        self.assertEqual(self.socket.phase, "large_tensor_1")
        self.socket.send_pyobj(self.bucket(last=True))
        self.socket.recv()

    def test_adapter_failure_never_clears_or_commits_and_lora_delegates(self):
        path = ROOT / "verl_omni/workers/rollout/vllm_rollout/vllm_omni_async_server.py"
        tree = ast.parse(path.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "vLLMOmniServerAdapter")
        method = next(node for node in cls.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "update_weights")
        observed = []

        class Parent:
            async def update_weights(self, *args, **kwargs):
                observed.append("legacy")

        class Sender:
            def __init__(self, **kwargs):
                observed.append("sender_created")

            async def async_send_weights(self, weights):
                raise TimeoutError("missing native ACK")

        wrapper = ast.ClassDef(name="Adapter", bases=[ast.Name(id="Parent", ctx=ast.Load())],
                               keywords=[], body=[method], decorator_list=[])
        namespace = {"Parent": Parent, "os": os, "asyncio": asyncio}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])), str(path), "exec"), namespace)
        adapter = namespace["Adapter"]()

        async def rpc(*args, **kwargs):
            observed.append("receiver_started")
            return None

        async def clear():
            observed.append("clear")

        async def commit(value):
            observed.append("commit")

        adapter._execute_method = rpc
        adapter.use_shm = False
        adapter.zmq_handle = "inproc://isolated"
        adapter._has_server = True
        adapter.config = SimpleNamespace(checkpoint_engine=SimpleNamespace(update_weights_bucket_megabytes=2048))
        adapter.server_handle = SimpleNamespace(clear_kv_cache=SimpleNamespace(remote=clear),
                                                set_global_steps=SimpleNamespace(remote=commit))
        with patch.dict(os.environ, {"OMNI_FULL_FINETUNE": "1"}), patch.dict(sys.modules, {
            "verl_omni.utils.full_weight_transfer": SimpleNamespace(FullBoundedWeightSender=Sender),
        }):
            with self.assertRaisesRegex(TimeoutError, "missing native ACK"):
                asyncio.run(adapter.update_weights(iter(()), global_steps=1))
        self.assertNotIn("clear", observed)
        self.assertNotIn("commit", observed)
        observed.clear()
        with patch.dict(os.environ, {"OMNI_FULL_FINETUNE": "0"}):
            asyncio.run(adapter.update_weights(iter(()), global_steps=1))
        self.assertEqual(observed, ["legacy"])

    def test_native_recv_sites_are_all_wrapped_by_init_socket(self):
        native = Path(sys.executable).parent.parent / "lib/python3.12/site-packages/verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py"
        tree = ast.parse(native.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BucketedWeightSender")
        recv = [node for node in ast.walk(cls) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "recv"]
        self.assertEqual(len(recv), 4)
        self.assertTrue(all(isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "socket" for node in recv))


if __name__ == "__main__":
    unittest.main()
