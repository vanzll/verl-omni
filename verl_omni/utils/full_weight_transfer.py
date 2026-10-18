import time

import zmq
from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightSender


class FullSyncSocket:
    def __init__(self, socket, timeout_seconds, receiver_future, policy_version):
        self.socket = socket
        self.timeout_seconds = timeout_seconds
        self.receiver_future = receiver_future
        self.policy_version = policy_version
        self.phase = "handshake"
        self.buckets = 0

    def __getattr__(self, name):
        return getattr(self.socket, name)

    def send_pyobj(self, payload, *args, **kwargs):
        if isinstance(payload, dict) and "bucket_meta" in payload:
            self.buckets += 1
            kind = "terminal" if payload["is_last"] else "bucket"
            if any(value.get("handle") is not None for value in payload["bucket_meta"].values()):
                kind = "large_tensor"
            self.phase = f"{kind}_{self.buckets}"
        return self.socket.send_pyobj(payload, *args, **kwargs)

    def recv(self, *args, **kwargs):
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            if self.receiver_future is not None and self.receiver_future.done():
                self.receiver_future.result()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Full weight ACK timeout: policy_version={self.policy_version}, phase={self.phase}, "
                    f"timeout_seconds={self.timeout_seconds}"
                )
            if self.socket.poll(timeout=min(1000, max(1, int(remaining * 1000))), flags=zmq.POLLIN):
                return self.socket.recv(*args, **kwargs, flags=zmq.NOBLOCK)


class FullBoundedWeightSender(BucketedWeightSender):
    def __init__(self, *args, receiver_future, policy_version, timeout_seconds=300, **kwargs):
        super().__init__(*args, **kwargs)
        self.receiver_future = receiver_future.future() if receiver_future is not None else None
        self.policy_version = policy_version
        self.timeout_seconds = timeout_seconds

    def _init_socket(self):
        super()._init_socket()
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.SNDTIMEO, int(self.timeout_seconds * 1000))
        self.socket.setsockopt(zmq.RCVTIMEO, int(self.timeout_seconds * 1000))
        self.socket = FullSyncSocket(self.socket, self.timeout_seconds, self.receiver_future, self.policy_version)

    async def async_send_weights(self, weights):
        try:
            await super().async_send_weights(weights)
        except zmq.Again as error:
            raise TimeoutError(f"Full weight send timed out: policy_version={self.policy_version}") from error
