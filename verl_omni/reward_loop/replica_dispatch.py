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
"""Phase-local dispatch across complete reward replicas."""

import asyncio
import time

from verl.protocol import DataProto, pad_dataproto_to_divisor


def project_reward_data(data: DataProto, excluded_tensor_keys) -> DataProto:
    excluded = set(excluded_tensor_keys or ())
    protected = {"responses", "audio", "audio_sample_rate"}
    if excluded & protected or any(name.startswith("media_") for name in excluded):
        raise ValueError("Reward transport cannot exclude response or media fields")
    if not excluded or data.batch is None:
        return data
    return DataProto(
        batch=data.batch.exclude(*excluded),
        non_tensor_batch=data.non_tensor_batch,
        meta_info=data.meta_info,
    )


def tensor_payload_inventory(data: DataProto) -> dict:
    if data.batch is None:
        return {}
    return {
        name: {"shape": list(value.shape), "dtype": str(value.dtype),
               "logical_bytes": value.numel() * value.element_size()}
        for name, value in data.batch.items()
        if hasattr(value, "numel")
    }


class SharedRewardBatchCache:
    def __init__(self, consumers, put, max_cached_batches=16):
        if type(max_cached_batches) is not int or max_cached_batches <= 0:
            raise ValueError("max_cached_batches must be a positive integer")
        self.consumers = frozenset(consumers)
        self.put = put
        self.max_cached_batches = max_cached_batches
        self.entries = {}
        self.condition = asyncio.Condition()
        self.stopped = False
        self.materializations = 0
        self.cache_hits = 0
        self.put_seconds = 0.0
        self.peak_cached_batches = 0

    async def acquire(self, key, consumer, chunk):
        async with self.condition:
            await self.condition.wait_for(
                lambda: self.stopped or key in self.entries or len(self.entries) < self.max_cached_batches
            )
            if self.stopped:
                return None
            if key not in self.entries:
                started = time.perf_counter()
                reference = self.put(chunk)
                self.put_seconds += time.perf_counter() - started
                self.materializations += 1
                self.entries[key] = (reference, set(self.consumers))
                self.peak_cached_batches = max(self.peak_cached_batches, len(self.entries))
            else:
                self.cache_hits += 1
            reference, remaining = self.entries[key]
            remaining.remove(consumer)
            if not remaining:
                del self.entries[key]
                self.condition.notify_all()
            return reference

    async def abort(self):
        async with self.condition:
            self.stopped = True
            self.condition.notify_all()

    def clear(self):
        self.entries.clear()

    def diagnostics(self):
        return {"materializations": self.materializations, "cache_hits": self.cache_hits,
                "put_seconds": self.put_seconds, "peak_cached_batches": self.peak_cached_batches,
                "max_cached_batches": self.max_cached_batches}


async def dispatch_reward_groups(
    data: DataProto, worker_groups: dict, batch_sizes: dict[str, int], *, diagnostics: dict | None = None,
    batch_cache: SharedRewardBatchCache | None = None,
) -> dict[str, list]:
    """Score groups, restore sample order and drain accepted RPCs before returning.

    Groups absent from ``batch_sizes`` retain static padded splitting. Other
    groups dispatch at most one microbatch per replica, replenishing on completion.
    Failure or caller cancellation stops new dispatch, but does not cancel an RPC.
    """
    outputs = {}
    next_index = {}
    static_chunks = {}
    stopped = False
    started = time.perf_counter()
    if batch_cache is not None:
        if (
            set(batch_sizes) != set(worker_groups)
            or len(set(batch_sizes.values())) != 1
            or any(type(size) is not int or size <= 0 for size in batch_sizes.values())
        ):
            raise ValueError("Shared reward batches require equal dynamic dispatch sizes for every group")
        if batch_cache.consumers != frozenset(worker_groups):
            raise ValueError("Shared reward batch consumers must match worker groups")
    if diagnostics is not None:
        diagnostics["groups"] = {}
    for name, workers in worker_groups.items():
        if not workers:
            raise ValueError(f"Reward worker group {name!r} has no replicas")
        batch_size = batch_sizes.get(name)
        if batch_size is not None and (type(batch_size) is not int or batch_size <= 0):
            raise ValueError(f"Reward worker group {name!r} dispatch_batch_size must be a positive integer")
        next_index[name] = 0
        if diagnostics is not None:
            diagnostics["groups"][name] = {
                "replicas": [
                    {"samples": 0, "rpcs": 0, "slice_seconds": 0.0,
                     "submit_seconds": 0.0, "rpc_inflight_seconds": 0.0,
                     "worker_compute_seconds": 0.0}
                    for worker in workers
                ]
            }
        if batch_size is None and len(data):
            padded, _ = pad_dataproto_to_divisor(data, len(workers))
            static_chunks[name] = padded.chunk(len(workers))
            outputs[name] = [None] * len(padded)
        else:
            outputs[name] = [None] * len(data)

    async def run_replica(name, worker, replica_index):
        nonlocal stopped
        try:
            while not stopped and next_index[name] < len(data):
                slice_start = time.perf_counter()
                if name in static_chunks:
                    chunk = static_chunks[name][replica_index]
                    start = replica_index * len(chunk)
                else:
                    start = next_index[name]
                    stop = min(start + batch_sizes[name], len(data))
                    next_index[name] = stop
                    chunk = data[start:stop]
                submit_start = time.perf_counter()
                argument = chunk
                if batch_cache is not None:
                    argument = await batch_cache.acquire((start, start + len(chunk)), name, chunk)
                    if argument is None:
                        break
                remote_start = time.perf_counter()
                reference = worker.compute_score_batch.remote(argument)
                submitted = time.perf_counter()
                result = await reference
                completed = time.perf_counter()
                if len(result) != len(chunk):
                    raise ValueError(
                        f"Reward worker group {name!r} returned {len(result)} results for {len(chunk)} samples"
                    )
                outputs[name][start : start + len(chunk)] = result
                if diagnostics is not None:
                    stats = diagnostics["groups"][name]["replicas"][replica_index]
                    stats["samples"] += len(chunk)
                    stats["rpcs"] += 1
                    stats["max_inflight"] = 1
                    stats["slice_seconds"] += submit_start - slice_start
                    stats["submit_seconds"] += submitted - remote_start
                    stats["shared_reference_wait_seconds"] = (
                        stats.get("shared_reference_wait_seconds", 0.0) + remote_start - submit_start
                    )
                    stats["rpc_inflight_seconds"] += completed - submitted
                    stats.setdefault("first_submit_seconds", remote_start - started)
                    stats["last_complete_seconds"] = completed - started
                    profile = result[0].get("_reward_worker_profile") if result and isinstance(result[0], dict) else None
                    if profile:
                        stats["worker_compute_seconds"] += profile["compute_seconds"]
                        stats["worker"] = {
                            key: value for key, value in profile.items() if key != "compute_seconds"
                        }
                if name in static_chunks:
                    break
        except BaseException:
            stopped = True
            if batch_cache is not None:
                await batch_cache.abort()
            raise

    pending = asyncio.gather(
        *(
            run_replica(name, worker, index)
            for name, workers in worker_groups.items()
            for index, worker in enumerate(workers)
        ),
        return_exceptions=True,
    )
    cancellation = None
    while True:
        try:
            results = await asyncio.shield(pending)
            break
        except asyncio.CancelledError as exc:
            # Remote work can outlive a cancelled await; keep its ownership until completion.
            stopped = True
            cancellation = exc
            if batch_cache is not None:
                await batch_cache.abort()
    if batch_cache is not None:
        if diagnostics is not None:
            diagnostics["shared_rpc_batches"] = batch_cache.diagnostics()
        batch_cache.clear()
    if cancellation is not None:
        raise cancellation
    for result in results:
        if isinstance(result, BaseException):
            raise result
    if diagnostics is not None:
        diagnostics["dispatch_seconds"] = time.perf_counter() - started
    return {name: values[: len(data)] for name, values in outputs.items()}
