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
"""CPU contracts for reward-replica dispatch."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
import torch
from verl.protocol import DataProto

from verl_omni.reward_loop.replica_dispatch import (
    SharedRewardBatchCache,
    dispatch_reward_groups,
    project_reward_data,
    tensor_payload_inventory,
)


def test_reward_projection_preserves_media_metadata_and_original_training_tensors():
    import numpy as np

    responses = torch.arange(48, dtype=torch.uint8).reshape(4, 3, 2, 2)
    cache = torch.arange(4 * 19 * 64, dtype=torch.float32).reshape(4, 19, 64)
    original = DataProto.from_dict(
        tensors={"responses": responses, "media_artifact__video": responses,
                 "all_conditional_velocities": cache, "all_latents": cache.clone(), "prompt_embeds": cache[:, 0]},
        non_tensors={"uid": np.array(["a", "b", "c", "d"], dtype=object),
                     "reward_inputs": np.array([{"text": "same"}] * 4, dtype=object)},
        meta_info={"validate": False, "sampling_rate": 48000},
    )
    projected = project_reward_data(original, ["all_conditional_velocities", "all_latents", "prompt_embeds", "missing"])
    assert set(projected.batch.keys()) == {"responses", "media_artifact__video"}
    assert projected.non_tensor_batch is original.non_tensor_batch
    assert projected.meta_info == original.meta_info
    torch.testing.assert_close(projected.batch["responses"], responses, rtol=0, atol=0)
    torch.testing.assert_close(original.batch["all_conditional_velocities"], cache, rtol=0, atol=0)
    torch.testing.assert_close(original.batch["prompt_embeds"], cache[:, 0], rtol=0, atol=0)
    assert tensor_payload_inventory(original)["all_conditional_velocities"]["logical_bytes"] == cache.numel() * 4


@pytest.mark.parametrize("protected", ["responses", "audio", "audio_sample_rate", "media_artifact__video"])
def test_reward_projection_rejects_media_exclusion(protected):
    with pytest.raises(ValueError, match="media"):
        project_reward_data(_data(2), [protected])


@pytest.mark.asyncio
@pytest.mark.parametrize("shared", [False, True])
async def test_projected_profiled_manager_preserves_raw_scores_and_metadata(monkeypatch, tmp_path, shared):
    import numpy as np

    from verl_omni.reward_loop.reward_loop import OmniRewardLoopManager

    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    (tmp_path / "logs").mkdir()
    data = DataProto.from_dict(
        tensors={"responses": torch.arange(12, dtype=torch.uint8).reshape(3, 4),
                 "all_latents": torch.ones(3, 19, 64)},
        non_tensors={"uid": np.array(["a", "b", "c"], dtype=object)},
    )
    received = []
    monkeypatch.setattr("verl_omni.reward_loop.reward_loop.ray.put", lambda chunk: SimpleNamespace(data=chunk))

    async def compute(chunk):
        chunk = chunk.data if shared else chunk
        assert "all_latents" not in chunk.batch
        received.extend(chunk.non_tensor_batch["uid"].tolist())
        return [{"reward_score": float(value), "reward_extra_info": {"reward/test": float(value)}}
                for value in chunk.batch["responses"].sum(dim=1)]

    async def lifecycle():
        return None

    manager = object.__new__(OmniRewardLoopManager)
    manager.config = SimpleNamespace(reward=SimpleNamespace(
        reward_functions={"test": {}}, profile_dispatch=True, rpc_excluded_tensor_keys=["all_latents"],
        share_rpc_batches=shared, shared_rpc_cache_batches=1,
    ))
    manager._preserve_reward_components = True
    manager._score_lock = asyncio.Lock()
    manager._reward_payload_profiled_phases = {(False, 3)}
    manager.multi_reward_model_manager = SimpleNamespace(
        models={"test": object()}, wake_up=lifecycle, sleep=lifecycle,
    )
    manager._reward_worker_groups = {"test": [_worker(compute)]}
    manager._reward_dispatch_batch_sizes = {"test": 2}
    result = await manager.async_compute_rm_score(data)
    torch.testing.assert_close(result.batch["rm_scores"][:, 0], data.batch["responses"].sum(dim=1).float())
    assert result.non_tensor_batch["reward/test"].tolist() == [6.0, 22.0, 38.0]
    assert received == ["a", "b", "c"]
    assert "all_latents" in data.batch
    assert result.meta_info["reward_names"] == ["test"]
    assert "perf/reward/wake_seconds" in result.meta_info["reward_extra_keys"]
    assert (tmp_path / "logs/reward_dispatch_profile.jsonl").is_file()


@pytest.mark.asyncio
async def test_profiled_dispatch_keeps_exact_scores_order_and_one_inflight_per_replica():
    calls = [[], []]
    diagnostics = {}
    result = await dispatch_reward_groups(
        _data(7), {"native": [_recording_worker(records) for records in calls]},
        {"native": 2}, diagnostics=diagnostics,
    )
    assert result == {"native": list(range(7))}
    replicas = diagnostics["groups"]["native"]["replicas"]
    assert sum(row["samples"] for row in replicas) == 7
    assert sum(row["rpcs"] for row in replicas) == 4
    assert all(row[key] >= 0 for row in replicas for key in
               ("slice_seconds", "submit_seconds", "rpc_inflight_seconds"))


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [0, 1, 7, 32])
async def test_shared_batches_preserve_every_group_order_and_bound_cache(size):
    writes = []
    received = {}

    def put(chunk):
        reference = SimpleNamespace(data=chunk)
        writes.append(reference)
        return reference

    groups = {}
    for name, replicas in {"va": 8, "hps": 8, "ab": 4, "clap": 4, "ds": 8}.items():
        async def compute(reference, name=name):
            received.setdefault(name, []).append(reference)
            await asyncio.sleep(0.002 if name == "hps" else 0)
            return _sample_ids(reference.data)

        groups[name] = [_worker(compute) for index in range(replicas)]
    cache = SharedRewardBatchCache(groups, put, max_cached_batches=1)
    diagnostics = {}
    result = await asyncio.wait_for(dispatch_reward_groups(
        _data(size), groups, dict.fromkeys(groups, 2), diagnostics=diagnostics, batch_cache=cache,
    ), timeout=5)
    assert result == {name: list(range(size)) for name in groups}
    assert len(writes) == (size + 1) // 2
    assert cache.cache_hits == 4 * len(writes)
    assert cache.peak_cached_batches <= 1
    assert not cache.entries
    for name in received:
        assert {id(reference) for reference in received[name]} == {id(reference) for reference in writes}
    assert diagnostics["shared_rpc_batches"]["materializations"] == len(writes)


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_kind", ["cancel", "rpc_failure", "put_failure"])
async def test_shared_batch_abort_wakes_waiters_and_drains_accepted_calls(stop_kind):
    release = asyncio.Event()
    accepted = []
    completed = []
    fail_rpc = asyncio.Event()
    writes = []

    def put(chunk):
        if stop_kind == "put_failure" and writes:
            raise RuntimeError("put failure")
        reference = SimpleNamespace(data=chunk)
        writes.append(reference)
        return reference

    async def compute(reference, fail=False):
        accepted.append(reference)
        try:
            if fail and stop_kind == "rpc_failure":
                await fail_rpc.wait()
                raise RuntimeError("rpc failure")
            await release.wait()
            return _sample_ids(reference.data)
        finally:
            completed.append(reference)

    groups = {"fast": [_worker(lambda reference: compute(reference, True)) for index in range(3)],
              "slow": [_worker(compute)]}
    cache = SharedRewardBatchCache(groups, put, max_cached_batches=1)
    task = asyncio.create_task(dispatch_reward_groups(
        _data(12), groups, dict.fromkeys(groups, 2), batch_cache=cache,
    ))
    for index in range(100):
        if len(accepted) >= (2 if stop_kind == "put_failure" else 3):
            break
        await asyncio.sleep(0.001)
    assert accepted
    if stop_kind == "cancel":
        task.cancel()
    else:
        fail_rpc.set()
    await asyncio.sleep(0.01)
    assert not task.done()
    accepted_before_release = len(accepted)
    release.set()
    with pytest.raises(asyncio.CancelledError if stop_kind == "cancel" else RuntimeError):
        await asyncio.wait_for(task, timeout=2)
    assert len(completed) == len(accepted) == accepted_before_release
    assert not cache.entries


@pytest.mark.asyncio
@pytest.mark.parametrize("sizes", [{"va": 2}, {"va": 2, "ab": 3}, {"va": None, "ab": None}])
async def test_shared_batches_reject_incompatible_dispatch_partition(sizes):
    groups = {"va": [_recording_worker([])], "ab": [_recording_worker([])]}
    cache = SharedRewardBatchCache(groups, lambda chunk: chunk)
    with pytest.raises(ValueError, match="equal dynamic"):
        await dispatch_reward_groups(_data(4), groups, sizes, batch_cache=cache)


def _data(size: int) -> DataProto:
    return DataProto.from_dict(tensors={"sample_id": torch.arange(size, dtype=torch.int64)})


def _sample_ids(data: DataProto) -> list[int]:
    return data.batch["sample_id"].tolist()


def _worker(compute):
    return SimpleNamespace(compute_score_batch=SimpleNamespace(remote=compute))


def _recording_worker(calls: list[list[int]]):
    async def compute(data):
        sample_ids = _sample_ids(data)
        calls.append(sample_ids)
        return sample_ids

    return _worker(compute)


@pytest.mark.asyncio
@pytest.mark.parametrize(("size", "replicas"), [(0, 3), (1, 3), (3, 5), (7, 4)])
async def test_static_dispatch_preserves_padded_one_chunk_per_replica(size, replicas):
    calls = [[] for _ in range(replicas)]
    workers = [_recording_worker(worker_calls) for worker_calls in calls]

    result = await dispatch_reward_groups(_data(size), {"static": workers}, {})

    assert result == {"static": list(range(size))}
    if size == 0:
        assert calls == [[] for _ in range(replicas)]
        return

    assert all(len(worker_calls) == 1 for worker_calls in calls)
    chunk_sizes = {len(worker_calls[0]) for worker_calls in calls}
    assert len(chunk_sizes) == 1
    assert sum(len(worker_calls[0]) for worker_calls in calls) >= size
    if size % replicas:
        assert sum(len(worker_calls[0]) for worker_calls in calls) > size


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [0, 1, 3, 7])
async def test_dynamic_dispatch_scores_each_sample_exactly_once_without_padding(size):
    calls = [[] for _ in range(4)]
    workers = [_recording_worker(worker_calls) for worker_calls in calls]

    result = await dispatch_reward_groups(_data(size), {"native": workers}, {"native": 2})

    dispatched = [sample_id for worker_calls in calls for batch in worker_calls for sample_id in batch]
    assert result == {"native": list(range(size))}
    assert sorted(dispatched) == list(range(size))
    assert len(dispatched) == size
    assert all(0 < len(batch) <= 2 for worker_calls in calls for batch in worker_calls)


@pytest.mark.asyncio
async def test_dynamic_dispatch_replenishes_the_first_free_replica_with_next_contiguous_slice():
    class GatedWorker:
        def __init__(self):
            self.calls = []
            self.entered = asyncio.Event()
            self.second_entered = asyncio.Event()
            self.release = asyncio.Event()
            self.active = 0

            async def compute(data):
                assert self.active == 0
                self.active += 1
                sample_ids = _sample_ids(data)
                self.calls.append(sample_ids)
                if len(self.calls) == 1:
                    self.entered.set()
                else:
                    self.second_entered.set()
                await self.release.wait()
                self.active -= 1
                return sample_ids

            self.compute_score_batch = SimpleNamespace(remote=compute)

    workers = [GatedWorker() for _ in range(3)]
    task = asyncio.create_task(dispatch_reward_groups(_data(7), {"native": workers}, {"native": 2}))
    await asyncio.gather(*(worker.entered.wait() for worker in workers))

    assert [worker.calls for worker in workers] == [[[0, 1]], [[2, 3]], [[4, 5]]]
    workers[1].release.set()
    await workers[1].second_entered.wait()
    assert workers[1].calls == [[2, 3], [6]]
    assert workers[0].calls == [[0, 1]]
    assert workers[2].calls == [[4, 5]]

    workers[0].release.set()
    workers[2].release.set()
    assert await task == {"native": list(range(7))}
    assert all(worker.active == 0 for worker in workers)


@pytest.mark.asyncio
async def test_mixed_static_and_dynamic_groups_keep_independent_dispatch_contracts():
    static_calls = [[], []]
    dynamic_calls = [[], []]
    groups = {
        "shared": [_recording_worker(calls) for calls in static_calls],
        "native": [_recording_worker(calls) for calls in dynamic_calls],
    }

    result = await dispatch_reward_groups(_data(3), groups, {"native": 1})

    assert result == {"shared": [0, 1, 2], "native": [0, 1, 2]}
    assert all(len(calls) == 1 for calls in static_calls)
    assert sum(len(calls[0]) for calls in static_calls) == 4
    dynamic_ids = [sample_id for calls in dynamic_calls for batch in calls for sample_id in batch]
    assert sorted(dynamic_ids) == [0, 1, 2]
    assert all(len(batch) == 1 for calls in dynamic_calls for batch in calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5, "2"])
async def test_invalid_dynamic_batch_size_fails_before_any_remote_send(batch_size):
    calls = []
    groups = {
        "valid": [_recording_worker(calls)],
        "invalid": [_recording_worker(calls)],
    }

    with pytest.raises(ValueError, match="dispatch_batch_size must be a positive integer"):
        await dispatch_reward_groups(_data(2), groups, {"invalid": batch_size})

    assert calls == []


@pytest.mark.asyncio
async def test_empty_group_fails_before_any_remote_send():
    calls = []

    with pytest.raises(ValueError, match="Reward worker group 'empty' has no replicas"):
        await dispatch_reward_groups(
            _data(2),
            {"valid": [_recording_worker(calls)], "empty": []},
            {},
        )

    assert calls == []


@pytest.mark.asyncio
async def test_result_cardinality_mismatch_is_not_retried():
    calls = []

    async def compute(data):
        sample_ids = _sample_ids(data)
        calls.append(sample_ids)
        return sample_ids[:-1]

    with pytest.raises(ValueError, match="returned 1 results for 2 samples"):
        await dispatch_reward_groups(_data(3), {"native": [_worker(compute)]}, {"native": 2})

    assert calls == [[0, 1]]


@pytest.mark.asyncio
async def test_worker_failure_stops_all_groups_and_drains_accepted_remote_calls():
    peer_entered = asyncio.Event()
    release_peer = asyncio.Event()
    failure_calls = []
    peer_calls = []

    async def fail(data):
        failure_calls.append(_sample_ids(data))
        await peer_entered.wait()
        raise RuntimeError("reward failed")

    async def peer(data):
        peer_calls.append(_sample_ids(data))
        peer_entered.set()
        await release_peer.wait()
        return _sample_ids(data)

    task = asyncio.create_task(
        dispatch_reward_groups(
            _data(4),
            {"failing": [_worker(fail)], "peer": [_worker(peer)]},
            {"failing": 1, "peer": 1},
        )
    )
    await peer_entered.wait()
    await asyncio.sleep(0)

    assert not task.done()
    release_peer.set()
    with pytest.raises(RuntimeError, match="reward failed"):
        await task

    assert failure_calls == [[0]]
    assert peer_calls == [[0]]


@pytest.mark.asyncio
async def test_synchronous_remote_failure_drains_an_active_peer_without_new_dispatch():
    peer_entered = asyncio.Event()
    release_peer = asyncio.Event()
    peer_calls = []
    failure_calls = []

    async def peer(data):
        peer_calls.append(_sample_ids(data))
        peer_entered.set()
        await release_peer.wait()
        return _sample_ids(data)

    class FailingRemote:
        def remote(self, data):
            assert peer_entered.is_set()
            failure_calls.append(_sample_ids(data))
            raise RuntimeError("remote submission failed")

    task = asyncio.create_task(
        dispatch_reward_groups(
            _data(4),
            {
                "peer": [_worker(peer)],
                "failing": [SimpleNamespace(compute_score_batch=FailingRemote())],
            },
            {"peer": 1, "failing": 1},
        )
    )
    await peer_entered.wait()
    await asyncio.sleep(0)
    assert not task.done()

    release_peer.set()
    with pytest.raises(RuntimeError, match="remote submission failed"):
        await task
    assert peer_calls == [[0]]
    assert failure_calls == [[0]]


@pytest.mark.asyncio
@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
@pytest.mark.parametrize("cancel_count", [1, 2], ids=["cancel-once", "cancel-twice"])
async def test_caller_cancellation_drains_each_accepted_rpc_without_replenishing(dynamic, cancel_count):
    calls = [[], []]
    entered = [asyncio.Event(), asyncio.Event()]
    release = [asyncio.Event(), asyncio.Event()]
    completed = [asyncio.Event(), asyncio.Event()]
    rpc_cancelled = [False, False]

    def blocking_worker(index):
        async def compute(data):
            sample_ids = _sample_ids(data)
            calls[index].append(sample_ids)
            entered[index].set()
            try:
                await release[index].wait()
            except asyncio.CancelledError:
                rpc_cancelled[index] = True
                raise
            completed[index].set()
            return sample_ids

        return _worker(compute)

    # Dynamic dispatch has queued samples beyond the two accepted RPCs.
    batch_sizes = {"native": 1} if dynamic else {}
    task = asyncio.create_task(
        dispatch_reward_groups(_data(5), {"native": [blocking_worker(0), blocking_worker(1)]}, batch_sizes)
    )
    try:
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in entered)), timeout=5)
        for _ in range(cancel_count):
            assert task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            assert rpc_cancelled == [False, False]
            assert not any(event.is_set() for event in completed)

        release[0].set()
        await asyncio.wait_for(completed[0].wait(), timeout=5)
        await asyncio.sleep(0)
        assert not task.done()
        assert not completed[1].is_set()
        assert [len(worker_calls) for worker_calls in calls] == [1, 1]

        release[1].set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(task), timeout=5)

        assert task.cancelled()
        assert all(event.is_set() for event in completed)
        assert rpc_cancelled == [False, False]
        assert [len(worker_calls) for worker_calls in calls] == [1, 1]
        if dynamic:
            assert calls == [[[0]], [[1]]]
    finally:
        for event in release:
            event.set()
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=5)


def test_dispatch_state_is_local_to_each_asyncio_run_call():
    calls = []
    worker = _recording_worker(calls)

    first = asyncio.run(dispatch_reward_groups(_data(3), {"native": [worker]}, {"native": 2}))
    second = asyncio.run(dispatch_reward_groups(_data(1), {"native": [worker]}, {"native": 2}))

    assert first == {"native": [0, 1, 2]}
    assert second == {"native": [0]}
    assert calls == [[0, 1], [2], [0]]
