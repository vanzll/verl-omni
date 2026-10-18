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
"""Verify the A1 permutation through the pinned DataProto reorder/chunk path."""

import numpy as np
import pytest
import torch

pytest.importorskip("verl")

from verl.protocol import DataProto  # noqa: E402
from verl_omni.trainer.diffusion.diffusion_art import art_prompt_balanced_indices  # noqa: E402


def _global_update_uids(shards, update):
    rows = [shard.non_tensor_batch["uid"][update * 8 : (update + 1) * 8] for shard in shards]
    return np.concatenate(rows)


def test_rank_major_a1_reorder_keeps_tensor_identity_and_prompt_coverage():
    uid = np.repeat(np.arange(32), 8)
    batch = DataProto.from_dict(
        tensors={"sample_id": torch.arange(256)},
        non_tensors={"uid": uid, "sample_id_copy": np.arange(256)},
    )
    order = art_prompt_balanced_indices(uid, group_size=8, seed=43, dp_size=16)
    batch.reorder(torch.as_tensor(order, dtype=torch.long))
    np.testing.assert_array_equal(batch.batch["sample_id"].numpy(), batch.non_tensor_batch["sample_id_copy"])

    shards = batch.chunk(16)
    assert len(shards) == 16 and all(len(shard) == 16 for shard in shards)
    for update in (0, 1):
        np.testing.assert_array_equal(
            np.bincount(_global_update_uids(shards, update), minlength=32), np.full(32, 4)
        )
    ids = np.concatenate([shard.batch["sample_id"].numpy() for shard in shards])
    np.testing.assert_array_equal(np.sort(ids), np.arange(256))


def test_original_contiguous_order_covers_half_the_prompts_per_update():
    uid = np.repeat(np.arange(32), 8)
    batch = DataProto.from_dict(tensors={"sample_id": torch.arange(256)}, non_tensors={"uid": uid})
    shards = batch.chunk(16)
    assert np.unique(_global_update_uids(shards, 0)).size == 16
    assert np.unique(_global_update_uids(shards, 1)).size == 16
