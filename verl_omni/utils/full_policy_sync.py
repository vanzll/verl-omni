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

import os


def full_policy_sync_tag(algorithm, updates_per_rollout=None):
    updates = {"omni_nft": 1, "diffusion_art": 2}
    if algorithm not in updates:
        raise ValueError("Unknown Full policy synchronization algorithm.")
    if updates_per_rollout is None:
        updates_per_rollout = 2 if algorithm == "omni_nft" and os.environ.get("OMNI_VARIANT") == "NFT_FULL32_U2" else updates[algorithm]
    valid = (1, 2) if algorithm == "omni_nft" else (2,)
    if type(updates_per_rollout) is not int or updates_per_rollout not in valid:
        raise ValueError("Unsupported Full optimizer update count per rollout.")
    return {"policy_algorithm": algorithm, "optimizer_updates_per_rollout": updates_per_rollout}


def full_policy_sync_clock(global_steps, *, nft=False, updates_per_rollout=None):
    if type(global_steps) is not int or global_steps < 0:
        raise ValueError("Full policy synchronization requires a nonnegative integer rollout step.")
    tag = full_policy_sync_tag("omni_nft" if nft else "diffusion_art", updates_per_rollout)
    return {**tag, "global_steps": global_steps,
            "policy_version": global_steps * tag["optimizer_updates_per_rollout"]}


def validate_full_policy_sync_clock(metadata, *, nft=False, global_steps=None, updates_per_rollout=None):
    if not isinstance(metadata, dict):
        raise ValueError("Full policy synchronization metadata must be a dictionary.")
    expected = full_policy_sync_clock(
        metadata.get("global_steps") if global_steps is None else global_steps, nft=nft, updates_per_rollout=updates_per_rollout
    )
    for key in ("global_steps", "policy_version"):
        if type(metadata.get(key)) is not int or metadata[key] != expected[key]:
            raise ValueError("Full policy version does not match the algorithm rollout boundary.")
    tags = ("policy_algorithm", "optimizer_updates_per_rollout")
    if any(metadata.get(key) != expected[key] for key in tags) or type(
        metadata.get("optimizer_updates_per_rollout")
    ) is not int:
        raise ValueError("Full policy synchronization algorithm or update clock tag mismatch.")
    return expected
