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
"""Atomic complete-rollout checkpoints for runs with an explicit update budget."""

import hashlib
import json
import os
import random
import re
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


def checkpoint_identity(config):
    """Keep code, data and loss identity separate from attempt output paths."""
    algorithm = OmegaConf.to_container(config.algorithm, resolve=True)
    algorithm.pop("art_sample_log_dir", None)
    return {
        "provenance": OmegaConf.to_container(config.trainer.campaign, resolve=True),
        "algorithm": algorithm,
        "loss": OmegaConf.to_container(config.actor_rollout_ref.actor.diffusion_loss, resolve=True),
        "optimizer": OmegaConf.to_container(config.actor_rollout_ref.actor.optim, resolve=True),
        "model": OmegaConf.to_container(config.actor_rollout_ref.model, resolve=True),
        "rollout": OmegaConf.to_container(config.actor_rollout_ref.rollout.pipeline, resolve=True),
        "validation": OmegaConf.to_container(config.actor_rollout_ref.rollout.val_kwargs, resolve=True),
        "reward": OmegaConf.to_container(config.reward, resolve=True),
        "data_seed": config.data.seed,
        "batch": [config.data.train_batch_size, config.actor_rollout_ref.rollout.n,
                  config.actor_rollout_ref.actor.ppo_mini_batch_size, config.actor_rollout_ref.actor.ppo_epochs],
    }


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path, text):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_complete_rollout(trainer):
    """Publish only after all worker shards and controller state are durable."""
    config = trainer.config
    if config.trainer.default_hdfs_dir is not None:
        raise ValueError("Atomic rollout checkpoints require a shared local filesystem")
    if config.actor_rollout_ref.actor.checkpoint.get("async_save", False):
        raise ValueError("Atomic rollout checkpoints require synchronous worker saves")
    root = Path(config.trainer.default_local_dir)
    root.mkdir(parents=True, exist_ok=True)
    step = trainer.global_steps
    destination = root / f"global_step_{step}"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite published checkpoint {destination}")
    staging = root / f".global_step_{step}.{uuid.uuid4().hex}.incomplete"
    staging.mkdir()
    # The controller owns rotation after publication. Worker-side rotation would
    # reference staging paths after the directory rename and could prune too soon.
    world = trainer.resource_pool_manager.get_n_gpus()
    expected_world = config.trainer.nnodes * config.trainer.n_gpus_per_node
    if world != expected_world:
        raise ValueError(f"Checkpoint allocation has {world} GPUs, configured {expected_world}")
    trainer.actor_rollout_wg.save_checkpoint(str(staging / "actor"), None, step, max_ckpt_to_keep=None)
    torch.save(trainer.train_dataloader.state_dict(), staging / "data.pt")
    for rank in range(world):
        for kind in ("model", "optim", "extra_state"):
            path = staging / "actor" / f"{kind}_world_size_{world}_rank_{rank}.pt"
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Missing or empty checkpoint shard: {path}")
    state = {
        "schema": 1, "rollout": step,
        "optimizer_updates": step * int(config.trainer.expected_optimizer_updates_per_rollout),
        "generated_samples": step * int(config.trainer.expected_generated_samples_per_rollout),
        "identity": checkpoint_identity(config),
        "training_gpu_seconds": trainer._campaign_training_gpu_seconds,
        "end_to_end_gpu_seconds": trainer._campaign_prior_end_to_end_gpu_seconds
        + world * (time.monotonic() - trainer._campaign_fit_start),
        "wandb": {"project": config.trainer.project_name, "name": config.trainer.experiment_name,
                  "run_id": os.environ.get("WANDB_RUN_ID")},
    }
    import wandb

    if wandb.run is not None:
        state["wandb"].update(run_id=wandb.run.id, url=wandb.run.url)
    files = {}
    for path in sorted(staging.rglob("*")):
        if path.is_file():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
            files[str(path.relative_to(staging))] = {"size": path.stat().st_size, "sha256": _digest(path)}
    state["end_to_end_gpu_seconds"] = trainer._campaign_prior_end_to_end_gpu_seconds + world * (
        time.monotonic() - trainer._campaign_fit_start
    )
    state["controller_rng"] = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                               "random": random.getstate()}
    controller_path = staging / "controller_state.pt"
    torch.save(state, controller_path)
    with controller_path.open("rb") as stream:
        os.fsync(stream.fileno())
    files[controller_path.name] = {"size": controller_path.stat().st_size, "sha256": _digest(controller_path)}
    manifest = {"schema": 1, "rollout": step, "world_size": world, "files": files}
    _atomic_text(staging / "checkpoint_manifest.json", json.dumps(manifest, indent=2) + "\n")
    for directory in sorted((path for path in staging.rglob("*") if path.is_dir()),
                            key=lambda path: len(path.parts), reverse=True):
        _sync_directory(directory)
    _sync_directory(staging)
    staging.rename(destination)
    _sync_directory(root)
    _atomic_text(root / "latest_checkpointed_iteration.txt", str(step))
    _sync_directory(root)
    keep = config.trainer.get("max_actor_ckpt_to_keep", None)
    if keep is not None and int(keep) > 0:
        published = sorted((p for p in root.iterdir() if p.is_dir()
                            and re.fullmatch(r"global_step_[0-9]+", p.name)
                            and (p / "checkpoint_manifest.json").is_file()),
                           key=lambda p: int(p.name.removeprefix("global_step_")))
        for old in published[:-int(keep)]:
            shutil.rmtree(old)


def validate_complete_rollout(path, config):
    """Check every published file before asking GPU workers to load any shards."""
    path = Path(path)
    manifest = json.loads((path / "checkpoint_manifest.json").read_text())
    if manifest["schema"] != 1 or manifest["rollout"] != int(path.name.removeprefix("global_step_")):
        raise ValueError("Invalid complete-rollout checkpoint identity")
    world = config.trainer.nnodes * config.trainer.n_gpus_per_node
    if manifest["world_size"] != world:
        raise ValueError("Checkpoint world size differs from this run")
    required = {"data.pt", "controller_state.pt"} | {
        f"actor/{kind}_world_size_{world}_rank_{rank}.pt"
        for rank in range(world) for kind in ("model", "optim", "extra_state")
    }
    if not required.issubset(manifest["files"]):
        raise ValueError("Checkpoint manifest omits required complete-state files")
    for name, record in manifest["files"].items():
        file = path / name
        if not file.resolve().is_relative_to(path.resolve()):
            raise ValueError("Checkpoint manifest path escapes checkpoint root")
        if file.stat().st_size != record["size"] or _digest(file) != record["sha256"]:
            raise ValueError(f"Checkpoint file integrity mismatch: {file}")
    state = torch.load(path / "controller_state.pt", map_location="cpu", weights_only=False)
    if state["schema"] != 1 or state["identity"] != checkpoint_identity(config) or state["rollout"] != manifest["rollout"]:
        raise ValueError("Checkpoint code/data/algorithm identity differs from this run")
    if state["optimizer_updates"] != state["rollout"] * config.trainer.expected_optimizer_updates_per_rollout:
        raise ValueError("Checkpoint update budget mismatch")
    if state["generated_samples"] != state["rollout"] * config.trainer.expected_generated_samples_per_rollout:
        raise ValueError("Checkpoint generated-sample budget mismatch")
    return state


def latest_complete_rollout(root):
    """Find a published directory even if a crash preceded the tracker rename."""
    root = Path(root)
    if not root.is_dir():
        return None
    published = [path for path in root.iterdir() if path.is_dir()
                 and re.fullmatch(r"global_step_[0-9]+", path.name)
                 and (path / "checkpoint_manifest.json").is_file()]
    return str(max(published, key=lambda path: int(path.name.removeprefix("global_step_")))) if published else None


def restore_controller_state(trainer, state):
    trainer._campaign_training_gpu_seconds = state["training_gpu_seconds"]
    trainer._campaign_prior_end_to_end_gpu_seconds = (
        state["end_to_end_gpu_seconds"] + trainer._campaign_attempt_startup_gpu_seconds
        + trainer._campaign_resume_extra_gpu_seconds
    )
    trainer._campaign_parent_wandb = state["wandb"]
    rng = state["controller_rng"]
    torch.set_rng_state(rng["torch"])
    np.random.set_state(rng["numpy"])
    random.setstate(rng["random"])
    import wandb

    if wandb.run is not None:
        wandb.config.update({"checkpoint_resume": {"parent_wandb": state["wandb"],
                            "rollout": state["rollout"], "optimizer_updates": state["optimizer_updates"]}},
                            allow_val_change=True)
