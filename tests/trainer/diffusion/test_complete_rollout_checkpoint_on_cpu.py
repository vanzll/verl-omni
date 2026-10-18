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
"""Failure/recovery checks for complete rollout publication without GPU workers."""

import json
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
import wandb

from verl_omni.trainer.diffusion.diffusion_checkpoint import save_complete_rollout, validate_complete_rollout
from verl_omni.trainer.diffusion import diffusion_checkpoint
from verl_omni.trainer.diffusion.ray_diffusion_trainer import BaseRayDiffusionTrainer
from verl_omni.trainer.diffusion import ray_diffusion_trainer


@pytest.fixture
def trainer(tmp_path, monkeypatch):
    monkeypatch.setattr(wandb, "run", None)
    config = OmegaConf.create({
        "trainer": {"default_local_dir": str(tmp_path), "default_hdfs_dir": None,
                    "expected_optimizer_updates_per_rollout": 2, "expected_generated_samples_per_rollout": 256,
                    "campaign": {"source_snapshot_sha256": "source", "train_parquet_sha256": "data"},
                    "project_name": "unit", "experiment_name": "unit", "max_actor_ckpt_to_keep": 2,
                    "resume_mode": "auto", "del_local_ckpt_after_load": False, "nnodes": 1, "n_gpus_per_node": 2},
        "data": {"train_batch_size": 32, "seed": 42},
        "algorithm": {"art_prompt_balanced_split": False},
        "reward": {"aggregation": "weighted_sum"},
        "actor_rollout_ref": {"model": {"lora_rank": 32},
                              "rollout": {"n": 8, "pipeline": {"steps": 20}, "val_kwargs": {"n": 1}},
                              "actor": {"ppo_mini_batch_size": 16, "ppo_epochs": 1,
                                "optim": {"lr": 3e-5}, "checkpoint": {"async_save": False},
                                "diffusion_loss": {"loss_mode": "example"}}},
    })
    cursor = {"value": 32}
    loads = []

    def save_actor(path, remote, step, max_ckpt_to_keep):
        assert max_ckpt_to_keep is None
        path = Path(path)
        path.mkdir()
        for rank in range(2):
            torch.save({"lora.default": torch.tensor([step]), "lora.old": torch.tensor([step - 1])},
                       path / f"model_world_size_2_rank_{rank}.pt")
            torch.save({"step": step}, path / f"optim_world_size_2_rank_{rank}.pt")
            torch.save({"rng": torch.get_rng_state(), "lr_scheduler": {"step": step}},
                       path / f"extra_state_world_size_2_rank_{rank}.pt")

    return SimpleNamespace(config=config, global_steps=10,
        resource_pool_manager=SimpleNamespace(get_n_gpus=lambda: 2),
        actor_rollout_wg=SimpleNamespace(save_checkpoint=save_actor, load_checkpoint=lambda *a, **k: loads.append(a)),
        train_dataloader=SimpleNamespace(state_dict=lambda: dict(cursor), load_state_dict=cursor.update),
        _campaign_training_gpu_seconds=120.0, _campaign_prior_end_to_end_gpu_seconds=60.0,
        _campaign_attempt_startup_gpu_seconds=0.0, _campaign_resume_extra_gpu_seconds=0.0,
        _campaign_fit_start=time.monotonic() - 10, cursor=cursor, loads=loads)


def test_actual_trainer_save_load_restores_rng_cursor_and_cost(trainer):
    random.seed(127)
    np.random.seed(127)
    torch.manual_seed(127)
    BaseRayDiffusionTrainer._save_checkpoint(trainer)
    expected = (random.random(), np.random.rand(), torch.rand(3))
    random.seed(9)
    np.random.seed(9)
    torch.manual_seed(9)
    trainer.cursor["value"] = 999
    trainer.global_steps = 0
    trainer._campaign_training_gpu_seconds = 0
    BaseRayDiffusionTrainer._load_checkpoint(trainer)
    assert trainer.global_steps == 10
    assert trainer.cursor == {"value": 32}
    assert trainer._campaign_training_gpu_seconds == 120
    assert trainer._campaign_prior_end_to_end_gpu_seconds >= 80
    assert trainer.loads
    assert (random.random(), np.random.rand()) == expected[:2]
    torch.testing.assert_close(torch.rand(3), expected[2])
    checkpoint = Path(trainer.config.trainer.default_local_dir) / "global_step_10"
    state = validate_complete_rollout(checkpoint, trainer.config)
    assert (state["optimizer_updates"], state["generated_samples"]) == (20, 2560)


def test_failed_new_save_preserves_last_published_checkpoint(trainer):
    save_complete_rollout(trainer)
    root = Path(trainer.config.trainer.default_local_dir)
    previous = (root / "global_step_10/checkpoint_manifest.json").read_bytes()

    def fail(*args, **kwargs):
        raise OSError("injected shared filesystem write failure")

    trainer.actor_rollout_wg.save_checkpoint = fail
    trainer.global_steps = 20
    with pytest.raises(OSError, match="injected"):
        save_complete_rollout(trainer)
    assert (root / "latest_checkpointed_iteration.txt").read_text() == "10"
    assert not (root / "global_step_20").exists()
    assert (root / "global_step_10/checkpoint_manifest.json").read_bytes() == previous
    assert list(root.glob(".global_step_20.*.incomplete"))


def test_corruption_rejected_before_worker_load(trainer):
    save_complete_rollout(trainer)
    path = Path(trainer.config.trainer.default_local_dir) / "global_step_10/data.pt"
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity mismatch"):
        BaseRayDiffusionTrainer._load_checkpoint(trainer)
    assert not trainer.loads


def test_cross_ablation_resume_rejected(trainer):
    save_complete_rollout(trainer)
    trainer.config.algorithm.art_prompt_balanced_split = True
    with pytest.raises(ValueError, match="identity differs"):
        BaseRayDiffusionTrainer._load_checkpoint(trainer)
    assert not trainer.loads


def test_rotation_happens_after_complete_publication(trainer):
    root = Path(trainer.config.trainer.default_local_dir)
    for step in [10, 20, 30]:
        trainer.global_steps = step
        save_complete_rollout(trainer)
    assert not (root / "global_step_10").exists()
    assert (root / "global_step_20").is_dir()
    assert (root / "global_step_30").is_dir()
    assert (root / "latest_checkpointed_iteration.txt").read_text() == "30"
    assert json.loads((root / "global_step_30/checkpoint_manifest.json").read_text())["world_size"] == 2


def test_missing_rank_cannot_be_published(trainer):
    original = trainer.actor_rollout_wg.save_checkpoint

    def missing_rank(*args, **kwargs):
        original(*args, **kwargs)
        (Path(args[0]) / "extra_state_world_size_2_rank_1.pt").unlink()

    trainer.actor_rollout_wg.save_checkpoint = missing_rank
    with pytest.raises(RuntimeError, match="Missing or empty checkpoint shard"):
        save_complete_rollout(trainer)
    assert not (Path(trainer.config.trainer.default_local_dir) / "latest_checkpointed_iteration.txt").exists()


def test_resume_recovers_directory_published_before_tracker_crash(trainer, monkeypatch):
    save_complete_rollout(trainer)
    original = diffusion_checkpoint._atomic_text
    trainer.global_steps = 20

    def fail_tracker(path, text):
        if path.name == "latest_checkpointed_iteration.txt":
            raise OSError("injected tracker crash")
        original(path, text)

    monkeypatch.setattr(diffusion_checkpoint, "_atomic_text", fail_tracker)
    with pytest.raises(OSError, match="tracker crash"):
        save_complete_rollout(trainer)
    root = Path(trainer.config.trainer.default_local_dir)
    assert (root / "latest_checkpointed_iteration.txt").read_text() == "10"
    BaseRayDiffusionTrainer._load_checkpoint(trainer)
    assert trainer.global_steps == 20


def test_optimizer_protocol_change_rejected(trainer):
    save_complete_rollout(trainer)
    trainer.config.actor_rollout_ref.actor.optim.lr = 1e-4
    with pytest.raises(ValueError, match="identity differs"):
        BaseRayDiffusionTrainer._load_checkpoint(trainer)


def test_missing_manifest_rank_rejected_before_loading(trainer):
    save_complete_rollout(trainer)
    path = Path(trainer.config.trainer.default_local_dir) / "global_step_10/checkpoint_manifest.json"
    manifest = json.loads(path.read_text())
    del manifest["files"]["actor/model_world_size_2_rank_1.pt"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="omits required"):
        BaseRayDiffusionTrainer._load_checkpoint(trainer)
    assert not trainer.loads


def test_real_stateful_dataloader_resumes_same_next_prompt_batch(trainer):
    from torchdata.stateful_dataloader import StatefulDataLoader

    def make_loader():
        return StatefulDataLoader(list(range(256)), batch_size=32, shuffle=True,
                                  generator=torch.Generator().manual_seed(42), num_workers=0)

    trainer.train_dataloader = make_loader()
    batches = iter(trainer.train_dataloader)
    for _ in range(3):
        next(batches)
    BaseRayDiffusionTrainer._save_checkpoint(trainer)
    expected = [next(batches), next(batches)]
    trainer.train_dataloader = make_loader()
    BaseRayDiffusionTrainer._load_checkpoint(trainer)
    resumed = iter(trainer.train_dataloader)
    for next_prompts in expected:
        torch.testing.assert_close(next(resumed), next_prompts)


def test_clock_includes_gpu_initialization_before_fit(trainer, monkeypatch):
    trainer.config.trainer.gpu_reservation_start_unix_mean = 950.0
    fake_clock = SimpleNamespace(time=lambda: 1000.0, monotonic=lambda: 500.0)
    monkeypatch.setattr(ray_diffusion_trainer, "time", fake_clock)
    BaseRayDiffusionTrainer._start_campaign_clock(trainer)
    assert trainer._campaign_attempt_startup_gpu_seconds == 100.0
    fake_clock.monotonic = lambda: 520.0
    metrics = {"actor/optimizer_updates": 2}
    BaseRayDiffusionTrainer._record_campaign_progress(trainer, metrics, {"step": 5.0}, 2, 256)
    assert metrics["cost/training_gpu_hours"] == pytest.approx(10.0 / 3600)
    assert metrics["cost/end_to_end_gpu_hours"] == pytest.approx(140.0 / 3600)


def test_resume_adds_new_startup_and_recorded_failed_attempt_cost(trainer):
    save_complete_rollout(trainer)
    trainer._campaign_attempt_startup_gpu_seconds = 17.0
    trainer._campaign_resume_extra_gpu_seconds = 23.0
    path = Path(trainer.config.trainer.default_local_dir) / "global_step_10"
    saved = validate_complete_rollout(path, trainer.config)
    BaseRayDiffusionTrainer._load_checkpoint(trainer)
    assert trainer._campaign_prior_end_to_end_gpu_seconds == pytest.approx(saved["end_to_end_gpu_seconds"] + 40.0)
