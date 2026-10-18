import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from verl import DataProto

from test_art_joint_latent_on_cpu import make_batch, make_config
from verl_omni.trainer.diffusion.diffusion_algos import DiffusionARTLoss


ROOT = Path(__file__).resolve().parents[3]


def test_full_cache_version_survives_balanced_order_and_rejects_mixed_versions():
    config = make_config(art_first_update_mask_exempt=True, art_prompt_group_std=True)
    config.actor_rollout_ref.model.full_finetune = True
    batch = make_batch()
    with pytest.raises(ValueError, match="native committed"):
        DiffusionARTLoss.prepare_actor_batch(batch, torch.zeros(256), config)
    batch = make_batch()
    batch.batch["art_policy_version"] = torch.zeros(256, dtype=torch.int64)
    batch.batch["art_policy_version"][1] = 2
    with pytest.raises(ValueError, match="different policies"):
        DiffusionARTLoss.prepare_actor_batch(batch, torch.zeros(256), config)
    batch = make_batch()
    batch.batch["art_policy_version"] = torch.full((256,), 4, dtype=torch.int64)
    result = DiffusionARTLoss.prepare_actor_batch(batch, torch.zeros(256), config)
    assert result.batch["art_policy_version"].tolist() == [4] * 256
    for rank in range(32):
        shard = result.batch[rank * 8:(rank + 1) * 8]
        assert shard["art_update_window"].tolist() == [0] * 4 + [1] * 4


def test_eval_dispatches_exactly_eight_real_cases_without_padding():
    path = ROOT / "verl_omni/agent_loop/full32_manager.py"
    tree = ast.parse(path.read_text())
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef))
    method.decorator_list = []
    namespace = {"asyncio": asyncio, "np": np, "DataProto": DataProto}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    observed = []

    async def generate(batch):
        observed.extend(batch.batch["sample_id"].tolist())
        batch.meta_info = {"metrics": [{"finite": True}] * len(batch)}
        return batch

    workers = [SimpleNamespace(generate_sequences=SimpleNamespace(remote=generate)) for _ in range(16)]
    manager = SimpleNamespace(model_config=SimpleNamespace(full_finetune=True), agent_loop_workers=workers,
                              _performance_metrics=lambda metrics, output: {"cases": sum(map(len, metrics))})
    prompts = DataProto.from_dict(tensors={"sample_id": torch.arange(8)}, meta_info={"validate": True})
    output = asyncio.run(namespace[method.name](manager, prompts))
    assert observed == list(range(8))
    assert output.batch["sample_id"].tolist() == list(range(8))
    assert output.meta_info["timing"]["cases"] == 8
