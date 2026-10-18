"""A restored OmniNFT actor must use the diffusion initialization contract."""

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl_omni.workers import engine_workers as workers


@pytest.mark.parametrize("model_type", ["omni_nft_model", "diffusion_art_model", "diffusion_nft_model"])
def test_diffusion_actor_dispatch_accepts_null_logprob_microbatch(monkeypatch, model_type):
    # This is the configuration that failed on real distributed N0 startup.
    # Stop at engine construction, before any distributed initialization or GPU use.
    config = OmegaConf.create({
        "model": {"model_type": model_type},
        "actor": {"ppo_micro_batch_size_per_gpu": 2},
        "rollout": {"log_prob_micro_batch_size_per_gpu": None},
    })
    model = config.model
    actor = SimpleNamespace(engine=SimpleNamespace(), optim=None, checkpoint=None)

    # DictConfig attribute lookup may return a new Python wrapper, so use the
    # semantic node rather than its identity when substituting the CPU fixtures.
    def instantiate(value):
        return model if "model_type" in value else actor

    monkeypatch.setattr(workers, "omega_conf_to_dataclass", instantiate)
    worker = object.__new__(workers.ActorRolloutRefWorker)
    worker.config = config
    worker.role = "actor"
    worker.distillation_enabled = False
    captured = []

    class BeforeGPU(Exception):
        pass

    def build_engine(*, config):
        captured.append(config)
        raise BeforeGPU

    monkeypatch.setattr(workers, "TrainingWorker", build_engine)
    with pytest.raises(BeforeGPU):
        worker.init_model()
    assert captured[0].model_type == model_type
    assert captured[0].engine_config.micro_batch_size_per_gpu == 2
    assert captured[0].engine_config.infer_micro_batch_size_per_gpu == 2
    assert worker.loss_fn.func is workers.diffusion_loss
