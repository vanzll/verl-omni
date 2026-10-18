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
"""The restored OmniNFT rollout must expose clean AV latents and named scorer media."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm_omni")

from vllm_omni.diffusion.data import DiffusionOutput  # noqa: E402
from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline  # noqa: E402
from vllm_omni.diffusion.request import OmniDiffusionRequest  # noqa: E402
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch  # noqa: E402
from vllm_omni.inputs.data import OmniDiffusionSamplingParams  # noqa: E402

from verl_omni.pipelines.ltx2_omni_nft.vllm_omni_rollout_adapter import LTX23OmniNFTPipeline  # noqa: E402
from verl_omni.pipelines.model_base import VllmOmniPipelineBase  # noqa: E402


def test_omni_nft_forward_preserves_clean_latents_and_named_media(monkeypatch):
    assert VllmOmniPipelineBase.get_class("LTX2Pipeline", "omni_nft") is LTX23OmniNFTPipeline
    pipeline = object.__new__(LTX23OmniNFTPipeline)
    pipeline.distributed_video_decode = False
    pipeline.vocoder = SimpleNamespace(config=SimpleNamespace(output_sampling_rate=48000))
    pipeline._prepare_request = lambda request: request
    clean_video = torch.randn(1, 4, 8)
    clean_audio = torch.randn(1, 3, 8)
    native_video = torch.randn(1, 4, 2, 4, 6)
    native_audio = torch.randn(1, 2, 3, 4)
    video = torch.zeros(1, 5, 3, 8, 12)
    audio = torch.ones(1, 2, 320)
    pipeline._omni_nft_clean_state = SimpleNamespace(video=clean_video, audio=clean_audio)
    prompt_context = SimpleNamespace(
        **{
            f"{polarity}_connector_{name}": torch.ones(1, 2, 3)
            for polarity in ("positive", "negative")
            for name in ("prompt_embeds", "audio_prompt_embeds", "attention_mask")
        }
    )
    pipeline._omni_nft_forward_context = SimpleNamespace(
        prompt_context=prompt_context,
        original_audio_num_frames=2,
        timesteps=torch.tensor([1000.0, 500.0]),
        request_inputs=SimpleNamespace(frame_rate=24.0),
    )
    pipeline._omni_nft_native_latents = (native_video, native_audio)
    monkeypatch.setattr(LTX2Pipeline, "forward", lambda self, req, **kwargs: DiffusionOutput(output=(video, audio)))
    request = OmniDiffusionRequest(
        request_id="ltx-nft",
        prompt={"prompt_ids": [1, 2]},
        sampling_params=OmniDiffusionSamplingParams(output_type="pt", extra_args={}, fps=24),
    )

    output = pipeline.forward(DiffusionRequestBatch(requests=[request]))
    metadata = output.output["metadata"]
    rl = metadata["rl"]
    torch.testing.assert_close(rl["video_latents_clean"], clean_video)
    torch.testing.assert_close(rl["audio_latents_clean"], clean_audio[:, :2])
    torch.testing.assert_close(rl["train_timesteps"], torch.tensor([[1000.0, 500.0]]))
    specs = metadata["media_artifacts"]["specs"]
    assert set(specs) == {"video_preview", "audio", "video_latent", "audio_latent"}
    assert specs["video_preview"]["fps"] == pytest.approx(24.0)
    assert specs["audio"]["sample_rate"] == 48000
    assert pipeline._omni_nft_clean_state is None
    assert pipeline._omni_nft_forward_context is None
    assert pipeline._omni_nft_native_latents is None
