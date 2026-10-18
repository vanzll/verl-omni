"""Private GPU probe pipeline; production algorithms never import this module."""
import os
from pathlib import Path

import torch

from ltx_velocity_capture import LTXVelocityCapture
from verl_omni.pipelines.ltx2_diffusion_art.vllm_omni_rollout_adapter import LTX23DiffusionARTPipeline


class LTXProbePipeline(LTX23DiffusionARTPipeline):
    @torch.no_grad()
    def forward(self, req, **kwargs):
        sampling = req.sampling_params
        if not (sampling.extra_args or {}).get('_isolated_velocity_capture', False):
            return super().forward(req, **kwargs)
        assert int(sampling.num_inference_steps) == 20
        seed = int(sampling.seed)
        identity = {
            'source_sha256': os.environ['SOURCE_SNAPSHOT_SHA256'],
            'model_revision': '8eee8edcf067e838b843f926ec4d4cc9b2be1aaf',
            'adapter_fingerprint': 'base-model-no-loaded-LoRA; initial-zero-LoRA equivalence not yet checked',
            'seed_identity': f'isolated-actual-native-seed-{seed}',
            'attention_backend': os.environ['OMNI_PROBE_ATTENTION_BACKEND'],
            'video_cfg': sampling.extra_args['video_cfg_scale'],
            'audio_cfg': sampling.extra_args['audio_cfg_scale'],
        }
        with LTXVelocityCapture(self, identity, (0, 9, 18)) as capture:
            result = super().forward(req, **kwargs)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        target = Path(os.environ['OUTPUT_DIR']) / 'captures' / f'seed_{seed}_rank_{rank}.pt'
        capture.publish(target)
        return result
