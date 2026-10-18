"""Bounded transparency check using native ART stepping and the real scheduler.

Predictions are synthetic CPU inputs. Passing is not a full-model GPU result.
"""
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch

from ltx_velocity_capture import LTXVelocityCapture, ROOT
from verl_omni.pipelines.ltx2_diffusion_art.vllm_omni_rollout_adapter import LTX23DiffusionARTPipeline
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from vllm_omni.diffusion.models.ltx2.ltx2_latents import LTXAVState


def make_pipeline():
    p = object.__new__(LTX23DiffusionARTPipeline)
    # Skip heavyweight model construction but initialize the real Module base;
    # method restoration exercises nn.Module.__delattr__ as on a real pipeline.
    torch.nn.Module.__init__(p)
    p._flow_grpo_noise_level = 0.0
    p._flow_grpo_sde_type = 'sde'
    p._flow_grpo_logprobs = False
    p._selected_sde_steps = {0, 1, 2}
    p._current_latents, p._next_latents, p._selected_timesteps, p._log_probs = [], [], [], []
    p._flow_grpo_condition_image_latents = None
    p.scheduler = FlowMatchSDEDiscreteScheduler(num_train_timesteps=1000)
    p.scheduler.set_timesteps(3, device='cpu')
    p._predict_noise_for_step = lambda index, timestep, state, *contexts: (
        state.video.float() * 0.25 + 0.137, state.audio.float() * 0.31 - 0.223)
    p._synchronize_guidance_parallel_step_output = lambda value, **kwargs: value
    return p


def make_context():
    positive = torch.arange(2 * 5 * 4).reshape(2, 5, 4).float() / 10
    prompts = SimpleNamespace(positive_connector_prompt_embeds=positive,
                              positive_connector_audio_prompt_embeds=positive + 1,
                              negative_connector_prompt_embeds=positive - 1,
                              negative_connector_audio_prompt_embeds=positive - 2)
    return SimpleNamespace(request_inputs=SimpleNamespace(generator=torch.Generator().manual_seed(42)),
                           guidance_parallel_ready=False, sampler='euler',
                           original_audio_num_frames=4, latent_num_frames=2, prompt_context=prompts)


def run(p, context):
    x = torch.arange(2 * 10 * 8).reshape(2, 10, 8).float() / 37 - 1
    state = LTXAVState(video=x[:, :6].to(torch.bfloat16), audio=x[:, 6:].to(torch.bfloat16))
    denoise = SimpleNamespace(latents=None, audio_latents=None, conditioning_mask=None)
    for i, timestep in enumerate(p.scheduler.timesteps):
        state = p._denoise_step(i, timestep, state, context, denoise)
    return state


identity = {'source_sha256': os.environ['SOURCE_SNAPSHOT_SHA256'], 'model_revision': 'synthetic-CPU-fixture',
            'adapter_fingerprint': 'no-model-synthetic-prediction', 'seed_identity': 'cpu-fixture-42',
            'attention_backend': 'none-CPU-fixture', 'video_cfg': 1.5, 'audio_cfg': 3.0}
p = make_pipeline()
try:
    LTXVelocityCapture(p, identity)
    raise AssertionError('Probe isolation guard failed')
except RuntimeError:
    pass
os.environ['OMNI_ISOLATED_VELOCITY_PROBE'] = '1'
baseline_context = make_context()
torch.manual_seed(71)
initial_rng = torch.get_rng_state()
baseline = run(p, baseline_context)
baseline_rng = torch.get_rng_state()
baseline_request_rng = baseline_context.request_inputs.generator.get_state()
probe = make_pipeline()
probe_context = make_context()
torch.set_rng_state(initial_rng)
with LTXVelocityCapture(probe, identity, (0, 1, 2)) as capture:
    captured = run(probe, probe_context)
assert torch.equal(captured.video, baseline.video) and torch.equal(captured.audio, baseline.audio)
assert torch.equal(torch.get_rng_state(), baseline_rng)
assert torch.equal(probe_context.request_inputs.generator.get_state(), baseline_request_rng)
assert '_denoise_step' not in vars(probe) and 'step' not in vars(probe.scheduler)
rounding_gap = []
for row in capture.records:
    euler = row['state'] + (row['sigma_next'] - row['sigma']) * row['actual_sampler_velocity']
    torch.testing.assert_close(row['scheduler_next_unrounded'], euler, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(row['actual_next_rounded'], euler.to(torch.bfloat16).float(), rtol=0, atol=0)
    rounding_gap.append(float((row['actual_next_rounded'] - euler).square().mean().sqrt()))
assert max(rounding_gap) > 0, 'Fixture failed to expose native BF16 rounding'
destination = ROOT / 'logs/velocity_capture_cpu_fixture.pt'
if destination.exists():
    raise FileExistsError('Do not overwrite prior diagnostic evidence')
metadata = capture.publish(destination)
loaded = torch.load(destination, weights_only=True)
assert len(loaded['records']) == 3 and loaded['identity'] == identity

# Prove scheduler interception is undone even if native prediction fails.
broken = make_pipeline()
def fail(*args, **kwargs):
    raise RuntimeError('deliberate CPU fixture failure')
broken._predict_noise_for_step = fail
try:
    with LTXVelocityCapture(broken, identity, (0,)):
        run(broken, make_context())
    raise AssertionError('Expected fixture exception')
except RuntimeError as error:
    assert str(error) == 'deliberate CPU fixture failure'
assert '_denoise_step' not in vars(broken) and 'step' not in vars(broken.scheduler)
assert not hasattr(broken, '_isolated_velocity_capture_active')
report = {'status': 'passed', 'scope': 'synthetic CPU predictions, native ART denoise and real scheduler',
          'output_and_rng_bitwise_equal': True, 'exception_restores_methods': True,
          'rounded_vs_unrounded_rms': rounding_gap, 'artifact': str(destination), 'metadata': metadata,
          'gpu_model_cache_equivalence': 'not tested', 'parameter_gradients': 'not tested'}
(ROOT / 'logs/velocity_capture_cpu_result.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report), flush=True)
