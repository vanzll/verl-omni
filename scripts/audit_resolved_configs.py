"""Validate the actual Hydra output and compare scientific ablation settings."""
import hashlib
import importlib
import inspect
import json
from pathlib import Path

from omegaconf import OmegaConf
from verl_omni.trainer.diffusion.diffusion_algos import get_diffusion_loss_fn

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')


def differences(a, b, path=''):
    if isinstance(a, dict) and isinstance(b, dict):
        result = []
        for key in sorted(a.keys() | b.keys()):
            result.extend(differences(a.get(key), b.get(key), f'{path}.{key}'.lstrip('.')))
        return result
    return [] if a == b else [{'path': path, 'from': a, 'to': b}]


def require(config, path, expected):
    actual = OmegaConf.select(config, path)
    if actual != expected:
        raise AssertionError(f'{path}: actual={actual!r}, expected={expected!r}')


configs, report = {}, {'variants': {}}
for variant in ['N0', 'A0', 'A1', 'A2']:
    raw = ROOT / 'logs' / ('config_N0_fps_fixed.raw' if variant == 'N0' else f'config_{variant}_resolved.raw')
    text = raw.read_text()
    # Keep original stdout, including import warnings, as evidence. Only strip the
    # prefix before the known complete root config; verify all root sections.
    text = text[text.index('model_engine: dp_diffusion\n'):]
    cfg = OmegaConf.create(text)
    assert {'model_engine', 'actor_rollout_ref', 'algorithm', 'data', 'reward', 'trainer', 'ray_kwargs'} <= set(cfg)
    expected = {
        'data.train_batch_size': 32, 'data.seed': 42, 'data.val_max_samples': 8,
        'data.val_batch_size': 8, 'data.validation_shuffle': False,
        'actor_rollout_ref.actor.ppo_mini_batch_size': 16,
        'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu': 2,
        'actor_rollout_ref.actor.ppo_epochs': 1, 'actor_rollout_ref.actor.shuffle': False,
        'actor_rollout_ref.actor.fsdp_config.fsdp_size': 1,
        'actor_rollout_ref.actor.optim.override_optimizer_config.foreach': False,
        'actor_rollout_ref.actor.optim.lr': 3e-5, 'actor_rollout_ref.actor.optim.weight_decay': 1e-4,
        'actor_rollout_ref.model.lora_rank': 32, 'actor_rollout_ref.model.lora_alpha': 64,
        'actor_rollout_ref.rollout.n': 8, 'actor_rollout_ref.rollout.tensor_model_parallel_size': 2,
        'trainer.nnodes': 2, 'trainer.n_gpus_per_node': 8, 'trainer.total_training_steps': 75,
        'trainer.test_freq': 10, 'trainer.save_freq': -1, 'trainer.val_before_train': True,
        'trainer.validation_data_max_samples': 8,
        'trainer.expected_optimizer_updates_per_rollout': 2,
        'trainer.expected_generated_samples_per_rollout': 256,
        'actor_rollout_ref.actor.checkpoint.save_lora_only': True,
        'actor_rollout_ref.actor.checkpoint.async_save': False,
        'trainer.campaign.checkpoint_format': 'disabled',
        'trainer.campaign.checkpoint_storage_policy_sha256': hashlib.sha256(
            (ROOT / 'assets/compact_checkpoint_storage_policy.json').read_bytes()).hexdigest(),
        'algorithm.art_prompt_balanced_split': variant == 'A1',
        'algorithm.art_joint_reward_routing': variant == 'A2',
        'algorithm.old_policy_decay_schedule': 'linear_to_0_5' if variant == 'N0' else 'copy',
        'reward.reward_functions.video_align.fps': 24.0,
        'reward.reward_functions.desync.fps': 24.0,
    }
    expected.update({f'reward.models.{name}.dispatch_batch_size': 2
                     for name in ('video_align', 'hpsv3', 'audiobox', 'clap', 'desync')})
    for key, value in expected.items():
        require(cfg, key, value)
    for prefix, h, w, steps, vcfg, acfg, modality, rescale in [
        ('actor_rollout_ref.rollout.pipeline', 256, 384, 20, 1.5, 3.0, 1.0, 0.0),
        ('actor_rollout_ref.rollout.val_kwargs.pipeline', 512, 768, 40, 3.0, 7.0, 3.0, 0.7),
    ]:
        values = {'height': h, 'width': w, 'num_inference_steps': steps, 'num_frames': 121,
                  'frame_rate': 24.0, 'video_cfg_scale': vcfg, 'audio_cfg_scale': acfg,
                  'video_modality_scale': modality, 'audio_modality_scale': modality,
                  'video_rescale_scale': rescale, 'audio_rescale_scale': rescale}
        for key, value in values.items():
            require(cfg, f'{prefix}.{key}', value)
    get_diffusion_loss_fn(cfg.actor_rollout_ref.actor.diffusion_loss.loss_mode).validate_trainer_config(cfg)
    for name, entry in cfg.reward.reward_functions.items():
        module = importlib.import_module(entry.path.removeprefix('pkg://'))
        fn = getattr(module, entry.name)
        for name_, parameter in inspect.signature(fn).parameters.items():
            if parameter.kind == inspect.Parameter.KEYWORD_ONLY and parameter.default is inspect.Parameter.empty:
                assert name_ == 'reward_model' or name_ in entry, (variant, name, name_)
    model_file = Path(cfg.actor_rollout_ref.model.path)
    assert cfg.trainer.campaign.model_revision == model_file.name
    for kind, key in [('train', 'train_files'), ('val', 'val_files')]:
        digest = hashlib.sha256(Path(cfg.data[key]).read_bytes()).hexdigest()
        assert digest == cfg.trainer.campaign[f'{kind}_parquet_sha256']
    out = ROOT / 'preflight' / 'configs' / variant / 'resolved.yaml'
    out.write_text(OmegaConf.to_yaml(cfg, resolve=True))
    configs[variant] = OmegaConf.to_container(cfg, resolve=True)
    report['variants'][variant] = {'resolved_sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
                                  'loss_validator': 'passed', 'required_reward_arguments': 'passed',
                                  'protocol_assertions': 'passed'}

identity_paths = {'algorithm.art_sample_log_dir', 'ray_kwargs.ray_init.runtime_env.env_vars.WANDB_DIR',
                  'trainer.default_local_dir', 'trainer.experiment_name', 'trainer.rollout_data_dir',
                  'trainer.validation_data_dir'}
for variant, flag in [('A1', 'art_prompt_balanced_split'), ('A2', 'art_joint_reward_routing')]:
    diff = differences(configs['A0'], configs[variant])
    scientific = [d for d in diff if d['path'] not in identity_paths]
    assert scientific == [{'path': f'algorithm.{flag}', 'from': False, 'to': True}], scientific
    report['variants'][variant]['all_differences_from_A0'] = diff
report['N0_differences_from_A0'] = differences(configs['A0'], configs['N0'])
report['status'] = 'passed'
report['native_reward_dispatch_batch_size'] = 2
report['checkpoint_storage_policy_sha256'] = hashlib.sha256(
    (ROOT / 'assets/compact_checkpoint_storage_policy.json').read_bytes()).hexdigest()
(ROOT / 'logs' / 'resolved_config_audit.json').write_text(json.dumps(report, indent=2) + '\n')
print('PASS: four resolved configs, protocol, loss validators, required scorer arguments, data hashes, single-switch ablations')
