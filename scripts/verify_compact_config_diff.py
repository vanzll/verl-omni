"""Verify the storage migration changes no scientific resolved config fields."""
import hashlib
import json
from pathlib import Path
import time

from omegaconf import OmegaConf

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')
operational = {'actor_rollout_ref.actor.fsdp_config.fsdp_size',
               'actor_rollout_ref.actor.optim.override_optimizer_config'}
allowed = operational | {'trainer.save_freq', 'actor_rollout_ref.actor.checkpoint.save_lora_only',
           'trainer.campaign.checkpoint_format', 'trainer.campaign.checkpoint_storage_policy_sha256',
           'trainer.campaign.source_snapshot_sha256', 'trainer.campaign.source_files_manifest_sha256',
           'trainer.campaign.source_checkout_dir', 'trainer.launch_files_manifest_sha256',
           'ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH', 'algorithm.art_sample_log_dir',
           'trainer.default_local_dir', 'trainer.experiment_name', 'trainer.rollout_data_dir',
           'trainer.validation_data_dir', 'ray_kwargs.ray_init.runtime_env.env_vars.WANDB_DIR'}
transport_fields = {f'reward.models.{name}.dispatch_batch_size'
                    for name in ('video_align', 'hpsv3', 'audiobox', 'clap', 'desync')}
allowed |= transport_fields


def difference(a, b, path=''):
    if isinstance(a, dict) and isinstance(b, dict):
        result = []
        for key in sorted(a.keys() | b.keys()):
            result += difference(a.get(key), b.get(key), f'{path}.{key}'.lstrip('.'))
        return result
    return [] if a == b else [{'path': path, 'from': a, 'to': b}]


audit = json.loads((ROOT / 'logs/resolved_config_audit.json').read_text())
assert audit['status'] == 'passed'
report = {'status': 'passed', 'observed_unix': time.time(), 'variants': {},
          'native_reward_dispatch_batch_size': 2,
          'checkpoint_storage_policy_sha256': audit['checkpoint_storage_policy_sha256'],
          'scope': 'all-field comparison; only storage, identity/source provenance and native reward RPC sample batching may differ'}
for variant in ('N0', 'A0', 'A1', 'A2'):
    old_path = ROOT / 'logs/config_before_compact_checkpoint' / variant / 'resolved.yaml'
    new_path = ROOT / 'preflight/configs' / variant / 'resolved.yaml'
    digest = hashlib.sha256(new_path.read_bytes()).hexdigest()
    assert digest == audit['variants'][variant]['resolved_sha256']
    old, new = [OmegaConf.to_container(OmegaConf.load(p), resolve=True) for p in (old_path, new_path)]
    changes = difference(old, new)
    assert all(OmegaConf.select(OmegaConf.load(new_path), field) == 2 for field in transport_fields)
    prior_transport_path = ROOT / 'logs/config_before_reward_dispatch2' / variant / 'resolved.yaml'
    prior_transport = OmegaConf.to_container(OmegaConf.load(prior_transport_path), resolve=True)
    transport_changes = difference(prior_transport, new)
    transport_identity = {'algorithm.art_sample_log_dir', 'trainer.default_local_dir',
                          'trainer.experiment_name', 'trainer.rollout_data_dir', 'trainer.validation_data_dir',
                          'ray_kwargs.ray_init.runtime_env.env_vars.WANDB_DIR'}
    assert not [d for d in transport_changes if d['path'] not in transport_fields | transport_identity | operational | {'trainer.save_freq', 'trainer.campaign.checkpoint_format', 'trainer.campaign.checkpoint_storage_policy_sha256'}]
    actual_transport = [d for d in transport_changes if d['path'] in transport_fields]
    assert {d['path'] for d in actual_transport} == transport_fields
    assert all(d['from'] is None and d['to'] == 2 for d in actual_transport)
    assert new['actor_rollout_ref']['actor']['fsdp_config']['fsdp_size'] == 1
    assert new['actor_rollout_ref']['actor']['optim']['override_optimizer_config'] == {'foreach': False}
    assert new['trainer']['save_freq'] == -1
    assert new['trainer']['campaign']['checkpoint_format'] == 'disabled'
    unexpected = [d for d in changes if d['path'] not in allowed]
    assert not unexpected, (variant, unexpected)
    assert any(d['path'] == 'actor_rollout_ref.actor.checkpoint.save_lora_only'
               and d['from'] is False and d['to'] is True for d in changes)
    report['variants'][variant] = {'all_differences': changes, 'scientific_recipe_unchanged': True,
                                   'changes_since_before_reward_dispatch2': transport_changes,
                                   'resolved_sha256': digest}
output = ROOT / 'logs/compact_checkpoint_semantic_config_diff.json'
output.write_text(json.dumps(report, indent=2) + '\n')
print('PASS: all scientific settings unchanged; native reward RPC sample batching changes None to2 for all five models')
