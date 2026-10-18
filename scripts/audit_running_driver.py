"""Record only actual storage flags after checking the live owner and attempt nonce."""
import json
from pathlib import Path
import socket
import sys
import time

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')
descriptor = Path(sys.argv[1]).resolve()
assert descriptor.is_relative_to(ROOT / 'attempts')
node = json.loads(descriptor.read_text())
assert node['hostname'] == socket.gethostname().split('.')[0]
output = Path(node['output'])
owner = json.loads((output / 'logs' / f'owner_identity_{node["hostname"]}.json').read_text())
assert owner['nonce'] == node['nonce'] and owner['attempt'] == node['attempt']
owner_stat = Path('/proc') / str(owner['pid']) / 'stat'
assert owner_stat.read_text().rsplit(') ', 1)[1].split()[19] == owner['start_ticks']
manifest = json.loads((output / 'pair_manifest.json').read_text())
expected = {'actor_rollout_ref.actor.checkpoint.save_lora_only=true',
            '++trainer.campaign.checkpoint_format=disabled',
            'trainer.save_freq=-1',
            'actor_rollout_ref.actor.fsdp_config.fsdp_size=1',
            'actor_rollout_ref.actor.optim.override_optimizer_config={foreach: false}',
            '++trainer.campaign.checkpoint_storage_policy_sha256=' + manifest['checkpoint_storage_policy_sha256']}
expected |= {f'++reward.models.{name}.dispatch_batch_size=2'
             for name in ('video_align', 'hpsv3', 'audiobox', 'clap', 'desync')}
rows = []
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        argv = proc.joinpath('cmdline').read_bytes().split(b'\0')
        if b'verl_omni.trainer.main_diffusion' not in argv:
            continue
        before = proc.joinpath('stat').read_text().rsplit(') ', 1)[1].split()
        environment = dict(entry.split(b'=', 1) for entry in proc.joinpath('environ').read_bytes().split(b'\0') if b'=' in entry)
        if environment.get(b'OMNI_ATTEMPT') != node['attempt'].encode():
            continue
        assert environment.get(b'OMNI_NONCE') == node['nonce'].encode()
        assert environment.get(b'NCCL_NET') == b'Socket'
        flags = sorted(arg.decode() for arg in argv if arg.decode(errors='replace') in expected)
        assert set(flags) == expected
        assert proc.joinpath('stat').read_text().rsplit(') ', 1)[1].split()[19] == before[19]
        rows.append({'pid': int(proc.name), 'start_ticks': before[19], 'whitelisted_storage_and_reward_transport_flags': flags})
    except (FileNotFoundError, ProcessLookupError):
        continue
assert len(rows) == 1, 'expected one exact live training driver'
assert owner_stat.read_text().rsplit(') ', 1)[1].split()[19] == owner['start_ticks']
evidence = {'observed_unix': time.time(), 'scope': 'Actual inherited identity, Socket network and no-checkpoint/common operational argv; not successful training proof', 'processes': rows}
(output / 'logs/actual_driver_no_checkpoint_runtime_flags.json').write_text(json.dumps(evidence, indent=2) + '\n')
print(json.dumps(evidence))
