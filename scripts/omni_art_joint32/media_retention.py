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

import json
import math
from pathlib import Path
import re
import time


RUNTIME_ROOTS = (
    Path('/m2v_intern/zhenglin/omni_art_ablation_20261016/extensions/nft_full32'),
    Path('/m2v_intern2/zhenglin/omni_art_ablation_20261016/extensions/nft_full32'),
)
MEDIA_SUFFIXES = {'.mp4', '.wav', '.png', '.jpg', '.jpeg'}


def minimum_committed_ema(output, updates_per_rollout=1):
    versions = []
    for rank in range(32):
        path = output / 'logs' / f'full32_rank_{rank}.jsonl'
        if not path.is_file() or path.is_symlink():
            return None
        with path.open('rb') as stream:
            size = path.stat().st_size
            stream.seek(max(0, size - 262144))
            lines = stream.read(262144).splitlines(keepends=True)
        if size > 262144:
            lines = lines[1:]
        latest = None
        for line in lines:
            if not line.endswith(b'\n'):
                continue
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                continue
            if row.get('event') != 'nft_ema_update':
                continue
            version = row.get('successful_optimizer_updates')
            if not isinstance(version, int) or isinstance(version, bool) or version < 0:
                return None
            if (version % updates_per_rollout or row.get('ema_updates') != version // updates_per_rollout
                    or any(row.get(key) != version for key in ('policy_version', 'old_version'))):
                return None
            decay = row.get('decay')
            if not isinstance(decay, (int, float)) or not math.isfinite(decay):
                return None
            if abs(decay - min(0.001 * (version // updates_per_rollout), 0.5)) > 1e-12:
                return None
            latest = version // updates_per_rollout
        if latest is None:
            return None
        versions.append(latest)
    return min(versions)


def prune_completed_media(output_root, *, now=None, allowed_runtime_roots=RUNTIME_ROOTS):
    output = Path(output_root)
    if output.is_symlink():
        raise ValueError('Media retention must not follow a symlinked attempt.')
    output = output.resolve(strict=True)
    manifest = json.loads((output / 'manifest.json').read_text())
    runtime = Path(manifest['runtime_root']).resolve(strict=True)
    if runtime not in tuple(path.resolve() for path in allowed_runtime_roots):
        raise ValueError('Media retention runtime is outside the authorized NFT extension.')
    if output.parent != runtime / 'attempts' or output.name != manifest['attempt']:
        raise ValueError('Media retention attempt path differs from its manifest.')
    if manifest.get('variant') not in ('nft_full32_u1', 'nft_full32_u2') or Path(manifest['output']).resolve() != output:
        raise ValueError('Media retention only supports its exact NFT U1/U2 attempt.')
    version = minimum_committed_ema(output, 2 if manifest['variant'] == 'nft_full32_u2' else 1)
    result = {'observed_unix': time.time() if now is None else now, 'attempt': manifest['attempt'],
              'minimum_committed_ema': version, 'protected_initial_eval': 0,
              'retained_recent_rollouts': 20, 'deleted_files': [], 'deleted_bytes': 0}
    if version is None or version <= 20:
        result['status'] = 'no_completed_media_eligible'
        return result
    for directory in (output / 'validation', output / 'rollout_media', output / 'validation/wandb_val_media'):
        if not directory.is_dir() or directory.is_symlink() or not directory.resolve().is_relative_to(output):
            continue
        for step_dir in directory.iterdir():
            match = re.fullmatch(r'(?:global_step_)?([0-9]+)', step_dir.name)
            if not match or step_dir.is_symlink() or not step_dir.is_dir():
                continue
            step = int(match.group(1))
            if step == 0 or step >= version - 20:
                continue
            for media in step_dir.iterdir():
                if media.is_symlink() or not media.is_file() or not media.resolve().is_relative_to(output):
                    continue
                optional_payload = media.suffix == '.pt' and media.stem.isdecimal()
                if media.suffix.lower() not in MEDIA_SUFFIXES and not optional_payload:
                    continue
                info = media.stat()
                if result['observed_unix'] - info.st_mtime < 300:
                    continue
                media.unlink()
                result['deleted_bytes'] += info.st_size
                if len(result['deleted_files']) < 64:
                    result['deleted_files'].append(str(media.relative_to(output)))
    result['status'] = 'completed_optional_media_retention_only'
    return result
