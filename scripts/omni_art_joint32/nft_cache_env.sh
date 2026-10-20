#!/usr/bin/env bash

_nft_cache_env() {
    local assignments
    assignments=$(timeout --foreground 15s python -I -S - <<'PY'
import os
from pathlib import Path
import shlex
import tempfile

roots = (
    Path('/m2v_intern/zhenglin/omni_art_ablation_20261016/extensions/nft_full32'),
    Path('/m2v_intern2/zhenglin/omni_art_ablation_20261016/extensions/nft_full32'),
)
if os.environ.get('OMNI_VARIANT') not in ('NFT_FULL32_U1', 'NFT_FULL32_U2'):
    raise SystemExit('NFT cache environment requires an NFT Full32 U1/U2 variant')
raw_runtime = os.environ.get('OMNI_RUNTIME_ROOT', '')
if not raw_runtime or not Path(raw_runtime).is_absolute():
    raise SystemExit('NFT cache environment requires an absolute OMNI_RUNTIME_ROOT')
runtime = Path(raw_runtime).resolve(strict=True)
if not any(runtime == root or runtime.is_relative_to(root) for root in roots):
    raise SystemExit('NFT cache runtime is outside the authorized extension roots')
if not runtime.is_dir():
    raise SystemExit('NFT cache runtime must be an existing directory')

def owned_directory(path, allow_alias=False):
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('NFT cache directory must be absolute')
    resolved = path.resolve()
    if resolved == runtime or not resolved.is_relative_to(runtime):
        raise ValueError('NFT cache directory escapes the runtime root')
    if allow_alias:
        path = resolved
    relative = path.relative_to(runtime)
    current = runtime
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise ValueError('NFT cache directory must not traverse symlinks')
    return path

raw_tmp = os.environ.get('TMPDIR', '')
if not raw_tmp:
    raise SystemExit('NFT cache environment requires an owned TMPDIR')
home = owned_directory(runtime / 'cachehome')
workspace = owned_directory(runtime / 'cache' / 'flashinfer')
temporary = owned_directory(Path(raw_tmp), allow_alias=True)
directories = (home, workspace, temporary)
for directory in directories:
    directory.mkdir(parents=True, exist_ok=True)
    owned_directory(directory)
    with tempfile.TemporaryFile(dir=directory) as probe:
        if probe.write(b'nft-cache-probe\n') != 16:
            raise OSError('NFT cache write probe was incomplete')
        probe.flush()
for key, value in (('HOME', home), ('FLASHINFER_WORKSPACE_BASE', workspace)):
    print('export ' + key + '=' + shlex.quote(str(value)))
PY
    ) || return $?
    eval "$assignments"
}

if _nft_cache_env; then
    unset -f _nft_cache_env
else
    unset -f _nft_cache_env
    return 1
fi
