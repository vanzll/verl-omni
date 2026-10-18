#!/usr/bin/env python3
"""Install hash-verified GLVND libraries into this campaign's private environment."""
import concurrent.futures
import ctypes
import hashlib
import io
import json
import tarfile
import zipfile
from pathlib import Path

import requests

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')
packages = json.loads((ROOT / 'assets/libgl_conda_packages.json').read_text())
cache = ROOT / 'cache/conda/native-gl'
cache.mkdir(exist_ok=True)


def fetch(item):
    name, metadata = item
    path = cache / Path(metadata['basename']).name
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == metadata['sha256']:
        return name, path
    response = requests.get('https://conda.anaconda.org/conda-forge/' + metadata['basename'], timeout=(8, 30))
    response.raise_for_status()
    data = response.content
    assert len(data) == metadata['size'] and hashlib.sha256(data).hexdigest() == metadata['sha256']
    temporary = path.with_suffix('.tmp')
    temporary.write_bytes(data)
    temporary.replace(path)
    print('Verified GL package', name, metadata['sha256'], flush=True)
    return name, path


with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    files = list(pool.map(fetch, packages.items()))
zstd = ctypes.CDLL('libzstd.so.1')
zstd.ZSTD_decompress.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
zstd.ZSTD_decompress.restype = ctypes.c_size_t
zstd.ZSTD_isError.argtypes = [ctypes.c_size_t]
zstd.ZSTD_isError.restype = ctypes.c_uint
installed = []
for name, path in files:
    with zipfile.ZipFile(path) as archive:
        payload = next(item for item in archive.namelist() if item.startswith('pkg-') and item.endswith('.tar.zst'))
        compressed = archive.read(payload)
    output = ctypes.create_string_buffer(64 * 1024**2)
    length = zstd.ZSTD_decompress(output, len(output), compressed, len(compressed))
    assert not zstd.ZSTD_isError(length), 'Zstd decompression failed'
    with tarfile.open(fileobj=io.BytesIO(output.raw[:length]), mode='r:') as archive:
        members = [item for item in archive.getmembers() if item.name.startswith('lib/')]
        for member in members:
            destination = ROOT / 'env' / member.name
            assert destination.parent.resolve().is_relative_to(ROOT / 'env')
            archive.extract(member, path=ROOT / 'env', filter='data')
            if destination.is_file():
                installed.append(dict(path=str(destination), sha256=hashlib.sha256(destination.read_bytes()).hexdigest()))
    print('Installed private GL libraries', name, flush=True)
(ROOT / 'assets/native_gl_manifest.json').write_text(json.dumps(dict(packages=packages, files=installed), indent=2) + '\n')
