#!/usr/bin/env python3
"""Pin compatible cu129 artifacts and publisher digests from the official index."""
import concurrent.futures
import json
import re
from pathlib import Path
from urllib.parse import parse_qs, unquote, urljoin, urlsplit, urlunsplit

import requests
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')
VERSIONS = {'torch': '2.13.0+cu129', 'torchvision': '0.28.0+cu129',
            'torchaudio': '2.11.0+cu129', 'torchcodec': '0.16.0+cu129'}
RANKS = {tag: i for i, tag in enumerate(sys_tags())}


def one(item):
    name, version = item
    index = f'https://download.pytorch.org/whl/cu129/{name}/'
    response = requests.get(index, timeout=(8, 20))
    response.raise_for_status()
    choices = []
    for href in re.findall(r'href="([^"]+)"', response.text):
        url = urlsplit(urljoin(index, href))
        filename = unquote(url.path.rsplit('/', 1)[-1])
        if not filename.endswith('.whl'):
            continue
        _, found_version, _, tags = parse_wheel_filename(filename)
        if str(found_version) != version:
            continue
        scores = [RANKS[tag] for tag in tags if tag in RANKS]
        if scores:
            hashes = parse_qs(url.fragment)
            if 'sha256' not in hashes:
                continue
            sha = hashes['sha256'][0]
            assert re.fullmatch('[0-9a-f]{64}', sha)
            choices.append((min(scores), filename, urlunsplit(url._replace(fragment='')), sha))
    if not choices:
        raise ValueError('No compatible official wheel for ' + name + '==' + version)
    _, filename, url, sha = min(choices)
    with requests.get(url, headers={'Range': 'bytes=0-0', 'Accept-Encoding': 'identity'},
                      stream=True, timeout=(8, 20)) as response:
        response.raise_for_status()
        match = re.fullmatch(r'bytes 0-0/(\d+)', response.headers.get('Content-Range', ''))
        assert response.status_code == 206 and match
        size = int(match[1])
    return dict(name=name + '==' + version, url=url, size=size, sha256=sha,
                destination=str(ROOT / 'assets/wheelhouse' / filename), publisher_index=index)


tasks, deferred = [], []
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    futures = {pool.submit(one, item): item for item in VERSIONS.items()}
    for future in concurrent.futures.as_completed(futures):
        try:
            tasks.append(future.result())
        except (ValueError, requests.RequestException) as error:
            name, version = futures[future]
            deferred.append(dict(name=name, version=version, reason=str(error)))
            print('Deferred to dependency resolver:', name, version, type(error).__name__, flush=True)
assert tasks, 'No publisher-hash-pinned artifacts were found'
path = ROOT / 'assets/torch-cu129-bootstrap-manifest.json'
temporary = path.with_suffix('.tmp')
temporary.write_text(json.dumps(tasks, indent=2) + '\n')
temporary.replace(path)
(ROOT / 'assets/torch-cu129-bootstrap-deferred.json').write_text(json.dumps(deferred, indent=2) + '\n')
print('Official cu129 Torch artifacts', len(tasks), 'GB', sum(t['size'] for t in tasks) / 1e9, flush=True)
