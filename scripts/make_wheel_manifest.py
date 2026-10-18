#!/usr/bin/env python3
"""Select this interpreter's compatible artifacts from a uv pylock file."""
import argparse
import hashlib
import json
import re
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlsplit

import requests
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('lock', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--only', action='append', help='Stage selected locked packages only')
    args = parser.parse_args()
    assert args.output.resolve().is_relative_to(ROOT)
    lock = tomllib.loads(args.lock.read_text())
    ranks = {tag: i for i, tag in enumerate(sys_tags())}
    tasks, sources, reused = [], [], 0
    session = requests.Session()
    for package in lock['packages']:
        if args.only and package['name'] not in args.only:
            continue
        if 'vcs' in package or 'directory' in package:
            sources.append(package)
            continue
        choices = []
        for wheel in package.get('wheels', []):
            location = wheel.get('url', wheel.get('path'))
            assert location, 'Wheel has no source location'
            filename = unquote(urlsplit(location).path.rsplit('/', 1)[-1])
            _, _, _, tags = parse_wheel_filename(filename)
            scores = [ranks[tag] for tag in tags if tag in ranks]
            if scores:
                choices.append((min(scores), filename, wheel))
        if choices:
            _, filename, artifact = min(choices, key=lambda value: (
                value[0], value[2].get('url', '').startswith('file:')))
        else:
            artifact = package.get('sdist')
            if not artifact and 'archive' in package:
                artifact = package['archive']
            if not artifact:
                raise ValueError('No supported artifact for ' + package['name'])
            filename = unquote(urlsplit(artifact.get('url', artifact.get('path', ''))).path.rsplit('/', 1)[-1])
        assert filename == Path(filename).name
        source_url = urlsplit(artifact.get('url', ''))
        if 'path' in artifact or source_url.scheme == 'file':
            assert not source_url.netloc or source_url.netloc == 'localhost'
            local_path = Path(artifact.get('path', unquote(source_url.path)))
            if not local_path.is_absolute():
                local_path = args.lock.parent / local_path
            assert local_path.resolve().is_relative_to(ROOT) and local_path.is_file()
            with local_path.open('rb') as file:
                assert hashlib.file_digest(file, 'sha256').hexdigest() == artifact['hashes']['sha256']
            sources.append(dict(name=package['name'], version=package['version'],
                                kind='verified-local-artifact', path=str(local_path.resolve()),
                                sha256=artifact['hashes']['sha256']))
            continue
        url = urlsplit(artifact['url'])
        assert url.scheme == 'https' and url.username is None and not url.query
        sha = artifact['hashes']['sha256']
        assert re.fullmatch('[0-9a-f]{64}', sha)
        size = artifact.get('size')
        if size is None:
            with session.get(artifact['url'], headers={'Range': 'bytes=0-0',
                             'Accept-Encoding': 'identity'}, stream=True, timeout=(10, 30)) as response:
                response.raise_for_status()
                content_range = response.headers.get('Content-Range', '')
                match = re.fullmatch(r'bytes 0-0/(\d+)', content_range)
                if response.status_code != 206 or not match:
                    raise ValueError('Size probe did not honor range for ' + package['name'])
                size = int(match[1])
        target = ROOT / 'assets/wheelhouse' / filename
        # Keep verified files in the manifest too; the downloader rechecks reuse.
        if target.exists() and target.stat().st_size == size:
            with target.open('rb') as file:
                if hashlib.file_digest(file, 'sha256').hexdigest() == sha:
                    reused += 1
        tasks.append(dict(name=package['name'] + '==' + package['version'],
                          url=artifact['url'], sha256=sha, size=size,
                          destination=str(target)))
    tasks.sort(key=lambda task: (task['size'] > 1024**2, -task['size']))
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(tasks, indent=2) + '\n')
    temporary.replace(args.output)
    source_path = args.output.with_name(args.output.stem + '-sources.json')
    source_path.write_text(json.dumps(sources, indent=2) + '\n')
    print('Artifacts', len(tasks), 'verified_reuse', reused,
          'source_packages', len(sources),
          'total_GB', round(sum(task['size'] for task in tasks) / 1e9, 3), flush=True)


if __name__ == '__main__':
    main()
