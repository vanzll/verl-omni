#!/usr/bin/env python3
"""Keep the pinned Hub blob/snapshot layout while replacing the transport."""
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_url

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')


def main():
    asset = next(item for item in json.loads((ROOT / 'assets/hf_asset_sizes.json').read_text())
                 if item['repo'] == 'diffusers/LTX-2.3-Diffusers')
    info = HfApi().model_info(asset['repo'], revision=asset['revision'],
                              files_metadata=True, timeout=25)
    assert info.sha == asset['revision']
    files = {item.rfilename: item for item in info.siblings}
    hub = ROOT / 'cache/hf/hub/models--diffusers--LTX-2.3-Diffusers'
    tasks, links = {}, []
    for item in asset['files']:
        sibling = files[item['path']]
        assert sibling.size == item['size']
        expected = sibling.lfs.sha256 if sibling.lfs else sibling.blob_id
        target = hub / 'blobs' / expected
        task = dict(name=asset['repo'] + '/' + item['path'], size=item['size'],
                    destination=str(target),
                    url=hf_hub_url(asset['repo'], item['path'], revision=asset['revision']),
                    hf=dict(repo=asset['repo'], revision=asset['revision'], filename=item['path']),
                    legacy_owner=dict(pid=184791, start_ticks=2416459204),
                    legacy_partials=[str(target.with_name(expected + '.incomplete'))])
        task['sha256' if sibling.lfs else 'git_blob_sha1'] = expected
        if expected in tasks:
            assert tasks[expected]['size'] == task['size']
        else:
            tasks[expected] = task
        links.append(dict(snapshot=str(hub / 'snapshots' / asset['revision'] / item['path']),
                          blob=str(target)))
    ordered = sorted(tasks.values(), key=lambda item: (item['size'] > 1024**2, -item['size']))
    for name, value in [('ltx-fast-manifest.json', ordered), ('ltx-snapshot-links.json', links)]:
        temporary = ROOT / 'assets' / (name + '.tmp')
        temporary.write_text(json.dumps(value, indent=2) + '\n')
        temporary.replace(ROOT / 'assets' / name)
    print('LTX unique blobs', len(ordered), 'snapshot files', len(links),
          'total_GB', round(sum(item['size'] for item in ordered) / 1e9, 2), flush=True)


if __name__ == '__main__':
    main()
