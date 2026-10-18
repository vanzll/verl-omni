#!/usr/bin/env python3
import concurrent.futures
import json
from pathlib import Path
from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')
DESTS = {'MizzenAI/HPSv3': 'HPSv3', 'KlingTeam/VideoReward': 'VideoReward',
         'facebook/audiobox-aesthetics': 'audiobox-aesthetics',
         'laion/clap-htsat-unfused': 'checkpoints/clap-htsat-unfused',
         'Qwen/Qwen2-VL-2B-Instruct': 'Qwen2-VL-2B-Instruct',
         'Qwen/Qwen2-VL-7B-Instruct': 'Qwen2-VL-7B-Instruct'}


def repo_tasks(asset):
    repo, revision = asset['repo'], asset['revision']
    info = HfApi().model_info(repo, revision=revision, files_metadata=True, timeout=25)
    assert info.sha == revision
    siblings = {s.rfilename: s for s in info.siblings}
    result = []
    dest = ROOT / 'assets/rewards' / DESTS[repo]
    for f in asset['files']:
        sibling = siblings[f['path']]
        assert sibling.size == f['size']
        task = dict(name=repo + '/' + f['path'], size=f['size'],
                    url=hf_hub_url(repo, f['path'], revision=revision),
                    destination=str(dest / f['path']),
                    hf=dict(repo=repo, revision=revision, filename=f['path']),
                    hf_metadata=str(dest / '.cache/huggingface/download' / (f['path'] + '.metadata')))
        if sibling.lfs:
            task['sha256'] = sibling.lfs.sha256
        else:
            task['git_blob_sha1'] = sibling.blob_id
        if repo == 'KlingTeam/VideoReward':
            expected = task.get('sha256', task.get('git_blob_sha1'))
            partial_dir = dest / '.cache/huggingface/download' / str(Path(f['path']).parent)
            task['legacy_partials'] = [str(p) for p in partial_dir.glob('*.' + expected + '.incomplete')]
            task['legacy_owner'] = dict(pid=297729, start_ticks=2417025926)
        result.append(task)
    return result


assets = [x for x in json.loads((ROOT / 'assets/hf_asset_sizes.json').read_text()) if x['repo'] in DESTS]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    tasks = [task for group in pool.map(repo_tasks, assets) for task in group]
repo, revision, filename = ('zghhui/OmniNFT-Reward-Series', '9e30061a1392d03bafdcf717e80a385ddf411b4d',
                            'synchformer/synchformer_state_dict.pth')
url = hf_hub_url(repo, filename, revision=revision)
metadata = get_hf_file_metadata(url, timeout=25)
assert metadata.commit_hash == revision and len(metadata.etag) == 64
tasks.append(dict(name=repo + '/' + filename, url=url, size=metadata.size, sha256=metadata.etag,
                  destination=str(ROOT / 'assets/rewards' / filename),
                  hf=dict(repo=repo, revision=revision, filename=filename),
                  hf_metadata=str(ROOT / 'assets/rewards/.cache/huggingface/download' / (filename + '.metadata'))))
tasks.sort(key=lambda x: (x['size'] > 1024**2, -x['size']))
(ROOT / 'assets/reward-fast-manifest.json').write_text(json.dumps(tasks, indent=2) + '\n')
print('pinned reward files:', len(tasks), 'total_GB:', round(sum(t['size'] for t in tasks) / 1e9, 2))
