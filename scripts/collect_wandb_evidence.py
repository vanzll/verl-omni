"""Read full W&B evidence for exact prepared attempts; never creates or edits runs."""
import argparse
import json
import math
from pathlib import Path
import time

import wandb

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')


def publish(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')
    temporary.replace(path)


def identity_values(value, wanted):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == wanted:
                yield child
            yield from identity_values(child, wanted)
    elif isinstance(value, list):
        for child in value:
            yield from identity_values(child, wanted)


def collect(api, manifest_path):
    manifest_path = manifest_path.resolve()
    assert manifest_path.is_relative_to(ROOT / 'attempts')
    manifest = json.loads(manifest_path.read_text())
    output = Path(manifest['output']).resolve()
    assert output == manifest_path.parent
    evidence = output / 'analysis/wandb'
    evidence.mkdir(parents=True, exist_ok=True)
    run_path = f"{manifest['wandb_entity']}/diffusionart-omni-ablation/{manifest['wandb_run_id']}"
    observed = {'attempt': manifest['attempt'], 'requested_path': run_path, 'unix': time.time()}
    print('Fetching exact run:', manifest['attempt'], flush=True)
    try:
        run = api.run(run_path)
        config = dict(run.config)
        summary = dict(run.summary)
        observed.update(status='fetched', url=run.url, run_name=run.name, state=run.state)
        observed['identity_verified'] = (
            run.name == manifest['attempt']
            and manifest['source']['SOURCE_SNAPSHOT_SHA256'] in list(identity_values(config, 'source_snapshot_sha256'))
            and manifest['launch_manifest_sha256'] in list(identity_values(config, 'launch_files_manifest_sha256'))
        )
        publish(evidence / 'config.json', config)
        publish(evidence / 'summary.json', summary)
        rows = 0
        columns = set()
        nonfinite = {}
        progress = {}
        temporary = evidence / 'history.jsonl.tmp'
        with temporary.open('w') as file:
            # No keys filter or sample limit: preserve rows with partial metric sets.
            for row in run.scan_history(page_size=1000):
                file.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')
                rows += 1
                columns.update(row)
                for key, value in row.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        if not math.isfinite(value):
                            nonfinite[key] = nonfinite.get(key, 0) + 1
                        elif key.startswith('progress/'):
                            progress[key] = max(progress.get(key, value), value)
        temporary.replace(evidence / 'history.jsonl')
        observed.update(rows=rows, columns=sorted(columns), nonfinite_numeric_metrics=nonfinite,
                        maximum_progress=progress, history_method='unsampled scan_history; all keys')
    except Exception as error:
        # A missing/private/unreachable run is an observation failure, not proof
        # that its training owner stopped. Never print credentials or HTTP bodies.
        observed.update(status='api_observation_unavailable', error_type=type(error).__name__)
    observed['finished_unix'] = time.time()
    publish(evidence / 'observation.json', observed)
    print(json.dumps({key: observed[key] for key in (
        'attempt', 'status', 'url', 'identity_verified', 'rows', 'maximum_progress', 'error_type'
    ) if key in observed}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('manifests', type=Path, nargs='+')
    args = parser.parse_args()
    api = wandb.Api(timeout=20)
    for manifest_path in args.manifests:
        collect(api, manifest_path)


if __name__ == '__main__':
    main()
