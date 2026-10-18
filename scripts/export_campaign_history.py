"""Export exact, identity-checked unsampled histories without inventing progress."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def export(manifests, destination):
    assert destination.resolve().is_relative_to(ROOT / 'analysis')
    destination.mkdir(parents=True, exist_ok=True)
    records, inputs, variants = [], {}, {}
    for manifest_path in manifests:
        manifest_path = manifest_path.resolve()
        assert manifest_path.is_relative_to(ROOT / 'attempts')
        manifest = json.loads(manifest_path.read_text())
        variant = manifest['variant']
        assert variant not in variants, 'one continuous run per scientific arm; do not splice attempts'
        evidence = manifest_path.parent / 'analysis/wandb'
        observation = json.loads((evidence / 'observation.json').read_text())
        assert observation['status'] == 'fetched' and observation['identity_verified'] is True
        assert observation['attempt'] == manifest['attempt']
        for path in [manifest_path, evidence / 'observation.json', evidence / 'config.json',
                     evidence / 'summary.json', evidence / 'history.jsonl']:
            inputs[str(path.relative_to(ROOT))] = digest(path)
        source_rows = [json.loads(line) for line in (evidence / 'history.jsonl').read_text().splitlines() if line.strip()]
        assert len(source_rows) == observation['rows']
        progress_seen, validation_rollouts, nonfinite = [], [], []
        for row_index, row in enumerate(source_rows):
            rollout = row.get('progress/rollout')
            updates = row.get('progress/optimizer_updates')
            generated = row.get('progress/generated_samples')
            progress_source = 'explicit campaign metrics'
            if rollout is not None:
                assert all(numeric(x) and math.isfinite(x) and x == int(x) for x in [rollout, updates, generated])
                assert 1 <= rollout <= 75 and updates == 2 * rollout and generated == 256 * rollout
                assert not progress_seen or rollout >= progress_seen[-1], 'nonmonotonic actual rollout counter'
                progress_seen.append(rollout)
            elif row.get('_step') == 0 and any(key.startswith('val/reward_raw/') for key in row):
                # Frozen trainer logs its initial validation with step=0 before
                # entering the training loop. Never use _step as later progress.
                rollout, updates, generated = 0, 0, 0
                progress_source = 'initial validation event; frozen trainer step0 semantics'
            else:
                progress_source = 'unknown; no mapping from generic W&B step'
            if any(key.startswith('val/reward_raw/') for key in row) and rollout is not None:
                validation_rollouts.append(rollout)
            for metric, value in row.items():
                if metric.startswith('_') or not numeric(value):
                    continue
                if not math.isfinite(value):
                    nonfinite.append({'history_row': row_index, 'metric': metric, 'value': str(value)})
                    continue
                records.append({'variant': variant, 'attempt': manifest['attempt'], 'history_row': row_index,
                                'wandb_step': row.get('_step'), 'rollout': rollout,
                                'optimizer_updates': updates, 'generated_samples': generated,
                                'training_gpu_hours': row.get('cost/training_gpu_hours'),
                                'end_to_end_gpu_hours': row.get('cost/end_to_end_gpu_hours'),
                                'progress_source': progress_source, 'metric': metric, 'value': value})
        max_rollout = max(progress_seen, default=0)
        expected_evals = [0, 10, 20, 30, 40, 50, 60, 70, 75]
        variants[variant] = {'attempt': manifest['attempt'], 'wandb_url': observation['url'],
                             'source_sha256': manifest['source']['SOURCE_SNAPSHOT_SHA256'],
                             'launch_sha256': manifest['launch_manifest_sha256'],
                             'history_rows': len(source_rows), 'observed_rollout': max_rollout,
                             'observed_optimizer_updates': 2 * max_rollout,
                             'duplicate_progress_rows': len(progress_seen) - len(set(progress_seen)),
                             'missing_training_rollouts': sorted(set(range(1, 76)) - set(progress_seen)),
                             'validation_rollouts': sorted(set(validation_rollouts)),
                             'missing_validation_rollouts': sorted(set(expected_evals) - set(validation_rollouts)),
                             'nonfinite_numeric_metrics': nonfinite,
                             'training_and_eval_history_budget_complete': max_rollout == 75
                             and set(range(1, 76)) <= set(progress_seen)
                             and set(expected_evals) <= set(validation_rollouts) and not nonfinite}
    assert len({item['source_sha256'] for item in variants.values()}) <= 1, 'comparison sources differ'
    for name, expected in inputs.items():
        assert digest(ROOT / name) == expected, 'evidence changed while exporting; collect again'
    fields = ['variant', 'attempt', 'history_row', 'wandb_step', 'rollout', 'optimizer_updates',
              'generated_samples', 'training_gpu_hours', 'end_to_end_gpu_hours',
              'progress_source', 'metric', 'value']
    with (destination / 'metrics_long.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    result = {'observed_unix': time.time(), 'variants': variants, 'input_sha256': inputs,
              'scope': 'Matched public-data re-comparison, single seed; not exact historical PR reproduction.',
              'campaign_complete': False,
              'completion_note': 'History budget alone does not verify media, distributed checkpoint GPU readback, UID/covariance audits or owner cleanup.'}
    (destination / 'history_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'variants': variants, 'metric_points': len(records)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifests', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'analysis/training_curves')
    args = parser.parse_args()
    export(args.manifests, args.output)
