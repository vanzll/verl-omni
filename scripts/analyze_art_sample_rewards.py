"""Describe actual complete K8 reward journals and verify their routing arithmetic."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')


def analyze(manifest_path, samples, window, output):
    manifest_path, samples, window, output = [p.resolve() for p in (manifest_path, samples, window, output)]
    assert manifest_path.is_relative_to(ROOT / 'attempts') and output.is_relative_to(ROOT / 'analysis')
    assert samples.is_relative_to(manifest_path.parent) and window.is_relative_to(manifest_path.parent)
    config_path = manifest_path.parent / 'analysis/wandb/config.json'
    observation = json.loads((config_path.parent / 'observation.json').read_text())
    assert observation['identity_verified'] is True
    manifest, config = json.loads(manifest_path.read_text()), json.loads(config_path.read_text())
    assert manifest['attempt'] == observation['attempt']
    rows = [json.loads(line) for line in samples.read_text().splitlines() if line.strip()]
    coverage = json.loads(window.read_text())
    counts = Counter(str(row['uid']) for row in rows)
    assert len(rows) == 256 and len(counts) == 32 and set(counts.values()) == {8}
    assert len({(str(row['uid']), row['sample']) for row in rows}) == 256
    a, b = coverage['windows']
    assert len(a['sample_ids']) == len(b['sample_ids']) == 128
    assert not set(a['sample_ids']) & set(b['sample_ids'])
    assert set(a['sample_ids']) | set(b['sample_ids']) == set(range(256))
    names = sorted(rows[0]['reward_raw'])
    assert set(names) == {'video_align', 'hpsv3', 'audiobox', 'clap', 'desync'}
    raw = np.asarray([[row['reward_raw'][name] for name in names] for row in rows], dtype=np.float64)
    standardized = np.asarray([[row['reward_standardized'][name] for name in names] for row in rows], dtype=np.float64)
    assert np.isfinite(raw).all() and np.isfinite(standardized).all()
    weights = {modality: np.asarray([config['reward']['reward_functions'][name]['routing_weights'].get(modality, 0.)
                                    for name in names], dtype=np.float64) for modality in ('video', 'audio')}
    joint_weights = np.maximum(weights['video'], weights['audio'])
    for index in range(len(names)):
        nonzero = {vector[index] for vector in weights.values() if vector[index] != 0}
        assert len(nonzero) == 1, 'shared component must have one consistent nonzero weight'
    assert joint_weights[names.index('desync')] == 1.
    joint = standardized @ joint_weights
    independent_routes = {modality: standardized @ vector for modality, vector in weights.items()}
    logged_routes = {modality: np.asarray([row[modality + '_routed'] for row in rows]) for modality in weights}
    route_checks = {}
    for modality in weights:
        expected = joint if config['algorithm'].get('art_joint_reward_routing', False) else independent_routes[modality]
        vector = joint_weights if config['algorithm'].get('art_joint_reward_routing', False) else weights[modality]
        bound = 16 * np.finfo(np.float32).eps * (np.abs(standardized) @ np.abs(vector) + 1e-30)
        difference = np.abs(expected - logged_routes[modality])
        route_checks[modality] = {'max_absolute_difference': float(difference.max()),
                                  'float32_dot_rounding_bound_satisfied': bool(np.all(difference <= bound))}
    assert all(check['float32_dot_rounding_bound_satisfied'] for check in route_checks.values())
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row['uid'])].append(index)
    def within(values):
        result = values.copy()
        for indices in groups.values():
            result[indices] -= values[indices].mean(axis=0)
        return result
    def correlations(values):
        centered = within(values)
        covariance = centered.T @ centered / len(rows)
        scale = np.sqrt(np.outer(np.diag(covariance), np.diag(covariance)))
        matrix = np.divide(covariance, scale, out=np.full_like(covariance, np.nan), where=scale > 0)
        return [[float(value) if np.isfinite(value) else None for value in row] for row in matrix]
    routes = np.column_stack([independent_routes['video'], independent_routes['audio'], joint])
    matrices = {'raw_components': correlations(raw), 'standardized_components': correlations(standardized),
                'standardized_routes': correlations(routes)}
    component_stats = {name: {'raw_mean': float(raw[:, i].mean()), 'raw_std_population': float(raw[:, i].std()),
                               'raw_min': float(raw[:, i].min()), 'raw_max': float(raw[:, i].max()),
                               'standardized_std_population': float(standardized[:, i].std())}
                       for i, name in enumerate(names)}
    result = {'status': 'passed_actual_journal_integrity_and_routing', 'observed_unix': time.time(),
              'attempt': manifest['attempt'], 'rollout_from_window_audit': coverage['rollout'],
              'journal_step_key': samples.stem, 'rows': 256, 'prompts': 32, 'samples_per_prompt': 8,
              'source_sha256': manifest['source']['SOURCE_SNAPSHOT_SHA256'],
              'input_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in (manifest_path, config_path, samples, window)},
              'routing_checks': route_checks, 'component_order': names, 'component_stats': component_stats,
              'raw_once_counted_weighted_total_mean': float((raw @ joint_weights).mean()),
              'raw_component_definition': 'Returned native reward components before advantage standardization, including original scorer preprocessing and HPS cap15.',
              'within_prompt_population_correlations': matrices,
              'within_prompt_exclusive_routes_correlation': correlations(np.column_stack([
                  independent_routes['video'] - standardized[:, names.index('desync')],
                  independent_routes['audio'] - standardized[:, names.index('desync')]]))[0][1],
              'route_order': ['video_route', 'audio_route', 'joint_once'],
              'windows': {'unique_prompts': [a['unique_prompts'], b['unique_prompts']],
                          'prompt_intersection': len(set(a['prompt_counts']) & set(b['prompt_counts'])),
                          'sample_intersection': 0, 'complete_union_samples': 256},
              'inactive_sample_counts': {modality: sum(not row[modality + '_group_active'] for row in rows)
                                         for modality in ('video', 'audio')},
              'scope': 'Actual scalar rewards from one pre-update rollout, one seed. Correlations pool equal complete K8 groups after within-prompt centering. Joint reward values are an offline arithmetic counterfactual, without rerunning training. Latent covariance requires separate endpoint measurements; reward correlation alone cannot validate A2 or an algorithm ranking.'}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'reward_analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    with (output / 'components.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['component', *next(iter(component_stats.values())).keys()])
        writer.writeheader()
        writer.writerows({'component': name, **stats} for name, stats in component_stats.items())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, (title, matrix) in zip(axes, matrices.items()):
        labels = names if title != 'standardized_routes' else result['route_order']
        image = ax.imshow(np.asarray(matrix, dtype=float), vmin=-1, vmax=1, cmap='coolwarm')
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        ax.set_title(title.replace('_', ' '), fontsize=10)
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                ax.text(j, i, 'NA' if value is None else f'{value:.2f}', ha='center', va='center', fontsize=8)
        fig.colorbar(image, ax=ax, fraction=.046, pad=.04)
    fig.suptitle('Observed within-prompt reward correlations · first pre-update rollout · seed42', fontsize=11)
    fig.tight_layout()
    for suffix in ('png', 'svg'):
        fig.savefig(output / f'reward_correlations.{suffix}', dpi=160)
    plt.close(fig)
    page = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Actual ART rollout reward diagnostics</title>',
            '<style>body{font:16px system-ui;max-width:1250px;margin:30px auto;padding:0 16px}img{max-width:100%}table{border-collapse:collapse}td,th{padding:7px;border:1px solid #ddd}</style>',
            '<h1>First pre-update ART rollout: actual scalar reward diagnostics</h1>',
            '<p>' + html.escape(result['scope']) + '</p>',
            '<p>' + html.escape(result['raw_component_definition']) + '</p>',
            '<p>Coverage uses the controller audit reconstructed from actual UID order and the frozen contiguous DP16 dispatch. These data do not certify optimizer completion.</p>',
            '<p><a href="reward_analysis.json">Full analysis and input hashes</a> · <a href="components.csv">Component CSV</a> · <a href="reward_correlations.svg">Figure SVG</a></p>',
            '<p>Attempt: ' + html.escape(manifest['attempt']) + '; rollout: ' + str(coverage['rollout'])
            + '; journal key: ' + html.escape(samples.stem) + '; 32 prompts × K8 = 256 samples.</p>',
            '<img src="reward_correlations.png" alt="Within-prompt scalar reward correlations">',
            '<p>Route correlation with shared DeSync: ' + str(matrices['standardized_routes'][0][1])
            + '; correlation of the two exclusive reward sums: ' + str(result['within_prompt_exclusive_routes_correlation'])
            + '. These are descriptive scalar observations from one rollout.</p>',
            '<table><tr><th>Component</th><th>Returned mean</th><th>Returned std</th><th>Min</th><th>Max</th><th>Standardized std</th></tr>']
    for name, stats in component_stats.items():
        page.append('<tr><td>' + html.escape(name) + '</td>' + ''.join('<td>' + f'{value:.6g}' + '</td>'
                    for value in stats.values()) + '</tr>')
    page.append('</table><h2>Source observations</h2>')
    for path in (samples, window):
        page.append('<p><a href="' + html.escape(os.path.relpath(path, output), quote=True) + '">'
                    + html.escape(path.name) + '</a></p>')
    page.append('</html>')
    (output / 'index.html').write_text('\n'.join(page) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('manifest', 'samples', 'window', 'output'):
        parser.add_argument('--' + field, type=Path, required=True)
    args = parser.parse_args()
    analyze(args.manifest, args.samples, args.window, args.output)
