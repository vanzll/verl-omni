"""Inventory and plot every observed numeric metric from an identity-checked export."""
import argparse
from collections import defaultdict
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import statistics
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')
COLORS = {'N0': '#444444', 'A0': '#2166ac', 'A1': '#238443', 'A2': '#d95f0e'}


def render(directory):
    directory = directory.resolve()
    assert directory.is_relative_to(ROOT / 'analysis')
    audit_path, data_path = directory / 'history_audit.json', directory / 'metrics_long.csv'
    inputs = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (audit_path, data_path)}
    audit = json.loads(audit_path.read_text())
    output = directory / 'diagnostics'
    output.mkdir(exist_ok=True)
    series = defaultdict(list)
    with data_path.open() as file:
        for row in csv.DictReader(file):
            assert row['variant'] in audit['variants']
            assert row['attempt'] == audit['variants'][row['variant']]['attempt']
            value = float(row['value'])
            assert math.isfinite(value)
            row['value'] = value
            series[row['variant'], row['metric']].append(row)
    summaries = []
    for (variant, metric), rows in sorted(series.items()):
        rows.sort(key=lambda row: int(row['history_row']))
        values = [row['value'] for row in rows]
        known = [row for row in rows if row['optimizer_updates'] != '']
        unknown = [row for row in rows if row['optimizer_updates'] == '']
        summaries.append({'variant': variant, 'metric': metric, 'observations': len(rows),
                          'known_update_coordinates': len(known), 'unknown_update_coordinates': len(unknown),
                          'minimum': min(values), 'maximum': max(values), 'mean': statistics.mean(values),
                          'std_population': statistics.pstdev(values), 'first': values[0], 'last': values[-1],
                          'last_minus_first': values[-1] - values[0],
                          'first_history_row': rows[0]['history_row'], 'last_history_row': rows[-1]['history_row'],
                          'unknown_progress_history_rows': ','.join(row['history_row'] for row in unknown)})
    fields = ['variant', 'metric', 'observations', 'known_update_coordinates', 'unknown_update_coordinates',
              'minimum', 'maximum', 'mean', 'std_population', 'first', 'last', 'last_minus_first',
              'first_history_row', 'last_history_row', 'unknown_progress_history_rows']
    with (output / 'metric_summary.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    groups = defaultdict(list)
    for metric in sorted({metric for _, metric in series}):
        groups[metric.split('/')[0]].append(metric)
    figures = []
    for group, metrics in sorted(groups.items()):
        for start in range(0, len(metrics), 6):
            selected = metrics[start:start + 6]
            fig, axes = plt.subplots(3, 2, figsize=(15, 10))
            plotted = 0
            for ax, metric in zip(axes.flat, selected):
                for variant in sorted(audit['variants']):
                    rows = series.get((variant, metric), [])
                    points = [(float(row['optimizer_updates']), row['value']) for row in rows
                              if row['optimizer_updates'] != '']
                    assert all(math.isfinite(x) for x, _ in points)
                    if points:
                        ax.plot([x for x, _ in points], [y for _, y in points], '.-',
                                color=COLORS.get(variant), label=variant, linewidth=1.1)
                        plotted += len(points)
                ax.set_title(metric, fontsize=8, wrap=True)
                ax.set_xlabel('Successful global optimizer updates (verified mapping)')
                ax.set_ylabel('Original metric scale')
                ax.grid(alpha=.2)
                if ax.lines:
                    ax.legend(fontsize=8)
                else:
                    ax.text(.5, .5, 'No known progress coordinate', ha='center', va='center', transform=ax.transAxes)
            for ax in list(axes.flat)[len(selected):]:
                ax.set_visible(False)
            fig.suptitle('Observed metrics; one seed; missing data remain missing', fontsize=11)
            fig.tight_layout()
            stem = 'metrics_' + hashlib.sha256(group.encode()).hexdigest()[:12] + '_' + str(start // 6)
            for suffix in ('png', 'svg'):
                fig.savefig(output / f'{stem}.{suffix}', dpi=150)
            plt.close(fig)
            figures.append({'group': group, 'metrics': selected, 'stem': stem, 'plotted_points': plotted})
    result = {'status': 'rendered_observed_metrics', 'observed_unix': time.time(),
              'input_sha256': inputs, 'numeric_metric_names': len({metric for _, metric in series}),
              'variant_metric_series': len(series), 'observations': sum(len(rows) for rows in series.values()),
              'unknown_progress_observations': sum(row['unknown_update_coordinates'] for row in summaries),
              'figures': figures, 'campaign_complete': False,
              'nonfinite_metrics_from_history_audit': {key: item['nonfinite_numeric_metrics'] for key, item in audit['variants'].items()},
              'scope': 'Descriptive observations only. No interpolation, fabricated update mapping, statistical significance or scientific completion claim.'}
    (output / 'diagnostic_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    esc = lambda value: html.escape(str(value), quote=True)
    page = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Observed diagnostic metrics</title>',
            '<style>body{font:15px system-ui;max-width:1450px;margin:25px auto;padding:0 15px}img{max-width:100%}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:5px}td{overflow-wrap:anywhere}details{margin:20px 0}</style>',
            '<h1>Observed diagnostic metrics</h1>',
            '<p>Incomplete matched public-data comparison, one seed. All numeric metrics retain their original scales. Summary rows include observations with unknown progress; plots require an explicit verified update coordinate. Missing or nonfinite metrics are recorded in the audit.</p>',
            '<p>Single observations cannot establish a trend. First/last/delta describe observed event order, not significance or causal effects. No cross-arm calibration or failure-attempt splicing is used.</p>',
            '<p><a href="../index.html">Reward curves</a> · <a href="../metrics_long.csv">All observations CSV</a> · <a href="metric_summary.csv">Metric summaries CSV</a> · <a href="diagnostic_audit.json">Diagnostic audit</a> · <a href="../history_audit.json">History audit</a></p>',
            '<h2>All observed metric summaries</h2><table><tr>' + ''.join('<th>' + esc(k) + '</th>' for k in fields[:12]) + '</tr>']
    for row in summaries:
        page.append('<tr>' + ''.join('<td>' + esc(row[k]) + '</td>' for k in fields[:12]) + '</tr>')
    page.append('</table>')
    if not figures:
        page.append('<p>No numeric metric observations in the selected verified histories.</p>')
    for figure in figures:
        stem = figure['stem']
        page.append('<details open><summary>' + esc(figure['group']) + ' · ' + str(figure['plotted_points']) + f' observed points</summary><img src="{stem}.png" alt="Observed metrics"><p><a href="{stem}.svg">SVG</a></p></details>')
    page.append('</html>')
    (output / 'index.html').write_text('\n'.join(page) + '\n')
    for name, expected in inputs.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, 'history export changed during render'
    print(json.dumps({k: v for k, v in result.items() if k not in ('figures', 'input_sha256')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    render(parser.parse_args().directory)
