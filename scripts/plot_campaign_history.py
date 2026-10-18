"""Render exported actual reward points; leave missing axes empty and label scope."""
import argparse
import csv
import html
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')
COLORS = {'N0': '#444444', 'A0': '#2166ac', 'A1': '#238443', 'A2': '#d95f0e'}


def render(directory):
    assert directory.resolve().is_relative_to(ROOT / 'analysis')
    audit = json.loads((directory / 'history_audit.json').read_text())
    with (directory / 'metrics_long.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    scorers = [('video_align', 'VideoAlign'), ('hpsv3', 'HPSv3'), ('audiobox', 'AudioBox'),
               ('clap', 'CLAP'), ('desync', 'DeSync'), ('weighted_total', 'Weighted total (DeSync once)')]
    figures = []
    for phase in ['train', 'val']:
        for axis, xlabel in [('rollout', 'Completed rollout'), ('optimizer_updates', 'Successful global optimizer updates'),
                             ('end_to_end_gpu_hours', 'Allocated run GPU hours (includes initialization)')]:
            fig, axes = plt.subplots(2, 3, figsize=(13, 7))
            total_points = 0
            for ax, (scorer, title) in zip(axes.flat, scorers):
                metric = f'{phase}/reward_weighted_total/mean' if scorer == 'weighted_total' else f'{phase}/reward_raw/{scorer}/mean'
                for variant, color in COLORS.items():
                    selected = [r for r in rows if r['variant'] == variant and r['metric'] == metric and r[axis] != '']
                    points = sorted((float(r[axis]), float(r['value'])) for r in selected)
                    assert all(math.isfinite(x) and math.isfinite(y) for x, y in points)
                    if points:
                        ax.plot([p[0] for p in points], [p[1] for p in points], '.-', color=color, label=variant, linewidth=1.2)
                        total_points += len(points)
                ax.set(title=title, xlabel=xlabel, ylabel='Raw scorer / configured weighted sum')
                ax.grid(alpha=.2)
                if ax.lines:
                    ax.legend(fontsize=8)
                else:
                    ax.text(.5, .5, 'No observed points', ha='center', va='center', transform=ax.transAxes)
            if not total_points:
                plt.close(fig)
                continue
            fig.suptitle(f'Observed {phase} rewards — matched public-data comparison, seed42; incomplete until audited', fontsize=10)
            fig.tight_layout()
            stem = f'{phase}_rewards_by_{axis}'
            for suffix in ['png', 'svg']:
                fig.savefig(directory / f'{stem}.{suffix}', dpi=170)
            plt.close(fig)
            figures.append((stem, total_points))
    def link(path, label):
        relative = os.path.relpath(path, directory)
        return '<a href="' + html.escape(relative, quote=True) + '">' + html.escape(label) + '</a>'
    page = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">',
            '<title>Observed four-arm LTX training histories</title>',
            '<style>body{font:16px system-ui;max-width:1200px;margin:30px auto;padding:0 18px;line-height:1.5}img{max-width:100%}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ddd}.scope{background:#fff4da;padding:16px}</style>',
            '<h1>Observed LTX training histories</h1>',
            '<p class="scope">Incomplete scientific campaign. Single-seed matched comparison on public data; not exact reproduction of historical PR numbers. Missing points remain missing. History counts alone do not establish checkpoint recovery, media integrity or successful completion.</p>',
            '<p>' + link(directory / 'metrics_long.csv', 'All numeric metrics CSV') + ' · ' + link(directory / 'history_audit.json', 'History identity / progress / evaluation audit') + ' · ' + link(ROOT / 'track.md', 'Task progress') + '</p>',
            '<p>Run GPU hours include initialization and checkpoint costs. Failed attempts are retained in the separate reservation ledger. Initial validation has no measured per-row GPU-hour coordinate and appears only on the rollout/update plots. Reward components keep their original scales; no plot-specific reward calibration is applied.</p>',
            '<table><tr><th>Arm</th><th>Attempt</th><th>Observed updates</th><th>Validation rollouts</th><th>W&amp;B</th></tr>']
    for variant, item in audit['variants'].items():
        page.append('<tr>' + ''.join('<td>' + html.escape(str(value)) + '</td>' for value in [variant, item['attempt'], str(item['observed_optimizer_updates']) + '/150', item['validation_rollouts']]) + '<td><a href="' + html.escape(item['wandb_url'], quote=True) + '">verified run</a></td></tr>')
    page.append('</table>')
    if (directory / 'diagnostics/index.html').is_file():
        page.append('<p><a href="diagnostics/index.html">All observed numeric metrics and diagnostic plots</a></p>')
    if not figures:
        page.append('<p>No reward points have been observed in the selected verified histories.</p>')
    for stem, count in figures:
        page.append(f'<h2>{html.escape(stem.replace("_", " "))}</h2><img src="{stem}.png" alt="Observed reward curves"><p>{count} plotted metric points · <a href="{stem}.svg">SVG</a></p>')
    page.append('</html>')
    (directory / 'index.html').write_text('\n'.join(page) + '\n')
    print(json.dumps({'figures': figures, 'page': str(directory / 'index.html'), 'scope': 'observed histories only; no completion claim'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    render(parser.parse_args().directory)
