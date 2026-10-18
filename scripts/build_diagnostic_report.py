"""Export measured preflight diagnostics; never substitutes for four-arm results."""
import csv
import hashlib
import html
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')
OUT = ROOT / 'analysis/preflight_diagnostics'
OUT.mkdir(parents=True, exist_ok=True)
native_dir = ROOT / 'preflight/runtime_gpu/ltx-rollout-20261016T145624Z/klingai-wlf2-ge103-134'
actor_dir = ROOT / 'preflight/runtime_gpu/actor-gpu-20261016T154937Z/klingai-wlf2-ge103-134'
native_path = native_dir / 'captures/native_sampler_euler_comparison.json'
actor_path = actor_dir / 'recompute_mb1.json'
native = json.loads(native_path.read_text())
actor = json.loads(actor_path.read_text())
state = json.loads((actor_dir / 'actor_gpu_state.json').read_text())
nft_dir = ROOT / 'preflight/runtime_gpu/nft-gpu-20261016T160455Z/klingai-wlf2-ge103-134'
nft = json.loads((nft_dir / 'nft_gpu_state.json').read_text())
assert nft['stage'] == 'diagnostics_completed' and (nft_dir / 'owner_exit_code').read_text().strip() == '0'
precision_dir = ROOT / 'preflight/runtime_gpu/art-precision-20261016T164637Z/klingai-wlf2-ge103-134'
precision = json.loads((precision_dir / 'precision_gpu_state.json').read_text())
assert precision['stage'] == 'diagnostics_completed' and (precision_dir / 'owner_exit_code').read_text().strip() == '0'
assert state['stage'] == 'diagnostics_completed'
assert (actor_dir / 'owner_exit_code').read_text().strip() == '0'
rows = []
for item in native['seeds']:
    seed = item['seed']
    assert seed in (42, 43)
    for record in item['velocity_comparisons']:
        sample = next(o for o in actor['observations'] if o['sample_start'] == seed - 42
                      and o['step_index'] == record['index'])
        for modality in ('video', 'audio'):
            stats = record['modalities'][modality]
            rows.append({'seed': seed, 'denoise_index': record['index'], 'modality': modality,
                         'sigma': record['sigma'], 'sigma_next': record['sigma_next'],
                         'unrounded_euler_relative_rms': stats['sampler_vs_unrounded_euler']['relative_error_rms'],
                         'rounded_euler_relative_rms': stats['sampler_vs_rounded_euler']['relative_error_rms'],
                         'actor_relative_rms': sample[modality]['actor_sampler_relative_rms'],
                         'initial_current_old_mse': sample[modality]['initial_restoration_mse']})
with (OUT / 'velocity_comparison.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
series = [('unrounded_euler_relative_rms', 'Euler before state rounding', '#238443'),
          ('rounded_euler_relative_rms', 'Euler from BF16 rounded states', '#d95f0e'),
          ('actor_relative_rms', 'Actor recomputed old', '#2166ac')]
for ax, modality in zip(axes, ('video', 'audio')):
    for seed, marker in ((42, 'o'), (43, 's')):
        selected = [r for r in rows if r['modality'] == modality and r['seed'] == seed]
        for key, label, color in series:
            ax.semilogy([r['denoise_index'] for r in selected], [r[key] for r in selected],
                        color=color, marker=marker, linestyle='-' if seed == 42 else '--',
                        label=label if seed == 42 else None, alpha=.85)
    ax.set(title=modality.capitalize(), xlabel='Denoise index (20-step schedule)', xticks=[0, 9, 18])
    ax.grid(alpha=.2, which='both')
axes[0].set_ylabel('Error RMS / actual sampler velocity RMS')
axes[1].legend(fontsize=8, loc='lower left')
fig.suptitle('Fixed base-model probe: two seeds, three visited states; o=42, square=43', fontsize=10)
fig.tight_layout()
fig.savefig(OUT / 'velocity_comparison.png', dpi=180)
fig.savefig(OUT / 'velocity_comparison.svg')
plt.close(fig)
details = precision['microbatch_partition']['sign_and_energy']
bins = details['magnitude_bins']
fig, ax = plt.subplots(figsize=(9, 4.6))
positions = list(range(len(bins)))
ax.bar([i - .19 for i in positions], [100 * b['gradient_energy_fraction'] for b in bins],
       width=.38, color='#2166ac', label='Share of pooled gradient energy')
ax.bar([i + .19 for i in positions], [100 * b['update_error_energy_fraction'] for b in bins],
       width=.38, color='#d95f0e', label='Share of Adam update difference energy')
ax.set_xticks(positions, ['<0.01', '0.01–0.1', '0.1–1', '1–10', '≥10'])
ax.set(xlabel='max(|g_mb2|, |g_mb1|) / pooled active-gradient RMS', ylabel='Fraction (%)',
       title='Fixed-input initial BF16 LoRA: small-gradient sign sensitivity')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=.2)
fig.tight_layout()
fig.savefig(OUT / 'gradient_sign_energy.png', dpi=180)
fig.savefig(OUT / 'gradient_sign_energy.svg')
plt.close(fig)


def link(path, label):
    relative = os.path.relpath(path, OUT)
    assert not relative.startswith('/')
    return f'<a href="{html.escape(relative, quote=True)}">{html.escape(label)}</a>'


def table(mapping):
    return '<table>' + ''.join('<tr><th>' + html.escape(str(k)) + '</th><td>' +
                              html.escape(str(v)) + '</td></tr>' for k, v in mapping.items()) + '</table>'


mb = state['microbatch']
cache = state['cache_target']
checkpoint = state['checkpoint']
page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>LTX preflight diagnostics — incomplete four-arm experiment</title>
<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 18px;color:#17202a;line-height:1.5}
img{max-width:100%%}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:7px;text-align:left}
.scope{padding:16px;background:#fff4da}code{overflow-wrap:anywhere}a{color:#145a96}</style>
<h1>Measured LTX preflight diagnostics</h1>
<p class="scope"><strong>Incomplete scientific campaign.</strong> These are independent fixed-input diagnostics,
not four-arm training curves or held-out validation scores. No conclusion about ART versus NFT quality follows from them.</p>
<h2>Three velocity definitions</h2><img src="velocity_comparison.png" alt="Actual velocity error by denoise index and modality">
<p>Both Euler errors use the actual sampler velocity as reference. The unrounded transition matches Euler closely.
Differencing BF16-rounded states amplifies rounding error when delta sigma is small. Actor recompute has a separate
measured mismatch. The same model revision, CFG, text conditions, states and sigma schedule were supplied;
the actor uses the Diffusers native attention stack and rollout uses vLLM-Omni TORCH_SDPA. This does not establish
which implementation detail causes their mismatch. Initial current-versus-old restoration MSE was exactly zero.</p>
<p>%s · %s · %s</p>
<h2>Microbatch comparison: numerical equivalence unproven</h2>
<p>Two fixed samples × three selected timesteps; native accumulation is three backward calls for microbatch two,
six for microbatch one. All formal arms continue using microbatch two. BF16 reduction and Adam sensitivity to small
gradient signs are hypotheses, not isolated explanations.</p>%s
<h2>Measured repeatability and initial Adam sign sensitivity</h2>
<img src="gradient_sign_energy.png" alt="Gradient energy and Adam update difference by gradient magnitude">
<p>A separate three-branch probe uses explicit initialization seed42 and identical starting adapters/RNG.
Parameters, gradients and Adam moments are all measured BF16. Repeating microbatch two is not bitwise identical.
The small-gradient sign decomposition directly measures Adam amplification; it does not isolate which forward or
backward kernel causes each gradient difference. This remains a two-sample, three-timestep diagnostic with synthetic
guidance. All formal arms keep microbatch two.</p>%s
<h2>Cached actual sampler target versus recomputed old</h2>
<p>Guidance is fixed synthetic unit-RMS per modality. Mask and full denominator are unchanged. Replacing only the
old target velocity changes loss and accumulated parameter gradients substantially. All production ART arms retain
old-policy recompute. This control still computes old and therefore does not measure a pure cache speedup.</p>%s
<h2>Checkpoint readback: world-one engine scope</h2>
<p>After deliberately perturbing both adapters, Adam state, scheduler and all RNGs, the existing native checkpoint
manager restored them exactly. Fixed-input current and old predictions were bitwise restored. Immutable base assets
are required. This is not a world-16 trainer/controller/dataloader readback test. The current common formal
storage policy is a separate operational artifact linked below.</p>%s
<h2>Native OmniNFT GPU path</h2>
<p>Two finite independent diagnostic Adam updates completed through the native OmniNFT FSDP2 engine.
Inputs are clean estimates from a recorded visited state minus sigma times its actual sampler velocity,
fixed precomputed forward noise, and explicitly synthetic per-modality reward probabilities.
These are not final generated clean endpoints or a scientific reward batch. Both partitions start from
the same parameters/RNG; numerical equivalence is unproven. Formal arms retain microbatch two.</p>%s
<h2>Actual generated AV</h2><p>%s</p>
<p>Three base-model fixed-prompt MP4s, all 121 frames and finite 48 kHz audio decoded during export.
They are not the final eight held-out cases or an arm comparison. Browser playback/audio listening is unconfirmed.</p>
<h2>Raw evidence</h2><p>%s</p>
<p>Source SHA: <code>%s</code></p></html>''' % (
    link(native_path, 'Native sampler/Euler JSON'), link(actor_path, 'Actual actor JSON'),
    link(OUT / 'velocity_comparison.csv', 'CSV'),
    table({'loss mb2': mb['loss_mb2'], 'loss mb1': mb['loss_mb1'],
           'gradient relative L2': mb['gradient']['relative_l2'], 'gradient cosine': mb['gradient']['cosine'],
           'gradient norm ratio': mb['gradient']['norm_ratio_second_over_first'],
           'Adam update relative L2': mb['parameter_update']['relative_l2'],
           'Adam update cosine': mb['parameter_update']['cosine']}),
    table({'same-microbatch repeat gradient relative L2': precision['same_microbatch_repeat']['gradient']['relative_l2'],
           'mb2 versus mb1 gradient relative L2': precision['microbatch_partition']['gradient']['relative_l2'],
           'mb2 versus mb1 actual update relative L2': precision['microbatch_partition']['parameter_update']['relative_l2'],
           'sign flips / active elements': details['sign_flip_fraction_among_active'],
           'pooled gradient energy at sign flips': details['gradient_energy_fraction_at_sign_flips'],
           'actual update difference energy at sign flips': details['update_difference_energy_fraction_at_sign_flips'],
           'actual B update versus first FP32 Adam formula relative L2': details['actual_B_update_vs_fp32_first_Adam_formula_relative_l2']}),
    table({'recomputed target loss': cache['recompute_loss'], 'cached sampler target loss': cache['actual_sampler_target_loss'],
           'gradient relative L2': cache['parameter_gradient_difference']['relative_l2'],
           'gradient cosine': cache['parameter_gradient_difference']['cosine'],
           'gradient norm ratio': cache['parameter_gradient_difference']['norm_ratio_second_over_first'],
           'cache bytes (2 samples × 3 steps)': cache['actual_sampler_cache_tensor_bytes']}),
    table({'checkpoint bytes': checkpoint['total_bytes'], **checkpoint['checks']}),
    table({'NFT loss mb2': nft['comparison']['mb2_loss'], 'NFT loss mb1': nft['comparison']['mb1_loss'],
           'NFT gradient relative L2': nft['comparison']['gradient']['relative_l2'],
           'NFT gradient cosine': nft['comparison']['gradient']['cosine'],
           'NFT Adam update relative L2': nft['comparison']['parameter_update']['relative_l2']}),
    link(native_dir / 'media_report/index.html', 'Three generated cases and media evidence'),
    link(actor_dir / 'actor_gpu_state.json', 'Actor diagnostic state') + ' · ' +
    link(actor_dir / 'checkpoint_roundtrip.json', 'Checkpoint file hashes and readback') + ' · ' +
    link(nft_dir / 'nft_gpu_state.json', 'Native OmniNFT diagnostic') + ' · ' +
    link(precision_dir / 'precision_gpu_state.json', 'Repeatability/sign-energy diagnostic') + ' · ' +
    link(ROOT / 'assets/compact_checkpoint_storage_policy.json', 'Common checkpoint storage policy') + ' · ' +
    link(ROOT / 'logs/compact_checkpoint_semantic_config_diff.json', 'Scientific configuration identity check') + ' · ' +
    link(ROOT / 'logs/gpu_reservation_ledger.json', 'Allocated GPU cost ledger'),
    state['selected_source'])
(OUT / 'index.html').write_text(page)
evidence = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [native_path, actor_path, actor_dir / 'actor_gpu_state.json', nft_dir / 'nft_gpu_state.json',
                      precision_dir / 'precision_gpu_state.json']}
(OUT / 'inputs_sha256.json').write_text(json.dumps(evidence, indent=2) + '\n')
print(OUT / 'index.html')
