"""Measure where actual parameter-gradient and Adam-update differences live."""
import math
import torch


def partition_details(first, second, update_first, update_second, *, lr, eps):
    assert first.keys() == second.keys()
    total_count = active_count = flip_count = zero_change_count = 0
    grad_energy = grad_error = update_error = flip_energy = flip_grad_error = 0.0
    flip_update_error = zero_update_error = 0.0
    magnitude_max = 0.0
    for name in first:
        a, b = first[name].float(), second[name].float()
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        active = (a != 0) | (b != 0)
        flip = (a * b) < 0
        zero_changed = (a == 0) != (b == 0)
        u, v = update_first[name].float(), update_second[name].float()
        energy, ge, ue = a.double().square() + b.double().square(), (a.double() - b.double()).square(), (u.double() - v.double()).square()
        total_count += a.numel(); active_count += int(active.sum())
        flip_count += int(flip.sum()); zero_change_count += int(zero_changed.sum())
        grad_energy += float(energy.sum()); grad_error += float(ge.sum()); update_error += float(ue.sum())
        flip_energy += float(energy[flip].sum()); flip_grad_error += float(ge[flip].sum())
        flip_update_error += float(ue[flip].sum()); zero_update_error += float(ue[zero_changed].sum())
        magnitude_max = max(magnitude_max, float(a.abs().max()), float(b.abs().max()))
    rms = math.sqrt(grad_energy / max(2 * active_count, 1))
    edges = [0.0, .01 * rms, .1 * rms, rms, 10 * rms, math.inf]
    bins = [{'lower_rms_multiple': lo, 'upper_rms_multiple': hi, 'count': 0,
             'sign_flip_count': 0, 'gradient_energy': 0.0, 'update_error_energy': 0.0}
            for lo, hi in zip([0, .01, .1, 1, 10], [.01, .1, 1, 10, 'inf'])]
    analytic_error = analytic_energy = 0.0
    for name in first:
        a, b = first[name].float(), second[name].float()
        mag = torch.maximum(a.abs(), b.abs())
        active, flip = mag > 0, a * b < 0
        u, v = update_first[name].float(), update_second[name].float()
        energy = a.double().square() + b.double().square()
        error = (u.double() - v.double()).square()
        for row, lo, hi in zip(bins, edges, edges[1:]):
            selected = active & (mag >= lo) & (mag < hi)
            row['count'] += int(selected.sum()); row['sign_flip_count'] += int((selected & flip).sum())
            row['gradient_energy'] += float(energy[selected].sum())
            row['update_error_energy'] += float(error[selected].sum())
        if 'lora_B' in name:
            # Initial B is exactly zero: AdamW decay contributes zero here.
            predicted = -lr * a / (a.abs() + eps)
            analytic_error += float((u.double() - predicted.double()).square().sum())
            analytic_energy += float(predicted.double().square().sum())
    for row in bins:
        row['gradient_energy_fraction'] = row.pop('gradient_energy') / max(grad_energy, 1e-30)
        row['update_error_energy_fraction'] = row.pop('update_error_energy') / max(update_error, 1e-30)
    return {'total_parameter_elements': total_count, 'nonzero_in_either_gradient': active_count,
            'sign_flip_count': flip_count, 'zero_transition_count': zero_change_count,
            'sign_flip_fraction_among_active': flip_count / max(active_count, 1),
            'gradient_energy_fraction_at_sign_flips': flip_energy / max(grad_energy, 1e-30),
            'gradient_difference_energy_fraction_at_sign_flips': flip_grad_error / max(grad_error, 1e-30),
            'update_difference_energy_fraction_at_sign_flips': flip_update_error / max(update_error, 1e-30),
            'update_difference_energy_fraction_at_zero_transitions': zero_update_error / max(update_error, 1e-30),
            'active_gradient_rms': rms, 'gradient_abs_max': magnitude_max, 'magnitude_bins': bins,
            'actual_B_update_vs_fp32_first_Adam_formula_relative_l2': math.sqrt(analytic_error / max(analytic_energy, 1e-30)),
            'formula_scope': 'first update, initially zero B; real optimizer uses BF16 state/parameter arithmetic'}
