"""Compare separately observed sampler velocities and Euler reconstructions.

No equality claim about actor recomputation or parameter gradients is inferred
from a scheduler-only capture. Native BF16 rounding is reported separately.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from ltx_velocity_capture import ROOT


def compare(reference, candidate):
    if reference.shape != candidate.shape or reference.numel() == 0:
        raise ValueError('Velocity tensors must have the same nonempty shape')
    reference, candidate = reference.double().reshape(-1), candidate.double().reshape(-1)
    if not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
        raise ValueError('Nonfinite probe input')
    error = candidate - reference
    rms = float(reference.square().mean().sqrt())
    candidate_rms = float(candidate.square().mean().sqrt())
    error_rms = float(error.square().mean().sqrt())
    denominator = float(reference.norm() * candidate.norm())
    return {'reference_rms': rms, 'candidate_rms': candidate_rms,
            'mse': float(error.square().mean()), 'error_rms': error_rms,
            'relative_error_rms': error_rms / rms if rms else None,
            'maximum_absolute_error': float(error.abs().max()),
            'cosine': float(torch.dot(reference, candidate)) / denominator if denominator else None,
            'elements': reference.numel()}


def analyze(path):
    path = path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Probe evidence must be under the campaign root')
    metadata = json.loads(path.with_suffix(path.suffix + '.json').read_text())
    with path.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    if digest != metadata['sha256']:
        raise ValueError('Capture hash mismatch')
    capture = torch.load(path, map_location='cpu', weights_only=True)
    if capture['format_version'] != 1 or capture['identity'] != metadata['identity']:
        raise ValueError('Unexpected capture format or input identity')
    report = {'identity': capture['identity'], 'capture_sha256': digest,
              'scope': capture['scope'], 'records': [],
              'actor_recomputed_old': 'not measured', 'target_loss_parameter_gradient_equivalence': 'not measured',
              'old_forward_timing': 'not measured', 'cache_transfer_timing': 'not measured',
              'capture_and_step_timing_note': 'Includes model/scheduler work and CPU copies; not an efficiency benchmark',
              'stored_tensor_bytes': capture['stored_tensor_bytes']}
    for row in capture['records']:
        ds = row['sigma_next'].float() - row['sigma'].float()
        if ds.numel() != 1 or not torch.isfinite(ds).all() or float(ds) == 0:
            raise ValueError('Expected one finite nonzero Euler sigma interval')
        actual = row['actual_sampler_velocity'].float()
        unrounded = (row['scheduler_next_unrounded'].float() - row['state'].float()) / ds
        rounded = (row['actual_next_rounded'].float() - row['state'].float()) / ds
        if actual.ndim != 3 or actual.shape[1] != row['video_seq_len'] + row['audio_seq_len']:
            raise ValueError('Unexpected unified latent layout')
        result = {'index': row['index'], 'sigma': float(row['sigma']),
                  'sigma_next': float(row['sigma_next']),
                  'sampler_velocity_tensor_bytes': row['actual_sampler_velocity'].numel()
                  * row['actual_sampler_velocity'].element_size(),
                  'capture_and_step_wall_seconds': row['capture_and_step_wall_seconds'], 'modalities': {}}
        split = row['video_seq_len']
        for name, section in [('video', slice(0, split)), ('audio', slice(split, None))]:
            result['modalities'][name] = {
                'sampler_vs_unrounded_euler': compare(actual[:, section], unrounded[:, section]),
                'sampler_vs_rounded_euler': compare(actual[:, section], rounded[:, section]),
            }
        report['records'].append(result)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('capture', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT) or args.output.exists():
        raise ValueError('Use a fresh report path under the campaign root')
    result = analyze(args.capture)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'identity': result['identity'], 'records': len(result['records']),
                      'actor_recomputed_old': result['actor_recomputed_old'], 'output': str(args.output)}))
