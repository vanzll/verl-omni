"""Decode actual held-out AV, retain hashes/raw scores and render relative HTML."""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import statistics
import time

import av
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')
WEIGHTS = {'video_align': 1., 'hpsv3': 1.5, 'audiobox': .5, 'clap': 1., 'desync': 1.}


def decode(path):
    pixels = hashlib.sha256()
    thumbnails, shapes, count = {}, set(), 0
    previous = None
    frame_brightness, frame_changes, duplicate_pairs = [], [], 0
    with av.open(path) as container:
        assert len(container.streams.video) == len(container.streams.audio) == 1
        video = container.streams.video[0]
        assert float(video.average_rate) == 24.
        for index, frame in enumerate(container.decode(video=0)):
            array = frame.to_ndarray(format='rgb24')
            shapes.add(array.shape)
            pixels.update(array.tobytes())
            frame_brightness.append(float(array.mean()) / 255.)
            if previous is not None:
                frame_changes.append(float(np.abs(array.astype(np.int16) - previous).mean()) / 255.)
                duplicate_pairs += int(np.array_equal(array, previous))
            previous = array.astype(np.int16)
            if index in (0, 60, 120):
                thumbnails[index] = Image.fromarray(array).resize((256, 171))
            count += 1
    assert count == 121 and shapes == {(512, 768, 3)}, (count, shapes)
    audio_hash, samples, energy, peak = hashlib.sha256(), 0, 0., 0.
    with av.open(path) as container:
        stream = container.streams.audio[0]
        assert stream.sample_rate == 48000 and stream.codec_context.channels == 2
        for frame in container.decode(audio=0):
            array = frame.to_ndarray()
            assert np.isfinite(array).all()
            audio_hash.update(array.tobytes())
            samples += frame.samples
            energy += float(np.square(array.astype(np.float64)).sum())
            peak = max(peak, float(np.abs(array).max()))
    assert samples > 0
    assert len(frame_changes) == 120 and len(frame_brightness) == 121
    assert np.isfinite(frame_changes).all() and np.isfinite(frame_brightness).all()
    return {'video_frames': count, 'fps': 24, 'height': 512, 'width': 768,
            'decoded_pixel_sha256': pixels.hexdigest(), 'audio_rate': 48000, 'audio_channels': 2,
            'audio_samples_per_channel_including_codec_padding': samples,
            'audio_decoded_seconds': samples / 48000,
            'decoded_audio_sha256': audio_hash.hexdigest(), 'audio_finite': True,
            'audio_rms': (energy / (2 * samples)) ** .5, 'audio_peak_abs': peak,
            'frame_mean_rgb_0_1': frame_brightness,
            'adjacent_frame_mean_abs_rgb_diff_0_1': frame_changes,
            'temporal_mean_abs_rgb_diff_0_1': float(np.mean(frame_changes)),
            'temporal_std_abs_rgb_diff_0_1': float(np.std(frame_changes)),
            'exact_duplicate_adjacent_frame_pairs': duplicate_pairs,
            'descriptive_media_metrics_scope': 'Decoded RGB changes and audio RMS/peak only. Camera motion, object motion and flicker are not distinguished; these are not semantic quality or synchronization scores.'}, thumbnails


def audit(manifests, rollout, output):
    assert output.resolve().is_relative_to(ROOT / 'analysis')
    output.mkdir(parents=True, exist_ok=True)
    cached, records, inputs = {}, [], {}
    sheet = Image.new('RGB', (768, 8 * 196), 'white')
    draw = ImageDraw.Draw(sheet)
    row_inputs, raw_scores = {}, {}
    for manifest_file in manifests:
        manifest_file = manifest_file.resolve()
        assert manifest_file.is_relative_to(ROOT / 'attempts')
        manifest = json.loads(manifest_file.read_text())
        assert manifest['variant'] not in row_inputs, 'select one continuous attempt per arm; compare cohorts separately'
        config_path = manifest_file.parent / 'analysis/wandb/config.json'
        observation = json.loads((config_path.parent / 'observation.json').read_text())
        assert observation['status'] == 'fetched' and observation['identity_verified'] is True
        config = json.loads(config_path.read_text())
        metadata = Path(config['trainer']['validation_data_dir']) / f'{rollout}.jsonl'
        assert metadata.resolve().is_relative_to(ROOT)
        values = [json.loads(line) for line in metadata.read_text().splitlines() if line.strip()]
        assert len(values) == 8 and all(row['step'] == rollout for row in values)
        inputs[str(metadata.resolve().relative_to(ROOT))] = hashlib.sha256(metadata.read_bytes()).hexdigest()
        row_inputs[manifest['variant']] = [row['input'] for row in values]
        raw_scores[manifest['variant']] = [{k: row['reward/' + k] for k in WEIGHTS} for row in values]
        for index, row in enumerate(values):
            scores = {key: float(row['reward/' + key]) for key in WEIGHTS}
            assert all(np.isfinite(score) for score in scores.values())
            path = Path(row['output']).resolve()
            assert path.is_relative_to(ROOT) and path.is_file()
            with path.open('rb') as stream:
                checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
            reused = checksum in cached
            if not reused:
                verified, thumbnails = decode(path)
                cached[checksum] = verified
                if len(records) < 8:
                    for column, frame in enumerate((0, 60, 120)):
                        sheet.paste(thumbnails[frame], (column * 256, index * 196 + 24))
                        draw.text((column * 256 + 3, index * 196 + 4), f'case {index} / frame {frame}', fill='black')
            record = {'variant': manifest['variant'], 'attempt': manifest['attempt'], 'case': index,
                      'rollout': rollout, 'input': row['input'], 'path': str(path), 'bytes': path.stat().st_size,
                      'file_sha256': checksum, 'raw_scores': scores,
                      'once_counted_weighted_total': sum(WEIGHTS[k] * scores[k] for k in WEIGHTS),
                      'decode_reused_for_byte_identical_file': reused, **cached[checksum]}
            records.append(record)
            print('VALIDATED_AV', manifest['variant'], rollout, index, checksum, flush=True)
    sheet.save(output / 'contact_sheet.png')
    variants = list(row_inputs)
    comparison = {'inputs_identical': all(row_inputs[v] == row_inputs[variants[0]] for v in variants),
                  'all_raw_reward_components_identical': all(raw_scores[v] == raw_scores[variants[0]] for v in variants),
                  'all_media_files_byte_identical_by_case': all(
                      next(r['file_sha256'] for r in records if r['variant'] == v and r['case'] == i)
                      == next(r['file_sha256'] for r in records if r['variant'] == variants[0] and r['case'] == i)
                      for v in variants for i in range(8))}
    report = {'status': 'passed_all_media_integrity_and_decode_checks', 'observed_unix': time.time(),
              'scope': 'Actual held-out AV exports; unique content decoded in full. Browser playback and audio listening not verified.',
              'rollout': rollout, 'export_count': len(records), 'unique_full_decodes': len(cached),
              'input_sha256': inputs, 'weights': WEIGHTS, 'comparison': comparison, 'records': records,
              'weighted_total_means': {v: statistics.mean(r['once_counted_weighted_total'] for r in records if r['variant'] == v) for v in variants}}
    (output / 'media_evidence.json').write_text(json.dumps(report, indent=2) + '\n')
    page = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Actual held-out LTX AV</title>',
            '<style>body{font:16px system-ui;max-width:1150px;margin:30px auto;padding:0 18px}img{max-width:100%}video{max-width:49%}section{margin:25px 0}</style>',
            f'<h1>Actual eight-case held-out AV — rollout {rollout}</h1>',
            '<p>Single-seed matched public-data comparison. All unique files fully decoded; browser playback/audio listening unverified. These exports alone do not establish the final experiment result.</p>',
            '<p><a href="media_evidence.json">Raw scores, hashes and decode evidence</a></p>',
            '<img src="contact_sheet.png" alt="Cases 0–7, first selected arm, frames 0/60/120">']
    for index in range(8):
        selected = [record for record in records if record['case'] == index]
        page.append(f'<section><h2>Case {index}</h2><p>{html.escape(str(selected[0]["input"]))}</p>')
        for record in selected:
            relative = html.escape(os.path.relpath(record['path'], output), quote=True)
            page.append(f'<p>{record["variant"]} · weighted total {record["once_counted_weighted_total"]:.6g} · adjacent RGB change {record["temporal_mean_abs_rgb_diff_0_1"]:.5g} · audio RMS {record["audio_rms"]:.5g} · audio peak {record["audio_peak_abs"]:.5g}</p><video controls preload="metadata" src="{relative}"></video>')
        page.append('</section>')
    page.append('</html>')
    (output / 'index.html').write_text('\n'.join(page) + '\n')
    print(json.dumps({'report': str(output / 'index.html'), 'comparison': comparison, 'weighted_total_means': report['weighted_total_means']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifests', nargs='+', type=Path)
    parser.add_argument('--rollout', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.manifests, args.rollout, args.output)
