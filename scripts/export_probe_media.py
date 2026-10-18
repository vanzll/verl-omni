"""Export every completed native generation probe case and verify real AV decode."""
import hashlib
import html
import json
import os
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw
import torch

from verl_omni.pipelines.rollout_media import MediaArtifact, MediaSpec
from verl_omni.utils.tracking import _export_video

ROOT = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016')
descriptor = json.loads((ROOT / 'logs/current_ltx_rollout_probe.json').read_text())
output = Path(descriptor['output'])
state = json.loads((output / 'gpu_generation_state.json').read_text())
assert state['stage'] == 'passed_generation' and len(state['requests']) == 3
report_dir = output / 'media_report'
report_dir.mkdir(exist_ok=True)
sheet = Image.new('RGB', (3 * 256, 3 * 196), 'white')
draw = ImageDraw.Draw(sheet)
records = []
sections = []
for row, request in enumerate(state['requests']):
    source = Path(request['artifact'])
    with source.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == request['sha256']
    data = torch.load(source, map_location='cpu', weights_only=True)['artifacts']
    video, audio = data['video_preview'], data['audio'].float()
    assert list(video.shape) == [121, 3, request['height'], request['width']]
    assert torch.isfinite(audio).all()
    target = report_dir / (request['name'] + '.mp4')
    assert not target.exists()
    artifact = MediaArtifact(MediaSpec('video', 'decoded', 'TCHW', fps=24), video)
    _export_video(artifact, str(target), fps=24, audio=audio, audio_sample_rate=48000)
    with av.open(target) as container:
        assert {s.type for s in container.streams} == {'video', 'audio'}
        frames = list(container.decode(video=0))
        assert len(frames) == 121
        assert frames[0].to_ndarray(format='rgb24').shape == (request['height'], request['width'], 3)
    with av.open(target) as container:
        audio_stream = container.streams.audio[0]
        decoded = list(container.decode(audio=0))
        assert decoded and audio_stream.sample_rate == 48000
        assert all(np.isfinite(f.to_ndarray()).all() for f in decoded)
    for column, index in enumerate((0, 60, 120)):
        image = Image.fromarray(video[index].permute(1, 2, 0).numpy()).resize((256, 171))
        sheet.paste(image, (column * 256, row * 196 + 24))
        draw.text((column * 256 + 3, row * 196 + 4), f"{request['name']} / frame {index}", fill='black')
    record = {'name': request['name'], 'frames': 121, 'fps': 24, 'audio_rate': 48000,
              'audio_source_samples': audio.shape[-1], 'audio_rms': float(audio.square().mean().sqrt()),
              'audio_max_abs': float(audio.abs().max()), 'pixel_mean': float(video.float().mean()),
              'pixel_std': float(video.float().std()), 'file_bytes': target.stat().st_size,
              'source_sha256': request['sha256'], 'path': str(target)}
    records.append(record)
    sections.append(f'<section><h2>{html.escape(request["name"])}</h2><p>{request["width"]}×{request["height"]}, 121 frames, 24 FPS</p>'
                    f'<video controls preload="metadata" width="768" src="{html.escape(target.name)}"></video></section>')
    print('REAL_MEDIA_VERIFIED', json.dumps(record), flush=True)
sheet.save(report_dir / 'contact_sheet.png')
(report_dir / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>LTX independent GPU probe</title>'
    '<h1>LTX 独立 GPU 预检样例</h1><p>基模、固定诊断提示词、完整三个 case。正式四组实验尚无有效更新。</p>'
    '<p>已验证 H.264/AAC 导出及 PyAV 音视频解码；未宣称浏览器播放验收。</p>'
    '<img src="contact_sheet.png" width="768">' + ''.join(sections), encoding='utf-8')
(report_dir / 'media_evidence.json').write_text(json.dumps({'status': 'passed_all_three_real_av_exports_and_decodes',
    'scope': 'Independent base-model GPU preflight, no scientific training/eval result', 'records': records}, indent=2) + '\n')
print('REPORT', str(report_dir / 'index.html'), flush=True)
