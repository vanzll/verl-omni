"""Exercise real FFmpeg export and PyAV decoding using the public media contract."""
import json
from pathlib import Path

import av
import torch

from verl_omni.pipelines.rollout_media import MediaArtifact, MediaSpec
from verl_omni.utils.tracking import _export_video

root = Path('/m2v_intern/zhenglin/omni_art_ablation_20261016/logs/media_preflight')
root.mkdir(exist_ok=True)
video = torch.linspace(0, 255, 24 * 3 * 64 * 96).reshape(24, 3, 64, 96).to(torch.uint8)
audio = 0.1 * torch.sin(torch.arange(12000, dtype=torch.float32) * 2 * torch.pi * 440 / 24000)
records = []
for kind, value in [('legacy_tensor', video), ('named_artifact', MediaArtifact(
        MediaSpec('video', 'decoded', 'TCHW', fps=24), video))]:
    path = root / f'synthetic_{kind}.mp4'
    _export_video(value, str(path), fps=24, audio=audio.unsqueeze(0), audio_sample_rate=24000)
    with av.open(path) as container:
        assert {stream.type for stream in container.streams} == {'video', 'audio'}
        frames = list(container.decode(video=0))
        assert len(frames) == 24, len(frames)
        assert frames[0].to_ndarray(format='rgb24').shape == (64, 96, 3)
    with av.open(path) as container:
        decoded_audio = list(container.decode(audio=0))
        assert decoded_audio and all(torch.isfinite(torch.from_numpy(frame.to_ndarray())).all()
                                     for frame in decoded_audio)
    records.append({'kind': kind, 'video_frames': 24, 'fps': 24, 'video_shape': [64, 96, 3],
                    'audio_frames': len(decoded_audio), 'file_bytes': path.stat().st_size, 'path': str(path)})
result = {'status': 'passed_export_and_decode', 'records': records}
(root / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
