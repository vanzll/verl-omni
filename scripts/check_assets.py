#!/usr/bin/env python3
"""Check the pinned HF files required by the common LTX experiment runtime."""

import json
import sys
from pathlib import Path


ROOT = Path("/m2v_intern/zhenglin/omni_art_ablation_20261016")
DESTINATIONS = {
    "MizzenAI/HPSv3": ROOT / "assets/rewards/HPSv3",
    "KlingTeam/VideoReward": ROOT / "assets/rewards/VideoReward",
    "facebook/audiobox-aesthetics": ROOT / "assets/rewards/audiobox-aesthetics",
    "laion/clap-htsat-unfused": ROOT / "assets/rewards/checkpoints/clap-htsat-unfused",
    "Qwen/Qwen2-VL-2B-Instruct": ROOT / "assets/rewards/Qwen2-VL-2B-Instruct",
    "Qwen/Qwen2-VL-7B-Instruct": ROOT / "assets/rewards/Qwen2-VL-7B-Instruct",
}


def main() -> int:
    assets = json.loads((ROOT / "assets/hf_asset_sizes.json").read_text())
    failures = []
    for asset in assets:
        repo, revision = asset["repo"], asset["revision"]
        if repo == "diffusers/LTX-2.3-Diffusers":
            root = ROOT / "cache/hf/hub/models--diffusers--LTX-2.3-Diffusers/snapshots" / revision
        else:
            root = DESTINATIONS[repo]
        complete = 0
        for item in asset["files"]:
            path = root / item["path"]
            if not path.is_file():
                failures.append(f"missing {repo}@{revision}: {item['path']}")
                continue
            size = path.stat().st_size
            if size != item["size"]:
                failures.append(f"wrong size {repo}: {item['path']} actual={size} expected={item['size']}")
                continue
            if repo != "diffusers/LTX-2.3-Diffusers":
                metadata = root / ".cache/huggingface/download" / f"{item['path']}.metadata"
                if not metadata.is_file():
                    failures.append(f"missing revision metadata {repo}: {item['path']}")
                    continue
                if metadata.open().readline().strip() != revision:
                    failures.append(f"wrong revision metadata {repo}: {item['path']}")
                    continue
            complete += 1
        print(f"{repo}@{revision}: {complete}/{len(asset['files'])} files with expected sizes")
    reward_root = ROOT / "assets/rewards"
    for path in reward_root.rglob("*.incomplete"):
        failures.append(f"unfinished reward transfer: {path}")
    model_blobs = ROOT / "cache/hf/hub/models--diffusers--LTX-2.3-Diffusers/blobs"
    for path in model_blobs.glob("*.incomplete"):
        failures.append(f"unfinished LTX transfer: {path}")
    desync = reward_root / "synchformer/synchformer_state_dict.pth"
    if not desync.is_file() or desync.stat().st_size == 0:
        failures.append(f"missing DeSync checkpoint: {desync}")
    else:
        desync_metadata = reward_root / ".cache/huggingface/download/synchformer/synchformer_state_dict.pth.metadata"
        if not desync_metadata.is_file() or desync_metadata.open().readline().strip() != "9e30061a1392d03bafdcf717e80a385ddf411b4d":
            failures.append("DeSync checkpoint revision metadata is missing or different")
    if failures:
        print(f"asset preflight failed ({len(failures)} issues):", file=sys.stderr)
        for item in failures[:30]:
            print(item, file=sys.stderr)
        if len(failures) > 30:
            print(f"... {len(failures) - 30} more", file=sys.stderr)
        return 1
    print("all pinned HF files and DeSync checkpoint present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
