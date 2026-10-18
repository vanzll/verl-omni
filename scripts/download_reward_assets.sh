#!/usr/bin/env bash
set -euo pipefail
source /m2v_intern2/zhenglin/my_bashrc.sh

campaign=/m2v_intern/zhenglin/omni_art_ablation_20261016
export HF_HOME="$campaign/cache/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export TMPDIR="$campaign/tmp"
export HF_HUB_ETAG_TIMEOUT=20
export HF_HUB_DOWNLOAD_TIMEOUT=120
reward_root="$campaign/assets/rewards"
mkdir -p "$reward_root" "$TMPDIR" "$HF_HUB_CACHE" "$HF_XET_CACHE"

download_repo() {
  local repo="$1" revision="$2" destination="$3"
  printf '[%s] START %s@%s\n' "$(date -u +%FT%TZ)" "$repo" "$revision"
  hf download "$repo" --revision "$revision" --local-dir "$reward_root/$destination"
  printf '[%s] DONE %s\n' "$(date -u +%FT%TZ)" "$repo"
}

download_repo KlingTeam/VideoReward 4f26600130683e6f1de9f5d463887f28e8ef995c VideoReward
download_repo MizzenAI/HPSv3 4f81e3e09edd82fe3c5f636444c721b592a735ca HPSv3
download_repo facebook/audiobox-aesthetics 9b1dd8e5df9af7216e836a98974fe3b82c56ded6 audiobox-aesthetics
download_repo laion/clap-htsat-unfused 8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a checkpoints/clap-htsat-unfused
download_repo Qwen/Qwen2-VL-2B-Instruct 895c3a49bc3fa70a340399125c650a463535e71c Qwen2-VL-2B-Instruct
download_repo Qwen/Qwen2-VL-7B-Instruct eed13092ef92e448dd6875b2a00151bd3f7db0ac Qwen2-VL-7B-Instruct
printf '[%s] START Synchformer checkpoint\n' "$(date -u +%FT%TZ)"
hf download zghhui/OmniNFT-Reward-Series synchformer/synchformer_state_dict.pth \
  --revision 9e30061a1392d03bafdcf717e80a385ddf411b4d \
  --local-dir "$reward_root"
printf '[%s] DONE all reward downloads\n' "$(date -u +%FT%TZ)"
