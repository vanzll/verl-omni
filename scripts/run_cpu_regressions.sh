#!/usr/bin/env bash
set -euo pipefail
source /m2v_intern2/zhenglin/my_bashrc.sh
campaign=/m2v_intern/zhenglin/omni_art_ablation_20261016
export OMNI_RUNTIME_CUDA_VARIANT=cu129 CUDA_VISIBLE_DEVICES=""
source "$campaign/scripts/runtime_env.sh"
repo=${OMNI_CPU_SOURCE_DIR:-$campaign/source}
[[ "$repo" == "$campaign/source" || "$repo" == "$campaign/snapshots/"*/runtime/source ]] || exit 2
test_plugin="$campaign/test_tools/pytest_asyncio_1_4_0"
[[ -f "$test_plugin/pytest_asyncio/plugin.py" ]] || exit 2
export PYTHONPATH="$repo:$test_plugin" PYTHONDONTWRITEBYTECODE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export TMPDIR="$campaign/tmp" XDG_CACHE_HOME="$campaign/cache/xdg"
export HF_HOME="$campaign/cache/hf" HF_HUB_CACHE="$campaign/cache/hf/hub"
export TORCH_EXTENSIONS_DIR="$campaign/cache/torch_extensions" TRITON_CACHE_DIR="$campaign/cache/triton"
export VLLM_CACHE_ROOT="$campaign/cache/vllm" TORCH_COMPILE_DISABLE=1 TORCHINDUCTOR_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 WANDB_MODE=disabled
cd "$repo"
timeout --kill-after=15 600 "$campaign/env/bin/python" -m pytest -q -ra -p no:cacheprovider -p pytest_asyncio.plugin --tb=short --maxfail=5 \
  --junitxml="$campaign/logs/cpu_runtime_results.xml" \
  tests/trainer/diffusion/test_complete_rollout_checkpoint_on_cpu.py \
  tests/trainer/diffusion/test_diffusionart_math_on_cpu.py \
  tests/trainer/diffusion/test_diffusionart_golden_on_cpu.py \
  tests/trainer/diffusion/test_diffusionart_on_cpu.py \
  tests/trainer/diffusion/test_art_dataproto_dispatch_on_cpu.py \
  tests/trainer/diffusion/test_diffusion_core_algos_on_cpu.py \
  tests/trainer/diffusion/test_diffusion_registry_on_cpu.py \
  tests/workers/engine/test_omni_nft_engine_on_cpu.py \
  tests/workers/engine/test_diffusion_worker_dispatch_on_cpu.py \
  tests/workers/rollout/rollout_vllm/test_vllm_omni_strategy_on_cpu.py \
  tests/workers/engine/test_diffusion_gradient_sync_on_cpu.py \
  tests/workers/engine/fsdp/test_art_engine_loss_inputs_on_cpu.py \
  tests/pipelines/test_ltx2_omni_nft_adapter_on_cpu.py \
  tests/pipelines/test_ltx2_diffusion_art_adapter_on_cpu.py \
  tests/pipelines/test_ltx2_flow_grpo_on_cpu.py \
  tests/pipelines/test_ltx2_output_type_on_cpu.py \
  tests/pipelines/test_boogu_image_diffusion_nft_engine_on_cpu.py
