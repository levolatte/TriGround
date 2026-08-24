#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data/aic}"
export DATA_ROOT
export HF_HOME="${HF_HOME:-${DATA_ROOT}/huggingface}"

bash scripts/setup_gpu.sh
source .venv/bin/activate
python tools/download_model.py \
  --repo-id Qwen/Qwen3-VL-2B-Instruct \
  --cache-dir "${HF_HOME}/hub"

echo "Upload city_detection_prepared beside the repository, then run:"
echo "python tools/preflight.py --config configs/multimodal.yaml --offline"
echo "python tools/preflight.py --config configs/multimodal.yaml --device cuda --backward"
