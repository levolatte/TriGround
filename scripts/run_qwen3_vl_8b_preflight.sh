#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo"

python_bin="${PYTHON:-python}"
"$python_bin" tools/preflight.py \
  --config configs/qwen3_vl_8b_stage1a_ir.yaml \
  --device cuda \
  --max-samples 64 \
  --optimizer-steps 2
