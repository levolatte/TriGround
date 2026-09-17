#!/usr/bin/env bash
# Run from any directory; all data/model/run paths are relative to this checkout.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHON="${PYTHON:-python}"
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export BACKBONE="${BACKBONE:-../models/Qwen3-VL-8B-Instruct}"
stamp="$(date +%Y%m%d-%H%M%S)"
export CONFIG_DIR="configs"
export RUN_ROOT="runs/cloud-smoke-$stamp"
bash scripts/qwen3_vl_8b_smoke.slurm
