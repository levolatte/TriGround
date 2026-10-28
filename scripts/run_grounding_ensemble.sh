#!/usr/bin/env bash
# Sequential Qwen multimodal + EGM RGB inference; each process releases its GPU.
# Required: QWEN_MODEL EGM_MODEL QUERIES DATA_ROOT OUTPUT_DIR.
# Optional GT is a normalized id->bbox/record JSON map for local evaluation.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
: "${QWEN_MODEL:?Set QWEN_MODEL}"
: "${EGM_MODEL:?Set EGM_MODEL}"
: "${QUERIES:?Set QUERIES}"
: "${DATA_ROOT:?Set DATA_ROOT}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"
PY="${PY:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"

qwen=("${PY}" tools/generate_grounding_candidates.py
  --model "${QWEN_MODEL}" --profile qwen --modality "${QWEN_MODALITY:-all}"
  --queries "${QUERIES}" --data-root "${DATA_ROOT}" --output-dir "${OUTPUT_DIR}/qwen"
  --max-pixels "${QWEN_MAX_PIXELS:-2073600}" --upscale "${QWEN_UPSCALE:-1}"
  --model-max-length "${QWEN_MAX_LENGTH:-16384}" --max-new-tokens 128
  --max-num-seqs "${MAX_NUM_SEQS:-16}" --chunk-size "${CHUNK_SIZE:-32}"
  --num-samples "${NUM_SAMPLES:-1}" --temperature "${TEMPERATURE:-0}")
if [[ -n "${QWEN_ADAPTER:-}" ]]; then
  qwen+=(--adapter "${QWEN_ADAPTER}" --max-lora-rank "${MAX_LORA_RANK:-32}")
fi
egm=("${PY}" tools/generate_grounding_candidates.py
  --model "${EGM_MODEL}" --profile egm --modality rgb
  --queries "${QUERIES}" --data-root "${DATA_ROOT}" --output-dir "${OUTPUT_DIR}/egm"
  --max-pixels "${EGM_MAX_PIXELS:-2073600}" --max-new-tokens "${EGM_MAX_NEW_TOKENS:-4096}"
  --model-max-length "${EGM_MAX_LENGTH:-16384}"
  --max-num-seqs "${MAX_NUM_SEQS:-16}" --chunk-size "${CHUNK_SIZE:-32}"
  --num-samples "${NUM_SAMPLES:-1}" --temperature "${TEMPERATURE:-0}")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${qwen[@]}"; printf '\n'
  printf '%q ' "${egm[@]}"; printf '\n'
  exit 0
fi
"${qwen[@]}"
"${egm[@]}"
"${PY}" - "${OUTPUT_DIR}/members.json" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"members": [
    {"name": "qwen", "path": "qwen/candidates.jsonl", "family": "qwen", "weight": 1},
    {"name": "egm", "path": "egm/candidates.jsonl", "family": "egm", "weight": 1},
]}, indent=2) + "\n", encoding="utf-8")
PY
aggregate=("${PY}" tools/ensemble_grounding.py --config "${OUTPUT_DIR}/members.json"
  --method "${METHOD:-wbf}" --iou-threshold "${IOU_THRESHOLD:-0.55}"
  --out-dir "${OUTPUT_DIR}/ensemble")
if [[ -n "${GT:-}" ]]; then
  aggregate+=(--gt "${GT}" --compare-all --baseline qwen)
fi
"${aggregate[@]}"
