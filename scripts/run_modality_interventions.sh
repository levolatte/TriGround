#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/rematch_20260922
OUT="$ROOT/results/modality_diagnostics_5090"
DATA="$ROOT/data/city/train"
PY=/root/miniconda3/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
cd "$ROOT/code"
mkdir -p "$OUT"
for arm in normal both_shuffle both_black ir_shuffle depth_shuffle ir_black depth_black; do
    [[ -f "$OUT/$arm.complete" ]] && continue
    printf '%s\t%s\tstart\n' "$(date -Is)" "$arm" >> "$OUT/stage_status.tsv"
    "$PY" tools/evaluate_pretrained_grounder.py \
      --model /root/rematch_models/Qwen3-VL-8B-Instruct \
      --adapter "$ROOT/results/first_batch/m2/checkpoint-928" \
      --manifest "$OUT/manifests/$arm.json" \
      --target-manifest "$DATA/target_v2/qwen_generation_val.json" --data-root "$DATA" \
      --output-dir "$OUT/$arm" --prompt-style native --max-pixels 602112 --min-pixels 200704 \
      --max-new-tokens 128 --resume > "$OUT/$arm.log" 2>&1
    touch "$OUT/$arm.complete"
    printf '%s\t%s\tcomplete\n' "$(date -Is)" "$arm" >> "$OUT/stage_status.tsv"
done
printf '%s\tall\tcomplete\n' "$(date -Is)" >> "$OUT/stage_status.tsv"
