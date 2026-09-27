#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/rematch_20260922
OUT="$ROOT/results/modality_diagnostics_5090"
cd "$ROOT/code"
pid=$(cat "$OUT/runner.pid")
while kill -0 "$pid" 2>/dev/null; do
    state=$(ps -o stat= -p "$pid" || true)
    [[ "$state" == Z* ]] && break
    sleep 30
done
/root/miniconda3/bin/python -m tools.report_modality_interventions \
  --manifest "$ROOT/data/city/train/target_v2/qwen_generation_val.json" \
  --experiment-dir "$OUT" \
  --old-m2 "$ROOT/results/first_batch/m2_epoch2/predictions.jsonl" \
  --rgb "$ROOT/results/first_batch/r2_epoch2/predictions.jsonl" \
  --output-dir "$OUT/report"
