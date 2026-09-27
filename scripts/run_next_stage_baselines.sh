#!/usr/bin/env bash
set -euo pipefail
ROOT="${REMATCH_ROOT:-/root/autodl-tmp/rematch_20260922}"
OUT="$ROOT/results/next_stage_20260925"
PY=/root/miniconda3/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
DATA="$ROOT/data/city/train"
mkdir -p "$OUT/logs"
cd "$ROOT/code"
stage=initialization
trap 'rc=$?; printf "%s\t%s\tfailed:%s\n" "$(date -Is)" "$stage" "$rc" >> "$OUT/stage_status.tsv"; exit "$rc"' ERR
for pair in 'm2 trimodal' 'r2 rgb'; do
  read -r name variant <<< "$pair"
  stage="baseline_$name"
  if [[ -f "$OUT/$stage.complete" ]]; then continue; fi
  printf '%s\t%s\tstart\n' "$(date -Is)" "$stage" >> "$OUT/stage_status.tsv"
  "$PY" tools/evaluate_pretrained_grounder.py \
    --model /root/rematch_models/Qwen3-VL-8B-Instruct \
    --adapter "$ROOT/results/first_batch/$name/checkpoint-928" \
    --manifest "$DATA/target_v2/qwen3vl_native_sft/${variant}_val.json" \
    --target-manifest "$DATA/target_v2/qwen_generation_val.json" \
    --data-root "$DATA" --output-dir "$OUT/$stage" --prompt-style native \
    --max-pixels 602112 --min-pixels 200704 --max-new-tokens 128 --resume \
    > "$OUT/logs/$stage.log" 2>&1
  "$PY" - "$OUT/$stage/summary.json" <<'PY'
import json,sys
s=json.load(open(sys.argv[1]))
assert s['samples']==412, s
PY
  touch "$OUT/$stage.complete"
  printf '%s\t%s\tcomplete\n' "$(date -Is)" "$stage" >> "$OUT/stage_status.tsv"
done
printf '%s\tbaselines\tcomplete\n' "$(date -Is)" >> "$OUT/stage_status.tsv"
