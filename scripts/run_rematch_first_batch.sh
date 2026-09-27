#!/usr/bin/env bash
set -euo pipefail
ROOT="${REMATCH_ROOT:-/root/autodl-tmp/rematch_20260922}"
PY=/root/miniconda3/bin/python
export PATH="/root/miniconda3/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export DATA_ROOT="$ROOT/data/city/train"
export QWEN_FINETUNE_DIR="$ROOT/third_party/Qwen3-VL/qwen-vl-finetune"
export MODEL_PATH="${NATIVE_MODEL_PATH:-/root/rematch_models/Qwen3-VL-8B-Instruct}"
export SEED=2026 EPOCHS=2 LEARNING_RATE=1e-5
GT="$DATA_ROOT/target_v2/qwen_generation_val.json"
SFT="$DATA_ROOT/target_v2/qwen3vl_native_sft"
OUT="$ROOT/results/first_batch"
mkdir -p "$OUT" "$ROOT/logs"
cd "$ROOT/code"
stage=initialization
trap 'rc=$?; printf "%s\t%s\tfailed:%s\n" "$(date -Is)" "$stage" "$rc" >> "$ROOT/logs/stage_status.tsv"; exit "$rc"' ERR
space_check() {
  "$PY" - "$ROOT" "$MODEL_PATH" <<'PY'
import shutil, sys
for path in sys.argv[1:]:
    free = shutil.disk_usage(path).free
    if free < 3 * 1024**3:
        raise RuntimeError(f'Less than 3 GiB free at {path}: {free}')
PY
}
run_stage() {
  stage="$1"; shift
  if [[ -f "$OUT/$stage.complete" ]]; then return; fi
  printf '%s\t%s\tstart\n' "$(date -Is)" "$stage" >> "$ROOT/logs/stage_status.tsv"
  "$@" > "$ROOT/logs/$stage.log" 2>&1
  touch "$OUT/$stage.complete"
  printf '%s\t%s\tcomplete\n' "$(date -Is)" "$stage" >> "$ROOT/logs/stage_status.tsv"
}
eval_native() {
  local manifest="$1" output="$2"; shift 2
  "$PY" tools/evaluate_pretrained_grounder.py --model "$MODEL_PATH" \
    --manifest "$manifest" --target-manifest "$GT" --data-root "$DATA_ROOT" \
    --output-dir "$OUT/$output" --prompt-style native --max-pixels 602112 \
    --min-pixels 200704 --max-new-tokens 128 --resume "$@"
}
run_stage e0_smoke "$PY" tools/evaluate_pretrained_grounder.py \
  --model "$ROOT/models/EGM-8B" --manifest "$DATA_ROOT/target_v2/expert_smoke_8.json" \
  --target-manifest "$GT" --data-root "$DATA_ROOT" --output-dir "$OUT/e0_smoke" \
  --prompt-style egm --max-pixels 602112 --min-pixels 200704 --max-new-tokens 4096 --resume
"$PY" - "$OUT/e0_smoke/summary.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
assert s['samples'] == 8 and s['parsed'] > 0, s
PY
run_stage e0 "$PY" tools/evaluate_pretrained_grounder.py \
  --model "$ROOT/models/EGM-8B" --manifest "$GT" --data-root "$DATA_ROOT" \
  --output-dir "$OUT/e0" --prompt-style egm --max-pixels 602112 --min-pixels 200704 \
  --max-new-tokens 4096 --resume
stage=waiting_native_download
printf '%s\t%s\tstart\n' "$(date -Is)" "$stage" >> "$ROOT/logs/stage_status.tsv"
while [[ ! -f "$MODEL_PATH/.download_complete" ]]; do
  if ! ps -p "$(cat "$ROOT/logs/native_download.pid")" -o stat= | grep -q '^[[:space:]]*[^Z[:space:]]'; then
    printf '%s\tnative_download\tinterrupted_requires_resume\n' "$(date -Is)" >> "$ROOT/logs/stage_status.tsv"
    exit 2
  fi
  sleep 30
done
space_check
run_stage r0 eval_native "$SFT/rgb_val.json" r0
run_stage prepare_pressure "$PY" tools/prepare_rematch_run.py --data-root "$DATA_ROOT" --output-dir "$OUT/manifests"
if [[ ! -f "$OUT/native_pressure.complete" ]]; then
  stage=native_pressure
  printf '%s\t%s\tstart\n' "$(date -Is)" "$stage" >> "$ROOT/logs/stage_status.tsv"
  if DATASET_VARIANT=trimodal ANNOTATION_PATH="$OUT/manifests/trimodal_pressure_16.json" \
     OUTPUT_DIR="$OUT/native_pressure" MAX_STEPS=2 EPOCHS=1 \
     bash scripts/run_qwen3vl_native_lora.sh > "$ROOT/logs/native_pressure.log" 2>&1; then
    touch "$OUT/native_pressure.complete"
    printf '%s\t%s\tcomplete\n' "$(date -Is)" "$stage" >> "$ROOT/logs/stage_status.tsv"
  else
    rc=$?
    if grep -qiE 'CUDA out of memory|OutOfMemoryError' "$ROOT/logs/native_pressure.log"; then
      printf '%s\tnative_pressure\tOOM_wait_L40_no_resolution_change\n' "$(date -Is)" >> "$ROOT/logs/stage_status.tsv"
      exit 3
    fi
    printf '%s\tnative_pressure\tfailed:%s\n' "$(date -Is)" "$rc" >> "$ROOT/logs/stage_status.tsv"
    exit "$rc"
  fi
fi
for repeat in 1 2; do
  run_stage "native_reload_$repeat" "$PY" tools/evaluate_pretrained_grounder.py \
    --model "$MODEL_PATH" --adapter "$OUT/native_pressure" \
    --manifest "$OUT/manifests/trimodal_pressure_8.json" \
    --target-manifest "$DATA_ROOT/target_v2/qwen_generation_train_100.json" \
    --data-root "$DATA_ROOT" --output-dir "$OUT/native_reload_$repeat" \
    --prompt-style native --max-new-tokens 128 --limit 2 --resume
done
"$PY" - "$OUT" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1])
a=[json.loads(x) for x in (p/'native_reload_1/predictions.jsonl').read_text().splitlines()]
b=[json.loads(x) for x in (p/'native_reload_2/predictions.jsonl').read_text().splitlines()]
assert [(r['id'],r['raw_text']) for r in a] == [(r['id'],r['raw_text']) for r in b]
assert len(a) == 2
PY
for variant in rgb trimodal; do
  space_check
  if [[ "$variant" == rgb ]]; then name=r2; else name=m2; fi
  run_stage "${name}_train" env DATASET_VARIANT="$variant" OUTPUT_DIR="$OUT/$name" \
    MAX_STEPS=-1 bash scripts/run_qwen3vl_native_lora.sh
  for pair in '1 464' '2 928'; do
    read -r epoch step <<< "$pair"
    run_stage "${name}_epoch${epoch}" eval_native "$SFT/${variant}_val.json" \
      "${name}_epoch${epoch}" --adapter "$OUT/$name/checkpoint-$step"
  done
done
run_stage report "$PY" tools/report_rematch_experiment.py --manifest "$GT" \
  --run "E0=$OUT/e0/predictions.jsonl" --run "R0=$OUT/r0/predictions.jsonl" \
  --run "R2_epoch1=$OUT/r2_epoch1/predictions.jsonl" --run "R2_epoch2=$OUT/r2_epoch2/predictions.jsonl" \
  --run "M2_epoch1=$OUT/m2_epoch1/predictions.jsonl" --run "M2_epoch2=$OUT/m2_epoch2/predictions.jsonl" \
  --output-dir "$OUT/report"
printf '%s\tfirst_batch\tcomplete_decision_required\n' "$(date -Is)" >> "$ROOT/logs/stage_status.tsv"
