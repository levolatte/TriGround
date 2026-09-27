#!/usr/bin/env bash
set -euo pipefail
ROOT="${REMATCH_ROOT:-/root/autodl-tmp/rematch_20260922}"
OUT="$ROOT/results/next_stage_20260925"
PY=/root/miniconda3/bin/python
export PATH="/root/miniconda3/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export DATA_ROOT="$ROOT/data/city/train"
export MODEL_PATH=/root/rematch_models/Qwen3-VL-8B-Instruct
export QWEN_FINETUNE_DIR="$ROOT/third_party/Qwen3-VL/qwen-vl-finetune"
export SEED=2026 DATASET_VARIANT=trimodal EPOCHS=1
GT="$DATA_ROOT/target_v2/qwen_generation_val.json"
SFT="$DATA_ROOT/target_v2/qwen3vl_native_sft"
M2="$ROOT/results/first_batch/m2/checkpoint-928"
mkdir -p "$OUT/logs" "$OUT/manifests"
cd "$ROOT/code"
stage=wait_baselines
trap 'rc=$?; printf "%s\t%s\tfailed:%s\n" "$(date -Is)" "$stage" "$rc" >> "$OUT/stage_status.tsv"; exit "$rc"' ERR
while [[ ! -f "$OUT/baseline_r2.complete" ]]; do
  pid=$(cat "$OUT/baseline_runner.pid")
  if ! ps -p "$pid" -o stat= | grep -q '^[[:space:]]*[^Z[:space:]]'; then
    echo 'Baseline runner exited without completing both baselines' >&2
    exit 2
  fi
  sleep 15
done

run_stage() {
  stage="$1"; shift
  if [[ -f "$OUT/$stage.complete" ]]; then return; fi
  printf '%s\t%s\tstart\n' "$(date -Is)" "$stage" >> "$OUT/stage_status.tsv"
  local attempt log rc
  for attempt in 1 2 3; do
    log="$OUT/logs/${stage}_$(date +%Y%m%dT%H%M%S)_attempt${attempt}.log"
    if "$@" > "$log" 2>&1; then
      touch "$OUT/$stage.complete"
      printf '%s\t%s\tcomplete\n' "$(date -Is)" "$stage" >> "$OUT/stage_status.tsv"
      return
    else rc=$?; fi
    # Only concrete transient transport failures qualify. Numerical/data/interface errors stop.
    if [[ "$attempt" == 3 ]] || ! grep -qE 'ConnectionResetError|ConnectionAbortedError|Temporary failure in name resolution|\[Errno 110\] Connection timed out' "$log"; then
      tail -n 30 "$log" >&2
      return "$rc"
    fi
    sleep "$((attempt * 60))"
  done
}

train_segment() {
  local output="$1" manifest="$2" total="$3" lr="$4" initial="$5" save="$6" stop="${7:-}"
  local resume
  resume=$("$PY" - "$output" <<'PY'
from pathlib import Path
import sys
paths=sorted(Path(sys.argv[1]).glob('checkpoint-*'), key=lambda p:int(p.name.split('-')[-1]))
print(paths[-1] if paths else '')
PY
)
  "$PY" - "$OUT" <<'PY'
import shutil,sys
assert shutil.disk_usage(sys.argv[1]).free >= 3*1024**3, 'Less than 3 GiB free'
PY
  if [[ -n "$resume" ]]; then initial=''; fi
  OUTPUT_DIR="$output" ANNOTATION_PATH="$manifest" MAX_STEPS="$total" \
    LEARNING_RATE="$lr" INIT_ADAPTER="$initial" RESUME_FROM_CHECKPOINT="$resume" \
    SAVE_STEPS="$save" STOP_AFTER_STEP="$stop" bash scripts/run_qwen3vl_native_lora.sh
}

eval_model() {
  local adapter="$1" output="$2" manifest="${3:-$SFT/trimodal_val.json}" target="${4:-$GT}" limit="${5:-0}"
  "$PY" tools/evaluate_pretrained_grounder.py --model "$MODEL_PATH" --adapter "$adapter" \
    --manifest "$manifest" --target-manifest "$target" --data-root "$DATA_ROOT" \
    --output-dir "$OUT/$output" --prompt-style native --max-pixels 602112 \
    --min-pixels 200704 --max-new-tokens 128 --resume --limit "$limit"
}

run_stage pressure_manifest "$PY" tools/prepare_rematch_run.py --data-root "$DATA_ROOT" --output-dir "$OUT/manifests"
run_stage init_smoke_step1 train_segment "$OUT/init_smoke" "$OUT/manifests/trimodal_pressure_16.json" 2 5e-6 "$M2" 1 1
run_stage init_smoke_resume2 train_segment "$OUT/init_smoke" "$OUT/manifests/trimodal_pressure_16.json" 2 5e-6 "$M2" 1
for repeat in 1 2; do
  run_stage "init_reload_$repeat" eval_model "$OUT/init_smoke/checkpoint-2" "init_reload_$repeat" \
    "$OUT/manifests/trimodal_pressure_8.json" "$DATA_ROOT/target_v2/qwen_generation_train_100.json" 2
done
run_stage init_smoke_verify "$PY" - "$OUT" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
state=json.loads((p/'init_smoke/checkpoint-2/trainer_state.json').read_text())
assert state['global_step']==2, state
a=[json.loads(x) for x in (p/'init_reload_1/predictions.jsonl').read_text().splitlines()]
b=[json.loads(x) for x in (p/'init_reload_2/predictions.jsonl').read_text().splitlines()]
assert len(a)==len(b)==2
assert [(r['id'],r['raw_text'],r['prediction']) for r in a] == [(r['id'],r['raw_text'],r['prediction']) for r in b]
PY

for phase in 1 2; do
  run_stage "c_manifest_$phase" "$PY" -m tools.prepare_next_stage_data --city-native "$SFT/trimodal_train.json" \
    --city-root "$DATA_ROOT" --output-dir "$OUT/manifests" --branch C --phase "$phase" --seed 2026
done
# MAX_STEPS remains 1000 across the midpoint pause, preserving the original LR schedule.
run_stage c_train_500 train_segment "$OUT/c_phase1" "$OUT/manifests/c_phase1.json" 1000 5e-6 "$M2" 500 500
run_stage c_eval_500 eval_model "$OUT/c_phase1/checkpoint-500" c_step500
run_stage c_train_1000 train_segment "$OUT/c_phase1" "$OUT/manifests/c_phase1.json" 1000 5e-6 "$M2" 500
run_stage c_eval_1000 eval_model "$OUT/c_phase1/checkpoint-1000" c_step1000
run_stage c_train_1500 train_segment "$OUT/c_phase2" "$OUT/manifests/c_phase2.json" 500 3e-6 "$OUT/c_phase1/checkpoint-1000" 500
run_stage c_eval_1500 eval_model "$OUT/c_phase2/checkpoint-500" c_step1500
run_stage c_report "$PY" tools/report_rematch_experiment.py --manifest "$GT" \
  --run "M2=$OUT/baseline_m2/predictions.jsonl" --run "R2=$OUT/baseline_r2/predictions.jsonl" \
  --run "C500=$OUT/c_step500/predictions.jsonl" --run "C1000=$OUT/c_step1000/predictions.jsonl" \
  --run "C1500=$OUT/c_step1500/predictions.jsonl" --output-dir "$OUT/c_report"
printf '%s\tcontrol_pipeline\tcomplete_wait_reviewed_external_data\n' "$(date -Is)" >> "$OUT/stage_status.tsv"
