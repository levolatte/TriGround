#!/usr/bin/env bash
# B/G*/U* diagnostic queue. Launch on the remote host as one independent nohup process.
set -euo pipefail

ROOT="${REMATCH_ROOT:-/root/autodl-tmp/rematch_20260922}"
OUT="${GU_DIAG_OUTPUT:-$ROOT/results/gu_diagnosis_20260927}"
OLD="$ROOT/results/gu_pilot_20260926"
PY="${PYTHON_BIN:-/root/miniconda3/bin/python}"
DATA_ROOT="$ROOT/data/city/train"
SFT="$DATA_ROOT/target_v2/qwen3vl_native_sft"
GT="$DATA_ROOT/target_v2/qwen_generation_val.json"
M2="$ROOT/results/first_batch/m2/checkpoint-928"
C="$ROOT/results/next_stage_20260925/c_phase2/checkpoint-500"
G_OLD="$OLD/g_2026/checkpoint-200"
U_OLD="$OLD/u_2026/checkpoint-200"
MODEL_PATH="${MODEL_PATH:-/root/rematch_models/Qwen3-VL-8B-Instruct}"
M="$OUT/manifests"

for path in "$PY" "$SFT/trimodal_train.json" "$SFT/trimodal_val.json" "$GT" \
  "$OLD/released_cloud.jsonl" "$OLD/released_metadata.jsonl" \
  "$OUT/source/pending_candidates.jsonl" \
  "$OLD/manifests/seed2026/pressure16.json" \
  "$ROOT/results/next_stage_20260925/baseline_m2/predictions.jsonl"; do test -f "$path"; done
for path in "$M2" "$C" "$G_OLD" "$U_OLD" "$MODEL_PATH"; do test -d "$path"; done
mkdir -p "$OUT/logs" "$M"
if [[ -f "$OUT/runner.pid" ]] && kill -0 "$(cat "$OUT/runner.pid")" 2>/dev/null; then
  echo "diagnosis queue is already running: $(cat "$OUT/runner.pid")" >&2
  exit 2
fi
echo $$ > "$OUT/runner.pid"

export PATH="/root/miniconda3/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export DATA_ROOT MODEL_PATH
export QWEN_FINETUNE_DIR="$ROOT/third_party/Qwen3-VL/qwen-vl-finetune"
export DATASET_VARIANT=trimodal EPOCHS=1 MAX_PIXELS=602112 MIN_PIXELS=200704
export PRESERVE_MANIFEST_ORDER=1 SEED=2026
cd "$ROOT/code"

stage=init
trap 'rc=$?; printf "%s\t%s\tfailed:%s\n" "$(date -Is)" "$stage" "$rc" >> "$OUT/stage_status.tsv"; exit "$rc"' ERR

disk_check() {
  "$PY" - "$OUT" "$1" <<'PY'
import shutil, sys
free = shutil.disk_usage(sys.argv[1]).free
required = int(sys.argv[2]) * 1024**3
assert free >= required, f"free data-disk {free / 1024**3:.2f} GiB < {sys.argv[2]} GiB"
PY
}

# The first GPU stage fixes an eight-hour wall-clock window. An estimate is
# checked before every subsequent stage; a running stage finishes at its safe
# checkpoint or complete prediction set, and the actual overrun is recorded.
budget_check() {
  local estimate="$1" now start remaining
  now=$(date +%s)
  if [[ ! -f "$OUT/gpu_start_epoch" ]]; then echo "$now" > "$OUT/gpu_start_epoch"; fi
  start=$(cat "$OUT/gpu_start_epoch")
  remaining=$((start + 28800 - now))
  if (( remaining < estimate )); then
    printf '%s\t%s\tbudget_stop:remaining=%s:estimate=%s\n' "$(date -Is)" "$stage" "$remaining" "$estimate" >> "$OUT/stage_status.tsv"
    echo "GPU budget stops before $stage: remaining ${remaining}s, estimated ${estimate}s" >&2
    exit 0
  fi
}

run_stage() {
  stage="$1"; shift
  [[ -f "$OUT/$stage.complete" ]] && return 0
  printf '%s\t%s\tstart\n' "$(date -Is)" "$stage" >> "$OUT/stage_status.tsv"
  local attempt log rc
  for attempt in 1 2 3; do
    log="$OUT/logs/${stage}_$(date +%Y%m%dT%H%M%S)_attempt${attempt}.log"
    if "$@" > "$log" 2>&1; then
      touch "$OUT/$stage.complete"
      printf '%s\t%s\tcomplete\n' "$(date -Is)" "$stage" >> "$OUT/stage_status.tsv"
      return 0
    else rc=$?; fi
    # Only transient transport errors permit two further attempts.
    if [[ "$attempt" == 3 ]] || ! grep -qE 'ConnectionResetError|ConnectionAbortedError|Temporary failure in name resolution|\[Errno 110\] Connection timed out' "$log"; then
      tail -n 40 "$log" >&2
      return "$rc"
    fi
    sleep "$((attempt * 60))"
  done
}

gpu_stage() {
  local id="$1" estimate="$2"; shift 2
  [[ -f "$OUT/$id.complete" ]] && return 0
  stage="$id"
  budget_check "$estimate"
  disk_check 3
  run_stage "$id" "$@"
}

verify_checkpoint() {
  "$PY" tools/check_gu_diagnosis.py checkpoint "$1" "$2"
}

train_segment() {
  local output="$1" manifest="$2" target="$3" total="$4" stop="${5:-}" save="${6:-100}" latest="" step
  if [[ -d "$output" ]]; then
    latest=$("$PY" - "$output" <<'PY'
from pathlib import Path
import sys
paths = sorted(Path(sys.argv[1]).glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1]))
print(paths[-1] if paths else '')
PY
) || return $?
    if [[ -z "$latest" ]] && [[ -n "$(ls -A "$output")" ]]; then
      echo "nonempty training directory without checkpoint: $output" >&2
      return 2
    fi
  fi
  if [[ -n "$latest" ]]; then
    step="${latest##*-}"
    verify_checkpoint "$latest" "$step" || return $?
    if (( step >= target )); then return 0; fi
  fi
  SEED=2026 OUTPUT_DIR="$output" ANNOTATION_PATH="$manifest" MAX_STEPS="$total" \
    LEARNING_RATE=5e-6 INIT_ADAPTER="$([[ -z "$latest" ]] && echo "$M2")" \
    RESUME_FROM_CHECKPOINT="$latest" SAVE_STEPS="$save" STOP_AFTER_STEP="$stop" \
    SAVE_TOTAL_LIMIT=2 bash scripts/run_qwen3vl_native_lora.sh || return $?
  verify_checkpoint "$output/checkpoint-$target" "$target" || return $?
}

evaluate() {
  local adapter="$1" name="$2" manifest="$3" gt="$4" limit="${5:-0}"
  local target_arg=()
  if [[ "$gt" != - ]]; then target_arg=(--target-manifest "$gt"); fi
  "$PY" tools/evaluate_pretrained_grounder.py --model "$MODEL_PATH" --adapter "$adapter" \
    --manifest "$manifest" "${target_arg[@]}" --data-root "$DATA_ROOT" \
    --output-dir "$OUT/$name" --prompt-style native --max-pixels 602112 \
    --min-pixels 200704 --max-new-tokens 128 --limit "$limit" --resume || return $?
  "$PY" tools/check_gu_diagnosis.py predictions "$manifest" "$OUT/$name/predictions.jsonl" "$limit" || return $?
}

prepare() {
  if [[ -n "$(ls -A "$M")" ]]; then
    echo "manifest destination exists without completed stage: $M" >&2
    return 2
  fi
  "$PY" tools/prepare_gu_diagnosis.py --old-root "$OLD" \
    --released-jsonl "$OLD/released_cloud.jsonl" \
    --released-metadata "$OLD/released_metadata.jsonl" \
    --released-candidates "$OUT/source/pending_candidates.jsonl" \
    --city-native "$SFT/trimodal_train.json" --city-root "$DATA_ROOT" \
    --city-val "$SFT/trimodal_val.json" --city-gt "$GT" --output-dir "$M" || return $?
}

if [[ -f "$OUT/preflight_verify.complete" ]]; then disk_check 3; else disk_check 10; fi
run_stage prepare prepare
run_stage manifests_verify "$PY" tools/check_gu_diagnosis.py manifests "$M"

# Historical prompt and eight original City validation IDs must reproduce the
# saved M2 run before new-prompt comparisons have any standing.
gpu_stage m2_env8 300 evaluate "$M2" m2_env8 "$M/env8.json" "$GT"
run_stage m2_env8_verify "$PY" tools/check_gu_diagnosis.py reproduce8 \
  "$ROOT/results/next_stage_20260925/baseline_m2/predictions.jsonl" "$OUT/m2_env8/predictions.jsonl"

export AUDIT_SAMPLE_COUNT=8
gpu_stage preflight_step1 600 train_segment "$OUT/preflight" "$M/pressure16.json" 1 2 1 1
gpu_stage preflight_resume2 600 train_segment "$OUT/preflight" "$M/pressure16.json" 2 2 "" 1
unset AUDIT_SAMPLE_COUNT
for repeat in 1 2; do
  gpu_stage "preflight_reload_$repeat" 300 evaluate "$OUT/preflight/checkpoint-2" \
    "preflight_reload_$repeat" "$M/pressure2city.json" -
done
run_stage preflight_verify "$PY" tools/check_gu_diagnosis.py preflight "$OUT/preflight" \
  "$M/pressure16.json" "$OUT/preflight_reload_1/predictions.jsonl" \
  "$OUT/preflight_reload_2/predictions.jsonl"

# Pretraining diagnosis. Different prompts are reported as paired only for the
# overlapping 83 IDs; the City variants share the same 96 IDs and original GT.
for model in m2 g_old u_old; do
  case "$model" in m2) adapter="$M2";; g_old) adapter="$G_OLD";; u_old) adapter="$U_OLD";; esac
  gpu_stage "${model}_fit_legacy181" 900 evaluate "$adapter" "${model}_fit_legacy181" \
    "$M/fit_legacy181.json" "$M/fit_legacy181_gt.json"
  gpu_stage "${model}_fit_canonical_aux83" 600 evaluate "$adapter" "${model}_fit_canonical_aux83" \
    "$M/fit_canonical_aux83.json" "$M/fit_canonical_aux83_gt.json"
done
for model in m2 c g_old u_old; do
  case "$model" in m2) adapter="$M2";; c) adapter="$C";; g_old) adapter="$G_OLD";; u_old) adapter="$U_OLD";; esac
  for variant in rgb rgb_ir rgb_depth trimodal; do
    gpu_stage "${model}_city96_${variant}" 600 evaluate "$adapter" "${model}_city96_${variant}" \
      "$M/city96_${variant}.json" "$M/city96_gt.json"
  done
done
run_stage report_pre "$PY" -m tools.report_gu_diagnosis --output "$OUT" --phase pre

# Each 200-update arm keeps the same M2 initialization and scheduler horizon.
# Step 100 saves a full checkpoint; there is no evaluation before step 200.
for model in b gstar ustar; do
  gpu_stage "${model}_train100" 2400 train_segment "$OUT/$model" "$M/train_${model}.json" 100 200 100
  gpu_stage "${model}_train200" 2400 train_segment "$OUT/$model" "$M/train_${model}.json" 200 200
  run_stage "${model}_trace" "$PY" tools/check_gu_diagnosis.py trace \
    "$OUT/$model/consumed_samples.jsonl" "$M/train_${model}.json" 200
  gpu_stage "${model}_city412" 1500 evaluate "$OUT/$model/checkpoint-200" "${model}_city412" \
    "$SFT/trimodal_val.json" "$GT"
done
run_stage report_city412 "$PY" -m tools.report_gu_diagnosis --output "$OUT" --root "$ROOT" --phase city412

for model in gstar ustar; do
  adapter="$OUT/$model/checkpoint-200"
  gpu_stage "${model}_fit_legacy181" 900 evaluate "$adapter" "${model}_fit_legacy181" \
    "$M/fit_legacy181.json" "$M/fit_legacy181_gt.json"
  gpu_stage "${model}_fit_canonical_aux83" 600 evaluate "$adapter" "${model}_fit_canonical_aux83" \
    "$M/fit_canonical_aux83.json" "$M/fit_canonical_aux83_gt.json"
  for variant in rgb rgb_ir rgb_depth trimodal; do
    gpu_stage "${model}_city96_${variant}" 600 evaluate "$adapter" "${model}_city96_${variant}" \
      "$M/city96_${variant}.json" "$M/city96_gt.json"
  done
done
run_stage report_final "$PY" -m tools.report_gu_diagnosis --output "$OUT" --phase final
printf '%s\tqueue\tcomplete\n' "$(date -Is)" >> "$OUT/stage_status.tsv"
