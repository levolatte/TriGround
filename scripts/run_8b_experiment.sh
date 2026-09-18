#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:?usage: run_8b_experiment.sh RUN_DIR [PRESSURE_COMMAND]}"
PRESSURE_COMMAND="${2:-${PRESSURE_COMMAND:-}}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:$REPO_DIR/src:$REPO_DIR/tools${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN_DIR="$($PYTHON_BIN -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$RUN_DIR")"
PLAN="$RUN_DIR/plan.json"
VAL_MANIFEST="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["manifests"]["reviewed_val"])' "$PLAN")"
STATUS="$RUN_DIR/stage_status.tsv"
START_FILE="$RUN_DIR/experiment_started_unix.txt"

[[ -f "$PLAN" ]] || { echo "missing plan: $PLAN" >&2; exit 2; }
if [[ -z "$PRESSURE_COMMAND" ]]; then
  printf -v PRESSURE_COMMAND 'set -e; for phase in stage1_ir stage1_depth c4; do %q tools/stress_8b_training.py --config %q/configs/$phase.yaml --output %q/results/pressure_$phase.json; done' "$PYTHON_BIN" "$RUN_DIR" "$RUN_DIR"
fi
mkdir -p "$RUN_DIR/logs"
[[ -f "$START_FILE" ]] || date +%s > "$START_FILE"
START_UNIX="$(<"$START_FILE")"
HARD_DEADLINE=$((START_UNIX + 48 * 3600))
TRAINING_CUTOFF=$((START_UNIX + 44 * 3600))
[[ -f "$STATUS" ]] || printf 'stage\tkind\tstarted_unix\tended_unix\tduration_seconds\tstatus\tpeak_gpu_mib\tfree_kib\n' > "$STATUS"

free_kib() { df -Pk "$RUN_DIR" | awk 'NR==2 {print $4}'; }
guard_space() {
  local free
  free="$(free_kib)"
  (( free >= 3 * 1024 * 1024 )) || { echo "free space below 3 GiB: $free KiB" >&2; exit 3; }
}
guard_deadline() {
  local kind="$1" now
  now="$(date +%s)"
  (( now < HARD_DEADLINE )) || { echo "48-hour hard deadline reached" >&2; exit 4; }
  if [[ "$kind" == train ]] && (( now >= TRAINING_CUTOFF )); then
    echo "44-hour training cutoff reached" >&2
    exit 5
  fi
}
run_stage() {
  local name="$1" kind="$2" command="$3" started ended rc peak=0 current remaining
  guard_deadline "$kind"
  guard_space
  started="$(date +%s)"
  if [[ "$kind" == train ]]; then
    remaining=$((TRAINING_CUTOFF - started))
  else
    remaining=$((HARD_DEADLINE - started))
  fi
  (( remaining > 15 )) || { echo "insufficient time remaining for $name" >&2; return 4; }
  echo "[$(date --iso-8601=seconds)] START $name"
  set +e
  timeout --signal=INT --kill-after=10 "$((remaining-10))" bash -c "$command" \
    > >(tee "$RUN_DIR/logs/$name.log") 2>&1 &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    current="$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END {print s+0}')"
    (( current > peak )) && peak="$current"
    sleep 5
  done
  wait "$pid"
  rc=$?
  set -e
  ended="$(date +%s)"
  if (( rc == 0 )); then status=ok; else status="failed:$rc"; fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$name" "$kind" "$started" "$ended" "$((ended-started))" "$status" "$peak" "$(free_kib)" >> "$STATUS"
  (( rc == 0 )) || { echo "$name failed with exit $rc" >&2; return "$rc"; }
  guard_space
  echo "[$(date --iso-8601=seconds)] DONE $name"
}

if [[ ! -f "$RUN_DIR/.pressure.ok" ]]; then
  if run_stage pressure train "$PRESSURE_COMMAND"; then
    pressure_rc=0
  else
    pressure_rc=$?
  fi
  if (( pressure_rc == 42 )); then
    current_pixels="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["max_pixels"])' "$PLAN")"
    [[ "$current_pixels" != 602112 ]] || { echo "602112 pressure OOM: investigate before retry" >&2; exit 42; }
    mkdir -p "$RUN_DIR/results/pressure_802816"
    for report in "$RUN_DIR"/results/pressure_*.json; do
      [[ ! -f "$report" ]] || mv "$report" "$RUN_DIR/results/pressure_802816/"
    done
    "$PYTHON_BIN" tools/prepare_8b_experiment.py --run-dir "$RUN_DIR" --set-max-pixels 602112
    run_stage pressure_602112 train "$PRESSURE_COMMAND"
  elif (( pressure_rc != 0 )); then
    exit "$pressure_rc"
  fi
  touch "$RUN_DIR/.pressure.ok"
fi

if [[ ! -f "$RUN_DIR/.native_rgb.ok" ]]; then
  run_stage native_rgb eval "$PYTHON_BIN evaluate.py --config '$RUN_DIR/configs/native_rgb.yaml' --rgb-only --output '$RUN_DIR/results/native_rgb.json' --rows-output '$RUN_DIR/results/native_rgb_rows.jsonl'"
  touch "$RUN_DIR/.native_rgb.ok"
fi

for stage in ${EXPERIMENT_STAGES:-stage1_ir stage1_depth c4 t4}; do
  case "$stage" in
    stage1_ir|stage1_depth|c4|t4|w4|c4_seed2027|t4_seed2027|w4_seed2027|weak|weak_clean|control|control_clean) ;;
    *) echo "unknown experiment stage: $stage" >&2; exit 2 ;;
  esac
  if [[ ! -f "$RUN_DIR/.$stage.ok" ]]; then
    run_stage "$stage" train "$PYTHON_BIN train.py --config '$RUN_DIR/configs/$stage.yaml'"
    touch "$RUN_DIR/.$stage.ok"
  fi
  if [[ "$stage" != stage1_ir && "$stage" != stage1_depth ]]; then
    case "$stage" in
      c4*|t4*|w4*) last_label=fixed4 ;;
      *) last_label=last ;;
    esac
    if [[ ! -f "$RUN_DIR/.${stage}_${last_label}_eval.ok" ]]; then
      run_stage "${stage}_${last_label}_eval" eval "$PYTHON_BIN evaluate.py --config '$RUN_DIR/configs/$stage.yaml' --checkpoint '$RUN_DIR/outputs/$stage/last_phase_a.pt' --output '$RUN_DIR/results/${stage}_${last_label}.json' --rows-output '$RUN_DIR/results/${stage}_${last_label}_rows.jsonl'"
      touch "$RUN_DIR/.${stage}_${last_label}_eval.ok"
    fi
    if [[ ! -f "$RUN_DIR/.${stage}_best_eval.ok" ]]; then
      run_stage "${stage}_best_eval" eval "$PYTHON_BIN evaluate.py --config '$RUN_DIR/configs/$stage.yaml' --checkpoint '$RUN_DIR/outputs/$stage/best_phase_a.pt' --output '$RUN_DIR/results/${stage}_best.json' --rows-output '$RUN_DIR/results/${stage}_best_rows.jsonl'"
      touch "$RUN_DIR/.${stage}_best_eval.ok"
    fi
    if [[ ! -f "$RUN_DIR/.${stage}_diagnostics.ok" ]]; then
      run_stage "${stage}_diagnostics" eval "$PYTHON_BIN tools/diagnose_8b_fusion.py --config '$RUN_DIR/configs/$stage.yaml' --checkpoint '$RUN_DIR/outputs/$stage/best_phase_a.pt' --manifest '$VAL_MANIFEST' --output '$RUN_DIR/results/${stage}_diagnostics.json'"
      touch "$RUN_DIR/.${stage}_diagnostics.ok"
    fi
  fi
done

"$PYTHON_BIN" tools/report_8b_experiment.py --run-dir "$RUN_DIR"
echo "Requested sequence complete. Select optional branches from results and remaining budget."
