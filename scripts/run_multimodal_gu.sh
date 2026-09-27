#!/usr/bin/env bash
# Invoke only after accepted data and the real mixed-task GPU preflight exist.
set -euo pipefail
ROOT="${REMATCH_ROOT:-/root/autodl-tmp/rematch_20260922}"
OUT="${GU_OUTPUT:-$ROOT/results/multimodal_gu_seed2026}"
PY="${PYTHON_BIN:-/root/miniconda3/bin/python}"
: "${ACCEPTED_MANIFEST_DIR:?Set the directory of accepted G/U phase manifests}"
: "${GPU_PREFLIGHT_COMPLETE:?Set the completed mixed-task GPU preflight record path}"
: "${EXTERNAL_NATIVE:?Set the isolated human-reviewed external native manifest}"
: "${EXTERNAL_GT:?Set the original external review ground-truth manifest}"
test -f "$GPU_PREFLIGHT_COMPLETE"
test -f "$EXTERNAL_NATIVE"
test -f "$EXTERNAL_GT"
export PATH="/root/miniconda3/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export DATA_ROOT="$ROOT/data/city/train"
export MODEL_PATH="${MODEL_PATH:-/root/rematch_models/Qwen3-VL-8B-Instruct}"
export QWEN_FINETUNE_DIR="$ROOT/third_party/Qwen3-VL/qwen-vl-finetune"
export SEED="${SEED:-2026}" DATASET_VARIANT=trimodal EPOCHS=1
export MAX_PIXELS=602112 MIN_PIXELS=200704
GT="$DATA_ROOT/target_v2/qwen_generation_val.json"
SFT="$DATA_ROOT/target_v2/qwen3vl_native_sft"
M2="$ROOT/results/first_batch/m2/checkpoint-928"
mkdir -p "$OUT/logs"
cd "$ROOT/code"

"$PY" - "$ACCEPTED_MANIFEST_DIR" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1])
for phase,n in ((1,8000),(2,4000)):
    metas=[]
    for branch in ('g','u'):
        rows=[json.loads(x) for x in (p/f'{branch}_phase{phase}_metadata.jsonl').read_text().splitlines()]
        assert len(rows)==n, (branch,phase,len(rows))
        assert all(r['review_status'] in ('human_accepted','approved_batch') for r in rows if r['source']=='rgbdt')
        assert len(json.loads((p/f'{branch}_phase{phase}.json').read_text()))==n
        metas.append([(r['source'],r['source_id'],r['augmentation']) for r in rows])
    assert metas[0]==metas[1], 'G/U scene/order/augmentation mismatch'
PY

stage=init
trap 'rc=$?; printf "%s\t%s\tfailed:%s\n" "$(date -Is)" "$stage" "$rc" >> "$OUT/stage_status.tsv"; exit "$rc"' ERR
run_stage() {
  stage="$1"; shift
  [[ -f "$OUT/$stage.complete" ]] && return
  printf '%s\t%s\tstart\n' "$(date -Is)" "$stage" >> "$OUT/stage_status.tsv"
  local attempt log rc
  for attempt in 1 2 3; do
    log="$OUT/logs/${stage}_$(date +%Y%m%dT%H%M%S)_attempt${attempt}.log"
    if "$@" > "$log" 2>&1; then
      touch "$OUT/$stage.complete"
      printf '%s\t%s\tcomplete\n' "$(date -Is)" "$stage" >> "$OUT/stage_status.tsv"
      return
    else rc=$?; fi
    if [[ "$attempt" == 3 ]] || ! grep -qE 'ConnectionResetError|ConnectionAbortedError|Temporary failure in name resolution|\[Errno 110\] Connection timed out' "$log"; then
      tail -n 30 "$log" >&2
      return "$rc"
    fi
    sleep "$((attempt * 60))"
  done
}
train_segment() {
  local output="$1" manifest="$2" total="$3" lr="$4" initial="$5" stop="${6:-}" resume
  resume=$("$PY" - "$output" <<'PY'
import shutil,sys
from pathlib import Path
p=Path(sys.argv[1]); p.mkdir(parents=True,exist_ok=True)
assert shutil.disk_usage(p).free >= 3*1024**3, 'Less than 3 GiB free'
paths=sorted(p.glob('checkpoint-*'),key=lambda x:int(x.name.split('-')[-1]))
print(paths[-1] if paths else '')
PY
)
  if [[ -n "$resume" ]]; then
    if (( ${resume##*-} >= ${stop:-$total} )); then return; fi
    initial=''
  fi
  OUTPUT_DIR="$output" ANNOTATION_PATH="$manifest" MAX_STEPS="$total" LEARNING_RATE="$lr" \
    INIT_ADAPTER="$initial" RESUME_FROM_CHECKPOINT="$resume" SAVE_STEPS=500 STOP_AFTER_STEP="$stop" \
    bash scripts/run_qwen3vl_native_lora.sh
}
evaluate() {
  "$PY" tools/evaluate_pretrained_grounder.py --model "$MODEL_PATH" --adapter "$1" \
    --manifest "$3" --target-manifest "$4" --data-root "$DATA_ROOT" --output-dir "$OUT/$2" \
    --prompt-style native --max-pixels 602112 --min-pixels 200704 --max-new-tokens 128 --resume
}
for branch in g u; do
  run_stage "${branch}_train500" train_segment "$OUT/${branch}_phase1" "$ACCEPTED_MANIFEST_DIR/${branch}_phase1.json" 1000 5e-6 "$M2" 500
  run_stage "${branch}_city500" evaluate "$OUT/${branch}_phase1/checkpoint-500" "${branch}_city500" "$SFT/trimodal_val.json" "$GT"
  run_stage "${branch}_train1000" train_segment "$OUT/${branch}_phase1" "$ACCEPTED_MANIFEST_DIR/${branch}_phase1.json" 1000 5e-6 "$M2"
  run_stage "${branch}_city1000" evaluate "$OUT/${branch}_phase1/checkpoint-1000" "${branch}_city1000" "$SFT/trimodal_val.json" "$GT"
  run_stage "${branch}_train1500" train_segment "$OUT/${branch}_phase2" "$ACCEPTED_MANIFEST_DIR/${branch}_phase2.json" 500 3e-6 "$OUT/${branch}_phase1/checkpoint-1000"
  run_stage "${branch}_city1500" evaluate "$OUT/${branch}_phase2/checkpoint-500" "${branch}_city1500" "$SFT/trimodal_val.json" "$GT"
  run_stage "${branch}_external1500" evaluate "$OUT/${branch}_phase2/checkpoint-500" "${branch}_external1500" "$EXTERNAL_NATIVE" "$EXTERNAL_GT"
done
run_stage city_report "$PY" tools/report_rematch_experiment.py --manifest "$GT" \
  --run "M2=$ROOT/results/next_stage_20260925/baseline_m2/predictions.jsonl" \
  --run "C1500=$ROOT/results/next_stage_20260925/c_step1500/predictions.jsonl" \
  --run "G1500=$OUT/g_city1500/predictions.jsonl" --run "U1500=$OUT/u_city1500/predictions.jsonl" \
  --output-dir "$OUT/city_report"
printf '%s\tgu_pipeline\tcomplete\n' "$(date -Is)" >> "$OUT/stage_status.tsv"
