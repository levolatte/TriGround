#!/usr/bin/env bash
# One bounded G/U queue. Run only after the reviewed training export is available.
set -euo pipefail
ROOT="${REMATCH_ROOT:-/root/autodl-tmp/rematch_20260922}"
OUT="${GU_OUTPUT:-$ROOT/results/gu_pilot_20260926}"
GU_RELEASED_JSONL="${GU_RELEASED_JSONL:-$OUT/released_cloud.jsonl}"
GU_RELEASED_METADATA="${GU_RELEASED_METADATA:-$OUT/released_metadata.jsonl}"
PY="${PYTHON_BIN:-/root/miniconda3/bin/python}"
DATA_ROOT="$ROOT/data/city/train"
SFT="$DATA_ROOT/target_v2/qwen3vl_native_sft"
GT="$DATA_ROOT/target_v2/qwen_generation_val.json"
M2="$ROOT/results/first_batch/m2/checkpoint-928"
C="$ROOT/results/next_stage_20260925/c_step1500/predictions.jsonl"
test -f "$GU_RELEASED_JSONL"
test -f "$GU_RELEASED_METADATA"
test -f "$C"
test -d "$M2"
export PATH="/root/miniconda3/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export DATA_ROOT MODEL_PATH="${MODEL_PATH:-/root/rematch_models/Qwen3-VL-8B-Instruct}"
export QWEN_FINETUNE_DIR="$ROOT/third_party/Qwen3-VL/qwen-vl-finetune"
export DATASET_VARIANT=trimodal EPOCHS=1 MAX_PIXELS=602112 MIN_PIXELS=200704
export PRESERVE_MANIFEST_ORDER=1
mkdir -p "$OUT/logs" "$OUT/manifests"
if [[ -f "$OUT/runner.pid" ]]; then
  old_pid=$(cat "$OUT/runner.pid")
  if kill -0 "$old_pid" 2>/dev/null; then
    echo "G/U queue already running as PID $old_pid" >&2
    exit 2
  fi
fi
echo $$ > "$OUT/runner.pid"
cd "$ROOT/code"
stage=init
trap 'rc=$?; printf "%s\t%s\tfailed:%s\n" "$(date -Is)" "$stage" "$rc" >> "$OUT/stage_status.tsv"; exit "$rc"' ERR

disk_check() {
  "$PY" - "$OUT" "$1" <<'PY'
import shutil, sys
free = shutil.disk_usage(sys.argv[1]).free
required = int(sys.argv[2]) * 1024**3
assert free >= required, f"free disk {free / 1024**3:.2f} GiB below {sys.argv[2]} GiB"
PY
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
    # Only transport failures get two additional attempts; numerical, OOM,
    # malformed data and interface failures stop immediately.
    if [[ "$attempt" == 3 ]] || ! grep -qE 'ConnectionResetError|ConnectionAbortedError|Temporary failure in name resolution|\[Errno 110\] Connection timed out' "$log"; then
      tail -n 35 "$log" >&2
      return "$rc"
    fi
    sleep "$((attempt * 60))"
  done
}

prepare() {
  local seed="$1" steps="$2" destination="$3"
  if [[ -e "$destination" ]]; then
    echo "manifest destination already exists without completed stage; refusing overwrite: $destination" >&2
    return 2
  fi
  "$PY" tools/prepare_gu_same_day_pilot.py --released-jsonl "$GU_RELEASED_JSONL" \
    --city-native "$SFT/trimodal_train.json" --city-root "$DATA_ROOT" \
    --metadata-jsonl "$GU_RELEASED_METADATA" \
    --output-dir "$destination" --seed "$seed" --steps "$steps"
}

train_segment() {
  local output="$1" manifest="$2" total="$3" save="$4" stop="${5:-}" resume=""
  disk_check 3
  if [[ -d "$output" ]]; then
    resume=$("$PY" - "$output" <<'PY'
from pathlib import Path
import sys
paths=sorted(Path(sys.argv[1]).glob('checkpoint-*'), key=lambda p:int(p.name.split('-')[-1]))
print(paths[-1] if paths else '')
PY
)
    if [[ -z "$resume" ]] && [[ -n "$(ls -A "$output")" ]]; then
      echo "non-empty stage without checkpoint; refusing to overwrite $output" >&2
      return 2
    fi
  fi
  if [[ -n "$resume" ]]; then
    local current="${resume##*-}"
    if (( current >= ${stop:-$total} )); then return 0; fi
  fi
  SEED="$SEED" OUTPUT_DIR="$output" ANNOTATION_PATH="$manifest" MAX_STEPS="$total" \
    LEARNING_RATE=5e-6 INIT_ADAPTER="$([[ -z "$resume" ]] && echo "$M2")" \
    RESUME_FROM_CHECKPOINT="$resume" SAVE_STEPS="$save" STOP_AFTER_STEP="$stop" \
    bash scripts/run_qwen3vl_native_lora.sh
}

evaluate() {
  local adapter="$1" name="$2" manifest="${3:-$SFT/trimodal_val.json}" target="${4-$GT}"
  local target_arg=()
  if [[ -n "$target" ]]; then target_arg=(--target-manifest "$target"); fi
  if [[ "$name" == external_* ]]; then
    mkdir -p "$OUT/$name"
    "$PY" - "$OUT/$name" <<'PY'
import json,socket,sys
from pathlib import Path
directory=Path(sys.argv[1]); marker=directory/'evaluation_host.json'; host=socket.gethostname()
if marker.exists():
    assert json.loads(marker.read_text())['hostname']==host, 'external evaluation host changed'
else:
    assert not (directory/'predictions.jsonl').exists(), 'external predictions lack host provenance'
    marker.write_text(json.dumps({'hostname':host})+'\n')
PY
  fi
  "$PY" tools/evaluate_pretrained_grounder.py --model "$MODEL_PATH" --adapter "$adapter" \
    --manifest "$manifest" "${target_arg[@]}" --data-root "$DATA_ROOT" \
    --output-dir "$OUT/$name" --prompt-style native --max-pixels 602112 \
    --min-pixels 200704 --max-new-tokens 128 --resume
}

verify_smoke() {
  "$PY" - "$OUT" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
manifest=json.loads((p/'manifests/seed2026/pressure16.json').read_text())
trace=[json.loads(line) for line in (p/'preflight/consumed_samples.jsonl').read_text().splitlines()]
assert len(trace)==16, len(trace)
assert [r['id'] for r in trace]==[r['id'] for r in manifest]
assert json.loads((p/'preflight/checkpoint-1/trainer_state.json').read_text())['global_step']==1
assert json.loads((p/'preflight/checkpoint-2/trainer_state.json').read_text())['global_step']==2
for step in (1,2):
    checkpoint=p/f'preflight/checkpoint-{step}'
    assert all((checkpoint/name).is_file() for name in ('optimizer.pt','scheduler.pt','trainer_state.json'))
    assert list(checkpoint.glob('rng_state*.pth'))
a=[json.loads(x) for x in (p/'preflight_reload_1/predictions.jsonl').read_text().splitlines()]
b=[json.loads(x) for x in (p/'preflight_reload_2/predictions.jsonl').read_text().splitlines()]
assert len(a)==len(b)==2
assert [(r['id'],r['raw_text'],r['prediction']) for r in a] == [(r['id'],r['raw_text'],r['prediction']) for r in b]
PY
}

check_trace() {
  "$PY" - "$1" "$2" "$3" <<'PY'
import json,sys
from pathlib import Path
output,manifest,steps=Path(sys.argv[1]),Path(sys.argv[2]),int(sys.argv[3])
rows=json.loads(manifest.read_text())
trace=[json.loads(line) for line in (output/'consumed_samples.jsonl').read_text().splitlines()]
assert len(trace)==steps*8,(len(trace),steps)
expected=[rows[index % len(rows)]['id'] for index in range(steps*8)]
assert [row['id'] for row in trace]==expected, 'actual sample order differs from manifest sequence'
PY
}

seed_pair() {
  local seed="$1" dir="$OUT/manifests/seed$1" branch
  for branch in g u; do
    SEED="$seed" run_stage "${branch}_${seed}_train100" train_segment "$OUT/${branch}_${seed}" "$dir/${branch}_train.json" 200 100 100
    run_stage "${branch}_${seed}_city100" evaluate "$OUT/${branch}_${seed}/checkpoint-100" "${branch}_${seed}_city100"
    SEED="$seed" run_stage "${branch}_${seed}_train200" train_segment "$OUT/${branch}_${seed}" "$dir/${branch}_train.json" 200 100
    run_stage "${branch}_${seed}_trace" check_trace "$OUT/${branch}_${seed}" "$dir/${branch}_train.json" 200
    run_stage "${branch}_${seed}_city200" evaluate "$OUT/${branch}_${seed}/checkpoint-200" "${branch}_${seed}_city200"
  done
  run_stage "report_${seed}" "$PY" tools/report_rematch_experiment.py --manifest "$GT" \
    --run "M2=$ROOT/results/next_stage_20260925/baseline_m2/predictions.jsonl" \
    --run "C=$C" \
    --run "G100=$OUT/g_${seed}_city100/predictions.jsonl" \
    --run "G200=$OUT/g_${seed}_city200/predictions.jsonl" \
    --run "U100=$OUT/u_${seed}_city100/predictions.jsonl" \
    --run "U200=$OUT/u_${seed}_city200/predictions.jsonl" \
    --output-dir "$OUT/report_${seed}"
  run_stage "gate_${seed}" "$PY" -m tools.gate_gu_pilot seed --gt "$GT" --c "$C" \
    --g "$OUT/g_${seed}_city200/predictions.jsonl" --u "$OUT/u_${seed}_city200/predictions.jsonl" \
    --output "$OUT/gate_${seed}.json"
}

if [[ ! -f "$OUT/preflight_verify.complete" ]]; then disk_check 10; else disk_check 3; fi
run_stage prepare_2026 prepare 2026 200 "$OUT/manifests/seed2026"
SEED=2026 AUDIT_SAMPLE_COUNT=8 run_stage preflight_step1 train_segment "$OUT/preflight" "$OUT/manifests/seed2026/pressure16.json" 2 1 1
SEED=2026 AUDIT_SAMPLE_COUNT=8 run_stage preflight_resume2 train_segment "$OUT/preflight" "$OUT/manifests/seed2026/pressure16.json" 2 1
for repeat in 1 2; do
  run_stage "preflight_reload_$repeat" evaluate "$OUT/preflight/checkpoint-2" "preflight_reload_$repeat" \
    "$OUT/manifests/seed2026/pressure2city.json" ""
done
run_stage preflight_verify verify_smoke
seed_pair 2026
if [[ "$("$PY" - "$OUT/gate_2026.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['status'])
PY
)" != continue ]]; then
  printf '%s\tqueue\tstop_2026_gate\n' "$(date -Is)" >> "$OUT/stage_status.tsv"
  exit 0
fi
run_stage prepare_2027 prepare 2027 200 "$OUT/manifests/seed2027"
seed_pair 2027
run_stage winner_gate "$PY" -m tools.gate_gu_pilot winner --seed2026 "$OUT/gate_2026.json" \
  --seed2027 "$OUT/gate_2027.json" --output "$OUT/winner_gate.json"
winner=$("$PY" - "$OUT/winner_gate.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1])).get('winner',''))
PY
)
if [[ -z "$winner" ]]; then
  printf '%s\tqueue\tstop_two_seed_gate\n' "$(date -Is)" >> "$OUT/stage_status.tsv"
  exit 0
fi
if [[ -z "${EXTERNAL_NATIVE:-}" || -z "${EXTERNAL_GT:-}" || -z "${EXTERNAL_REVIEW_JSONL:-}" || -z "${EXTERNAL_SOURCE:-}" || \
      ! -f "${EXTERNAL_NATIVE:-/dev/null}" || ! -f "${EXTERNAL_GT:-/dev/null}" || ! -f "${EXTERNAL_REVIEW_JSONL:-/dev/null}" ]]; then
  printf '%s\tqueue\tWAIT_reviewed_external_100\n' "$(date -Is)" >> "$OUT/stage_status.tsv"
  exit 0
fi
stage=external_review_gate
"$PY" -m tools.gate_gu_pilot review --gt "$EXTERNAL_GT" \
  --native "$EXTERNAL_NATIVE" --review "$EXTERNAL_REVIEW_JSONL" \
  --expected-source "$EXTERNAL_SOURCE" --output "$OUT/external_review_gate.json"
if [[ "$("$PY" - "$OUT/external_review_gate.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['status'])
PY
)" == wait ]]; then
  printf '%s\tqueue\tWAIT_reviewed_external_100\n' "$(date -Is)" >> "$OUT/stage_status.tsv"
  exit 0
fi
run_stage external_m2 evaluate "$M2" external_m2 "$EXTERNAL_NATIVE" "$EXTERNAL_GT"
run_stage external_winner evaluate "$OUT/${winner,,}_2027/checkpoint-200" external_winner "$EXTERNAL_NATIVE" "$EXTERNAL_GT"
run_stage external_gate "$PY" -m tools.gate_gu_pilot external --gt "$EXTERNAL_GT" \
  --native "$EXTERNAL_NATIVE" --review "$EXTERNAL_REVIEW_JSONL" \
  --m2 "$OUT/external_m2/predictions.jsonl" --winner "$OUT/external_winner/predictions.jsonl" \
  --expected-source "${EXTERNAL_SOURCE:?Set the original external source name}" --output "$OUT/external_gate.json"
if [[ "$("$PY" - "$OUT/external_gate.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['status'])
PY
)" != pass ]]; then
  printf '%s\tqueue\tstop_external_gate\n' "$(date -Is)" >> "$OUT/stage_status.tsv"
  exit 0
fi
run_stage prepare_600 prepare 2026 600 "$OUT/manifests/seed2026_600"
for step in 200 400 600; do
  if [[ "$step" == 600 ]]; then pause=""; else pause="$step"; fi
  SEED=2026 SAVE_TOTAL_LIMIT=3 run_stage "winner_600_train${step}" train_segment "$OUT/winner_600" \
    "$OUT/manifests/seed2026_600/${winner,,}_train.json" 600 200 "$pause"
  run_stage "winner_600_trace${step}" check_trace "$OUT/winner_600" \
    "$OUT/manifests/seed2026_600/${winner,,}_train.json" "$step"
  run_stage "winner_600_city${step}" evaluate "$OUT/winner_600/checkpoint-$step" "winner_600_city${step}"
done
printf '%s\tqueue\tcomplete_600\n' "$(date -Is)" >> "$OUT/stage_status.tsv"
