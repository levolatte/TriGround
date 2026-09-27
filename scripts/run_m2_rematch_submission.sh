#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/rematch_20260922
PY=/root/miniconda3/bin/python
OUT="$ROOT/submissions/m2_epoch2_4090_20260924"
MODEL=/root/rematch_models/Qwen3-VL-8B-Instruct
ADAPTER="$ROOT/results/first_batch/m2/checkpoint-928"
DATA="$ROOT/data/official_test_rematch_20260924"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
cd "$ROOT/code"
mkdir -p "$OUT"
stage=preflight
trap 'rc=$?; printf "%s\t%s\tfailed:%s\n" "$(date -Is)" "$stage" "$rc" >> "$OUT/stage_status.tsv"; exit "$rc"' ERR
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
run_stage() {
  stage="$1"; shift
  printf '%s\t%s\tstart\n' "$(date -Is)" "$stage" >> "$OUT/stage_status.tsv"
  local attempt log rc
  for attempt in 1 2 3; do
    log="$OUT/${stage}_attempt${attempt}_$(date +%Y%m%dT%H%M%S).log"
    if "$@" > "$log" 2>&1; then
      printf '%s\t%s\tcomplete\t%s\n' "$(date -Is)" "$stage" "$log" >> "$OUT/stage_status.tsv"
      return 0
    else rc=$?; fi
    if (( attempt == 3 )) || grep -Eqi 'out of memory|OutOfMemoryError|ValueError|TypeError|AssertionError|ImportError|ModuleNotFoundError' "$log" || ! grep -Eqi 'ConnectionResetError|TimeoutError|temporarily unavailable' "$log"; then return "$rc"; fi
    sleep "$((attempt * 60))"
  done
}
run_stage validation8 "$PY" tools/predict_native_submission.py \
  --model "$MODEL" --adapter "$ADAPTER" --modality trimodal \
  --queries "$OUT/validation8_queries.json" --data-root "$ROOT/data/city/train" \
  --output-dir "$OUT/validation8" --expected-query-count 8 --resume
stage=entrypoint_parity
"$PY" - "$OUT" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
read=lambda name: {r['id']:r for r in (json.loads(s) for s in (p/name/'predictions.jsonl').read_text().splitlines())}
a,b=read('reference8'),read('validation8')
assert len(a)==len(b)==8 and set(a)==set(b)
differences={k:[field for field,other in [('raw_text','raw_text'),('prediction','bbox'),('image_grid_thw','image_grid_thw')] if a[k][field]!=b[k][other]] for k in a}
assert not any(differences.values()),differences
(p/'entrypoint_parity.json').write_text(json.dumps({'samples':8,'identical_fields':['raw_text','prediction','image_grid_thw'],'passed':True},indent=2))
PY
printf '%s\t%s\tcomplete\n' "$(date -Is)" "$stage" >> "$OUT/stage_status.tsv"
run_stage official5690 "$PY" tools/predict_native_submission.py \
  --model "$MODEL" --adapter "$ADAPTER" --modality trimodal \
  --queries "$DATA/queries/queries.json" --data-root "$DATA" \
  --output-dir "$OUT/full" --expected-query-count 5690 --resume
printf '%s\tsubmission_queue\tcomplete\n' "$(date -Is)" >> "$OUT/stage_status.tsv"
