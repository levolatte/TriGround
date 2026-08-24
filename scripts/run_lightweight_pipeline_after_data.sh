#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/final_training_upload/mm-grounding-adapters
dataset=/root/autodl-tmp/datasets/RoboRefIt
pipeline_log="$repo/logs/lightweight_pipeline.log"
mkdir -p "$repo/logs"
cd "$repo"

while [[ ! -e "$dataset/PREPARED" ]]; do
  if [[ -s "$dataset/prepare.pid" ]]; then
    prepare_pid=$(cat "$dataset/prepare.pid")
    if ! kill -0 "$prepare_pid" 2>/dev/null; then
      echo "RoboRefIt preparation exited without PREPARED marker" >&2
      exit 1
    fi
  fi
  sleep 15
done

if [[ -s logs/stage1a_ir_eval512_waiter.pid ]]; then
  eval_waiter_pid=$(cat logs/stage1a_ir_eval512_waiter.pid)
  while kill -0 "$eval_waiter_pid" 2>/dev/null; do
    state=$(awk '{print $3}' "/proc/$eval_waiter_pid/stat" 2>/dev/null || true)
    [[ "$state" == Z ]] && break
    sleep 10
  done
fi
grep -q 'evaluation_exit=0' logs/stage1a_ir_eval512.log || {
  echo "Stage 1A fixed evaluation did not finish successfully" >&2
  exit 1
}
[[ -s runs/stage1a_ir_50/best_phase_a.pt ]]

generated_config=configs/generated_stage1b_depth_light.yaml
/root/miniconda3/bin/python - "$dataset/manifests/conversion_report.json" \
  configs/stage1b_depth_light.yaml "$generated_config" <<'PY'
import json
import sys
from pathlib import Path

import yaml

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
maximum = max(split["depth_max"] for split in report.values())
dtypes = sorted({dtype for split in report.values() for dtype in split["depth_dtypes"]})
if maximum <= 0:
    raise SystemExit("RoboRefIt depth audit found no positive depth values")
config = yaml.safe_load(Path(sys.argv[2]).read_text(encoding="utf-8"))
if maximum <= 255 and all(dtype == "uint8" for dtype in dtypes):
    # The official RoboRefIt loader consumes its depth PNGs as 8-bit images.
    # Preserve that relative-depth signal instead of interpreting 255 as mm.
    config["data"]["depth_scale"] = 1.0
    config["data"]["depth_clip"] = 255.0
    encoding = "relative_uint8"
else:
    config["data"]["depth_scale"] = 1000.0
    config["data"]["depth_clip"] = 20.0
    encoding = "metric_mm"
Path(sys.argv[3]).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
print(json.dumps({"depth_max": maximum, "depth_dtypes": dtypes, "encoding": encoding}))
PY

export PYTHONPATH="$repo/src"
export HF_HOME=/root/autodl-tmp/aic/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

run_stage() {
  local name=$1
  local config=$2
  local checkpoint=$3
  local stage_log="$repo/logs/${name}.log"
  [[ ! -e "$checkpoint" ]] || {
    echo "refusing to overwrite an existing lightweight result: $checkpoint" >&2
    return 1
  }
  echo "$(date '+%F %T') starting $name config=$config" | tee -a "$pipeline_log"
  /root/miniconda3/bin/python -u train.py --config "$config" > "$stage_log" 2>&1
  [[ -s "$checkpoint" ]] || {
    echo "$name exited without checkpoint" >&2
    return 1
  }
  echo "$(date '+%F %T') completed $name" | tee -a "$pipeline_log"
  tail -n 8 "$stage_log" >> "$pipeline_log"
}

run_stage stage1b_depth_light "$generated_config" runs/stage1b_depth_light/best_phase_a.pt
run_stage stage2_joint_light configs/stage2_joint_light.yaml runs/stage2_joint_light/best_phase_a.pt
run_stage stage3_city_light configs/stage3_city_light.yaml runs/stage3_city_light/best_phase_a.pt

touch runs/LIGHTWEIGHT_PIPELINE_COMPLETE
echo "$(date '+%F %T') lightweight pipeline complete" | tee -a "$pipeline_log"
