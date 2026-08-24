#!/usr/bin/env bash
set -uo pipefail

train_pid=4804
project=/root/autodl-tmp/final_training_upload/mm-grounding-adapters
model=/root/autodl-tmp/aic/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203
manifest=/root/autodl-tmp/final_training_upload/city_detection_prepared/train/grounding_final_val.json
output="$project/runs/qwen_rgb_only/val_subset_512.jsonl"

if ! kill -0 "$train_pid" 2>/dev/null; then
  echo "training PID $train_pid is not running; refusing to manage an unknown process" >&2
  exit 2
fi

resume_training() {
  if kill -0 "$train_pid" 2>/dev/null; then
    kill -CONT "$train_pid" 2>/dev/null || true
    echo "resumed training PID $train_pid at $(date --iso-8601=seconds)"
  fi
}

trap resume_training EXIT INT TERM
kill -STOP "$train_pid"
echo "paused training PID $train_pid at $(date --iso-8601=seconds)"
sleep 3
ps -p "$train_pid" -o pid,etime,stat,%cpu,%mem --no-headers
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader

cd "$project"
mkdir -p runs/qwen_rgb_only
export PYTHONPATH=src
export HF_HOME=/root/autodl-tmp/aic/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
/root/miniconda3/bin/python tools/eval_qwen_rgb_grounding.py \
  --model "$model" \
  --manifest "$manifest" \
  --output "$output" \
  --samples 512 \
  --seed 2026 \
  --min-pixels 200704 \
  --max-pixels 802816 \
  --max-new-tokens 40
status=$?
echo "RGB-only evaluation exit status: $status"
exit "$status"
