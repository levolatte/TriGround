#!/usr/bin/env bash
set -uo pipefail

train_pid="${1:?training PID required}"
project="/root/autodl-tmp/final_training_upload/mm-grounding-adapters"
model="/root/autodl-tmp/aic/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203"
manifest="/root/autodl-tmp/final_training_upload/city_detection_prepared/train/grounding_final_val.json"
output="runs/qwen_rgb_full_val/results.jsonl"

resume_training() {
  if kill -0 "$train_pid" 2>/dev/null; then
    kill -CONT "$train_pid" 2>/dev/null || true
    echo "RESUMED_TRAINING pid=$train_pid at $(date '+%F %T %Z')"
  fi
}
trap resume_training EXIT INT TERM

if ! kill -0 "$train_pid" 2>/dev/null; then
  echo "training PID $train_pid is not running" >&2
  exit 2
fi

kill -STOP "$train_pid"
for _ in $(seq 1 20); do
  state=$(ps -o stat= -p "$train_pid" 2>/dev/null || true)
  case "$state" in
    T*) break ;;
  esac
  sleep 0.5
done
state=$(ps -o stat= -p "$train_pid" 2>/dev/null || true)
case "$state" in
  T*) ;;
  *) echo "failed to suspend training process; state=$state" >&2; exit 3 ;;
esac
echo "SUSPENDED_TRAINING pid=$train_pid at $(date '+%F %T %Z')"

cd "$project"
mkdir -p runs/qwen_rgb_full_val
PYTHONPATH=src \
HF_HOME=/root/autodl-tmp/aic/huggingface \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
/root/miniconda3/bin/python tools/eval_qwen_rgb_grounding.py \
  --model "$model" \
  --manifest "$manifest" \
  --output "$output" \
  --samples 0 \
  --seed 2026 \
  --min-pixels 200704 \
  --max-pixels 802816 \
  --max-new-tokens 40
