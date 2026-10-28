#!/usr/bin/env bash
# ms-swift 4.5.3 grounding SFT. ARM=language or vision_language.
# MODEL is a full base/SFT checkpoint; ADAPTER optionally continues an existing
# LoRA with a fresh optimizer. RESUME_FROM_CHECKPOINT instead restores full state.
# Short-box training disables thinking by default. Set ENABLE_THINKING=true only
# for a dedicated experiment whose assistant targets retain reasoning traces.
# Example: MODEL=/models/EGM-8B-SFT DATASET=/data/train.jsonl \
#   OUTPUT_DIR=/runs/sft ARM=language DRY_RUN=1 bash scripts/train_grounding_sft.sh
set -euo pipefail

: "${MODEL:?Set MODEL to a full base or SFT model directory}"
: "${DATASET:?Set DATASET to the prepared train.jsonl}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a fresh run directory}"
ARM="${ARM:-language}"
case "${ARM}" in
  language) FREEZE_VIT=true; FREEZE_ALIGNER=true ;;
  vision_language) FREEZE_VIT=false; FREEZE_ALIGNER=false ;;
  *) echo "ARM must be language or vision_language" >&2; exit 2 ;;
esac
if [[ -n "${ADAPTER:-}" && -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  echo "Choose ADAPTER (weights only) or RESUME_FROM_CHECKPOINT (complete state)" >&2
  exit 2
fi
if [[ "${ARM}" == vision_language && -n "${ADAPTER:-}" ]]; then
  echo "vision_language needs a full/merged MODEL and a fresh LoRA; loading ADAPTER retains its old target modules" >&2
  exit 2
fi

export MAX_PIXELS="${MAX_PIXELS:-1204224}"
export MIN_PIXELS="${MIN_PIXELS:-200704}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cmd=("${SWIFT_CLI:-swift}" sft
  --model "${MODEL}" --dataset "${DATASET}" --output_dir "${OUTPUT_DIR}"
  --enable_thinking "${ENABLE_THINKING:-false}"
  --tuner_type lora --target_modules all-linear
  --freeze_vit "${FREEZE_VIT}" --freeze_aligner "${FREEZE_ALIGNER}"
  --lora_rank "${LORA_RANK:-32}" --lora_alpha "${LORA_ALPHA:-64}" --lora_dropout "${LORA_DROPOUT:-0.05}"
  --torch_dtype bfloat16 --attn_impl "${ATTN_IMPL:-sdpa}"
  --learning_rate "${LEARNING_RATE:-5e-6}" --lr_scheduler_type cosine --warmup_ratio 0.03
  --weight_decay 0 --max_grad_norm 1.0 --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}"
  --max_steps "${MAX_STEPS:-100}"
  --per_device_train_batch_size "${PER_DEVICE_BATCH:-1}" --gradient_accumulation_steps "${ACCUM:-8}"
  --max_length "${MAX_LENGTH:-8192}" --max_pixels "${MAX_PIXELS}"
  --gradient_checkpointing true --split_dataset_ratio 0
  --save_strategy steps --save_steps "${SAVE_STEPS:-50}" --save_total_limit 3
  --logging_steps 1 --dataloader_num_workers "${NUM_WORKERS:-4}"
  --dataset_num_proc "${DATASET_NUM_PROC:-4}" --seed "${SEED:-2026}"
  --report_to none --ignore_args_error false --strict true)
if [[ -n "${ADAPTER:-}" ]]; then
  cmd+=(--adapters "${ADAPTER}")
fi
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  cmd+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi
if [[ -n "${VAL_DATASET:-}" ]]; then
  cmd+=(--val_dataset "${VAL_DATASET}" --eval_strategy steps --eval_steps "${SAVE_STEPS:-50}")
fi
printf '%q ' "${cmd[@]}"
printf '\n'
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  exit 0
fi
mkdir -p "${OUTPUT_DIR}"
"${cmd[@]}" 2>&1 | tee "${OUTPUT_DIR}/train.log"
