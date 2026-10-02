#!/usr/bin/env bash
# Qwen3.6-27B 原生三模态 LoRA 训练（ms-swift）。
#
# 与已验证基线 A 的唯一差别是「基座」与「每图像素预算」；其余超参逐项对齐 A：
#   语言侧 LoRA r32/alpha64/dropout0.05、lr 5e-6、batch1 x accum8、seed2026、
#   线性调度 warmup0、梯度裁剪 1.0、bf16 冻结主干、600 步、同一份 A.json 清单。
#
# 之所以换 ms-swift：旧训练栈 third_party/Qwen3-VL/qwen-vl-finetune 只支持 qwen3_vl，
# 而 Qwen3.6-27B 的 model_type 是 qwen3_5（官方 ms-swift 已支持）。
#
# 用法：
#   MAX_PIXELS=1204224 MAX_LENGTH=4096 OUTPUT_DIR=/root/runs/q36_27b_1m2 \
#     bash scripts/train_qwen36_27b_swift.sh
set -euo pipefail

MODEL="${MODEL:-/root/rematch_models/Qwen3.6-27B}"
DATASET="${DATASET:-/root/sprint_qwen36/a_city_swift.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR to a fresh run directory}"
PY="${PY:-/root/miniconda3/bin/python}"
SWIFT_CLI="${SWIFT_CLI:-/root/miniconda3/bin/swift}"

# 分辨率与上下文：602112(=588 tok/图) / 1204224(=1176) / 2073600(=2025, 原生 1920x1080)
MAX_PIXELS="${MAX_PIXELS:-1204224}"
MIN_PIXELS="${MIN_PIXELS:-200704}"
MAX_LENGTH="${MAX_LENGTH:-4096}"

# 与 A 对齐的超参
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
MAX_STEPS="${MAX_STEPS:-600}"
SEED="${SEED:-2026}"
ACCUM="${ACCUM:-8}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
TARGET_MODULES="${TARGET_MODULES:-all-linear}"

test -f "${MODEL}/config.json"
test -f "${DATASET}"

# 训练时图像会按该上限缩放（ms-swift 读 MAX_PIXELS/MIN_PIXELS 环境变量）
export MAX_PIXELS MIN_PIXELS
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${OUTPUT_DIR}"
echo "model=${MODEL}"
echo "dataset=${DATASET}"
echo "output=${OUTPUT_DIR}"
echo "max_pixels=${MAX_PIXELS}  min_pixels=${MIN_PIXELS}  max_length=${MAX_LENGTH}"
echo "lora r=${LORA_RANK} a=${LORA_ALPHA} dropout=${LORA_DROPOUT} target=${TARGET_MODULES}"

# LoRA 实际命中的模块名必须存档：Qwen3.5/3.6 是混合线性注意力（Gated DeltaNet + 全注意力），
# 投影层命名与 Qwen3-VL 的 q/k/v/o_proj 不同，不能沿用旧清单。
"${PY}" - <<EOF | tee "${OUTPUT_DIR}/target_modules.log"
import json
from transformers import AutoConfig
config = AutoConfig.from_pretrained("${MODEL}")
text = config.text_config
print("architectures:", config.architectures)
print("model_type:", config.model_type)
print("num_hidden_layers:", text.num_hidden_layers)
print("layer_types 分布:", {t: text.layer_types.count(t) for t in set(text.layer_types)})
EOF

"${SWIFT_CLI}" sft \
    --model "${MODEL}" \
    --train_type lora \
    --dataset "${DATASET}" \
    --split_dataset_ratio 0 \
    --torch_dtype bfloat16 \
    --attn_impl sdpa \
    --num_train_epochs 1 \
    --max_steps "${MAX_STEPS}" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps "${ACCUM}" \
    --learning_rate "${LEARNING_RATE}" \
    --lr_scheduler_type linear \
    --warmup_ratio 0 \
    --max_grad_norm 1.0 \
    --weight_decay 0.0 \
    --lora_rank "${LORA_RANK}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    --target_modules "${TARGET_MODULES}" \
    --max_length "${MAX_LENGTH}" \
    --gradient_checkpointing true \
    --save_strategy steps \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --logging_steps 1 \
    --dataloader_num_workers 0 \
    --dataset_num_proc 1 \
    --seed "${SEED}" \
    --report_to none \
    --output_dir "${OUTPUT_DIR}" \
    2>&1 | tee "${OUTPUT_DIR}/train.log"

echo "TRAIN_DONE"
