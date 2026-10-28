#!/usr/bin/env bash
# Qwen3.6/3.8-27B SFT LoRA -> 短框 RL，ms-swift 4.5.3 接口。
# 单96GB卡采用Transformers生成，与训练共用权重；GBS4不是16个三图输入同时生成。
# 每优化步：batch1 × accum16 = 16个response = G4下4次query呈现。
# 初次运行设置 MODEL / SFT_ADAPTER / DATASET / OUTPUT_DIR；ALGORITHM=gspo 切第二训练臂。
# DRY_RUN=1 仅输出完整命令，不加载ms-swift、不检查云端路径、不创建目录。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
MODEL="${MODEL:-/root/rematch_models/Qwen3.6-27B}"
SFT_ADAPTER="${SFT_ADAPTER:?Set SFT_ADAPTER to the cold-start LoRA checkpoint}"
DATASET="${DATASET:?Set DATASET to the prepared RL jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR to a fresh RL run directory}"
PY="${PY:-/root/miniconda3/bin/python}"
SWIFT_CLI="${SWIFT_CLI:-/root/miniconda3/bin/swift}"
ALGORITHM="${ALGORITHM:-dapo}"
REWARD_FUNC="${REWARD_FUNC:-acc05_bbox}"
MAX_STEPS="${MAX_STEPS:-200}"
MAX_PIXELS="${MAX_PIXELS:-602112}"
MIN_PIXELS="${MIN_PIXELS:-200704}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-64}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-4}"
ACCUM="${ACCUM:-16}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
BETA="${BETA:-0.01}"
TEMPERATURE="${TEMPERATURE:-0.8}"
SAVE_STEPS="${SAVE_STEPS:-50}"
SEED="${SEED:-2026}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
DRY_RUN="${DRY_RUN:-0}"

if (( GENERATION_BATCH_SIZE % NUM_GENERATIONS != 0 || ACCUM % GENERATION_BATCH_SIZE != 0 )); then
    echo 'GENERATION_BATCH_SIZE must be divisible by G; ACCUM must be divisible by GENERATION_BATCH_SIZE.' >&2
    exit 2
fi

case "${ALGORITHM}" in
    dapo)
        # DAPO式变体：token损失归一化、非对称clip；保留SFT KL和未缩放阈值奖励。
        ALGORITHM_ARGS=(--loss_type dapo --importance_sampling_level token
                        --epsilon 0.2 --epsilon_high 0.28 --scale_rewards none)
        ;;
    gspo)
        ALGORITHM_ARGS=(--loss_type grpo --importance_sampling_level sequence
                        --epsilon 3e-4 --epsilon_high 4e-4 --scale_rewards group)
        ;;
    *) echo "Unsupported ALGORITHM: ${ALGORITHM}" >&2; exit 2 ;;
esac

START_ARGS=(--adapters "${SFT_ADAPTER}")
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
    START_ARGS=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MAX_PIXELS MIN_PIXELS
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

COMMAND=("${SWIFT_CLI}" rlhf \
    --rlhf_type grpo \
    --model "${MODEL}" \
    --tuner_type lora \
    "${START_ARGS[@]}" \
    --ref_adapters "${SFT_ADAPTER}" \
    --ignore_args_error false \
    --dataset "${DATASET}" \
    --split_dataset_ratio 0 \
    --eval_strategy no \
    --external_plugins "${REPO_ROOT}/tools/swift_grounding_reward.py" \
    --reward_funcs "${REWARD_FUNC}" \
    "${ALGORITHM_ARGS[@]}" \
    --beta "${BETA}" \
    --use_vllm false \
    --num_generations "${NUM_GENERATIONS}" \
    --generation_batch_size "${GENERATION_BATCH_SIZE}" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps "${ACCUM}" \
    --num_iterations 1 \
    --dynamic_sample true \
    --max_resample_times 1 \
    --overlong_filter true \
    --enable_thinking false \
    --temperature "${TEMPERATURE}" \
    --top_p 0.95 \
    --top_k -1 \
    --repetition_penalty 1.0 \
    --max_completion_length "${MAX_COMPLETION_LENGTH}" \
    --max_length "${MAX_LENGTH}" \
    --truncation_strategy delete \
    --max_pixels "${MAX_PIXELS}" \
    --freeze_vit true \
    --freeze_aligner true \
    --torch_dtype bfloat16 \
    --attn_impl sdpa \
    --gradient_checkpointing true \
    --learning_rate "${LEARNING_RATE}" \
    --lr_scheduler_type constant \
    --warmup_ratio 0 \
    --weight_decay 0.0 \
    --max_grad_norm 1.0 \
    --max_steps "${MAX_STEPS}" \
    --save_strategy steps \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit 4 \
    --logging_steps 1 \
    --log_completions true \
    --dataloader_num_workers 4 \
    --dataloader_persistent_workers true \
    --dataset_num_proc 4 \
    --seed "${SEED}" \
    --report_to none \
    --output_dir "${OUTPUT_DIR}")

if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
    exit 0
fi

test -f "${MODEL}/config.json"
test -f "${SFT_ADAPTER}/adapter_config.json"
test -f "${DATASET}"
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
    test -f "${RESUME_FROM_CHECKPOINT}/trainer_state.json"
fi

# 只检查真实参数定义与环境，不创建模型。未知参数由CLI明确拒绝。
"${PY}" - <<'PY'
from importlib.metadata import version
from dataclasses import fields
from swift.arguments import RLHFArguments

required = {
    'adapters', 'ref_adapters', 'tuner_type', 'reward_funcs', 'external_plugins',
    'loss_type', 'importance_sampling_level', 'epsilon', 'epsilon_high', 'scale_rewards',
    'generation_batch_size', 'num_generations', 'num_iterations', 'dynamic_sample',
    'max_resample_times', 'overlong_filter', 'max_completion_length', 'enable_thinking',
}
missing = required - {f.name for f in fields(RLHFArguments)}
if missing:
    raise RuntimeError(f'Installed ms-swift lacks required RL fields: {sorted(missing)}')
print({name: version(name) for name in ('ms-swift', 'transformers', 'trl', 'torch')})
PY

mkdir -p "${OUTPUT_DIR}"
"${COMMAND[@]}" 2>&1 | tee "${OUTPUT_DIR}/train.log"
