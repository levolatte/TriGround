#!/usr/bin/env bash
# 首轮辅助数据 A/B：相同初始化和超参，每次只运行 ARM 指定的一臂。
# DATA_DIR 中需有 A.jsonl、B.jsonl；ADAPTER 为 Arm2 step500 的 LoRA 权重。
# SMOKE=1 各臂只取前 16 条、训练 2 步，写入独立 smoke 目录。
# PREFLIGHT_ONLY=1 只用 Qwen3.6 processor/template 检查两份完整清单，不加载模型权重。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${DATA_DIR:?Set DATA_DIR to the prepared A/B jsonl directory}"
MODEL="${MODEL:-/root/rematch_models/Qwen3.6-27B}"
PY="${PY:-/root/miniconda3/bin/python}"
test -f "${MODEL}/config.json"
test -f "${DATA_DIR}/A.jsonl"
test -f "${DATA_DIR}/B.jsonl"
export DATA_DIR MODEL PY
export MAX_PIXELS=2073600 MIN_PIXELS=200704 MAX_LENGTH=8192

"${PY}" - <<'PY'
import json
import os
from pathlib import Path
from statistics import median
from swift import get_processor, get_template

data_dir = Path(os.environ['DATA_DIR'])
processor = get_processor(os.environ['MODEL'], model_type='qwen3_5', download_model=False)
# max_length=None is deliberate: count the full Qwen3.6 tokenization *before*
# comparing with the training limit. The template still resizes images as in SFT.
template = get_template(processor, template_type='qwen3_5', max_length=None,
                        max_pixels=int(os.environ['MAX_PIXELS']),
                        loss_scale='default+ignore_empty_think', padding_free=False)
assert template.template_meta.template_type == 'qwen3_5', template.template_meta.template_type
assert template.loss_scale.is_binary_loss_scale
template.set_mode('train')
print(f'template={template.template_meta.template_type} image_token_id={template.image_token_id}')
cache = {}
for arm in ('A', 'B'):
    lengths, image_tokens, answer_tokens = [], [], []
    with (data_dir / f'{arm}.jsonl').open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            if any('loss_scale' in message for message in row['messages']):
                raise ValueError(f'{arm} line {line_number}: scalar message loss_scale is forbidden')
            key = json.dumps([row['images'], row['messages']], ensure_ascii=False, sort_keys=True)
            if key not in cache:
                encoded = template.encode(row)
                labels = encoded['labels']
                length = len(encoded['input_ids'])
                n_answer = sum(label != -100 for label in labels[1:])
                n_image = encoded['input_ids'].count(template.image_token_id)
                if (encoded.get('loss_scale') is not None or n_answer == 0
                        or labels[0] != -100 or labels[-1] == -100):
                    raise ValueError(f'{arm} line {line_number}: missing binary supervision or final EOS')
                cache[key] = (length, n_image, n_answer)
            length, n_image, n_answer = cache[key]
            if length > int(os.environ['MAX_LENGTH']):
                raise ValueError(f'{arm} line {line_number}: raw length {length} exceeds max_length')
            lengths.append(length)
            image_tokens.append(n_image)
            answer_tokens.append(n_answer)
            if line_number % 100 == 0:
                print(f'{arm} encoded {line_number}', flush=True)
    if len(lengths) != 1920:
        raise ValueError(f'{arm}: expected 1920 rows, got {len(lengths)}')
    print(f'{arm}: rows={len(lengths)} unique_encoded_so_far={len(cache)} length(min/median/max)={min(lengths)}/{median(lengths)}/{max(lengths)} '
          f'image_tokens={min(image_tokens)}/{median(image_tokens)}/{max(image_tokens)} '
          f'answer_tokens={min(answer_tokens)}/{median(answer_tokens)}/{max(answer_tokens)}', flush=True)
PY

if [ "${PREFLIGHT_ONLY:-0}" = 1 ]; then exit 0; fi
: "${ADAPTER:?Set ADAPTER to the Arm2 step500 adapter directory}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to a fresh experiment root}"
ARM="${ARM:?Set ARM=A or ARM=B}"
case "${ARM}" in
  A|B) ;;
  *) echo "ARM must be A or B" >&2; exit 2 ;;
esac
if [ -n "${RESUME_FROM_CHECKPOINT:-}" ]; then
  echo "This experiment starts from --adapters with a fresh optimizer, not --resume_from_checkpoint" >&2
  exit 2
fi

DATASET="${DATA_DIR}/${ARM}.jsonl"
if [ "${SMOKE:-0}" = 1 ]; then
  SMOKE_DATASET="$(mktemp --suffix=.jsonl)"
  trap 'rm -f "${SMOKE_DATASET}"' EXIT
  head -n 16 "${DATASET}" > "${SMOKE_DATASET}"
  DATASET="${SMOKE_DATASET}"
  OUTPUT_DIR="${OUTPUT_ROOT}/${ARM,,}-smoke"
  MAX_STEPS=2
  SAVE_STEPS=2
else
  OUTPUT_DIR="${OUTPUT_ROOT}/${ARM,,}"
  MAX_STEPS=240
  SAVE_STEPS=80
fi

test -f "${ADAPTER}/adapter_config.json"
if [ -e "${OUTPUT_DIR}" ]; then
  echo "Output directory already exists: ${OUTPUT_DIR}" >&2; exit 2
fi

export DATASET OUTPUT_DIR MAX_STEPS SAVE_STEPS
export LORA_RANK=32 LORA_ALPHA=64 LORA_DROPOUT=0.05
export LEARNING_RATE=5e-6 PER_DEVICE_BATCH=1 ACCUM=8
export FREEZE_VIT=true FREEZE_ALIGNER=true
export SEED=2026 DATA_SEED=2026
export DATASET_SHUFFLE=false TRAIN_DATALOADER_SHUFFLE=false GROUP_BY_LENGTH=false
export PACKING=false PADDING_FREE=false
export LOSS_SCALE=default+ignore_empty_think
export EXTERNAL_PLUGINS="${REPO_ROOT}/tools/swift_sample_mean_loss.py"
export LOSS_TYPE=grounding_sample_mean
export RESUME_FROM_CHECKPOINT=""

bash "${REPO_ROOT}/scripts/train_qwen36_27b_swift.sh"
