#!/usr/bin/env bash
# B3 → B4 自动衔接：先跑 4 步预检，用客观判据通过后才启动正式训练。
#
# 为什么要写成脚本而不是由人手动敲两条命令：
#   * 4 步预检的产出是日志，靠肉眼读容易漏掉"其实没在训 LoRA"或"已贴近显存上限"；
#   * 预检不通过就必须停，绝不能"看起来还行就开训"，否则几小时后才发现问题。
#
# 用法：
#   PIXELS=1204224 MAX_LEN=4096 bash scripts/run_training_with_preflight.sh
#   （预检通过后自动以后台方式启动 600 步正式训练）
set -uo pipefail

PY=/root/miniconda3/bin/python
CODE_DIR="${CODE_DIR:-/root/AIC_code}"
PIXELS="${PIXELS:-1204224}"
MAX_LEN="${MAX_LEN:-4096}"
# 用像素数本身做标签，不要格式化——曾用 printf "%dM" 把 2073600 截断成 "2M"，
# 结果连自己都找错目录。
TAG="${TAG:-${PIXELS}}"

PREFLIGHT_DIR="${PREFLIGHT_DIR:-/root/runs/q36_preflight_${TAG}}"
PREFLIGHT_LOG="${PREFLIGHT_LOG:-/root/train_preflight_${TAG}.log}"
TRAIN_DIR="${TRAIN_DIR:-/root/runs/q36_600_${TAG}}"
TRAIN_LOG="${TRAIN_LOG:-/root/train_q36_${TAG}.log}"
STEPS_BUDGET="${STEPS_BUDGET:-600}"
# 单卡 83.6GiB 上 600 步的合理上限。超过就必须先降分辨率或减步数，不能直接开跑。
MAX_HOURS="${MAX_HOURS:-12}"

say() { echo "[$(date +%H:%M:%S)] $*"; }

cd "${CODE_DIR}" || exit 2

say "B3 训练前 4 步预检（max_pixels=${PIXELS}, max_length=${MAX_LEN}）"
rm -rf "${PREFLIGHT_DIR}"
MAX_STEPS=4 SAVE_STEPS=4 SAVE_TOTAL_LIMIT=1 \
OUTPUT_DIR="${PREFLIGHT_DIR}" \
MAX_PIXELS="${PIXELS}" MAX_LENGTH="${MAX_LEN}" \
  bash scripts/train_qwen36_27b_swift.sh > "${PREFLIGHT_LOG}" 2>&1
preflight_status=$?

# ms-swift 4.5.3 不打印 trainable params 行，改用预检真正落盘的 adapter 来确认 LoRA 生效
LATEST_CHECKPOINT="$(find "${PREFLIGHT_DIR}" -maxdepth 1 -type d -name 'v*' | sort | tail -1)"
say "预检退出码 ${preflight_status}，开始判读"

# 判读只看客观数字：可训练参数、峰值显存、**外推总时长**、是否 OOM。
# --max-hours 是这里最要紧的一道闸：本机实测原生分辨率下 27B 约 180 秒/步，
# 600 步即 30 小时——这种量级必须在开训前拦住，而不是跑完一天才发现。
"${PY}" scripts/check_training.py "${PREFLIGHT_LOG}" --steps-budget "${STEPS_BUDGET}" \
  --max-hours "${MAX_HOURS}" --adapter-dir "${LATEST_CHECKPOINT}" 2>&1 | tee "${PREFLIGHT_LOG%.log}.verdict.txt"
verdict=${PIPESTATUS[0]}

if [ "${preflight_status}" -ne 0 ] || [ "${verdict}" -ne 0 ]; then
  say "预检未通过，按纪律不启动正式训练"
  echo "--- 日志尾部"
  tail -30 "${PREFLIGHT_LOG}"
  exit 1
fi

say "预检通过，启动 B4 正式训练 600 步 -> ${TRAIN_DIR}"
rm -rf "${TRAIN_DIR}"
setsid bash -c "cd ${CODE_DIR} && OUTPUT_DIR='${TRAIN_DIR}' MAX_PIXELS='${PIXELS}' MAX_LENGTH='${MAX_LEN}' \
  bash scripts/train_qwen36_27b_swift.sh > '${TRAIN_LOG}' 2>&1" < /dev/null > /dev/null 2>&1 &
say "已后台启动，日志 ${TRAIN_LOG}"
echo "TRAINING_LAUNCHED"
