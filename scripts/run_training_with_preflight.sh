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
TAG="${TAG:-$(echo "${PIXELS}" | awk '{printf "%dM", $1/1000000}')}"

PREFLIGHT_DIR="${PREFLIGHT_DIR:-/root/runs/q36_preflight_${TAG}}"
PREFLIGHT_LOG="${PREFLIGHT_LOG:-/root/train_preflight_${TAG}.log}"
TRAIN_DIR="${TRAIN_DIR:-/root/runs/q36_600_${TAG}}"
TRAIN_LOG="${TRAIN_LOG:-/root/train_q36_${TAG}.log}"

say() { echo "[$(date +%H:%M:%S)] $*"; }

cd "${CODE_DIR}" || exit 2

say "B3 训练前 4 步预检（max_pixels=${PIXELS}, max_length=${MAX_LEN}）"
rm -rf "${PREFLIGHT_DIR}"
MAX_STEPS=4 SAVE_STEPS=4 SAVE_TOTAL_LIMIT=1 \
OUTPUT_DIR="${PREFLIGHT_DIR}" \
MAX_PIXELS="${PIXELS}" MAX_LENGTH="${MAX_LEN}" \
  bash scripts/train_qwen36_27b_swift.sh > "${PREFLIGHT_LOG}" 2>&1
preflight_status=$?
say "预检退出码 ${preflight_status}，开始判读"

# 判读只看客观数字：可训练参数、峰值显存、每步耗时、是否 OOM
"${PY}" scripts/check_training.py "${PREFLIGHT_LOG}" --steps-budget 600
verdict=$?

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
