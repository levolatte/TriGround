#!/usr/bin/env bash
# 训练侧一键拉起：Arm 4（数据集修复）→ 三条流水线。
#
# ## 为什么单独写成脚本
#
# 实例克隆后 screen 会话全部消失，训练/归档/评估都得手工重建；而手敲长命令
# 在凌晨最容易漏参数（少一个 `SAVE_STEPS` 就会覆盖掉早先的检查点）。
# 把已在用的那条命令固化下来，比每次重新拼更可靠。
#
# ## 与推理侧的分工
#
# **只有一个 GPU**。推理侧（放大阶梯）需要约 35 分钟显存，本脚本因此先**等 GPU 空闲**
# 再开训，不去抢占；训练一旦开始会持续约 7 小时，推理侧应在那之后再跑。
#
# ## 为什么输出必须在数据盘
#
# overlay 只有约 10G，而每个检查点 2.7G、`save_total_limit=6` 需要约 16G——
# 放 overlay 会在保存检查点时写满根分区，且**不是启动就报错**，而是跑到一半才炸。
set -uo pipefail

CODE_DIR="${CODE_DIR:-/root/AIC_code}"
RUNS="${RUNS:-/root/autodl-tmp/runs/q36_arm4_matched}"
DATASET="${DATASET:-/root/sprint_qwen36/slices_arm4/mixture.jsonl}"
SWEEP_OUT="${SWEEP_OUT:-/root/sprint_qwen36/ckpt_sweep_arm4}"
# 对比基线：Arm 2 的最佳检查点（本地 314，官方 0.7870）
BASE_PRED="${BASE_PRED:-/root/sprint_qwen36/ckpt_sweep_arm2/checkpoint-500/predictions.jsonl}"
MAX_STEPS="${MAX_STEPS:-700}"
SAVE_STEPS="${SAVE_STEPS:-100}"
FREEZE_VIT="${FREEZE_VIT:-true}"
FREEZE_ALIGNER="${FREEZE_ALIGNER:-true}"
WAIT_GPU_SECONDS="${WAIT_GPU_SECONDS:-7200}"

say() { echo "[$(date '+%F %T')] $*"; }

cd "${CODE_DIR}" || exit 2

# ---- 前置检查：缺一个都别开训，开了也是白烧 7 小时 ----
for required in "${DATASET}" /root/rematch_models/Qwen3.6-27B/config.json \
                "${CODE_DIR}/scripts/train_qwen36_27b_swift.sh" \
                "${CODE_DIR}/scripts/archive_checkpoints.sh" \
                "${CODE_DIR}/scripts/after_arm2.sh"; do
  [ -e "${required}" ] || { say "缺少必需文件：${required}"; exit 3; }
done
say "前置检查通过（数据集 $(wc -l < "${DATASET}") 条）"

# ---- 等 GPU：推理侧可能正在跑放大阶梯，不抢 ----
say "等待 GPU 空闲（最多 ${WAIT_GPU_SECONDS} 秒）"
waited=0
while [ "${waited}" -lt "${WAIT_GPU_SECONDS}" ]; do
  busy="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)"
  [ "${busy}" -eq 0 ] && break
  if [ "${waited}" -eq 0 ]; then
    say "GPU 上有 ${busy} 个进程（可能是推理侧的放大阶梯），等待其结束"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | sed 's/^/    /'
  fi
  sleep 30
  waited=$((waited + 30))
done
busy="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)"
[ "${busy}" -eq 0 ] || { say "等待超时，GPU 仍被 ${busy} 个进程占用，退出"; exit 4; }
say "GPU 空闲"

# ---- 三条流水线 ----
# 训练：从零开始（上次只跑了 12 步、无检查点，续训没有意义）
rm -rf "${RUNS}"
screen -dmS train4 bash -c "cd ${CODE_DIR}; OUTPUT_DIR=${RUNS} DATASET=${DATASET} \
  MAX_PIXELS=2073600 MAX_LENGTH=8192 MIN_PIXELS=200704 \
  MAX_STEPS=${MAX_STEPS} SAVE_STEPS=${SAVE_STEPS} \
  FREEZE_VIT=${FREEZE_VIT} FREEZE_ALIGNER=${FREEZE_ALIGNER} \
  bash scripts/train_qwen36_27b_swift.sh > /root/train_arm4.log 2>&1"
say "已启动训练 train4"

sleep 15
[ -d "${RUNS}" ] || { say "训练目录未出现，训练可能启动失败"; }

# 归档看护：save_total_limit=6 会让训练器删掉早期检查点，
# 而 Arm 2 的教训正是"最佳不在最后一步"（ck500 优于 ck600）。
# 硬链接把它们保住，且不额外占空间。
screen -dmS ckptarch4 bash -c "RUN_DIR=${RUNS} ARCHIVE_DIR=${RUNS}/archived \
  LOG=/root/ckpt_archive_arm4.log bash ${CODE_DIR}/scripts/archive_checkpoints.sh"
say "已启动归档看护 ckptarch4"

# 评估链条：训练结束 → 逐检查点扫描 → 失败模式对比 → 出包 → 格式闸门
# （出包不因统计区间阻塞；那条门槛是我们自己加的且口径错误，已于 8b0c66b 修正）
screen -dmS arm4chain bash -c "RUN_DIR=${RUNS} ARCHIVE_DIR=${RUNS}/archived \
  SWEEP_OUT=${SWEEP_OUT} BASE_PRED=${BASE_PRED} \
  LOG=/root/after_arm4.log MAX_WAIT_HOURS=14 \
  exec bash ${CODE_DIR}/scripts/after_arm2.sh"
say "已启动评估链条 arm4chain"

sleep 10
echo
say "=== 当前 screen 会话 ==="
screen -ls | grep Detached
echo
say "=== 训练日志尾 ==="
tail -c 300 /root/train_arm4.log 2>/dev/null | tr '\r' '\n' | grep -avE '^\s*$' | tail -3
echo
say "=== 磁盘 ==="
df -h /root/autodl-tmp / | tail -2
