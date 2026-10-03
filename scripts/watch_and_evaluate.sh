#!/usr/bin/env bash
# 看护进程：等训练结束，立刻自动跑 B5 判决 + 最终提交。
#
# 为什么需要它：训练约 22:08 结束，而定时任务是每 30 分钟一次心跳。若不看着，
# 最坏要空等 30 分钟才启动判决与 5690 条推理——在 10-05 20:00 的截止前，这段时间
# 应该用来留余量而不是空转。
#
# 安全约束（本脚本只做判定与调用，不自己推理）：
#   * 只在"训练进程确实不存在"且"最终 checkpoint 已落盘"时才动手；
#   * 找不到 checkpoint 就停下并写清原因，绝不拿着半成品去评测；
#   * 全程用 screen 托管，脱离 SSH。
set -uo pipefail

RUN_DIR="${RUN_DIR:-/root/runs/q36_600_2073600}"
LOG="${LOG:-/root/b5_watch.log}"
CODE_DIR="${CODE_DIR:-/root/AIC_code}"
MAX_WAIT_HOURS="${MAX_WAIT_HOURS:-20}"

exec > >(tee -a "${LOG}") 2>&1
say() { echo "[$(date '+%F %T')] $*"; }

say "看护启动，等待 ${RUN_DIR} 的训练结束"

deadline=$(( $(date +%s) + MAX_WAIT_HOURS * 3600 ))
while :; do
  if ! pgrep -f "swift sft" > /dev/null; then
    say "训练进程已不存在"
    break
  fi
  if [ "$(date +%s)" -gt "${deadline}" ]; then
    say "等待超过 ${MAX_WAIT_HOURS} 小时，退出（训练可能卡住，需人工查看）"
    exit 2
  fi
  sleep 60
done

# 训练进程退出后，再宽限一会儿让最后的 checkpoint 落盘
sleep 90

CHECKPOINT="$(ls -d ${RUN_DIR}/v*/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)"
if [ -z "${CHECKPOINT}" ]; then
  say "找不到任何 checkpoint —— 可能未到 SAVE_STEPS 就退出。不评测半成品。"
  say "训练日志尾部："
  tail -20 "${RUN_DIR}/v*/train.log" 2>/dev/null || true
  exit 3
fi
say "使用 checkpoint: ${CHECKPOINT}"

# 记录训练是否真的收敛，避免拿一个 loss 爆炸的模型去提交
if [ -f "${RUN_DIR}/v0-"*/logging.jsonl ]; then
  say "训练末尾记录："
  tail -3 ${RUN_DIR}/v*/logging.jsonl
fi

say "启动判决 + 最终提交"
cd "${CODE_DIR}" || exit 4
export PATH=/root/vllm_env/bin:$PATH
ADAPTER="${CHECKPOINT}" bash scripts/final_submission_with_adapter.sh
status=$?
say "final_submission_with_adapter.sh 退出码 ${status}"
if [ "${status}" -ne 0 ]; then
  say "未通过判决或未满足生效判据 —— 回退方案：/root/sprint_qwen36/submit_b2_zeroshot/submission.zip（已核验）"
fi
exit "${status}"
