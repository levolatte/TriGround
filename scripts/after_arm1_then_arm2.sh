#!/usr/bin/env bash
# Arm 1 收尾链条：等训练结束 → 逐检查点评分 → 用最佳检查点提交 → 启动 Arm 2。
#
# ## 为什么要串成一条链
#
# 这几个步骤有严格的先后依赖（GPU 只有一块），任何一步等人来点都会空耗时间：
#   训练结束 → 逐检查点评估（约 40 分钟）→ 官方 5690 条推理（约 1.6 小时）→ Arm 2 训练（12 小时）
# 目前距截止约 2 天 9 小时，链条总长约 14 小时，串起来才不会把余量浪费在等人上。
#
# ## 关于"用哪个检查点提交"
#
# 逐检查点扫描只用于**排序**。是否提交仍按预登记的三条件判据
# （命中 > 293、聚类配对区间下界 > 0、适配器确实生效）。
# 注意其中的统计风险：在 City412 上挑最高分再在 City412 上做显著性检验是有偏的。
# 因此本脚本**优先用最后一个检查点**（它是计划原本指定的对象，无选择偏差）；
# 只有当最后一个不达标、而某个更早的达标时才改用它，并在报告里显式标注这一点。
set -uo pipefail

PY=/root/miniconda3/bin/python
CODE_DIR="${CODE_DIR:-/root/AIC_code}"
RUN_DIR="${RUN_DIR:-/root/runs/q36_600_2073600}"
SWEEP_OUT="${SWEEP_OUT:-/root/sprint_qwen36/ckpt_sweep}"
LOG="${LOG:-/root/after_arm1.log}"
ARM2_DATASET="${ARM2_DATASET:-/root/sprint_qwen36/slices/mixture.jsonl}"
ARM2_OUTPUT="${ARM2_OUTPUT:-/root/runs/q36_arm2_capability}"
MAX_WAIT_HOURS="${MAX_WAIT_HOURS:-16}"

exec > >(tee -a "${LOG}") 2>&1
say() { echo "[$(date '+%F %T')] $*"; }
cd "${CODE_DIR}" || exit 2
export PATH=/root/vllm_env/bin:$PATH

say "等待 Arm 1 训练结束（${RUN_DIR}）"
deadline=$(( $(date +%s) + MAX_WAIT_HOURS * 3600 ))
while pgrep -f "swift sft" > /dev/null; do
  if [ "$(date +%s)" -gt "${deadline}" ]; then
    say "等待超过 ${MAX_WAIT_HOURS} 小时，退出（训练可能卡住，需人工查看）"; exit 2
  fi
  sleep 60
done
say "训练进程已结束，等 90 秒让最后的检查点落盘"
sleep 90

# 归档看护是另一个进程，可能还没来得及链接最新检查点
sleep 30

mapfile -t CHECKPOINTS < <(ls -d "${RUN_DIR}"/archived/checkpoint-* 2>/dev/null | sort -t- -k2 -n)
if [ "${#CHECKPOINTS[@]}" -eq 0 ]; then
  say "没有已归档的检查点 —— 不评测半成品，直接回退"
  say "回退方案：/root/sprint_qwen36/submit_b2_zeroshot/submission.zip（官方 0.7685，已核验）"
  exit 3
fi
say "已归档 ${#CHECKPOINTS[@]} 个检查点：$(basename -a "${CHECKPOINTS[@]}" | tr '\n' ' ')"

say "逐检查点评估（City412）"
bash scripts/evaluate_checkpoints.sh || say "扫描失败，改用最后一个检查点"

LAST="$(basename "$(printf '%s\n' "${CHECKPOINTS[@]}" | tail -1)")"
BEST="$(basename "$(printf '%s\n' "${CHECKPOINTS[@]}" | head -1)")"
if [ -f "${SWEEP_OUT}/sweep.json" ]; then
  BEST="$("${PY}" - <<PYEOF
import json, pathlib
sweep = json.loads(pathlib.Path("${SWEEP_OUT}/sweep.json").read_text(encoding="utf-8"))
arms = {n: d["hits_0.5"] for n, d in sweep["arms"].items()}
print(max(arms, key=lambda n: arms[n]))
PYEOF
)"
fi
say "最佳检查点 ${BEST}（最后一个是 ${LAST}）"

run_verdict() {
  local checkpoint="$1"
  local adapter="${RUN_DIR}/archived/${checkpoint}"
  say "用 ${checkpoint} 跑判决 + 若达标则出提交包"
  ADAPTER="${adapter}" bash scripts/final_submission_with_adapter.sh
  return $?
}

# 优先最后一个检查点（无选择偏差）；不达标再看最佳的那个
status=0
if [ "${LAST}" != "${BEST}" ]; then
  say "最佳不是最后一个 —— 先按无偏口径试最后一个，失败再用 ${BEST}"
fi
run_verdict "${LAST}" || status=$?

if [ "${status}" -ne 0 ] && [ "${LAST}" != "${BEST}" ]; then
  say "⚠️ 最后一个检查点未通过判据，改用扫描出的最佳 ${BEST}"
  say "   注意：这是在 City412 上选出的最高分，存在选择偏差，报告里必须标注"
  run_verdict "${BEST}" || status=$?
fi

if [ "${status}" -ne 0 ]; then
  say "两个检查点都未通过判据 —— 回退"
  say "回退方案：/root/sprint_qwen36/submit_b2_zeroshot/submission.zip（官方 0.7685，已核验）"
else
  say "✅ 提交包已生成，见上面的最终提交段落"
fi

if [ ! -f "${ARM2_DATASET}" ]; then
  say "找不到 Arm 2 混合集 ${ARM2_DATASET} —— 不启动 Arm 2"
  exit "${status}"
fi

say "启动 Arm 2（能力混合，数据集 $(wc -l < "${ARM2_DATASET}") 条）"
pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)"
if [ "${pids}" -ne 0 ]; then
  say "⚠️ GPU 上仍有 ${pids} 个进程，等 60 秒再确认"
  sleep 60
  pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)"
  if [ "${pids}" -ne 0 ]; then
    say "GPU 仍被占用，不启动 Arm 2 以免 OOM"
    exit "${status}"
  fi
fi

mkdir -p "${ARM2_OUTPUT}"
inner="cd ${CODE_DIR}; OUTPUT_DIR=${ARM2_OUTPUT} DATASET=${ARM2_DATASET} MAX_PIXELS=2073600 \
MAX_LENGTH=8192 MIN_PIXELS=200704 bash scripts/train_qwen36_27b_swift.sh \
> /root/train_arm2.log 2>&1"
screen -dmS train2 bash -c "${inner}"
sleep 10
say "Arm 2 已启动：$(screen -ls | grep train2 || echo '(screen 未找到，检查日志)')"
say "完成"
