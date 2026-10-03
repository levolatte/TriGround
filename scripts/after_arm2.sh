#!/usr/bin/env bash
# 收尾链条（各臂通用，用环境变量区分）：等训练结束 → 逐检查点扫描 → 失败模式对比 → 出包 → 格式闸门。
#
# 与 Arm 1 链条（`after_arm1_then_arm2.sh`）的区别只有两点，但都很关键：
#
# 1. **多一道副判据**：计划规定"小目标桶命中数不得下降"。总体涨而极小桶跌，
#    等于用易题换难题——只看总数看不出来。因此这里显式跑 `report_failure_modes.py`
#    把 Arm 1 与 Arm 2 的分桶结果对照。
# 2. **不启动下一臂**：Arm 2 之后没有排队的训练，GPU 释放出来供人工决策。
set -uo pipefail

PY=/root/miniconda3/bin/python
CODE_DIR="${CODE_DIR:-/root/AIC_code}"
RUN_DIR="${RUN_DIR:-/root/autodl-tmp/runs/q36_arm2_capability}"
ARCHIVE_DIR="${ARCHIVE_DIR:-${RUN_DIR}/archived}"
SWEEP_OUT="${SWEEP_OUT:-/root/sprint_qwen36/ckpt_sweep_arm2}"
# 对比基线用**零样本 27B**，而不是原计划的 Arm 1：Arm 1（City-only）因实例迁移未跑完，
# 没有可用的预测。零样本 27B 在 City412 上是 297 命中，正是本轮要超越的对象。
BASE_PRED="${BASE_PRED:-/root/sprint_qwen36/probes/q36_27b_zeroshot/pixels_2073600/predictions.jsonl}"
GT="${GT:-/root/sprint_qwen36/baseline_preds/city412_gt.json}"
LOG="${LOG:-/root/after_arm2.log}"
MAX_WAIT_HOURS="${MAX_WAIT_HOURS:-12}"

exec > >(tee -a "${LOG}") 2>&1
say() { echo "[$(date '+%F %T')] $*"; }
cd "${CODE_DIR}" || exit 2
export PATH=/root/vllm_env/bin:$PATH

say "等待 Arm 2 训练开始（目录 ${RUN_DIR} 出现）"
#
# 两道等待缺一不可：只等 `swift sft` 退出的话，**Arm 1 结束就会误触发**
# （那时 Arm 2 还没开始）。所以先等 Arm 2 的运行目录出现，再等训练进程退出。
start_deadline=$(( $(date +%s) + MAX_WAIT_HOURS * 3600 ))
while [ ! -d "${RUN_DIR}" ]; do
  if [ "$(date +%s)" -gt "${start_deadline}" ]; then
    say "等待 ${MAX_WAIT_HOURS} 小时仍未见 ${RUN_DIR} —— Arm 2 可能未启动，退出"
    exit 2
  fi
  sleep 60
done
say "运行目录已出现，等待训练结束"
deadline=$(( $(date +%s) + MAX_WAIT_HOURS * 3600 ))
while pgrep -f "swift sft" > /dev/null; do
  if [ "$(date +%s)" -gt "${deadline}" ]; then
    say "等待超过 ${MAX_WAIT_HOURS} 小时，退出（训练可能卡住）"; exit 2
  fi
  sleep 60
done
say "训练结束，等 120 秒让最后的检查点落盘并被归档"
sleep 120

mapfile -t CHECKPOINTS < <(ls -d "${ARCHIVE_DIR}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n)
if [ "${#CHECKPOINTS[@]}" -eq 0 ]; then
  say "没有已归档的检查点 —— 不评测半成品"
  say "回退：/root/sprint_qwen36/submit_b2_zeroshot/submission.zip（官方 0.7685）"
  exit 3
fi
say "已归档 ${#CHECKPOINTS[@]} 个：$(basename -a "${CHECKPOINTS[@]}" | tr '\n' ' ')"

say "逐检查点在 City412 上扫描"
ARCHIVE_DIR="${ARCHIVE_DIR}" OUT_ROOT="${SWEEP_OUT}" \
  bash scripts/evaluate_checkpoints.sh || say "扫描失败，稍后用最后一个检查点"

LAST="$(basename "$(printf '%s\n' "${CHECKPOINTS[@]}" | tail -1)")"
BEST="${LAST}"
if [ -f "${SWEEP_OUT}/sweep.json" ]; then
  BEST="$("${PY}" - <<PYEOF
import json, pathlib
sweep = json.loads(pathlib.Path("${SWEEP_OUT}/sweep.json").read_text(encoding="utf-8"))
arms = {n: d["hits_0.5"] for n, d in sweep["arms"].items()}
print(max(arms, key=lambda n: arms[n]))
PYEOF
)"
fi
say "最佳 ${BEST}（最后一个是 ${LAST}）"

# ---- 副判据：与对比基线按失败模式对比 ----------------------------------------
# 臂名用 `base` / `arm3` 而不是 `zeroshot` / `arm2`：`BASE_PRED` 现在指向的是
# **Arm 2 的最佳检查点**（本地 314），不是零样本（297）。名字与实际不符会让
# 读 `failure_modes.json` 的人误判对照对象——比较逻辑本身是对的，只是标签错了。
say "按失败模式对比（对比基线 vs 本轮）"
ARM3_BEST_PRED="${SWEEP_OUT}/${BEST}/predictions.jsonl"
if [ -f "${BASE_PRED}" ] && [ -f "${ARM3_BEST_PRED}" ]; then
  "${PY}" tools/report_failure_modes.py --gt "${GT}" \
    --arm base="${BASE_PRED}" --arm arm3="${ARM3_BEST_PRED}" \
    --out-json "${SWEEP_OUT}/failure_modes.json" || say "失败模式报告出错"
  TINY_REGRESSED="$("${PY}" - <<PYEOF
import json, pathlib
p = pathlib.Path("${SWEEP_OUT}/failure_modes.json")
if not p.is_file():
    print("unknown"); raise SystemExit
d = json.loads(p.read_text(encoding="utf-8"))["buckets"]
tiny = d["base"].get("极小 <0.2%", {}).get("命中", 0) - d["arm3"].get("极小 <0.2%", {}).get("命中", 0)
print("yes" if tiny > 0 else "no")
PYEOF
)"
  say "极小目标桶是否退化: ${TINY_REGRESSED}"
  if [ "${TINY_REGRESSED}" = "yes" ]; then
    say "⚠️ 极小目标桶相对对比基线退化 —— 按计划这属于'用易题换难题'。"
    say "   仍然会生成候选提交包（本地与官方的映射关系很弱，官方才是真裁判），"
    say "   但**不要默认提交它**，请先看下面的分桶对比再决定。"
  fi
else
  say "缺少对比基线预测（${BASE_PRED}），跳过副判据"
  TINY_REGRESSED="unknown"
fi

# ---- 主判据 + 提交 ----------------------------------------------------------
#
# 只用**扫描最佳**跑提交，不再"先试最后一个、失败再试最佳"。
#
# 原因：出包这一步现在不因区间被判失败（见 final_submission_with_adapter.sh 的说明），
# 所以"先试最后一个"会**为最后一个检查点白跑一次 5690**（约 1.6 小时），
# 而且它本地分数更低。直接对最佳检查点出包既省时又是我们真正想提交的那个。
#
# 代价要写明：**按本地分数挑检查点是在开发集上做选择**，本地数字会偏乐观。
# 这是有意的权衡——官方才是真裁判，而 Arm 2 已证明本地确实能选出比"最后一个"更好的
# （ck500 本地 314 > ck600 的 311，官方 0.7870）。
say "用扫描最佳 ${BEST} 出提交包"
ADAPTER="${ARCHIVE_DIR}/${BEST}" DEV_OUT="${SWEEP_OUT}/verdict_${BEST}" \
  bash scripts/final_submission_with_adapter.sh
status=$?

if [ "${status}" -ne 0 ]; then
  say "⚠️ 出包流程返回 ${status} —— 请看上面的日志定位是推理、闸门还是显存问题"
  say "   回退包仍在：/root/sprint_qwen36/submit_b2_zeroshot/submission.zip（官方 0.7685）"
  exit "${status}"
fi

if [ "${TINY_REGRESSED}" = "yes" ]; then
  say "⚠️⚠️ 总体达标但**极小目标桶退化** —— 按计划这属于'用易题换难题'"
  say "   包已生成，但**不要默认提交它**，请先看失败模式对比再决定"
  exit 5
fi

say "✅ 出包完成；极小目标桶未退化"
say "对照基线：零样本 297 / Arm 2 ck500 314（官方 0.7870）"
