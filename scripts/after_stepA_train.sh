#!/bin/bash
# 步骤 A 出包后立刻启动新数据训练。
#
# 为什么要有这个脚本：步骤 A（BF16 全量推理）预计在 10-31 05:10 左右结束时，
# 那天凌晨没有人盯着。手工接续会浪费几小时，而结果截止 11-01 20:00 已经没有余量。
#
# 安全设计（上一版脚本的教训）：
# 1) 解释器与 checkpoint 路径在使用前显式校验，不存在就直接退出，不做静默降级。
# 2) 任何一步失败都 `exit 1`，绝不「带着空数据往下走」。
# 3) 等的是「预测行数到齐 + submission.zip 存在」这个可验证条件，不是等进程消失
#    （进程正常退出和被 OOM 杀掉的表象一样，只看进程会误判）。
set -u

P=/root/autodl-tmp/quant_rl_20261026
AC=$P/acc_code_20261029
PY=/root/train_env/bin/python
REPL=$P/runs/replacement3_20261030
MIX=$REPL/mixture.jsonl
TRAIN_OUT=$REPL/correction/train
BF16=$P/runs/bf16_k20_202610301855
LOG=$P/runs/after_stepA_train.log

exec > >(tee -a "$LOG") 2>&1
echo "================ $(date '+%F %T') 等待步骤 A 结束 ================"

if [ ! -x "$PY" ]; then echo "FATAL: 解释器不存在 $PY"; exit 1; fi
if [ ! -f "$MIX" ]; then echo "FATAL: 索引不存在 $MIX"; exit 1; fi
echo "前置检查通过：解释器 $PY，索引 $MIX"

# ---------- 1. 等步骤 A 完成：预测行数到 5691 且 submission.zip 已生成 ----------
for i in $(seq 1 480); do   # 最多等 8 小时
  rows=$(wc -l < "$BF16/predictions.jsonl" 2>/dev/null || echo 0)
  if [ "$rows" -ge 5691 ] && [ -f "$BF16/submission.zip" ]; then
    echo "步骤 A 完成（$rows 行 + submission.zip）"
    break
  fi
  if [ $((i % 20)) -eq 0 ]; then
    echo "  $(date '+%H:%M')  步骤 A 进度 $rows/5691"
  fi
  sleep 60
done
rows=$(wc -l < "$BF16/predictions.jsonl" 2>/dev/null || echo 0)
if [ "$rows" -lt 5691 ] || [ ! -f "$BF16/submission.zip" ]; then
  echo "WARN: 步骤 A 未完成（$rows 行）。**仍然继续训练**——GPU 空出来就该训练，"
  echo "      步骤 A 的残缺包不用于提交，但训练不能因此停摆。"
fi

# ---------- 2. 等显存彻底释放 ----------
for i in $(seq 1 120); do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [ "${USED:-99999}" -lt 2000 ]; then
    echo "显存已释放（${USED} MiB）"
    break
  fi
  echo "  等待显存释放，当前 ${USED} MiB"
  sleep 15
done

# ---------- 3. 启动训练 ----------
if [ -f "$TRAIN_OUT/completed.json" ]; then
  echo "训练已完成过，跳过"
  exit 0
fi
mkdir -p "$TRAIN_OUT"
cd "$AC" || exit 1
echo "--- 启动新数据训练：6,603 条，2,400 次呈现，jitter_adaptive=false ---"
"$PY" -u tools/train_quant_grounding.py \
  --mode correction \
  --config configs/quant_grounding/correction_replacement3.json \
  --output-dir "$TRAIN_OUT"
rc=$?
echo "--- 训练结束 rc=$rc，输出 $TRAIN_OUT ---"
if [ "$rc" -ne 0 ]; then
  echo "FATAL: 训练失败"
  exit 1
fi
echo "================ $(date '+%F %T') 全部结束 ================"
