#!/bin/bash
# 抖动重训的收尾：评外部88 → 配对过闸 → 用选中的适配器启动 BF16 全量推理。
#
# 前置：City378 评测已在 screen jeval 里跑（由外部启动）。本脚本只负责后续。
#
# **上一版脚本的教训**：它把解释器写成 `$P/train_env/bin/python`，而实际 venv 在
# `/root/train_env`，导致两次评测都在第 52 行直接报 "No such file or directory"，
# 被 `|| echo WARN` 吞掉，于是过闸在无数据的情况下默认 reject、
# 步骤 A 用错适配器起跑。两处要改：**解释器路径统一走 PY 变量**、
# **评测失败必须中止而不是继续**（不能靠 WARN 继续跑下去）。
set -u

P=/root/autodl-tmp/quant_rl_20261026
AC=$P/acc_code_20261029
J=$P/runs/jitter_20261030
DATA=/root/autodl-tmp/rematch_20260922/data
DEV=$DATA/quant_grounding_20261026_formal
CK=$J/correction/train/checkpoint-2400
BASE=$P/runs/acc_correction_20261029/correction/eval/presentations-2400/city378/attempt-1
PY=/root/train_env/bin/python
LOG=$P/runs/after_jitter_v2.log

exec > >(tee -a "$LOG") 2>&1
echo "================ $(date '+%F %T') 收尾开始 ================"

if [ ! -x "$PY" ]; then echo "FATAL: 解释器不存在 $PY"; exit 1; fi
if [ ! -d "$CK" ]; then echo "FATAL: 缺 $CK"; exit 1; fi
echo "解释器 $PY 与 checkpoint 均就绪"

# ---------- 1. 等 City378 评测结束 ----------
for i in $(seq 1 120); do
  screen -ls 2>/dev/null | grep -qE '^\s+[0-9]+\.jeval' || break
  sleep 30
done
if [ ! -f "$J/eval/city378/summary.json" ]; then
  echo "FATAL: City378 评测未产出 summary.json，中止（不再盲跑）"
  exit 1
fi
echo "City378 评测完成"

# ---------- 2. 评外部88 ----------
echo "--- 评测外部88 ---"
"$PY" -u "$AC/tools/eval_quant_grounding.py" \
  --dataset "$DEV/external94_dev.jsonl" --data-root "$DATA" \
  --model /root/autodl-tmp/rematch_models/Qwen3.6-27B --adapter "$CK" \
  --engine hf --precision nf4 --answer-mode short --reward-mode acc05 \
  --format-policy punctuation --samples-per-query 1 --hf-max-batch-size 2 --seed 2026 \
  --max-tokens 64 --max-length 8192 --min-pixels 200704 --max-pixels 1204224 \
  --output-dir "$J/eval/external88" > "$J/eval_ext.log" 2>&1
if [ ! -f "$J/eval/external88/summary.json" ]; then
  echo "FATAL: 外部88 评测失败，中止"
  exit 1
fi
echo "外部88 评测完成"

# ---------- 3. 配对过闸 ----------
echo "--- 配对比较（近失集由基线定义）---"
"$PY" "$AC/tools/compare_jitter_eval.py" \
  --baseline "$BASE/predictions.jsonl" \
  --candidate "$J/eval/city378/predictions.jsonl" \
  --out-json "$J/eval/jitter_vs_baseline.json" | tail -22
VERDICT=$("$PY" -c "
import json
try:
    d = json.load(open('$J/eval/jitter_vs_baseline.json'))
    print(d['verdict']); print('  尺寸比变化', d['delta']['near_miss_mean_size_ratio'],
          '总命中变化', d['delta']['hits'], file=__import__('sys').stderr)
except Exception as e:
    print('reject')")
echo "过闸判定: $VERDICT"

# ---------- 4. 按判定选适配器，启动步骤 A ----------
if [ "$VERDICT" = "adopt" ]; then
  ADAPTER=$CK; echo "采用抖动适配器"
else
  ADAPTER=$P/runs/acc_correction_20261029/correction/train/checkpoint-1600; echo "回退原纠错适配器"
fi

OUT=$P/runs/bf16_k20_$(date '+%Y%m%d%H%M')
mkdir -p "$OUT"
cd /root/AIC_code || exit 1
echo "--- 步骤 A：BF16 + 2× × 5691，输出 $OUT ---"
/root/vllm_env/bin/python tools/predict_submission_vllm.py \
  --model /root/autodl-tmp/rematch_models/Qwen3.6-27B \
  --adapter "$ADAPTER" \
  --modality trimodal \
  --queries /root/autodl-tmp/semifinal_20261028/queries/queries.json \
  --data-root /root/autodl-tmp/semifinal_20261028 \
  --expected-query-count 5691 \
  --min-pixels 200704 --max-pixels 8294400 \
  --model-max-length 32768 --max-num-seqs 12 \
  --max-lora-rank 32 --chunk-size 256 --upscale 2.0 \
  --gpu-memory-utilization 0.95 \
  --output-dir "$OUT"
echo "--- 步骤 A 结束 rc=$? 输出 $OUT ---"
echo "================ $(date '+%F %T') 收尾结束 ================"
