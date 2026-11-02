#!/bin/bash
# 抖动重训结束后的自动串联：评测 → 配对过闸 → 启动 BF16 全量推理。
#
# 设计要点：
# 1) 过闸判据不看总命中数（City378 标准误约 8.5 题，整条无抖动轨迹只在 311~313 摆动），
#    而看机制指标：近失题平均尺寸比要上移 >= 0.03（实测基线中位 0.758），且总命中不降。
# 2) **无论过闸与否都启动步骤 A**——BF16 全量推理在"抖动适配器"和"原纠错适配器"上都值得跑，
#    所以不存在因为过闸失败而空转 GPU 的情况。
# 3) 步骤 A 用 /root/AIC_code 的复赛 BF16 工具，它是产出 0.8267 的同一份代码；
#    纠错适配器走的是 acc_code 的路径，两者互不影响。
set -u

P=/root/autodl-tmp/quant_rl_20261026
AC=$P/acc_code_20261029
TRAIN=$P/runs/jitter_20261030/correction/train
CK=$TRAIN/checkpoint-2400
EVAL=$P/runs/jitter_20261030/eval
BASE=$P/runs/acc_correction_20261029/correction/eval/presentations-2400
CITY=/root/autodl-tmp/rematch_20260922/data/quant_grounding_20261026_formal/city400_dev.jsonl
EXT=/root/autodl-tmp/rematch_20260922/data/quant_grounding_20261026_formal/external94_dev.jsonl
LOG=$P/runs/after_jitter.log

exec > >(tee -a "$LOG") 2>&1
echo "================ $(date '+%F %T') 开始串联 ================"

# ---------- 1. 等训练结束 ----------
for i in $(seq 1 240); do
  [ -f "$TRAIN/completed.json" ] && break
  sleep 30
done
if [ ! -d "$CK" ]; then
  echo "FATAL: 训练未产出 $CK，终止"
  exit 1
fi
echo "训练完成，checkpoint: $CK"

# completed.json 可能比进程真正退出早几秒。评测要加载模型，必须等显存彻底释放，
# 否则会与残留训练进程抢显存而 OOM。
for i in $(seq 1 60); do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [ "${USED:-99999}" -lt 2000 ]; then
    echo "显存已释放（${USED} MiB），开始评测"
    break
  fi
  echo "  等待显存释放，当前 ${USED} MiB"
  sleep 10
done

# ---------- 2. 评测（与历史同口径：HF + NF4 + 低分辨率） ----------
mkdir -p "$EVAL"
run_eval () {  # $1=dataset  $2=输出子目录
  "$P/train_env/bin/python" "$AC/tools/eval_quant_grounding.py" \
    --dataset "$1" --data-root /root/autodl-tmp/rematch_20260922/data \
    --model /root/autodl-tmp/rematch_models/Qwen3.6-27B --adapter "$CK" \
    --engine hf --precision nf4 --answer-mode short --reward-mode acc05 \
    --format-policy punctuation --samples-per-query 1 --hf-max-batch-size 2 --seed 2026 \
    --max-tokens 64 --max-length 8192 --min-pixels 200704 --max-pixels 1204224 \
    --output-dir "$2"
}
echo "--- 评测 City378 ---"
run_eval "$CITY" "$EVAL/city378" || echo "WARN: city378 评测失败"
echo "--- 评测外部88 ---"
run_eval "$EXT" "$EVAL/external88" || echo "WARN: external88 评测失败"

# ---------- 3. 配对过闸 ----------
VERDICT=reject
if [ -f "$EVAL/city378/predictions.jsonl" ]; then
  "$P/train_env/bin/python" "$AC/tools/compare_jitter_eval.py" \
    --baseline "$BASE/city378/attempt-1/predictions.jsonl" \
    --candidate "$EVAL/city378/predictions.jsonl" \
    --out-json "$EVAL/jitter_vs_baseline.json" || true
  VERDICT=$("$P/train_env/bin/python" -c "
import json,sys
try:
    print(json.load(open('$EVAL/jitter_vs_baseline.json'))['verdict'])
except Exception:
    print('reject')")
fi
echo "过闸判定: $VERDICT"

# ---------- 4. 步骤 A：BF16 全量推理（两种判定都跑） ----------
if [ "$VERDICT" = "adopt" ]; then
  ADAPTER=$CK
  echo "采用抖动适配器: $ADAPTER"
else
  ADAPTER=$P/runs/acc_correction_20261029/correction/train/checkpoint-1600
  echo "抖动未过闸，回退原纠错适配器: $ADAPTER"
fi

OUT=$P/runs/bf16_k20_$(date '+%Y%m%d%H%M')
mkdir -p "$OUT"
cd /root/AIC_code || exit 1
echo "--- 步骤 A 启动：BF16 + 2× × 5691 ---"
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
echo "================ $(date '+%F %T') 串联结束 ================"
