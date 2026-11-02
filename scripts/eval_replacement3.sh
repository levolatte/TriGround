#!/bin/bash
# 新数据模型的开发集评测：checkpoint-2400 与 checkpoint-1600 各跑 City378 + 外部88。
#
# 为什么两个 checkpoint 都评：历史上「1600 比 2400 好」这个说法只差 2 题，
# 落在 378 题标准误（约 8.5）以内，并不可靠。这次同时评两个，看这个说法能否复现。
#
# 参数严格照抄历史（presentations-2400 基线），特别是 --hf-max-batch-size 2 与
# --samples-per-query 1 —— 复赛实测过批大小/分块能造成 2~5 题差异，改了就破坏可比性。
set -u

P=/root/autodl-tmp/quant_rl_20261026
AC=$P/acc_code_20261029
PY=/root/train_env/bin/python
DATA=/root/autodl-tmp/rematch_20260922/data
DEV=$DATA/quant_grounding_20261026_formal
TRAIN=$P/runs/replacement3_20261030/correction/train
EVAL=$P/runs/replacement3_20261030/eval
LOG=$P/runs/replacement3_eval.log

exec > >(tee -a "$LOG") 2>&1
echo "================ $(date '+%F %T') 开始评测 ================"

if [ ! -x "$PY" ]; then echo "FATAL: 缺 $PY"; exit 1; fi

run_eval () {  # $1=checkpoint 号  $2=数据集文件  $3=输出子目录
    local ck="$1" dataset="$2" outdir="$3"
    echo "--- checkpoint-$ck / $(basename "$dataset") ---"
    "$PY" -u "$AC/tools/eval_quant_grounding.py" \
        --dataset "$dataset" --data-root "$DATA" \
        --model /root/autodl-tmp/rematch_models/Qwen3.6-27B \
        --adapter "$TRAIN/checkpoint-$ck" \
        --engine hf --precision nf4 --answer-mode short --reward-mode acc05 \
        --format-policy punctuation --samples-per-query 1 --hf-max-batch-size 2 --seed 2026 \
        --max-tokens 64 --max-length 8192 --min-pixels 200704 --max-pixels 1204224 \
        --output-dir "$outdir"
    if [ ! -f "$outdir/summary.json" ]; then
        echo "FATAL: checkpoint-$ck 在 $(basename "$dataset") 上评测失败"
        exit 1
    fi
    "$PY" -c "
import json
d = json.load(open('$outdir/summary.json'))
o = d['overall']
print('  结果 acc05', round(o['acc05_first'], 4), ' 命中', round(o['acc05_first'] * o['queries']), '/', o['queries'])
"
}

for ck in 2400 1600; do
    if [ ! -d "$TRAIN/checkpoint-$ck" ]; then
        echo "FATAL: 缺 checkpoint-$ck"; exit 1
    fi
    run_eval "$ck" "$DEV/city400_dev.jsonl" "$EVAL/ck$ck/city378"
    run_eval "$ck" "$DEV/external94_dev.jsonl" "$EVAL/ck$ck/external88"
done

echo "================ $(date '+%F %T') 评测结束 ================"
echo "基线（correction1600, presentations-2400）: city378 311/378, external88 84/88"
