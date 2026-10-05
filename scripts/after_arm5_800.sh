#!/usr/bin/env bash
# Arm 5 续训到 800 步之后的全自动收尾。
#
# ## 为什么需要这个链条
#
# 截止 20:00，而续训约 16:55 结束。之后的每一步都必须尽快：
#   评价 ck800 @ k1.0（约 7 分钟）→ 与 ck700 的 318 比较 → 若更好则跑 5690（约 1.6 小时）
# 手工串这些步骤会浪费十几分钟，而十几分钟正是我们的余量。
#
# ## 判据为什么是"严格更好"
#
# ck700 @ k1.0 = 318，它的 5690 包**已经做好并通过格式闸门**。
# 所以要换模型，ck800 必须**严格超过 318**——否则重新跑 1.6 小时的 5690 毫无意义。
#
# ## 为什么用 k1.0 而不是放大
#
# 已证实：Arm 5 在 k1.0 下的"尺寸估计越界率"最低（12.2%），放大反而把它推到 16.3%。
# 队长据此决定不再放大。
set -uo pipefail

RUNS=/root/autodl-tmp/runs/q36_arm5_corrected
ARCH=${RUNS}/archived
OUT=/root/sprint_qwen36/arm5_ck800_k10
Q=/root/sprint_qwen36/city_val_queries.json
R=/root/autodl-tmp/rematch_20260922/data/city/train
OFF_Q=/root/autodl-tmp/rematch_20260922/data/official_test_rematch_20260924/queries/queries.json
OFF_R=/root/autodl-tmp/rematch_20260922/data/official_test_rematch_20260924
LOG=/root/arm5_800_chain.log
BASELINE=318   # ck700 @ k1.0 @ chunk64 的命中数

say() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }

cd /root/AIC_code || exit 2
export PATH=/root/vllm_env/bin:$PATH

# --- 1. 等训练结束 ---
say "等待续训结束（train5b 消失）"
while screen -ls 2>/dev/null | grep -q 'train5b'; do sleep 30; done
sleep 20
say "续训已结束"

# --- 2. 归档 ck800（save_total_limit=8 会保留，但硬链接一份更稳）---
if [ -d "${RUNS}"/v*/checkpoint-800 ]; then
    ck=$(ls -d "${RUNS}"/v*/checkpoint-800 | head -1)
    mkdir -p "${ARCH}/checkpoint-800"
    for f in adapter_model.safetensors adapter_config.json additional_config.json args.json; do
        [ -f "${ck}/${f}" ] && ln -f "${ck}/${f}" "${ARCH}/checkpoint-800/${f}" 2>/dev/null
    done
    say "已归档 checkpoint-800"
else
    say "找不到 checkpoint-800，退出"; exit 3
fi

# --- 3. 评 ck800 @ k1.0 City412（chunk=64，与 ck700 严格同口径）---
say "评价 ck800 @ k1.0 @ City412"
/root/vllm_env/bin/python -u tools/predict_submission_vllm.py \
    --model /root/rematch_models/Qwen3.6-27B \
    --adapter "${ARCH}/checkpoint-800" --modality trimodal \
    --queries "${Q}" --data-root "${R}" --expected-query-count 412 \
    --min-pixels 200704 --max-pixels 2073600 \
    --model-max-length 8192 --max-num-seqs 256 \
    --chunk-size 64 --upscale 1.0 --output-dir "${OUT}" > /root/arm5_ck800_k10.log 2>&1

/root/miniconda3/bin/python tools/verify_acc05_dev.py \
    --gt /root/sprint_qwen36/baseline_preds/city412_gt.json \
    --arm ck700=/root/sprint_qwen36/arm5_ck700_k10_chunk64/predictions.jsonl \
    --arm ck800="${OUT}/predictions.jsonl" \
    --out-json /root/sprint_qwen36/arm5_800_vs_700.json > /dev/null 2>&1

H800=$(/root/miniconda3/bin/python -c "
import json;d=json.load(open('/root/sprint_qwen36/arm5_800_vs_700.json'))['arms']
print(d.get('ck800',{}).get('hits_0.5',0))" 2>/dev/null || echo 0)
say "ck800 @ k1.0 = ${H800}/412   （ck700 = ${BASELINE}）"

# --- 4. 只有严格更好才重跑 5690 ---
if [ "${H800}" -gt "${BASELINE}" ]; then
    say "ck800 更好，开始出 5690 包（k1.0）"
    /root/vllm_env/bin/python -u tools/predict_submission_vllm.py \
        --model /root/rematch_models/Qwen3.6-27B \
        --adapter "${ARCH}/checkpoint-800" --modality trimodal \
        --queries "${OFF_Q}" --data-root "${OFF_R}" --expected-query-count 5690 \
        --min-pixels 200704 --max-pixels 2073600 \
        --model-max-length 8192 --max-num-seqs 256 \
        --chunk-size 128 --upscale 1.0 \
        --output-dir /root/sprint_qwen36/submit_arm5_ck800_k10 \
        > /root/submit_arm5_ck800_k10.log 2>&1

    /root/miniconda3/bin/python tools/check_submission_format.py \
        --queries "${OFF_Q}" \
        --submission /root/sprint_qwen36/submit_arm5_ck800_k10/predictions.json \
        --zip /root/sprint_qwen36/submit_arm5_ck800_k10/submission.zip 2>&1 | tail -8 | tee -a "${LOG}"
    say "包已出：/root/sprint_qwen36/submit_arm5_ck800_k10/submission.zip"
else
    say "ck800 未超过 ${BASELINE}，沿用 ck700 的包（submit_trained/submission.zip）"
fi
say "链条结束"
