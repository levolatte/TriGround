#!/usr/bin/env bash
# Arm 5 ck700 的 City412 全档位扫描——**统一 chunk=64**，得到可比较的一组数。
#
# ## 为什么必须重跑 k1.0
#
# 我先前那次 k1.0 = 316 是用 `--chunk-size 128` 测的（逐检查点扫描脚本的默认值），
# 而 k1.25 / k1.5 / k2.0 都是 64。**已知 chunk 会影响几题（实测 312 vs 309），
# 而我们正在测的放大效应本身也只有几题**——两者同量级，混在一起就分不清是哪个造成的。
#
# 所以这一轮把 k1.0 用 chunk=64 重测，**与其余档位严格同口径**。
#
# ## 档位参数（max_pixels 必须随 k² 联动，否则处理器把图缩回原尺寸、实验静默失效）
#
#   k      max_pixels   max_length   max_num_seqs
#   1.00   2073600       8192        256
#   1.25   3240000      12288        128
#   1.50   4665600      16384         64
#   1.75   6350400      24576         48
#   2.00   8294400      32768         32
set -uo pipefail

CK=/root/autodl-tmp/runs/q36_arm5_corrected/archived/checkpoint-700
Q=/root/sprint_qwen36/city_val_queries.json
R=/root/autodl-tmp/rematch_20260922/data/city/train
LOG=/root/arm5_ladder64.log

say() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }

run_one() {
    local k="$1" maxpx="$2" maxlen="$3" nseq="$4" tag="$5"
    local out="/root/sprint_qwen36/arm5_ck700_${tag}"
    say "=== k=${k} chunk=64 -> ${out} ==="
    cd /root/AIC_code || return 2
    /root/vllm_env/bin/python -u tools/predict_submission_vllm.py \
        --model /root/rematch_models/Qwen3.6-27B \
        --adapter "${CK}" \
        --modality trimodal \
        --queries "${Q}" --data-root "${R}" \
        --expected-query-count 412 \
        --min-pixels 200704 --max-pixels "${maxpx}" \
        --model-max-length "${maxlen}" --max-num-seqs "${nseq}" \
        --chunk-size 64 --upscale "${k}" \
        --output-dir "${out}" > "/root/arm5_${tag}_c64.log" 2>&1
    say "=== k=${k} 完成 ==="
}

# 只补缺失的档位：k1.0 用 chunk=64 重测；其余若已完成则跳过
if [ ! -f /root/sprint_qwen36/arm5_ck700_k10_chunk64/predictions.jsonl ] || \
   [ "$(wc -l < /root/sprint_qwen36/arm5_ck700_k10_chunk64/predictions.jsonl 2>/dev/null || echo 0)" -lt 412 ]; then
    run_one 1.0 2073600 8192 256 k10_chunk64
fi

say "全部完成"
