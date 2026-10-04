#!/usr/bin/env bash
# Arm 4 探针：ck600 @ k2.0 × City412，外加回答训练机提出的 chunk-size 口径疑问。
#
# ## 第一项：Arm 4 ck600 @ k2.0
#
# 训练机在 k1.0 上测过全轨迹，Arm 4 最佳是 ck600 = 313（与 ck500 的 314 持平）。
# 但 **k2.0 会把视觉 token 密度提到 4 倍**，而 Arm 4 训练过视觉塔（249.46M 可训练参数
# vs ck500 的 233.46M）——**视觉 LoRA 是在原生分辨率下标定的，能否泛化到 4× 密度没有证据**。
# 这是方向明确的怀疑，所以必须实测。
#
# **对照基线必须是 k2.0 下的 ck500 = 321**（已由官方 0.8012 确证）。
#
# ## 第二项：chunk-size 口径
#
# 同一个 ck500、同样 k1.0：训练机测得 314（chunk 128），我们阶梯是 312（chunk 64）。
# vLLM 的 chunk 只影响每批提交多少请求，贪心解码下理论上结果应一致，但批组成可能带来数值差异。
# 跑一次 chunk=128 的 ck500 @ k1.0 即可判定这 2 题是"chunk 效应"还是"运行噪声"。
#
# ## 为什么探针用 chunk=64
#
# 我们的 321 基线是 chunk=64 测出来的。**探针要与基线严格配对**，所以用同一 chunk。
set -uo pipefail

PY=/root/miniconda3/bin/python
VLLM_PY=/root/vllm_env/bin/python
cd /root/AIC_code || exit 2
export PATH=/root/vllm_env/bin:$PATH
LOG=/root/arm4_probe.log
exec > >(tee -a "${LOG}") 2>&1
say() { echo "[$(date '+%F %T')] $*"; }

MODEL=/root/rematch_models/Qwen3.6-27B
CITY_Q=/root/sprint_qwen36/city_val_queries.json
CITY_ROOT=/root/autodl-tmp/rematch_20260922/data/city/train
ARM4=/root/autodl-tmp/runs/q36_arm4_matched/archived/checkpoint-600
CK500=/root/autodl-tmp/runs/q36_arm2_capability/archived/checkpoint-500

wait_for_gpu() {
  local waited=0 pids
  while [ "${waited}" -lt 240 ]; do
    pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)"
    [ "${pids}" -eq 0 ] && return 0
    [ "${waited}" -eq 0 ] && say "GPU 上有 ${pids} 个进程，等待"
    sleep 15; waited=$((waited + 15))
  done
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    kill -9 "${pid}" 2>/dev/null || true
  done
  sleep 10
  [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)" -ne 0 ] && return 1
  return 0
}

# ---- 1. Arm 4 ck600 @ k2.0 ------------------------------------------------
OUT4=/root/sprint_qwen36/arm4_ck600_k20
if [ -f "${OUT4}/predictions.jsonl" ] && [ "$(wc -l < "${OUT4}/predictions.jsonl")" -ge 412 ]; then
  say "Arm 4 ck600 @ k2.0 已有完整结果，跳过"
else
  wait_for_gpu || { say "GPU 不可用"; exit 5; }
  say "=== 探针 1：Arm 4 ck600 @ k2.0 × City412 ==="
  rm -rf "${OUT4}"
  "${VLLM_PY}" -u tools/predict_submission_vllm.py \
    --model "${MODEL}" --adapter "${ARM4}" --modality trimodal \
    --queries "${CITY_Q}" --data-root "${CITY_ROOT}" --expected-query-count 412 \
    --min-pixels 200704 --max-pixels 8294400 \
    --model-max-length 32768 --max-num-seqs 32 \
    --chunk-size 64 --upscale 2.0 \
    --output-dir "${OUT4}" > "${OUT4}.log" 2>&1
  st=$?
  say "退出码 ${st}"
  [ "${st}" -ne 0 ] && { tail -8 "${OUT4}.log" | sed 's/^/  /'; exit "${st}"; }
fi

# ---- 2. chunk-size 口径：ck500 @ k1.0，chunk=128 ---------------------------
OUT1=/root/sprint_qwen36/ck500_k10_chunk128
if [ -f "${OUT1}/predictions.jsonl" ] && [ "$(wc -l < "${OUT1}/predictions.jsonl")" -ge 412 ]; then
  say "ck500 @ k1.0 (chunk 128) 已有结果，跳过"
else
  wait_for_gpu || { say "GPU 不可用"; exit 5; }
  say "=== 探针 2：ck500 @ k1.0，chunk=128（对照我们 chunk=64 的 312）==="
  rm -rf "${OUT1}"
  "${VLLM_PY}" -u tools/predict_submission_vllm.py \
    --model "${MODEL}" --adapter "${CK500}" --modality trimodal \
    --queries "${CITY_Q}" --data-root "${CITY_ROOT}" --expected-query-count 412 \
    --min-pixels 200704 --max-pixels 2073600 \
    --model-max-length 8192 --max-num-seqs 256 \
    --chunk-size 128 --upscale 1.0 \
    --output-dir "${OUT1}" > "${OUT1}.log" 2>&1
  st=$?
  say "退出码 ${st}"
  [ "${st}" -ne 0 ] && { tail -8 "${OUT1}.log" | sed 's/^/  /'; exit "${st}"; }
fi

# ---- 3. 汇总评分 ----------------------------------------------------------
wait_for_gpu || true
say "=== 评分 ==="
"${PY}" tools/verify_acc05_dev.py \
  --gt /root/sprint_qwen36/baseline_preds/city412_gt.json \
  --arm ck500_k10_chunk64=/root/sprint_qwen36/upscale_ladder/k1.0/predictions.jsonl \
  --arm ck500_k10_chunk128=${OUT1}/predictions.jsonl \
  --arm ck500_k20_chunk64=/root/sprint_qwen36/upscale_ladder/k2.0/predictions.jsonl \
  --arm arm4_ck600_k20=${OUT4}/predictions.jsonl \
  --out-json /root/sprint_qwen36/arm4_probe_scores.json 2>&1 | tail -18

say "完成"
