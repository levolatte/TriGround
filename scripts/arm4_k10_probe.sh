#!/usr/bin/env bash
# 等 Arm 4 的 5690 出包结束后，补跑 **Arm 4 ck600 @ k1.0 × City412**。
#
# ## 为什么必须补这一格
#
# 我们此前把"Arm 4 @ k1.0 = 313"（**训练机的机器**）与"Arm 4 @ k2.0 = 317"（**本机**）
# 相减，得出"k2.0 对 Arm 4 只增益 +4，小于 ck500 的 +9"，并据此推断视觉 LoRA 削弱了尺度泛化。
#
# **但同一配置在不同机器/不同运行之间的方差实测是 ±3~5 题**
# （ck500 @ k1.0 @ chunk128：本机 309、训练机 314；本机 chunk64 又是 312）。
#
# 所以那个 +4 是**跨机器差值**，落在噪声里——**结论并未成立**。
# 补上本机的 Arm 4 @ k1.0 后，才是同机同代码的四格配对：
#
#              k1.0        k2.0
#   ck500      312         321
#   Arm4 ck600  ?          317
#
# ## 为什么放最后跑
#
# 5690 那轮已占用 ~92 GiB 显存，两者不能并行。k1.0 只需约 7 分钟，等它结束再跑即可。
set -uo pipefail

PY=/root/miniconda3/bin/python
VLLM_PY=/root/vllm_env/bin/python
cd /root/AIC_code || exit 2
export PATH=/root/vllm_env/bin:$PATH
LOG=/root/arm4_k10_probe.log
exec > >(tee -a "${LOG}") 2>&1
say() { echo "[$(date '+%F %T')] $*"; }

OUT=/root/sprint_qwen36/arm4_ck600_k10
MODEL=/root/rematch_models/Qwen3.6-27B
ARM4=/root/autodl-tmp/runs/q36_arm4_matched/archived/checkpoint-600
CITY_Q=/root/sprint_qwen36/city_val_queries.json
CITY_ROOT=/root/autodl-tmp/rematch_20260922/data/city/train

say "等待 Arm 4 的 5690 出包链结束（screen 会话 arm4sub 消失）"
waited=0
while screen -ls 2>/dev/null | grep -q '\.arm4sub'; do
  [ "${waited}" -ge 28800 ] && { say "等待超过 8 小时，退出"; exit 2; }
  sleep 60
  waited=$((waited + 60))
done
say "arm4sub 已结束"

# 释放检查：5690 那轮留下过 ~92 GiB 显存，直接开跑会 OOM
waited=0
while [ "${waited}" -lt 240 ]; do
  pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)"
  [ "${pids}" -eq 0 ] && break
  [ "${waited}" -eq 0 ] && say "GPU 上仍有 ${pids} 个进程，等待释放"
  sleep 15; waited=$((waited + 15))
done
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  say "清理残留 ${pid}"; kill -9 "${pid}" 2>/dev/null || true
done
sleep 10

if [ -f "${OUT}/predictions.jsonl" ] && [ "$(wc -l < "${OUT}/predictions.jsonl")" -ge 412 ]; then
  say "已有结果，跳过推理"
else
  say "=== 补跑：Arm 4 ck600 @ k1.0 × City412（与 ck500 的 312 同机配对）==="
  rm -rf "${OUT}"
  "${VLLM_PY}" -u tools/predict_submission_vllm.py \
    --model "${MODEL}" --adapter "${ARM4}" --modality trimodal \
    --queries "${CITY_Q}" --data-root "${CITY_ROOT}" --expected-query-count 412 \
    --min-pixels 200704 --max-pixels 2073600 \
    --model-max-length 8192 --max-num-seqs 256 \
    --chunk-size 64 --upscale 1.0 \
    --output-dir "${OUT}" > "${OUT}.log" 2>&1
  st=$?
  say "退出码 ${st}"
  [ "${st}" -ne 0 ] && { tail -8 "${OUT}.log" | sed 's/^/  /'; exit "${st}"; }
fi

say "=== 四格同机配对评分 ==="
"${PY}" tools/verify_acc05_dev.py \
  --gt /root/sprint_qwen36/baseline_preds/city412_gt.json \
  --arm ck500_k10=/root/sprint_qwen36/upscale_ladder/k1.0/predictions.jsonl \
  --arm ck500_k20=/root/sprint_qwen36/upscale_ladder/k2.0/predictions.jsonl \
  --arm arm4_k10=${OUT}/predictions.jsonl \
  --arm arm4_k20=/root/sprint_qwen36/arm4_ck600_k20/predictions.jsonl \
  --out-json /root/sprint_qwen36/arm4_k10_k20_grid.json 2>&1 | tail -14

say "完成"
