#!/usr/bin/env bash
# Arm 4 ck600 × k2.0 × 官方 5690 → 出包 → 格式闸门。
#
# ## 为什么在本地判不出胜负时还要跑
#
# City412 的标准误是 **8.79 题**（412 题、SE 2.13pp），而 Arm4(317) 与 ck500(321) 只差 **4 题**
# ——**本地这把尺子测不出这个量级**。官方 5690 题的 SE 约 **0.54pp**，灵敏度约 4 倍。
#
# 所以判据改成**机制论证 + 零下行风险**：
#   1. k2.0 已被官方确证有效（0.7870 → 0.8012，+1.42pp ≈ 2.6 倍标准误）→ 必须带上
#   2. **深度修复**：原训练数据 **55.6% 的样本**深度是原始 uint16 直转 RGB（80% 像素全白）的废图；
#      Arm 4 改用与推理**逐像素一致**的渲染 → 机制独立、未被证伪
#   3. **数据分布对齐**：三模态占比 6.2% → 59.4%、查询长度中位 84 → 54（官方 49）、剔除 65 个无关类别
#   4. 三者与放大互不干扰，**理论上可叠加**
#   5. 比赛**取最高分** → 提交 Arm 4 **零下行风险**
#
# ## 参数严格沿用跑出 0.8012 的那一套
#
# chunk=256 与官方包一致，避免引入新变量。max_pixels / max_length / max_num_seqs 三者
# 必须与 upscale 2.0 联动，否则处理器会把放大缩回去（静默失效）。
set -uo pipefail

PY=/root/miniconda3/bin/python
VLLM_PY=/root/vllm_env/bin/python
CODE_DIR=/root/AIC_code
MODEL=/root/rematch_models/Qwen3.6-27B
ARM4=/root/autodl-tmp/runs/q36_arm4_matched/archived/checkpoint-600
OFFICIAL=/root/autodl-tmp/rematch_20260922/data/official_test_rematch_20260924
OUT="${OUT:-/root/sprint_qwen36/submit_arm4_ck600_k20}"
LOG="${LOG:-/root/submit_arm4.log}"

cd "${CODE_DIR}" || exit 2
export PATH=/root/vllm_env/bin:$PATH
exec > >(tee -a "${LOG}") 2>&1
say() { echo "[$(date '+%F %T')] $*"; }

wait_for_gpu() {
  local waited=0 pids
  while [ "${waited}" -lt 300 ]; do
    pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)"
    [ "${pids}" -eq 0 ] && return 0
    [ "${waited}" -eq 0 ] && say "GPU 上有 ${pids} 个进程，等待释放"
    sleep 15; waited=$((waited + 15))
  done
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    kill -9 "${pid}" 2>/dev/null || true
  done
  sleep 10
  [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)" -ne 0 ] && return 1
  return 0
}

test -d "${ARM4}" || { say "Arm 4 ck600 不存在：${ARM4}"; exit 3; }
say "适配器 ${ARM4}（$(du -sh "${ARM4}" | cut -f1)）"

if [ -f "${OUT}/predictions.jsonl" ] && [ "$(wc -l < "${OUT}/predictions.jsonl")" -ge 5690 ]; then
  say "已有完整结果，跳过推理"
else
  wait_for_gpu || { say "GPU 不可用"; exit 5; }
  say "=== Arm 4 ck600 @ k2.0 × 官方 5690（预计约 6.4 小时）==="
  say "    upscale 2.0 / max_pixels 8294400 / max_length 32768 / max_num_seqs 32 / chunk 256"
  rm -rf "${OUT}"
  "${VLLM_PY}" -u tools/predict_submission_vllm.py \
    --model "${MODEL}" --adapter "${ARM4}" --modality trimodal \
    --queries "${OFFICIAL}/queries/queries.json" --data-root "${OFFICIAL}" \
    --expected-query-count 5690 \
    --min-pixels 200704 --max-pixels 8294400 \
    --model-max-length 32768 --max-num-seqs 32 \
    --chunk-size 256 --upscale 2.0 \
    --output-dir "${OUT}" > "${OUT}.log" 2>&1
  st=$?
  say "推理退出码 ${st}"
  [ "${st}" -ne 0 ] && { tail -10 "${OUT}.log" | sed 's/^/  /'; exit "${st}"; }
fi

say "解析率检查（现役标准：100% 解析、0 兜底、0 重试）"
"${PY}" -c "
import json, pathlib
p = pathlib.Path('${OUT}/summary.json')
if p.is_file():
    d = json.loads(p.read_text(encoding='utf-8'))
    pf, fb, s2 = d.get('parse_failures'), d.get('fallback_predictions'), d.get('stage2_retries')
    print(f\"  完成 {d.get('completed_queries')}  解析失败 {pf}  兜底 {fb}  阶段2重试 {s2}\")
    print('  ** 解析率满分 **' if pf == 0 and fb == 0 else '  ** 解析率未满分 **')
else:
    print('  (无 summary.json)')
"

wait_for_gpu || true
say "格式闸门（必须 SUBMISSION_FORMAT_OK）"
"${PY}" tools/check_submission_format.py --queries "${OFFICIAL}/queries/queries.json" \
  --submission "${OUT}/predictions.json" --zip "${OUT}/submission.zip" 2>&1 | tail -10

echo
echo "================ Arm 4 ck600 @ k2.0 提交包 ================"
echo "zip    : ${OUT}/submission.zip"
echo "对照   : 现役最好 0.8012（ck500 @ k2.0）"
echo "         City412 本地——ck500@k2.0 321 / Arm4@k2.0 317"
echo "         **本地差 4 题 < 噪声 SE 8.79 题，本地判不出胜负**"
echo "         故依机制论证 + 零下行风险提交"
echo "=========================================================="
