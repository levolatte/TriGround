#!/usr/bin/env bash
# 训练后的最终提交：先判决，通过才跑 5690 条出正式 zip。
#
# 判据（事先登记，不事后调整）：
#   A. 训练后适配器在 City412 上的命中数 > 293（本机基线）
#   B. 与基线 A 的 78 图组聚类配对 bootstrap 95% 区间下界 > 0
#   C. **适配器确实被应用了**：同一个 vLLM 加不加适配器结果必须有明显差异。
#      只看 ACC 无法区分"适配器没用上"和"适配器没用"——前者是静默故障，必须单独抓。
#
# 为什么不用"HF vs vLLM 逐题一致率"来判断 LoRA 覆盖：这两个引擎之间有约 1% 的固有数值
# 差异（实测一致率 99.0%），拿它当阈值会**误报**。用同引擎加/不加适配器对比才干净。
set -uo pipefail

PY=/root/miniconda3/bin/python
VLLM_PY=/root/vllm_env/bin/python
CODE_DIR="${CODE_DIR:-/root/AIC_code}"
ADAPTER="${ADAPTER:?请设置 ADAPTER}"
MODEL="${MODEL:-/root/rematch_models/Qwen3.6-27B}"
OFFICIAL="${OFFICIAL:-/root/autodl-tmp/rematch_20260922/data/official_test_rematch_20260924}"
CITY_ROOT="${CITY_ROOT:-/root/autodl-tmp/rematch_20260922/data/city/train}"
CITY_QUERIES="${CITY_QUERIES:-/root/sprint_qwen36/city_val_queries.json}"
GT="${GT:-/root/sprint_qwen36/baseline_preds/city412_gt.json}"
DEV_OUT="${DEV_OUT:-/root/sprint_qwen36/b5_verdict}"
FINAL_OUT="${FINAL_OUT:-/root/sprint_qwen36/submit_trained}"
BASELINE_HITS="${BASELINE_HITS:-293}"
NO_ADAPTER_HITS="${NO_ADAPTER_HITS:-296}"

say() { echo "[$(date +%H:%M:%S)] $*"; }
cd "${CODE_DIR}" || exit 2
export PATH=/root/vllm_env/bin:$PATH

# 每次起 vLLM 前确认 GPU 干净。逐检查点扫描会反复启停引擎，任何一次留下孤儿
# EngineCore 都会让后续推理 OOM——而 vLLM 的失败信息往往只说"显存不足"，
# 不指出是上一次没退干净。这里先等、再清、最后硬失败，避免把问题带到提交阶段。
wait_for_gpu() {
  local waited=0
  while [ "${waited}" -lt 180 ]; do
    local pids
    pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)"
    if [ "${pids}" -eq 0 ]; then
      return 0
    fi
    if [ "${waited}" -eq 0 ]; then
      say "GPU 上有 ${pids} 个进程，等待其退出（可能是上一次 vLLM 的残留）"
      nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | sed 's/^/    /'
    fi
    sleep 15
    waited=$((waited + 15))
  done
  # 等待无效再清一次；仍不清就硬失败，绝不带着不干净的显存去跑正式推理
  say "等待 180 秒仍未释放，强制清理残留进程"
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    kill -9 "${pid}" 2>/dev/null || true
  done
  sleep 10
  local left
  left="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)"
  if [ "${left}" -ne 0 ]; then
    say "!! GPU 仍被 ${left} 个进程占用，停止以免 OOM"
    return 1
  fi
  return 0
}

[ -f "${ADAPTER}/adapter_config.json" ] || { echo "适配器不存在: ${ADAPTER}" >&2; exit 3; }
[ -f "${CITY_QUERIES}" ] || { echo "City412 官方格式清单不存在: ${CITY_QUERIES}" >&2; exit 3; }

wait_for_gpu || exit 4

say "1/4 用训练后适配器跑 City412"
rm -rf "${DEV_OUT}_trained"
"${VLLM_PY}" -u tools/predict_submission_vllm.py --model "${MODEL}" --adapter "${ADAPTER}" \
  --modality trimodal --queries "${CITY_QUERIES}" --data-root "${CITY_ROOT}" \
  --expected-query-count 412 --max-pixels 2073600 --model-max-length 8192 \
  --max-num-seqs 256 --chunk-size 128 \
  --output-dir "${DEV_OUT}_trained" > "${DEV_OUT}_trained.log" 2>&1
trained_status=$?
tail -4 "${DEV_OUT}_trained.log"

say "2/4 判决：ACC + 聚类配对区间 + 适配器是否真的生效"
"${PY}" tools/verify_acc05_dev.py --gt "${GT}" \
  --arm trained="${DEV_OUT}_trained/predictions.jsonl" \
  --arm baseline=/root/sprint_qwen36/probes/A_8b/pixels_2073600/predictions.jsonl \
  --taxonomy trained --compare trained:baseline --bootstrap 5000 \
  --out-json "${DEV_OUT}_verdict.json" > /dev/null 2>&1

result=$("${PY}" - <<PYEOF
import json
d = json.load(open("${DEV_OUT}_verdict.json"))
hits = d["arms"]["trained"]["hits_0.5"]
pair = d["paired_bootstrap"]
lo = pair["ci95_pp"][0]
base, no_adapter = ${BASELINE_HITS}, ${NO_ADAPTER_HITS}
print(f"训练后命中      : {hits}  (基线 {base}，未加适配器 {no_adapter})")
print(f"聚类配对 95% 区间: [{lo:.3f}, {pair['ci95_pp'][1]:.3f}] pp  跨0={pair['crosses_zero']}")
print(f"分桶            : {d['taxonomy']['buckets']}")
# C：适配器是否真的被应用。若加不加适配器的命中数几乎相同，说明 vLLM 静默忽略了它。
applied = abs(hits - no_adapter) >= 3
print(f"适配器生效判据  : 与未加适配器相差 {hits - no_adapter:+d} 题 -> "
      f"{'生效' if applied else '未见效（疑似未应用！）'}")
ok = hits > base and lo > 0 and applied
print("VERDICT_" + ("PASS" if ok else "FAIL"))
PYEOF
)
echo "${result}"
verdict_line=$(echo "${result}" | tail -1)

if [ "${verdict_line}" != "VERDICT_PASS" ]; then
  say "B5 不达标或不满足生效判据 —— 不提交训练后模型"
  echo "回退：使用已核验的 B2 零样本包 /root/sprint_qwen36/submit_b2_zeroshot/submission.zip"
  exit 1
fi

say "3/4 跑官方 5690 条"
# 上一步的 vLLM 若没退干净，这里会 OOM；正式提交推理绝不能带着残留显存开跑
wait_for_gpu || exit 4
rm -rf "${FINAL_OUT}"
"${VLLM_PY}" -u tools/predict_submission_vllm.py --model "${MODEL}" --adapter "${ADAPTER}" \
  --modality trimodal --queries "${OFFICIAL}/queries/queries.json" --data-root "${OFFICIAL}" \
  --expected-query-count 5690 --max-pixels 2073600 --model-max-length 8192 \
  --max-num-seqs 256 --chunk-size 256 \
  --output-dir "${FINAL_OUT}" > "${FINAL_OUT}.log" 2>&1
say "推理退出码 $?"

say "4/4 格式闸门（必须 SUBMISSION_FORMAT_OK）"
"${PY}" tools/check_submission_format.py --queries "${OFFICIAL}/queries/queries.json" \
  --submission "${FINAL_OUT}/predictions.json" --zip "${FINAL_OUT}/submission.zip" 2>&1 | tail -10

echo
echo "================ 最终提交 ================"
echo "zip   : ${FINAL_OUT}/submission.zip"
echo "截止  : 2026-10-05 20:00"
echo "=========================================="
