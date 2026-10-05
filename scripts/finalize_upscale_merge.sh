#!/usr/bin/env bash
# 串接完成"分路合并"提交包：等 k=1.25 官方出包 → 跑 k=1.0 官方 → 分路合并 → 格式闸门。
#
# ## 为什么要两源
#
# 分路规则是"逐题看**基线**预测框的面积"，所以必须同时有：
#   * 基线版（k=1.0，与已提交的 0.7870 同参数）
#   * 放大版（k=1.25）
# 只有一源就无法分路。City412 演练：基线 312 / 纯放大 316 / **分路 320**。
#
# ## 为什么串接而不是等人决定
#
# 两个源的官方推理各要 1.6~2.5 小时。串接执行能把这段时间与阶梯实验的尾巴重叠，
# 且**只产出制品、不提交**——决定权仍留给人。
set -uo pipefail

PY=/root/miniconda3/bin/python
VLLM_PY=/root/vllm_env/bin/python
CODE_DIR="${CODE_DIR:-/root/AIC_code}"
MODEL="${MODEL:-/root/rematch_models/Qwen3.6-27B}"
ADAPTER="${ADAPTER:-/root/autodl-tmp/runs/q36_arm2_capability/archived/checkpoint-500}"
OFFICIAL="${OFFICIAL:-/root/autodl-tmp/rematch_20260922/data/official_test_rematch_20260924}"
BASE_OUT="${BASE_OUT:-/root/sprint_qwen36/submit_upscale_base}"
UP_OUT="${UP_OUT:-/root/sprint_qwen36/submit_upscale_k1.25}"
MERGED_OUT="${MERGED_OUT:-/root/sprint_qwen36/submit_upscale_merged}"
THRESHOLD="${THRESHOLD:-0.006}"
LOG="${LOG:-/root/finalize_merge.log}"

exec > >(tee -a "${LOG}") 2>&1
say() { echo "[$(date '+%F %T')] $*"; }
cd "${CODE_DIR}" || exit 2
export PATH=/root/vllm_env/bin:$PATH

wait_for_gpu() {
  local waited=0 pids
  while [ "${waited}" -lt 240 ]; do
    pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)"
    [ "${pids}" -eq 0 ] && return 0
    [ "${waited}" -eq 0 ] && say "GPU 上有 ${pids} 个进程，等待释放"
    sleep 15
    waited=$((waited + 15))
  done
  say "等待超时，强制清理残留"
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    kill -9 "${pid}" 2>/dev/null || true
  done
  sleep 10
  pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)"
  [ "${pids}" -ne 0 ] && { say "!! GPU 仍被占用，停止"; return 1; }
  return 0
}

# ---- 1. 等 k=1.25 出包完成 -------------------------------------------------
say "等待 k=1.25 官方出包结束（screen 会话 submitsub 消失）"
waited=0
while screen -ls 2>/dev/null | grep -E '\.submitsub' | grep -qE '(Attached|Detached)'; do
  [ "${waited}" -ge 14400 ] && { say "等待超过 4 小时，k1.25 可能卡住，退出"; exit 2; }
  sleep 60
  waited=$((waited + 60))
done
if [ -f "${UP_OUT}/predictions.jsonl" ]; then
  say "k=1.25 已有 $(wc -l < "${UP_OUT}/predictions.jsonl") 条"
else
  say "!! k=1.25 没有产出结果——分路无法进行，退出"
  exit 3
fi

# ---- 2. 跑 k=1.0 官方（基线版，参数与已提交的 0.7870 一致）------------------
if [ -f "${BASE_OUT}/predictions.jsonl" ]; then
  say "k=1.0 已有结果，跳过"
else
  wait_for_gpu || exit 5
  say "跑 k=1.0 官方 5690（2073600 / 8192 / 256，与已提交包同参数）"
  rm -rf "${BASE_OUT}"
  "${VLLM_PY}" -u tools/predict_submission_vllm.py \
    --model "${MODEL}" --adapter "${ADAPTER}" --modality trimodal \
    --queries "${OFFICIAL}/queries/queries.json" --data-root "${OFFICIAL}" \
    --expected-query-count 5690 --min-pixels 200704 --max-pixels 2073600 \
    --model-max-length 8192 --max-num-seqs 256 --chunk-size 256 --upscale 1.0 \
    --output-dir "${BASE_OUT}" > "${BASE_OUT}.log" 2>&1
  status=$?
  say "k=1.0 推理退出码 ${status}"
  [ "${status}" -ne 0 ] && { tail -8 "${BASE_OUT}.log" | sed 's/^/  /'; exit "${status}"; }
fi

# ---- 3. 分路合并 -----------------------------------------------------------
wait_for_gpu || exit 5
say "分路合并（阈值 ${THRESHOLD}）"
rm -rf "${MERGED_OUT}"
"${PY}" tools/merge_by_predicted_size.py \
  --base "${BASE_OUT}/predictions.jsonl" --upscaled "${UP_OUT}/predictions.jsonl" \
  --queries "${OFFICIAL}/queries/queries.json" --expected-query-count 5690 \
  --area-threshold "${THRESHOLD}" --out-dir "${MERGED_OUT}" || exit 6

# ---- 4. 两源的解析率 + 合并包的格式闸门 ------------------------------------
say "两源解析率（现役标准是 100%、0 兜底、0 重试）"
for d in "${BASE_OUT}" "${UP_OUT}"; do
  [ -f "$d/summary.json" ] || continue
  "${PY}" -c "
import json, pathlib
d = json.loads(pathlib.Path('$d/summary.json').read_text(encoding='utf-8'))
print(f\"  $(basename $d): 完成 {d.get('completed_queries')} 解析失败 {d.get('parse_failures')} \"
      f\"兜底 {d.get('fallback_predictions')} 阶段2重试 {d.get('stage2_retries')}\")
"
done

say "格式闸门（必须 SUBMISSION_FORMAT_OK）"
"${PY}" tools/check_submission_format.py --queries "${OFFICIAL}/queries/queries.json" \
  --submission "${MERGED_OUT}/predictions.json" --zip "${MERGED_OUT}/submission.zip" 2>&1 | tail -10

echo
echo "================ 分路合并提交包 ================"
echo "zip    : ${MERGED_OUT}/submission.zip"
echo "组成   : 基线 ${BASE_OUT}  +  放大 ${UP_OUT}  阈值 ${THRESHOLD}"
echo "对照   : 现役最好 0.7870（本地 314）"
echo "         City412 演练——基线 312 / 纯放大 316 / 分路 320"
echo "说明   : 本包尚未提交，等队里决定"
echo "================================================"
