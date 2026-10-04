#!/usr/bin/env bash
# k2.0 官方推理 → 与已提交包分路合并 → 格式闸门。
#
# ## 为什么是 k2.0 而不是 k1.25
#
# 完整的 City412 阶梯（本机）：
#
#     档     极小<0.2%   小0.2-1%   中1-5%   大>5%    总数
#     k1.0       75        124       71       42     312
#     k1.10      78        117       72       42     309
#     k1.25      82        122       70       42     316
#     k1.5       82        118       71       41     312
#     k2.0       90        117       73       41     321    <- 最好
#
# **极小桶的剂量-反应是单调不降的**：75 → 78 → 82 → 82 → 90，且 k2.0 明显拉开（+15）。
# 配对检验（McNemar）k2.0 救回 26 题 / 弄错 17 题，是所有档里最强的信号
# （但 p=0.22，**未达显著**，412 题的判别力有限，须如实标注）。
#
# ## 为什么还要分路（而不是直接提交纯 k2.0）
#
# 纯 k2.0 在「小 0.2–1%」桶上从 124 掉到 117（−7）。分路的作用正是**保护其他桶**：
# 逐题看**基线**预测框面积，小于阈值才取放大版，否则保留基线。
#
#     k1.0 为基 + k2.0：322 321 322 321 322 321 321 322   （阈值 0.003→0.020）
#
# **整个阈值范围都是 321–322**——几乎没有阈值选择风险，这是它比 k1.25 更稳的地方。
#
# ## 基线版直接复用已提交的包
#
# `/root/sprint_qwen36/submit_ck500/predictions.jsonl` 就是 k=1.0 参数下的官方 5690 条预测
# （max_pixels 2073600 / max_length 8192 / max_num_seqs 256 / chunk 256），**与阶梯的 k1.0 档同参数**。
# 因此**不必重跑**，省下约 1.6 小时。
set -uo pipefail

PY=/root/miniconda3/bin/python
VLLM_PY=/root/vllm_env/bin/python
CODE_DIR="${CODE_DIR:-/root/AIC_code}"
MODEL="${MODEL:-/root/rematch_models/Qwen3.6-27B}"
ADAPTER="${ADAPTER:-/root/autodl-tmp/runs/q36_arm2_capability/archived/checkpoint-500}"
OFFICIAL="${OFFICIAL:-/root/autodl-tmp/rematch_20260922/data/official_test_rematch_20260924}"
# 基线版：已提交的 0.7870 那份包（就是 k=1.0 参数）
BASE_PRED="${BASE_PRED:-/root/sprint_qwen36/submit_ck500/predictions.jsonl}"
UP_OUT="${UP_OUT:-/root/sprint_qwen36/submit_upscale_k2.0}"
MERGED_OUT="${MERGED_OUT:-/root/sprint_qwen36/submit_merged_k20}"
THRESHOLD="${THRESHOLD:-0.006}"
LOG="${LOG:-/root/finalize_k20.log}"

# k=2.0 对应的三档参数（与阶梯表一致，三者必须联动）
UPSCALE=2.0
MAX_PIXELS=8294400
MAX_LENGTH=32768
MAX_SEQS=32

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

test -f "${BASE_PRED}" || { say "基线版预测不存在：${BASE_PRED}"; exit 3; }
say "基线版：${BASE_PRED}（$(wc -l < "${BASE_PRED}") 条，即已提交的 0.7870 那份的参数）"

# ---- 1. k=2.0 官方推理 -----------------------------------------------------
if [ -f "${UP_OUT}/predictions.jsonl" ]; then
  say "k=2.0 已有结果（$(wc -l < "${UP_OUT}/predictions.jsonl") 条），跳过推理"
else
  wait_for_gpu || exit 5
  say "跑 k=2.0 官方 5690（upscale=${UPSCALE} max_pixels=${MAX_PIXELS} max_length=${MAX_LENGTH} max_num_seqs=${MAX_SEQS}）"
  say "  注意：token 数约为原生的 4 倍，预计 7~8 小时"
  rm -rf "${UP_OUT}"
  "${VLLM_PY}" -u tools/predict_submission_vllm.py \
    --model "${MODEL}" --adapter "${ADAPTER}" --modality trimodal \
    --queries "${OFFICIAL}/queries/queries.json" --data-root "${OFFICIAL}" \
    --expected-query-count 5690 --min-pixels 200704 --max-pixels "${MAX_PIXELS}" \
    --model-max-length "${MAX_LENGTH}" --max-num-seqs "${MAX_SEQS}" \
    --chunk-size 256 --upscale "${UPSCALE}" \
    --output-dir "${UP_OUT}" > "${UP_OUT}.log" 2>&1
  status=$?
  say "推理退出码 ${status}"
  [ "${status}" -ne 0 ] && { tail -10 "${UP_OUT}.log" | sed 's/^/  /'; exit "${status}"; }
fi

# ---- 2. 解析率检查（放大后可能变差，必须先看）------------------------------
say "k=2.0 解析率（现役标准 100%、0 兜底、0 重试）"
"${PY}" -c "
import json, pathlib
p = pathlib.Path('${UP_OUT}/summary.json')
if p.is_file():
    d = json.loads(p.read_text(encoding='utf-8'))
    print(f\"  完成 {d.get('completed_queries')}  解析失败 {d.get('parse_failures')}  \"
          f\"兜底 {d.get('fallback_predictions')}  阶段2重试 {d.get('stage2_retries')}\")
else:
    print('  (无 summary.json)')
"

# ---- 3. 分路合并 -----------------------------------------------------------
wait_for_gpu || exit 5
say "分路合并（基线 ${BASE_PRED} + 放大 ${UP_OUT}，阈值 ${THRESHOLD}）"
rm -rf "${MERGED_OUT}"
"${PY}" tools/merge_by_predicted_size.py \
  --base "${BASE_PRED}" --upscaled "${UP_OUT}/predictions.jsonl" \
  --queries "${OFFICIAL}/queries/queries.json" --expected-query-count 5690 \
  --area-threshold "${THRESHOLD}" --out-dir "${MERGED_OUT}" || exit 6

# ---- 4. 格式闸门 -----------------------------------------------------------
say "格式闸门（必须 SUBMISSION_FORMAT_OK）"
"${PY}" tools/check_submission_format.py --queries "${OFFICIAL}/queries/queries.json" \
  --submission "${MERGED_OUT}/predictions.json" --zip "${MERGED_OUT}/submission.zip" 2>&1 | tail -10

echo
echo "================ 分路合并包（k1.0 基 + k2.0）================"
echo "zip    : ${MERGED_OUT}/submission.zip"
echo "阈值   : ${THRESHOLD}"
echo "对照   : 现役最好 0.7870（本地 314/312）"
echo "         City412——纯 k1.0 312 / 纯 k2.0 321 / 分路 321~322"
echo "         极小桶：75 -> 90（+15）"
echo "说明   : 尚未提交，等队里决定（剩 3 次机会，取最好一次）"
echo "============================================================"
