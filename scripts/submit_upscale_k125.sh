#!/usr/bin/env bash
# 等阶梯实验结束后，自动用 k=1.25 跑官方 5690 并过格式闸门。
#
# ## 为什么要排队而不是等结果再决定
#
# k1.25 是本地最佳（City412 = 316），且分桶分析显示**极小目标桶随倍数呈剂量-反应改善**
# （75 → 78 → 82 → 82），机制方向明确。而跑一次官方 5690 要 2~3 小时——
# 排队执行可以把这段时间与阶梯实验的尾巴重叠，不必等人回来再启动。
#
# **它只产出制品，不提交。** 提交与否由人决定。
#
# ## 参数必须与 City412 阶梯完全一致
#
# 放大的效果依赖 max_pixels / max_length / max_num_seqs 三者联动；
# 官方推理若用不同参数，City412 上的结论就不能迁移过去。
set -uo pipefail

PY=/root/miniconda3/bin/python
VLLM_PY=/root/vllm_env/bin/python
CODE_DIR="${CODE_DIR:-/root/AIC_code}"
MODEL="${MODEL:-/root/rematch_models/Qwen3.6-27B}"
ADAPTER="${ADAPTER:-/root/autodl-tmp/runs/q36_arm2_capability/archived/checkpoint-500}"
OFFICIAL="${OFFICIAL:-/root/autodl-tmp/rematch_20260922/data/official_test_rematch_20260924}"
# k=1.25 对应的一组参数（与阶梯表一致）
UPSCALE=1.25
MAX_PIXELS=3240000
MAX_LENGTH=12288
MAX_SEQS=128
OUT="${OUT:-/root/sprint_qwen36/submit_upscale_k1.25}"
LOG="${LOG:-/root/submit_upscale.log}"

exec > >(tee -a "${LOG}") 2>&1
say() { echo "[$(date '+%F %T')] $*"; }
cd "${CODE_DIR}" || exit 2
export PATH=/root/vllm_env/bin:$PATH

say "等待阶梯实验结束（screen 会话 ladder 消失）"
waited=0
while screen -ls 2>/dev/null | grep -q '\.ladder'; do
  [ "${waited}" -ge 7200 ] && { say "等待超过 2 小时，阶梯可能卡住，退出"; exit 2; }
  sleep 60
  waited=$((waited + 60))
done
say "阶梯已结束，等 60 秒让 GPU 释放"
sleep 60

# 阶梯每一档都起停过 vLLM；正式提交推理绝不能带着残留显存开跑。
waited=0
while [ "${waited}" -lt 180 ]; do
  pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)"
  [ "${pids}" -eq 0 ] && break
  [ "${waited}" -eq 0 ] && say "GPU 上有 ${pids} 个进程，等待释放"
  sleep 15
  waited=$((waited + 15))
done
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  say "强制清理残留进程 ${pid}"
  kill -9 "${pid}" 2>/dev/null || true
done
sleep 10
pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)"
[ "${pids}" -ne 0 ] && { say "!! GPU 仍被占用，停止"; exit 5; }

say "用 k=${UPSCALE} 跑官方 5690（max_pixels=${MAX_PIXELS} max_length=${MAX_LENGTH} max_num_seqs=${MAX_SEQS}）"
rm -rf "${OUT}"
"${VLLM_PY}" -u tools/predict_submission_vllm.py \
  --model "${MODEL}" --adapter "${ADAPTER}" --modality trimodal \
  --queries "${OFFICIAL}/queries/queries.json" --data-root "${OFFICIAL}" \
  --expected-query-count 5690 --min-pixels 200704 --max-pixels "${MAX_PIXELS}" \
  --model-max-length "${MAX_LENGTH}" --max-num-seqs "${MAX_SEQS}" \
  --chunk-size 256 --upscale "${UPSCALE}" \
  --output-dir "${OUT}" > "${OUT}.log" 2>&1
status=$?
say "推理退出码 ${status}"
[ "${status}" -ne 0 ] && { say "推理失败，日志尾部："; tail -8 "${OUT}.log" | sed 's/^/  /'; exit "${status}"; }

say "解析率检查（现役是 100%、0 兜底、0 重试；若退步则不应提交）"
"${PY}" -c "
import json, pathlib
d = json.loads(pathlib.Path('${OUT}/summary.json').read_text(encoding='utf-8'))
total = d.get('completed_queries'); pf = d.get('parse_failures'); fb = d.get('fallback_predictions')
s2 = d.get('stage2_retries')
print(f'  完成 {total}  解析失败 {pf}  兜底 {fb}  阶段2重试 {s2}')
print('  解析率 100%' if pf == 0 and fb == 0 else f'  ** 解析率非满分：{pf} 条失败 / {fb} 条兜底 **')
"

say "格式闸门（必须 SUBMISSION_FORMAT_OK）"
"${PY}" tools/check_submission_format.py --queries "${OFFICIAL}/queries/queries.json" \
  --submission "${OUT}/predictions.json" --zip "${OUT}/submission.zip" 2>&1 | tail -10

echo
echo "================ 候选提交包（k=${UPSCALE}）================"
echo "zip   : ${OUT}/submission.zip"
echo "对照  : 现役最好 0.7870（submit_ck500，本地 314）"
echo "       本包本地 City412 316（极小桶 82 vs 基线 75）"
echo "=========================================================="
