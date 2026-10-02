#!/usr/bin/env bash
# B5 判决：评测训练后的 LoRA 适配器，并按预登记判据给出结论。
#
# 为什么写成脚本：训练结束的那一刻是决策点，现场拼命令容易出错、也容易忘记做
# "vLLM 是否真的把 LoRA 应用到全部目标模块"这一步验证——而那个错误是静默的。
#
# 判据（事先登记，不得事后调整）：
#   1) City412 命中数 > 293（本机 602112 档基线）
#   2) 与基线 A 的 78 图组聚类配对 bootstrap 95% 区间下界 > 0
#   3) vLLM 与 HF 两条路径用**同一个适配器**的 ACC 差 <= 2 题（否则说明 vLLM 的 LoRA
#      模块覆盖不全，必须改用 merge_and_unload 合并后再跑）
#
# 用法：
#   ADAPTER=/root/runs/q36_600_2073600/v0-xxx/checkpoint-600 bash scripts/evaluate_trained_adapter.sh
set -uo pipefail

PY=/root/miniconda3/bin/python
VLLM_PY=/root/vllm_env/bin/python
CODE_DIR="${CODE_DIR:-/root/AIC_code}"
ADAPTER="${ADAPTER:?请设置 ADAPTER 指向训练输出的 checkpoint 目录}"
MODEL="${MODEL:-/root/rematch_models/Qwen3.6-27B}"
CITY_ROOT="${CITY_ROOT:-/root/autodl-tmp/rematch_20260922/data/city/train}"
QUERIES="${QUERIES:-/root/sprint_qwen36/city_val_queries.json}"
GT="${GT:-/root/sprint_qwen36/baseline_preds/city412_gt.json}"
OUT_ROOT="${OUT_ROOT:-/root/sprint_qwen36/b5}"
BASELINE_HITS="${BASELINE_HITS:-293}"

say() { echo "[$(date +%H:%M:%S)] $*"; }
cd "${CODE_DIR}" || exit 2

if [ ! -f "${ADAPTER}/adapter_config.json" ]; then
  echo "适配器不存在: ${ADAPTER}/adapter_config.json" >&2
  exit 3
fi
if [ ! -f "${QUERIES}" ]; then
  say "City412 官方格式清单不存在，先从开发清单转换"
  "${PY}" scripts/convert_dev_manifest_to_queries.py \
    --manifest /root/autodl-tmp/rematch_20260922/results/triground_abv_20260927/inputs/city_val.json \
    --out "${QUERIES}"
fi

say "1/4 vLLM + LoRA 跑 City412（验证 vLLM 的 LoRA 模块覆盖）"
rm -rf "${OUT_ROOT}_vllm"
export PATH=/root/vllm_env/bin:$PATH
"${VLLM_PY}" tools/predict_submission_vllm.py --model "${MODEL}" --adapter "${ADAPTER}" \
  --modality trimodal --queries "${QUERIES}" --data-root "${CITY_ROOT}" \
  --expected-query-count 412 --max-pixels 2073600 --model-max-length 8192 \
  --max-num-seqs 256 --chunk-size 128 \
  --output-dir "${OUT_ROOT}_vllm" 2>&1 | tail -20
vllm_status=${PIPESTATUS[0]}

say "2/4 HF 路径用同一适配器跑 City412（作为 LoRA 覆盖的参照）"
rm -rf "${OUT_ROOT}_hf"
"${PY}" tools/predict_native_submission.py --model "${MODEL}" --adapter "${ADAPTER}" \
  --modality trimodal --queries "${QUERIES}" --data-root "${CITY_ROOT}" \
  --expected-query-count 412 --max-pixels 2073600 --model-max-length 8192 \
  --no-thinking --no-stop-on-bbox \
  --output-dir "${OUT_ROOT}_hf" 2>&1 | tail -8
hf_status=${PIPESTATUS[0]}

say "3/4 两条路径逐题比对（LoRA 覆盖验证）"
"${PY}" scripts/compare_predictions.py \
  "${OUT_ROOT}_hf/predictions.jsonl" "${OUT_ROOT}_vllm/predictions.jsonl" \
  --label-a HF --label-b vLLM 2>&1 | tail -12
parity_status=${PIPESTATUS[0]}

say "4/4 ACC 判决（对比本机基线 ${BASELINE_HITS}）"
"${PY}" tools/verify_acc05_dev.py --gt "${GT}" \
  --arm trained="${OUT_ROOT}_vllm/predictions.jsonl" \
  --arm baseline=/root/sprint_qwen36/probes/A_8b/pixels_2073600/predictions.jsonl \
  --taxonomy trained --compare trained:baseline --bootstrap 5000 \
  --out-json "${OUT_ROOT}_verdict.json" 2>&1 | tail -20

say "结论汇总"
"${PY}" - <<PYEOF
import json, sys
try:
    d = json.load(open("${OUT_ROOT}_verdict.json"))
except Exception as error:
    print("无法读取判决文件:", error); sys.exit(1)
base = ${BASELINE_HITS}
hits = d["arms"]["trained"]["hits_0.5"]
pair = d["paired_bootstrap"]
lo = pair["ci95_pp"][0]
print(f"训练后命中 {hits}  vs 基线 {base}  (差 {hits - base:+d})")
print(f"聚类配对 95% 区间 [{lo:.3f}, {pair['ci95_pp'][1]:.3f}] pp  跨0={pair['crosses_zero']}")
print(f"分桶: {d['taxonomy']['buckets']}")
ok = hits > base and lo > 0
print()
print("B5_VERDICT: " + ("PASS —— 用训练后模型跑 5690 条出最终提交" if ok
                        else "FAIL —— 回退用 B2 零样本的 submission.zip"))
if ${parity_status} != 0:
    print("注意：vLLM 与 HF 路径不一致，请确认 vLLM 的 LoRA 模块覆盖；"
          "必要时改用 merge_and_unload 合并后再跑。")
PYEOF
echo "vllm=${vllm_status} hf=${hf_status} parity=${parity_status}"
