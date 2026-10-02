#!/usr/bin/env bash
# 阶段 B 的探针套件：对「冻结基线 A」与「新基座零样本」各跑三档像素，并输出判决表。
#
# 判决只看两件事（总分被噪声支配，不作判据）：
#   1) 错误实例数 other_instance（基线 A 在 602112 下是 30）
#   2) 小目标子集与 near_miss 的变化
#
# 用法：
#   bash scripts/run_probe_suite.sh b1          # 第一跳：冻结 A（8B）× 三档
#   bash scripts/run_probe_suite.sh b2          # 第二跳：27B 零样本 × 三档
set -euo pipefail

STAGE="${1:?用法: run_probe_suite.sh b1|b2}"
CODE=/root/AIC_code
PY=/root/miniconda3/bin/python
ROOT=/root/autodl-tmp/rematch_20260922
OUT_ROOT=/root/sprint_qwen36/probes
VAL="${ROOT}/results/triground_abv_20260927/inputs/city_val.json"
GT="${ROOT}/results/triground_abv_20260927/inputs/city_gt.json"
DATA_ROOT="${ROOT}/data/city/train"
PIXELS="602112 1204224 2073600"
ADAPTER="${ROOT}/results/triground_abv_execution_20260928/runs/seed2026_600_deterministic/A/main"

cd "${CODE}"
mkdir -p "${OUT_ROOT}"

run_probe() {  # $1=标签 $2=模型 $3=额外参数
  local tag="$1" model="$2" extra="$3"
  local out="${OUT_ROOT}/${tag}"
  echo "=== 探针 ${tag}  模型=${model} ==="
  # shellcheck disable=SC2086
  "${PY}" tools/probe_native_resolution.py \
      --model "${model}" ${extra} \
      --manifest "${VAL}" --target-manifest "${GT}" --data-root "${DATA_ROOT}" \
      --output-dir "${out}" --pixels ${PIXELS} --model-max-length 16384 --no-thinking \
      2>&1 | tail -20
}

verdict() {  # $1=标签：对每个像素档出判决
  local tag="$1"
  for px in ${PIXELS}; do
    local pred="${OUT_ROOT}/${tag}/pixels_${px}/predictions.jsonl"
    [ -f "${pred}" ] || continue
    "${PY}" tools/verify_acc05_dev.py --gt "${GT}" \
        --arm "${tag}@${px}=${pred}" --taxonomy "${tag}@${px}" \
        --out-json "${OUT_ROOT}/${tag}/verdict_${px}.json" > /dev/null
    "${PY}" - "${OUT_ROOT}/${tag}/verdict_${px}.json" "${tag}@${px}" <<'EOF'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
arm = data["arms"][sys.argv[2]]
buckets = data["taxonomy"]["buckets"]
print(f"{sys.argv[2]:>28s}  hits={arm['hits_0.5']:3d}/412  acc={arm['acc_0.5']:.4f}  "
      f"mIoU={arm['mean_iou']:.4f}  parse_fail={arm['parse_failures']:3d}  "
      f"other_instance={buckets.get('other_instance', 0):3d}  "
      f"near_miss={buckets.get('near_miss', 0):3d}  gross={buckets.get('gross_miss', 0):3d}")
EOF
  done
}

case "${STAGE}" in
  b1)
    run_probe "A_8b" "/root/rematch_models/Qwen3-VL-8B-Instruct" "--adapter ${ADAPTER}"
    verdict "A_8b"
    ;;
  b2)
    run_probe "q36_27b_zeroshot" "/root/rematch_models/Qwen3.6-27B" ""
    verdict "q36_27b_zeroshot"
    ;;
  *)
    echo "只支持 b1 / b2" >&2; exit 2;;
esac

echo "=== 参考：基线 A @602112 为 hits=296 acc=0.7184 other_instance=30 ==="
echo "PROBE_SUITE_DONE"
