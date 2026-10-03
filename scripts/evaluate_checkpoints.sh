#!/usr/bin/env bash
# 逐检查点评估：在 City412 上给训练过程中**每一个**检查点打分，选出最好的那个。
#
# ## 为什么需要它
#
# 本轮训练此前只保留最后两个检查点，而且没有任何验证环节——等于"跑完才知道好不好，
# 且只能看最后一步"。小数据 + 大模型恰恰是最可能"最佳步数不在最后"的情形。
# `archive_checkpoints.sh` 已经把各检查点保住了，这个脚本负责**用同一把尺子量它们**。
#
# ## 判据
#
# 与预登记口径保持一致：ACC@0.5 命中数。这里**只做排序**，是否值得提交仍按
# `final_submission_with_adapter.sh` 的三条件判据（命中 > 293、聚类配对区间下界 > 0、
# 适配器确实生效）判断——避免"在 412 条上挑最高分"变成对开发集的过拟合。
set -uo pipefail

PY=/root/miniconda3/bin/python
VLLM_PY=/root/vllm_env/bin/python
CODE_DIR="${CODE_DIR:-/root/AIC_code}"
MODEL="${MODEL:-/root/rematch_models/Qwen3.6-27B}"
ARCHIVE_DIR="${ARCHIVE_DIR:-/root/runs/q36_600_2073600/archived}"
CITY_ROOT="${CITY_ROOT:-/root/autodl-tmp/rematch_20260922/data/city/train}"
CITY_QUERIES="${CITY_QUERIES:-/root/sprint_qwen36/city_val_queries.json}"
GT="${GT:-/root/sprint_qwen36/baseline_preds/city412_gt.json}"
OUT_ROOT="${OUT_ROOT:-/root/sprint_qwen36/ckpt_sweep}"

say() { echo "[$(date +%H:%M:%S)] $*"; }
cd "${CODE_DIR}" || exit 2
export PATH=/root/vllm_env/bin:$PATH

if [ ! -f "${CITY_QUERIES}" ]; then
  say "City412 清单不存在: ${CITY_QUERIES}"; exit 3
fi

mapfile -t CHECKPOINTS < <(ls -d "${ARCHIVE_DIR}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n)
if [ "${#CHECKPOINTS[@]}" -eq 0 ]; then
  say "没有已归档的检查点：${ARCHIVE_DIR}"; exit 3
fi
say "待评估检查点 ${#CHECKPOINTS[@]} 个：$(basename -a "${CHECKPOINTS[@]}" | tr '\n' ' ')"

mkdir -p "${OUT_ROOT}"
for checkpoint in "${CHECKPOINTS[@]}"; do
  name="$(basename "${checkpoint}")"
  out="${OUT_ROOT}/${name}"
  if [ -f "${out}/predictions.jsonl" ]; then
    say "${name} 已有结果，跳过"
    continue
  fi
  say "评估 ${name}"
  rm -rf "${out}"
  "${VLLM_PY}" -u tools/predict_submission_vllm.py --model "${MODEL}" --adapter "${checkpoint}" \
    --modality trimodal --queries "${CITY_QUERIES}" --data-root "${CITY_ROOT}" \
    --expected-query-count 412 --max-pixels 2073600 --model-max-length 8192 \
    --max-num-seqs 256 --chunk-size 128 \
    --output-dir "${out}" > "${out}.log" 2>&1
  status=$?
  if [ "${status}" -ne 0 ]; then
    say "${name} 推理失败（退出码 ${status}），日志：${out}.log"
    continue
  fi
  tail -1 "${out}.log"
done

say "汇总各检查点得分"

# 用仓库里**已验证过**的判决工具，而不是自己重写 IoU：它已经处理了
# city412_gt.json 的嵌套结构（id -> {bbox, query, ...}）以及 bbox/prediction
# 两种字段名。自己写一遍只会重复踩这两个坑。
ARMS=()
for checkpoint in "${CHECKPOINTS[@]}"; do
  name="$(basename "${checkpoint}")"
  path="${OUT_ROOT}/${name}/predictions.jsonl"
  [ -f "${path}" ] || continue
  ARMS+=(--arm "${name}=${path}")
done
if [ "${#ARMS[@]}" -eq 0 ]; then
  say "没有任何可用结果"; exit 1
fi

"${PY}" tools/verify_acc05_dev.py --gt "${GT}" "${ARMS[@]}" \
  --out-json "${OUT_ROOT}/sweep.json" > "${OUT_ROOT}/sweep.txt" 2>&1
status=$?
if [ "${status}" -ne 0 ]; then
  say "判决工具失败（退出码 ${status}），见 ${OUT_ROOT}/sweep.txt"; exit "${status}"
fi

"${PY}" - <<'PYEOF'
import json
import pathlib

sweep = json.loads(pathlib.Path("/root/sprint_qwen36/ckpt_sweep/sweep.json")
                   .read_text(encoding="utf-8"))
arms = {name: data["hits_0.5"] for name, data in sweep["arms"].items()}
rows = sorted(arms.items(), key=lambda item: -item[1])
print(f"\n{'检查点':>16s} {'命中':>6s} {'ACC':>8s}")
for name, hits in rows:
    print(f"{name:>16s} {hits:>6d} {hits / 412:>8.4f}")

best_name, best_hits = rows[0]
last_name = max(arms, key=lambda n: int(n.split("-")[1]))
print(f"\n最佳: {best_name}  命中 {best_hits}/412")
print(f"目录: /root/sprint_qwen36/ckpt_sweep/{best_name}")
print("提交时用最佳检查点，但判据仍按预登记口径（命中>293 且聚类配对区间下界>0）。")
if last_name != best_name:
    print(f"⚠️ 最佳**不是**最后一步（最后是 {last_name}，命中 {arms[last_name]}）——"
          "只看最后一步会漏掉更好的模型，这正是补可观测性的意义。")
PYEOF

