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

# 本脚本会**连续启停 vLLM 六次**（每个检查点一次）。只要有一次留下孤儿 EngineCore
# 占着显存，后续检查点就会 OOM，而 vLLM 只会说"显存不足"，不指出是上一次没退干净。
# 因此在每一轮开始前都确认 GPU 干净：先等、再清、仍不干净就硬失败——
# 宁可少评一个检查点，也不要让整轮扫描的结果不可信。
wait_for_gpu() {
  local waited=0 pids
  while [ "${waited}" -lt 180 ]; do
    pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)"
    [ "${pids}" -eq 0 ] && return 0
    if [ "${waited}" -eq 0 ]; then
      say "GPU 上有 ${pids} 个进程，等待其退出（可能是上一轮 vLLM 的残留）"
      nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | sed 's/^/    /'
    fi
    sleep 15
    waited=$((waited + 15))
  done
  say "等待 180 秒仍未释放，强制清理残留进程"
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    kill -9 "${pid}" 2>/dev/null || true
  done
  sleep 10
  pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)"
  if [ "${pids}" -ne 0 ]; then
    say "!! GPU 仍被 ${pids} 个进程占用，停止以免产出不可信的扫描结果"
    return 1
  fi
  return 0
}

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
  wait_for_gpu || exit 5
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

# 汇总表：路径必须用 ${OUT_ROOT}，**不能硬编码**。
# 这里曾经写死 `/root/sprint_qwen36/ckpt_sweep/sweep.json`（那是默认值目录），
# 于是换成 Arm 3 的 `ckpt_sweep_arm3` 后，`sweep.json` 明明写对了（见上一行），
# 这段汇总却去读一个不存在的文件而抛 FileNotFoundError——日志里就只剩 traceback，
# 看不到各检查点分数。没有 `set -e` 所以不至于中断，但等于把最该看的输出弄丢了。
"${PY}" - <<PYEOF
import json
import pathlib

sweep = json.loads(pathlib.Path("${OUT_ROOT}/sweep.json").read_text(encoding="utf-8"))
arms = {name: data["hits_0.5"] for name, data in sweep["arms"].items()}
rows = sorted(arms.items(), key=lambda item: -item[1])
print(f"\n{'检查点':>16s} {'命中':>6s} {'ACC':>8s}")
for name, hits in rows:
    print(f"{name:>16s} {hits:>6d} {hits / 412:>8.4f}")

best_name, best_hits = rows[0]
last_name = max(arms, key=lambda n: int(n.split("-")[1]))
print(f"\n最佳: {best_name}  命中 {best_hits}/412")
print(f"目录: ${OUT_ROOT}/{best_name}")
print("提交时用扫描最佳检查点（注意：按本地分数挑 = 在开发集上做选择，本地数字偏乐观）。")
if last_name != best_name:
    print(f"⚠️ 最佳**不是**最后一步（最后是 {last_name}，命中 {arms[last_name]}）——"
          "只看最后一步会漏掉更好的模型，这正是补可观测性的意义。")
PYEOF

