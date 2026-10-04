#!/usr/bin/env bash
# 超原生放大推理的阶梯实验：在 City412 上逐档放大，与基线 314 对比。
#
# ## 为什么做这个
#
# 失分 **65% 集中在极小目标桶**（面积 <0.2%，1920×1080 下约 65×65 像素）。
# 实测像素档位 602112 / 1204224 / 2073600 → 293 / 298 / 299，**每次提高都在涨**，
# 但 2073600 是"原生不放大"的上限。再往上从未测过。
#
# 放大让极小目标占据更多视觉 token（每 token 覆盖 32×32 像素），理论上有利于定位。
#
# ## 每档的三个参数必须联动，否则实验静默失效
#
# 处理器在图像**超出 `max_pixels` 时会下采样**。只放大而不提高 `max_pixels`，
# 图会被缩回原尺寸——模型收到的还是原图，而日志、耗时、显存全都正常。
# `predict_submission_vllm.py` 里的 `check_upscale_consistency` 会挡住这种误用，
# 这里把对应关系写死在同一张表里，避免手工配错。
#
#   upscale  max_pixels  max_length  max_num_seqs
#   1.00     2073600      8192        256     （基线）
#   1.10     2509056      8192        256
#   1.25     3240000     12288        128
#   1.50     4665600     16384         64
#   2.00     8294400     32768         32
#
# max_length 必须随之提高：视觉 token = 像素/32²，1.5× 时三图就要 14,070 token。
# max_num_seqs 随之下调：全注意力层每 token 的 KV 是 64 KB，长序列会挤爆 KV 池。
#
# ## 用法
#
#   ADAPTER=/root/.../checkpoint-500 bash scripts/upscale_ladder.sh
#   LADDER="1.0 1.5" ADAPTER=... bash scripts/upscale_ladder.sh   # 只跑指定档
set -uo pipefail

PY=/root/miniconda3/bin/python
# vLLM 装在独立环境里，**不能**用 miniconda3 跑推理——那里没有 vllm 模块。
# （首次运行就是踩了这个：ModuleNotFoundError: No module named 'vllm'。）
VLLM_PY=/root/vllm_env/bin/python
CODE_DIR="${CODE_DIR:-/root/AIC_code}"
MODEL="${MODEL:-/root/rematch_models/Qwen3.6-27B}"
ADAPTER="${ADAPTER:?必须指定 ADAPTER（Arm 2 的最佳检查点）}"
CITY_QUERIES="${CITY_QUERIES:-/root/sprint_qwen36/city_val_queries.json}"
CITY_ROOT="${CITY_ROOT:-/root/autodl-tmp/rematch_20260922/data/city/train}"
GT="${GT:-/root/sprint_qwen36/baseline_preds/city412_gt.json}"
OUT_ROOT="${OUT_ROOT:-/root/sprint_qwen36/upscale_ladder}"
LOG="${LOG:-/root/upscale_ladder.log}"
BASELINE_HITS="${BASELINE_HITS:-314}"
# 判定"有效"的门槛：基线 +4 题，超出该开发集 ±3 的噪声量级
PROMOTE_HITS="${PROMOTE_HITS:-318}"

exec > >(tee -a "${LOG}") 2>&1
say() { echo "[$(date '+%F %T')] $*"; }
cd "${CODE_DIR}" || exit 2
export PATH=/root/vllm_env/bin:$PATH

# 每一档：upscale:max_pixels:max_length:max_num_seqs
#
# 2.5 与 3.0 是**官方 0.8012 之后追加的**：k2.0 在极小桶上给到 90（基线 75），
# 且这一段是唯一还在明显上涨的区间（75→78→82→82→90），所以必须试 2.0 以上。
# 官方实测把本地 +9 兑现成了 +0.0142（约 +81 题），转移率约 65%，说明这个小实验有真实预测力。
#
# 显存账（全注意力层每 token 64 KB KV：16 层 × 4 KV头 × 256 head_dim × 2 × 2B）：
#   k2.0  24,480 token → 约 1.6 GB/序列，max_num_seqs 32
#   k2.5  38,250 token → 约 2.4 GB/序列，max_num_seqs 16
#   k3.0  55,080 token → 约 3.5 GB/序列，max_num_seqs 8
ALL_LADDER=("1.0:2073600:8192:256" "1.10:2509056:8192:256"
            "1.25:3240000:12288:128" "1.5:4665600:16384:64"
            "2.0:8294400:32768:32" "2.5:12960000:40960:16"
            "3.0:18662400:65536:8")

# 允许用 LADDER 覆盖（空格分隔的倍数），只跑关心的档
LADDER=("${ALL_LADDER[@]}")
if [ -n "${LADDER_FACTORS:-}" ]; then
  LADDER=()
  for want in ${LADDER_FACTORS}; do
    for row in "${ALL_LADDER[@]}"; do
      [ "${row%%:*}" = "${want}" ] && LADDER+=("${row}")
    done
  done
  [ "${#LADDER[@]}" -eq 0 ] && { say "LADDER_FACTORS=${LADDER_FACTORS} 没有匹配的档"; exit 3; }
fi

say "实验阶梯：${#LADDER[@]} 档   适配器=$(basename "${ADAPTER}")"
say "基线 City412 = ${BASELINE_HITS}；判定有效需 > ${PROMOTE_HITS}"
mkdir -p "${OUT_ROOT}"

# vLLM 每档都要重新起停一次。只要有一次留下孤儿 EngineCore，后面就会 OOM，
# 而 vLLM 只会说"显存不足"、不指出是上一档没退干净，最终得到一个残缺的阶梯。
wait_for_gpu() {
  local waited=0 pids
  while [ "${waited}" -lt 180 ]; do
    pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)"
    [ "${pids}" -eq 0 ] && return 0
    [ "${waited}" -eq 0 ] && say "GPU 上有 ${pids} 个进程，等待退出（可能是上一档残留）"
    sleep 15
    waited=$((waited + 15))
  done
  say "等待 180 秒仍未释放，强制清理"
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    kill -9 "${pid}" 2>/dev/null || true
  done
  sleep 10
  pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)"
  [ "${pids}" -ne 0 ] && { say "!! GPU 仍被占用，停止以免产出不可信的阶梯结果"; return 1; }
  return 0
}

declare -a ARMS=()
for row in "${LADDER[@]}"; do
  IFS=':' read -r factor max_pixels max_length max_seqs <<< "${row}"
  name="k${factor}"
  out="${OUT_ROOT}/${name}"
  say "=== 档 ${name}：upscale=${factor} max_pixels=${max_pixels} max_length=${max_length} max_num_seqs=${max_seqs} ==="

  if [ -f "${out}/predictions.jsonl" ]; then
    say "${name} 已有结果，跳过推理"
  else
    wait_for_gpu || exit 5
    rm -rf "${out}"
    "${VLLM_PY}" -u tools/predict_submission_vllm.py --model "${MODEL}" --adapter "${ADAPTER}" \
      --modality trimodal --queries "${CITY_QUERIES}" --data-root "${CITY_ROOT}" \
      --expected-query-count 412 --min-pixels 200704 --max-pixels "${max_pixels}" \
      --model-max-length "${max_length}" --max-num-seqs "${max_seqs}" \
      --chunk-size 64 --upscale "${factor}" \
      --output-dir "${out}" > "${out}.log" 2>&1
    status=$?
    if [ "${status}" -ne 0 ]; then
      say "${name} 推理失败（退出码 ${status}）—— 可能是该倍数下 OOM 或超长被拒。"
      say "  日志尾部："; tail -5 "${out}.log" | sed 's/^/    /'
      say "  跳过该档，继续下一档"
      continue
    fi
    tail -2 "${out}.log" | sed 's/^/  /'
  fi
  [ -f "${out}/predictions.jsonl" ] && ARMS+=(--arm "${name}=${out}/predictions.jsonl")
done

if [ "${#ARMS[@]}" -eq 0 ]; then
  say "没有任何一档产出结果"; exit 1
fi

say "汇总各档得分"
"${PY}" tools/verify_acc05_dev.py --gt "${GT}" "${ARMS[@]}" \
  --out-json "${OUT_ROOT}/ladder.json" > "${OUT_ROOT}/ladder.txt" 2>&1
status=$?
[ "${status}" -ne 0 ] && { say "判决工具失败，见 ${OUT_ROOT}/ladder.txt"; exit "${status}"; }

"${PY}" - <<PYEOF
import json, pathlib
data = json.loads(pathlib.Path("${OUT_ROOT}/ladder.json").read_text(encoding="utf-8"))
arms = {k: v["hits_0.5"] for k, v in data["arms"].items()}
baseline, promote = ${BASELINE_HITS}, ${PROMOTE_HITS}

def key(name):
    return float(name.lstrip("k"))

rows = sorted(arms.items(), key=lambda kv: key(kv[0]))
print(f"\n{'放大倍数':>10s} {'命中':>6s} {'ACC':>8s} {'vs 基线':>9s}")
for name, hits in rows:
    print(f"{name:>10s} {hits:>6d} {hits/412:>8.4f} {hits-baseline:>+9d}")

best_name, best_hits = max(arms.items(), key=lambda kv: kv[1])
print(f"\n最佳: {best_name}  命中 {best_hits}/412（基线 {baseline}）")

mono = all(rows[i][1] <= rows[i+1][1] for i in range(len(rows)-1))
if best_hits > promote:
    print(f"结论: 有效（{best_hits} > {promote}）—— 用 {best_name} 跑官方 5690")
    print(f"  参数: upscale={key(best_name)}  max_pixels≈{int(2073600*key(best_name)**2)}")
elif best_hits > baseline:
    print(f"结论: 略优但不达门槛（{best_hits} vs 需 >{promote}）—— 收益在噪声量级内，建议不提交")
else:
    print(f"结论: 无效（最佳 {best_hits} ≤ 基线 {baseline}）—— **不提交，保留现有 0.7870**")
print(f"单调性: {'分数随倍数单调不降' if mono else '非单调（说明模型对超训练分布的处理不稳定）'}")
PYEOF
