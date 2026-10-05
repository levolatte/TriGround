#!/usr/bin/env bash
# 07:00 起的自动链条（用户 2026-10-05 批准）：
#   1. 等 Arm 4 的 5690 出包结束
#   2. 补跑 Arm4 ck600 @ k1.0 × City412  → 填上四格配对里唯一空着的那格
#   3. **边到边扫** Arm 5 的七个检查点 @ k1.0 × City412（训练机在增量推送）
#   4. 取 Arm 5 最佳检查点 → k2.0 × 官方 5690 → 出包 → 格式闸门
#
# ## 为什么是"边到边扫"而不是等齐再扫
#
# 训练机的排期假设我 07:00 就能扫 Arm 5，但**Arm 5 要到 09:35 训练结束才归档完**
# （现在只到了 checkpoint-100）。等齐再扫会白等 2.5 小时。
# 改成"每个检查点一到就扫"，最后一个到达时前面的都已扫完 → **零等待**。
#
# ## 为什么 Arm4@k1.0 一定要跑
#
# 它是四格配对里唯一空着的一格：
#
#              k1.0        k2.0
#   ck500      312         321
#   Arm4 ck600  ?          317
#
# 只有填上它，才能回答"**放大对不同模型是否同样有效**"——
# 这正是视觉侧 LoRA 最该被检验的假设。7 分钟的成本。
# （此前把它与 317 相减得到的"+4"是**跨机器差值**，实测跨机器方差 ±3~5 题，故无效。）
set -uo pipefail

PY=/root/miniconda3/bin/python
VLLM_PY=/root/vllm_env/bin/python
cd /root/AIC_code || exit 2
export PATH=/root/vllm_env/bin:$PATH
LOG=/root/auto_chain_0505.log
exec > >(tee -a "${LOG}") 2>&1
say() { echo "[$(date '+%F %T')] $*"; }

MODEL=/root/rematch_models/Qwen3.6-27B
CITY_Q=/root/sprint_qwen36/city_val_queries.json
CITY_ROOT=/root/autodl-tmp/rematch_20260922/data/city/train
GT=/root/sprint_qwen36/baseline_preds/city412_gt.json
ARM4=/root/autodl-tmp/runs/q36_arm4_matched/archived/checkpoint-600
ARM5_ROOT=/root/autodl-tmp/runs/q36_arm5_corrected/archived
OFFICIAL=/root/autodl-tmp/rematch_20260922/data/official_test_rematch_20260924
S=/root/sprint_qwen36
# 等 Arm 5 检查点的上限（训练机说 09:35 训完，给到 11:00 富余）
ARM5_WAIT_LIMIT=15000   # 秒 ≈ 4.2 小时

wait_for_gpu() {
  local waited=0
  while [ "${waited}" -lt 300 ]; do
    [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)" -eq 0 ] && return 0
    sleep 15; waited=$((waited + 15))
  done
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    kill -9 "${pid}" 2>/dev/null || true
  done
  sleep 10
  [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)" -ne 0 ] && return 1
  return 0
}

# 单次 City412 推理。$1=适配器 $2=输出目录 $3=upscale $4=max_pixels $5=max_length $6=max_seqs $7=tag
run_city() {
  local adapter="$1" out="$2" up="$3" mp="$4" ml="$5" ms="$6" tag="$7"
  if [ -f "${out}/predictions.jsonl" ] && [ "$(wc -l < "${out}/predictions.jsonl")" -ge 412 ]; then
    say "${tag} 已有结果，跳过"; return 0
  fi
  wait_for_gpu || { say "${tag} GPU 不可用"; return 1; }
  say "跑 ${tag}（upscale=${up} max_pixels=${mp} max_length=${ml} max_seqs=${ms}）"
  rm -rf "${out}"
  "${VLLM_PY}" -u tools/predict_submission_vllm.py \
    --model "${MODEL}" --adapter "${adapter}" --modality trimodal \
    --queries "${CITY_Q}" --data-root "${CITY_ROOT}" --expected-query-count 412 \
    --min-pixels 200704 --max-pixels "${mp}" \
    --model-max-length "${ml}" --max-num-seqs "${ms}" \
    --chunk-size 64 --upscale "${up}" \
    --output-dir "${out}" > "${out}.log" 2>&1
  local st=$?
  [ "${st}" -ne 0 ] && { say "${tag} 失败（退出码 ${st}）"; tail -6 "${out}.log" | sed 's/^/    /'; return "${st}"; }
  say "${tag} 完成"
  return 0
}

# ---- 1. 等 Arm 4 的 5690 结束 ---------------------------------------------
say "等待 Arm 4 的 5690 出包链结束（screen arm4sub 消失）"
w=0
while screen -ls 2>/dev/null | grep -E '\.arm4sub' | grep -qE '(Attached|Detached)'; do
  [ "${w}" -ge 28800 ] && { say "等待超过 8 小时，退出"; exit 2; }
  sleep 60; w=$((w + 60))
done
say "arm4sub 已结束"

# ---- 2. 补 Arm4 ck600 @ k1.0（四格配对的空格）-----------------------------
run_city "${ARM4}" "${S}/arm4_ck600_k10" 1.0 2073600 8192 256 "Arm4 ck600 @ k1.0"
"${PY}" tools/verify_acc05_dev.py --gt "${GT}" \
  --arm ck500_k10=${S}/upscale_ladder/k1.0/predictions.jsonl \
  --arm ck500_k20=${S}/upscale_ladder/k2.0/predictions.jsonl \
  --arm arm4_k10=${S}/arm4_ck600_k10/predictions.jsonl \
  --arm arm4_k20=${S}/arm4_ck600_k20/predictions.jsonl \
  --out-json ${S}/grid_2x2.json 2>&1 | tail -12
say "四格配对表已写入 ${S}/grid_2x2.json"

# ---- 3. 边到边扫 Arm 5 的七个检查点 ---------------------------------------
say "开始边到边扫 Arm 5（每个检查点到达即扫）"
declare -a SCANNED=()
scanned_all=0
w=0
while [ "${w}" -lt "${ARM5_WAIT_LIMIT}" ]; do
  pending=0
  for ck in 100 200 300 400 500 600 700; do
    d="${ARM5_ROOT}/checkpoint-${ck}"
    # 只有适配器文件齐了才认为"到了"
    if [ -f "${d}/adapter_model.safetensors" ] && [ -f "${d}/adapter_config.json" ]; then
      if [ ! -f "${S}/arm5_ck${ck}_k10/predictions.jsonl" ]; then
        say "  checkpoint-${ck} 已到达，开始扫描"
        run_city "${d}" "${S}/arm5_ck${ck}_k10" 1.0 2073600 8192 256 "Arm5 ck${ck} @ k1.0" || true
      fi
    else
      pending=$((pending + 1))
    fi
  done
  if [ "${pending}" -eq 0 ]; then scanned_all=1; break; fi
  sleep 120; w=$((w + 120))
done
# 收尾：把已扫的分数汇总
have=$(ls -d ${S}/arm5_ck*_k10 2>/dev/null | wc -l)
say "Arm 5 扫描阶段结束：已扫 ${have} 个检查点（pending=${pending}，超时=${scanned_all}）"

declare -a ARMS=()
for ck in 100 200 300 400 500 600 700; do
  f="${S}/arm5_ck${ck}_k10/predictions.jsonl"
  [ -f "${f}" ] && ARMS+=(--arm "arm5_ck${ck}=${f}")
done
if [ "${#ARMS[@]}" -eq 0 ]; then say "!! 没有任何 Arm 5 检查点被扫到，退出"; exit 3; fi
"${PY}" tools/verify_acc05_dev.py --gt "${GT}" "${ARMS[@]}" \
  --arm ck500_k20=${S}/upscale_ladder/k2.0/predictions.jsonl \
  --out-json ${S}/arm5_sweep.json 2>&1 | tail -18

# ---- 4. 取 Arm 5 最佳检查点，跑 k2.0 × 5690 -------------------------------
BEST=$("${PY}" -c "
import json, pathlib
d = json.loads(pathlib.Path('${S}/arm5_sweep.json').read_text(encoding='utf-8'))
arms = {k: v['hits_0.5'] for k, v in d['arms'].items() if k.startswith('arm5_')}
if not arms:
    print('')
else:
    best = max(arms, key=arms.get)
    print(best.split('_')[1])
")
if [ -z "${BEST}" ]; then say "!! 无法确定最佳检查点，退出"; exit 4; fi
say "Arm 5 最佳 = ${BEST}"
say "对照：ck500 @ k2.0 = 321；若 Arm 5 明显更好，其 k2.0 值得提交"

OUT5="${S}/submit_arm5_${BEST}_k20"
wait_for_gpu || { say "GPU 不可用"; exit 5; }
if [ -f "${OUT5}/predictions.jsonl" ] && [ "$(wc -l < "${OUT5}/predictions.jsonl")" -ge 5690 ]; then
  say "Arm 5 的 5690 已有结果，跳过"
else
  say "=== Arm 5 ${BEST} @ k2.0 × 官方 5690（约 6.4 小时）==="
  rm -rf "${OUT5}"
  "${VLLM_PY}" -u tools/predict_submission_vllm.py \
    --model "${MODEL}" --adapter "${ARM5_ROOT}/checkpoint-${BEST}" --modality trimodal \
    --queries "${OFFICIAL}/queries/queries.json" --data-root "${OFFICIAL}" \
    --expected-query-count 5690 \
    --min-pixels 200704 --max-pixels 8294400 \
    --model-max-length 32768 --max-num-seqs 32 \
    --chunk-size 256 --upscale 2.0 \
    --output-dir "${OUT5}" > "${OUT5}.log" 2>&1
  st=$?
  say "5690 退出码 ${st}"
  [ "${st}" -ne 0 ] && { tail -10 "${OUT5}.log" | sed 's/^/  /'; exit "${st}"; }
fi

say "解析率检查"
"${PY}" -c "
import json, pathlib
d = json.loads(pathlib.Path('${OUT5}/summary.json').read_text(encoding='utf-8'))
pf, fb, s2 = d.get('parse_failures'), d.get('fallback_predictions'), d.get('stage2_retries')
print(f\"  完成 {d.get('completed_queries')}  解析失败 {pf}  兜底 {fb}  阶段2重试 {s2}\")
print('  ** 解析率满分 **' if pf == 0 and fb == 0 else '  ** 解析率未满分 **')
"
wait_for_gpu || true
say "格式闸门"
"${PY}" tools/check_submission_format.py --queries "${OFFICIAL}/queries/queries.json" \
  --submission "${OUT5}/predictions.json" --zip "${OUT5}/submission.zip" 2>&1 | tail -10

{
  echo ""
  echo "## 【推理机 → 训练机】$(date '+%F %T') —— 自动链条完成"
  echo ""
  echo "* \`Arm4 ck600 @ k1.0\` 已补：结果在 \`${S}/grid_2x2.json\`（四格配对表）"
  echo "* \`Arm5 七点扫描\`：结果在 \`${S}/arm5_sweep.json\`，最佳 = **${BEST}**"
  echo "* \`Arm5 ${BEST} @ k2.0 × 5690\` 出包：\`${OUT5}/submission.zip\`"
  echo "* 请核验并把 zip 路径转达队长提交。"
} >> /root/INTER_AGENT.md
say "已把结果写进 /root/INTER_AGENT.md"
say "全部完成"
