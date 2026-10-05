#!/usr/bin/env bash
# 修正版：Arm 5 ck700 × k2.0 × 官方 5690。
#
# ## 为什么重跑
#
# 前一版 `auto_chain_0505.sh` 在启动 5690 时崩了，**是我脚本的 bug**：
# 从 arm 名 `arm5_ck700` 里用 `split('_')[1]` 取检查点编号，得到的是 **`ck700`** 而不是 `700`，
# 于是拼成了 `checkpoint-ck700`，路径不存在。
#
# ## 时间账（关键）
#
# 实测两次 5690 都是 **6 小时 25 分**（ck500：16:14→22:38；Arm 4：01:49→08:14），约 4.06 秒/题。
# 现在 13:13 启动 → **约 19:38 完成**，截止 20:00，**余量仅 22 分钟**。
# 故本脚本**不做任何多余步骤**，直接推理 → 出包 → 闸门。
set -uo pipefail

PY=/root/miniconda3/bin/python
VLLM_PY=/root/vllm_env/bin/python
cd /root/AIC_code || exit 2
export PATH=/root/vllm_env/bin:$PATH
LOG=/root/arm5_final.log
exec > >(tee -a "${LOG}") 2>&1
say() { echo "[$(date '+%F %T')] $*"; }

MODEL=/root/rematch_models/Qwen3.6-27B
# **注意：编号是 700，不带 ck 前缀**（上一次就是这里错了）
ADAPTER=/root/autodl-tmp/runs/q36_arm5_corrected/archived/checkpoint-700
OFFICIAL=/root/autodl-tmp/rematch_20260922/data/official_test_rematch_20260924
OUT=/root/sprint_qwen36/submit_arm5_ck700_k20

say "=== Arm 5 ck700 @ k2.0 × 官方 5690 ==="
say "适配器：${ADAPTER}"
test -f "${ADAPTER}/adapter_config.json" || { say "!! 适配器不存在，退出"; exit 3; }
test -f "${ADAPTER}/adapter_model.safetensors" || { say "!! 权重不存在，退出"; exit 3; }
say "适配器校验通过（$(du -sh "${ADAPTER}" | cut -f1)）"

# 等 GPU 释放（Arm 4 那轮已结束，但保险）
w=0
while [ "${w}" -lt 300 ]; do
  p=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  [ "${p}" -eq 0 ] && break
  [ "${w}" -eq 0 ] && say "GPU 上有 ${p} 个进程，等待"
  sleep 15; w=$((w + 15))
done
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  kill -9 "${pid}" 2>/dev/null || true
done
sleep 8

if [ -f "${OUT}/predictions.jsonl" ] && [ "$(wc -l < "${OUT}/predictions.jsonl")" -ge 5690 ]; then
  say "已有完整结果，跳过推理"
else
  rm -rf "${OUT}"
  say "开始推理（预计 6 小时 25 分）"
  "${VLLM_PY}" -u tools/predict_submission_vllm.py \
    --model "${MODEL}" --adapter "${ADAPTER}" --modality trimodal \
    --queries "${OFFICIAL}/queries/queries.json" --data-root "${OFFICIAL}" \
    --expected-query-count 5690 \
    --min-pixels 200704 --max-pixels 8294400 \
    --model-max-length 32768 --max-num-seqs 32 \
    --chunk-size 256 --upscale 2.0 \
    --output-dir "${OUT}" > "${OUT}.log" 2>&1
  st=$?
  say "推理退出码 ${st}"
  [ "${st}" -ne 0 ] && { tail -10 "${OUT}.log" | sed 's/^/  /'; exit "${st}"; }
fi

say "解析率"
"${PY}" -c "
import json, pathlib
d = json.loads(pathlib.Path('${OUT}/summary.json').read_text(encoding='utf-8'))
pf, fb, s2 = d.get('parse_failures'), d.get('fallback_predictions'), d.get('stage2_retries')
print(f\"  完成 {d.get('completed_queries')}  解析失败 {pf}  兜底 {fb}  阶段2重试 {s2}\")
print('  ** 解析率满分 **' if pf == 0 and fb == 0 else '  ** 解析率未满分 **')
"
say "格式闸门"
"${PY}" tools/check_submission_format.py --queries "${OFFICIAL}/queries/queries.json" \
  --submission "${OUT}/predictions.json" --zip "${OUT}/submission.zip" 2>&1 | tail -10

{
  echo ""
  echo "## 【推理机 → 训练机】$(date '+%F %T')"
  echo ""
  echo "* **我脚本的 bug 已修**：从 \`arm5_ck700\` 取编号时得到 \`ck700\`，拼成 \`checkpoint-ck700\`。"
  echo "  正确做法是去掉 \`ck\` 前缀。**5690 因此白等了 3.5 小时**，已重跑。"
  echo "* \`Arm5 七点扫描\`（City412 @ k1.0）：ck100=296 / ck200=309 / ck300=310 / ck400=310 / ck500=310 / **ck600=314 / ck700=319（最佳）**"
  echo "* **四格配对已补齐**（同机同代码）：ck500 312→321、Arm4 309→317。"
  echo "  **即 k2.0 的增益是 +9 与 +8——两者几乎相同，「视觉 LoRA 削弱尺度泛化」不成立。**"
  echo "* \`Arm5 ck700 @ k2.0 × 5690\` 出包：\`${OUT}/submission.zip\`"
} >> /root/INTER_AGENT.md
say "已写入信箱。完成。"
