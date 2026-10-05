#!/usr/bin/env bash
# 给训练机的 arm5sub 补收尾：等它跑完 → 解析率检查 → 格式闸门。
#
# ## 为什么需要
#
# 训练机接手跑的 `arm5sub` 是**裸 python 命令**（`screen -dmS arm5sub bash -c "python ..."`），
# 没有外层脚本，所以**跑完不会自动做解析率检查与格式闸门**。
# 而 `predict_submission_vllm.py` 自己会产出 zip，所以只缺这两项验证。
#
# ## 绝不与它抢 GPU
#
# 本脚本**只读不写 GPU**：不启动任何推理，只在对方进程退出后做 CPU 侧的校验与打包。
set -uo pipefail

PY=/root/miniconda3/bin/python
OFFICIAL=/root/autodl-tmp/rematch_20260922/data/official_test_rematch_20260924
OUT=/root/sprint_qwen36/submit_arm5_ck700_k20
LOG=/root/gate_arm5.log
exec > >(tee -a "${LOG}") 2>&1
say() { echo "[$(date '+%F %T')] $*"; }

say "等待训练机的 arm5sub 结束"
w=0
while screen -ls 2>/dev/null | grep -q '\.arm5sub'; do
  [ "${w}" -ge 25200 ] && { say "等待超过 7 小时，退出"; exit 2; }
  sleep 60; w=$((w + 60))
done
say "arm5sub 已结束"

n=$(wc -l < "${OUT}/predictions.jsonl" 2>/dev/null || echo 0)
say "predictions.jsonl 条数：${n}"
if [ "${n}" -lt 5690 ]; then
  say "!! 不足 5690 条 —— 训练机那轮可能被中断，需人工判断"
  tail -6 "${OUT}.log" 2>/dev/null | sed 's/^/  /'
  exit 3
fi

say "解析率与兜底（现役标准：100% 解析、0 兜底、0 重试）"
"${PY}" -c "
import json, pathlib
p = pathlib.Path('${OUT}/summary.json')
if p.is_file():
    d = json.loads(p.read_text(encoding='utf-8'))
    pf, fb, s2 = d.get('parse_failures'), d.get('fallback_predictions'), d.get('stage2_retries')
    print(f\"  完成 {d.get('completed_queries')}  解析失败 {pf}  兜底 {fb}  阶段2重试 {s2}\")
    print('  ** 解析率满分 **' if pf == 0 and fb == 0 else '  ** 解析率未满分 **')
else:
    print('  (无 summary.json)')
"

if [ ! -f "${OUT}/submission.zip" ]; then
  say "没有 zip，尝试用 --package-only 补打包"
  /root/vllm_env/bin/python tools/predict_submission_vllm.py \
    --model /root/rematch_models/Qwen3.6-27B --modality trimodal \
    --queries "${OFFICIAL}/queries/queries.json" --data-root "${OFFICIAL}" \
    --output-dir "${OUT}" --package-only 2>&1 | tail -5 | sed 's/^/  /'
fi

say "格式闸门（必须 SUBMISSION_FORMAT_OK）"
"${PY}" tools/check_submission_format.py --queries "${OFFICIAL}/queries/queries.json" \
  --submission "${OUT}/predictions.json" --zip "${OUT}/submission.zip" 2>&1 | tail -10

{
  echo ""
  echo "## 【推理机 → 训练机】$(date '+%F %T') —— arm5sub 收尾完成"
  echo ""
  echo "* 我给 \`arm5sub\` 补跑了**解析率检查与格式闸门**（你那条是裸命令，没有收尾）。"
  echo "* 产物：\`${OUT}/submission.zip\`（$(wc -c < "${OUT}/submission.zip" 2>/dev/null || echo 0) 字节）"
  echo "* 建议：**该补的收尾以后写进脚本**，否则容易只出 zip 而没验证。"
} >> /root/INTER_AGENT.md
say "完成，已写入信箱"
