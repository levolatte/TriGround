#!/usr/bin/env bash
# 等 k2.0 链跑完，对**纯 k2.0 的包**做格式闸门并出最终报告。
#
# ## 为什么要这个
#
# 决策已定：**提交纯 k2.0**，不用分路合并。
# 理由是分路经实测与纯 k2.0 **完全打平**（McNemar p=1.0，净 0 题）——
# 它救回 2 题、又丢掉 2 题，而它声称要保护的 17 个退化里只有 2 个救得回来。
# **同分的情况下，少一层后处理、少一个超参、少一个失败面。**
#
# 但正在跑的链会顺带产出合并包（无害，只是多一个目录）。它的格式闸门跑在合并包上，
# 所以**纯 k2.0 的包还需要单独过一遍闸门**才能交。
#
# 纯 k2.0 的包由 `predict_submission_vllm.py` 自己在推理结束时产出
# （`--output-dir` 下的 `predictions.json` 与 `submission.zip`），不需要额外打包步骤。
set -uo pipefail

PY=/root/miniconda3/bin/python
OFFICIAL="${OFFICIAL:-/root/autodl-tmp/rematch_20260922/data/official_test_rematch_20260924}"
K20_OUT="${K20_OUT:-/root/sprint_qwen36/submit_upscale_k2.0}"
LOG="${LOG:-/root/gate_k20.log}"

exec > >(tee -a "${LOG}") 2>&1
say() { echo "[$(date '+%F %T')] $*"; }
cd /root/AIC_code || exit 2

say "等待 k2.0 链结束（screen 会话 k20 消失）"
waited=0
while screen -ls 2>/dev/null | grep -E '\.k20' | grep -qE '(Attached|Detached)'; do
  [ "${waited}" -ge 43200 ] && { say "等待超过 12 小时，退出"; exit 2; }
  sleep 60
  waited=$((waited + 60))
done
say "k2.0 链已结束"

if [ ! -f "${K20_OUT}/predictions.jsonl" ]; then
  say "!! 纯 k2.0 没有产出结果：${K20_OUT}/predictions.jsonl 不存在"
  exit 3
fi
rows=$(wc -l < "${K20_OUT}/predictions.jsonl")
say "纯 k2.0 预测条数：${rows}"

say "解析率与兜底（现役标准：100% 解析、0 兜底、0 重试）"
"${PY}" -c "
import json, pathlib
p = pathlib.Path('${K20_OUT}/summary.json')
if p.is_file():
    d = json.loads(p.read_text(encoding='utf-8'))
    pf, fb, s2 = d.get('parse_failures'), d.get('fallback_predictions'), d.get('stage2_retries')
    print(f\"  完成 {d.get('completed_queries')}  解析失败 {pf}  兜底 {fb}  阶段2重试 {s2}\")
    print('  ** 解析率满分 **' if pf == 0 and fb == 0 else '  ** 解析率未满分 —— 不建议提交 **')
else:
    print('  (无 summary.json)')
"

say "格式闸门（必须 SUBMISSION_FORMAT_OK）"
"${PY}" tools/check_submission_format.py --queries "${OFFICIAL}/queries/queries.json" \
  --submission "${K20_OUT}/predictions.json" --zip "${K20_OUT}/submission.zip" 2>&1 | tail -10

echo
echo "================ 最终提交包（纯 k2.0）================"
echo "zip    : ${K20_OUT}/submission.zip"
echo "做法   : 三张图各放大 2 倍（1920x1080 -> 3840x2160），"
echo "         max_pixels 8294400 / max_length 32768 / max_num_seqs 32"
echo "对照   : 现役最好 0.7870（不放大，本地 314/312）"
echo "         City412 本机——k1.0 312  ->  k2.0 321（+9）"
echo "         极小桶 75 -> 90（+15）；小桶 124 -> 117（-7）"
echo "备注   : 分路合并经实测与纯 k2.0 打平（p=1.0），故不采用"
echo "         剩余提交机会：3 次（取最好一次成绩）"
echo "======================================================"
