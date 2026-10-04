#!/usr/bin/env bash
# 让 k2.5 跑完，然后**在切到 k3.0 之前**截停整条阶梯。
#
# ## 为什么
#
# k2.5 已在同一批 256 题上给出明确结论（相对 k2.0 极小桶 −12、总数 −11），
# 决策已定：**k=2.0 是峰值**。
# 但 k2.5 只差约 24 分钟就能跑满 412 题，完整口径的记录更干净，成本近乎为零（GPU 本就空闲）。
#
# 而 **k3.0 不跑**：即便它有效，官方推理需约 20 小时，性价比不成立（用户决策）。
# 阶梯脚本会在 k2.5 结束后自动切到 k3.0，所以必须在这个窗口里截停。
set -uo pipefail

K25=/root/sprint_qwen36/upscale_ladder2/k2.5/predictions.jsonl
LOG=/root/stop_before_k30.log
exec > >(tee -a "${LOG}") 2>&1
say() { echo "[$(date '+%F %T')] $*"; }

say "等待 k2.5 跑满 412 条（当前 $(wc -l < "${K25}" 2>/dev/null || echo 0) 条）"
waited=0
while true; do
  n=$(wc -l < "${K25}" 2>/dev/null || echo 0)
  [ "${n}" -ge 412 ] && break
  # 若 k2.5 的 screen 已经消失且条数不够，说明阶梯异常结束
  if ! screen -ls 2>/dev/null | grep -q '\.ladder2'; then
    say "阶梯会话已消失，而 k2.5 只有 ${n} 条 —— 不再等待，直接收尾"
    break
  fi
  [ "${waited}" -ge 7200 ] && { say "等待超过 2 小时，强制收尾"; break; }
  sleep 30
  waited=$((waited + 30))
done
say "k2.5 现有 $(wc -l < "${K25}" 2>/dev/null || echo 0) 条"

# 截停：先杀阶梯会话，再清 vLLM 残留。
# 顺序重要——先杀会话才能阻止脚本继续启动 k3.0。
say "截停阶梯（阻止 k3.0 启动）"
screen -S ladder2 -X quit 2>/dev/null || true
sleep 3
pkill -9 -f upscale_ladder.sh 2>/dev/null || true
sleep 2
pkill -9 -f predict_submission_vllm 2>/dev/null || true
sleep 5
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  say "清理残留 GPU 进程 ${pid}"
  kill -9 "${pid}" 2>/dev/null || true
done
sleep 6

say "收尾状态："
say "  剩余 screen     : $(screen -ls 2>/dev/null | grep -c Detached)"
say "  GPU 计算进程    : $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)"
say "  k2.5 条数       : $(wc -l < "${K25}" 2>/dev/null || echo 0)"
say "  k3.0 是否被创建 : $([ -f /root/sprint_qwen36/upscale_ladder2/k3.0/predictions.jsonl ] && echo '是（应清理）' || echo '否 ✓')"
