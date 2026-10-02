#!/usr/bin/env bash
# 从旧实例（原 4090 机）把冲刺所需的全部数据与权重拉到新实例（RTX PRO 6000）。
#
# 前置：新机 ~/.ssh/transfer_ed25519 的公钥已被授权到旧机。
# 关键设计：目标路径与旧机完全一致，因此所有 manifest 里的绝对路径无需重写。
set -euo pipefail

OLD_HOST="${OLD_HOST:-connect.westc.seetacloud.com}"
OLD_PORT="${OLD_PORT:-46057}"
OLD_USER="${OLD_USER:-root}"
OLD_KEY="${OLD_KEY:-$HOME/.ssh/transfer_ed25519}"
OLD_ROOT="${OLD_ROOT:-/root/autodl-tmp/rematch_20260922}"

ROOT="/root/autodl-tmp/rematch_20260922"
SSH_CMD="ssh -i ${OLD_KEY} -p ${OLD_PORT} -o StrictHostKeyChecking=accept-new -o BatchMode=yes"
RSYNC="rsync -a --partial --info=progress2 -e \"${SSH_CMD}\""

say() { echo "[$(date +%H:%M:%S)] $*"; }

say "0/6 带宽抽样（200MB）"
timeout 120 rsync -a --partial -e "${SSH_CMD}" \
  --bwlimit=0 "${OLD_USER}@${OLD_HOST}:${OLD_ROOT}/data/city/train/target_v2/" \
  /root/autodl-tmp/_speedtest/ 2>&1 | tail -3 || true
du -sh /root/autodl-tmp/_speedtest 2>/dev/null || true

say "1/6 City 训练与开发图像（5.4G）"
mkdir -p "${ROOT}/data/city/train"
eval ${RSYNC} "${OLD_USER}@${OLD_HOST}:${ROOT}/data/city/train/" "${ROOT}/data/city/train/"

say "2/6 评估输入清单（city_val.json / city_gt.json）"
mkdir -p "${ROOT}/results/triground_abv_20260927/inputs"
eval ${RSYNC} "${OLD_USER}@${OLD_HOST}:${ROOT}/results/triground_abv_20260927/inputs/" \
  "${ROOT}/results/triground_abv_20260927/inputs/"

say "3/6 训练清单 A.json"
mkdir -p "${ROOT}/results/triground_abv_execution_20260928/deployment_600_seed2026/manifests"
eval ${RSYNC} "${OLD_USER}@${OLD_HOST}:${ROOT}/results/triground_abv_execution_20260928/deployment_600_seed2026/manifests/A.json" \
  "${ROOT}/results/triground_abv_execution_20260928/deployment_600_seed2026/manifests/"

say "4/6 基线 A 的语言 LoRA（排除中间 checkpoint-*，只留最终 adapter）"
mkdir -p "${ROOT}/results/triground_abv_execution_20260928/runs/seed2026_600_deterministic/A/main"
eval ${RSYNC} --exclude 'checkpoint-*' \
  "${OLD_USER}@${OLD_HOST}:${ROOT}/results/triground_abv_execution_20260928/runs/seed2026_600_deterministic/A/main/" \
  "${ROOT}/results/triground_abv_execution_20260928/runs/seed2026_600_deterministic/A/main/"

say "5/6 Qwen3-VL-8B-Instruct 基座（17G，用于第一跳探针与兜底）"
mkdir -p /root/rematch_models
eval ${RSYNC} "${OLD_USER}@${OLD_HOST}:/root/rematch_models/Qwen3-VL-8B-Instruct/" \
  "/root/rematch_models/Qwen3-VL-8B-Instruct/"

say "6/6 旧训练框架留档（248K）+ 旧 code_snapshot"
eval ${RSYNC} "${OLD_USER}@${OLD_HOST}:${ROOT}/third_party/" "${ROOT}/third_party/"

say "=== 迁移完成核对 ==="
du -sh "${ROOT}/data/city/train"/* 2>/dev/null
echo "--- 训练/开发图像文件数"
for d in visible infrared target_v2/qwen3vl_native_sft/depth_rgb; do
  printf '%-42s %s\n' "$d" "$(find "${ROOT}/data/city/train/$d" -type f | wc -l)"
done
echo "--- 评估输入"
ls -la "${ROOT}/results/triground_abv_20260927/inputs/"
echo "--- A adapter"
ls -la "${ROOT}/results/triground_abv_execution_20260928/runs/seed2026_600_deterministic/A/main/" | head -12
echo "--- 8B 基座"
du -sh /root/rematch_models/Qwen3-VL-8B-Instruct
echo "--- 磁盘"
df -h /root/autodl-tmp | tail -1
rm -rf /root/autodl-tmp/_speedtest
say "ALL_DONE"
