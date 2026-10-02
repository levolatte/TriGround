#!/usr/bin/env bash
# 从旧实例把官方复赛测试集与辅助数据拉到新实例（云间直传，走内网级带宽）。
#
# 前置：新机 ~/.ssh/transfer_ed25519 已授权到旧机（阶段 A 建立的一次性传输密钥）。
# 目标路径与旧机完全一致，因此 predict_native_submission.py 的 queries 路径无需改写。
set -uo pipefail

OLD_HOST="${OLD_HOST:-connect.westc.seetacloud.com}"
OLD_PORT="${OLD_PORT:-46057}"
OLD_USER="${OLD_USER:-root}"
OLD_KEY="${OLD_KEY:-$HOME/.ssh/transfer_ed25519}"
ROOT="/root/autodl-tmp/rematch_20260922"
SSH_CMD="ssh -i ${OLD_KEY} -p ${OLD_PORT} -o StrictHostKeyChecking=accept-new -o BatchMode=yes"

say() { echo "[$(date +%H:%M:%S)] $*"; }

if [ ! -f "${OLD_KEY}" ]; then
  echo "缺少传输密钥 ${OLD_KEY}，无法直传" >&2
  exit 2
fi

# 先做一次连通性与路径存在性检查，避免传一半才发现路径写错
say "0/4 连通性检查"
${SSH_CMD} "${OLD_USER}@${OLD_HOST}" \
  "du -sh ${ROOT}/data/official_test_rematch_20260924 ${ROOT}/data/city_depth_expansion_20260928 ${ROOT}/data/city_object_extension" \
  || { echo "旧机不可达或路径不存在" >&2; exit 3; }

say "1/4 官方复赛测试集（13G，5690 条查询）"
mkdir -p "${ROOT}/data"
rsync -a --partial --info=progress2 -e "${SSH_CMD}" \
  "${OLD_USER}@${OLD_HOST}:${ROOT}/data/official_test_rematch_20260924/" \
  "${ROOT}/data/official_test_rematch_20260924/"

say "2/4 深度扩展数据（148M）"
rsync -a --partial -e "${SSH_CMD}" \
  "${OLD_USER}@${OLD_HOST}:${ROOT}/data/city_depth_expansion_20260928/" \
  "${ROOT}/data/city_depth_expansion_20260928/"

say "3/4 目标扩展数据（99M）"
rsync -a --partial -e "${SSH_CMD}" \
  "${OLD_USER}@${OLD_HOST}:${ROOT}/data/city_object_extension/" \
  "${ROOT}/data/city_object_extension/"

say "4/4 校验"
for d in official_test_rematch_20260924 city_depth_expansion_20260928 city_object_extension; do
  printf '  %-38s %s\n' "${d}" "$(du -sh ${ROOT}/data/${d} 2>/dev/null | cut -f1)"
done
echo "--- 模态文件数"
for m in visible infrared depth; do
  printf '  %-10s %s\n' "${m}" \
    "$(find ${ROOT}/data/official_test_rematch_20260924/Images/${m} -type f 2>/dev/null | wc -l)"
done
echo "--- queries"
ls -la "${ROOT}/data/official_test_rematch_20260924/queries/queries.json" 2>/dev/null
echo "PULL_OFFICIAL_TEST_DONE"
