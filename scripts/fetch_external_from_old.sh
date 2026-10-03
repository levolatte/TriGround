#!/usr/bin/env bash
# 从旧机拉取全部外部数据集（新机主动拉，旧机只读——这是当时建立的通道方向）。
#
# 为什么废弃新机的下载：云端互传实测约 43 MB/s，而新机直连 HF 镜像只有 1.6 MB/s，
# 差 27 倍。旧机上不仅已下全，还带好了转换产物（converted/*.jsonl），
# 直接拉过来比重新下载再加转换省事得多。
#
# 方向很重要：旧机**无法**主动连新机（试过，不通）；通道是"新机持私钥去拉旧机"。
set -uo pipefail

SSH_CMD="ssh -i ~/.ssh/transfer_ed25519 -p 46057 -o StrictHostKeyChecking=accept-new -o BatchMode=yes -o ConnectTimeout=15"
OLD="root@connect.westc.seetacloud.com"
OLD_EXT="/root/autodl-tmp/rematch_20260922/data/external"
DEST="${DEST:-/root/autodl-tmp/rematch_20260922/data/external}"

mkdir -p "${DEST}"
say() { echo "[$(date '+%F %T')] $*"; }

say "先核对旧机上的实际内容"
${SSH_CMD} "${OLD}" "du -sh ${OLD_EXT}/* 2>/dev/null; echo '---'; find ${OLD_EXT} -maxdepth 2 -name '*.jsonl' -printf '%s %p\n' 2>/dev/null | sort -rn | head -8"

for dataset in RGBDT500 RGBT-GroundBench RoboRefIt; do
  say "拉取 ${dataset}"
  rsync -a --partial --info=progress2 \
    -e "${SSH_CMD}" \
    "${OLD}:${OLD_EXT}/${dataset}/" "${DEST}/${dataset}/"
  status=$?
  if [ "${status}" -ne 0 ]; then
    say "!! ${dataset} 拉取失败（rsync 退出码 ${status}）"
    exit "${status}"
  fi
  say "${dataset} 完成：$(du -sh "${DEST}/${dataset}" | cut -f1)"
done

say "全部完成"
du -sh "${DEST}"/*
