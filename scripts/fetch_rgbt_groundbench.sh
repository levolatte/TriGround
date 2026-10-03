#!/usr/bin/env bash
# 从 hf-mirror.com 下载 RGBT-GroundBench（HF 直连在云端被墙，镜像可用）。
#
# 已由调研子 Agent 实测确认的结构：
#   ann_{flir,m3fd,mfad}.tar  -> *.pth（torch ZIP 归档，标准库 zipfile+pickle 可读，无需 torch）
#   data_{flir,m3fd,mfad}.tar -> image_data/<src>/{rgb,ir}/*.jpg，同名一一对应
# ⚠️ 标注里 bbox 是 [x, y, w, h] 绝对像素，不是 xyxy —— 转换时必须换算式。
set -uo pipefail

DEST="${DEST:-/root/autodl-tmp/rematch_20260922/data/external/RGBT-GroundBench}"
BASE="https://hf-mirror.com/datasets/JiawenXi/RGBT-Ground-Dataset/resolve/main"
FILES=(ann_flir.tar ann_m3fd.tar ann_mfad.tar data_flir.tar data_mfad.tar data_m3fd.tar)

mkdir -p "${DEST}"
cd "${DEST}" || exit 2

for name in "${FILES[@]}"; do
  echo "[$(date +%H:%M:%S)] === ${name} ==="
  # -C - 断点续传；--retry 抗抖动；下载到 .part 再改名，避免半成品被当成完整文件
  curl -L -C - --retry 8 --retry-delay 5 --retry-all-errors \
       --connect-timeout 30 --speed-time 120 --speed-limit 1024 \
       -o "${name}.part" "${BASE}/${name}" 2>&1 | tail -2
  status=${PIPESTATUS[0]}
  if [ "${status}" -ne 0 ]; then
    echo "[$(date +%H:%M:%S)] !! ${name} 下载失败，退出码 ${status}（保留 .part 以便续传）"
    exit "${status}"
  fi
  mv -f "${name}.part" "${name}"
  echo "[$(date +%H:%M:%S)] 完成 ${name}  $(du -h "${name}" | cut -f1)"
done

echo "[$(date +%H:%M:%S)] 全部下载完成"
du -sh "${DEST}"
ls -la "${DEST}"
