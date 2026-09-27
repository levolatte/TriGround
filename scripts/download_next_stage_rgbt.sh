#!/usr/bin/env bash
set -euo pipefail
ROOT="${REMATCH_ROOT:-/root/autodl-tmp/rematch_20260922}"
OUT="$ROOT/data/external/RGBT-GroundBench"
mkdir -p "$OUT/archives" "$OUT/raw"
# Use the platform's accelerator when it is installed; no subscription is needed here.
if [[ -f /etc/network_turbo ]]; then source /etc/network_turbo > /dev/null 2>&1; fi
BASE=https://huggingface.co/datasets/JiawenXi/RGBT-Ground-Dataset/resolve/main
if ! curl -fsSL --range 0-1023 --max-time 30 "$BASE/ann_flir.tar" -o "$OUT/connection_probe.bin"; then
  BASE=https://hf-mirror.com/datasets/JiawenXi/RGBT-Ground-Dataset/resolve/main
fi
printf '%s\tendpoint\t%s\n' "$(date -Is)" "$BASE" >> "$OUT/download_status.tsv"
for file in ann_flir.tar ann_m3fd.tar ann_mfad.tar data_flir.tar data_mfad.tar data_m3fd.tar; do
  case "$file" in
    ann_flir.tar) size=3307520;; ann_m3fd.tar) size=4567040;; ann_mfad.tar) size=8529920;;
    data_flir.tar) size=935802880;; data_mfad.tar) size=2191175680;; data_m3fd.tar) size=7246704640;;
  esac
  if [[ ! -f "$OUT/archives/$file" ]] || [[ "$(stat -c %s "$OUT/archives/$file")" != "$size" ]]; then
    printf '%s\t%s\tdownload\n' "$(date -Is)" "$file" >> "$OUT/download_status.tsv"
    curl -fL --connect-timeout 30 --retry 2 --retry-delay 30 --continue-at - \
      "$BASE/$file" -o "$OUT/archives/$file"
    [[ "$(stat -c %s "$OUT/archives/$file")" == "$size" ]]
  fi
  if [[ ! -f "$OUT/$file.extracted" ]]; then
    /root/miniconda3/bin/python - "$OUT" "$OUT/archives/$file" <<'PY'
import shutil,sys,tarfile
from pathlib import Path
root=Path(sys.argv[1]); archive=Path(sys.argv[2])
with tarfile.open(archive) as tf:
    needed=sum(m.size for m in tf.getmembers())
    assert shutil.disk_usage(root).free >= needed+3*1024**3, 'Insufficient extraction space'
    tf.extractall(root/'raw', filter='data')
PY
    touch "$OUT/$file.extracted"
    printf '%s\t%s\tcomplete\n' "$(date -Is)" "$file" >> "$OUT/download_status.tsv"
  fi
done
printf '%s\trgbt_archives\tcomplete_prepare_manifest_next\n' "$(date -Is)" >> "$OUT/download_status.tsv"
