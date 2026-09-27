#!/usr/bin/env bash
set -euo pipefail
ROOT="${REMATCH_ROOT:-/root/autodl-tmp/rematch_20260922}"
OUT="$ROOT/data/external/RGBT-GroundBench"
FILE="$OUT/archives/data_m3fd.tar"
URL=https://hf-mirror.com/datasets/JiawenXi/RGBT-Ground-Dataset/resolve/main/data_m3fd.tar
# The original platform transfer ended with curl 18. These are its two allowed extra attempts.
for attempt in 2 3; do
  printf '%s\tdata_m3fd.tar\tdirect_mirror_resume_attempt%s\n' "$(date -Is)" "$attempt" >> "$OUT/download_status.tsv"
  if env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    curl -fL --connect-timeout 30 --continue-at - "$URL" -o "$FILE"; then break; else rc=$?; fi
  case "$rc" in 6|7|18|28|52|56) ;; *) exit "$rc";; esac
  if [[ "$attempt" == 3 ]]; then exit "$rc"; fi
  sleep 60
done
[[ "$(stat -c %s "$FILE")" == 7246704640 ]]
/root/miniconda3/bin/python - "$OUT" "$FILE" <<'PY'
import shutil,sys,tarfile
from pathlib import Path
p=Path(sys.argv[1])
with tarfile.open(sys.argv[2]) as tf:
    needed=sum(m.size for m in tf.getmembers())
    assert shutil.disk_usage(p).free >= needed+3*1024**3
    tf.extractall(p/'raw',filter='data')
PY
touch "$OUT/data_m3fd.tar.extracted"
printf '%s\trgbt_archives\tcomplete_prepare_manifest_next\n' "$(date -Is)" >> "$OUT/download_status.tsv"
