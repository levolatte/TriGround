#!/usr/bin/env bash
set -euo pipefail
ROOT="${REMATCH_ROOT:-/root/autodl-tmp/rematch_20260922}"
OUT="$ROOT/data/external/RoboRefIt"
mkdir -p "$OUT"
if [[ -f /etc/network_turbo ]]; then source /etc/network_turbo > /dev/null 2>&1; fi
URL='https://drive.usercontent.google.com/download?id=1pdGF1HaU_UiKfh5Z618hy3nRjVbq_VuW&export=download&confirm=t'
printf '%s\trobo\tdownload_start\n' "$(date -Is)" >> "$OUT/download_status.tsv"
curl -fL --connect-timeout 30 --retry 2 --retry-delay 30 --continue-at - "$URL" -o "$OUT/final_dataset.rar"
[[ "$(stat -c %s "$OUT/final_dataset.rar")" == 6084791269 ]]
printf '%s\trobo\tdownload_complete_extract_next\n' "$(date -Is)" >> "$OUT/download_status.tsv"
