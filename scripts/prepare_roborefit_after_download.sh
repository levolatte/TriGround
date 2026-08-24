#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/final_training_upload/mm-grounding-adapters
dataset=/root/autodl-tmp/datasets/RoboRefIt
archive="$dataset/RoboRefIt.tar.gz"
download_pid_file="$dataset/download.pid"
expected_size=5023484669
expected_sha256=60053d01a7df8f64d4bfdc33bad502bf23f64200f0d12cdcfee7d31ba92d8066
extract_root="$dataset/extracted"
manifest_root="$dataset/manifests"

if [[ -s "$download_pid_file" ]]; then
  download_pid=$(cat "$download_pid_file")
  while kill -0 "$download_pid" 2>/dev/null; do
    state=$(awk '{print $3}' "/proc/$download_pid/stat" 2>/dev/null || true)
    [[ "$state" == Z ]] && break
    sleep 15
  done
fi

actual_size=$(stat -c %s "$archive")
[[ "$actual_size" == "$expected_size" ]] || {
  echo "archive size mismatch: expected=$expected_size actual=$actual_size"
  exit 1
}
echo "$expected_sha256  $archive" | sha256sum --check

archive_bytes=$(/root/miniconda3/bin/python - "$archive" <<'PY'
import pathlib
import sys
import tarfile

archive = pathlib.Path(sys.argv[1])
total = 0
with tarfile.open(archive, "r:gz") as handle:
    for member in handle:
        path = pathlib.PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise SystemExit(f"unsafe archive path: {member.name}")
        if member.issym() or member.islnk() or member.isdev():
            raise SystemExit(f"unsupported archive member: {member.name}")
        total += member.size
print(total)
PY
)
free_bytes=$(df --output=avail -B1 "$dataset" | tail -n 1 | tr -d ' ')
reserve_bytes=$((5 * 1024 * 1024 * 1024))
(( free_bytes >= archive_bytes + reserve_bytes )) || {
  echo "insufficient extraction space: free=$free_bytes archive_contents=$archive_bytes"
  exit 1
}
echo "validated archive: compressed=$actual_size contents=$archive_bytes free=$free_bytes"

if [[ ! -e "$extract_root/.complete" ]]; then
  [[ ! -e "$extract_root" ]] || {
    echo "refusing to reuse incomplete extraction directory: $extract_root"
    exit 1
  }
  mkdir -p "$extract_root"
  tar -xzf "$archive" -C "$extract_root"
  touch "$extract_root/.complete"
fi

mkdir -p "$manifest_root"
cd "$repo"
PYTHONPATH=src /root/miniconda3/bin/python tools/prepare_roborefit.py \
  --dataset-root "$extract_root" \
  --output-dir "$manifest_root" \
  --audit-depth-images 100

/root/miniconda3/bin/python tools/build_grouped_subsets.py \
  --manifest "$manifest_root/train.jsonl" \
  --output-dir "$manifest_root" \
  --validation-fraction 0 \
  --fractions 0.0138697 \
  --seed 2026
mv "$manifest_root/train_1.38697.jsonl" "$manifest_root/train_light_512.jsonl"
mv "$manifest_root/split_report.json" "$manifest_root/train_light_report.json"

/root/miniconda3/bin/python tools/build_grouped_subsets.py \
  --manifest "$manifest_root/testA.jsonl" \
  --output-dir "$manifest_root" \
  --validation-fraction 0 \
  --fractions 0.0150182 \
  --seed 2026
mv "$manifest_root/train_1.50182.jsonl" "$manifest_root/val_light_128.jsonl"
mv "$manifest_root/split_report.json" "$manifest_root/val_light_report.json"

PYTHONPATH=src /root/miniconda3/bin/python - <<'PY'
import json
from pathlib import Path
from mm_grounding.data import GroundingDataset

root = Path("/root/autodl-tmp/datasets/RoboRefIt/manifests")
for name in ("train_light_512.jsonl", "val_light_128.jsonl"):
    dataset = GroundingDataset(root / name, stage="depth")
    sample = dataset[0]
    print(json.dumps({
        "manifest": name,
        "samples": len(dataset),
        "first_id": sample["sample_id"],
        "rgb_size": sample["rgb"].size,
        "depth_size": sample["depth"].size,
    }))
PY
touch "$dataset/PREPARED"
echo "RoboRefIt lightweight manifests are ready; inspect depth audit before training."
