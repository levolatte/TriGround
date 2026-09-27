#!/usr/bin/env bash
set -euo pipefail
ROOT="${REMATCH_ROOT:-/root/autodl-tmp/rematch_20260922}"
DEST="${NATIVE_MODEL_PATH:-/root/rematch_models/Qwen3-VL-8B-Instruct}"
source "$ROOT/scripts/network_env.sh"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
mkdir -p "$DEST"
printf '%s\tnative_download\tstart\n' "$(date -Is)" >> "$ROOT/logs/stage_status.tsv"
/root/miniconda3/bin/hf download Qwen/Qwen3-VL-8B-Instruct \
  --local-dir "$DEST" --max-workers 2 \
  --include '*.json' --include '*.safetensors' --include '*.txt' --include '*.jinja'
/root/miniconda3/bin/python - "$DEST" <<'PY'
import json, sys
from pathlib import Path
from safetensors import safe_open
root = Path(sys.argv[1])
index = json.loads((root / 'model.safetensors.index.json').read_text())
for name in set(index['weight_map'].values()):
    with safe_open(root / name, framework='pt', device='cpu') as f:
        assert len(f.keys()), name
(root / '.download_complete').touch()
print('NATIVE_WEIGHTS_READY', root, flush=True)
PY
printf '%s\tnative_download\tcomplete\n' "$(date -Is)" >> "$ROOT/logs/stage_status.tsv"
