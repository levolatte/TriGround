#!/usr/bin/env bash
set -euo pipefail
ROOT="${REMATCH_ROOT:-/root/autodl-tmp/rematch_20260922}"
DEST="${NATIVE_MODEL_PATH:-/root/rematch_models/Qwen3-VL-8B-Instruct}"
if [[ "${DOWNLOAD_ROUTE:-platform}" == backup ]]; then
  source "$ROOT/scripts/start_backup_proxy.sh"
else
  source "$ROOT/scripts/network_env.sh"
fi
export DEST
printf '%s\tnative_download\tsegmented_%s_start\n' "$(date -Is)" "${DOWNLOAD_ROUTE:-platform}" >> "$ROOT/logs/stage_status.tsv"
/root/miniconda3/bin/python - <<'PY'
import json, os, urllib.request
from pathlib import Path
root = Path(os.environ['DEST'])
root.mkdir(parents=True, exist_ok=True)
meta = root / 'download_files.json'
if not meta.exists():
    with urllib.request.urlopen('https://huggingface.co/api/models/Qwen/Qwen3-VL-8B-Instruct?blobs=true', timeout=60) as response:
        info = json.load(response)
    selected = [entry for entry in info['siblings'] if '/' not in entry['rfilename'] and
                Path(entry['rfilename']).suffix in {'.json', '.safetensors', '.txt', '.jinja'}]
    meta.write_text(json.dumps(selected, indent=2))
lines=[]
for entry in json.loads(meta.read_text()):
    path = root / entry['rfilename']
    if path.exists() and path.stat().st_size == entry['size'] and not Path(str(path)+'.aria2').exists():
        continue
    lines.extend(['https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/resolve/main/'+entry['rfilename'],
                  ' dir='+str(root), ' out='+path.name])
(root/'download_urls.txt').write_text('\n'.join(lines)+'\n')
print('PENDING_FILES',len(lines)//3,flush=True)
PY
aria2c --input-file="$DEST/download_urls.txt" --continue=true --auto-file-renaming=false \
  --file-allocation=none --max-concurrent-downloads=2 --max-connection-per-server=4 --split=4 \
  --min-split-size=4M --connect-timeout=20 --timeout=30 --max-tries=3 --retry-wait=5 \
  --summary-interval=30 --enable-color=false --console-log-level=warn
/root/miniconda3/bin/python - <<'PY'
import json, os
from pathlib import Path
from safetensors import safe_open
root=Path(os.environ['DEST'])
for entry in json.loads((root/'download_files.json').read_text()):
    p=root/entry['rfilename']
    assert p.stat().st_size == entry['size'] and not Path(str(p)+'.aria2').exists(), p.name
index=json.loads((root/'model.safetensors.index.json').read_text())
for name in set(index['weight_map'].values()):
    with safe_open(root/name,framework='pt',device='cpu') as f:
        assert len(f.keys()),name
(root/'.download_complete').touch()
print('NATIVE_WEIGHTS_READY',flush=True)
PY
printf '%s\tnative_download\tcomplete\n' "$(date -Is)" >> "$ROOT/logs/stage_status.tsv"
