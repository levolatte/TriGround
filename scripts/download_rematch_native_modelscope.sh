#!/usr/bin/env bash
set -euo pipefail
ROOT="${REMATCH_ROOT:-/root/autodl-tmp/rematch_20260922}"
DEST=/root/rematch_models/Qwen3-VL-8B-Instruct
# The official domestic endpoint is reachable directly; other jobs keep their proxies.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export DEST
printf '%s\tnative_download\tmodelscope_official_start\n' "$(date -Is)" >> "$ROOT/logs/stage_status.tsv"
/root/miniconda3/bin/python - <<'PY'
import json, os, shutil, urllib.request
from pathlib import Path
root=Path(os.environ['DEST'])
old=root.with_name(root.name+'.hf_partial')
assert root.is_dir() and not old.exists()
assert shutil.disk_usage(root).free > 20*1024**3, 'Need room for a complete independent 8B copy'
expected=json.loads((root/'download_files.json').read_text())
hf_config=json.loads((root/'config.json').read_text())
base='https://modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct/resolve/master/'
with urllib.request.urlopen(base+'config.json',timeout=30) as f:
    assert json.load(f)==hf_config, 'Official model configs differ'
root.rename(old)
root.mkdir()
(root/'download_files.json').write_text(json.dumps(expected,indent=2))
(root/'download_origin.json').write_text(json.dumps({'repo':'Qwen/Qwen3-VL-8B-Instruct','source':base,'previous_partial':str(old)},indent=2))
lines=[]
for entry in expected:
    lines.extend([base+entry['rfilename'],' dir='+str(root),' out='+entry['rfilename']])
(root/'download_urls.txt').write_text('\n'.join(lines)+'\n')
PY
aria2c --input-file="$DEST/download_urls.txt" --continue=true --auto-file-renaming=false \
  --file-allocation=none --max-concurrent-downloads=3 --max-connection-per-server=4 --split=4 \
  --min-split-size=4M --connect-timeout=20 --timeout=30 --max-tries=3 --retry-wait=5 \
  --summary-interval=30 --enable-color=false --console-log-level=warn
/root/miniconda3/bin/python - <<'PY'
import json,os
from pathlib import Path
from safetensors import safe_open
root=Path(os.environ['DEST'])
for entry in json.loads((root/'download_files.json').read_text()):
    p=root/entry['rfilename']
    assert p.is_file() and p.stat().st_size==entry['size'] and not Path(str(p)+'.aria2').exists(),p.name
index=json.loads((root/'model.safetensors.index.json').read_text())
for name in set(index['weight_map'].values()):
    with safe_open(root/name,framework='pt',device='cpu') as f:
        assert len(f.keys()),name
(root/'.download_complete').touch()
print('NATIVE_MODELSCOPE_WEIGHTS_READY',flush=True)
PY
printf '%s\tnative_download\tcomplete_modelscope\n' "$(date -Is)" >> "$ROOT/logs/stage_status.tsv"
