#!/usr/bin/env bash
set -euo pipefail
ROOT="${REMATCH_ROOT:-/root/autodl-tmp/rematch_20260922}"
OUT="$ROOT/data/external/RGBT-GroundBench"
while [[ ! -f "$OUT/data_m3fd.tar.extracted" ]]; do
  if ! ps -p "$(cat "$OUT/download.pid")" -o stat= | grep -q '^[[:space:]]*[^Z[:space:]]'; then
    echo 'Download exited before extraction completed' >&2; exit 2
  fi
  sleep 15
done
cd "$ROOT/code"
/root/miniconda3/bin/python - "$OUT" <<'PY'
import json,random,sys
from collections import defaultdict,Counter
from pathlib import Path
from tools.prepare_rgbt_groundbench import convert_split
root=Path(sys.argv[1]); out=root/'converted';out.mkdir(exist_ok=True)
report=convert_split(root/'raw','train',out/'train_full.jsonl',verify_images=True)
rows=[json.loads(x) for x in (out/'train_full.jsonl').read_text().splitlines()]
rng=random.Random(2026); selected=[]
for source in sorted({r['source'] for r in rows}):
    groups=defaultdict(list)
    for row in rows:
        if row['source']==source: groups[row['scene_id']].append(row)
    for values in groups.values(): rng.shuffle(values)
    order=list(groups);rng.shuffle(order);picked=[]
    while len(picked)<2000:
        progress=False
        for key in order:
            if groups[key]: picked.append(groups[key].pop());progress=True
            if len(picked)==2000: break
        if not progress: raise ValueError(f'Fewer than 2000 available rows in {source}')
    selected.extend(picked)
assert len(selected)==len({r['id'] for r in selected})==6000
rng.shuffle(selected)
(out/'train6000.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
report.update(selected=6000,seed=2026,selected_sources=dict(Counter(r['source'] for r in selected)),
              selected_image_groups=len({r['scene_id'] for r in selected}))
(out/'selection_report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
PY
printf '%s\trgbt_train6000\tprepared\n' "$(date -Is)" >> "$OUT/download_status.tsv"
