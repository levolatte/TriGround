"""Make an offline human-review packet; never infer human acceptance from agents."""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw

from tools.prepare_multimodal_evidence import MODALITIES, read_rows, validate_bundle, view_paths


def render(bundles: list[dict], output: Path) -> dict:
    # Review decisions must not be overwritten by subsequent production batches.
    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir()
    sections, decisions = [], []
    totals = {"bundles": len(bundles), "objects": 0, "queries": 0,
              "relations": 0, "human_accepted": 0}
    esc = lambda value: html.escape(str(value))
    for index, bundle in enumerate(bundles):
        first_decision = len(decisions)
        validate_bundle(bundle)
        views = view_paths(bundle)
        figures = []
        totals["objects"] += len(bundle["objects"])
        for modality in MODALITIES:
            if modality not in views:
                continue
            with Image.open(views[modality]) as source:
                original = source.convert("RGB")
                original.thumbnail((720, 480))
            original_name = f"images/{index:04d}_{modality}_original.jpg"
            original.save(output / original_name, quality=90)
            annotated = original.copy()
            draw = ImageDraw.Draw(annotated)
            captions = []
            for oi, obj in enumerate(bundle["objects"]):
                box = obj["boxes"].get(modality)
                captions.append(f"{esc(obj['object_id'])}: {esc(obj.get('evidence', {}).get(modality, '未记录'))}")
                if box is None:
                    captions.append(f"{esc(obj['object_id'])}: 无可确认区域")
                    continue
                x1, y1, x2, y2 = [v * (annotated.width if n % 2 == 0 else annotated.height) for n, v in enumerate(box)]
                color = ("#ff5050", "#54dd83", "#65a9ff", "#ffc44a")[oi]
                draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
                draw.text((x1, max(0, y1 - 13)), str(obj["object_id"]), fill=color)
                decisions.append({"case_id": f"{bundle['id']}::{obj['object_id']}::{modality}",
                                  "bundle_id": bundle["id"], "category": "correspondence",
                                  "object_id": obj["object_id"], "modality": modality,
                                  "query": "", "decision": "", "note": ""})
            annotated_name = f"images/{index:04d}_{modality}_proposed.jpg"
            annotated.save(output / annotated_name, quality=90)
            figures.append(f'<figure><figcaption>{esc(modality)}：原图 / 提议框</figcaption>'
                           f'<a href="{original_name}"><img src="{original_name}"></a>'
                           f'<a href="{annotated_name}"><img src="{annotated_name}"></a>'
                           f'<p>{"<br>".join(captions)}</p></figure>')
        query_html = []
        for query in bundle["queries"]:
            totals["queries"] += 1
            query_html.append(f"<li><b>{esc(query['id'])}</b>：{esc(query['query'])} → {esc(query['target_object_id'])}"
                              f"；机器复核：{esc(query.get('review_status', 'pending'))}</li>")
            decisions.append({"case_id": f"{bundle['id']}::{query['id']}", "bundle_id": bundle["id"],
                              "category": "natural_query", "object_id": query["target_object_id"],
                              "modality": "rgb_output", "query": query["query"], "decision": "", "note": ""})
        relation_html = []
        for relation in bundle.get("relations", []):
            totals["relations"] += 1
            relation_html.append(f"<li>{esc(relation['query'])}；提议答案 {esc(relation['answer_object_id'])}；"
                                 f"依据 {esc(relation.get('evidence', '未记录'))}</li>")
            decisions.append({"case_id": f"{bundle['id']}::{relation['id']}", "bundle_id": bundle["id"],
                              "category": "relation", "object_id": relation["answer_object_id"],
                              "modality": "candidate_id", "query": relation["query"], "decision": "", "note": ""})
        form_rows = []
        for row in decisions[first_decision:]:
            form_rows.append(f'<tr data-case="{esc(row["case_id"])}"><td>{esc(row["category"])} / '
                             f'{esc(row["object_id"])} / {esc(row["modality"])}</td>'
                             f'<td>{esc(row["query"])}</td><td><select><option value="">未核对</option>'
                             '<option value="accept">接受</option><option value="reject">拒绝</option>'
                             '<option value="uncertain">不确定</option></select></td>'
                             '<td><input placeholder="原因或修正建议"></td></tr>')
        sections.append(f'<section><h2>{esc(bundle["id"])} · {esc(bundle["split"])} · 场景 {esc(bundle["scene_id"])}</h2>'
                        f'<div class="views">{"".join(figures)}</div><ul>{"".join(query_html)}</ul>'
                        f'<ul>{"".join(relation_html)}</ul><table>{"".join(form_rows)}</table></section>')
    page = '<!doctype html><meta charset="utf-8"><title>跨模态对象证据人工复核</title><style>' \
           'body{font:16px system-ui;margin:24px;background:#f5f6f8;color:#18222c}' \
           'section{background:white;padding:20px;margin:20px 0;border-radius:12px}.views{display:flex;gap:12px}' \
           'figure{margin:0;flex:1;min-width:0}img{width:100%}figcaption{font-weight:700}p{font-size:14px}' \
           'td{padding:8px;border-bottom:1px solid #ddd}input{min-width:200px}select,button{padding:9px}' \
           'nav{position:sticky;top:0;background:#edf2f7;padding:10px;z-index:1}' \
           '@media(max-width:900px){.views{display:block}}</style>' \
           '<h1>对象证据：人工复核包</h1><p>框是待确认提议，非已验收标签。请对照原图核对对象范围、IR对应、Depth区域及Query唯一性。' \
           '未知深度编码不能仅凭亮暗判断远近。机器通过不代表人工通过。</p>' \
           '<p>可直接在页面核对并导出人工CSV，也可填写同目录 decisions.csv。' \
           '一条记录是一个检查项；不把多个框检查项冒充多条独立随机验收Query。抽样验收与争议处理分开统计。</p>'
    payload = json.dumps(decisions, ensure_ascii=False).replace("<", "\\u003c")
    script = '''<script>
const records=RECORDS;
const fields=['case_id','bundle_id','category','object_id','modality','query','decision','note'];
document.getElementById('export').onclick=()=>{
  const forms=new Map([...document.querySelectorAll('tr[data-case]')].map(tr=>[tr.dataset.case,tr]));
  const quote=v=>'"'+String(v??'').replaceAll('"','""')+'"';
  const rows=records.map(r=>{const tr=forms.get(r.case_id);return {...r,decision:tr.querySelector('select').value,note:tr.querySelector('input').value}});
  const csv='\\uFEFF'+[fields,...rows.map(r=>fields.map(f=>r[f]))].map(r=>r.map(quote).join(',')).join('\\r\\n');
  const link=document.createElement('a');link.href=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));
  link.download='human_decisions.csv';link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000);
};
</script>'''.replace("RECORDS", payload)
    nav = '<nav><button id="export">导出当前人工核对CSV</button> 关闭页面前请导出，未填写项保持空白。</nav>'
    (output / "review.html").write_text(page + nav + "".join(sections) + script, encoding="utf-8")
    with (output / "decisions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "bundle_id", "category", "object_id", "modality", "query", "decision", "note"])
        writer.writeheader()
        writer.writerows(decisions)
    totals["review_items"] = len(decisions)
    (output / "summary.json").write_text(json.dumps(totals, ensure_ascii=False, indent=2), encoding="utf-8")
    return totals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(read_rows(args.evidence), args.output_dir), ensure_ascii=False))
