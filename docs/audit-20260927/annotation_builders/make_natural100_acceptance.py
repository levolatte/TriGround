from __future__ import annotations
import csv
import html
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from PIL import Image, ImageDraw

SOURCE = Path(r"F:\AIC\results\multimodal_data_20260925\merged_20260926_trial200_final\reviewed_evidence.jsonl")
OUT = Path(r"F:\AIC\results\multimodal_data_20260925\human_acceptance\natural100_seed2026")
FIELDS = ["case_id", "bundle_id", "category", "object_id", "modality", "query", "decision", "note"]
DECISIONS = [("correct_unique", "正确且唯一"), ("wrong", "错误"), ("multiple", "存在多个合理指代"), ("uncertain", "不确定")]

assert SOURCE.is_file(), SOURCE
assert not OUT.exists(), f"Refusing to overwrite existing output: {OUT}"
source_bytes = SOURCE.read_bytes()
rows = [json.loads(line) for line in source_bytes.decode("utf-8").splitlines() if line.strip()]
candidates = []
for bundle in rows:
    for query in bundle.get("queries", []):
        if query.get("modality_reviews", {}).get("rgb", {}).get("status") == "blind_passed":
            candidates.append((bundle, query))
candidates.sort(key=lambda pair: (pair[0]["id"], pair[1]["id"]))
assert len(candidates) == 121, f"Expected final RGB blind-passed pool of 121, found {len(candidates)}"
keys = [(b["id"], q["id"]) for b, q in candidates]
assert len(keys) == len(set(keys)), "Duplicate candidate bundle/query IDs"
rng = random.Random(2026)
sampled = rng.sample(list(enumerate(candidates, start=1)), 100)

OUT.mkdir(parents=True)
assets = OUT / "images"
assets.mkdir()
provenance = OUT / "provenance"
provenance.mkdir()
snapshot = provenance / "reviewed_evidence.jsonl"
snapshot.write_bytes(source_bytes)

# Exact source-image copies are shared when the same image is used by more than one query.
asset_for_source: dict[str, str] = {}
asset_index = 0
records = []
sections = []
for sample_order, (sorted_rank, pair) in enumerate(sampled, start=1):
    bundle, query = pair
    bundle_id, query_id = bundle["id"], query["id"]
    target_id = query["target_object_id"]
    obj = next((o for o in bundle.get("objects", []) if o.get("object_id") == target_id), None)
    assert obj is not None, f"Missing target object {bundle_id}::{query_id} -> {target_id}"
    rgb_box = obj.get("boxes", {}).get("rgb")
    assert isinstance(rgb_box, list) and len(rgb_box) == 4, f"Missing original RGB proposal for {bundle_id}::{query_id}"
    paths = bundle["images"]
    view_keys = [("rgb", "RGB原图"), ("infrared", "IR原图"), ("depth_visual", "Depth可视化预览")]
    asset_paths = {}
    for key, label in view_keys:
        source_path = Path(paths[key])
        assert source_path.is_file(), f"Missing source image: {source_path}"
        cache_key = str(source_path.resolve())
        if cache_key not in asset_for_source:
            asset_index += 1
            suffix = source_path.suffix or ".png"
            asset_name = f"source_{asset_index:04d}_{key}{suffix.lower()}"
            shutil.copy2(source_path, assets / asset_name)
            asset_for_source[cache_key] = f"images/{asset_name}"
        asset_paths[key] = asset_for_source[cache_key]

    # A comparison copy with only this query target's original RGB proposal box.
    rgb_source = Path(paths["rgb"])
    with Image.open(rgb_source) as image:
        rgb_img = image.convert("RGB")
    width, height = rgb_img.size
    x1, y1, x2, y2 = rgb_box
    rect = (round(x1 * width), round(y1 * height), round(x2 * width), round(y2 * height))
    draw = ImageDraw.Draw(rgb_img)
    line_width = max(3, round(min(width, height) / 140))
    draw.rectangle(rect, outline=(255, 40, 40), width=line_width)
    overlay_name = f"item_{sample_order:03d}_rgb_gt_proposal.png"
    rgb_img.save(assets / overlay_name, format="PNG")
    asset_paths["rgb_proposal"] = f"images/{overlay_name}"

    case_id = f"{bundle_id}::{query_id}"
    record = {
        "case_id": case_id,
        "bundle_id": bundle_id,
        "category": "natural_query",
        "object_id": target_id,
        "modality": "rgb_output",
        "query": query["query"],
        "decision": "",
        "note": "",
    }
    records.append(record)
    esc = html.escape
    img_panel = "".join(
        f'<figure><figcaption>{esc(label)}</figcaption><a href="{esc(asset_paths[key])}" target="_blank">'
        f'<img loading="lazy" src="{esc(asset_paths[key])}" alt="{esc(label)}"></a></figure>'
        for key, label in view_keys
    )
    overlay_panel = (f'<figure><figcaption>原RGB GT提议框（仅供核对）</figcaption>'
                     f'<a href="{esc(asset_paths["rgb_proposal"])}" target="_blank">'
                     f'<img loading="lazy" src="{esc(asset_paths["rgb_proposal"])}" alt="原RGB GT提议框"></a></figure>')
    option_html = '<option value="">未核对</option>' + "".join(
        f'<option value="{esc(value)}">{esc(label)}</option>' for value, label in DECISIONS
    )
    sections.append(
        f'<article class="review-item" data-case="{esc(case_id)}">'
        f'<h2>第 {sample_order:03d} 条 <small>{esc(bundle_id)} / {esc(query_id)}</small></h2>'
        f'<p class="query">{esc(query["query"])}</p>'
        f'<div class="views">{img_panel}{overlay_panel}</div>'
        f'<table><thead><tr><th>验收判定</th><th>备注</th></tr></thead><tbody><tr>'
        f'<td><select aria-label="验收判定">{option_html}</select></td>'
        f'<td><textarea aria-label="备注" rows="2" placeholder="可选"></textarea></td>'
        f'</tr></tbody></table></article>'
    )

# Static CSV with the same field order and a blank human decision for every item.
with (OUT / "decisions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(records)

manifest_items = []
for order, ((sorted_rank, pair), record) in enumerate(zip(sampled, records), start=1):
    bundle, query = pair
    obj = next(o for o in bundle["objects"] if o["object_id"] == query["target_object_id"])
    manifest_items.append({
        "sample_order": order,
        "sorted_candidate_rank_1based": sorted_rank,
        "case_id": record["case_id"],
        "bundle_id": record["bundle_id"],
        "query_id": query["id"],
        "target_object_id": query["target_object_id"],
        "query": query["query"],
        "rgb_eligibility_status": query["modality_reviews"]["rgb"]["status"],
        "original_rgb_gt_proposal_xyxy_normalized": obj["boxes"]["rgb"],
        "images": {k: bundle["images"][k] for k in ("rgb", "infrared", "depth_visual")},
    })
manifest = {
    "title": "Natural Query human acceptance sample",
    "seed": 2026,
    "sampling_method": "Python random.Random(2026).sample after stable sort by (bundle_id, query_id)",
    "candidate_filter": "modality_reviews.rgb.status == blind_passed",
    "candidate_count": len(candidates),
    "sample_size": len(records),
    "source_snapshot_original_path": str(SOURCE),
    "source_snapshot_preserved_copy": str(snapshot),
    "sample_items": manifest_items,
}
(OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

summary = {
    "source_snapshot_original_path": str(SOURCE),
    "source_snapshot_preserved_copy": str(snapshot),
    "source_bundle_count": len(rows),
    "rgb_blind_passed_candidate_count": len(candidates),
    "sample_seed": 2026,
    "review_items": len(records),
    "unique_case_count": len({r["case_id"] for r in records}),
    "natural_query_items": len(records),
    "object_box_review_items": 0,
    "relation_review_items": 0,
    "human_decisions_initially_blank": len(records),
    "human_decision_counts_initial": {key: 0 for key, _ in DECISIONS},
    "images_source_copies_unique": len(asset_for_source),
    "rgb_gt_proposal_overlays": len(records),
    "scope": "100 natural Query items only; this is not a claim that all 500 budget items are complete.",
}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

readme = """# Natural Query 人工随机验收包（seed 2026）

本包包含从最终 trial200 合并快照中按固定规则抽取的 100 条自然 Query。抽样候选仅为 RGB modality review 状态 `blind_passed` 的 Query；先按 `(bundle_id, query_id)` 稳定排序，再由 Python `random.Random(2026).sample` 抽取。抽样来源与逐条清单见 `manifest.json`，原始来源快照副本保存在 `provenance/reviewed_evidence.jsonl`。

请逐条判断 Query 是否正确且唯一指向其意图目标，可选择“正确且唯一”“错误”“存在多个合理指代”或“不确定”，并可填写备注。请仅依据图片与自然语言 Query 做人工判断。页面显示 RGB、IR、Depth 可视化预览；另有一张只覆盖原 RGB GT 目标框的对照图，该框是待核对提议，不等同于人工验收。Depth 可视化只用于目视参考，不据亮暗推断近远。

所有判定初始为空。可在页面中导出 CSV，或使用同目录的空白 `decisions.csv`。两种方式的字段顺序一致：`case_id,bundle_id,category,object_id,modality,query,decision,note`。请保存填写后的 CSV 作为人工结果。

范围限定：这是 trial200 最终合并数据中的 100 条自然 Query 人工验收样本，不代表 500 条预算全部完成；本包没有额外对象框或关系验收项。
"""
(OUT / "说明.md").write_text(readme, encoding="utf-8")

# HTML page: one natural-query human acceptance item per sample and no automatic verdict fields.
records_json = json.dumps(records, ensure_ascii=False).replace("<", "\\u003c")
page = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Natural Query 人工随机验收 · 100</title>
<style>
body{font:16px system-ui,-apple-system,"Segoe UI",sans-serif;margin:24px;background:#f4f6f8;color:#18222c}header,article{background:#fff;padding:20px;margin:16px 0;border-radius:12px}nav{position:sticky;top:0;z-index:2;background:#eaf0f6;padding:10px;border-radius:8px}button{padding:10px 14px;font-size:15px}.views{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}figure{margin:0;min-width:0}figcaption{font-weight:700;margin:4px 0 8px}img{width:100%;height:auto;border:1px solid #ccd3da;background:#eee}h2 small{font-size:13px;font-weight:400;color:#536170}.query{font-size:18px;line-height:1.45}table{width:100%;border-collapse:collapse;margin-top:14px}th,td{text-align:left;padding:8px;border-bottom:1px solid #dce1e6}select,textarea{font:inherit;padding:8px}select{min-width:230px}textarea{width:min(700px,95%)}.help{color:#536170}@media(max-width:1000px){.views{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:600px){.views{grid-template-columns:1fr}}
</style></head><body><header><h1>自然 Query 人工随机验收（100条）</h1><p>每条只评估该自然语言 Query 是否正确且唯一指向意图目标。RGB、IR 与 Depth 可视化预览为源图副本；第四张图只叠加该目标的原 RGB GT 提议框，供对照，不代表人工验收结果。请根据原图和 Query 作答，所有判定均从空白开始。</p><p class="help">选项：正确且唯一 / 错误 / 存在多个合理指代 / 不确定。Depth预览不用于从亮暗推断距离。此包只覆盖100条Query，不表示500条预算全部完成。</p></header><nav><button id="export">导出当前人工验收 CSV</button> <span>也可填写同目录 decisions.csv；未选择的判定保持空白。</span></nav>SECTIONS
<script>
const records=RECORDS;
const fields=['case_id','bundle_id','category','object_id','modality','query','decision','note'];
document.getElementById('export').onclick=()=>{
 const forms=new Map([...document.querySelectorAll('article[data-case]')].map(a=>[a.dataset.case,a]));
 const quote=v=>'"'+String(v??'').replaceAll('"','""')+'"';
 const rows=records.map(r=>{const a=forms.get(r.case_id);return {...r,decision:a.querySelector('select').value,note:a.querySelector('textarea').value}});
 const csv='\\uFEFF'+[fields,...rows.map(r=>fields.map(f=>r[f]))].map(r=>r.map(quote).join(',')).join('\\r\\n');
 const link=document.createElement('a');link.href=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));link.download='human_decisions.csv';link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000);
};
</script></body></html>'''.replace("SECTIONS", "".join(sections)).replace("RECORDS", records_json)
(OUT / "review.html").write_text(page, encoding="utf-8")
print(json.dumps({"output": str(OUT), "items": len(records), "eligible": len(candidates), "unique_source_images": len(asset_for_source)}, ensure_ascii=False))
