"""Self-contained offline flip atlas; never used by inference."""
from __future__ import annotations
import argparse
import html
import json
import random
from pathlib import Path
from PIL import Image, ImageDraw
from tools.prepare_aux_selection import read_rows, dump, dump_rows
from tools.report_rematch_experiment import load_manifest, load_run, box_iou
from tools.predict_aux_selection import annotate_candidates


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'fixed16']:
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    out = a.root/'results/aux_selection_refine_20260927'
    dest = out/'atlas'
    assets = dest/'assets'
    assets.mkdir(parents=True, exist_ok=True)
    gt = load_manifest(out/'manifests/scoring/city412_gt.json')
    manifest = {r['id']: r for r in read_rows(out/'manifests/city412.jsonl')}
    evidence = {r['id']: r for r in read_rows(out/'city412/evidence.jsonl')}
    predictions = out/'city412/trimodal'
    paths = {'C': out/'manifests/c_report_predictions.jsonl', 'selection': predictions/'selected_predictions.jsonl',
             'full': predictions/'final_predictions.jsonl'}
    runs = {n: load_run(n, path, set(gt), allow_partial=False) for n,path in paths.items()}
    finals = {r['id']: r for r in read_rows(paths['full'])}
    selected = {r['id']: r for r in read_rows(paths['selection'])}
    fixed = {r['id'] for r in read_rows(a.fixed16)}
    assert len(fixed) == 16 and fixed <= set(gt)
    transitions = {}
    for key in gt:
        truth = gt[key]['bbox']
        hits = {n: run['rows'][key]['primary'] is not None and box_iou(run['rows'][key]['primary'], truth)>=.5 for n,run in runs.items()}
        flips = []
        for left,right in [('C','selection'), ('C','full'), ('selection','full')]:
            if hits[left] != hits[right]: flips.append(f'{left}→{right}: '+('纠正' if hits[right] else '退化'))
        if flips: transitions[key] = flips
    ids = set(transitions)|fixed
    depth_changed = []
    diag = out/'city96/rgb_ir/selected_predictions.jsonl'
    if diag.exists():
        other = {r['id']: r for r in read_rows(diag)}
        depth_changed = [key for key,r in other.items() if r.get('selected_box') != selected[key].get('selected_box')]
        rng = random.Random(2026)
        depth_changed = rng.sample(sorted(depth_changed), min(8,len(depth_changed)))
        ids.update(depth_changed)
    dump_rows(dest/'depth_condition_changed_review.jsonl', [{'id': key, 'claim': 'input condition changed selection; not internal causal proof'} for key in depth_changed])
    cards, case_rows = [], []
    for key in gt:
        if key not in ids: continue
        row, ev, final = manifest[key], evidence[key], finals[key]
        figures = []
        rgb = Image.open(row['images']['rgb']).convert('RGB')
        truth = gt[key]['bbox']
        for name,run in runs.items():
            image = rgb.copy()
            d = ImageDraw.Draw(image)
            def mark(box, color):
                if box is not None: d.rectangle([box[0]*image.width,box[1]*image.height,box[2]*image.width,box[3]*image.height],outline=color,width=4)
            mark(truth,'#00bf75')
            pred = run['rows'][key]['primary']
            mark(pred,'#f02842')
            image.thumbnail((960,640))
            filename = key+'_'+name+'.jpg'
            image.save(assets/filename,quality=86)
            iou = box_iou(pred,truth) if pred is not None else 0
            figures.append(f'<figure><img loading="lazy" src="assets/{filename}"><figcaption>{name} · IoU {iou:.3f}（绿GT、红预测）</figcaption></figure>')
        for modality in ['rgb','ir','depth_visual']:
            image = Image.open(row['images'][modality]).convert('RGB')
            image = annotate_candidates(image,ev['candidates'])
            image.thumbnail((960,640))
            filename = key+'_'+modality+'.jpg'
            image.save(assets/filename,quality=86)
            figures.append(f'<figure><img loading="lazy" src="assets/{filename}"><figcaption>{modality} · 实际候选编号（投影框不保证跨模态边界）</figcaption></figure>')
        candidate_rows = []
        for c in ev['candidates']:
            depth = c.get('depth',{})
            mask_path = c.get('mask_path')
            link = ''
            if mask_path:
                mask = Image.open(mask_path).convert('L')
                mask.thumbnail((640,480))
                filename = key+f"_mask{c['id']}.png"
                mask.save(assets/filename)
                link = f'<a href="assets/{filename}">主体掩码</a>'
            candidate_rows.append(f"<tr><td>{c['id']} · {c['role']}</td><td>{html.escape(json.dumps(depth,ensure_ascii=False))}</td><td>{link}</td></tr>")
        details = {field: final.get(field) for field in ['selection_id','selection_kind','selection_raw','changed_from_c','refine_status','refine_raw','crop_transform']}
        info = {'id': key,'query': row['query'],'transitions':transitions.get(key,[]),'fixed16':key in fixed,'depth_condition_changed':key in depth_changed,**details}
        case_rows.append(info)
        cards.append(f'<section id="{key}"><h2>{key}</h2><p>{html.escape(row["query"])}</p><p>{html.escape("；".join(transitions.get(key,[])) or "固定观察例")}</p><div class="grid">'+''.join(figures)+'</div><details><summary>选择、细化和主体证据</summary><pre>'+html.escape(json.dumps(details,ensure_ascii=False,indent=2))+'</pre><table>'+''.join(candidate_rows)+'</table></details></section>')
    title='IR / Depth选择与局部细化：全部翻转图册'
    page=f'<!doctype html><html lang="zh"><meta charset="utf-8"><title>{title}</title><style>body{{font:16px/1.5 system-ui;max-width:1600px;margin:24px auto;color:#242424;background:#f5f5f4}}section{{background:white;margin:22px 0;padding:20px}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}figure{{margin:0}}img{{width:100%}}td{{border-bottom:1px solid #ddd;padding:8px;font-size:12px;overflow-wrap:anywhere}}pre{{white-space:pre-wrap}}@media(max-width:800px){{.grid{{grid-template-columns:1fr}}}}</style><h1>{title}</h1><p>全412离线评分后生成。GT仅在本页展示；未用于候选、分割、选择或裁剪。所有翻转与固定16例保留。自动框及深度统计不是人工真值，尚未目视确认的错误类型不预判。</p>'+''.join(cards)+'</html>'
    (dest/'index.html').write_text(page,encoding='utf-8')
    dump_rows(dest/'case_rows.jsonl',case_rows)
    dump(dest/'summary.json',{'cases':len(ids),'all_flip_cases':len(transitions),'fixed16':len(fixed),'depth_condition_changed_review':len(depth_changed),'missing_images':0})
    print(json.dumps({'atlas_cases':len(ids),'flip_cases':len(transitions)}))


if __name__ == '__main__': main()
