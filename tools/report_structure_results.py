"""Merge matched R/S predictions, isolate structure/head gains, and render all flips."""
from __future__ import annotations
import argparse
import csv
from copy import deepcopy
import html
import json
from pathlib import Path
from PIL import Image,ImageDraw
from tools.train_grounding_structure import read,write
from tools.evaluate_grounding_structure import summarize,comparison


def merge(left,right):
    if [r['id'] for r in left]!=[r['id'] for r in right]: raise ValueError('R/S evaluation IDs/order differ')
    rows=[]
    for a,b in zip(left,right,strict=True):
        if a['target']!=b['target'] or a['A']!=b['A']: raise ValueError('R/S do not share the same target and real A prediction')
        row=deepcopy(a);row['S']=b['S'];row['S+G']=b['S+G'];rows.append(row)
    return rows


def paired(rows,left,right):
    copied=[{**r,'A':r[right]} for r in rows]
    return comparison(copied,left)


def report(args):
    rows=merge(read(args.r/'predictions.json'),read(args.s/'predictions.json'))
    inputs={r['id']:r for r in read(args.inputs)}
    arms=['A','R','S','S+G']; normal=[r for r in rows if r['condition']=='normal']
    result=summarize(rows,arms); pairs={}
    for left,right in [('R','A'),('S','R'),('S+G','S'),('S+G','R')]:
        pairs[left+' vs '+right]={}
        for group,subset in [('city412',[r for r in normal if r['task_category']=='old_city']),
                             ('diagnostic119',[r for r in normal if r['task_category']!='old_city'])]:
            pairs[left+' vs '+right][group]=paired(subset,left,right)
    increments={}
    for left,right in [('R','A'),('S','R'),('S+G','S')]:
        increments[left+' vs '+right]={}
        for modality in ('ir','depth'):
            if modality not in result[left]['modality_effect']:continue
            a,b=result[left]['modality_effect'][modality],result[right]['modality_effect'][modality]
            increments[left+' vs '+right][modality]={
                'normal_correct_delta':a['normal_correct']-b['normal_correct'],
                'blank_correct_delta':a['blank_correct']-b['blank_correct'],
                'net_dependence_delta':a['net']-b['net']}
    write(args.output/'summary.json',{'arms':result,'paired':pairs,'modality_increment':increments,
        'limits':['City412 is repeatedly used development data; 47 known reused sequences and other365 are not certified independent.',
                  'Blank-view intervention measures model dependence, not a human proof of modal necessity.',
                  'Increased normal-minus-blank dependence can also result from harming blank inputs; inspect both correct-count deltas.',
                  'The 16 RGB-sufficient IR controls are descriptive; they do not satisfy the old 32-control promotion gate.']})
    write(args.output/'predictions.json',rows)
    with (args.output/'flips.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['id','scene_id','category','comparison','change','left_iou','right_iou'])
        writer.writeheader()
        for row in normal:
            for left,right in [('R','A'),('S','R'),('S+G','S'),('S+G','A')]:
                if row[left]['hit']!=row[right]['hit']:
                    writer.writerow({'id':row['id'],'scene_id':row['scene_id'],'category':row['task_category'],
                        'comparison':left+' vs '+right,'change':'rescue' if row[left]['hit'] else 'harm',
                        'left_iou':row[left]['iou'],'right_iou':row[right]['iou']})
    cards=[]; colors={'GT':'#26a269','A':'#1c71d8','R':'#e5a50a','S':'#e01b24','S+G':'#9141ac'}
    for row in normal:
        if len({row[arm]['hit'] for arm in arms})==1:continue
        source=inputs[row['id']]
        with Image.open(source['image'][source['modalities'].index('rgb')]) as original:
            image=original.convert('RGB');image.thumbnail((900,600))
        draw=ImageDraw.Draw(image);w,h=image.size
        boxes={'GT':row['target'],**{arm:row[arm]['prediction'] for arm in arms}}
        for name,box in boxes.items():
            if box is None:continue
            points=(box[0]*w,box[1]*h,box[2]*w,box[3]*h)
            draw.rectangle(points,outline=colors[name],width=3)
            draw.text((points[0]+2,points[1]+2),name,fill=colors[name],stroke_width=1,stroke_fill='white')
        path=args.output/'atlas'/f'{len(cards):04d}.jpg';path.parent.mkdir(parents=True,exist_ok=True);image.save(path,quality=90)
        scores=' | '.join(f"{arm}: IoU={row[arm]['iou']:.3f}, {'命中' if row[arm]['hit'] else '失败'}" for arm in arms)
        cards.append(f'<article><h3>{html.escape(row["id"])}</h3><p>{html.escape(source["query"])}</p><p>{scores}</p><img src="atlas/{path.name}" loading="lazy"></article>')
    document='<!doctype html><meta charset="utf-8"><title>结构实验全部命中翻转</title><style>body{max-width:1100px;margin:auto;font-family:sans-serif}article{border-bottom:1px solid #aaa;padding:16px}img{max-width:100%}</style><h1>结构实验全部命中翻转</h1><p>绿色 GT，蓝色 A，黄色 R，红色 S，紫色 S+G。GT 仅用于此评估图册。</p>'+''.join(cards)
    (args.output/'atlas.html').write_text(document,encoding='utf-8')
    lines=['# 结构实验完整评估','', '| 模型 | City412 命中 | 诊断119 命中 | City救回/损害 |','|---|---:|---:|---:|']
    for arm in arms:
        item=result[arm];city=item['city412'];diag=item['diagnostic119']
        lines.append(f"| {arm} | {city['correct']}/412 | {diag['correct']}/119 | {city['rescue']}/{city['harm']} |")
    lines+=['','S 对 R、S+G 对 S 的配对翻转和按场景重采样区间见 summary.json；全部命中翻转见 flips.csv 与 atlas.html。',
        '', 'City412 是反复使用的开发集；已知重用47与其余365不等于独立泛化划分。辅助置空净收益不等于人工已证明模态必要性。16条真实IR阴性对照不足以触发旧32条自动升级门槛。']
    (args.output/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('r','s','inputs','output'):parser.add_argument('--'+name,type=Path,required=True)
    report(parser.parse_args())
