import contextlib, io, json, re, runpy, statistics
from collections import Counter
from pathlib import Path
from zipfile import ZipFile
from PIL import Image
import numpy as np

ROOT=Path('F:/AIC')
OUT=ROOT/'code/docs/research/2026-10-28-rgbdt-annotation/evidence'
with contextlib.redirect_stdout(io.StringIO()):
    base=runpy.run_path(str(OUT/'compare_styles.py'))
base['patterns']['ordinal_only']=r'\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|\d+(?:st|nd|rd|th))\b'
base['patterns']['ordinal_extreme']=r'\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|\d+(?:st|nd|rd|th)|leftmost|rightmost|nearest|farthest|furthest|closest)\b'
source=ROOT/'data/quant_grounding_source'
with ZipFile(source/'original_city_labels.zip') as z:
    train=json.loads(z.read('qwen_generation_train_100.json'))
    val=json.loads(z.read('qwen_generation_val.json'))
    readme=z.read('qwen_generation_DATASET_README.md').decode('utf-8-sig')
    reports={n:json.loads(z.read(n)) for n in ['qwen_generation_split_report.json','qwen_generation_overlap_report.json']}
def rows(d): return [dict(id=k,**v) for k,v in d.items()]
def frame(r): return Path(r['visible'].replace('\\','/')).stem
exclude=set(json.loads((source/'recovered_city_test_exclusions.json').read_text())['excluded_frame_keys'])
devframes={frame(r) for r in val.values()}
clean=[r for r in rows(train) if frame(r) not in exclude|devframes]
corrected=json.loads((source/'city_val_corrected.json').read_text())
corrections=json.loads((source/'corrections.json').read_text())
deferred={r['id'] for r in corrections['cases'] if r['status'] in ['deferred','partial']}
dev=[r for r in rows(corrected) if r['id'] not in deferred and frame(r) not in exclude]
original=[dict(r,query=r.get('original_query',r['query'])) for r in rows(train)]
sets={'fine98':base['fine'],'teammate919':base['teammate'],'city_train3707':rows(train),'city_clean3178':clean,'city_original_expression':original,'city_val412_original':rows(val),'city_dev378_corrected':dev,'semifinal5691':base['semifinal']}
result={'method':{'patterns':base['patterns'],'note':'Numeric ordinals such as 4th added for all datasets; keyword rates are proxies, not semantic ground truth. City clean uses previous exclusions, not a completed latest semifinal overlap audit.'},'datasets':{k:base['stats'](v) for k,v in sets.items()},'city_groups':{k:len({frame(r) for r in v}) for k,v in sets.items() if k.startswith('city')},'city_decisions':dict(Counter(r.get('review_decision') for r in train.values())),'city_query_changes':sum(r.get('original_query',r['query'])!=r['query'] for r in train.values()),'city_reports':reports}
semi_root=ROOT/'data/半决赛数据集-基于大模型的多模态视觉理解与推理'
groups={r['visible']:r for r in base['semifinal']}
counts=Counter(r['visible'] for r in base['semifinal'])
result['queries_per_image']={k:{'groups':len(c),'median':statistics.median(c.values()),'mean':sum(c.values())/len(c),'max':max(c.values())} for k,c in [('city_train',Counter(frame(r) for r in train.values())),('semifinal',counts)]}
sample=[groups[k] for k in sorted(groups)][::20]
media=[]
for r in sample:
    item={'image':r['visible']}
    for m in ['visible','infrared','depth']:
        with Image.open(semi_root/r[m]) as im:
            a=np.array(im); item[m]={'mode':im.mode,'size':im.size,'dtype':str(a.dtype)}
            if m=='depth': item[m].update(zero_fraction=float((a==0).mean()),min=int(a.min()),max=int(a.max()))
            if m=='infrared': item[m]['channels_equal']=bool(np.array_equal(a[:,:,0],a[:,:,1]) and np.array_equal(a[:,:,1],a[:,:,2])) if a.ndim==3 else None
    media.append(item)
result['semifinal_systematic_media_sample']=media
raw_media=[r for r in media if r['depth']['dtype']=='uint16' and r['depth']['mode'].startswith('I')]
result['depth_sample_summary']={'n':len(raw_media),'scope':'only sampled uint16 single-channel depth; zero fraction over whole image, not target ROI','zero_fraction_median':statistics.median(r['depth']['zero_fraction'] for r in raw_media),'zero_fraction_over_half':sum(r['depth']['zero_fraction']>.5 for r in raw_media),'zero_fraction_range':[min(r['depth']['zero_fraction'] for r in raw_media),max(r['depth']['zero_fraction'] for r in raw_media)]}
formats=Counter()
for r in groups.values():
    key=[]
    for m in ['visible','infrared','depth']:
        with Image.open(semi_root/r[m]) as im: key.append(f'{m}:{im.size}/{im.mode}/{im.format}')
    formats['; '.join(key)]+=1
result['semifinal_all_image_formats']=dict(formats)
(OUT/'city_semifinal_comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(OUT/'city_source_readme.md').write_text(readme,encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ['method','semifinal_systematic_media_sample','city_reports','datasets']},ensure_ascii=False,indent=2))
for k,v in result['datasets'].items(): print(k,v['n'],v['words']['median'],v['lexical'])
