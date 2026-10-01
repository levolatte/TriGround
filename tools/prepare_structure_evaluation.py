"""Create full City412 and diagnostic interventions without changing frozen train data."""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
from PIL import Image

from tools.prepare_triground_structure import read, write, city_row, cloud_rows
from tools.report_gu_diagnosis import SHARED_PATTERN


def prepare(data, output, city_manifest, city_gt):
    output=Path(output)
    if (output/'inputs.cloud.json').exists(): raise FileExistsError('evaluation release already exists')
    raw=read(city_gt)
    city=[city_row(row,raw[row['id']],split='val') for row in read(city_manifest)]
    for row in city:
        row['id']=row['source_task_id']
        row['city_subset']='shared47' if SHARED_PATTERN.search(raw[row['id']]['visible']) else 'other365'
    if Counter(r['city_subset'] for r in city) != {'shared47':47,'other365':365}:
        raise ValueError('City evaluation must preserve the complete 47/365 partition')
    normal=read(Path(data)/'diagnostic_normal.json')
    rows=city+normal
    for row in rows:
        row['condition']='normal'
        row['evaluation_mother_id']=row['id']
        row['supervision']={}
    for source in normal:
        for modality in ('ir','depth'):
            if modality not in source['modalities'] or modality in source.get('missing_modalities_actual',[]):
                continue
            row=deepcopy(source)
            index=row['modalities'].index(modality)
            with Image.open(source['image'][index]) as original:
                path=output/'assets'/f"{source['id'].replace(':','_')}_{modality}.png"
                path.parent.mkdir(parents=True,exist_ok=True)
                Image.new('RGB',original.size).save(path)
            row['image'][index]=str(path)
            row['missing_modalities_actual']=list(row.get('missing_modalities_actual',[]))+[modality]
            row['condition']='without_'+modality
            row['id']=source['id']+':without_'+modality
            rows.append(row)
    if len(city)!=412 or len(normal)!=119: raise ValueError('evaluation denominator changed')
    write(output/'inputs.json',rows)
    write(output/'inputs.cloud.json',cloud_rows(rows))
    summary={'normal_city':len(city),'normal_diagnostic':len(normal),'conditions':dict(Counter(r['condition'] for r in rows)),
             'rows':len(rows),'blank_policy':'same query/RGB/GT; only actual existing auxiliary slots blanked; raw depth disabled synchronously'}
    write(output/'release.json',summary)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('data','output','city-manifest','city-gt'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    print(prepare(args.data,args.output,args.city_manifest,args.city_gt))
