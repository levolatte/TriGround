"""Post-score CPU audit. Diagnostic counterfactual coordinates are never predictions."""
import json
import math
from collections import Counter
from pathlib import Path
from tools.report_rematch_experiment import box_iou

OUT = Path('F:/AIC/results/aux_selection_refine_20260927')

def rows(p):
    return {r['id']:r for r in map(json.loads,p.read_text(encoding='utf-8').splitlines())}

def main():
    gt=json.loads((OUT/'manifests/scoring/city412_gt.json').read_text())
    c=rows(OUT/'manifests/c_report_predictions.jsonl')
    s=rows(OUT/'city412/trimodal/selected_predictions.jsonl')
    f=rows(OUT/'city412/trimodal/final_predictions.jsonl')
    ev=rows(OUT/'city412/evidence.jsonl')
    assert set(gt)==set(c)==set(s)==set(f)==set(ev) and len(gt)==412
    iou=lambda b,t: box_iou(b,t) if b is not None else 0
    records=[]; mappings=[]; counts=Counter(); refinements=[]; depth=[]; crop_policy=[]
    for key,g in gt.items():
        a,b,z=c[key]['prediction'],s[key]['prediction'],f[key]['prediction']
        ia,ib,iz=[iou(v,g['bbox']) for v in (a,b,z)]
        ref=f[key]; evidence=ev[key]
        if ref.get('crop_transform'):
            tr=ref['crop_transform']; pixel=ref['crop_pixel_box']; selected=ref['selected_box']
            width=round(pixel[2]/tr[2]); height=round(pixel[3]/tr[3])
            cw=max(2*(selected[2]-selected[0]),.12); ch=max(2*(selected[3]-selected[1]),.12)
            cx=(selected[0]+selected[2])/2; cy=(selected[1]+selected[3])/2
            expected=[max(0,math.floor((cx-cw/2)*width)),max(0,math.floor((cy-ch/2)*height)),
                      min(width,math.ceil((cx+cw/2)*width)),min(height,math.ceil((cy+ch/2)*height))]
            if pixel!=expected:
                crop_policy.append({'id':key,'actual_crop':pixel,'centered_clipped_crop':expected,
                                    'crosses_edge':cx-cw/2<0 or cy-ch/2<0 or cx+cw/2>1 or cy+ch/2>1,
                                    'refine_status':ref['refine_status'],'selected_iou':ib,'full_iou':iz})
        r={'id':key,'query':g['query'],'c_iou':ia,'selected_iou':ib,'full_iou':iz,
           'selection_full_box_iou':iou(b,z) if z is not None else 0,
           'c_selection_box_iou':iou(a,b) if b is not None else 0,
           'selection_id':s[key]['selection_id'],'refine_status':ref['refine_status']}
        if ia>=.5 and ib<.5:
            counts['selection_regressions']+=1
            counts['selection_regressions_c_selected_overlap_ge05' if r['c_selection_box_iou']>=.5 else 'selection_regressions_c_selected_overlap_lt05']+=1
        if ref['refine_status']=='refined':
            raw=[v/1000 for v in json.loads(ref['refine_raw'])['bbox_2d']]
            l,t,rr,bb=ref['crop_transform']; x1,y1,x2,y2=raw
            expected=[l+x1*(rr-l),t+y1*(bb-t),l+x2*(rr-l),t+y2*(bb-t)]
            err=max(abs(v-w) for v,w in zip(expected,z))
            mappings.append(err)
            assert err<1e-12, (key,expected,z)
            raw_global_iou=iou(raw,g['bbox'])
            r.update(raw_bbox=raw,selected_bbox=b,final_bbox=z,gt_bbox=g['bbox'],crop_transform=ref['crop_transform'],
                     raw_interpreted_as_global_iou_DIAGNOSTIC_ONLY=raw_global_iou,
                     raw_global_vs_selected_iou=iou(raw,b),mapping_max_error=err)
            refinements.append(r)
            if ib>=.5 and iz<.5:
                counts['refinement_regressions']+=1
                counts['refinement_regressions_raw_global_would_hit' if raw_global_iou>=.5 else 'refinement_regressions_raw_global_would_not_hit']+=1
            if ib<.5 and iz>=.5: counts['refinement_corrections']+=1
        targets=[q for q in evidence['candidates'] if q['role']=='target']
        near_duplicate=sum(iou(q['bbox'],w['bbox'])>=.5 for idx,q in enumerate(targets) for w in targets[idx+1:])
        r['target_pairs_iou_ge05']=near_duplicate
        if near_duplicate: counts['queries_with_overlapping_target_boxes_ge05']+=1
        relation=evidence['query_info']['relation_type']
        prompt=s[key].get('selection_prompt') or ''
        if relation in ['camera_near','camera_far']:
            depth.append({'id':key,'relation':relation,'reliable_candidates':sum(q.get('depth',{}).get('status')=='reliable' for q in evidence['candidates']),
                'supported_pairs':sum(q.get('status')=='supported' for q in evidence['depth_pairs']),
                'numeric_table_in_actual_prompt':'full[q1=' in prompt,
                'supported_order_in_actual_prompt':' is nearer than candidate ' in prompt,
                'c_hit':ia>=.5,'selected_hit':ib>=.5,'full_hit':iz>=.5})
        records.append(r)
    before=rows(OUT/'city412/trimodal_before_numeric_id_fix/final_predictions.jsonl')
    repair=json.loads((OUT/'numeric_id_interface_repair.json').read_text())
    changed={k for k in f if f[k]!=before[k]}
    assert changed==set(repair['ids']) and len(changed)==3
    report=json.loads((OUT/'report412/summary.json').read_text())
    newly=report['candidate_diagnostics']['newly_covered_c_error_ids']
    rgb_covers_new=[k for k in newly if any(src['modality']=='rgb' and iou(src['bbox'],gt[k]['bbox'])>=.5
                    for q in ev[k]['candidates'] if q['role']=='target' for src in q['sources'])]
    summary={'queries':412,'refined_numeric_boxes':len(refinements),'mapping_max_error':max(mappings),
       'counts':dict(counts),'camera_depth_query_count':len(depth),
       'camera_queries_with_numeric_table':sum(r['numeric_table_in_actual_prompt'] for r in depth),
       'camera_queries_with_supported_pair':sum(r['supported_order_in_actual_prompt'] for r in depth),
       'camera_query_scores':{n:sum(r[n] for r in depth) for n in ['c_hit','selected_hit','full_hit']},
       'generation_cap_hits':sum(bool(r.get('generation_cap_hit')) for r in f.values()),
       'generation_seconds_from_final_records':sum(r.get('latency_seconds',0) for r in f.values()),
       'newly_covered_c_errors':len(newly),'also_covered_by_retained_rgb_detection':len(rgb_covers_new),
       'numeric_id_repair_changed_ids':sorted(changed),'unchanged_ids':412-len(changed),
       'crop_policy_difference_count':len(crop_policy),
       'crop_policy_boundary_difference_count':sum(r['crosses_edge'] for r in crop_policy),
       'crop_policy_differing_refinement_regressions':sum(r['selected_iou']>=.5 and r['full_iou']<.5 for r in crop_policy),
       'diagnostic_limit':'Raw output reinterpreted as global uses GT after scoring, is not a deployable rule or a new method score; overlap does not prove same-instance identity.'}
    dest=OUT/'output_audit'; dest.mkdir(exist_ok=True)
    (dest/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (dest/'border_crop_policy_audit.json').write_text(json.dumps({'interpretation':'Measured v1 shifts border windows; approved centered clipping differs. No GT used to choose rectangles. Corrected CPU code not GPU-evaluated.','affected':crop_policy},ensure_ascii=False,indent=2),encoding='utf-8')
    for name,data in [('per_query',records),('refinements',refinements),('depth_queries',depth)]:
        (dest/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
