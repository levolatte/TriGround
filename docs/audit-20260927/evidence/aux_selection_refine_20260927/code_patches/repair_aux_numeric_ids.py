"""Repair the documented numeric-ID interface bug, without reading GT."""
import json
import shutil
import time
from pathlib import Path
from tools.prepare_aux_selection import read_rows, dump_rows, dump
from tools.predict_aux_selection import (_load_qwen, _make_selection_row, _refine_row,
                                         parse_selection, needs_refine)

root=Path('/root/autodl-tmp/rematch_20260922')
out=root/'results/aux_selection_refine_20260927'
folder=out/'city412/trimodal'
old=out/'city412/trimodal_before_numeric_id_fix'
deadline=json.loads((out/'execution_window.json').read_text())['deadline_epoch']
assert time.time()<deadline
assert not (out/'report412/summary.json').exists(), 'decide interface correction before seeing scores'
assert not old.exists()
manifest={r['id']:{**r,'_id':r['id']} for r in read_rows(out/'manifests/city412.jsonl')}
evidence={r['id']:r for r in read_rows(out/'city412/evidence.jsonl')}
query={r['id']:r for r in read_rows(out/'city412/query/query_info.jsonl')}
selected=read_rows(folder/'selected_predictions.jsonl')
final=read_rows(folder/'final_predictions.jsonl')
assert len(selected)==len(final)==len(manifest)==412
assert (folder/'summary.json').exists(), 'wait until original model process has finished'
affected=[]
for r in selected:
    raw=r.get('selection_raw')
    if r['selection_kind']!='invalid' or not raw: continue
    try: value=json.loads(raw)
    except json.JSONDecodeError: continue
    if isinstance(value,int) and not isinstance(value,bool) and value in r['target_candidate_ids']:
        affected.append(r['id'])
assert affected, 'no valid scalar IDs to repair'
folder.rename(old)
shutil.copytree(old,folder)
new_selected={r['id']:r for r in selected}
new_final={r['id']:r for r in final}
started=time.time()
model=processor=torch=None
for r in selected:
    key=r['id']
    if key not in affected: continue
    ev=evidence[key]
    output={'raw_text':r['selection_raw'],'image_grid_thw':r['selection_image_grid_thw'],
            'latency_seconds':r['selection_latency_seconds'],'generated_tokens':r['selection_generated_tokens'],
            'input_tokens':r['selection_input_tokens'],'generation_cap_hit':r['selection_generation_cap_hit']}
    s=_make_selection_row(manifest[key],ev,query[key],'trimodal',output,r['selection_prompt'],r['elapsed_seconds'],ev['candidates'])
    assert s['parsed'] and parse_selection(output['raw_text'],set(r['target_candidate_ids']))
    s['interface_repair']='accept_valid_numeric_id_without_regenerating_selection'
    requires=needs_refine(s['selection_kind'],s['selected_box'],changed_from_c=s['changed_from_c'])
    if requires and model is None:
        torch,processor,model=_load_qwen('/root/rematch_models/Qwen3-VL-8B-Instruct',root/'results/next_stage_20260925/c_phase2/checkpoint-500')
    assert time.time()<deadline
    if torch is not None:
        with torch.inference_mode(): f=_refine_row(torch,processor,model,manifest[key],ev,ev['candidates'],s,out/'manifests/city412.jsonl')
    else:
        f=_refine_row(None,None,None,manifest[key],ev,ev['candidates'],s,out/'manifests/city412.jsonl')
    f['latency_seconds']=float(f.get('selection_latency_seconds',0))+float(f.get('refine_latency_seconds',0))
    f['input_tokens']=int(f.get('selection_input_tokens',0))+int(f.get('refine_input_tokens',0))
    f['generation_cap_hit']=bool(f.get('selection_generation_cap_hit',False) or f.get('refine_generation_cap_hit',False))
    new_selected[key],new_final[key]=s,f
    print('REPAIRED',key,s['selection_id'],f['refine_status'],flush=True)
for r in selected:
    if r['id'] not in affected: assert new_selected[r['id']]==r
for r in final:
    if r['id'] not in affected: assert new_final[r['id']]==r
dump_rows(folder/'selected_predictions.jsonl',[new_selected[r['id']] for r in selected])
dump_rows(folder/'final_predictions.jsonl',[new_final[r['id']] for r in final])
summary=json.loads((folder/'summary.json').read_text())
summary['parse_failures']=sum(not r['parsed'] for r in new_final.values())
summary['interface_repair_ids']=affected
summary['interface_repair_seconds_including_load']=time.time()-started
summary['pre_repair_directory']=str(old)
dump(folder/'summary.json',summary)
dump(out/'numeric_id_interface_repair.json',{'ids':affected,'corrected_before_scoring':True,'selection_regenerated':False,
     'gt_read':False,'other_query_records_unchanged':412-len(affected),'elapsed_seconds':time.time()-started,
     'meaning':'Only valid integer IDs are accepted, exactly as required by the method; reference IDs remain errors.'})
print(json.dumps({'repaired':affected,'remaining_parse_failures':summary['parse_failures']}))
