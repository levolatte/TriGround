from __future__ import annotations
import json
from pathlib import Path
BATCH="batch004"; ROOT=Path(r"F:\AIC\results\multimodal_data_20260925\next_sources\city")/BATCH; INPUT=ROOT/"generation_inputs.json"; PART_NO=5
MANUAL={
 "city_group_001488":[
  {"original_id":"city_001488_002","object_id":"target","role":"target","semantic_label":"man holding objects","extent":"whole",
   "infrared":[0.47,0.10,0.76,0.95],"depth":[0.49,0.08,0.78,0.97],
   "evidence":{"rgb":"A man in a white shirt stands centrally holding a keyboard and a small object; original query and RGB box retained.","infrared":"A bright full-body human silhouette is separate from the background and small floor items.","depth":"A connected person-shaped contour is visible from head to legs; no metric distance is inferred."},
   "uncertainties":["The held objects overlap the arm/body and are not independently bounded.","Auxiliary boundaries are approximate."]},
  {"original_id":"city_001488_008","object_id":"reference1","role":"reference","semantic_label":"ceiling light panel","extent":"whole",
   "infrared":[0.10,0.04,0.26,0.11],"depth":None,
   "evidence":{"rgb":"A rectangular ceiling panel is visible at the far left of the upper ceiling row; original query and RGB box retained.","infrared":"A bright rectangular panel is visible in the upper-left ceiling area.","depth":"The top holes/patches do not isolate this panel reliably; no box assigned."},
   "uncertainties":["IR panel edges are blurred.","Depth preview has no separate panel boundary."]},
 ],
 "city_group_001015":[
  {"original_id":"city_001015_007","object_id":"target","role":"target","semantic_label":"octagonal road sign back","extent":"part",
   "infrared":[0.02,0.00,0.31,0.47],"depth":[0.02,0.00,0.32,0.43],
   "evidence":{"rgb":"A large octagonal sign back fills the upper-left foreground and is cut by the top/left image borders; original query and RGB box retained.","infrared":"The large octagonal plate and its pole form a distinct high-contrast silhouette in the upper-left region.","depth":"A polygon-shaped sign contour is clearly isolated from the background; the depth preview is used for shape only, not distance."},
   "uncertainties":["The sign is clipped at the top/left frame edges.","The IR/Depth boxes estimate the visible plate extent independently; the pole is not the intended object."]},
 ]
}
rows=json.loads(INPUT.read_text(encoding="utf-8")); by_id={r["id"]:r for r in rows}; assert set(MANUAL)=={r["id"] for r in rows[8:10]}
bundles=[]
for bundle_id,specs in MANUAL.items():
 src=by_id[bundle_id]; originals={r["id"]:r for r in src["original_records"]}; objects=[]; queries=[]
 for idx,s in enumerate(specs,start=1):
  orig=originals[s["original_id"]]; confirmed=["rgb"]+[m for m in ("infrared","depth") if s[m] is not None]
  objects.append({"object_id":s["object_id"],"role":s["role"],"semantic_label":s["semantic_label"],"boxes":{"rgb":orig["bbox"],"infrared":s["infrared"],"depth":s["depth"]},"confirmed_modalities":confirmed,"extent":s["extent"],"evidence":s["evidence"],"uncertainties":s["uncertainties"]})
  queries.append({"id":f"q{idx:02d}","origin_query_id":orig["id"],"query":orig["query"],"target_object_id":s["object_id"],"answerable_modalities":confirmed,"augmentation_eligible":False,"review_status":"provisional"})
 bundle={"id":bundle_id,"source":"city","split":"train","scene_id":bundle_id,"images":src["images"],"depth_policy":src["depth_policy"],"review_status":"provisional","revision":0,"objects":objects,"queries":queries,"relations":[]}
 if len(objects)<2:
  bundle["selection_note"]="Only the octagonal sign had a reliable independent IR/Depth contour. Other candidate signs/vehicles remained blurred or mixed in the full frames and focused crop, so no RGB-only competitor Query was added."
 bundles.append(bundle)
part=ROOT/f"evidence_part{PART_NO:02d}.jsonl"; master=ROOT/"object_evidence.jsonl"; review=ROOT/f"review_part{PART_NO:02d}"
assert not part.exists() and not review.exists(); existing=[json.loads(x) for x in master.read_text(encoding="utf-8").splitlines() if x.strip()]; assert not ({b["id"] for b in existing}&set(MANUAL))
text="".join(json.dumps(b,ensure_ascii=False)+"\n" for b in bundles);part.write_text(text,encoding="utf-8");master.write_text(master.read_text(encoding="utf-8")+text,encoding="utf-8")
print(json.dumps({"part":str(part),"master":str(master),"bundles":len(bundles),"cumulative":len(existing)+len(bundles),"objects":sum(len(b['objects']) for b in bundles)},ensure_ascii=False))
