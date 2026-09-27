from __future__ import annotations
import json
from pathlib import Path
BATCH="batch002"; ROOT=Path(r"F:\AIC\results\multimodal_data_20260925\next_sources\city")/BATCH; INPUT=ROOT/"generation_inputs.json"; PART_NO=5
MANUAL={
 "city_group_001965":[
  {"original_id":"city_001965_006","object_id":"target","role":"target","semantic_label":"person holding plush toys","extent":"whole",
   "infrared":[0.43,0.12,0.70,0.94],"depth":[0.45,0.13,0.77,0.97],
   "evidence":{"rgb":"A man holds plush toys while bending near the floor; original RGB query and box retained.","infrared":"A distinct bright person silhouette is visible with arms and legs separated from the background.","depth":"A large connected person-shaped contour includes head, torso, arms, and legs; no metric depth inference."},
   "uncertainties":["The held plush toys overlap the body outline and are not separately annotated.","Depth boundary is jagged; box follows the visible silhouette approximately."]},
  {"original_id":"city_001965_008","object_id":"reference1","role":"reference","semantic_label":"ceiling light panel","extent":"whole",
   "infrared":[0.44,0.05,0.56,0.10],"depth":None,
   "evidence":{"rgb":"A ceiling light panel appears above the man's head; original query and box retained.","infrared":"A bright rectangular panel is separately visible in the ceiling row above the person.","depth":"Several dark ceiling patches are present but this panel's boundary is not reliably isolated; no box assigned."},
   "uncertainties":["IR panel edges are soft.","Depth preview does not provide a reliable panel box."]},
 ],
 "city_group_shuming_612_00000323":[
  {"original_id":"city_shuming_612_00000323_001","object_id":"target","role":"target","semantic_label":"large quadruped in the enclosure","extent":"whole",
   "infrared":[0.00,0.49,0.23,0.78],"depth":None,
   "evidence":{"rgb":"A large four-legged animal is visible under the enclosure roof at the left; original RGB query and box retained.","infrared":"A single large animal-shaped thermal silhouette is visible at the far left behind the enclosure boundary.","depth":"The preview is dominated by noisy pen and ground regions without a separate animal contour; no box assigned."},
   "uncertainties":["Species is not clear enough to independently confirm from the view; the source query wording is preserved unchanged.","A crop of the lower IR scene showed only this one reliable animal; nearby small figures were not independently visible in IR and no second object was added.","Depth does not provide a reliable animal extent."]},
 ],
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
 if len(objects)<2: bundle["selection_note"]="Only one distinct object was reliably confirmed in auxiliary views after full-frame review and one necessary crop; no ordinary RGB-only object was added to fill the set."
 bundles.append(bundle)
part=ROOT/f"evidence_part{PART_NO:02d}.jsonl"; master=ROOT/"object_evidence.jsonl"; review=ROOT/f"review_part{PART_NO:02d}"
assert not part.exists() and not review.exists()
existing=[json.loads(x) for x in master.read_text(encoding="utf-8").splitlines() if x.strip()]; assert not ({b["id"] for b in existing}&set(MANUAL))
text="".join(json.dumps(b,ensure_ascii=False)+"\n" for b in bundles)
part.write_text(text,encoding="utf-8"); master.write_text(master.read_text(encoding="utf-8")+text,encoding="utf-8")
print(json.dumps({"part":str(part),"master":str(master),"bundles":len(bundles),"cumulative":len(existing)+len(bundles),"objects":sum(len(b['objects']) for b in bundles)},ensure_ascii=False))
