from __future__ import annotations
import json
from pathlib import Path
BATCH="batch002"; ROOT=Path(r"F:\AIC\results\multimodal_data_20260925\next_sources\city")/BATCH; INPUT=ROOT/"generation_inputs.json"; PART_NO=2
MANUAL={
 "city_group_001211":[
  {"original_id":"city_001211_006","object_id":"target","role":"target","semantic_label":"man pushing a cart","extent":"whole",
   "infrared":[0.55,0.12,0.70,0.84],"depth":[0.53,0.10,0.73,0.87],
   "evidence":{"rgb":"A single man stands behind and pushes a blue cart; source RGB query and bbox retained.","infrared":"A bright human figure is distinct from the cart and floor row.","depth":"A large connected person-shaped contour with head, torso, and legs is visible; no depth direction inferred."},
   "uncertainties":["IR and depth outlines are approximate because the views are softened and slightly offset."]},
  {"original_id":"city_001211_008","object_id":"reference1","role":"reference","semantic_label":"tracked robot on the cart","extent":"whole",
   "infrared":[0.59,0.56,0.73,0.70],"depth":None,
   "evidence":{"rgb":"A dark tracked device sits on the blue cart; original query and RGB box retained.","infrared":"A distinct dark mass on the cart bed is visible below and to the right of the person.","depth":"The preview does not separate this device from the cart and floor; no depth box assigned."},
   "uncertainties":["The IR robot boundary partly blends with the cart and is approximate.","Depth silhouette is insufficient to isolate the robot independently."]},
 ],
 "city_group_000005_026_00000001":[
  {"original_id":"city_000005_026_00000001_001","object_id":"target","role":"target","semantic_label":"person in a dark puffer jacket","extent":"whole",
   "infrared":[0.59,0.23,0.76,0.95],"depth":None,
   "evidence":{"rgb":"A large foreground person in a dark puffer jacket walks along the stepping-stone path; original RGB query and box retained.","infrared":"The larger foreground human silhouette is distinct from the smaller person ahead.","depth":"Tree trunks, foliage, and ground dominate; a separate full-body boundary is not reliable enough to box."},
   "uncertainties":["The person is distinguishable in IR, but the scene is blurred and the extent is estimated.","Depth preview does not provide a clean person contour."]},
  {"original_id":"city_000005_026_00000001_004","object_id":"reference1","role":"reference","semantic_label":"person carrying a bag","extent":"whole",
   "infrared":[0.42,0.28,0.54,0.67],"depth":None,
   "evidence":{"rgb":"A smaller person ahead of the foreground walker carries an item at their side; original RGB query and box retained.","infrared":"A separate smaller human silhouette is visible to the left/ahead of the larger foreground figure.","depth":"No reliable isolated contour for this smaller person is visible in the depth preview; no box assigned."},
   "uncertainties":["IR clothing color is unavailable; the carried bag is not separately resolved.","Depth preview does not provide a reliable target extent."]},
 ]
}
rows=json.loads(INPUT.read_text(encoding="utf-8")); by_id={r["id"]:r for r in rows}; assert set(MANUAL)=={r["id"] for r in rows[2:4]}
bundles=[]
for bundle_id,specs in MANUAL.items():
 src=by_id[bundle_id]; originals={r["id"]:r for r in src["original_records"]}; objects=[]; queries=[]
 for idx,s in enumerate(specs,start=1):
  orig=originals[s["original_id"]]; confirmed=["rgb"]+[m for m in ("infrared","depth") if s[m] is not None]
  objects.append({"object_id":s["object_id"],"role":s["role"],"semantic_label":s["semantic_label"],"boxes":{"rgb":orig["bbox"],"infrared":s["infrared"],"depth":s["depth"]},"confirmed_modalities":confirmed,"extent":s["extent"],"evidence":s["evidence"],"uncertainties":s["uncertainties"]})
  queries.append({"id":f"q{idx:02d}","origin_query_id":orig["id"],"query":orig["query"],"target_object_id":s["object_id"],"answerable_modalities":confirmed,"augmentation_eligible":False,"review_status":"provisional"})
 bundles.append({"id":bundle_id,"source":"city","split":"train","scene_id":bundle_id,"images":src["images"],"depth_policy":src["depth_policy"],"review_status":"provisional","revision":0,"objects":objects,"queries":queries,"relations":[]})
part=ROOT/f"evidence_part{PART_NO:02d}.jsonl"; master=ROOT/"object_evidence.jsonl"; review=ROOT/f"review_part{PART_NO:02d}"
assert not part.exists() and not review.exists()
existing=[json.loads(x) for x in master.read_text(encoding="utf-8").splitlines() if x.strip()]
assert not ({b["id"] for b in existing}&set(MANUAL))
text="".join(json.dumps(b,ensure_ascii=False)+"\n" for b in bundles)
part.write_text(text,encoding="utf-8"); master.write_text(master.read_text(encoding="utf-8")+text,encoding="utf-8")
print(json.dumps({"part":str(part),"master":str(master),"bundles":len(bundles),"cumulative":len(existing)+len(bundles)},ensure_ascii=False))
