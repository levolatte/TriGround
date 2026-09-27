from __future__ import annotations
import json
from pathlib import Path
BATCH="batch002"; ROOT=Path(r"F:\AIC\results\multimodal_data_20260925\next_sources\city")/BATCH; INPUT=ROOT/"generation_inputs.json"; PART_NO=4
MANUAL={
 "city_group_001873":[
  {"original_id":"city_001873_006","object_id":"target","role":"target","semantic_label":"man pushing a cart","extent":"whole",
   "infrared":[0.28,0.17,0.50,0.85],"depth":None,
   "evidence":{"rgb":"A man in a white t-shirt stands at and pushes a blue cart; original query and RGB box retained.","infrared":"A separate human silhouette is visible near the cart, with a clear torso and legs.","depth":"No reliable person boundary separates from the surrounding depth artifacts; no box assigned."},
   "uncertainties":["IR boundary is blurred around the body/cart overlap.","Depth preview does not support an independent person box."]},
  {"original_id":"city_001873_002","object_id":"reference1","role":"reference","semantic_label":"ceiling light panel","extent":"whole",
   "infrared":[0.40,0.07,0.53,0.13],"depth":None,
   "evidence":{"rgb":"A rectangular ceiling light is directly above the man; source query and RGB box retained.","infrared":"A bright horizontal rectangular panel is distinct from the person in the upper ceiling row.","depth":"Top dark marks in the depth preview do not establish this panel's exact extent; no box assigned."},
   "uncertainties":["IR panel edges are soft and estimated.","Depth preview does not provide an isolated panel boundary."]},
 ],
 "city_group_000581":[
  {"original_id":"city_000581_001","object_id":"target","role":"target","semantic_label":"light-colored standing quadruped","extent":"part",
   "infrared":[0.00,0.65,0.15,1.00],"depth":[0.02,0.68,0.14,1.00],
   "evidence":{"rgb":"A light-colored animal stands partly cut off at the far-left lower edge; original RGB query and box retained.","infrared":"A separate upright animal-shaped thermal silhouette appears along the left edge.","depth":"A boundary-like contour of the left-edge animal is visible down to the bottom frame; no distance ordering inferred."},
   "uncertainties":["The animal is clipped by the left and lower image edges; species is not independently confirmed.","Auxiliary box only covers visible extent and is approximate."]},
  {"original_id":"city_000581_003","object_id":"reference1","role":"reference","semantic_label":"light-colored grazing quadruped","extent":"part",
   "infrared":[0.32,0.82,0.54,1.00],"depth":[0.34,0.84,0.52,1.00],
   "evidence":{"rgb":"A larger light-colored animal grazes in the foreground and is cropped at the bottom edge; original RGB query and box retained.","infrared":"A second, larger animal-shaped region is visible at the lower right, separated from the left-edge animal.","depth":"A partial body contour appears at the lower right of the preview and is distinct from the left animal; no distance ordering inferred."},
   "uncertainties":["The animal is cropped by the bottom frame; species is not independently confirmed.","IR/Depth contour is partial, so these boxes are approximate."]},
 ]
}
rows=json.loads(INPUT.read_text(encoding="utf-8")); by_id={r["id"]:r for r in rows}; assert set(MANUAL)=={r["id"] for r in rows[6:8]}
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
existing=[json.loads(x) for x in master.read_text(encoding="utf-8").splitlines() if x.strip()]; assert not ({b["id"] for b in existing}&set(MANUAL))
text="".join(json.dumps(b,ensure_ascii=False)+"\n" for b in bundles)
part.write_text(text,encoding="utf-8"); master.write_text(master.read_text(encoding="utf-8")+text,encoding="utf-8")
print(json.dumps({"part":str(part),"master":str(master),"bundles":len(bundles),"cumulative":len(existing)+len(bundles)},ensure_ascii=False))
