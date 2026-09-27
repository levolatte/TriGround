from __future__ import annotations
import json
from pathlib import Path
BATCH="batch002"; ROOT=Path(r"F:\AIC\results\multimodal_data_20260925\next_sources\city")/BATCH; INPUT=ROOT/"generation_inputs.json"; PART_NO=3
MANUAL={
 "city_group_001240":[
  {"original_id":"city_001240_006","object_id":"target","role":"target","semantic_label":"man pushing a cart","extent":"whole",
   "infrared":[0.46,0.25,0.61,0.79],"depth":None,
   "evidence":{"rgb":"A man in a white t-shirt stands at the cart handle; original query and RGB box retained.","infrared":"A bright human silhouette is separately visible beside the cart.","depth":"The displayed depth view does not show a clean independent person boundary; no box assigned."},
   "uncertainties":["IR person outline is blurred around the hands and cart handle.","No reliable person boundary confirmed in depth preview."]},
  {"original_id":"city_001240_009","object_id":"reference1","role":"reference","semantic_label":"robot on the cart","extent":"whole",
   "infrared":[0.36,0.58,0.53,0.73],"depth":None,
   "evidence":{"rgb":"A small robot is on the blue cart; source query and RGB box retained.","infrared":"A dark device-shaped cluster lies on the cart bed, spatially separate from the man's bright legs.","depth":"The visible depth contours in this area do not isolate the robot from the cart/floor; no box assigned."},
   "uncertainties":["IR device boundary is fuzzy and the box is approximate.","Depth preview does not provide a reliable independent robot extent."]},
 ],
 "city_group_shuming_52_00000032":[
  {"original_id":"city_shuming_52_00000032_005","object_id":"target","role":"target","semantic_label":"sedan","extent":"whole",
   "infrared":[0.08,0.65,0.80,0.99],"depth":[0.07,0.66,0.62,0.96],
   "evidence":{"rgb":"A large sedan fills the lower foreground; original RGB query and box retained.","infrared":"A broad vehicle body and window boundary are visible across the foreground.","depth":"A continuous vehicle-shaped contour follows the car body in the lower foreground; no distance ordering inferred."},
   "uncertainties":["Depth contour is incomplete near the bottom/right image boundary; box follows the visible vehicle outline.","IR and depth extents are independent estimates due view registration differences."]},
  {"original_id":"city_shuming_52_00000032_002","object_id":"reference1","role":"reference","semantic_label":"red hanging lantern","extent":"whole",
   "infrared":[0.68,0.36,0.75,0.47],"depth":None,
   "evidence":{"rgb":"The uppermost of three red lanterns hangs on the pole; original query and RGB box retained.","infrared":"The upper lantern appears as a separate warm-colored rounded patch above two similar lanterns.","depth":"The depth preview does not isolate the lantern shape from the pole/background; no box assigned."},
   "uncertainties":["IR blur merges the lantern edge with the vertical pole; auxiliary box is approximate.","No reliable lantern boundary confirmed in depth preview."]},
 ]
}
rows=json.loads(INPUT.read_text(encoding="utf-8")); by_id={r["id"]:r for r in rows}; assert set(MANUAL)=={r["id"] for r in rows[4:6]}
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
