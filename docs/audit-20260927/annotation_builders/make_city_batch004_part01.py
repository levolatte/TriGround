from __future__ import annotations
import json
from pathlib import Path
BATCH="batch004"; ROOT=Path(r"F:\AIC\results\multimodal_data_20260925\next_sources\city")/BATCH; INPUT=ROOT/"generation_inputs.json"; PART_NO=1
MANUAL={
 "city_group_001453":[
  {"original_id":"city_001453_002","object_id":"target","role":"target","semantic_label":"man holding an object","extent":"whole",
   "infrared":[0.42,0.08,0.73,0.94],"depth":[0.45,0.07,0.79,0.97],
   "evidence":{"rgb":"A man in a white t-shirt stands centrally holding an object; original RGB query and bbox retained.","infrared":"The central person is a distinct bright full-body silhouette with extended arms.","depth":"A connected person-shaped contour is visible from head through legs; no metric distance inference."},
   "uncertainties":["The held keyboard/object is not separately bounded from the person.","Auxiliary outlines are jagged and estimated independently from the RGB box."]},
  {"original_id":"city_001453_009","object_id":"reference1","role":"reference","semantic_label":"ceiling light panel","extent":"whole",
   "infrared":[0.45,0.04,0.57,0.10],"depth":None,
   "evidence":{"rgb":"A rectangular ceiling panel is above the man's head; original query and bbox retained.","infrared":"A bright rectangular panel is separately visible directly over the person's head.","depth":"This top-region panel cannot be separated from nearby dark ceiling marks in the depth preview; no box assigned."},
   "uncertainties":["IR panel edges are soft and approximate.","Depth preview does not show a reliable panel boundary."]},
 ],
 "city_group_001593":[
  {"original_id":"city_001593_001","object_id":"target","role":"target","semantic_label":"shaggy light-colored quadruped","extent":"whole",
   "infrared":[0.38,0.46,0.57,0.77],"depth":None,
   "evidence":{"rgb":"A shaggy light-colored animal stands near the foreground fence; original RGB query and bbox retained.","infrared":"One large light animal silhouette is visible behind the railing, with a distinct body and legs.","depth":"The preview is dominated by railing/floor and does not isolate an animal contour; no box assigned."},
   "uncertainties":["The precise species is not independently established from the image; source Query text is kept unchanged.","A second dark animal behind the fence blends into the rails in IR and is not reliably separable; no second object was added.","Depth preview provides no reliable animal extent."]},
 ]
}
rows=json.loads(INPUT.read_text(encoding="utf-8")); by_id={r["id"]:r for r in rows}; assert set(MANUAL)=={r["id"] for r in rows[:2]}
bundles=[]
for bundle_id,specs in MANUAL.items():
 src=by_id[bundle_id]; originals={r["id"]:r for r in src["original_records"]}; objects=[]; queries=[]
 for idx,s in enumerate(specs,start=1):
  orig=originals[s["original_id"]]; confirmed=["rgb"]+[m for m in ("infrared","depth") if s[m] is not None]
  objects.append({"object_id":s["object_id"],"role":s["role"],"semantic_label":s["semantic_label"],"boxes":{"rgb":orig["bbox"],"infrared":s["infrared"],"depth":s["depth"]},"confirmed_modalities":confirmed,"extent":s["extent"],"evidence":s["evidence"],"uncertainties":s["uncertainties"]})
  queries.append({"id":f"q{idx:02d}","origin_query_id":orig["id"],"query":orig["query"],"target_object_id":s["object_id"],"answerable_modalities":confirmed,"augmentation_eligible":False,"review_status":"provisional"})
 bundle={"id":bundle_id,"source":"city","split":"train","scene_id":bundle_id,"images":src["images"],"depth_policy":src["depth_policy"],"review_status":"provisional","revision":0,"objects":objects,"queries":queries,"relations":[]}
 if len(objects)<2: bundle["selection_note"]="Only one distinct object was reliable in RGB/IR after full-frame review and one focused crop; the second candidate remains merged with the railing, so no object was added from RGB alone."
 bundles.append(bundle)
part=ROOT/f"evidence_part{PART_NO:02d}.jsonl"; master=ROOT/"object_evidence.jsonl"; review=ROOT/f"review_part{PART_NO:02d}"
for p in (part,master,review): assert not p.exists(),f"Refusing to overwrite {p}"
text="".join(json.dumps(b,ensure_ascii=False)+"\n" for b in bundles);part.write_text(text,encoding="utf-8");master.write_text(text,encoding="utf-8")
print(json.dumps({"part":str(part),"master":str(master),"bundles":len(bundles),"objects":sum(len(b['objects']) for b in bundles)},ensure_ascii=False))
