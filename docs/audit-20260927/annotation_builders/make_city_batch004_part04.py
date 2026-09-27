from __future__ import annotations
import json
from pathlib import Path
BATCH="batch004"; ROOT=Path(r"F:\AIC\results\multimodal_data_20260925\next_sources\city")/BATCH; INPUT=ROOT/"generation_inputs.json"; PART_NO=4
MANUAL={
 "city_group_001466":[
  {"original_id":"city_001466_002","object_id":"target","role":"target","semantic_label":"man holding a keyboard","extent":"whole",
   "infrared":[0.49,0.08,0.72,0.95],"depth":[0.49,0.07,0.78,0.97],
   "evidence":{"rgb":"A man in a white t-shirt holds a keyboard; original query and RGB box retained.","infrared":"A distinct full-body person silhouette is visible, with an object by one hand.","depth":"A connected human-shaped boundary is visible from head to legs; no distance ordering inferred."},
   "uncertainties":["The keyboard overlaps the person's arm and is not separately bounded.","Auxiliary person bounds are approximate."]},
  {"original_id":"city_001466_009","object_id":"reference1","role":"reference","semantic_label":"ceiling light panel","extent":"whole",
   "infrared":[0.44,0.04,0.56,0.10],"depth":None,
   "evidence":{"rgb":"A rectangular ceiling panel is at the top center; original query and RGB box retained.","infrared":"A distinct bright rectangle appears in the ceiling row above the person.","depth":"The preview's dark ceiling marks do not isolate a reliable matching panel boundary; no box assigned."},
   "uncertainties":["IR panel is blurred at its edges.","Depth preview does not confirm this panel's extent."]},
 ],
 "city_group_shuming_278_00000186":[
  {"original_id":"city_shuming_278_00000186_001","object_id":"target","role":"target","semantic_label":"hanging lantern","extent":"whole","infrared":None,"depth":None,
   "evidence":{"rgb":"A small lantern hangs under the leftmost eaves; original RGB GT retained.","infrared":"The IR facade is blurred and the lantern does not have a separable outline.","depth":"The depth preview shows broad building surfaces without an isolated lantern contour."},
   "uncertainties":["No reliable auxiliary boundary is confirmed; RGB-only query omitted."]},
  {"original_id":"city_shuming_278_00000186_002","object_id":"reference1","role":"reference","semantic_label":"hanging lantern","extent":"whole","infrared":None,"depth":None,
   "evidence":{"rgb":"A second lantern hangs to the right of the leftmost eaves lantern; original RGB GT retained.","infrared":"The IR facade does not resolve this small lantern separately from the roof and wall.","depth":"No distinct lantern contour is present in the depth preview."},
   "uncertainties":["No reliable auxiliary boundary is confirmed; RGB-only query omitted."]},
 ]
}
rows=json.loads(INPUT.read_text(encoding="utf-8")); by_id={r["id"]:r for r in rows}; assert set(MANUAL)=={r["id"] for r in rows[6:8]}
bundles=[]
for bundle_id,specs in MANUAL.items():
 src=by_id[bundle_id]; originals={r["id"]:r for r in src["original_records"]}; objects=[]; queries=[]
 for idx,s in enumerate(specs,start=1):
  orig=originals[s["original_id"]]; confirmed=["rgb"]+[m for m in ("infrared","depth") if s[m] is not None]
  objects.append({"object_id":s["object_id"],"role":s["role"],"semantic_label":s["semantic_label"],"boxes":{"rgb":orig["bbox"],"infrared":s["infrared"],"depth":s["depth"]},"confirmed_modalities":confirmed,"extent":s["extent"],"evidence":s["evidence"],"uncertainties":s["uncertainties"]})
  if s["infrared"] is not None or s["depth"] is not None:
   queries.append({"id":f"q{idx:02d}","origin_query_id":orig["id"],"query":orig["query"],"target_object_id":s["object_id"],"answerable_modalities":confirmed,"augmentation_eligible":False,"review_status":"provisional"})
 bundle={"id":bundle_id,"source":"city","split":"train","scene_id":bundle_id,"images":src["images"],"depth_policy":src["depth_policy"],"review_status":"provisional","revision":0,"objects":objects,"queries":queries,"relations":[]}
 if not queries:
  bundle["selection_outcome"]="no_confirmed_auxiliary_evidence"
  bundle["selection_note"]="Preserved original RGB GT for two distinct lanterns, but full-view and focused IR/Depth review showed only blurred facade/building structure without reliable lantern boundaries; original RGB-only Query texts were not emitted as grounding tasks."
 bundles.append(bundle)
part=ROOT/f"evidence_part{PART_NO:02d}.jsonl"; master=ROOT/"object_evidence.jsonl"; review=ROOT/f"review_part{PART_NO:02d}"
assert not part.exists() and not review.exists(); existing=[json.loads(x) for x in master.read_text(encoding="utf-8").splitlines() if x.strip()]; assert not ({b["id"] for b in existing}&set(MANUAL))
text="".join(json.dumps(b,ensure_ascii=False)+"\n" for b in bundles);part.write_text(text,encoding="utf-8");master.write_text(master.read_text(encoding="utf-8")+text,encoding="utf-8")
print(json.dumps({"part":str(part),"master":str(master),"bundles":len(bundles),"cumulative":len(existing)+len(bundles),"queries":sum(len(b['queries']) for b in bundles)},ensure_ascii=False))
