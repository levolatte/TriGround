from __future__ import annotations
import json
from pathlib import Path
BATCH="batch004"; ROOT=Path(r"F:\AIC\results\multimodal_data_20260925\next_sources\city")/BATCH; INPUT=ROOT/"generation_inputs.json"; PART_NO=3
MANUAL={
 "city_group_001461":[
  {"original_id":"city_001461_002","object_id":"target","role":"target","semantic_label":"man holding keyboards","extent":"whole",
   "infrared":[0.48,0.08,0.73,0.94],"depth":[0.49,0.07,0.78,0.96],
   "evidence":{"rgb":"A man stands centrally holding two keyboards; original query and RGB box retained.","infrared":"A distinct full-body person silhouette is visible with arms and legs separated.","depth":"The depth preview shows a connected person-shaped contour from head to lower legs; no numeric distance inference."},
   "uncertainties":["The keyboards overlap the body/arms and are not independently bounded.","IR/depth boundaries are approximate due blur and contour artifacts."]},
  {"original_id":"city_001461_009","object_id":"reference1","role":"reference","semantic_label":"ceiling light panel","extent":"whole",
   "infrared":[0.45,0.04,0.56,0.10],"depth":None,
   "evidence":{"rgb":"A rectangular ceiling panel appears in the top center; original query and RGB box retained.","infrared":"A bright rectangular panel is separated from the person's silhouette in the upper ceiling row.","depth":"Dark top marks do not isolate this panel reliably; no depth box assigned."},
   "uncertainties":["IR panel edges are soft.","Depth preview does not provide an isolated panel boundary."]},
 ],
 "city_group_000953":[
  {"original_id":"city_000953_002","object_id":"target","role":"target","semantic_label":"zebra","extent":"whole","infrared":None,"depth":None,
   "evidence":{"rgb":"A striped animal stands in profile under the shelter; original RGB GT retained.","infrared":"The shelter and foreground foliage dominate; a separate full animal outline is not reliable after full view and focused crop.","depth":"The preview has strong vegetation and structure contours but no distinct zebra boundary."},
   "uncertainties":["Auxiliary modalities do not confirm this object's extent; RGB-only query not generated."]},
  {"original_id":"city_000953_003","object_id":"reference1","role":"reference","semantic_label":"zebra","extent":"whole","infrared":None,"depth":None,
   "evidence":{"rgb":"A second striped animal is visible near the trough on the right, distinct from the zebra under the shelter; original RGB GT retained.","infrared":"No reliable separated contour for this right-side animal is identifiable in IR.","depth":"No distinct animal-shaped depth contour is visible apart from enclosure/foliage structure."},
   "uncertainties":["Auxiliary modalities do not confirm this object's extent; RGB-only query not generated."]},
 ]
}
rows=json.loads(INPUT.read_text(encoding="utf-8")); by_id={r["id"]:r for r in rows}; assert set(MANUAL)=={r["id"] for r in rows[4:6]}
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
  bundle["selection_note"]="Two different RGB-visible zebras were preserved with their original RGB GT boxes, but IR and Depth could not independently confirm separate object extents after full-frame review and one focused crop; no RGB-only Query was generated."
 bundles.append(bundle)
part=ROOT/f"evidence_part{PART_NO:02d}.jsonl"; master=ROOT/"object_evidence.jsonl"; review=ROOT/f"review_part{PART_NO:02d}"
assert not part.exists() and not review.exists(); existing=[json.loads(x) for x in master.read_text(encoding="utf-8").splitlines() if x.strip()]; assert not ({b["id"] for b in existing}&set(MANUAL))
text="".join(json.dumps(b,ensure_ascii=False)+"\n" for b in bundles);part.write_text(text,encoding="utf-8");master.write_text(master.read_text(encoding="utf-8")+text,encoding="utf-8")
print(json.dumps({"part":str(part),"master":str(master),"bundles":len(bundles),"cumulative":len(existing)+len(bundles),"queries":sum(len(b['queries']) for b in bundles)},ensure_ascii=False))
