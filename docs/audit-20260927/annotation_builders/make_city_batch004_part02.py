from __future__ import annotations
import json
from pathlib import Path
BATCH="batch004"; ROOT=Path(r"F:\AIC\results\multimodal_data_20260925\next_sources\city")/BATCH; INPUT=ROOT/"generation_inputs.json"; PART_NO=2
MANUAL={
 "city_group_001459":[
  {"original_id":"city_001459_002","object_id":"target","role":"target","semantic_label":"man holding a red disc","extent":"whole",
   "infrared":[0.48,0.08,0.73,0.94],"depth":[0.46,0.08,0.77,0.96],
   "evidence":{"rgb":"A man holds a round red object in one hand; original query and RGB box retained.","infrared":"A bright human silhouette is visible and a smaller bright feature appears by the raised hand.","depth":"The person forms a connected contour from head through legs; the held disc is not isolated."},
   "uncertainties":["The red disc boundary merges with the hand in IR and is not separately boxed.","Depth box covers the person only; no numeric distance inference."]},
  {"original_id":"city_001459_009","object_id":"reference1","role":"reference","semantic_label":"ceiling light panel","extent":"whole",
   "infrared":[0.44,0.04,0.56,0.10],"depth":None,
   "evidence":{"rgb":"A rectangular ceiling panel appears above the man's head; original query and RGB box retained.","infrared":"A bright horizontal ceiling panel is visible above the person in the top row.","depth":"Top dark patches are not sufficiently separable to assign this panel's depth extent; no box assigned."},
   "uncertainties":["IR panel edges are blurred.","Depth preview does not provide a reliable boundary for this panel."]},
 ],
 "city_group_000005_018_00000405":[
  {"original_id":"city_000005_018_00000405_001","object_id":"target","role":"target","semantic_label":"person in a blue jacket and light hat","extent":"whole",
   "infrared":[0.37,0.77,0.42,0.95],"depth":None,
   "evidence":{"rgb":"A small person in a blue jacket and light hat walks among the group at the lower center; original query and RGB bbox retained.","infrared":"A separate small human-shaped thermal figure is visible in the lower group, left of the adjacent dark-clothed person.","depth":"The preview is dominated by vegetation and ground; no isolated person contour is reliable."},
   "uncertainties":["IR figures are small and blurred; identity is assigned by preserved left-to-right relation, with approximate bounds.","Depth preview does not confirm individual people."]},
  {"original_id":"city_000005_018_00000405_002","object_id":"reference1","role":"reference","semantic_label":"person in dark clothing","extent":"whole",
   "infrared":[0.43,0.77,0.48,0.95],"depth":None,
   "evidence":{"rgb":"A second person in dark clothing walks beside the blue-jacketed person; original query and RGB bbox retained.","infrared":"A separate small thermal human figure is visible immediately to the right of the blue-jacketed figure.","depth":"No reliable individual silhouette separates from the background in this depth preview; no box assigned."},
   "uncertainties":["The two adjacent IR silhouettes are soft, so their boxes are approximate.","Depth preview does not confirm individual people."]},
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
assert not part.exists() and not review.exists(); existing=[json.loads(x) for x in master.read_text(encoding="utf-8").splitlines() if x.strip()]; assert not ({b["id"] for b in existing}&set(MANUAL))
text="".join(json.dumps(b,ensure_ascii=False)+"\n" for b in bundles);part.write_text(text,encoding="utf-8");master.write_text(master.read_text(encoding="utf-8")+text,encoding="utf-8")
print(json.dumps({"part":str(part),"master":str(master),"bundles":len(bundles),"cumulative":len(existing)+len(bundles)},ensure_ascii=False))
