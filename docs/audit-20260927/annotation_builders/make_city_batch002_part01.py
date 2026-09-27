from __future__ import annotations
import json
from pathlib import Path

BATCH = "batch002"
ROOT = Path(r"F:\AIC\results\multimodal_data_20260925\next_sources\city") / BATCH
INPUT = ROOT / "generation_inputs.json"
OUT = ROOT
PART_NO = 1
MANUAL = {
    "city_group_001813": [
        {"original_id":"city_001813_008","object_id":"target","role":"target","semantic_label":"man","extent":"whole",
         "infrared":[0.40,0.16,0.73,0.92],"depth":[0.46,0.15,0.79,0.98],
         "evidence":{"rgb":"A man stands centrally with arms extended, holding small objects; preserved RGB target box covers the described man and held items.",
                     "infrared":"A separate bright human silhouette is visible from head through legs, with both arms extended.",
                     "depth":"The preview contains a large connected person-shaped contour with head, torso, arms, and legs; no distance ordering inferred."},
         "uncertainties":["IR and depth bounds are independent visual estimates; held items are small and not separately bounded."]},
        {"original_id":"city_001813_004","object_id":"reference1","role":"reference","semantic_label":"ceiling light panel","extent":"whole",
         "infrared":[0.48,0.08,0.60,0.13],"depth":None,
         "evidence":{"rgb":"A rectangular ceiling panel is directly above the man's head; original query and RGB box are retained.",
                     "infrared":"A separate bright rectangular panel is visible above the person in the upper ceiling row.",
                     "depth":"The top-region dark patches do not isolate this panel with a reliable boundary; no depth box assigned."},
         "uncertainties":["The IR panel boundary is blurred and estimated from the visible rectangle.","Depth preview does not provide a reliable panel extent."]},
    ],
    "city_group_shuming_1180_00000231": [
        {"original_id":"city_shuming_1180_00000231_001","object_id":"target","role":"target","semantic_label":"spotted deer","extent":"whole",
         "infrared":[0.52,0.49,0.70,0.73],"depth":[0.52,0.48,0.70,0.75],
         "evidence":{"rgb":"The right member of the two foreground spotted deer is visible as a separate animal; original RGB GT retained.",
                     "infrared":"A distinct foreground animal silhouette appears to the right of the neighboring foreground deer.",
                     "depth":"Two side-by-side body contours are visible; this box follows the right contour independently of RGB."},
         "uncertainties":["The animal legs and lower body overlap slightly with the ground; auxiliary box edges are approximate.","Depth preview is used only for silhouette correspondence, not near/far ordering."]},
        {"original_id":"city_shuming_1180_00000231_002","object_id":"reference1","role":"reference","semantic_label":"spotted deer","extent":"whole",
         "infrared":[0.39,0.49,0.55,0.74],"depth":[0.36,0.48,0.54,0.75],
         "evidence":{"rgb":"The left member of the two foreground spotted deer is separately visible; original RGB GT retained.",
                     "infrared":"A distinct foreground animal silhouette appears left of the neighboring deer, though blurred.",
                     "depth":"A separate left body contour is visible beside the right foreground animal; box is estimated from its contour."},
         "uncertainties":["The left animal is partly mixed with branches/debris in IR; box is approximate.","Depth preview is used only for silhouette correspondence, not near/far ordering."]},
    ],
}
rows=json.loads(INPUT.read_text(encoding="utf-8"))
by_id={r["id"]:r for r in rows}
selected_ids=list(MANUAL)
assert len(selected_ids)==2
bundles=[]
for bundle_id in selected_ids:
    src=by_id[bundle_id]
    objects=[]; queries=[]
    originals={r["id"]:r for r in src["original_records"]}
    for idx, spec in enumerate(MANUAL[bundle_id],start=1):
        original=originals[spec["original_id"]]
        confirmed=["rgb"]
        for modality in ("infrared","depth"):
            if spec[modality] is not None:
                confirmed.append(modality)
        objects.append({
            "object_id":spec["object_id"], "role":spec["role"], "semantic_label":spec["semantic_label"],
            "boxes":{"rgb":original["bbox"],"infrared":spec["infrared"],"depth":spec["depth"]},
            "confirmed_modalities":confirmed,"extent":spec["extent"],"evidence":spec["evidence"],"uncertainties":spec["uncertainties"]
        })
        queries.append({"id":f"q{idx:02d}","origin_query_id":original["id"],"query":original["query"],
                        "target_object_id":spec["object_id"],"answerable_modalities":confirmed,
                        "augmentation_eligible":False,"review_status":"provisional"})
    bundles.append({"id":bundle_id,"source":"city","split":"train","scene_id":bundle_id,"images":src["images"],
                    "depth_policy":src["depth_policy"],"review_status":"provisional","revision":0,
                    "objects":objects,"queries":queries,"relations":[]})
part=OUT/f"evidence_part{PART_NO:02d}.jsonl"
master=OUT/"object_evidence.jsonl"
review=OUT/f"review_part{PART_NO:02d}"
for p in (part,master,review): assert not p.exists(), f"Refusing to overwrite {p}"
text="".join(json.dumps(b,ensure_ascii=False)+"\n" for b in bundles)
part.write_text(text,encoding="utf-8")
master.write_text(text,encoding="utf-8")
print(json.dumps({"part":str(part),"master":str(master),"bundles":len(bundles),"objects":sum(len(b['objects']) for b in bundles),"queries":sum(len(b['queries']) for b in bundles)},ensure_ascii=False))
