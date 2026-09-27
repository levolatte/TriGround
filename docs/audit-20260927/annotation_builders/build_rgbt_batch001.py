import argparse, json
from pathlib import Path

INPUT = Path(r"F:\AIC\results\multimodal_data_20260925\next_sources\rgbt\batch001\generation_inputs.json")
OUT = Path(r"F:\AIC\results\multimodal_data_20260925\next_sources\rgbt\batch001")

META = {
"rgbt_flir_train_000011": dict(label="white van", ir=[0.365,0.426,0.455,0.557], extent="whole", confirmed=["rgb","infrared"], rgb="A white van with bright headlights is centered on the nighttime roadway.", infrared="A vehicle silhouette and hot headlights are visible at the corresponding mid-road location.", uncertainty=[]),
"rgbt_flir_train_000012": dict(label="person beside another person", ir=[0.585,0.478,0.688,0.805], extent="group", confirmed=["rgb","infrared"], rgb="Two people stand close together in the crosswalk; the cited light-topped person is among them.", infrared="Two upright warm human silhouettes stand close together at the matching crosswalk position.", uncertainty=["The RGB proposal covers the close pair, so the individual boundary within the group is not isolated."]),
"rgbt_flir_train_000034": dict(label="person near illuminated trees", ir=None, extent="whole", confirmed=[], rgb="At the cited sidewalk spot beside the lit trees, a possible person is too small and dark to separate confidently.", infrared="The corresponding area contains bright tree and street-side structures; no separate human contour is clear.", uncertainty=["Person identity and boundary are not visually resolved in either image."]),
"rgbt_flir_train_000035": dict(label="distant car", ir=None, extent="whole", confirmed=["rgb"], rgb="A tiny distant vehicle or headlight mark lies near the road vanishing point.", infrared="Several distant roadway heat spots overlap in this area, without a separable car boundary.", uncertainty=["The distant car is only a small indistinct mark; an independent IR box cannot be placed reliably."]),
"rgbt_flir_train_000046": dict(label="person near restaurant", ir=[0.472,0.402,0.544,0.62], extent="whole", confirmed=["rgb","infrared"], rgb="A person in the group beside the restaurant is visible near the center of the parking area.", infrared="A warm upright figure is visible in the corresponding central group by the building.", uncertainty=["Several people stand close together; the cited individual is partly adjacent to others."]),
"rgbt_flir_train_000047": dict(label="parked white SUV", ir=[0.022,0.42,0.17,0.54], extent="whole", confirmed=["rgb","infrared"], rgb="A white SUV is parked in the row along the left side of the lot.", infrared="Several parked vehicles are visible along the left edge; the cited SUV has a separate vehicle outline.", uncertainty=[]),
"rgbt_flir_train_000048": dict(label="bicycle by railing", ir=[0.887,0.43,0.998,0.573], extent="whole", confirmed=["rgb","infrared"], rgb="A bicycle is parked beside the railing at the far right.", infrared="A bicycle frame and wheel outlines are visible beside the right-side railing.", uncertainty=["The bicycle is close to the right image boundary."]),
"rgbt_m3fd_train_000016": dict(label="person by bushes", ir=[0.676,0.388,0.724,0.568], extent="whole", confirmed=["rgb","infrared"], rgb="A person stands beside the bushes at the lower right of the building.", infrared="A bright upright human silhouette is visible at the corresponding spot beside the vehicle.", uncertainty=[]),
"rgbt_m3fd_train_000017": dict(label="dark sedan", ir=[0.728,0.433,0.96,0.568], extent="whole", confirmed=["rgb","infrared"], rgb="A dark sedan is parked beside the building and partly screened by a bush.", infrared="A long low vehicle outline is visible next to the person and bushes.", uncertainty=["Bushes partly obscure the vehicle outline."]),
"rgbt_m3fd_train_000036": dict(label="gray sedan on road", ir=[0.45,0.39,0.598,0.546], extent="whole", confirmed=["rgb","infrared"], rgb="A gray sedan travels along the marked multi-lane road.", infrared="The nearer sedan has a distinct warm body and wheel outline on the roadway.", uncertainty=[]),
"rgbt_m3fd_train_000037": dict(label="distant roadside figure", ir=None, extent="whole", confirmed=[], rgb="A pin-sized mark lies beside distant vehicles near the right roadside; it is not clearly separable as a person.", infrared="Several tiny distant heat spots merge with vehicles and roadside structures; no independent figure contour is clear.", uncertainty=["The proposed figure is too small to identify confidently and nearly coincides with another source proposal."]),
"rgbt_m3fd_train_000038": dict(label="distant vehicle light", ir=None, extent="part", confirmed=["rgb"], rgb="A tiny light-like mark appears among distant roadway traffic.", infrared="Multiple distant vehicle heat spots are present, but this point cannot be isolated as a unique light target.", uncertainty=["Its box nearly overlaps the source proposal for a distant figure; target identity and separation remain uncertain."]),
"rgbt_m3fd_train_000062": dict(label="white SUV", ir=[0.787,0.506,0.98,0.695], extent="whole", confirmed=["rgb","infrared"], rgb="A white SUV is parked in the row along the right side of the road.", infrared="A separate SUV body and wheel contour is visible in the right-side row.", uncertainty=["The vehicle is close to the right frame edge."]),
"rgbt_m3fd_train_000063": dict(label="two people walking together", ir=[0.232,0.493,0.272,0.614], extent="group", confirmed=["rgb","infrared"], rgb="Two people walk together beside the building at the left side of the road.", infrared="Two adjacent upright warm human shapes are visible at the corresponding roadside position.", uncertainty=["The RGB proposal may include both adjacent people."]),
"rgbt_mfad_train_000004": dict(label="electric bicycle rider", ir=None, extent="group", confirmed=["rgb"], rgb="A small helmeted rider on an electric bicycle is near the bus stop on the right.", infrared="Several nearby human heat silhouettes overlap; the rider and bicycle outline cannot be isolated from the adjacent pedestrian.", uncertainty=["The source proposals for the rider and pedestrian overlap substantially; no distinct IR rider extent is reliable."]),
"rgbt_mfad_train_000005": dict(label="pedestrian by bus stop", ir=[0.658,0.603,0.701,0.691], extent="whole", confirmed=["rgb","infrared"], rgb="A pedestrian in dark clothing stands beside the bus stop near the rider.", infrared="An upright warm human silhouette is visible among the separate roadside figures.", uncertainty=["The pedestrian stands close to the rider, so their adjacent boundaries are not crisp."]),
"rgbt_mfad_train_000006": dict(label="red sedan", ir=[0.348,0.619,0.414,0.716], extent="whole", confirmed=["rgb","infrared"], rgb="A red sedan is among the vehicles near the bus stop lights.", infrared="A distinct vehicle body with bright rear lamps is visible in the matching lane area.", uncertainty=["Several vehicles are close together in the distance."]),
"rgbt_mfad_train_000007": dict(label="white bus", ir=[0.25,0.591,0.318,0.668], extent="whole", confirmed=["rgb","infrared"], rgb="A white bus is partly screened by reflections and roadside traffic.", infrared="A boxy vehicle silhouette is visible in the corresponding left-center roadway area.", uncertainty=["The vehicle is distant and partly mixed with nearby traffic."]),
"rgbt_mfad_train_000013": dict(label="small pedestrian near traffic lights", ir=[0.505,0.647,0.541,0.714], extent="whole", confirmed=["rgb","infrared"], rgb="A small standing figure is near the traffic lights around the center-left distance.", infrared="A compact bright upright human silhouette is visible at the corresponding road-side location.", uncertainty=["The figure is small and distant."]),
"rgbt_mfad_train_000014": dict(label="cyclist at right edge", ir=[0.968,0.655,1.0,0.786], extent="part", confirmed=["rgb","infrared"], rgb="A cyclist is partially cut off at the far right edge of the road.", infrared="A partial warm human-and-bicycle silhouette is visible at the far-right boundary.", uncertainty=["The cyclist is clipped by the image edge."]),
"rgbt_mfad_train_000015": dict(label="red sedan", ir=[0.638,0.674,0.865,0.936], extent="whole", confirmed=["rgb","infrared"], rgb="A red Audi sedan fills the foreground lane.", infrared="The foreground car has a clear broad body and rear-light contour.", uncertainty=[]),
"rgbt_mfad_train_000020": dict(label="small roadside person", ir=[0.035,0.56,0.091,0.692], extent="whole", confirmed=["rgb","infrared"], rgb="A small standing figure is beside the traffic barriers at the far left.", infrared="A bright upright human silhouette is visible at the corresponding far-left roadside position.", uncertainty=["The person is small and distant."]),
"rgbt_mfad_train_000021": dict(label="partially visible red bus", ir=None, extent="part", confirmed=["rgb"], rgb="A partially visible bus occupies the far-left edge under the overpass.", infrared="The far-left roadway contains overlapping vehicle heat signatures; the bus boundary is not independently separable.", uncertainty=["The bus is clipped by the image edge and overlaps roadside traffic in IR."]),
"rgbt_mfad_train_000022": dict(label="silver sedan", ir=[0.468,0.61,0.652,0.803], extent="whole", confirmed=["rgb","infrared"], rgb="A silver sedan with a visible rear license plate is under the overpass.", infrared="A centered sedan body and rear lights are clearly visible in the corresponding lane.", uncertainty=[]),
"rgbt_mfad_train_000028": dict(label="red sedan", ir=[0.37,0.82,0.545,0.975], extent="whole", confirmed=["rgb","infrared"], rgb="A red sedan is parked in the lower middle of the nighttime scene.", infrared="A car body and rear lights occupy the matching lower-middle position.", uncertainty=["Low-light background makes the outer edges slightly soft."]),
"rgbt_mfad_train_000029": dict(label="white pickup truck", ir=[0.60,0.76,0.77,0.945], extent="whole", confirmed=["rgb","infrared"], rgb="A light-colored pickup is parked to the right of the sedan.", infrared="A taller pickup silhouette with a distinct rear bed is visible on the right.", uncertainty=["The pickup is near other vehicles in the lower-right area."]),
}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--part",type=int,required=True)
    a=p.parse_args()
    src=json.loads(INPUT.read_text(encoding="utf-8-sig"))
    selected=src[(a.part-1)*2:a.part*2]
    if len(selected)!=2:
        raise ValueError(f"part {a.part} must contain exactly two source groups")
    rows=[]
    for bundle in selected:
        recs=bundle["original_records"]
        ids=[r["id"] for r in recs]
        if any(i not in META for i in ids):
            raise ValueError(f"missing visual notes for {bundle['id']}")
        refs={}
        nref=0
        for r in recs:
            if r["id"]==bundle["id"]:
                refs[r["id"]]="target"
            else:
                nref+=1
                refs[r["id"]]=f"reference{nref}"
        objects=[]
        queries=[]
        for j,r in enumerate(recs,1):
            m=META[r["id"]]
            objid=refs[r["id"]]
            objects.append({
                "object_id":objid,
                "role":"target" if objid=="target" else "reference",
                "semantic_label":m["label"],
                "boxes":{"rgb":r["bbox"],"infrared":m["ir"],"depth":None},
                "confirmed_modalities":m["confirmed"],
                "extent":m["extent"],
                "evidence":{"rgb":m["rgb"],"infrared":m["infrared"],"depth":"No depth image is provided for this RGB-IR source."},
                "uncertainties":m["uncertainty"],
            })
            mods=["rgb","infrared"] if m["ir"] is not None else ["rgb"]
            queries.append({
                "id":f"q{j:02d}",
                "origin_query_id":r["id"],
                "query":r["query"],
                "target_object_id":objid,
                "answerable_modalities":mods,
                "augmentation_eligible":False,
                "review_status":"provisional",
            })
        rows.append({
            "id":bundle["id"],"source":"rgbt","split":"train","scene_id":bundle["scene_id"],
            "images":bundle["images"],"depth_policy":"visual","review_status":"provisional","revision":0,
            "objects":objects,"queries":queries,"relations":[],
            "selection_outcome":"partial_auxiliary_evidence" if any(META[i]["ir"] is not None for i in ids) else "no_confirmed_auxiliary_evidence",
            "selection_note":"Original source queries and RGB boxes are preserved. Infrared boxes were estimated independently only where a corresponding contour was visually separable; otherwise they remain null. This source has no depth image.",
        })
    OUT.mkdir(parents=True,exist_ok=True)
    part=OUT/f"evidence_part{a.part:02d}.jsonl"
    if part.exists():
        raise FileExistsError(part)
    part.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows),encoding="utf-8")
    master=OUT/"object_evidence.jsonl"
    prior=[]
    if master.exists():
        prior=[json.loads(s) for s in master.read_text(encoding="utf-8-sig").splitlines() if s.strip()]
    old={x["id"] for x in prior}
    if old.intersection(x["id"] for x in rows):
        raise ValueError("duplicate bundle IDs in cumulative master")
    allrows=prior+rows
    master.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in allrows),encoding="utf-8")
    print(json.dumps({"part":a.part,"bundles":[x["id"] for x in rows],"objects":sum(len(x["objects"]) for x in rows),"queries":sum(len(x["queries"]) for x in rows),"master_bundles":len(allrows)},ensure_ascii=False))
if __name__=="__main__":
    main()
