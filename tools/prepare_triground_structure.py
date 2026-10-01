"""Freeze R/S inputs from reviewed mothers and original floating City labels."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import random
import zipfile

from tools.prepare_triground_tm_data import _bbox_task
from tools.prepare_triground_abv_data import city_raw_labels, CITY, CITY_RAW_ZIP
from tools.prepare_qwen3vl_native_sft import native_prompt


ROOT = Path(__file__).resolve().parents[2]
TM = ROOT/"results/triground_tm_20260929/data/release_600_seed2028"
REVISION = ROOT/"results/triground_revision_20260929"
CLOUD = "/root/autodl-tmp/rematch_20260922"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_lines(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8-sig").splitlines() if x]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def cloud_path(path):
    path = str(path).replace("\\", "/")
    if path.startswith("F:/AIC/"):
        return CLOUD + path[len("F:/AIC"):]
    return path


def cloud_rows(rows):
    rows = deepcopy(rows)
    for row in rows:
        row["image"] = [cloud_path(p) for p in row["image"]]
        if row.get("depth_raw"):
            row["depth_raw"] = cloud_path(row["depth_raw"])
    return rows


def structural_row(sample, gt, *, source=None, all_sources=(), raw_city=None, ir_review=None):
    row = {k: deepcopy(v) for k,v in sample.items() if k not in ("conversations", "expected_answer")}
    row["prompt"] = sample["conversations"][0]["value"]
    row["rgb_gt_bbox"] = list(gt)
    row["depth_encoding"] = "millimeter" if source and source.get("depth_policy") in ("millimeter", "city_mm") else "unknown"
    row["depth_raw"] = source.get("images", {}).get("depth_raw") if source else None
    supervision = {"rgb_objects": [list(gt)], "bindings": {}}
    # City/Robo ordinary task boxes do not certify thermal/unknown-depth identities.
    if source and source.get("relation"):
        relation, evidence = source["relation"], source["depth_evidence"]["by_source_id"]
        target = relation["target_id"]
        references = [relation["reference_id"]] if relation.get("reference_id") else list(relation["competitors"])
        if len(references) > 2:
            raise ValueError("relation has more than two reference subjects")
        rgb_refs = [raw_city[x]["bbox"] for x in references if x in raw_city]
        supervision["rgb_objects"] += rgb_refs
        supervision["bindings"]["rgb"] = {"target": list(gt), "references": rgb_refs,
                                                 "unordered": not bool(relation.get("reference_id"))}
        supervision["bindings"]["depth"] = {"target": evidence[target]["region"],
            "references": [evidence[x]["region"] for x in references], "unordered": not bool(relation.get("reference_id"))}
    if source and source.get("opponent_object_id"):
        opponents = [s["bbox"] for s in all_sources if s.get("target_object_id") == source["opponent_object_id"]]
        if opponents:
            supervision["rgb_objects"].append(opponents[0])
            supervision["bindings"]["rgb"] = {"target": list(gt), "references": [opponents[0]], "unordered": False}
    # An IR-only rewritten query is NOT a joint input; only its verified box is reused.
    if ir_review and ir_review.get("accepted") and ir_review["split"] == "train":
        supervision["bindings"]["ir"] = {"target": ir_review["ir_bbox"], "references": []}
    # Never bind objects in a synthetic blank/low-quality auxiliary view.
    degraded = row.get("degraded_modality")
    for modality in {*row.get("missing_modalities_actual", []), degraded}:
        supervision["bindings"].pop(modality, None)
    row["supervision"] = supervision if row["split"] == "train" else {}
    return row


def scene_round_robin(rows, count, seed):
    groups = defaultdict(list)
    for row in rows:
        groups[Path(row["image"][0]).stem].append(row)
    rng = random.Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    for group in groups.values():
        rng.shuffle(group)
    selected = []
    while len(selected) < count:
        for key in keys:
            if groups[key]:
                selected.append(groups[key].pop())
                if len(selected) == count:
                    return selected
        if not any(groups.values()):
            raise ValueError("insufficient distinct City queries after holdout exclusions")
    return selected


def city_row(row, raw, split="train"):
    stem = Path(row["image"][0]).stem
    cloud_root = CLOUD+"/data/city/train/"
    paths = [cloud_root+p for p in row["image"]]
    sample = {"id": "structure:"+row["id"], "source_task_id": row["id"], "mother_id": row["id"],
              "source_id": row["id"], "scene_id": "city:"+stem, "source": "city", "split": split,
              "task_category": "old_city", "modalities": ["rgb","ir","depth"], "image": paths,
              "query": raw["query"], "prompt": row["conversations"][0]["value"],
              "rgb_gt_bbox": raw["bbox"], "depth_encoding": "millimeter",
              "depth_raw": cloud_root+"depth/"+stem+".png", "missing_modalities_actual": [],
              "requested_condition": "normal", "supervision": {}}
    # Confirm the source snapshot uses the original/corrected query, not a rounded GT.
    if raw["query"] not in sample["prompt"]:
        raise ValueError(f"City query mismatch: {row['id']}")
    return sample


def build(output, seed=2030):
    output = Path(output)
    if (output/"release.json").exists():
        raise FileExistsError("use a new release directory instead of rewriting frozen inputs")
    release = read(TM/"release.json")
    sources = read_lines(TM/"source_reviews.jsonl")
    by_id = {r["task_id"]:r for r in sources}
    ir_reviews = {r["task_id"]:r for r in read_lines(TM/"ir_reviews.jsonl")}
    old_special = read(REVISION/"data/verified_normal/train.json")
    selected = [by_id[r["source_task_id"]] for r in old_special]
    raw = city_raw_labels()
    diagnostics = read(TM/"diagnostics/normal.json") + read(REVISION/"data/rgb_ir_controls/normal.json")
    excluded_locations = {"000003","000005","000016"} | {
        by_id[r["source_task_id"]]["location_group"] for r in diagnostics
        if r["source_task_id"] in by_id and by_id[r["source_task_id"]]["source"] == "city"}
    excluded_scenes = {r["scene_id"] for r in diagnostics}
    excluded_ids = {r["source_id"] for r in selected}
    with zipfile.ZipFile(CITY_RAW_ZIP) as archive:
        validation = json.loads(archive.read("qwen_generation_val.json"))
    excluded_stems = {Path(r["visible"]).stem for r in validation.values()}
    city_pool = [r for r in read(CITY) if Path(r["image"][0]).stem.split("_")[0] not in excluded_locations
                 and "city:"+Path(r["image"][0]).stem not in excluded_scenes
                 and Path(r["image"][0]).stem not in excluded_stems and r["id"] not in excluded_ids]
    chosen_city = scene_round_robin(city_pool, 740, seed)
    city = [city_row(r, raw[r["id"]]) for r in chosen_city]
    variants, variant_ids = [], {}
    for source in selected:
        conditions = [(None,"normal")]
        if source["category"] == "reliability":
            if "infrared" in source["images"]:
                conditions.append(("infrared","reliability_ir_low"))
            if "depth" in source["images"]:
                conditions.append(("depth","reliability_depth_low"))
        for low, condition in conditions:
            sample = _bbox_task(source, "structure:"+source["task_id"]+":"+condition,
                               output/"assets", low=low, condition=condition)
            row = structural_row(sample, source["bbox"], source=source, all_sources=sources,
                                 raw_city=raw, ir_review=ir_reviews.get(source["task_id"]))
            row["mother_id"] = source["task_id"]
            variants.append(row)
            variant_ids[(source["task_id"],condition)] = row["id"]
    rows = city + variants
    all_ids = {r["id"] for r in rows}
    for steps, repeat, city_count in ((400,8,740),(200,4,370)):
        order = [r["id"] for r in city[:city_count] for _ in range(2)]
        for source in selected:
            normal = variant_ids[(source["task_id"],"normal")]
            if source["category"] != "reliability":
                order.extend([normal]*repeat)
            else:
                extra = [variant_ids[(source["task_id"], c)] for c in ("reliability_ir_low","reliability_depth_low")
                         if (source["task_id"],c) in variant_ids]
                order.extend([normal]*(repeat//2))
                for index in range(repeat//2):
                    order.append(extra[index%len(extra)])
        random.Random(seed).shuffle(order)
        if len(order) != steps*8 or not set(order) <= all_ids:
            raise ValueError("schedule does not match the frozen horizon")
        write(output/f"schedule_{steps}.json", order)
    write(output/"train_inputs.json", rows)
    write(output/"train_inputs.cloud.json", cloud_rows(rows))
    # Two known ambiguous descriptions are fixed against the already-inspected RGB.
    query_fixes = {
        "rgbt_mfad_train_011699:ir_complement": ("The light-colored car directly ahead in the camera vehicle's lane, to the right of the red car.",
            "本车车道正前方、位于红车右侧的浅色汽车。"),
        "robo:0004596:diag_aux": ("The black computer mouse with the green indicator light, at the lower left of the white container.",
            "白色容器左下方、带绿色指示灯的黑色电脑鼠标。"),
    }
    gt103 = read(release["gt_manifest"])
    control_gt = read(REVISION/"data/rgb_ir_controls/gt.json")
    diag = []
    corrections = []
    for original in diagnostics:
        sample = deepcopy(original)
        source = by_id.get(sample["source_task_id"])
        if sample["source_task_id"] in query_fixes:
            query, zh = query_fixes[sample["source_task_id"]]
            corrections.append({"id":sample["id"],"original_query":sample["query"],"query":query,"query_zh":zh,"gt_changed":False})
            sample["query"] = query
            sample["query_zh"] = zh
            sample["conversations"][0]["value"] = native_prompt(query, ["infrared" if m=="ir" else m for m in sample["modalities"]], source["depth_policy"])
        entry = gt103.get(sample["id"], control_gt.get(sample["id"]))
        gt = entry["bbox"] if isinstance(entry, dict) else entry
        row = structural_row(sample, gt, source=source, raw_city=raw)
        row["task_category"] = "rgb_sufficient_ir" if sample["id"].startswith("rgb-ir-control:") else row["task_category"]
        diag.append(row)
    if len({r['id'] for r in diag}) != 119 or {r['scene_id'] for r in rows} & {r['scene_id'] for r in diag}:
        raise ValueError("diagnostic overlap or unexpected denominator")
    write(output/"diagnostic_normal.json",diag)
    write(output/"diagnostic_normal.cloud.json",cloud_rows(diag))
    write(output/"diagnostic_query_corrections.json",corrections)
    summary = {"status":"inputs_ready_proposals_pending","seed":seed,"steps":[400,200],
        "unique_city":len(city),"special_mothers":len(selected),"unique_train_conditions":len(rows),
        "special_categories":dict(Counter(r['category'] for r in selected)),"diagnostic":len(diag),
        "excluded_city_locations":sorted(excluded_locations),"raw_gt_source":str(CITY_RAW_ZIP),
        "ir_binding_mothers":sum(r['task_id'] in ir_reviews and ir_reviews[r['task_id']]['accepted'] and ir_reviews[r['task_id']]['split']=='train' for r in selected),
        "train_inputs":str(output/'train_inputs.json'),"cloud_train_inputs":str(output/'train_inputs.cloud.json'),
        "diagnostic_normal":str(output/'diagnostic_normal.json'),"new_gpu_budget_seconds":43200}
    write(output/"release.json",summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(build(args.output_dir),ensure_ascii=False,indent=2))
