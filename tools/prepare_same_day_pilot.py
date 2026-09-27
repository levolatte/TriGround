"""Build pending RGBDT pilot candidates; export native samples only after human decisions."""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

from tools.prepare_multimodal_evidence import read_rows, write_rows
from tools.prepare_next_stage_data import _native_sample, _source_row
from tools.prepare_qwen3vl_native_sft import bbox_to_qwen1000


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/multimodal_data_20260925/merged_20260926_trial200_final/reviewed_evidence.jsonl"
ACCEPTANCE = ROOT / "results/multimodal_data_20260925/human_acceptance"
NATURAL = ACCEPTANCE / "natural100_seed2026/decisions.csv"
CORRESPONDENCE = ACCEPTANCE / "correspondence100_seed2026/decisions_blank.csv"
OUT = ROOT / "results/same_day_multimodal_pilot_20260926"
MODALITIES = ("rgb", "infrared", "depth")


def csv_by_id(path: Path, key: str = "case_id") -> dict[str, dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    ids = [row[key] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate {key}: {path}")
    return {row[key]: row for row in rows}


def candidates() -> tuple[list[dict], dict]:
    natural = csv_by_id(NATURAL)
    correspondence = csv_by_id(CORRESPONDENCE)
    source = read_rows(SOURCE)
    tasks = []
    seen = set()
    for bundle in source:
        objects = {o["object_id"]: o for o in bundle["objects"]}
        for query in bundle["queries"]:
            target = objects[query["target_object_id"]]
            rgb_passed = query.get("modality_reviews", {}).get("rgb", {}).get("status") == "blind_passed"
            if query.get("review_status") != "blind_passed" or not rgb_passed:
                continue
            for modality in MODALITIES:
                review = query.get("modality_reviews", {}).get(modality, {})
                box = target["boxes"].get(modality)
                if review.get("status") != "blind_passed" or box is None:
                    continue
                bbox_to_qwen1000(box)
                if review["review"]["query_id"] != f"{bundle['id']}::{query['id']}::{modality}":
                    raise ValueError("review ID does not match candidate")
                if any(not Path(bundle["images"][m]).is_file() for m in MODALITIES) or not Path(bundle["images"]["depth_visual"]).is_file():
                    raise FileNotFoundError(bundle["id"])
                key = (bundle["id"], query.get("origin_query_id") or query["id"], modality)
                if key in seen:
                    continue
                seen.add(key)
                natural_id = f"{bundle['id']}::{query['id']}"
                correspondence_id = f"{bundle['id']}::{target['object_id']}::{modality}" if modality != "rgb" else None
                task = {
                    "task_id": f"{bundle['id']}::{query['id']}::{modality}", "task_type": "direct_rgb_bbox" if modality == "rgb" else "auxiliary_cross_bbox",
                    "status": "pending_human", "query_id": query["id"], "origin_query_id": key[1],
                    "bundle_id": bundle["id"], "object_id": target["object_id"], "source": bundle["source"],
                    "split": bundle["split"], "scene_id": bundle["scene_id"], "query": query["query"],
                    "output_modality": modality, "input_modalities": list(MODALITIES),
                    "task_instruction": f"Use the RGB, infrared and depth views to locate the query target; return its bbox in {modality} coordinates.",
                    "images_original": {m: bundle["images"][m] for m in MODALITIES},
                    "depth_preview_review_only": bundle["images"]["depth_visual"],
                    "depth_policy": bundle["depth_policy"], "bbox_xyxy_normalized": box,
                    "rgb_gt_bbox_xyxy_normalized": target["boxes"]["rgb"],
                    "blind_review": review, "natural_case_id": natural_id,
                    "natural_existing_item": natural_id in natural,
                    "correspondence_case_id": correspondence_id,
                    "correspondence_existing_item": correspondence_id in correspondence if correspondence_id else None,
                    "source_evidence": str(SOURCE),
                }
                tasks.append(task)
    tasks.sort(key=lambda t: (t["bundle_id"], t["origin_query_id"], MODALITIES.index(t["output_modality"])))
    summary = {
        "source_bundles": len(source), "source_queries": sum(len(b["queries"]) for b in source),
        "source_blind_passed_by_modality": {m: sum(q.get("modality_reviews", {}).get(m, {}).get("status") == "blind_passed" for b in source for q in b["queries"]) for m in MODALITIES},
        "candidate_tasks": len(tasks), "candidate_bundles": len({t["bundle_id"] for t in tasks}),
        "candidate_by_modality": dict(Counter(t["output_modality"] for t in tasks)),
        "candidate_unique_origin_queries": len({(t["bundle_id"], t["origin_query_id"]) for t in tasks}),
        "natural_existing_items_linked": sum(t["natural_existing_item"] for t in tasks),
        "correspondence_existing_items_linked": sum(t["correspondence_existing_item"] is True for t in tasks),
        "existing_acceptance_coverage_by_modality": {
            m: sum(t["output_modality"] == m and t["natural_existing_item"]
                   and (m == "rgb" or t["correspondence_existing_item"] is True) for t in tasks)
            for m in MODALITIES},
        "existing_acceptance_coverage_tasks": sum(t["natural_existing_item"]
            and (t["output_modality"] == "rgb" or t["correspondence_existing_item"] is True) for t in tasks),
        "supplemental_natural_items": sum(t["output_modality"] == "rgb" and not t["natural_existing_item"] for t in tasks),
        "supplemental_correspondence_items": sum(t["output_modality"] != "rgb" and not t["correspondence_existing_item"] for t in tasks),
        "human_accepted": 0, "training_released": 0,
    }
    return tasks, summary


def build() -> None:
    tasks, summary = candidates()
    OUT.mkdir(parents=True, exist_ok=True)
    write_rows(OUT / "pending_candidates.jsonl", tasks)
    fields = ["task_id", "natural_decision", "correspondence_decision", "note"]
    supplemental_path = OUT / "supplemental_decisions.csv"
    if not supplemental_path.exists():
        with supplemental_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for task in tasks:
                if (task["output_modality"] == "rgb" and not task["natural_existing_item"]
                        or task["output_modality"] != "rgb" and not task["correspondence_existing_item"]):
                    writer.writerow({"task_id": task["task_id"], "natural_decision": "", "correspondence_decision": "", "note": ""})
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


def export(natural_path: Path, correspondence_path: Path, supplemental_path: Path) -> None:
    tasks = [json.loads(line) for line in (OUT / "pending_candidates.jsonl").read_text(encoding="utf-8").splitlines() if line]
    natural = csv_by_id(natural_path)
    correspondence = csv_by_id(correspondence_path)
    supplemental = csv_by_id(supplemental_path, "task_id")
    source = {b["id"]: b for b in read_rows(SOURCE)}
    accepted = []
    pending = Counter()
    for task in tasks:
        extra = supplemental.get(task["task_id"], {})
        rgb_task_id = f"{task['bundle_id']}::{task['query_id']}::rgb"
        n = natural[task["natural_case_id"]]["decision"] if task["natural_existing_item"] else supplemental.get(rgb_task_id, {}).get("natural_decision", "")
        c = (correspondence[task["correspondence_case_id"]]["decision"] if task["correspondence_existing_item"]
             else extra.get("correspondence_decision", "")) if task["output_modality"] != "rgb" else "accept"
        if n == "" or c == "":
            pending["pending"] += 1
            continue
        if n != "correct_unique" or c != "accept":
            pending["rejected"] += 1
            continue
        accepted.append(task)
    if not accepted:
        raise ValueError("no accepted tasks; no export written")
    native = []
    metadata = []
    cache = {}
    for index, task in enumerate(accepted):
        bundle = source[task["bundle_id"]]
        row = {"id": f"{task['bundle_id']}::{task['origin_query_id']}", "source": "rgbdt", "split": "train",
               "scene_id": bundle["scene_id"], "query": task["query"],
               "bbox": task["rgb_gt_bbox_xyxy_normalized"], "review_status": "human_accepted",
               "rgb": task["images_original"]["rgb"], "infrared": task["images_original"]["infrared"],
               "depth": task["images_original"]["depth"], "depth_policy": task["depth_policy"]}
        prepared = _source_row(row, "rgbdt", SOURCE, ROOT)
        aux = None if task["output_modality"] == "rgb" else {
            "id": task["task_id"], "query": task["query"], "target_modality": task["output_modality"],
            "bbox": task["bbox_xyxy_normalized"], "review_status": "human_accepted"}
        sample, sidecar = _native_sample(prepared, index + 1, OUT / "released", cache,
                                          "bbox" if aux is None else "cross_bbox", random.Random(2026), aux)
        if aux is None:
            sample["conversations"][0]["value"] = sample["conversations"][0]["value"].replace(
                "Return only JSON in this exact form", "Return the box in RGB image coordinates. Return only JSON in this exact form")
        sample["id"] = task["task_id"]
        sidecar["id"] = task["task_id"]
        native.append(sample)
        metadata.append(sidecar)
    released = OUT / "released"
    write_rows(released / "native_sft.jsonl", native)
    write_rows(released / "metadata.jsonl", metadata)
    (released / "summary.json").write_text(json.dumps({"released": len(native), "rejected": pending["rejected"], "still_pending": pending["pending"]}, indent=2) + "\n", encoding="utf-8")
    print(f"released {len(native)} accepted tasks; {pending['pending']} still pending")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "export"))
    parser.add_argument("--natural-decisions", type=Path, default=NATURAL)
    parser.add_argument("--correspondence-decisions", type=Path, default=CORRESPONDENCE)
    parser.add_argument("--supplemental-decisions", type=Path, default=OUT / "supplemental_decisions.csv")
    args = parser.parse_args()
    build() if args.command == "build" else export(args.natural_decisions, args.correspondence_decisions, args.supplemental_decisions)
