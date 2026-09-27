"""Object evidence -> isolated review jobs -> existing native-SFT source rows.

Observed boxes are proposals. Review jobs never contain them; production export
requires a human/batch acceptance record as well as independent review results.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from tools.prepare_qwen3vl_native_sft import bbox_to_qwen1000

MODALITIES = ("rgb", "infrared", "depth")
ACCEPTED = {"human_accepted", "approved_batch"}


def read_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate evidence IDs: {path}")
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def validate_bundle(bundle: dict) -> None:
    if bundle["split"] not in {"train", "external_review"}:
        raise ValueError("explicit train/external_review split required")
    if not 1 <= len(bundle["objects"]) <= 4:
        raise ValueError("one target and at most three other objects")
    objects = {obj["object_id"]: obj for obj in bundle["objects"]}
    if len(objects) != len(bundle["objects"]):
        raise ValueError("duplicate object IDs")
    for modality, path in bundle["images"].items():
        if modality in (*MODALITIES, "depth_visual") and path and not Path(path).is_file():
            raise FileNotFoundError(path)
    if not bundle["images"].get("rgb"):
        raise ValueError("RGB is required")
    for obj in objects.values():
        bbox_to_qwen1000(obj["boxes"]["rgb"])
        for modality in MODALITIES:
            box = obj["boxes"].get(modality)
            if box is not None:
                bbox_to_qwen1000(box)
                if not bundle["images"].get(modality):
                    raise ValueError("a box needs its actual modality image")
        if obj.get("extent") not in {"whole", "part", "group"}:
            raise ValueError("object extent must be explicit")
    query_ids = [q["id"] for q in bundle["queries"]]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("duplicate query IDs")
    counts = Counter(q["target_object_id"] for q in bundle["queries"])
    if any(count > 2 for count in counts.values()):
        raise ValueError("at most two distinct natural queries per object")
    for query in bundle["queries"]:
        if query["target_object_id"] not in objects or not query["query"].strip():
            raise ValueError("query must name an existing target object")


def view_paths(bundle: dict) -> dict:
    images = {m: bundle["images"][m] for m in MODALITIES if bundle["images"].get(m)}
    if "depth" in images and bundle.get("depth_policy") != "visual":
        preview = bundle["images"].get("depth_visual")
        if not preview:
            raise ValueError("blind viewing raw numeric depth needs an explicit preview")
        images["depth"] = preview
    return images


def prepare_review(bundles: list[dict], output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs, private = [], []
    for bundle in bundles:
        validate_bundle(bundle)
        objects = {obj["object_id"]: obj for obj in bundle["objects"]}
        for query in bundle["queries"]:
            target = objects[query["target_object_id"]]
            for modality in MODALITIES:
                if target["boxes"].get(modality) is None:
                    continue
                case_id = f"{bundle['id']}::{query['id']}::{modality}"
                jobs.append({"query_id": case_id, "images": view_paths(bundle),
                             "query": query["query"], "output_modality": modality})
                private.append({"id": case_id, "bundle_id": bundle["id"],
                                "source_query_id": query["id"], "object_id": target["object_id"],
                                "modality": modality, "proposed_bbox": target["boxes"][modality]})
    if len({job["query_id"] for job in jobs}) != len(jobs):
        raise ValueError("duplicate review case ID")
    safe_dir = output_dir / "blind"
    safe_dir.mkdir(exist_ok=True)
    task = {"instructions": "Independently locate the Query target in output_modality coordinates. "
            "Return query_id, predicted_bbox_xyxy_normalized (or null), ambiguous, evidence_confirmed, "
            "review_note. Do not infer depth distance from display intensity. No annotation boxes are supplied.",
            "jobs": jobs}
    (safe_dir / "tasks.json").write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")
    write_rows(output_dir / "private_index.jsonl", private)
    return {"bundles": len(bundles), "review_cases": len(jobs), "safe_task": str(safe_dir / "tasks.json")}


def iou(a: list, b: list) -> float:
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def merge_review(bundles: list[dict], index: list[dict], reviews: list[dict]) -> tuple[list[dict], dict]:
    cases = {row["id"]: row for row in index}
    by_id = {row["query_id"]: row for row in reviews}
    if len(by_id) != len(reviews) or set(by_id) - set(cases):
        raise ValueError("duplicate or unknown review ID")
    result = json.loads(json.dumps(bundles))
    lookup = {bundle["id"]: bundle for bundle in result}
    passed = 0
    for case_id, case in cases.items():
        review = by_id.get(case_id)
        box = review.get("predicted_bbox_xyxy_normalized") if review else None
        if box is not None:
            bbox_to_qwen1000(box)
        score = iou(case["proposed_bbox"], box) if box is not None else None
        okay = bool(review and score is not None and score >= .5
                    and review.get("ambiguous") is False and review.get("evidence_confirmed") is True)
        passed += int(okay)
        bundle = lookup[case["bundle_id"]]
        query = next(q for q in bundle["queries"] if q["id"] == case["source_query_id"])
        query.setdefault("modality_reviews", {})[case["modality"]] = {
            "status": "blind_passed" if okay else "needs_review" if review else "pending",
            "iou": score, "review": review}
        if case["modality"] == "rgb":
            query["review_status"] = "blind_passed" if okay else "needs_review" if review else "pending"
    return result, {"cases": len(cases), "reviewed": len(by_id), "blind_passed": passed,
                    "human_approved": 0}


def export_rows(bundles: list[dict], approvals: dict, preview: bool = False) -> list[dict]:
    rows = []
    for bundle in bundles:
        validate_bundle(bundle)
        approval = approvals.get(bundle["id"])
        accepted_tasks = set(approval.get("tasks", [])) if approval else set()
        if bundle["split"] != "train":
            raise ValueError("external review data cannot enter training export")
        if not preview and (not approval or approval.get("status") not in ACCEPTED
                            or not approval.get("record") or "bbox" not in accepted_tasks):
            raise ValueError(f"human/batch acceptance missing: {bundle['id']}")
        objects = {o["object_id"]: o for o in bundle["objects"]}
        for query in bundle["queries"]:
            if not preview and query.get("review_status") not in {"blind_passed", "human_accepted"}:
                continue
            obj = objects[query["target_object_id"]]
            row = {"id": f"{bundle['id']}::{query['id']}", "source": bundle["source"],
                   "split": "train", "scene_id": bundle["scene_id"], "query": query["query"],
                   "bbox": obj["boxes"]["rgb"], "depth_policy": bundle.get("depth_policy", "visual"),
                   "review_status": "provisional" if preview else approval["status"],
                   "evidence_id": bundle["id"], "origin_query_id": query.get("origin_query_id"),
                   "task_pool": [], "augmentation_eligible": False}
            for modality in MODALITIES:
                if bundle["images"].get(modality):
                    row[modality] = bundle["images"][modality]
            for modality in ("infrared", "depth"):
                passed = query.get("modality_reviews", {}).get(modality, {}).get("status") == "blind_passed"
                if obj["boxes"].get(modality) is not None and passed and "cross_bbox" in accepted_tasks:
                    row["task_pool"].append({"id": f"{query['id']}:{modality}", "type": "cross_bbox",
                                            "query": query["query"], "target_modality": modality,
                                            "bbox": obj["boxes"][modality], "review_status": approval["status"]})
            for relation in bundle.get("relations", []):
                if relation.get("answer_object_id") != obj["object_id"] or relation.get("review_status") not in ACCEPTED:
                    continue
                if "relation" not in accepted_tasks:
                    continue
                candidates = [objects[oid] for oid in relation["candidate_object_ids"]]
                if not 2 <= len(candidates) <= 6 or len({c['object_id'] for c in candidates}) != len(candidates):
                    raise ValueError("relation needs distinct confirmed candidates")
                if relation.get("kind") == "nearfar" and relation.get("depth_order_verified") is not True:
                    raise ValueError("near/far relation requires independently verified depth order")
                row["task_pool"].append({"id": relation["id"], "type": "relation", "query": relation["query"],
                                        "candidates": [{"object_id": c["object_id"], "bbox": c["boxes"]["rgb"]} for c in candidates],
                                        "answer_object_id": obj["object_id"], "review_status": relation["review_status"]})
            # A reviewed natural query also supplies an instance-choice task.
            # Only independently confirmed objects enter the candidate bank;
            # the human relation/candidate acceptance explicitly covers this use.
            candidates = [o for o in objects.values() if "rgb" in o.get("confirmed_modalities", [])]
            if "relation" in accepted_tasks and 2 <= len(candidates) <= 6 and obj in candidates:
                row["task_pool"].append({"id": f"{query['id']}:candidates", "type": "relation",
                                        "query": query["query"], "answer_object_id": obj["object_id"],
                                        "candidates": [{"object_id": c["object_id"], "bbox": c["boxes"]["rgb"]} for c in candidates],
                                        "review_status": approval["status"]})
            augmentation = query.get("augmentation", {})
            if "augmentation" in accepted_tasks and augmentation.get("review_status") in ACCEPTED:
                row["augmentation_eligible"] = bool(augmentation.get("auxiliary_support_confirmed") is True
                                                    and augmentation.get("answer_survives") is True)
                row["augmentation_review_status"] = augmentation["review_status"]
            rows.append(row)
    return rows


def attach_city_tasks(native: list[dict], evidence_rows: list[dict]) -> list[dict]:
    """Attach accepted evidence by original ID, preserving City conversations."""
    lookup = {str(row["id"]): row for row in native}
    result = json.loads(json.dumps(native))
    output = {str(row["id"]): row for row in result}
    seen = set()
    for evidence in evidence_rows:
        key = str(evidence.get("origin_query_id"))
        if key not in lookup or key in seen:
            raise ValueError(f"missing or duplicate original City query ID: {key}")
        if evidence["source"] != "city" or evidence["review_status"] not in ACCEPTED:
            raise ValueError("City attachment requires accepted City evidence")
        seen.add(key)
        output[key]["task_pool"] = evidence["task_pool"]
        for field in ("augmentation_eligible", "augmentation_review_status", "evidence_id"):
            if field in evidence:
                output[key][field] = evidence[field]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare-review", "merge-review", "export"))
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-index", type=Path)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--approvals", type=Path, help="JSON mapping bundle ID to status, tasks and human review record")
    parser.add_argument("--preview", action="store_true", help="Provisional formatting only; cannot train")
    parser.add_argument("--city-native", type=Path, help="Attach accepted City task pools to original native conversations")
    args = parser.parse_args()
    bundles = read_rows(args.evidence)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "prepare-review":
        summary = prepare_review(bundles, args.output_dir)
    elif args.command == "merge-review":
        index = read_rows(args.private_index)
        reviews = [json.loads(s) for s in args.reviews.read_text(encoding="utf-8-sig").splitlines() if s.strip()]
        merged, summary = merge_review(bundles, index, reviews)
        write_rows(args.output_dir / "reviewed_evidence.jsonl", merged)
    else:
        approvals = json.loads(args.approvals.read_text(encoding="utf-8-sig")) if args.approvals else {}
        rows = export_rows(bundles, approvals, args.preview)
        write_rows(args.output_dir / ("preview_rows.jsonl" if args.preview else "train_rows.jsonl"), rows)
        summary = {"queries": len(rows), "auxiliary_tasks": sum(len(r['task_pool']) for r in rows),
                   "preview_only": args.preview, "augmented_presentations": 0}
        if args.city_native:
            native = json.loads(args.city_native.read_text(encoding="utf-8-sig"))
            city = attach_city_tasks(native, rows)
            (args.output_dir / "city_native_with_evidence.json").write_text(json.dumps(city, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
