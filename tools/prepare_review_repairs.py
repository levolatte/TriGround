"""Build an independent, review-only package for rejected A/B/V candidates."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from tools.prepare_triground_abv_data import _thumbnail, read_jsonl, render_review, write_jsonl


def _read_specs(paths: list[Path]) -> list[dict[str, Any]]:
    specs = []
    for path in paths:
        items = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(items, list):
            raise ValueError(f"spec file must contain a JSON list: {path}")
        specs.extend(items)
    return specs


def _validate_bbox(bbox: Any, task_id: str) -> list[float]:
    if (not isinstance(bbox, list) or len(bbox) != 4
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   for value in bbox)):
        raise ValueError(f"invalid normalized bbox for {task_id}: {bbox!r}")
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError(f"bbox is outside normalized image coordinates for {task_id}: {bbox!r}")
    return bbox


def _nonempty_text(spec: dict[str, Any], key: str, task_id: str) -> str:
    value = spec.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be nonempty for {task_id}")
    return value.strip()


def _validate_spec(spec: Any) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise ValueError("each repair spec must be a JSON object")
    task_id = spec.get("task_id")
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("each repair spec needs a nonempty task_id")
    if spec.get("action") not in {"repair", "drop"}:
        raise ValueError(f"action must be repair or drop for {task_id}")
    _nonempty_text(spec, "reason_zh", task_id)
    _nonempty_text(spec, "visual_evidence", task_id)
    inspected_paths = spec.get("inspected_paths")
    if (not isinstance(inspected_paths, list)
            or any(not isinstance(path, str) or not path.strip() for path in inspected_paths)):
        raise ValueError(f"inspected_paths must be a list of nonempty paths for {task_id}")
    if spec["action"] == "repair":
        _nonempty_text(spec, "query", task_id)
        _nonempty_text(spec, "query_zh", task_id)
    if spec.get("bbox") is not None:
        _validate_bbox(spec["bbox"], task_id)
    return spec


def prepare_review_repairs(source_output: Path, specs_paths: list[Path],
                           output: Path) -> dict[str, Any]:
    """Write repaired proposals and a fresh review atlas without touching source_output."""
    source_output = source_output.resolve()
    output = output.resolve()
    if output == source_output:
        raise FileExistsError("repair review needs a separate output directory")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"repair review output is not empty: {output}")

    source_rows = read_jsonl(source_output / "data/candidates.jsonl")
    candidates = {row["task_id"]: row for row in source_rows}
    if len(candidates) != len(source_rows):
        raise ValueError("duplicate candidate task ID in source package")

    with (source_output / "review/decisions.csv").open(
            encoding="utf-8-sig", newline="") as stream:
        decisions = list(csv.DictReader(stream))
    decision_by_id = {row["task_id"]: row for row in decisions}
    if len(decision_by_id) != len(decisions):
        raise ValueError("duplicate decision task ID")
    unknown_decisions = set(decision_by_id) - set(candidates)
    if unknown_decisions:
        raise ValueError(f"unknown decision task IDs: {sorted(unknown_decisions)[:5]}")
    rejected_ids = [row["task_id"] for row in decisions
                    if row.get("decision", "").strip().casefold() == "reject"]
    rejected_set = set(rejected_ids)

    specs = [_validate_spec(spec) for spec in _read_specs(specs_paths)]
    specs_by_id: dict[str, dict[str, Any]] = {}
    for spec in specs:
        task_id = spec["task_id"]
        if task_id in specs_by_id:
            raise ValueError(f"duplicate repair spec task ID: {task_id}")
        if task_id not in candidates:
            raise ValueError(f"unknown repair spec task ID: {task_id}")
        if task_id not in rejected_set:
            raise ValueError(f"repair spec is not for a rejected task: {task_id}")
        specs_by_id[task_id] = spec
    if set(specs_by_id) != rejected_set:
        missing = sorted(rejected_set - set(specs_by_id))
        extra = sorted(set(specs_by_id) - rejected_set)
        raise ValueError(f"repair specs must cover all rejected IDs; missing={missing}, extra={extra}")

    rows = []
    drops = []
    bbox_changed_count = 0
    for old in source_rows:
        spec = specs_by_id.get(old["task_id"])
        if spec is None:
            continue
        decision = decision_by_id[old["task_id"]]
        user_note = decision.get("note", "")
        if spec["action"] == "drop":
            drops.append({"task_id": old["task_id"], "action": "drop",
                          "user_note": user_note, "reason_zh": spec["reason_zh"],
                          "visual_evidence": spec["visual_evidence"],
                          "inspected_paths": spec["inspected_paths"]})
            continue

        row = dict(old)
        row["task_id"] = f"{old['task_id']}:repair1"
        row["bundle_id"] = f"{old['bundle_id']}:repair1"
        row["proposed_query"] = spec["query"].strip()
        row["query_zh"] = spec["query_zh"].strip()
        row["review_status"] = "draft_unapproved"
        previous_bbox = list(old["bbox"])
        new_bbox = previous_bbox if spec.get("bbox") is None else spec["bbox"]
        box_changed = new_bbox != old["bbox"]
        row["bbox"] = new_bbox
        repair_info = {
            "previous_task_id": old["task_id"],
            "previous_query": old["proposed_query"],
            "previous_bbox": previous_bbox,
            "user_note": user_note,
            "reason_zh": spec["reason_zh"],
            "box_changed": box_changed,
            "visual_evidence": spec["visual_evidence"],
            "inspected_paths": spec["inspected_paths"],
        }
        if box_changed:
            bbox_changed_count += 1
            previous_object_id = old["target_object_id"]
            row["target_object_id"] = f"{previous_object_id}:repair1"
            repair_info["object_id_provenance"] = {
                "previous_target_object_id": previous_object_id,
                "source_id_preserved": old["source_id"],
                "basis": "target identity follows the visually revised bbox",
            }
            image_path = output / "review/assets" / f"repair_previous_{bbox_changed_count:03d}.jpg"
            if not _thumbnail(Path(old["images"]["rgb"]), image_path, previous_bbox):
                raise FileNotFoundError(f"cannot render previous bbox image: {old['images']['rgb']}")
            repair_info["previous_answer_image"] = f"assets/{image_path.name}"
        row["repair_info"] = repair_info
        row["predecessor_task_ids"] = [old["task_id"]]
        rows.append(row)

    task_ids = [row["task_id"] for row in rows]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("duplicate repaired task ID")
    write_jsonl(output / "data/candidates.jsonl", rows)
    (output / "data/repair_specs.json").write_text(
        json.dumps(specs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    atlas = render_review(rows, output)
    audit = {
        "source_output": str(source_output),
        "source_candidates": len(source_rows),
        "source_decisions": len(decisions),
        "source_rejected_ids": rejected_ids,
        "spec_files": [str(path.resolve()) for path in specs_paths],
        "repair_ids": [row["repair_info"]["previous_task_id"] for row in rows],
        "drop_records": drops,
        "bbox_changed_ids": [row["repair_info"]["previous_task_id"] for row in rows
                             if row["repair_info"]["box_changed"]],
        "original_decisions_migrated": False,
    }
    summary = {
        "status": "draft_unapproved",
        "rejected_candidates": len(rejected_ids),
        "repair_candidates": len(rows),
        "dropped_candidates": len(drops),
        "bbox_changed_candidates": bbox_changed_count,
        "review": atlas,
        "release_created": False,
    }
    (output / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--specs", type=Path, nargs="+", required=True,
                        help="one or more JSON lists of repair specs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare_review_repairs(args.source_output, args.specs, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
