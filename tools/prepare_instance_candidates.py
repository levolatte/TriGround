"""Build same-image, same-class known-object candidate boxes.

The input is a manifest ZIP.  A candidate group is keyed only by the complete
``(visible, infrared, depth, class_name)`` tuple.  ``sequence_id`` is neither
required nor consulted: two images from one sequence must never share a
candidate pool.

The generated JSON keeps the source record fields and adds ``positive_bbox``
and ``negative_bboxes``.  The latter contains distinct known boxes from the
same image/class group whose IoU with the positive box is below the threshold;
an empty list is valid when the image has no other known box.
"""

from __future__ import annotations

import argparse
import json
import math
import ntpath
import posixpath
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any


NEW_TRAIN_MEMBER = "qwen_generation_train_100.json"
NEW_VAL_MEMBER = "qwen_generation_val.json"
OLD_TRAIN_MEMBER = "target_v2/manual_split/train_100.json"
OLD_VAL_MEMBER = "target_v2/manual_split/val.json"

PathBox = tuple[float, float, float, float]
GroupKey = tuple[str, str, str, str]


def _read_member(archive: zipfile.ZipFile, member: str) -> dict[str, dict[str, Any]]:
    payload = json.loads(archive.read(member).decode("utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{member} must contain a sample-id -> record object")
    for sample_id, record in payload.items():
        if not isinstance(sample_id, str) or not isinstance(record, dict):
            raise ValueError(f"{member} contains an invalid record for {sample_id!r}")
    return payload


def _relative_path(value: Any, *, sample_id: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{sample_id}: {field} must be a non-empty relative path")
    path = value.replace("\\", "/")
    if path.startswith("/") or ntpath.isabs(path):
        raise ValueError(f"{sample_id}: {field} must stay relative: {value!r}")
    return posixpath.normpath(path)


def _bbox(value: Any, *, sample_id: str) -> PathBox:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{sample_id}: bbox must contain four values")
    box = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in box):
        raise ValueError(f"{sample_id}: bbox contains a non-finite value")
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError(f"{sample_id}: bbox is not a valid normalized xyxy box: {value!r}")
    return box


def _iou(left: PathBox, right: PathBox) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / (left_area + right_area - intersection)


def _canonical_record(sample_id: str, source: dict[str, Any]) -> dict[str, Any]:
    required = ("visible", "infrared", "depth", "query", "bbox", "class_name")
    missing = [field for field in required if field not in source]
    if missing:
        raise ValueError(f"{sample_id}: missing required fields {missing}")
    if not isinstance(source["query"], str) or not source["query"].strip():
        raise ValueError(f"{sample_id}: query must be a non-empty string")
    if not isinstance(source["class_name"], str) or not source["class_name"].strip():
        raise ValueError(f"{sample_id}: class_name must be a non-empty string")

    record = dict(source)
    for field in ("visible", "infrared", "depth"):
        record[field] = _relative_path(source[field], sample_id=sample_id, field=field)
    record["bbox"] = list(_bbox(source["bbox"], sample_id=sample_id))
    record["positive_bbox"] = list(record["bbox"])
    return record


def _group_key(record: dict[str, Any]) -> GroupKey:
    return (
        str(record["visible"]),
        str(record["infrared"]),
        str(record["depth"]),
        str(record["class_name"]),
    )


def build_candidates(
    source: dict[str, dict[str, Any]], *, iou_threshold: float = 0.5
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Return one candidate record per source query and aggregate statistics."""

    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be between 0 and 1")

    records = {
        sample_id: _canonical_record(sample_id, source[sample_id])
        for sample_id in sorted(source)
    }
    groups: defaultdict[GroupKey, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for sample_id, record in records.items():
        groups[_group_key(record)].append((sample_id, record))

    output: dict[str, dict[str, Any]] = {}
    duplicate_occurrences = 0
    overlap_excluded = 0
    candidate_edges = 0
    records_with_candidates = 0
    groups_with_distinct_boxes = 0
    groups_with_overlap_exclusions = 0
    max_candidates = 0

    for group_key in sorted(groups):
        rows = groups[group_key]
        distinct_boxes = sorted(
            {_bbox(record["bbox"], sample_id=sample_id) for sample_id, record in rows}
        )
        duplicate_occurrences += len(rows) - len(distinct_boxes)
        if len(distinct_boxes) > 1:
            groups_with_distinct_boxes += 1

        group_overlap_excluded = 0
        for sample_id, record in rows:
            positive = tuple(record["positive_bbox"])
            negatives: list[list[float]] = []
            for candidate in distinct_boxes:
                if candidate == positive:
                    continue
                if _iou(positive, candidate) >= iou_threshold:
                    overlap_excluded += 1
                    group_overlap_excluded += 1
                    continue
                negatives.append(list(candidate))
            record["negative_bboxes"] = negatives
            record["candidate_count"] = len(negatives)
            output[sample_id] = record
            candidate_edges += len(negatives)
            if negatives:
                records_with_candidates += 1
            max_candidates = max(max_candidates, len(negatives))
        if group_overlap_excluded:
            groups_with_overlap_exclusions += 1

    summary = {
        "records": len(records),
        "complete_image_class_groups": len(groups),
        "groups_with_multiple_known_boxes": sum(
            len(rows) > 1 for rows in groups.values()
        ),
        "groups_with_distinct_known_boxes": groups_with_distinct_boxes,
        "records_with_negative_candidates": records_with_candidates,
        "records_without_negative_candidates": len(records) - records_with_candidates,
        "unique_known_bboxes": len(
            {(group_key, tuple(record["bbox"])) for group_key, rows in groups.items() for _, record in rows}
        ),
        "duplicate_bbox_occurrences_removed": duplicate_occurrences,
        "overlap_bboxes_excluded": overlap_excluded,
        "negative_candidate_edges": candidate_edges,
        "max_negative_candidates_per_query": max_candidates,
        "iou_threshold": iou_threshold,
        "group_key_fields": ["visible", "infrared", "depth", "class_name"],
        "sequence_id_used_for_grouping": False,
        "empty_negative_list_is_allowed": True,
    }
    return output, summary


def confirmed_instance_task(record: dict[str, Any]) -> dict[str, Any]:
    """Use explicitly verified same-frame objects, never legacy class_name.

    The annotation author must identify each object and the referred target.
    The returned task is private training metadata; the model sees only boxes
    and shuffled letters generated by the mixed-data converter.
    """
    if record.get("composite_target"):
        raise ValueError("composite target cannot be used as a single-ID task")
    objects = record["confirmed_objects"]
    if not 2 <= len(objects) <= 6:
        raise ValueError("instance choice requires 2-6 confirmed objects")
    if not all(obj.get("confirmed") is True for obj in objects):
        raise ValueError("every instance candidate must be explicitly confirmed")
    ids = [str(obj["object_id"]) for obj in objects]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate object IDs in confirmed instance task")
    target = str(record["target_object_id"])
    if target not in ids:
        raise ValueError("target_object_id is absent from confirmed candidates")
    boxes = [_bbox(obj["bbox"], sample_id=obj["object_id"]) for obj in objects]
    for index, box in enumerate(boxes):
        for other in boxes[index + 1 :]:
            if _iou(box, other) >= 0.5:
                raise ValueError("overlapping confirmed objects are ambiguous for the first ID task")
    return {
        "candidates": [
            {"object_id": object_id, "bbox": list(box)}
            for object_id, box in zip(ids, boxes, strict=True)
        ],
        "positive_object_id": target,
    }


def _image_id(value: str) -> str:
    """Compare image names across manifests with different relative prefixes."""

    path = PurePosixPath(value.replace("\\", "/"))
    parts = list(path.parts)
    if "visible" in parts:
        return "/".join(parts[parts.index("visible") + 1 :])
    return path.name


def _split_summary(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "records": len(records),
        "unique_visible_images": len({_image_id(str(record["visible"])) for record in records.values()}),
        "unique_image_class_groups": len({_group_key(_canonical_record(sample_id, record)) for sample_id, record in records.items()}),
        "unique_query_texts": len({str(record.get("query")) for record in records.values()}),
        "unique_exact_bboxes": len(
            {tuple(float(value) for value in record["bbox"]) for record in records.values()}
        ),
        "classes": dict(sorted(Counter(str(record.get("class_name")) for record in records.values()).items())),
        "review_decisions": dict(sorted(Counter(str(record.get("review_decision")) for record in records.values()).items())),
        "query_field_present": sum("query" in record for record in records.values()),
        "bbox_field_present": sum("bbox" in record for record in records.values()),
    }


def compare_manifests(
    new_splits: dict[str, dict[str, dict[str, Any]]],
    old_splits: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Summarize old/new counts and label differences without using old data to fill new."""

    new_all = {sample_id: record for split in ("train", "val") for sample_id, record in new_splits[split].items()}
    old_all = {sample_id: record for split in ("train", "val") for sample_id, record in old_splits[split].items()}
    common_ids = sorted(set(new_all) & set(old_all))
    old_only = sorted(set(old_all) - set(new_all))
    new_only = sorted(set(new_all) - set(old_all))
    new_split_by_id = {sample_id: split for split, rows in new_splits.items() for sample_id in rows}
    old_split_by_id = {sample_id: split for split, rows in old_splits.items() for sample_id in rows}

    changes = Counter()
    for sample_id in common_ids:
        new_record = new_all[sample_id]
        old_record = old_all[sample_id]
        if new_record.get("query") != old_record.get("query"):
            changes["query"] += 1
        if tuple(float(value) for value in new_record["bbox"]) != tuple(float(value) for value in old_record["bbox"]):
            changes["bbox"] += 1
        if new_record.get("class_name") != old_record.get("class_name"):
            changes["class_name"] += 1
        new_paths = tuple(_image_id(str(new_record[field])) for field in ("visible", "infrared", "depth"))
        old_paths = tuple(_image_id(str(old_record[field])) for field in ("visible", "infrared", "depth"))
        if new_paths != old_paths:
            changes["modal_paths"] += 1
        if new_split_by_id[sample_id] != old_split_by_id[sample_id]:
            changes["split"] += 1

    new_queries = {str(record.get("query")) for record in new_all.values()}
    old_queries = {str(record.get("query")) for record in old_all.values()}
    new_boxes = {tuple(float(value) for value in record["bbox"]) for record in new_all.values()}
    old_boxes = {tuple(float(value) for value in record["bbox"]) for record in old_all.values()}
    new_images = {_image_id(str(record["visible"])) for record in new_all.values()}
    old_images = {_image_id(str(record["visible"])) for record in old_all.values()}

    return {
        "new_splits": {split: _split_summary(rows) for split, rows in new_splits.items()},
        "old_splits": {split: _split_summary(rows) for split, rows in old_splits.items()},
        "new_total": _split_summary(new_all),
        "old_total": _split_summary(old_all),
        "id_comparison": {
            "common_ids": len(common_ids),
            "old_only_ids": len(old_only),
            "new_only_ids": len(new_only),
            "old_only_examples": old_only[:5],
            "new_only_examples": new_only[:5],
        },
        "common_id_field_changes": {
            "common_ids": len(common_ids),
            "query_changed": changes["query"],
            "bbox_changed": changes["bbox"],
            "class_name_changed": changes["class_name"],
            "modal_paths_changed": changes["modal_paths"],
            "split_changed": changes["split"],
            "note": "No per-ID change is defined when common_ids is zero.",
        },
        "set_differences": {
            "shared_visible_image_names": len(new_images & old_images),
            "shared_query_texts": len(new_queries & old_queries),
            "shared_exact_bboxes": len(new_boxes & old_boxes),
            "shared_class_names": len(
                {str(record.get("class_name")) for record in new_all.values()}
                & {str(record.get("class_name")) for record in old_all.values()}
            ),
        },
        "old_manifest_used_for_candidate_generation": False,
    }


def _load_split_manifests(
    zip_path: Path, train_member: str, val_member: str
) -> dict[str, dict[str, dict[str, Any]]]:
    with zipfile.ZipFile(zip_path) as archive:
        return {
            "train": _read_member(archive, train_member),
            "val": _read_member(archive, val_member),
        }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare same-image/same-class known-object candidate boxes from a manifest ZIP"
    )
    parser.add_argument("--zip", dest="source_zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--legacy-zip", type=Path, help="Optional old target_v2 ZIP for count/label comparison only")
    parser.add_argument("--split", choices=("train", "val", "both"), default="both")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--train-member", default=NEW_TRAIN_MEMBER)
    parser.add_argument("--val-member", default=NEW_VAL_MEMBER)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source_zip = args.source_zip.resolve()
    output_dir = args.output_dir.resolve()
    selected_splits = ("train", "val") if args.split == "both" else (args.split,)
    source_splits = _load_split_manifests(source_zip, args.train_member, args.val_member)

    output_dir.mkdir(parents=True, exist_ok=True)
    generated: dict[str, dict[str, Any]] = {}
    for split in selected_splits:
        records, split_stats = build_candidates(
            source_splits[split], iou_threshold=args.iou_threshold
        )
        filename = f"{split}_instance_candidates.json"
        _write_json(output_dir / filename, records)
        generated[split] = {
            **split_stats,
            "output_file": filename,
            "source_member": args.train_member if split == "train" else args.val_member,
        }

    comparison = None
    if args.legacy_zip is not None:
        legacy_splits = _load_split_manifests(
            args.legacy_zip.resolve(), OLD_TRAIN_MEMBER, OLD_VAL_MEMBER
        )
        comparison = compare_manifests(source_splits, legacy_splits)
        _write_json(output_dir / "source_comparison_summary.json", comparison)

    summary = {
        "source_archive": source_zip.name,
        "source_members": {
            "train": args.train_member,
            "val": args.val_member,
        },
        "generated_splits": list(selected_splits),
        "train_and_val_outputs_are_separate": True,
        "output_schema": (
            "sample_id -> source fields plus positive_bbox, negative_bboxes, candidate_count; "
            "negative_bboxes are distinct known same-image/same-class boxes with IoU < threshold"
        ),
        "class_name_is_candidate_pool_key_only": True,
        "candidate_boxes_are_not_semantic_negative_labels": True,
        "candidate_source": "ground_truth_annotations",
        "validation_use": "oracle instance-choice diagnostics only; not end-to-end grounding ACC",
        "generated": generated,
        "source_comparison_file": "source_comparison_summary.json" if comparison is not None else None,
        "legacy_archive_used_for_candidates": False,
        "hashes_written": False,
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
