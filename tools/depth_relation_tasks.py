"""Build manually specified City depth-relation drafts from original GT.

The spec supplies the wording and object binding. Measurements only check that
the stated *bounded* relation agrees with the raw millimetre depth image; they
cannot establish that the visible-depth rendering makes it answerable.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _patch(image: np.ndarray, region: list[float]) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = region
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError(f"invalid normalized evidence region: {region}")
    left = min(width - 1, int(x1 * width))
    top = min(height - 1, int(y1 * height))
    right = min(width, max(left + 1, int(np.ceil(x2 * width))))
    bottom = min(height, max(top + 1, int(np.ceil(y2 * height))))
    return image[top:bottom, left:right]


def _bbox_core(bbox: list[float]) -> list[float]:
    x1, y1, x2, y2 = bbox
    return [x1 + (x2 - x1) / 4, y1 + (y2 - y1) / 4,
            x2 - (x2 - x1) / 4, y2 - (y2 - y1) / 4]


def region_depth_evidence(raw: np.ndarray, visual: np.ndarray,
                          region: list[float], provenance: str) -> dict[str, Any]:
    """Describe raw zero/saturated pixels and the separately displayed gray patch."""
    raw_patch = _patch(raw, region).reshape(-1)
    gray_patch = _patch(visual, region)
    if gray_patch.ndim == 3:
        gray_patch = np.asarray(Image.fromarray(gray_patch).convert("L"))
    gray_patch = gray_patch.reshape(-1)
    valid = raw_patch[(raw_patch > 0) & (raw_patch < 20_000)]
    nonzero = raw_patch[raw_patch > 0]
    saturated = raw_patch >= 20_000
    return {
        "region": list(region), "region_source": provenance,
        "raw_total_px": int(raw_patch.size),
        "raw_zero_px": int(np.count_nonzero(raw_patch == 0)),
        "raw_saturated_px": int(np.count_nonzero(saturated)),
        "raw_valid_px": int(valid.size),
        "raw_valid_fraction": round(float(valid.size / raw_patch.size), 4),
        "raw_zero_fraction": round(float(np.mean(raw_patch == 0)), 4),
        "raw_saturated_fraction": round(float(np.mean(saturated)), 4),
        "raw_median_mm": float(np.median(valid)) if valid.size else None,
        "raw_valid_iqr_mm": (float(np.percentile(valid, 75) - np.percentile(valid, 25))
                             if valid.size else None),
        # A median including saturation exposes a mostly saturated patch.
        "raw_nonzero_median_mm": float(np.median(nonzero)) if nonzero.size else None,
        "visual_gray_median": float(np.median(gray_patch)),
        "visual_gray_iqr": float(np.percentile(gray_patch, 75) - np.percentile(gray_patch, 25)),
    }


def _relation_holds(kind: str, target: float, competitors: list[float],
                    reference: float | None) -> bool:
    if kind == "middle":
        return min(competitors) < target < max(competitors)
    other = competitors[0]
    if reference is None:
        return target < other if kind == "nearer" else target > other
    if kind == "nearer":
        return target < reference < other
    return other < reference < target


def build_depth_tasks(specs: list[dict[str, Any]],
                      source_records: dict[str, dict[str, Any]],
                      scene_rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Return draft tasks; invalid bindings or measurements fail with spec ID.

    ``scene_rows`` maps ``city:stem`` to a retained City candidate carrying its
    original split, images, and location_group. ``source_records`` is the
    untouched ``city_raw_labels()`` mapping. A spec has ``spec_id``,
    ``scene_id``, ``split``, ``scope_description``, ``visual_review``, and
    ``tasks``. Each task has ``key``, English ``query``, ``query_zh``,
    ``target_id``, and ``relation`` with kind/competitors/optional reference_id.
    Optional ``evidence_regions`` maps source IDs to normalized RGB rectangles.
    """
    output: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for spec in specs:
        spec_id = spec["spec_id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", spec_id):
            raise ValueError(f"{spec_id}: spec_id must be a simple ID")
        scene_id = spec["scene_id"]
        if not scene_id.startswith("city:"):
            scene_id = f"city:{scene_id}"
        stem = scene_id.removeprefix("city:")
        if scene_id not in scene_rows:
            raise ValueError(f"{spec_id}: scene is absent from retained candidates")
        scene = scene_rows[scene_id]
        if spec["split"] != scene["split"]:
            raise ValueError(f"{spec_id}: split conflicts with retained scene")
        if not spec["scope_description"].strip():
            raise ValueError(f"{spec_id}: missing bounded comparison scope")
        tasks = spec["tasks"]
        auto_pair = (len(tasks) == 2 and {t["relation"]["kind"] for t in tasks} ==
                     {"nearer", "farther"} and tasks[0]["target_id"] != tasks[1]["target_id"])
        pair_required = spec.get("pair_required", auto_pair)
        if pair_required and len(tasks) != 2:
            raise ValueError(f"{spec_id}: paired bundle must contain two tasks")
        bundle_id = f"{scene_id}:depthv2:{spec_id}"
        raw = np.asarray(Image.open(scene["images"]["depth_raw"]))
        visual = np.asarray(Image.open(scene["images"]["depth"]).convert("L"))
        regions = spec.get("evidence_regions", {})
        participants = {source_id for task in tasks
                        for source_id in ([task["target_id"]] + task["relation"]["competitors"] +
                                          ([task["relation"]["reference_id"]]
                                           if "reference_id" in task["relation"] else []))}
        evidence: dict[str, dict[str, Any]] = {}
        for source_id in participants:
            if source_id not in source_records:
                raise ValueError(f"{spec_id}: unknown source object {source_id}")
            record = source_records[source_id]
            if Path(record["visible"]).stem != stem:
                raise ValueError(f"{spec_id}: {source_id} belongs to another scene")
            bbox = record["bbox"]
            region = regions.get(source_id)
            provenance = "manual_subject_interior" if region is not None else "bbox_core_low_confidence"
            if region is None:
                region = _bbox_core(bbox)
            elif not (bbox[0] <= region[0] < region[2] <= bbox[2] and
                      bbox[1] <= region[1] < region[3] <= bbox[3]):
                raise ValueError(f"{spec_id}: {source_id} evidence region outside GT bbox")
            evidence[source_id] = region_depth_evidence(raw, visual, region, provenance)
            item = evidence[source_id]
            if item["raw_valid_px"] == 0 or item["raw_nonzero_median_mm"] >= 20_000:
                raise ValueError(f"{spec_id}: {source_id} has no usable unsaturated depth")
        for task in tasks:
            key = task["key"]
            if not re.fullmatch(r"[A-Za-z0-9_-]+", key):
                raise ValueError(f"{spec_id}: task key must be a simple ID")
            if not task["query"].strip() or not task["query_zh"].strip():
                raise ValueError(f"{spec_id}: missing query text in {key}")
            task_id = f"{bundle_id}:{key}"
            if task_id in seen_ids:
                raise ValueError(f"{spec_id}: duplicate task ID {task_id}")
            seen_ids.add(task_id)
            relation = task["relation"]
            kind = relation["kind"]
            competitors = relation["competitors"]
            reference_id = relation.get("reference_id")
            expected = 2 if kind == "middle" else 1
            if kind not in {"nearer", "farther", "middle"} or len(competitors) != expected:
                raise ValueError(f"{spec_id}: unsupported relation in {key}")
            if reference_id is not None and kind == "middle":
                raise ValueError(f"{spec_id}: middle relation cannot use a reference")
            ids = [task["target_id"], *competitors, *([reference_id] if reference_id else [])]
            if len(set(ids)) != len(ids):
                raise ValueError(f"{spec_id}: relation repeats an object in {key}")
            medians = {source_id: evidence[source_id]["raw_median_mm"] for source_id in ids}
            if not _relation_holds(kind, medians[task["target_id"]],
                                   [medians[s] for s in competitors],
                                   medians[reference_id] if reference_id else None):
                raise ValueError(f"{spec_id}: raw depth contradicts {key} {kind} relation")
            target = source_records[task["target_id"]]
            relation_type = ("middle" if kind == "middle" else
                             "reference" if reference_id else "bounded_pair")
            output.append({
                "task_id": task_id, "bundle_id": bundle_id, "scene_id": scene_id,
                "split": scene["split"],
                "category": "diag_depth" if scene["split"] == "diagnostic" else "depth_relation",
                "source": "city", "images": dict(scene["images"]),
                "proposed_query": task["query"], "query_zh": task["query_zh"],
                "bbox": list(target["bbox"]),
                "target_object_id": f"city:{stem}:{task['target_id']}",
                "source_id": task["target_id"], "depth_policy": "millimeter",
                "review_status": "draft_unapproved", "original_query": target["query"],
                "location_group": scene["location_group"],
                "construction_version": "depth_relations_v2",
                "relation_type": relation_type, "pair_required": bool(pair_required),
                "relation": {"kind": kind, "target_id": task["target_id"],
                             "competitors": list(competitors), "reference_id": reference_id},
                "scope_description": spec["scope_description"],
                "visual_review": spec["visual_review"],
                "depth_evidence": {"by_source_id": {source_id: evidence[source_id] for source_id in ids},
                                   "raw_order_checked": True,
                                   "all_manual_regions": all(evidence[source_id]["region_source"] ==
                                                             "manual_subject_interior" for source_id in ids),
                                   "visible_depth_order_reviewed": False},
                "predecessor_task_ids": list(task.get("predecessor_task_ids",
                                                       spec.get("predecessor_task_ids", []))),
            })
    return output
