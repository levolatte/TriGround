import random

import numpy as np
import pytest


def test_empty_detection_tokenizer_label_is_not_a_box():
    from tools.aux_selection_evidence import aligned_detection_rows
    assert aligned_detection_rows({'boxes': [], 'scores': [], 'text_labels': ['']}) == []
    assert aligned_detection_rows({'boxes': [], 'scores': [], 'text_labels': []}) == []
    with pytest.raises(ValueError):
        aligned_detection_rows({'boxes': [[0,0,1,1]], 'scores': [.7], 'text_labels': ['a','b']})

import tools.aux_selection_evidence as evidence
from tools.aux_selection_evidence import (
    assess_depth_mask,
    build_candidates_stage,
    build_candidate_row,
    build_depth_pairs,
    build_depth_stage,
    compare_depth_pair,
    depth_statistics,
    merge_candidate_boxes,
    read_jsonl,
    select_candidate_cap,
    write_jsonl,
)


def test_iou_dedup_merges_sources_and_keeps_the_c_box_exact():
    c_bbox = [0.1, 0.1, 0.5, 0.5]
    candidates = merge_candidate_boxes(
        c_bbox,
        [
            {"role": "target", "modality": "rgb", "bbox": [0.11, 0.11, 0.51, 0.51], "score": 0.8},
            {"role": "target", "modality": "ir", "bbox": c_bbox, "score": 0.7},
        ],
    )

    assert len(candidates) == 1
    assert candidates[0]["bbox"] == c_bbox
    assert candidates[0]["is_baseline"] is True
    assert {source["modality"] for source in candidates[0]["sources"]} == {"c", "rgb", "ir"}
    ir_source = next(source for source in candidates[0]["sources"] if source["modality"] == "ir")
    assert ir_source["projection"] == "normalized_shared_frame"


def test_target_cap_keeps_c_and_top_three_from_each_modality_first():
    detections = []
    for index in range(5):
        x = 0.1 + index * 0.12
        detections.append(
            {
                "role": "target",
                "modality": "rgb",
                "bbox": [x, 0.1, x + 0.05, 0.15],
                "score": 0.99 - index * 0.01,
            }
        )
        y = 0.5 + index * 0.08
        detections.append(
            {
                "role": "target",
                "modality": "ir",
                "bbox": [0.7, y, 0.75, y + 0.05],
                "score": 0.89 - index * 0.01,
            }
        )
    candidates = merge_candidate_boxes([0.01, 0.01, 0.06, 0.06], detections)

    selected = select_candidate_cap(candidates, "target", 8)
    selected_boxes = {tuple(candidate["bbox"]) for candidate in selected}
    rgb_top = {
        tuple(detection["bbox"])
        for detection in sorted(
            (item for item in detections if item["modality"] == "rgb"),
            key=lambda item: -item["score"],
        )[:3]
    }
    ir_top = {
        tuple(detection["bbox"])
        for detection in sorted(
            (item for item in detections if item["modality"] == "ir"),
            key=lambda item: -item["score"],
        )[:3]
    }
    assert len(selected) == 8
    assert selected[0]["is_baseline"] is True
    assert rgb_top | ir_top <= selected_boxes


def test_reference_cap_reserves_two_distinct_candidates_per_modality():
    detections = []
    for index in range(4):
        detections.extend(
            [
                {
                    "role": "reference",
                    "modality": "rgb",
                    "bbox": [0.1 + index * 0.1, 0.1, 0.15 + index * 0.1, 0.15],
                    "score": 0.99 - index * 0.01,
                },
                {
                    "role": "reference",
                    "modality": "ir",
                    "bbox": [0.1 + index * 0.1, 0.6, 0.15 + index * 0.1, 0.65],
                    "score": 0.89 - index * 0.01,
                },
            ]
        )
    candidates = merge_candidate_boxes([0.01, 0.01, 0.06, 0.06], detections)

    selected = select_candidate_cap(candidates, "reference", 4, modality_top_k=2)

    assert len(selected) == 4
    assert sum(any(source["modality"] == "rgb" for source in c["sources"]) for c in selected) == 2
    assert sum(any(source["modality"] == "ir" for source in c["sources"]) for c in selected) == 2


def test_candidate_ids_are_deterministic_and_manifest_gt_is_ignored():
    manifest = {
        "id": "sample-1",
        "query": "the red car",
        "images": {"rgb": "rgb.png", "ir": "ir.png", "depth_raw": "depth.png"},
        "depth_encoding": "city_mm",
        "bbox": [0.8, 0.8, 0.95, 0.95],  # Deliberately unrelated manifest annotation.
    }
    baseline = {"id": "sample-1", "bbox": [0.1, 0.1, 0.3, 0.3]}
    query_info = {
        "id": "sample-1",
        "target_category": "car",
        "reference_categories": [],
        "relation_type": "none",
        "scope": "single",
    }
    detections = [
        {"role": "target", "modality": "rgb", "bbox": [0.4, 0.4, 0.6, 0.6], "score": 0.7},
        {"role": "target", "modality": "rgb", "bbox": [0.65, 0.4, 0.75, 0.5], "score": 0.6},
        {"role": "target", "modality": "rgb", "bbox": [0.4, 0.65, 0.5, 0.75], "score": 0.5},
        {"role": "target", "modality": "rgb", "bbox": [0.7, 0.7, 0.8, 0.8], "score": 0.4},
    ]

    first = build_candidate_row(manifest, baseline, query_info, detections, random.Random(2028))
    second = build_candidate_row(manifest, baseline, query_info, detections, random.Random(2028))

    assert first == second
    assert first["c_bbox"] == baseline["bbox"]
    ids = [candidate["id"] for candidate in first["candidates"]]
    assert ids == sorted(ids)
    assert set(ids) == set(range(1, len(ids) + 1))
    baseline_index = next(i for i, candidate in enumerate(first["candidates"]) if candidate["is_baseline"])
    assert baseline_index > 0
    assert "bbox" not in first


def test_part_scope_keeps_only_c_and_does_not_use_detector_proposals():
    row = build_candidate_row(
        {
            "id": "part-1",
            "query": "the left wing",
            "images": {"rgb": "rgb.png", "ir": "ir.png", "depth_raw": "depth.png"},
            "depth_encoding": "city_mm",
        },
        {"id": "part-1", "bbox": [0.2, 0.2, 0.5, 0.5]},
        {"scope": "part", "target_category": "wing"},
        [
            {"role": "target", "modality": "rgb", "bbox": [0.6, 0.6, 0.8, 0.8], "score": 0.9}
        ],
        random.Random(2026),
    )

    assert len(row["candidates"]) == 1
    assert row["candidates"][0]["bbox"] == [0.2, 0.2, 0.5, 0.5]
    assert row["candidates"][0]["is_baseline"] is True


def test_depth_statistics_ignore_invalid_pixels_and_pixels_outside_mask():
    depth = np.full((40, 40), 9000, dtype=np.uint16)
    mask = np.zeros((40, 40), dtype=bool)
    mask[5:35, 5:35] = True
    depth[mask] = 1000
    depth[5, 5] = 0
    depth[6, 6] = 25000

    stats = depth_statistics(depth, mask)

    assert stats["median_m"] == 1.0
    assert stats["valid_count"] == int(mask.sum()) - 2
    assert stats["valid_ratio"] > 0.99


def test_full_mask_background_contamination_marks_depth_unstable():
    depth = np.full((20, 100), 2000, dtype=np.uint16)
    mask = np.zeros_like(depth, dtype=bool)
    mask[6:14, 3:97] = True
    depth[6:9, 3:97] = 8000
    depth[11:14, 3:97] = 8000

    result, core = assess_depth_mask(depth, mask, 0.95, depth_encoding="city_mm")

    assert core.sum() >= 32
    assert result["full"]["median_m"] == 8.0
    assert result["core"]["median_m"] == 2.0
    assert result["status"] == "unreliable"
    assert "full_core_median_unstable" in result["reason"]


def test_unknown_encoding_never_exposes_numeric_depth_or_near_far():
    depth = np.full((20, 20), 1000, dtype=np.uint16)
    mask = np.ones((20, 20), dtype=bool)
    unknown, _ = assess_depth_mask(depth, mask, 0.99, depth_encoding="unknown")
    target = {"id": 1, "role": "target", "depth": unknown}
    reference = {"id": 2, "role": "reference", "depth": unknown}

    pair = compare_depth_pair(target, reference, "unknown")

    assert unknown["status"] == "unknown_encoding"
    assert unknown["full"] is None
    assert pair["status"] == "unknown_encoding"
    assert "near_median_m" not in pair


def test_near_far_pair_requires_separated_quantiles_and_median_gap():
    target = {
        "id": 101,
        "role": "target",
        "depth": {
            "status": "reliable",
            "core": {"q1_m": 1.0, "median_m": 1.1, "q3_m": 1.2},
        },
    }
    reference = {
        "id": 202,
        "role": "reference",
        "depth": {
            "status": "reliable",
            "core": {"q1_m": 2.0, "median_m": 2.2, "q3_m": 2.4},
        },
    }

    pair = compare_depth_pair(target, reference, "city_mm")

    assert pair["status"] == "supported"
    assert pair["near_candidate_id"] == target["id"]
    assert pair["far_candidate_id"] == reference["id"]


def test_depth_pairs_include_unique_target_pairs_and_target_reference_pairs():
    def candidate(candidate_id, role, depth_m):
        return {
            "id": candidate_id,
            "role": role,
            "depth": {
                "status": "reliable",
                "core": {
                    "q1_m": depth_m - 0.1,
                    "median_m": depth_m,
                    "q3_m": depth_m + 0.1,
                },
            },
        }

    targets = [candidate(1, "target", 1), candidate(2, "target", 3), candidate(3, "target", 5)]
    references = [candidate(4, "reference", 7), candidate(5, "reference", 9)]

    pairs = build_depth_pairs(targets, references, "city_mm")

    target_pairs = [pair for pair in pairs if pair["pair_kind"] == "target_target"]
    cross_pairs = [pair for pair in pairs if pair["pair_kind"] == "target_reference"]
    pair_ids = [(pair["candidate_a_id"], pair["candidate_b_id"]) for pair in pairs]
    assert len(target_pairs) == 3
    assert len(cross_pairs) == 6
    assert len(pair_ids) == len(set(pair_ids))


def test_stages_append_each_id_and_skip_completed_ids_on_resume(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.jsonl"
    baseline = tmp_path / "baseline.jsonl"
    query_info = tmp_path / "query_info.jsonl"
    candidates = tmp_path / "candidates.jsonl"
    evidence_path = tmp_path / "evidence.jsonl"
    manifest_rows = [
        {
            "id": f"sample-{index}",
            "query": "the group",
            "images": {"rgb": "rgb.png", "ir": "ir.png", "depth_raw": "depth.png"},
            "depth_encoding": "city_mm",
        }
        for index in range(2)
    ]
    write_jsonl(manifest, manifest_rows)
    write_jsonl(
        baseline,
        [{"id": row["id"], "bbox": [0.1, 0.1, 0.3, 0.3]} for row in manifest_rows],
    )
    write_jsonl(
        query_info,
        [{"id": row["id"], "scope": "group"} for row in manifest_rows],
    )
    append_counts = []
    original_append = evidence._append_jsonl_row

    def observe_append(path, row):
        original_append(path, row)
        append_counts.append(len(read_jsonl(path)))

    monkeypatch.setattr(evidence, "_append_jsonl_row", observe_append)
    first_candidates = build_candidates_stage(
        manifest, baseline, query_info, tmp_path, candidates, model_name="local-dino"
    )
    assert append_counts == [1, 2]
    append_counts.clear()
    resumed_candidates = build_candidates_stage(
        manifest, baseline, query_info, tmp_path, candidates, model_name="local-dino"
    )
    assert append_counts == []
    assert first_candidates == resumed_candidates
    with pytest.raises(ValueError, match="config mismatch"):
        build_candidates_stage(
            manifest, baseline, query_info, tmp_path, candidates, model_name="other-dino"
        )

    append_counts.clear()
    first_evidence = build_depth_stage(candidates, tmp_path, evidence_path, model_name="local-sam")
    assert append_counts == [1, 2]
    assert [row["candidates"][0]["depth"]["status"] for row in first_evidence] == [
        "skipped_scope",
        "skipped_scope",
    ]
    assert len(read_jsonl(evidence_path)) == 2
    append_counts.clear()
    resumed_evidence = build_depth_stage(candidates, tmp_path, evidence_path, model_name="local-sam")
    assert first_evidence == resumed_evidence
    assert append_counts == []
    assert len(read_jsonl(evidence_path)) == 2
