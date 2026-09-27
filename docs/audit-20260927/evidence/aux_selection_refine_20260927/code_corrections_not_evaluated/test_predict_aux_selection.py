import json
from argparse import Namespace
from pathlib import Path

import pytest
from PIL import Image

from tools.predict_aux_selection import (
    _make_selection_row,
    build_query_parse_prompt,
    build_selection_prompt,
    make_crop_transform,
    map_crop_bbox_to_original,
    needs_refine,
    parse_crop_bbox,
    parse_query_json,
    parse_selection,
    select_bbox,
    selection_changed_from_c,
    run_select_refine,
)


def candidates():
    return [
        {"id": 4, "role": "target", "bbox": [0.2, 0.2, 0.5, 0.6], "is_baseline": True,
         "sources": [{"modality": "c", "tag": "PRIVATE_SOURCE"}], "mask_path": "PRIVATE_MASK"},
        {"id": 2, "role": "target", "bbox": [0.6, 0.2, 0.8, 0.4], "is_baseline": False,
         "sources": [{"modality": "ir", "tag": "PRIVATE_SOURCE"}], "mask_path": "PRIVATE_MASK"},
        {"id": 9, "role": "reference", "bbox": [0.1, 0.7, 0.3, 0.9], "is_baseline": False,
         "sources": [{"modality": "depth", "tag": "PRIVATE_SOURCE"}], "mask_path": "PRIVATE_MASK"},
    ]


def query_info(**updates):
    result = {
        "target_category": "car",
        "reference_categories": ["person"],
        "relation_type": "camera_near",
        "scope": "single",
    }
    result.update(updates)
    return result


def test_crop_transform_round_trips_and_stays_inside_image():
    original_box = [0.21, 0.19, 0.31, 0.29]
    pixel_crop, transform = make_crop_transform(original_box, 1000, 500)
    assert pixel_crop[0] >= 0 and pixel_crop[1] >= 0
    assert pixel_crop[2] <= 1000 and pixel_crop[3] <= 500
    local_box = [
        (original_box[0] - transform[0]) / (transform[2] - transform[0]),
        (original_box[1] - transform[1]) / (transform[3] - transform[1]),
        (original_box[2] - transform[0]) / (transform[2] - transform[0]),
        (original_box[3] - transform[1]) / (transform[3] - transform[1]),
    ]
    assert map_crop_bbox_to_original(local_box, transform) == pytest.approx(original_box)

    edge_crop, edge_transform = make_crop_transform([0.01, 0.94, 0.05, 0.98], 1000, 500)
    assert edge_crop[0] == 0 and edge_crop[3] == 500
    assert all(0 <= value <= 1 for value in edge_transform)
    # Minimum size applies before clipping; the window must not shift its center.
    assert edge_crop == (0, 449, 90, 500) or edge_crop == (0, 450, 90, 500)
    assert edge_transform[2] == pytest.approx(0.09)


def test_crop_expands_about_center_before_clipping_large_boxes():
    crop, transform = make_crop_transform([0.0, 0.0, 0.6, 0.8], 1000, 1000)
    assert crop == (0, 0, 900, 1000)
    assert transform == pytest.approx([0, 0, .9, 1])


def test_selection_uses_candidate_id_after_candidate_order_changes():
    boxes = candidates()
    chosen = parse_selection('{"id":2}', {2, 4})
    assert chosen == ("candidate", 2)
    assert select_bbox(chosen, boxes, [0.05, 0.05, 0.15, 0.15]) == [0.6, 0.2, 0.8, 0.4]
    shuffled = [boxes[2], boxes[0], boxes[1]]
    assert select_bbox(chosen, shuffled, [0.05, 0.05, 0.15, 0.15]) == [0.6, 0.2, 0.8, 0.4]


def test_refine_only_when_box_is_small_or_target_candidate_changes_c():
    boxes = candidates()
    keep = ("keep", None)
    baseline_choice = ("candidate", 4)
    changed_choice = ("candidate", 2)
    large_keep_box = [0.1, 0.1, 0.4, 0.5]
    large_c_box = boxes[0]["bbox"]
    large_non_c_box = boxes[1]["bbox"]
    assert selection_changed_from_c(keep, boxes) is False
    assert needs_refine("keep", large_keep_box, changed_from_c=False) is False
    assert selection_changed_from_c(baseline_choice, boxes) is False
    assert needs_refine("candidate", large_c_box, changed_from_c=False) is False
    assert selection_changed_from_c(changed_choice, boxes) is True
    assert needs_refine("candidate", large_non_c_box, changed_from_c=True) is True

    small_c = [{"id": 1, "role": "target", "bbox": [0.1, 0.1, 0.14, 0.16], "is_baseline": True}]
    assert needs_refine("candidate", small_c[0]["bbox"],
                        changed_from_c=selection_changed_from_c(("candidate", 1), small_c)) is True


def test_reference_candidates_are_context_only_and_never_valid_selections():
    prompt = build_selection_prompt("the car beside the person", query_info(), candidates(), "rgb_ir", {})
    assert "Reference candidates (context only; never return their IDs)" in prompt
    assert "9: bbox_2d=" in prompt
    assert parse_selection('{"id":9}', {2, 4}) is None
    assert parse_selection('{"id":2}', {2, 4}) == ("candidate", 2)


def test_condition_prompts_hide_source_baseline_and_depth_metadata():
    evidence = {
        "depth_pairs": [{
            "status": "supported", "target_candidate_id": 4,
            "near_candidate_id": 4, "far_candidate_id": 2,
            "near_q3_m": 4.4, "far_q1_m": 5.8,
            "median_difference_m": 1.7, "required_difference_m": 0.5,
            "role": "PRIVATE_ROLE",
        }],
    }
    with_depth = candidates()
    for candidate in with_depth:
        candidate["depth"] = {
            "status": "reliable", "reason": "PRIVATE_RELIABILITY_REASON",
            "full": {"q1_m": 4.1, "median_m": 4.3, "q3_m": 4.5},
            "core": {"q1_m": 4.2, "median_m": 4.3, "q3_m": 4.4},
        }
    for condition in ("rgb_ir", "rgb_evidence"):
        prompt = build_selection_prompt("a red car", query_info(), with_depth, condition, evidence)
        assert "PRIVATE_SOURCE" not in prompt
        assert "PRIVATE_MASK" not in prompt
        assert "PRIVATE_ROLE" not in prompt
        assert "PRIVATE_RELIABILITY_REASON" not in prompt
        assert "4.300m" not in prompt and "supported" not in prompt
        assert "target_candidate_id" not in prompt
    prompt = build_selection_prompt("a red car", query_info(), with_depth, "trimodal", evidence)
    assert "PRIVATE_SOURCE" not in prompt and "PRIVATE_MASK" not in prompt
    assert "4.300m" in prompt and "candidate 4 is nearer than candidate 2" in prompt
    assert "target_candidate_id" not in prompt and "PRIVATE_ROLE" not in prompt
    no_relation_prompt = build_selection_prompt(
        "a red car", query_info(relation_type="other"), with_depth, "trimodal", evidence
    )
    assert "4.300m" not in no_relation_prompt


def test_non_single_scope_keeps_c_bbox_without_model_selection():
    source = {"id": "q1", "_id": "q1", "query": "the group of cars"}
    evidence = {"c_bbox": [0.1, 0.2, 0.7, 0.8]}
    row = _make_selection_row(
        source, evidence, query_info(scope="group"), "trimodal",
        output=None, prompt=None, elapsed=0.0, candidates=candidates(),
    )
    assert row["selection_kind"] == "scope_policy_keep"
    assert row["prediction"] == evidence["c_bbox"]
    assert row["selection_raw"] is None
    assert row["selection_status"] == "scope_group_kept_c_bbox"


def test_group_scope_run_skips_model_load_and_preserves_c_bbox(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.jsonl"
    evidence_path = tmp_path / "evidence.jsonl"
    query_info_path = tmp_path / "query_info.jsonl"
    row = {
        "id": "q-group", "query": "the group of cars",
        "images": {"rgb": "rgb.png", "ir": "ir.png", "depth_visual": "depth.png", "depth_raw": "raw.png"},
        "depth_encoding": "city_mm",
    }
    evidence = {
        **row, "c_bbox": [0.1, 0.2, 0.7, 0.8], "depth_pairs": [],
        "candidates": [{"id": 1, "role": "target", "bbox": [0.1, 0.2, 0.7, 0.8],
                        "is_baseline": True, "sources": []}],
    }
    query = {
        "id": "q-group", "query": row["query"], "target_category": "car",
        "reference_categories": [], "relation_type": "none", "scope": "group", "parsed": True,
    }
    manifest_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    evidence_path.write_text(json.dumps(evidence) + "\n", encoding="utf-8")
    query_info_path.write_text(json.dumps(query) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        "tools.predict_aux_selection._load_qwen",
        lambda *_: pytest.fail("group scope must bypass model loading"),
    )
    run_select_refine(Namespace(
        manifest=manifest_path, evidence=evidence_path, query_info=query_info_path,
        output_dir=tmp_path / "run", condition="trimodal", model="local-model",
        adapter=tmp_path / "adapter", resume=False, limit=0, deadline_epoch=None,
    ))
    prediction = json.loads((tmp_path / "run/final_predictions.jsonl").read_text().strip())
    assert prediction["prediction"] == evidence["c_bbox"]
    assert prediction["refine_status"] == "skipped_scope_policy"
    assert prediction["generation_cap_hit"] is False


def test_model_output_parsers_are_strict_and_keep_is_explicit():
    parsed = parse_query_json(json.dumps({
        "target_category": "car", "reference_categories": ["person"],
        "relation_type": "camera_far", "scope": "single",
    }))
    assert parsed["target_category"] == "car"
    assert parse_query_json("not json") is None
    assert parse_query_json('{"target_category":"car","reference_categories":[],"relation_type":"none","scope":"single","bbox":[0,0,1,1]}') is None

    assert parse_selection("KEEP", {1}) == ("keep", None)
    assert parse_selection('{"id":1}', {1}) == ("candidate", 1)
    assert parse_selection('1', {1}) == ("candidate", 1)
    assert parse_selection('2', {1}) is None
    assert parse_selection('true', {1}) is None
    assert parse_selection('{"id":2}', {1}) is None
    assert parse_selection('```json {"id":1}```', {1}) is None

    assert parse_crop_bbox('{"bbox_2d":[100,200,800,900]}') == [0.1, 0.2, 0.8, 0.9]
    assert parse_crop_bbox("KEEP") == "KEEP"
    assert parse_crop_bbox('{"bbox_2d":[900,200,100,900]}') is None


def test_query_parse_v2_examples_and_exact_target_reference_deduplication():
    prompt = build_query_parse_prompt("the back of an overhead sign")
    assert '"the silver sedan parked at the far left"' in prompt
    assert '"a person beside a tree"' in prompt
    assert '"the front surface of a cabinet"' in prompt
    assert "different category" in prompt
    assert "Foreground/background must describe camera depth" in prompt

    result = parse_query_json(json.dumps({
        "target_category": "Light",
        "reference_categories": ["light", "ceiling"],
        "relation_type": "none",
        "scope": "single",
    }))
    assert result["reference_categories"] == ["ceiling"]


def test_candidate_annotation_returns_a_separate_rgb_image():
    from tools.predict_aux_selection import annotate_candidates

    image = Image.new("RGB", (320, 240), "black")
    annotated = annotate_candidates(image, candidates())
    assert annotated.size == image.size
    assert image.getbbox() is None
    assert annotated.getbbox() is not None
