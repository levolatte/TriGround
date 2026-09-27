import sys

import pytest

from tools.evaluate_locateanything import (
    DEFAULT_GROUND_KWARGS,
    DEFAULT_WORKER_CONFIG,
    build_run_config,
    parse_boxes,
    parse_first_box,
    resolve_rgb_path,
    rescore_rows,
    validate_resume_rows,
)


def test_parse_first_valid_box_and_keep_all_candidates():
    answer = (
        "<ref>red object</ref><box><800><700><200><100></box> "
        "<box><100><200><400><600></box> <box><0><0><1001><500></box>"
    )
    assert parse_boxes(answer) == [[0.1, 0.2, 0.4, 0.6]]
    assert parse_first_box(answer) == [0.1, 0.2, 0.4, 0.6]
    assert parse_first_box("<box>none</box>") is None
    assert parse_boxes(None) == []


def test_relative_visible_path_uses_manifest_parent(tmp_path):
    manifest_path = tmp_path / "labels" / "val.json"
    manifest_path.parent.mkdir()
    assert resolve_rgb_path(manifest_path, "../visible/0001.jpg") == (
        tmp_path / "visible" / "0001.jpg"
    ).resolve()


def test_config_records_official_worker_and_generation_kwargs(tmp_path):
    config = build_run_config(
        model="local",
        manifest=tmp_path / "manifest.json",
        output_dir=tmp_path / "out",
        limit=0,
        worker_dir=tmp_path / "worker",
    )
    assert config["worker_constructor_kwargs"] == DEFAULT_WORKER_CONFIG
    assert config["ground_multi_kwargs"] == DEFAULT_GROUND_KWARGS
    assert config["image_processing"]["resize"] is False
    assert config["minimum_dependencies"]["installed_by_this_tool"] is False
    assert "locateanything_worker" not in sys.modules


def test_resume_keeps_original_gt_and_rescores_stale_fields():
    record = {
        "visible": "visible/0001.jpg",
        "infrared": "infrared/0001.jpg",
        "depth": "depth/0001.png",
        "query": "the object",
        "bbox": [0.0, 0.0, 0.4, 0.4],
    }
    records = [("sample-1", record)]
    row = {
        "id": "sample-1",
        "target": list(record["bbox"]),
        "prediction": [0.6, 0.6, 0.9, 0.9],
        "iou": 1.0,
        "acc_0.5": True,
    }
    existing = validate_resume_rows([row], records)
    summary = rescore_rows(existing, records)
    assert summary["mean_iou"] == pytest.approx(0.0)
    assert summary["acc_0.5"] == 0.0


def test_multiple_valid_boxes_keep_order_and_failed_sample_in_denominator():
    boxes = parse_boxes("<box><600><600><900><900></box><box><0><0><400><400></box>")
    assert len(boxes) == 2
    assert boxes[0] == [0.6, 0.6, 0.9, 0.9]
    records = [("wrong_first", {"bbox": [0, 0, .4, .4]}), ("missing", {"bbox": [0, 0, .4, .4]})]
    summary = rescore_rows({"wrong_first": {"prediction": boxes[0]}, "missing": {"prediction": None}}, records)
    assert summary["samples"] == 2
    assert summary["hits"] == 0
    assert summary["parse_failures"] == 1
