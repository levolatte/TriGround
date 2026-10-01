from __future__ import annotations

import json

import pytest
from PIL import Image

from tools.evaluate_aux_reading import load_reading_records, score_reading, summarize_reading
from tools.evaluate_pretrained_grounder import (
    build_run_config,
    ground_with_optional_ir,
    load_records,
    load_model_images,
    parse_ir_reading,
    summarize_rows,
)


def _record(modalities):
    return {
        "id": "one", "query": "the red sign with brand X", "modalities": modalities,
        "prompt_has_image_placeholders": False,
    }


def _fake_generator(answers, calls):
    def generate(prompt, images, placeholders, cap):
        calls.append((prompt, images, placeholders, cap))
        return {
            "raw_text": answers[len(calls) - 1], "latency_seconds": 0.5,
            "generated_tokens": 4, "generation_cap_hit": False,
        }
    return generate


def test_ir_is_selected_only_by_explicit_modality_and_preserves_rgb_prompt():
    calls = []
    record = _record(["rgb", "depth"])
    result = ground_with_optional_ir(
        record, "original prompt", ["RGB", "DEPTH"], "ir_then_ground",
        _fake_generator(['{"bbox_2d":[1,2,3,4]}'], calls), 4096,
    )
    assert result["first_status"] == "no_ir"
    assert result["calls"] == 1
    assert calls == [("original prompt", ["RGB", "DEPTH"], False, 4096)]
    placeholder = _record(["rgb", "ir", "depth"])
    placeholder["missing_modalities_actual"] = ["ir"]
    calls.clear()
    result = ground_with_optional_ir(
        placeholder, "original prompt", ["RGB", "BLACK", "DEPTH"], "ir_then_ground",
        _fake_generator(['{"bbox_2d":[1,2,3,4]}'], calls), 4096,
    )
    assert result["first_status"] == "no_ir"
    assert len(calls) == 1


def test_ir_box_guidance_keeps_ir_coordinates_separate_from_rgb():
    calls = []
    result = ground_with_optional_ir(
        _record(["rgb", "ir", "depth"]), "original RGB prompt",
        ["RGB", "IR", "DEPTH"], "ir_then_ground",
        _fake_generator(['{"bbox_2d":[100,200,300,400]}', '{"bbox_2d":[500,600,700,800]}'], calls), 4096,
    )
    assert result["first_box_ir"] == [100.0, 200.0, 300.0, 400.0]
    assert result["calls"] == 2
    assert result["total_latency_seconds"] == 1.0
    assert calls[0][1:] == (["IR"], False, 64)
    assert "the red sign with brand X" in calls[0][0]
    assert "color, brand" in calls[0][0]
    assert calls[1][1:] == (["RGB", "IR", "DEPTH"], False, 128)
    assert "IR model prediction may be wrong" in calls[1][0]
    assert "infrared-image coordinates, not RGB coordinates" in calls[1][0]
    assert "RGB 0-1000" in calls[1][0]


def test_ir_null_and_invalid_do_not_augment_final_prompt():
    for answer, status in (( '{"bbox_2d":null}', "unknown"), ("uncertain", "reading_parse_failure")):
        calls = []
        result = ground_with_optional_ir(
            _record(["rgb", "ir"]), "original prompt", ["RGB", "IR"],
            "ir_then_ground", _fake_generator([answer, '{"bbox_2d":[1,2,3,4]}'], calls), 4096,
        )
        assert result["first_status"] == status
        assert result["first_box_ir"] is None
        assert result["first_raw_text"] == answer
        assert calls[1] == ("original prompt", ["RGB", "IR"], False, 128)


def test_direct_mode_remains_single_call_with_original_token_cap():
    calls = []
    result = ground_with_optional_ir(
        _record(["rgb", "ir"]), "old prompt", ["RGB", "IR"], "direct",
        _fake_generator(['{"bbox_2d":[1,2,3,4]}'], calls), 4096,
    )
    assert result["calls"] == 1
    assert result["first_status"] == "not_requested"
    assert calls == [("old prompt", ["RGB", "IR"], False, 4096)]
    config = build_run_config(
        model="model", adapter=None, manifest="manifest", target_manifest=None,
        data_root="data", min_pixels=100, max_pixels=200,
        max_new_tokens=4096, prompt_style="egm",
    )
    assert "inference_mode" not in config


def test_ir_mode_requires_explicit_multi_image_order():
    record = _record(None)
    with pytest.raises(ValueError, match="explicit modalities"):
        ground_with_optional_ir(
            record, "prompt", ["RGB", "OTHER"], "ir_then_ground",
            _fake_generator([], []), 4096,
        )


def test_ir_parser_requires_single_valid_json_answer():
    assert parse_ir_reading('{"bbox_2d":null}') == ("unknown", None)
    for answer in ('{"bbox_2d":[900,1,100,2]}', '{"bbox_2d":[1,2,1001,4]}',
                   '{"bbox_2d":[1,2,3,4]} extra', '{"bbox_2d":[1,2,3,4],"confidence":1}'):
        assert parse_ir_reading(answer)[0] == "reading_parse_failure"


def test_mixed_reading_manifest_never_uses_expected_answer_as_prompt(tmp_path):
    manifest = tmp_path / "reading.jsonl"
    records = [
        {
            "id": "ir1", "image": ["rgb.png", "ir.png"], "modalities": ["rgb", "ir"],
            "task_type": "ir_bbox", "coordinate_system": "qwen_0_1000",
            "conversations": [
                {"from": "human", "value": "<image><image>Which sign is hotter?"},
                {"from": "gpt", "value": "SECRET_SUPERVISION"},
            ],
            "expected_answer": {"bbox_2d": [100, 200, 900, 800]},
            "ir_gt_bbox": [0.1004, 0.2, 0.9004, 0.8],
        },
        {
            "id": "ir_unknown", "image": "ir2.png", "modalities": ["ir"],
            "task_type": "ir_bbox", "coordinate_system": "qwen_0_1000",
            "conversations": [{"from": "human", "value": "<image>Find brand X."}],
            "expected_answer": {"bbox_2d": None},
            "ir_gt_bbox": None,
        },
        {
            "id": "depth1", "image": ["depth_a.png", "depth_b.png"], "modalities": ["depth", "depth"],
            "task_type": "depth_relation", "coordinate_system": "categorical",
            "conversations": [{"from": "human", "value": "<image><image>Which is nearer?"}],
            "expected_answer": "B_nearer",
        },
    ]
    manifest.write_text("\n".join(json.dumps(item) for item in records), encoding="utf-8")
    loaded = load_reading_records(manifest)
    assert len(loaded) == 3
    assert "SECRET_SUPERVISION" not in loaded[0]["prompt"]
    assert "900" not in loaded[0]["prompt"]
    assert loaded[0]["ir_gt_bbox"] == [0.1004, 0.2, 0.9004, 0.8]
    assert "0.1004" not in loaded[0]["prompt"]
    scored = []
    for record, answer in zip(loaded, ('{"bbox_2d":[100,200,900,800]}', '{"bbox_2d":null}', "B_nearer")):
        scored.append({**record, **score_reading(record, answer), "latency_seconds": 0.2})
    summary = summarize_reading(scored)
    assert summary["ir_bbox"]["known_acc_0.5"] == 1.0
    assert summary["ir_bbox"]["known_mean_iou"] < 1.0
    assert summary["ir_bbox"]["unknown_recognition"] == 1.0
    assert summary["depth_relation"]["by_answer"]["B_nearer"]["correct"] == 1


def test_reading_null_and_parse_failure_keep_full_denominators(tmp_path):
    empty = tmp_path / "empty.json"
    empty.write_text("[]", encoding="utf-8")
    assert load_reading_records(empty) == []
    assert summarize_reading([])["ir_bbox"]["unknown_recognition"] is None
    known = {"task_type": "ir_bbox", "expected_status": "box",
             "expected_box_ir": [100, 200, 900, 800]}
    unknown = {"task_type": "ir_bbox", "expected_status": "unknown",
               "expected_box_ir": None}
    rows = [
        {**known, **score_reading(known, "invalid"), "latency_seconds": 0.1},
        {**unknown, **score_reading(unknown, '{"bbox_2d":null}'), "latency_seconds": 0.1},
    ]
    summary = summarize_reading(rows)["ir_bbox"]
    assert summary["known_targets"] == 1
    assert summary["known_acc_0.5"] == 0.0
    assert summary["parse_failures"] == 1
    assert summary["unknown_recognition"] == 1.0


def test_formal_parse_failure_keeps_full_denominator():
    target = [0.1, 0.2, 0.9, 0.8]
    rows = [
        {"target": target, "prediction": target, "parsed": True,
         "latency_seconds": 0.1, "generated_tokens": 2, "generation_cap_hit": False,
         "first_status": "no_ir", "calls": 1, "total_latency_seconds": 0.1},
        {"target": target, "prediction": None, "parsed": False,
         "latency_seconds": 0.1, "generated_tokens": 2, "generation_cap_hit": False,
         "first_status": "reading_parse_failure", "calls": 2,
         "total_latency_seconds": 0.2},
    ]
    summary = summarize_rows(rows)
    assert summary["samples"] == 2
    assert summary["parse_failures"] == 1
    assert summary["acc_0.5"] == 0.5
    assert summary["ir_available"] == 1
    assert summary["reading_parse_failures"] == 1
    assert summary["total_calls"] == 3


def test_formal_and_reading_share_image_conversion(tmp_path):
    gray = tmp_path / "ir.png"
    Image.new("L", (3, 2), color=73).save(gray)
    loaded = load_model_images([gray])[0]
    assert loaded.mode == "RGB"
    assert loaded.size == (3, 2)
    assert loaded.getpixel((0, 0)) == (73, 73, 73)


def test_sft_bbox_has_explicit_modality_mapping(tmp_path):
    manifest = tmp_path / "bbox.jsonl"
    record = {
        "id": "x", "image": ["rgb.png", "depth.png"], "modalities": ["rgb", "depth"],
        "query": "the yellow bus", "conversations": [
            {"from": "human", "value": "<image><image>Locate the yellow bus."},
            {"from": "gpt", "value": '{"bbox_2d":[100,200,900,800]}'},
        ],
    }
    manifest.write_text(json.dumps(record), encoding="utf-8")
    loaded = load_records(manifest)[0]
    assert loaded["modalities"] == ["rgb", "depth"]
    assert loaded["query"] == "the yellow bus"
