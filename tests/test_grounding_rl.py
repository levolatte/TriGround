from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.grounding_reward import grounding_reward, grounding_rewards, parse_short_bbox
from tools.prepare_grounding_rl import convert, convert_record, load_targets
from tools.verify_acc05_dev import iou_xyxy


PROMPT = ('<image>\n<image>\n<image>\nRGB, infrared, depth. Locate the person. '
          'Return only JSON: {"bbox_2d":[x1,y1,x2,y2]}, coordinates 0-1000.')


def swift_record():
    return {
        "id": "sample:1",
        "origin_query_id": "original:1",
        "messages": [{"role": "user", "content": PROMPT},
                     {"role": "assistant", "content": '{"bbox_2d":[100,200,700,800]}'}],
        "images": ["/root/rgb/a.png", "/root/ir/a.png", "/root/depth_rgb/a.png"],
        "modalities": ["rgb", "infrared", "depth"],
    }


def test_reward_matches_official_inclusive_threshold_and_saturates():
    target = [0.0, 0.0, 1.0, 1.0]
    at_half = '{"bbox_2d":[0,0,500,1000]}'
    below_half = '{"bbox_2d":[0,0,499,1000]}'
    perfect = '{"bbox_2d":[0,0,1000,1000]}'
    assert iou_xyxy(parse_short_bbox(at_half), target) == 0.5
    assert grounding_reward(at_half, target) == pytest.approx(1.1)
    assert grounding_reward(perfect, target) == pytest.approx(1.1)
    assert grounding_reward(below_half, target) == pytest.approx(0.2 * 0.499)
    assert grounding_reward(at_half, target, "binary") == 1.0
    assert grounding_reward(below_half, target, "binary") == 0.0
    assert grounding_reward(at_half, target, "iou") == 0.5


@pytest.mark.parametrize("text", [
    'The box is {"bbox_2d":[0,0,500,500]}',
    '```json\n{"bbox_2d":[0,0,500,500]}\n```',
    '{"bbox_2d":[0,0,500,500]} {"bbox_2d":[0,0,600,600]}',
    '{"bbox_2d":[0,0,500,500],"confidence":1}',
    '{"bbox_2d":[0,0,100,100],"bbox_2d":[0,0,500,500]}',
    '{"bbox_2d":[0,0,1001,500]}',
    '{"bbox_2d":[-1,0,500,500]}',
    '{"bbox_2d":[500,0,0,500]}',
    '{"bbox_2d":[0,0,0,500]}',
    '{"bbox_2d":[0,0,NaN,500]}',
    '{"bbox_2d":[0,0,Infinity,500]}',
    '{"bbox_2d":[false,0,500,500]}',
    '{"bbox_2d":["0",0,500,500]}',
    '{"bbox_2d":null}',
    '[0,0,500,500]',
    'null',
])
def test_malformed_output_gets_negative_reward_without_repair(text):
    assert parse_short_bbox(text) is None
    assert grounding_reward(text, [0, 0, 1, 1]) == -0.1


def test_json_whitespace_and_fractional_0_1000_coordinates_are_valid():
    assert parse_short_bbox(' \n {"bbox_2d": [1.5, 2, 500, 600]} \n') == [0.0015, 0.002, 0.5, 0.6]


def test_bad_gt_fails_instead_of_becoming_zero_reward():
    with pytest.raises(ValueError, match="solution"):
        grounding_reward("invalid output", [0, 0, 1000, 1000])
    with pytest.raises(ValueError, match="solution"):
        grounding_reward("invalid output", [False, 0, 1, 1])
    with pytest.raises(ValueError, match="same length"):
        grounding_rewards(["a", "b"], [[0, 0, 1, 1]])


def test_original_float_gt_and_independent_id_override_quantized_answer():
    original = [0.10049, 0.19951, 0.70049, 0.79951]
    row = convert_record(swift_record(), targets={"original:1": original},
                         gt_id_key="origin_query_id")
    assert row["solution"] == original
    assert row["solution_source"] == "original_gt_map"
    assert row["solution_id"] == "original:1"
    assert row["messages"] == [{"role": "user", "content": PROMPT}]
    assert row["images"] == swift_record()["images"]
    assert "0.10049" not in json.dumps(row["messages"])


def test_quantized_target_requires_explicit_choice_and_keeps_provenance():
    with pytest.raises(ValueError, match="allow-quantized-targets"):
        convert_record(swift_record())
    row = convert_record(swift_record(), allow_quantized_targets=True)
    assert row["solution"] == [0.1, 0.2, 0.7, 0.8]
    assert row["solution_source"] == "quantized_sft_bbox_2d_1000"
    with pytest.raises(ValueError, match="no original GT"):
        convert_record(swift_record(), targets={}, allow_quantized_targets=True)


def test_record_bbox_keeps_original_precision_and_map_has_precedence():
    source = swift_record()
    source["bbox"] = [0.10049, 0.19951, 0.70049, 0.79951]
    row = convert_record(source)
    assert row["solution"] == source["bbox"]
    assert row["solution_source"] == "original_record_bbox"
    assert "bbox" not in row and "0.10049" not in json.dumps(row["messages"])
    mapped = [0.1, 0.2, 0.7, 0.8]
    assert convert_record(source, targets={source["id"]: mapped})["solution"] == mapped
    source["bbox"] = [0, 0, 1000, 1000]
    with pytest.raises(ValueError, match="solution"):
        convert_record(source, allow_quantized_targets=True)


def test_native_and_swift_conversions_match():
    swift = swift_record()
    native = {"id": swift["id"], "image": swift["images"], "conversations": [
        {"from": "human", "value": PROMPT},
        {"from": "gpt", "value": swift["messages"][1]["content"]},
    ]}
    converted = convert_record(native, allow_quantized_targets=True)
    expected = convert_record(swift, allow_quantized_targets=True)
    for key in ("messages", "images", "solution", "solution_source"):
        assert converted[key] == expected[key]


def test_single_rgb_native_image_string():
    native = {"id": "rgb:1", "image": "/root/rgb/a.png", "conversations": [
        {"from": "human", "value": '<image>\nLocate the person, output bbox_2d 0-1000.'},
        {"from": "gpt", "value": '{"bbox_2d":[100,200,700,800]}'},
    ]}
    row = convert_record(native, allow_quantized_targets=True)
    assert row["images"] == ["/root/rgb/a.png"]
    assert row["solution"] == [0.1, 0.2, 0.7, 0.8]


def test_actual_swift_registration_when_available():
    rewards = pytest.importorskip("swift.rewards", reason="本机未安装ms-swift，不模拟训练接口")
    from tools.swift_grounding_reward import Acc05BBoxReward

    assert rewards.orms["acc05_bbox"] is Acc05BBoxReward
    assert isinstance(Acc05BBoxReward(), rewards.ORM)
    assert Acc05BBoxReward()(['{"bbox_2d":[0,0,500,1000]}'], [[0, 0, 1, 1]]) == [1.1]


def test_no_hidden_assistant_answers_or_image_count_mismatch():
    row = swift_record()
    row["messages"] += [{"role": "user", "content": "check again"}]
    with pytest.raises(ValueError, match="single user turn"):
        convert_record(row, allow_quantized_targets=True)
    row = swift_record()
    row["images"] = row["images"][:2]
    with pytest.raises(ValueError, match="placeholder count"):
        convert_record(row, allow_quantized_targets=True)


def test_conversion_preserves_order_repetitions_and_relative_paths(tmp_path):
    source = tmp_path / "sft.jsonl"
    output = tmp_path / "rl.jsonl"
    row = swift_record()
    row["images"] = ["rgb/a.png", "ir/a.png", "depth_rgb/a.png"]
    source.write_text((json.dumps(row) + "\n") * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="相对路径"):
        convert(source, output, allow_quantized_targets=True)
    report = convert(source, output, data_root=tmp_path, allow_quantized_targets=True)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert report["records"] == 2 and report["unique_ids"] == 1
    assert report["images_per_record"] == {3: 2}
    assert rows[0] == rows[1]
    assert rows[0]["images"][0] == str(tmp_path / "rgb/a.png")


def test_gt_map_rejects_bad_raw_values(tmp_path):
    path = tmp_path / "gt.json"
    path.write_text(json.dumps({"sample:1": {"bbox": [False, 0, 1, 1]}}), encoding="utf-8")
    with pytest.raises(ValueError, match="solution"):
        load_targets(path)


REAL_MANIFEST = Path(__file__).resolve().parents[2] / "results" / "arm5_artifacts_20261005" / "slices_arm5_mixture.jsonl"


@pytest.mark.skipif(not REAL_MANIFEST.is_file(), reason="Arm 5训练清单不在本机")
def test_real_arm5_manifest_conversion_and_self_reward(tmp_path):
    output = tmp_path / "arm5_rl.jsonl"
    report = convert(REAL_MANIFEST, output, allow_quantized_targets=True)
    source = [json.loads(line) for line in REAL_MANIFEST.read_text(encoding="utf-8").splitlines()]
    converted = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert report["records"] == len(source) == len(converted)
    assert report["solution_sources"] == {"quantized_sft_bbox_2d_1000": len(source)}
    for original, row in zip(source, converted):
        assert row["id"] == original["id"]
        assert row["messages"] == original["messages"][:-1]
        assert row["images"] == original["images"]
        assert grounding_reward(original["messages"][-1]["content"], row["solution"]) == pytest.approx(1.1)
