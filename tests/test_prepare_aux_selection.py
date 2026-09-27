import json

from tools.prepare_aux_selection import inference_row, select_training_smoke, verify_env


def source_row(i):
    return {"id": str(i), "image": [f"visible/{i}.png", f"infrared/{i}.png", f"target_v2/qwen3vl_native_sft/depth_rgb/{i}.png"],
            "class_name": "DO_NOT_COPY", "bbox": [0, 0, 1, 1],
            "conversations": [{"from": "human", "value": "<image>\nLocate the object described by this query: Near car\nReturn JSON"},
                              {"from": "gpt", "value": "DO_NOT_COPY"}]}


def test_gt_free_boundary_and_raw_depth_path():
    row = inference_row(source_row(3), "/data/city/train")
    assert set(row) == {"id", "query", "images", "depth_encoding"}
    assert row["images"]["depth_raw"] == "/data/city/train/depth/3.png"
    assert "DO_NOT_COPY" not in json.dumps(row)
    assert row["query"] == "Near car"


def test_smoke_is_fixed_unique_groups():
    rows = [source_row(i) for i in range(20)]
    rows += [dict(rows[0], id="duplicate_image")]
    a = select_training_smoke(rows)
    assert a == select_training_smoke(rows)
    assert len(a) == len({tuple(row["image"]) for row in a}) == 16


def test_environment_requires_exact_outputs(tmp_path):
    row = {"prediction": [.1, .2, .3, .4], "raw_text": "answer", "image_grid_thw": [[1, 2, 3]], "prompt": "prompt"}
    expected, actual = tmp_path / "expected.json", tmp_path / "actual.jsonl"
    expected.write_text(json.dumps({"a": row}))
    actual.write_text(json.dumps({"id": "a", **row}) + "\n")
    verify_env(expected, actual)
