"""清单转换器的回归测试。

覆盖两件事：
1. 单位行为：绝对/相对路径、占位符数量、答案必须含合法 0--1000 bbox；
2. 真实清单的完整性：4800 次呈现 / 660 个唯一图组 / 每条 3 图 / 文本逐字不变 /
   与 412 条开发集**图组零交集**（这是防泄漏的硬门槛）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.convert_manifest_to_swift import convert, convert_record, is_absolute_path

REPO = Path(__file__).resolve().parents[1]
TRAIN_MANIFEST = Path(
    r"F:\AIC\results\triground_abv_execution_20260928\deployment_600_seed2026\manifests\A.json"
)
VAL_MANIFEST = Path(r"F:\AIC\results\triground_abv_20260927\inputs\city_val.json")

PROMPT = ("<image>\n<image>\n<image>\nLocate the object described by this query: a person\n"
          'Return only JSON in this exact form with coordinates normalized to 0-1000: {"bbox_2d":[x1,y1,x2,y2]}')


def make_record(**overrides) -> dict:
    record = {
        "id": "unit:0",
        "scene_id": "city:000001_001_00000001",
        "image": ["/root/data/visible/a.png", "/root/data/infrared/a.png", "/root/data/depth_rgb/a.png"],
        "conversations": [
            {"from": "human", "value": PROMPT},
            {"from": "gpt", "value": '{"bbox_2d":[1,2,3,4]}'},
        ],
    }
    record.update(overrides)
    return record


def test_posix_absolute_paths_are_not_treated_as_relative():
    # Windows 的 Path.is_absolute() 对 /root/... 返回 False，这里锁住跨平台行为
    assert is_absolute_path("/root/data/visible/a.png")
    assert not is_absolute_path("visible/a.png")


def test_converts_one_record_to_swift_messages(tmp_path):
    row = convert_record(make_record(), None)
    assert [m["role"] for m in row["messages"]] == ["user", "assistant"]
    assert row["messages"][0]["content"].count("<image>") == 3
    assert row["images"] == make_record()["image"]
    assert row["messages"][1]["content"] == '{"bbox_2d":[1,2,3,4]}'
    assert row["scene_id"] == "city:000001_001_00000001"


def test_relative_paths_require_and_use_data_root():
    record = make_record(image=["visible/a.png", "infrared/a.png", "depth_rgb/a.png"])
    with pytest.raises(ValueError, match="相对路径"):
        convert_record(record, None)
    row = convert_record(record, Path("/root/data"))
    assert row["images"][0] == str(Path("/root/data") / "visible/a.png")


def test_placeholder_count_must_match_images():
    record = make_record()
    record["conversations"][0]["value"] = PROMPT.replace("<image>\n", "", 1)
    with pytest.raises(ValueError, match="占位符"):
        convert_record(record, None)


def test_answer_must_contain_valid_bbox():
    record = make_record()
    record["conversations"][1]["value"] = "the person is on the left"
    with pytest.raises(ValueError, match="bbox"):
        convert_record(record, None)


@pytest.mark.skipif(not TRAIN_MANIFEST.exists(), reason="训练清单不在本机")
def test_real_training_manifest_is_lossless(tmp_path):
    source = json.loads(TRAIN_MANIFEST.read_text(encoding="utf-8"))
    output = tmp_path / "a_city_swift.jsonl"
    report = convert(TRAIN_MANIFEST, output, None)

    assert report["records"] == 4800
    assert report["unique_scene_ids"] == 660
    assert report["images_per_record"] == {3: 4800}

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len(source) == 4800
    # 保序 + 文本逐字不变（只换外壳，不改内容）
    for row, original in ((rows[0], source[0]), (rows[-1], source[-1])):
        assert row["messages"][0]["content"] == original["conversations"][0]["value"]
        assert row["messages"][1]["content"] == original["conversations"][1]["value"]
        assert row["images"] == original["image"]
        assert row["id"] == original["id"]
    assert all(len(row["images"]) == 3 for row in rows)


@pytest.mark.skipif(not (TRAIN_MANIFEST.exists() and VAL_MANIFEST.exists()),
                    reason="训练或开发清单不在本机")
def test_train_and_dev_image_groups_do_not_overlap():
    """防泄漏硬门槛：训练图组与 412 条开发集图组必须零交集。"""
    def groups(path: Path) -> set[str]:
        found: set[str] = set()

        def walk(node) -> None:
            if isinstance(node, dict):
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)
            elif isinstance(node, str) and value_looks_like_image(node):
                found.add(Path(node).stem)

        def value_looks_like_image(text: str) -> bool:
            return text.lower().endswith((".png", ".jpg", ".jpeg"))

        walk(json.loads(path.read_text(encoding="utf-8")))
        return found

    assert groups(TRAIN_MANIFEST) & groups(VAL_MANIFEST) == set()
