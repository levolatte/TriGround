"""`convert_rgbt_groundbench` 的回归测试。

存在的理由：这次转换里有三个**写错不会报错、只会静默降低质量或整源丢失**的地方——
（1）源标注是 xywh 绝对像素，按 xyxy 解释不抛异常但框全错；
（2）类别 id 每个源独立，跨源复用会得到错误类别名；
（3）实测的 tar 结构是 `rgbtvg_<source>/rgbtvg_<source>_train.pth`，
     路径写错会让整个源被静默跳过（stats 里只记一行 skipped）。
"""

from __future__ import annotations

import json
import pickle
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import convert_rgbt_groundbench as module  # noqa: E402
from tools.convert_rgbt_groundbench import (  # noqa: E402
    CLASS_NAMES,
    convert,
    is_degenerate,
    read_annotations,
)

ROW = ["A.jpg", {"height": 10, "width": 10}, [0, 0, 5, 5], "a car", 2, "small", 0, 1, 3, 0]


def write_pth(path: Path, rows: list) -> None:
    """造一个与真实 .pth 同结构的 torch ZIP 归档。成员名带源前缀，按后缀匹配。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("flir_train/data.pkl", pickle.dumps(rows))


def write_source(root: Path, source: str, rows: list) -> None:
    """写进**转换器实际会读的那个路径**（由 ANN_LAYOUT 决定），避免测试与实现脱节。"""
    directory, member = module.ANN_LAYOUT[source]
    write_pth(root / directory / member, rows)


def make_root(tmp_path: Path, source: str = "flir") -> Path:
    """按实测确认的真实结构造目录（不写假 .pth，由 write_source 负责）。"""
    root = tmp_path / "rgbt"
    (root / f"data_{source}" / "image_data" / source / "rgb").mkdir(parents=True)
    (root / f"data_{source}" / "image_data" / source / "ir").mkdir(parents=True)
    (root / f"ann_{source}.tar").write_bytes(b"placeholder")
    return root


def add_images(root: Path, source: str, stems: list[str]) -> None:
    for stem in stems:
        for sub in ("rgb", "ir"):
            (root / f"data_{source}" / "image_data" / source / sub / f"{stem}.jpg").write_bytes(b"x")


def test_layout_matches_measured_tar_members() -> None:
    """目录/分片名必须与 `tar -tf ann_*.tar` 实测结果一致，否则整个源被静默跳过。"""
    assert module.ANN_LAYOUT["flir"] == ("rgbtvg_flir", "rgbtvg_flir_train.pth")
    assert module.ANN_LAYOUT["m3fd"] == ("rgbtvg_m3fd", "rgbtvg_m3fd_train.pth")
    assert module.ANN_LAYOUT["mfad"] == ("rgbtvg_mfad", "rgbtvg_mfad_train.pth")


def test_read_annotations_without_torch(tmp_path: Path) -> None:
    """真实 .pth 是 torch ZIP 归档，标准库就能读——不能假设 torch 可用。"""
    path = tmp_path / "x.pth"
    write_pth(path, [ROW])
    assert read_annotations(path) == [ROW]


def test_missing_unpacked_annotation_is_reported(tmp_path: Path) -> None:
    """没解包时必须记进 skipped，而不是静默产出 0 条。"""
    root = make_root(tmp_path)
    add_images(root, "flir", ["A"])
    stats = convert(root, tmp_path / "o.jsonl", 0)
    assert stats["total"] == 0
    assert "未解包" in stats["skipped"]["flir"]


@pytest.mark.parametrize(
    ("box", "size", "expected"),
    [
        # [x, y, w, h] 绝对像素 -> 归一化 xyxy
        ([0, 0, 10, 10], {"height": 10, "width": 10}, [0, 0, 1000, 1000]),
        ([5, 2, 3, 4], {"height": 10, "width": 20}, [250, 200, 400, 600]),
    ],
)
def test_bbox_is_xywh_not_xyxy(tmp_path: Path, box, size, expected) -> None:
    """把源标注当 xyxy 解释不会报错，但框会整体错位——必须按 xywh 换算式处理。"""
    root = make_root(tmp_path)
    add_images(root, "flir", ["FLIR_00001"])
    write_source(root, "flir",
                 [["FLIR_00001.jpg", size, box, "a person", 3, "small", 0, 1, 3, 0]])
    out = tmp_path / "out.jsonl"
    stats = convert(root, out, 0)
    assert stats["total"] == 1, stats
    row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert json.loads(row["messages"][1]["content"])["bbox_2d"] == expected


def test_out_of_range_box_is_skipped(tmp_path: Path) -> None:
    """越界框不能进训练集。"""
    root = make_root(tmp_path)
    add_images(root, "flir", ["A"])
    write_source(root, "flir",
                 [["A.jpg", {"height": 10, "width": 10}, [0, 0, 999, 5], "q", 2, "small", 0, 1, 3, 0]])
    assert convert(root, tmp_path / "o.jsonl", 0)["total"] == 0


def test_class_ids_are_per_source() -> None:
    """同一个 id 在不同源里是不同类别，映射表必须分开。"""
    assert CLASS_NAMES["flir"][3] == "person"
    assert CLASS_NAMES["mfad"][3] == "truck"
    assert CLASS_NAMES["flir"] != CLASS_NAMES["mfad"]


def test_unknown_class_id_is_skipped(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    add_images(root, "flir", ["A"])
    write_source(root, "flir",
                 [["A.jpg", {"height": 10, "width": 10}, [0, 0, 5, 5], "q", 99, "small", 0, 1, 3, 0]])
    stats = convert(root, tmp_path / "o.jsonl", 0)
    assert stats["total"] == 0
    assert stats["by_source"]["flir"]["unknown_class"] == 1


@pytest.mark.parametrize(
    ("query", "degenerate"),
    [
        ("Person standing near a fence", False),
        ("No person is present in this scene.", True),
        ("There is no car here", True),
    ],
)
def test_degenerate_templates_filtered(query: str, degenerate: bool) -> None:
    assert is_degenerate(query) is degenerate


def test_depth_is_marked_missing_and_prompt_says_so(tmp_path: Path) -> None:
    """RGBT 没有深度：必须标为缺失且 prompt 显式告知，否则模型以为深度真的存在。"""
    root = make_root(tmp_path)
    add_images(root, "flir", ["A"])
    write_source(root, "flir", [ROW])
    out = tmp_path / "o.jsonl"
    assert convert(root, out, 0)["total"] == 1
    row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert row["missing_modalities"] == ["depth"]
    assert "unavailable" in row["messages"][0]["content"]
    assert len(row["images"]) == 2  # 只有 rgb + ir


def test_output_is_valid_swift_jsonl(tmp_path: Path) -> None:
    """产物必须与现有 swift 流程同构：一轮 user/assistant，三模态模板三个占位符。"""
    root = make_root(tmp_path)
    add_images(root, "flir", ["A"])
    write_source(root, "flir", [ROW])
    out = tmp_path / "o.jsonl"
    convert(root, out, 0)
    row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert [m["role"] for m in row["messages"]] == ["user", "assistant"]
    assert row["messages"][0]["content"].count("<image>") == 3
    assert module.NORMALIZE == 1000.0


def test_limit_per_source(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    add_images(root, "flir", ["A", "B", "C"])
    write_source(root, "flir",
                 [["A.jpg", {"height": 10, "width": 10}, [0, 0, 5, 5], "q", 2, "s", 0, 1, 3, 0],
                  ["B.jpg", {"height": 10, "width": 10}, [0, 0, 5, 5], "q", 2, "s", 0, 1, 3, 0],
                  ["C.jpg", {"height": 10, "width": 10}, [0, 0, 5, 5], "q", 2, "s", 0, 1, 3, 0]])
    assert convert(root, tmp_path / "o.jsonl", 2)["total"] == 2
