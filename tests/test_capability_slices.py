"""`build_capability_slices` 的回归测试。

覆盖计划里列出的四个必测项，外加两个**写错不会报错、只会静默产出坏数据**的地方：
（1）RGBT 源标注是 `[x, y, w, h]` 绝对像素，按 xyxy 解释不抛异常但框整体错位；
（2）`尺寸档 == 'small'` 只是极小目标的**超集**（实测 small 里仅约 47% 真的 <0.2%），
     若只信标签就会把大量小目标混进"极小目标"切片，污染配比。
另外 `normal` 标签中面积 <0.2% 的占比实测为 0.0%，这是"标签是安全超集"的依据，也钉住。
"""

from __future__ import annotations

import json
import pickle
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import build_capability_slices as module  # noqa: E402
from tools.build_capability_slices import (  # noqa: E402
    SliceError,
    build_city,
    build_rgbt,
    make_record,
    mix,
    read_pth,
    to_answer,
)


def write_pth(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("flir_train/data.pkl", pickle.dumps(rows))


def rgbt_row(filename: str, size: dict, box: list, query: str, class_id: int,
             size_class: str = "small", occlusion: int = 0) -> list:
    return [filename, size, box, query, class_id, size_class, occlusion, 1, 2, 0]


def make_rgbt_root(tmp_path: Path, rows: list, source: str = "flir") -> Path:
    root = tmp_path / "rgbt"
    write_pth(root / "raw" / f"rgbtvg_{source}" / f"rgbtvg_{source}_train.pth", rows)
    for sub in ("rgb", "ir"):
        (root / "raw" / "image_data" / source / sub).mkdir(parents=True, exist_ok=True)
    for row in rows:
        for sub in ("rgb", "ir"):
            (root / "raw" / "image_data" / source / sub / row[0]).write_bytes(b"x")
    return root


def test_read_pth_without_torch(tmp_path: Path) -> None:
    """真实 .pth 是 torch ZIP 归档，标准库即可读。"""
    path = tmp_path / "x.pth"
    rows = [rgbt_row("A.jpg", {"height": 10, "width": 10}, [1, 2, 3, 4], "q", 3)]
    write_pth(path, rows)
    assert read_pth(path) == rows


def test_to_answer_rejects_degenerate_box() -> None:
    for bad in ((0.5, 0.5, 0.5, 0.9), (0.9, 0.1, 0.2, 0.5), (-0.1, 0.1, 0.5, 0.5),
                (0.1, 0.1, 1.2, 0.5)):
        with pytest.raises(SliceError):
            to_answer(bad)


def test_to_answer_quantizes_to_1000() -> None:
    assert json.loads(to_answer((0.0, 0.0, 1.0, 1.0)))["bbox_2d"] == [0, 0, 1000, 1000]
    assert json.loads(to_answer((0.25, 0.5, 0.5, 0.75)))["bbox_2d"] == [250, 500, 500, 750]


def test_rgbt_bbox_is_xywh_not_xyxy(tmp_path: Path) -> None:
    """把 xywh 当 xyxy 解释不会报错，但框会整体错位——必须按换算式处理。

    1000x1000 图上 [10, 20, 30, 40] 作为 xywh -> 归一化 (0.01, 0.02, 0.04, 0.06)，
    面积 0.0012 < 0.002，因此落在极小目标切片里，便于直接检查换算结果。
    """
    rows = [rgbt_row("A.jpg", {"height": 1000, "width": 1000}, [10, 20, 30, 40],
                     "a small person", 3, "small", 0)]
    root = make_rgbt_root(tmp_path, rows)
    slices = build_rgbt(root, __import__("random").Random(0), sources=("flir",))
    assert len(slices["small"]) == 1
    answer = json.loads(slices["small"][0]["messages"][1]["content"])["bbox_2d"]
    assert answer == [10, 20, 40, 60]


def test_tiny_slice_uses_real_area_not_the_label(tmp_path: Path) -> None:
    """核心：`small` 标签是超集，切片必须按**真实面积**精筛。

    构造两条都标 `small` 但面积不同的样本：只有真正 <0.2% 的那条该进 small 切片。
    """
    tiny = rgbt_row("tiny.jpg", {"height": 1000, "width": 1000}, [0, 0, 10, 10],
                    "a tiny object", 3, "small", 0)          # 面积 1e-4 < 0.002
    labelled_small_but_big = rgbt_row("big.jpg", {"height": 1000, "width": 1000},
                                      [0, 0, 100, 100], "a bigger object", 3, "small", 0)
    root = make_rgbt_root(tmp_path, [tiny, labelled_small_but_big])
    slices = build_rgbt(root, __import__("random").Random(0), sources=("flir",))
    ids = [row["id"] for row in slices["small"]]
    assert any("tiny" in i for i in ids)
    assert not any("big" in i for i in ids), "标了 small 但面积超阈值的不该进极小切片"
    for row in slices["small"]:
        assert row["area"] < module.TINY_AREA


def test_normal_label_never_enters_tiny_slice(tmp_path: Path) -> None:
    """实测依据：三个源的 normal 标签里面积 <0.2% 的占比均为 0.0%。"""
    normal = rgbt_row("N.jpg", {"height": 1000, "width": 1000}, [0, 0, 300, 300],
                      "a large object", 3, "normal", 0)
    root = make_rgbt_root(tmp_path, [normal])
    slices = build_rgbt(root, __import__("random").Random(0), sources=("flir",))
    assert slices["small"] == []


def test_long_query_and_occlusion_slices(tmp_path: Path) -> None:
    long_q = "x" * (module.LONG_QUERY_CHARS + 5)
    rows = [rgbt_row("A.jpg", {"height": 1000, "width": 1000}, [0, 0, 300, 300],
                     long_q, 3, "normal", 0),
            rgbt_row("B.jpg", {"height": 1000, "width": 1000}, [0, 0, 300, 300],
                     "short query", 3, "normal", 2)]
    root = make_rgbt_root(tmp_path, rows)
    slices = build_rgbt(root, __import__("random").Random(0), sources=("flir",))
    assert len(slices["longquery"]) == 1
    assert len(slices["discriminate"]) == 1          # occlusion > 0
    assert slices["longquery"][0]["query_chars"] > module.LONG_QUERY_CHARS


def test_degenerate_query_filtered(tmp_path: Path) -> None:
    rows = [rgbt_row("A.jpg", {"height": 100, "width": 100}, [0, 0, 10, 10],
                     "No person is present in this scene.", 3, "small", 0)]
    root = make_rgbt_root(tmp_path, rows)
    result = build_rgbt(root, __import__("random").Random(0), sources=("flir",))
    assert all(not v for v in result.values()), result


def test_unknown_class_filtered(tmp_path: Path) -> None:
    rows = [rgbt_row("A.jpg", {"height": 100, "width": 100}, [0, 0, 10, 10],
                     "a mystery", 99, "small", 0)]
    root = make_rgbt_root(tmp_path, rows)
    result = build_rgbt(root, __import__("random").Random(0), sources=("flir",))
    assert all(not v for v in result.values()), result


def test_missing_image_raises_not_silently_skipped(tmp_path: Path) -> None:
    """图像缺失必须硬失败：静默跳过的切片会让人以为数据都用上了。"""
    rows = [rgbt_row("A.jpg", {"height": 1000, "width": 1000}, [0, 0, 10, 10],
                     "a tiny thing", 3, "small", 0)]
    root = make_rgbt_root(tmp_path, rows)
    (root / "raw" / "image_data" / "flir" / "ir" / "A.jpg").unlink()
    with pytest.raises(SliceError, match="图像缺失"):
        build_rgbt(root, __import__("random").Random(0), sources=("flir",))


def test_make_record_marks_missing_modalities(tmp_path: Path) -> None:
    """两模态来源必须显式标注缺失模态，否则模型以为深度真的存在。"""
    image = tmp_path / "a.png"
    image.write_bytes(b"x")
    record = make_record("s1", [str(image)], "a car", (0.1, 0.1, 0.5, 0.5),
                         depth_policy="millimeter", missing=("depth",),
                         capability="small", source="rgbt_flir")
    assert record["missing_modalities"] == ["depth"]
    assert "unavailable" in record["messages"][0]["content"]
    assert record["messages"][0]["content"].count("<image>") == 3


def test_make_record_raises_on_missing_image(tmp_path: Path) -> None:
    with pytest.raises(SliceError, match="图像不存在"):
        make_record("s1", [str(tmp_path / "nope.png")], "q", (0.1, 0.1, 0.5, 0.5),
                    depth_policy="millimeter", missing=(), capability="small", source="x")


def test_mix_respects_quotas_and_warns(monkeypatch, capsys) -> None:
    slices = {"a": [{"id": f"a{i}"} for i in range(10)],
              "b": [{"id": f"b{i}"} for i in range(3)]}
    mixed = mix(slices, {"a": 5, "b": 8}, seed=1)
    assert sum(1 for r in mixed if r["id"].startswith("a")) == 5
    assert sum(1 for r in mixed if r["id"].startswith("b")) == 3   # 不足则全用
    assert "少于配额" in capsys.readouterr().out


def test_mix_raises_on_empty_slice() -> None:
    with pytest.raises(SliceError, match="切片为空"):
        mix({"a": []}, {"a": 5}, seed=1)


def test_city_scene_cap_limits_per_scene(tmp_path: Path) -> None:
    """scene_cap 的取舍：提高上限多用数据，但不取消上限（取消会改变分布）。"""
    image_root = tmp_path / "city"
    for sub in ("visible", "infrared", "depth"):
        (image_root / sub).mkdir(parents=True, exist_ok=True)
    records = {}
    for scene in ("s1", "s2"):
        for i in range(5):
            name = f"{scene}_0000000{i}.png"
            for sub in ("visible", "infrared", "depth"):
                (image_root / sub / name).write_bytes(b"x")
            records[f"{scene}_{i}"] = {
                "visible": f"visible/{name}", "infrared": f"infrared/{name}",
                "depth": f"depth/{name}", "query": f"q{i}",
                "bbox": [0.1, 0.1, 0.5, 0.5], "class_name": "car"}
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps(records), encoding="utf-8")
    rows = build_city(manifest, image_root, scene_cap=2,
                      rng=__import__("random").Random(0))
    assert len(rows) == 4, "2 个场景 × 上限 2 = 4 条"
    assert all(r["capability"] == "city_anchor" for r in rows)
    assert all(r["missing_modalities"] == [] for r in rows)
