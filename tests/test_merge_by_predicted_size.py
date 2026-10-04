"""`merge_by_predicted_size` 的回归测试。

重点在**两端的恒等性**：阈值 0 必须逐题等于基线、阈值 ≥1 必须逐题等于放大版。
这两条把"分路"这个动作钉死了——如果面积算错口径（例如误用像素而非归一化坐标）、
或者分路方向写反，两端恒等就会失败。

打包走的是已验证的 `package_submission`，这里只验证产物结构。
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.merge_by_predicted_size import box_area, index_rows, route  # noqa: E402

SOURCE_IDS = ["q0", "q1", "q2"]


def _rows(bboxes: dict[str, list[float]]) -> dict[str, dict]:
    return {k: {"id": k, "bbox": v} for k, v in bboxes.items()}


# 面积：q0 = 0.1*0.1 = 0.01（大）；q1 = 0.01*0.01 = 0.0001（极小）；q2 = 0.05*0.05 = 0.0025（中）
BASE = _rows({"q0": [0.0, 0.0, 0.1, 0.1],
              "q1": [0.0, 0.0, 0.01, 0.01],
              "q2": [0.0, 0.0, 0.05, 0.05]})
UP = _rows({"q0": [0.2, 0.2, 0.3, 0.3],
            "q1": [0.4, 0.4, 0.41, 0.41],
            "q2": [0.6, 0.6, 0.65, 0.65]})


def test_box_area_uses_normalized_coordinates() -> None:
    assert box_area([0.0, 0.0, 0.1, 0.1]) == pytest.approx(0.01)
    assert box_area([0.4, 0.4, 0.41, 0.41]) == pytest.approx(0.0001)


def test_threshold_zero_takes_base_everywhere() -> None:
    """阈值 0 -> 恒等变换为基线。分路方向写反会在这里暴露。"""
    chosen, stats = route(SOURCE_IDS, BASE, UP, 0.0)
    assert [chosen[s]["bbox"] for s in SOURCE_IDS] == [BASE[s]["bbox"] for s in SOURCE_IDS]
    assert stats["upscaled"] == 0
    assert stats["base"] == 3


def test_threshold_one_takes_upscaled_everywhere() -> None:
    """阈值 >=1 -> 恒等变换为放大版。"""
    chosen, stats = route(SOURCE_IDS, BASE, UP, 1.0)
    assert [chosen[s]["bbox"] for s in SOURCE_IDS] == [UP[s]["bbox"] for s in SOURCE_IDS]
    assert stats["upscaled"] == 3


def test_intermediate_threshold_routes_only_small_base_boxes() -> None:
    """只把**基线框面积小于阈值**的题换成放大版。"""
    chosen, stats = route(SOURCE_IDS, BASE, UP, 0.001)
    # 只有 q1（0.0001）小于 0.001
    assert chosen["q1"]["bbox"] == UP["q1"]["bbox"]
    assert chosen["q0"]["bbox"] == BASE["q0"]["bbox"]
    assert chosen["q2"]["bbox"] == BASE["q2"]["bbox"]
    assert stats == {"base": 2, "upscaled": 1, "missing_in_upscaled": 0}


def test_intermediate_threshold_boundary_is_exclusive() -> None:
    """面积恰好等于阈值时**不**替换（严格小于），避免边界行为含糊。"""
    chosen, _ = route(["q2"], BASE, UP, box_area(BASE["q2"]["bbox"]))
    assert chosen["q2"]["bbox"] == BASE["q2"]["bbox"]


def test_missing_upscaled_falls_back_and_counts() -> None:
    """放大版缺题时回落基线并计数——丢题会让提交少记录，绝不能静默。"""
    partial = _rows({"q1": UP["q1"]["bbox"]})
    chosen, stats = route(SOURCE_IDS, BASE, partial, 1.0)
    assert chosen["q0"]["bbox"] == BASE["q0"]["bbox"]
    assert chosen["q2"]["bbox"] == BASE["q2"]["bbox"]
    assert chosen["q1"]["bbox"] == UP["q1"]["bbox"]
    assert stats["missing_in_upscaled"] == 2
    assert len(chosen) == 3, "缺题必须回落，而不是从结果里消失"


def test_base_missing_is_a_hard_error() -> None:
    """基线缺题是硬错误：分路的设计前提就是两套都完整。"""
    with pytest.raises(ValueError, match="基线预测缺失"):
        route(SOURCE_IDS, _rows({"q0": BASE["q0"]["bbox"]}), UP, 0.006)


def test_index_rows_rejects_duplicates(tmp_path: Path) -> None:
    rows = [{"id": "a", "bbox": [0, 0, 1, 1]}, {"id": "a", "bbox": [0, 0, 1, 1]}]
    with pytest.raises(ValueError, match="id 重复"):
        index_rows(rows, tmp_path / "x.jsonl")


def test_index_rows_rejects_missing_bbox(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="缺少 bbox"):
        index_rows([{"id": "a"}], tmp_path / "x.jsonl")


def test_end_to_end_packages_a_valid_zip(tmp_path: Path) -> None:
    """跑一次完整的 main()，确认 zip 结构正确且记录齐全。"""
    from tools import merge_by_predicted_size as module

    queries = {s: {"visible": "v.png", "infrared": "i.png", "depth": "d.png",
                   "query": f"the object {s}"} for s in SOURCE_IDS}
    q_path = tmp_path / "queries.json"
    q_path.write_text(json.dumps(queries, ensure_ascii=False), encoding="utf-8")

    def write_rows(path: Path, table: dict[str, dict]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for s in SOURCE_IDS:
                handle.write(json.dumps({**table[s], "parsed": True}) + "\n")

    base_path, up_path = tmp_path / "base.jsonl", tmp_path / "up.jsonl"
    write_rows(base_path, BASE)
    write_rows(up_path, UP)

    out_dir = tmp_path / "out"
    monkey = sys.argv
    try:
        sys.argv = ["merge_by_predicted_size", "--base", str(base_path),
                    "--upscaled", str(up_path), "--queries", str(q_path),
                    "--out-dir", str(out_dir),
                    "--expected-query-count", "3", "--area-threshold", "0.001"]
        module.main()
    finally:
        sys.argv = monkey

    with zipfile.ZipFile(out_dir / "submission.zip") as archive:
        assert archive.namelist() == ["predictions.json"]
        packed = json.loads(archive.read("predictions.json"))
    assert set(packed) == set(SOURCE_IDS)
    # q1 走了放大版，其余保持基线
    assert packed["q1"]["bbox"] == UP["q1"]["bbox"]
    assert packed["q0"]["bbox"] == BASE["q0"]["bbox"]
    summary = json.loads((out_dir / "merge_summary.json").read_text(encoding="utf-8"))
    assert summary["took_upscaled"] == 1 and summary["took_base"] == 2
