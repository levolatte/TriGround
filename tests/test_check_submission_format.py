"""`tools/check_submission_format.py` 的边界测试。

这个检查器是提交前的最后一道闸门：官方明确写着"不可修改除 bbox 字段外的其他字段，
否则将导致评测程序匹配失败"，而这类错误在本地不会报错、只会在平台上静默失分。
因此每条规则都要有对应的反例测试。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.check_submission_format import check_box, main  # noqa: E402


def write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def base_queries() -> dict:
    return {
        "q1": {"visible": "Images/visible/000001.png", "infrared": "Images/infrared/000001.png",
               "depth": "Images/depth/000001.png", "query": "A squatting person"},
        "q2": {"visible": "Images/visible/000002.png", "infrared": "Images/infrared/000002.png",
               "depth": "Images/depth/000002.png", "query": "A red car"},
    }


def valid_submission() -> dict:
    payload = json.loads(json.dumps(base_queries()))
    payload["q1"]["bbox"] = [0.1, 0.2, 0.3, 0.4]
    payload["q2"]["bbox"] = [0.5, 0.5, 0.9, 0.9]
    return payload


def run_check(tmp_path: Path, queries: dict, submission: dict) -> int:
    """返回退出码：合法路径正常返回（不调 sys.exit），不合格路径抛 SystemExit。"""
    query_path = write_json(tmp_path / "queries.json", queries)
    submission_path = write_json(tmp_path / "predictions.json", submission)
    sys.argv = ["check", "--queries", str(query_path), "--submission", str(submission_path)]
    try:
        main()
    except SystemExit as exit_error:
        return int(exit_error.code or 0)
    return 0


def test_valid_submission_passes(tmp_path: Path) -> None:
    assert run_check(tmp_path, base_queries(), valid_submission()) == 0


def test_changed_query_text_is_rejected(tmp_path: Path) -> None:
    submission = valid_submission()
    submission["q1"]["query"] = "A standing person"
    assert run_check(tmp_path, base_queries(), submission) == 1


def test_removed_field_is_rejected(tmp_path: Path) -> None:
    submission = valid_submission()
    del submission["q2"]["infrared"]
    assert run_check(tmp_path, base_queries(), submission) == 1


def test_reordered_ids_are_rejected(tmp_path: Path) -> None:
    submission = valid_submission()
    reordered = {"q2": submission["q2"], "q1": submission["q1"]}
    assert run_check(tmp_path, base_queries(), reordered) == 1


@pytest.mark.parametrize(
    ("box", "reason"),
    [
        ([0.5, 0.5, 0.4, 0.6], "反向坐标"),
        ([0.1, 0.1, 1.2, 0.5], "越界"),
        ([0.1, 0.1, 0.5], "空框"),
        ([float("nan"), 0.1, 0.5, 0.5], "NaN"),
        ("not-a-box", "空框"),
    ],
)
def test_invalid_boxes_are_caught(box: object, reason: str) -> None:
    assert check_box(box) is not None


def test_valid_box_returns_none() -> None:
    assert check_box([0.0, 0.0, 1.0, 1.0]) is None
