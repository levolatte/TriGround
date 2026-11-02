"""`dev_gt_from_jsonl` 的任务边界：只做格式转换，不做静默跳过。

存在理由：开发集是裁判。它出错必须立刻暴露——缺 `solution`、框非法、id 重复都被直接拒绝，
而不是跳过某一题。跳过会让分母变小、分数虚高，这类错误在竞赛里比程序报错危险得多。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import dev_gt_from_jsonl as dg  # noqa: E402


def test_converts_solution_to_normalized_box():
    rows = [{"id": "q1", "solution": [0.1, 0.2, 0.3, 0.4]},
            {"id": "q2", "solution": [0, 0, 1, 1]}]
    gt = dg.convert(rows)
    assert gt == {"q1": [0.1, 0.2, 0.3, 0.4], "q2": [0.0, 0.0, 1.0, 1.0]}


def test_missing_solution_fails_instead_of_skipping():
    with pytest.raises(ValueError, match="solution"):
        dg.convert([{"id": "q1", "solution": [0.1, 0.2, 0.3, 0.4]}, {"id": "q2"}])


def test_duplicate_id_fails():
    with pytest.raises(ValueError, match="重复"):
        dg.convert([{"id": "q", "solution": [0.1, 0.2, 0.3, 0.4]},
                    {"id": "q", "solution": [0.1, 0.2, 0.3, 0.4]}])


@pytest.mark.parametrize("box", [
    [0.5, 0.5, 0.4, 0.6],   # x2 < x1
    [-0.1, 0.1, 0.2, 0.2],  # 越界
    [0.1, 0.2, 0.3],        # 长度不足
    [0.1, 0.2, 0.3, 0.4, 0.5],
])
def test_invalid_box_is_rejected(box):
    with pytest.raises(ValueError):
        dg.convert([{"id": "q1", "solution": box}])


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="为空"):
        dg.convert([])


def test_cli_writes_json(tmp_path):
    source = tmp_path / "dev.jsonl"
    source.write_text(json.dumps({"id": "q1", "solution": [0.1, 0.2, 0.3, 0.4]}) + "\n",
                      encoding="utf-8")
    out_path = tmp_path / "gt.json"
    argv = sys.argv
    sys.argv = ["dev_gt_from_jsonl", "--input", str(source), "--output", str(out_path)]
    try:
        dg.main()
    finally:
        sys.argv = argv
    assert json.loads(out_path.read_text(encoding="utf-8")) == {"q1": [0.1, 0.2, 0.3, 0.4]}
