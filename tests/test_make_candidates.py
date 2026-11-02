"""`make_candidates` 的任务边界：解析失败的样本如何处理、代表框如何取。

存在理由：自一致性采样投票的链路里，这一步是唯一把「一题多样本」变成「一题多候选」的地方。
它错的方式很隐蔽——把不可解析的样本当成合法候选，会让聚合器在噪声上投票，
分数看起来正常但没有意义；把全失败的题丢掉，会让分母变小、分数虚高。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import make_candidates as mc  # noqa: E402

GOOD_A = '{"bbox_2d":[100,200,300,400]}'
GOOD_B = '{"bbox_2d":[110,210,310,410]}'


def test_parses_each_sample_and_keeps_sample_index():
    rows = [{"id": "q1", "samples": [{"text": "garbage"}, {"text": GOOD_A}, {"text": GOOD_B}]}]
    out, stats = mc.convert(rows, "short")

    assert [c["bbox"] for c in out[0]["candidates"]] == [
        [0.1, 0.2, 0.3, 0.4], [0.11, 0.21, 0.31, 0.41]]
    # sample_index 要指向原始样本序号，否则事后无法回查是第几个样本胜出
    assert [c["sample_index"] for c in out[0]["candidates"]] == [1, 2]
    assert stats["samples"] == 3 and stats["parsed_samples"] == 2
    assert stats["sample_parse_rate"] == pytest.approx(2 / 3, abs=1e-4)


def test_question_with_no_parseable_sample_is_kept_empty():
    """全失败的题**保留但 candidates 为空**：聚合器会记为 all_members_invalid 并拒绝打包。"""
    rows = [{"id": "q1", "samples": [{"text": "nope"}]},
            {"id": "q2", "samples": [{"text": GOOD_A}]}]
    out, stats = mc.convert(rows, "short")

    assert out[0] == {"id": "q1", "candidates": []}
    assert len(out) == 2  # 分母不变
    assert stats["questions_without_any_candidate"] == 1
    assert stats["empty_examples"] == ["q1"]


def test_prefers_upstream_bbox_and_rejects_invalid_one():
    """上游已给 bbox 时用它；但非法框仍要当作解析失败，不能放行。"""
    rows = [{"id": "q1", "samples": [
        {"bbox": [0.1, 0.1, 0.2, 0.2], "text": "ignored"},
        {"bbox": [0.5, 0.5, 0.4, 0.6], "text": "inverted"},       # x2 < x1，非法
        {"bbox": None, "text": GOOD_B},
    ]}]
    out, _ = mc.convert(rows, "short")
    boxes = [c["bbox"] for c in out[0]["candidates"]]
    assert boxes[0] == [0.1, 0.1, 0.2, 0.2]
    assert boxes[-1] == [0.11, 0.21, 0.31, 0.41]
    assert len(boxes) == 2


def test_text_with_surrounding_prose_is_not_parsed():
    """`parse_bbox` 要求整段文本恰好是一个 JSON——前后有任何文字都算失败。

    这条不是恒真式：思考模式下推理文字在 `</think>` 之前、由 thinking 解析器单独处理，
    短模式下则必然失败。把这件事钉成测试，避免以后有人误以为可以容忍前缀。
    """
    rows = [{"id": "q1", "samples": [{"text": f"答案是 {GOOD_A} 谢谢"}]}]
    out, stats = mc.convert(rows, "short")
    assert out[0]["candidates"] == []
    assert stats["parsed_samples"] == 0


def test_missing_samples_and_blank_id_fail_loudly():
    with pytest.raises(ValueError, match="缺少 samples"):
        mc.convert([{"id": "q1"}], "short")
    with pytest.raises(ValueError, match="非空 id"):
        mc.convert([{"id": "", "samples": [{"text": GOOD_A}]}], "short")


def test_cli_writes_candidates_and_stats(tmp_path):
    source = tmp_path / "raw.jsonl"
    source.write_text(
        json.dumps({"id": "q1", "samples": [{"text": GOOD_A}, {"text": "bad"}]}) + "\n"
        + json.dumps({"id": "q2", "samples": [{"text": "bad"}]}) + "\n", encoding="utf-8")
    out_path = tmp_path / "cands.jsonl"
    stats_path = tmp_path / "stats.json"
    argv = sys.argv
    sys.argv = ["make_candidates", "--input", str(source), "--output", str(out_path),
                "--out-stats", str(stats_path)]
    try:
        mc.main()
    finally:
        sys.argv = argv

    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert [len(row["candidates"]) for row in rows] == [1, 0]
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    assert stats["questions"] == 2 and stats["questions_without_any_candidate"] == 1
