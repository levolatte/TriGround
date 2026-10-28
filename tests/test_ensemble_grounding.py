"""集成的任务边界：家族票权、无真值聚合、失败分母和正式输出。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import ensemble_grounding as eg  # noqa: E402


LEFT = [0.1, 0.2, 0.3, 0.5]
RIGHT = [0.7, 0.2, 0.9, 0.5]


def make_members(names, families=None):
    return [eg.Member(name, Path(name), (families or names)[i], 1) for i, name in enumerate(names)]


def test_load_three_formats_and_explicit_failed_prediction(tmp_path):
    rows = [
        ("a.jsonl", '{"id":"q","prediction":null,"bbox":[0,0,1,1]}\n'),
        ("b.json", json.dumps({"q": {"bbox": LEFT, "query": "person"}})),
        ("c.jsonl", json.dumps({"id": "q", "candidates": [{"bbox": RIGHT}, {"bbox": LEFT}]})),
    ]
    loaded = []
    for name, text in rows:
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        loaded.append(eg.load_predictions(path))
    assert eg.valid_boxes(loaded[0]["q"]) == []
    assert eg.valid_boxes(loaded[1]["q"]) == [tuple(LEFT)]
    assert len(eg.valid_boxes(loaded[2]["q"])) == 2


def test_input_duplicate_and_missing_member_ids_fail(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text(json.dumps({"id": "q", "bbox": LEFT}) + "\n", encoding="utf-8")
    b.write_text(json.dumps({"id": "other", "bbox": RIGHT}) + "\n", encoding="utf-8")
    config = tmp_path / "members.json"
    config.write_text(json.dumps({"members": [{"name": "a", "path": a.name}, {"name": "b", "path": b.name}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="ID集合"):
        eg.load_members(config)
    a.write_text(a.read_text() * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="id 重复"):
        eg.load_predictions(a)


@pytest.mark.parametrize("method", eg.METHODS)
def test_repeated_same_family_cannot_overvote_other_experts(method):
    # 坏模型复制十个同源视图仍仅一票；两个独立专家应决定位置。
    names = ["bad"] + [f"copy{i}" for i in range(9)] + ["good1", "good2"]
    members = make_members(names, ["bad"] * 10 + ["g1", "g2"])
    tables = {name: {"q": {"bbox": LEFT if i < 10 else RIGHT}} for i, name in enumerate(names)}
    candidates = eg.candidates_for_id("q", members, tables)
    assert sum(c.mass for c in candidates if c.family == "bad") == pytest.approx(1)
    fused = eg.aggregate(candidates, method)["bbox"]
    if method == "mean":
        assert fused == pytest.approx((np.array(LEFT) + 2 * np.array(RIGHT)) / 3)
    else:
        assert fused == pytest.approx(RIGHT)


def test_multiple_candidates_share_member_vote_and_failed_member_has_no_mass():
    members = make_members(["a1", "a2", "b"], ["a", "a", "b"])
    tables = {"a1": {"q": {"candidates": [{"bbox": LEFT}] * 20}},
              "a2": {"q": {"bbox": RIGHT, "fallback": True}}, "b": {"q": {"bbox": RIGHT}}}
    candidates = eg.candidates_for_id("q", members, tables)
    assert sum(c.mass for c in candidates if c.family == "a") == pytest.approx(0.5)
    assert eg.aggregate(candidates, "cluster_medoid")["bbox"] == RIGHT


@pytest.mark.parametrize("method", eg.METHODS)
def test_all_failed_is_explicit_and_kept_in_denominator(method):
    bad = {"parsed": True, "candidates": [{"bbox": [float("nan"), 0, 1, 1]}, {"bbox": [1, 0, 0, 1]}]}
    assert eg.valid_boxes(bad) == []
    row = eg.aggregate([], method)
    assert not row["parsed"] and row["bbox"] is None
    score = eg.score_predictions({"q": row}, {"q": LEFT})
    assert score["denominator"] == 1 and score["hits"] == 0
    with pytest.raises(ValueError, match="无合法预测"):
        eg.package_predictions({"q": row}, {"q": {"query": "person"}})


def test_ground_truth_and_old_hit_fields_cannot_change_aggregation():
    members = make_members(["a", "b", "c"])
    tables = {name: {"q": {"bbox": LEFT if name == "a" else RIGHT}} for name in ["a", "b", "c"]}
    before = eg.aggregate_table(members, tables, "medoid")
    for name in tables:
        tables[name]["q"].update(target=LEFT, hit=True, iou=1, confidence=0.99)
    assert eg.aggregate_table(members, tables, "medoid") == before


def test_wbf_can_succeed_when_all_originals_miss():
    # 同一对象的左右偏移都低于.5，但融合后命中：原框oracle不是融合上限。
    gt = [0.4, 0.4, 0.6, 0.6]
    a, b = [0.33, 0.4, 0.53, 0.6], [0.47, 0.4, 0.67, 0.6]
    assert eg.iou_xyxy(a, gt) < .5 and eg.iou_xyxy(b, gt) < .5
    candidates = [eg.Candidate(tuple(x), str(i), str(i), 1, i) for i, x in enumerate([a, b])]
    result = eg.aggregate(candidates, "wbf", .1)
    assert eg.iou_xyxy(result["bbox"], gt) > .99


def test_submission_only_updates_bbox_without_mutating_queries():
    queries = {"q": {"query": "the left person", "visible": "v", "bbox": [0, 0, 0, 0], "extra": 42}}
    rows = {"q": {"bbox": LEFT, "parsed": True}}
    final = eg.package_predictions(rows, queries)
    assert final["q"] == {**queries["q"], "bbox": LEFT}
    assert queries["q"]["bbox"] == [0, 0, 0, 0]


def test_weighted_median_equal_even_and_single_vote():
    boxes = np.array([LEFT, RIGHT])
    assert eg.weighted_median(boxes, np.ones(2)) == pytest.approx(np.median(boxes, axis=0))
    assert eg.weighted_median(boxes, np.array([1, 3])) == pytest.approx(RIGHT)


def test_cli_all_methods_evaluation_and_formal_json(tmp_path, monkeypatch):
    for name, box in [("a", LEFT), ("b", LEFT)]:
        (tmp_path / f"{name}.json").write_text(json.dumps({"q": {"bbox": box}}), encoding="utf-8")
    config = {"members": [{"name": name, "path": f"{name}.json"} for name in ["a", "b"]]}
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "gt.json").write_text(json.dumps({"q": LEFT}), encoding="utf-8")
    (tmp_path / "queries.json").write_text(json.dumps({"q": {"query": "person", "visible": "v"}}), encoding="utf-8")
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["ensemble", "--config", str(tmp_path / "config.json"),
                                     "--compare-all", "--out-dir", str(out), "--gt", str(tmp_path / "gt.json"),
                                     "--queries", str(tmp_path / "queries.json")])
    eg.main()
    report = json.loads((out / "ensemble_report.json").read_text(encoding="utf-8"))
    assert report["evaluation"]["original_candidate_oracle_hits"] == 1
    for method in eg.METHODS:
        assert report["evaluation"]["methods"][method]["hits"] == 1
        assert json.loads((out / method / "predictions.json").read_text())["q"]["query"] == "person"
