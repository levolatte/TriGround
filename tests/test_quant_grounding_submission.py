from copy import deepcopy

import pytest

from tools.export_quant_submission import make_submission


def test_export_changes_only_bbox_and_keeps_order():
    source = {"b": {"query": "target", "visible": "b.png", "infrared": "i.png", "depth": "d.png"},
              "a": {"query": "second", "bbox": [0, 0, 0, 0], "extra": {"keep": 1}}}
    before = deepcopy(source)
    predictions = [{"id": key, "samples": [{"text": '{"bbox_2d":[100,200,900,800]}', "finish_reason": "stop"}]}
                   for key in ["a", "b"]]
    result = make_submission(source, predictions)
    assert list(result) == ["b", "a"] and source == before
    assert result["b"] == {**source["b"], "bbox": [0.1, 0.2, 0.9, 0.8]}
    assert result["a"]["extra"] == source["a"]["extra"]


@pytest.mark.parametrize("samples", [
    [{"text": '{"bbox_2d":[0,0,1000,1000]}', "finish_reason": "length"}],
    [{"text": "not a box", "finish_reason": "stop"}],
    [{"text": '{"bbox_2d":[0,0,1000,1000]}', "finish_reason": "stop"}] * 8,
])
def test_cannot_submit_truncation_fallback_or_best_of_eight(samples):
    with pytest.raises(ValueError):
        make_submission({"a": {"query": "q"}}, [{"id": "a", "samples": samples}])


def test_incomplete_prediction_set_cannot_be_packaged():
    with pytest.raises(ValueError, match="every official Query"):
        make_submission({"a": {"query": "q"}}, [])


def test_punctuation_policy_exports_same_coordinates_without_mutating_raw_answer():
    source = {"a": {"query": "q", "visible": "rgb.png", "bbox": [0, 0, 0, 0]}}
    predictions = [{"id": "a", "samples": [{
        "text": '{"bbox_2d":[593,362),(670,416]}', "finish_reason": "stop"}]}]
    before = deepcopy(predictions)
    with pytest.raises(ValueError, match="invalid answer"):
        make_submission(source, predictions)
    result = make_submission(source, predictions, format_policy="punctuation")
    assert result["a"] == {**source["a"], "bbox": [0.593, 0.362, 0.670, 0.416]}
    assert predictions == before


@pytest.mark.parametrize("text,finish_reason", [
    ('{"bbox_2d":[593,362),(670,416]}', "length"),
    ('{"bbox_2d":[593,362),(670,416],"label":"target"}', "stop"),
    ('{"bbox_2d":[593,362),(670,416]} {"bbox_2d":[0,0,1000,1000]}', "stop"),
])
def test_punctuation_policy_still_rejects_incomplete_or_ambiguous_answers(text, finish_reason):
    with pytest.raises(ValueError):
        make_submission({"a": {"query": "q"}}, [
            {"id": "a", "samples": [{"text": text, "finish_reason": finish_reason}]}],
            format_policy="punctuation")
