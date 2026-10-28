import json
from pathlib import Path

import pytest
from PIL import Image

from tools import evaluate_grounding_aux_ab as evaluation


def _write_json(path: Path, value) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def _manifest_rows(count: int = 412):
    rows = {}
    for index in range(count):
        sample_id = f"city_{index:04d}_001"
        rows[sample_id] = {
            "visible": f"../visible/{index:04d}.png",
            "infrared": f"../infrared/{index:04d}.png",
            "depth": f"../depth/{index:04d}.png",
            "query": f"target {index}",
            "bbox": [0.1, 0.1, 0.5, 0.5],
        }
    return rows


def test_prepare_builds_city_partitions_and_27_new_query_manifest(tmp_path: Path):
    source = _manifest_rows()
    gt_path = _write_json(tmp_path / "city.json", source)
    statuses = (["keep"] * 28 + ["corrected"] * 51 + ["deferred"] * 11 + ["partial"])
    cases = []
    changed_ids = []
    for index, status in enumerate(statuses):
        sample_id = f"city_{index:04d}_001"
        changed = index < 27
        new_query = f"reviewed target {index}" if changed else source[sample_id]["query"]
        source[sample_id]["query"] = new_query
        cases.append({"id": sample_id, "status": status, "query_changed": changed,
                      "requires_new_inference": changed, "new_query": new_query})
        if changed:
            changed_ids.append(sample_id)
    # Rewrite GT after applying the 27 reviewed descriptions.
    _write_json(gt_path, source)
    corrections_path = _write_json(tmp_path / "corrections.json", {
        "summary": {"requires_new_inference_ids": changed_ids}, "cases": cases,
    })

    output = tmp_path / "prepared"
    image_root = tmp_path / "remote-data-root"  # Deliberately absent: preparation is path-only.
    summary = evaluation.prepare(gt_path, corrections_path, output, image_root)
    queries = json.loads((output / "queries.json").read_text(encoding="utf-8"))
    gt = json.loads((output / "gt.json").read_text(encoding="utf-8"))
    groups = json.loads((output / "groups.json").read_text(encoding="utf-8"))
    subsets = json.loads((output / "subsets.json").read_text(encoding="utf-8"))

    assert summary["query_changed_count"] == 27
    assert len(queries) == len(gt) == len(groups) == 412
    assert "bbox" not in queries["city_0000_001"]
    assert queries["city_0000_001"]["visible"] == "visible/0000.png"
    assert groups["city_0000_001"] == "0000"
    assert {name: len(subsets[name]) for name in
            ("city400", "regression321", "reviewed79", "unresolved12", "all412_descriptive_only")} == {
                "city400": 400, "regression321": 321, "reviewed79": 79,
                "unresolved12": 12, "all412_descriptive_only": 412,
            }
    assert subsets["_metadata"]["primary"] == "city400"
    assert subsets["_metadata"]["descriptive_only"] == ["all412_descriptive_only"]


def test_threshold_invalid_missing_and_legal_fallback_score_as_boxes():
    gt = {
        "edge": [0.0, 0.0, 1.0, 1.0],
        "invalid": [0.0, 0.0, 1.0, 1.0],
        "fallback": [0.0, 0.0, 1.0, 1.0],
        "missing": [0.0, 0.0, 1.0, 1.0],
    }
    rows = {
        "edge": {"id": "edge", "bbox": [0.0, 0.0, 0.5, 1.0], "parsed": True, "fallback": False},
        "invalid": {"id": "invalid", "bbox": [0.8, 0.1, 0.2, 0.9], "parsed": True, "fallback": False},
        "fallback": {"id": "fallback", "bbox": [0, 0, 1, 1], "parsed": False, "fallback": True},
    }
    items = evaluation.score_items(list(gt), gt, rows)
    metrics = evaluation._arm_metrics(list(gt), items)

    assert items["edge"]["iou"] == pytest.approx(0.5)
    assert items["edge"]["hit"] is True
    assert items["invalid"]["iou"] == 0 and items["invalid"]["hit"] is False
    assert items["fallback"]["hit"] is True
    assert items["missing"]["missing"] is True
    assert metrics["denominator"] == 4
    assert metrics["missing_count"] == 1 and metrics["invalid_count"] == 1
    assert metrics["fallback_rate"] == pytest.approx(1 / 3)


def test_compare_excludes_unresolved_12_from_city400_denominator(tmp_path: Path):
    ids = ["resolved_a", "resolved_b", *[f"unresolved_{i}" for i in range(12)]]
    gt_path = _write_json(tmp_path / "gt.json", {sample_id: [0.1, 0.1, 0.5, 0.5] for sample_id in ids})
    groups_path = _write_json(tmp_path / "groups.json", {sample_id: sample_id for sample_id in ids})
    subsets_path = _write_json(tmp_path / "subsets.json", {
        "_metadata": {"primary": "city400", "prediction_only": ["unresolved12"],
                      "descriptive_only": ["all412_descriptive_only"]},
        "city400": ids[:2], "unresolved12": ids[2:], "all412_descriptive_only": ids,
    })
    arm_a = tmp_path / "a.jsonl"
    arm_b = tmp_path / "b.jsonl"
    rows_a = [{"id": sample_id, "bbox": [0.1, 0.1, 0.5, 0.5], "parsed": True, "fallback": False}
              for sample_id in ids]
    rows_b = [{"id": sample_id, "bbox": [0.1, 0.1, 0.5, 0.5], "parsed": True, "fallback": False}
              for sample_id in ids]
    arm_a.write_text("".join(json.dumps(row) + "\n" for row in rows_a), encoding="utf-8")
    arm_b.write_text("".join(json.dumps(row) + "\n" for row in rows_b), encoding="utf-8")

    summary = evaluation.compare(gt_path, groups_path, subsets_path,
                                 [("A", arm_a), ("B", arm_b)], tmp_path / "out")
    assert summary["subsets"]["city400"]["arms"]["A"]["denominator"] == 2
    assert summary["subsets"]["city400"]["arms"]["A"]["acc_0.5"] == 1.0
    assert summary["subsets"]["unresolved12"]["role"] == "prediction_only"
    assert "acc_0.5" not in summary["subsets"]["unresolved12"]["arms"]["A"]
    assert summary["subsets"]["all412_descriptive_only"]["role"] == "descriptive_only"
    assert "mean_iou" not in summary["subsets"]["all412_descriptive_only"]["arms"]["A"]
    assert (tmp_path / "out" / "cases.jsonl").is_file()
    assert "descriptive only" in (tmp_path / "out" / "report.md").read_text(encoding="utf-8")


def test_duplicate_extra_ids_and_query_mismatch_are_rejected(tmp_path: Path):
    gt = {"known": [0.1, 0.1, 0.5, 0.5]}
    with pytest.raises(ValueError, match="duplicate"):
        evaluation.index_predictions([{"id": "known"}, {"id": "known"}], gt)
    with pytest.raises(ValueError, match="extra ID"):
        evaluation.index_predictions([{"id": "extra"}], gt)
    with pytest.raises(ValueError, match="query mismatch"):
        evaluation.index_predictions([{"id": "known", "query": "old wording"}], gt,
                                     {"known": {"query": "new wording"}})
    with pytest.raises(ValueError, match="no query"):
        evaluation.index_predictions([{"id": "known"}], gt,
                                     {"known": {"query": "new wording"}})


def test_paired_bootstrap_resamples_groups():
    ids = ["a1", "a2", "b1", "b2"]
    groups = {"a1": "image_a", "a2": "image_a", "b1": "image_b", "b2": "image_b"}
    items_a = {sample_id: {"hit": False} for sample_id in ids}
    items_b = {sample_id: {"hit": sample_id.startswith("a")} for sample_id in ids}

    result = evaluation.paired_bootstrap(ids, groups, items_a, items_b, rounds=1000, seed=2026)
    assert result["unit"] == "image_group"
    assert result["groups"] == 2
    assert result["point_delta_pp"] == 50
    assert result["ci95_pp"] == [0, 100]


def test_aux_unknown_is_not_neither(tmp_path: Path):
    manifest = tmp_path / "aux_eval.jsonl"
    manifest.write_text(json.dumps({
        "id": "eval:item:aux", "category": "extent",
        "expected": {"choice": "unknown", "reason": "unclear"},
    }) + "\n", encoding="utf-8")
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(json.dumps({
        "id": "eval:item:aux", "response": [{"role": "assistant",
                                                    "content": "{\"choice\":\"neither\",\"reason\":\"both wrong\"}"}],
    }) + "\n", encoding="utf-8")

    summary = evaluation.score_aux(manifest, predictions, tmp_path / "out")
    assert summary["accuracy"] == 0
    assert summary["confusion"]["unknown"]["neither"] == 1
    assert summary["by_category"]["extent"]["samples"] == 1
    assert "not a bbox gain claim" in summary["interpretation"]


class _FakeAuxTokenizer:
    instances = []

    def __init__(self):
        self.calls = []
        self.instances.append(self)

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        text = "".join(
            block.get("text", "")
            for message in messages for block in message["content"]
            if isinstance(block, dict) and block.get("type") == "text"
        )
        return "<fake>" + text + "<think>\n\n</think>\n"


class _FakeAuxSamplingParams:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.instances.append(self)


class _FakeAuxLLM:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.tokenizer = _FakeAuxTokenizer()
        self.instances.append(self)

    def get_tokenizer(self):
        return self.tokenizer

    def generate(self, requests, sampling_params, **kwargs):
        self.calls.append({"requests": requests, "sampling_params": sampling_params,
                           "kwargs": kwargs})
        output = type("Output", (), {
            "text": '{"choice":"A","reason":"mock response"}',
            "token_ids": [1], "finish_reason": "stop",
        })()
        return [type("RequestOutput", (), {"outputs": [output]})() for _ in requests]


def test_infer_aux_is_single_pass_and_does_not_send_hidden_labels(tmp_path: Path, monkeypatch):
    import sys
    import types

    rgb = tmp_path / "rgb.png"
    depth = tmp_path / "depth.png"
    Image.new("RGB", (16, 12), (80, 90, 100)).save(rgb)
    Image.new("RGB", (16, 12), (20, 30, 40)).save(depth)
    prepared_prompt = (
        "<image>\n<image>\nThese are aligned RGB and depth views.\n"
        "Query: the left red object\nA: [100, 200, 300, 400]\nB: [500, 200, 700, 400]\n"
        "Return only short JSON with choice and reason."
    )
    manifest = tmp_path / "aux.jsonl"
    manifest.write_text(json.dumps({
        "id": "aux-001", "category": "wrong_object",
        "images": [str(rgb), str(depth)],
        "messages": [{"role": "user", "content": prepared_prompt}],
        "expected": {"choice": "B", "reason": "hidden-choice-secret"},
        "bbox": [0.81, 0.82, 0.91, 0.92], "evidence": "hidden-evidence-secret",
    }) + "\n", encoding="utf-8")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = _FakeAuxLLM
    fake_vllm.SamplingParams = _FakeAuxSamplingParams
    fake_lora = types.ModuleType("vllm.lora")
    fake_lora_request = types.ModuleType("vllm.lora.request")
    fake_lora_request.LoRARequest = lambda name, lid, path: (name, lid, path)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.lora", fake_lora)
    monkeypatch.setitem(sys.modules, "vllm.lora.request", fake_lora_request)
    _FakeAuxLLM.instances.clear()
    _FakeAuxTokenizer.instances.clear()
    _FakeAuxSamplingParams.instances.clear()

    output = tmp_path / "predictions.jsonl"
    summary = evaluation.infer_aux(manifest, output, "fake-model", adapter, chunk_size=1)

    llm = _FakeAuxLLM.instances[-1]
    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call["sampling_params"].kwargs == {
        "temperature": 0.0, "max_tokens": evaluation.MAX_NEW_TOKENS,
        "skip_special_tokens": True,
    }
    assert call["kwargs"]["lora_request"] == ("trained", 1, str(adapter.resolve()))
    assert llm.kwargs["mm_processor_kwargs"] == {"min_pixels": evaluation.MIN_PIXELS,
                                                   "max_pixels": 8294400}
    assert llm.kwargs["max_model_len"] == 32768
    tokenizer_messages, tokenizer_kwargs = llm.tokenizer.calls[0]
    assert [message["role"] for message in tokenizer_messages] == ["user"]
    assert tokenizer_kwargs["enable_thinking"] is False
    prompt = call["requests"][0]["prompt"]
    actual_user_content = llm.tokenizer.calls[0][0][0]["content"]
    restored_user_prompt = "".join(
        "<image>" if block.get("type") == "image" else block["text"]
        for block in actual_user_content
    )
    assert restored_user_prompt == prepared_prompt
    assert "the left red object" in prompt
    assert "hidden-choice-secret" not in prompt
    assert "hidden-evidence-secret" not in prompt
    assert "0.81,0.82,0.91,0.92" not in prompt
    assert [Image.open(path).size for path in (rgb, depth)] == [(16, 12), (16, 12)]
    assert summary["upscale"] == 2.0

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows == [{"id": "aux-001", "response": '{"choice":"A","reason":"mock response"}'}]


def test_build_aux_requests_rejects_assistant_manifest_message():
    with pytest.raises(ValueError, match="must not contain assistant messages"):
        evaluation.build_aux_requests(None, [{
            "id": "aux-001", "images": [],
            "messages": [
                {"role": "user", "content": "<image> prompt"},
                {"role": "assistant", "content": '{"choice":"A"}'},
            ],
        }], upscale=1.0)
