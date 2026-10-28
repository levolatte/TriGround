import json
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

import pytest

from tools import eval_quant_grounding as ev


GOOD = '{"bbox_2d":[100,100,500,500]}'
BAD = '{"bbox_2d":[600,600,900,900]}'
GT = [0.1, 0.1, 0.5, 0.5]


def row(sample_id="q1", source="city", negative_kind=None):
    return {"id": sample_id, "source": source, "group_id": "frame1", "images": ["rgb.png"],
            "messages": [{"role": "user", "content": "<image>find object"}],
            "solution": GT, "negative_boxes": [[0.6, 0.6, 0.9, 0.9]],
            "negative_kind": negative_kind or ["wrong_object"]}


def pred(sample_id, *texts, reasons=None):
    reasons = reasons or ["stop"] * len(texts)
    return {"id": sample_id, "samples": [
        {"text": text, "finish_reason": reason, "tokens": 10, "seconds": 0.2}
        for text, reason in zip(texts, reasons)]}


def test_strict_scoring_retains_invalid_denominator_and_reward_crosscheck():
    data = [row()]
    predictions = [pred("q1", GOOD, BAD, GOOD + " extra", GOOD, reasons=["stop", "stop", "stop", "length"])]
    scored = ev.score_rows(data, predictions, "short", "contrast")
    samples = scored[0]["samples"]
    assert [s["hit"] for s in samples] == [True, False, False, False]
    assert [s["score"]["invalid_reason"] for s in samples] == [None, None, "format", "truncated"]
    assert samples[0]["bbox"] == GT and samples[0]["iou"] == pytest.approx(1)
    assert samples[3]["iou"] == pytest.approx(1)  # geometry retained, but truncation scores zero
    summary = ev.summarize(scored)
    assert summary["overall"]["acc05_all_samples"] == 0.25
    assert summary["overall"]["acc05_first"] == 1
    assert summary["overall"]["mixed_groups"] == 1
    assert summary["overall"]["invalid"] == 2
    assert summary["by_source"]["city"]["queries"] == 1
    assert summary["by_negative_kind"]["wrong_object"]["samples"] == 4


def test_all_same_reward_and_all_wrong_counted():
    scored = ev.score_rows([row()], [pred("q1", BAD, BAD)], "short", "contrast")
    total = ev.summarize(scored)["overall"]
    assert total["all_wrong_groups"] == 1
    assert total["all_same_reward_groups"] == total["zero_advantage_groups"] == 1
    assert total["acc05_first"] == 0


def test_thinking_requires_native_end_and_unique_final_json():
    scored = ev.score_rows(
        [row()], [pred("q1", "inspect</think>\n" + GOOD, "inspect " + GOOD,
                           "inspect</think>" + GOOD + BAD)], "thinking", "acc05")
    assert [sample["hit"] for sample in scored[0]["samples"]] == [True, False, False]
    assert scored[0]["samples"][0]["score"]["contrast_reward"] == 0


def test_punctuation_policy_uses_repaired_text_and_keeps_raw_text():
    malformed = '{"bbox_2d":[100,100),(500,500]}'
    predictions = [pred("q1", malformed, malformed, reasons=["stop", "length"])]
    strict = ev.score_rows([row()], predictions, "short", "contrast")
    assert strict[0]["samples"][0]["score"]["invalid_reason"] == "format"
    assert "normalized_text" not in strict[0]["samples"][0]

    scored = ev.score_rows([row()], predictions, "short", "contrast", "punctuation")
    first, truncated = scored[0]["samples"]
    assert first["text"] == malformed
    assert first["normalized_text"] == GOOD
    assert first["format_repaired"] is True
    assert first["bbox"] == GT and first["hit"] is True
    assert truncated["normalized_text"] == GOOD
    assert truncated["format_repaired"] is True
    assert truncated["score"]["truncated"] is True
    assert truncated["score"]["valid"] is False and truncated["hit"] is False
    assert ev.summarize(scored)["overall"]["format_repaired"] == 2


def test_punctuation_normalization_does_not_depend_on_ground_truth():
    malformed = '{"bbox_2d":[100,100),(500,500]}'
    predictions = [pred("q1", malformed)]
    first = ev.score_rows([row()], predictions, "short", "contrast", "punctuation")
    other_target = dict(row(), solution=[0.2, 0.2, 0.7, 0.7])
    second = ev.score_rows([other_target], predictions, "short", "contrast", "punctuation")
    assert first[0]["samples"][0]["normalized_text"] == second[0]["samples"][0]["normalized_text"]
    assert first[0]["samples"][0]["iou"] != second[0]["samples"][0]["iou"]


def test_format_policy_cli_rescores_with_requested_policy(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    dataset.write_text(json.dumps(row()) + "\n", encoding="utf-8")
    malformed = '{"bbox_2d":[100,100),(500,500]}'
    predictions.write_text(json.dumps(pred("q1", malformed)) + "\n", encoding="utf-8")
    out = tmp_path / "out"
    summary = ev.main(["--dataset", str(dataset), "--predictions", str(predictions),
                       "--output-dir", str(out), "--format-policy", "punctuation"])
    assert summary["format_policy"] == "punctuation"
    assert summary["overall"]["format_repaired"] == 1
    saved = ev.read_jsonl(out / "predictions.jsonl")[0]["samples"][0]
    assert saved["text"] == malformed
    assert saved["normalized_text"] == GOOD and saved["format_repaired"] is True


def test_exact_coverage_and_baseline_flips():
    dataset = [row("a"), row("b", source="robo", negative_kind=["extent"])]
    with pytest.raises(ValueError, match="coverage"):
        ev.score_rows(dataset, [pred("a", GOOD)], "short", "contrast")
    scored = ev.score_rows(dataset, [pred("a", GOOD), pred("b", BAD)], "short", "contrast")
    comparison = ev.compare_baseline(scored, [
        {"id": "a", "prediction": [0.6, 0.6, 0.9, 0.9]},
        {"id": "b", "prediction": GT},
    ])
    assert comparison["counts"] == {"rescue": 1, "regression": 1}
    assert comparison["baseline_acc05"] == comparison["model_acc05"] == 0.5
    assert comparison["per_query"][0]["change"] == "rescue"


def test_rescore_cli_uses_saved_text_not_saved_score(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    dataset.write_text(json.dumps(row()) + "\n", encoding="utf-8")
    saved = pred("q1", GOOD)
    saved["samples"][0]["score"] = {"reward": -999, "hit": False}
    predictions.write_text(json.dumps(saved) + "\n", encoding="utf-8")
    out = tmp_path / "out"
    summary = ev.main(["--dataset", str(dataset), "--predictions", str(predictions),
                       "--output-dir", str(out)])
    assert summary["overall"]["acc05_first"] == 1
    assert summary["resource"] == {"rescore_only": True}
    saved_again = ev.read_jsonl(out / "predictions.jsonl")
    assert saved_again[0]["samples"][0]["score"]["reward"] > 1


def test_adapter_audit_requires_paired_language_weights(tmp_path):
    safetensors = pytest.importorskip("safetensors.torch")
    torch = pytest.importorskip("torch")
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "modules_to_save": [], "r": 32,
                    "target_modules": ["q_proj"]}), encoding="utf-8")
    safetensors.save_file({
        "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(2, 2),
        "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(2, 2),
    }, tmp_path / "adapter_model.safetensors")
    assert ev.audit_adapter(tmp_path, vllm=True)["module_count"] == 1
    safetensors.save_file({
        "base_model.model.model.visual.q_proj.lora_A.weight": torch.ones(2, 2),
        "base_model.model.model.visual.q_proj.lora_B.weight": torch.ones(2, 2),
    }, tmp_path / "adapter_model.safetensors")
    with pytest.raises(ValueError, match="visual"):
        ev.audit_adapter(tmp_path, vllm=True)


def test_swift_decode_omits_synthetic_thinking_prefix():
    pytest.importorskip("swift.infer_engine")
    from swift.model import get_processor
    from swift.template import get_template

    metadata = ev.Path(__file__).resolve().parents[2] / ".work/grounding_aux_round1/Qwen3.6-27B-metadata"
    if not metadata.is_dir():
        pytest.skip("Qwen3.6 tokenizer metadata unavailable")
    processor = get_processor(str(metadata), download_model=False, model_type="qwen3_5")
    template = get_template(processor, template_type="qwen3_5")
    ids = processor.tokenizer.encode(GOOD, add_special_tokens=False)
    assert template.decode_generate_ids(ids).startswith("<think>")
    assert template.decode_generate_ids(ids, first_token=False) == GOOD
    assert ev.score_sample({"text": template.decode_generate_ids(ids, first_token=False),
                            "finish_reason": "stop"}, row(), "short", "contrast")["hit"]


def test_swift_hf_g8_serial_and_raw_saved_before_next_failure(tmp_path, monkeypatch):
    pytest.importorskip("swift.infer_engine")
    import torch

    class FakeEngine:
        max_batch_size = 1

        def __init__(self):
            self.template = SimpleNamespace(decode_generate_ids=lambda ids, first_token: GOOD)
            self.generation_config = SimpleNamespace(top_k=20)
            self.calls = 0

        def _prepare_generation_config(self, request):
            return SimpleNamespace(temperature=request.temperature, top_p=request.top_p,
                                   top_k=self.generation_config.top_k, do_sample=True)

        def infer(self, requests, config, use_tqdm):
            self.calls += 1
            assert len(requests) == 8 and config.n == 1 and config.stop == []
            assert all(request.chat_template_kwargs == {"enable_thinking": False} for request in requests)
            if self.calls == 2:
                raise RuntimeError("simulated GPU failure")
            choice = SimpleNamespace(
                message=SimpleNamespace(content="<think>synthetic prefix", reasoning_content=None),
                token_ids=[1, 2], finish_reason="stop")
            return [SimpleNamespace(choices=[choice]) for _ in requests]

    engine = FakeEngine()
    monkeypatch.setattr(ev, "create_engine", lambda args: engine)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (9, 10))
    (tmp_path / "rgb.png").write_bytes(b"image")
    output = tmp_path / "out"
    output.mkdir()
    args = Namespace(engine="hf", precision="bf16", adapter=None, max_pixels=1204224,
                     min_pixels=200704,
                     max_tokens=64, samples_per_query=8, seed=2026, data_root=tmp_path,
                     answer_mode="short", output_dir=output)
    with pytest.raises(RuntimeError, match="simulated"):
        ev.generate_rows(args, [row("a"), row("b")])
    raw = ev.read_jsonl(output / "predictions_raw.jsonl")
    assert len(raw) == 1 and raw[0]["id"] == "a" and len(raw[0]["samples"]) == 8
    assert all(sample["text"] == GOOD for sample in raw[0]["samples"])


@pytest.fixture
def fake_generation_modules(monkeypatch):
    torch = ModuleType("torch")
    torch.manual_seed = lambda seed: None
    torch.cuda = SimpleNamespace(is_available=lambda: True, manual_seed_all=lambda seed: None,
                                 mem_get_info=lambda: (9, 10), max_memory_allocated=lambda: 0,
                                 max_memory_reserved=lambda: 0)
    swift = ModuleType("swift")
    swift.__path__ = []
    infer = ModuleType("swift.infer_engine")
    infer.InferRequest = lambda **kwargs: SimpleNamespace(**kwargs)
    infer.RequestConfig = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "swift", swift)
    monkeypatch.setitem(sys.modules, "swift.infer_engine", infer)


def fake_rows(tmp_path):
    rows = []
    for sample_id in ("a", "b", "c"):
        (tmp_path / f"{sample_id}.png").write_bytes(sample_id.encode())
        item = row(sample_id)
        item["images"] = [f"{sample_id}.png"]
        item["messages"][0]["content"] = f"<image>find {sample_id}"
        rows.append(item)
    return rows


def fake_args(tmp_path, *, engine="hf", cap=2, g=1):
    output = tmp_path / "out"
    output.mkdir()
    return Namespace(engine=engine, precision="bf16", adapter=None, max_pixels=1204224,
                     min_pixels=200704, max_tokens=64, samples_per_query=g, seed=2026,
                     data_root=tmp_path, answer_mode="short", output_dir=output,
                     hf_max_batch_size=cap)


class MappingEngine:
    def __init__(self, cap, fail_on=None):
        self.max_batch_size = cap
        self.fail_on = fail_on
        self.calls = []
        self.generation_config = SimpleNamespace(top_k=20)
        self.template = SimpleNamespace(decode_generate_ids=lambda ids, first_token: f"answer_{ids[0]}")

    def _prepare_generation_config(self, request):
        return SimpleNamespace(temperature=1 if request.temperature == 0 else request.temperature,
                               top_p=request.top_p, top_k=self.generation_config.top_k,
                               do_sample=request.temperature != 0)

    def infer(self, requests, config, use_tqdm):
        ids = [Path(request.images[0]).stem for request in requests]
        assert [request.messages[0]["content"] for request in requests] == [
            f"<image>find {sample_id}" for sample_id in ids]
        assert all(request.chat_template_kwargs == {"enable_thinking": False} for request in requests)
        assert config.n == 1 and config.stop == []
        self.calls.append(ids)
        if len(self.calls) == self.fail_on:
            raise RuntimeError("simulated inference failure")
        return [SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(reasoning_content=None), token_ids=[sample_id], finish_reason="stop")])
                for sample_id in ids]


@pytest.mark.parametrize("cap,expected_calls", [
    (1, [["a"], ["b"], ["c"]]),
    (2, [["a", "b"], ["c"]]),
])
def test_hf_g1_batches_distinct_queries_without_response_mix(
    tmp_path, monkeypatch, fake_generation_modules, cap, expected_calls
):
    engine = MappingEngine(cap)
    monkeypatch.setattr(ev, "create_engine", lambda args: engine)
    args = fake_args(tmp_path, cap=cap)
    generated, resource = ev.generate_rows(args, fake_rows(tmp_path))
    assert engine.calls == expected_calls
    assert [item["id"] for item in generated] == ["a", "b", "c"]
    assert [item["samples"][0]["text"] for item in generated] == ["answer_a", "answer_b", "answer_c"]
    assert ev.read_jsonl(args.output_dir / "predictions_raw.jsonl") == generated
    assert resource["sample_seconds_semantics"] == "infer call wall time divided equally among its samples"


def test_hf_g1_flushes_completed_batch_before_later_failure(tmp_path, monkeypatch, fake_generation_modules):
    engine = MappingEngine(2, fail_on=2)
    monkeypatch.setattr(ev, "create_engine", lambda args: engine)
    args = fake_args(tmp_path)
    with pytest.raises(RuntimeError, match="simulated"):
        ev.generate_rows(args, fake_rows(tmp_path))
    assert engine.calls == [["a", "b"], ["c"]]
    assert [item["id"] for item in ev.read_jsonl(args.output_dir / "predictions_raw.jsonl")] == ["a", "b"]


def test_hf_g8_keeps_one_query_per_infer_and_n1(tmp_path, monkeypatch, fake_generation_modules):
    engine = MappingEngine(2, fail_on=2)
    monkeypatch.setattr(ev, "create_engine", lambda args: engine)
    args = fake_args(tmp_path, cap=2, g=8)
    with pytest.raises(RuntimeError, match="simulated"):
        ev.generate_rows(args, fake_rows(tmp_path)[:2])
    assert engine.calls == [["a"] * 8, ["b"] * 8]
    raw = ev.read_jsonl(args.output_dir / "predictions_raw.jsonl")
    assert len(raw) == 1 and raw[0]["id"] == "a" and len(raw[0]["samples"]) == 8


def test_vllm_g1_keeps_per_query_calls(tmp_path, monkeypatch, fake_generation_modules):
    engine = MappingEngine(2)
    engine.generation_config.top_k = -1
    monkeypatch.setattr(ev, "create_engine", lambda args: engine)
    args = fake_args(tmp_path, engine="vllm", cap=2)
    generated, _ = ev.generate_rows(args, fake_rows(tmp_path))
    assert engine.calls == [["a"], ["b"], ["c"]]
    assert [item["samples"][0]["text"] for item in generated] == ["answer_a", "answer_b", "answer_c"]


def test_prequantized_base_rejected_for_nf4(monkeypatch):
    from transformers import AutoConfig

    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *a, **kw: SimpleNamespace(
        to_dict=lambda: {"quantization_config": {"quant_method": "fp8"}}))
    with pytest.raises(ValueError, match="unquantized"):
        ev.model_quantization("prequantized", "nf4")
    ev.model_quantization("prequantized", "fp8")


def test_full_generation_budget_must_fit():
    class FakeEngine:
        template = SimpleNamespace()
        _get_num_tokens = staticmethod(lambda inputs: inputs["count"])

        def set_default_max_tokens(self, config, inputs):
            config.max_tokens = 1  # Swift would silently reduce it

    engine = FakeEngine()
    ev.enforce_max_length(engine, 8192, 64)
    assert engine.template.truncation_strategy == "raise"
    with pytest.raises(ValueError, match="exceeds max_length"):
        engine.set_default_max_tokens(SimpleNamespace(max_tokens=64), {"count": 8130})
    config = SimpleNamespace(max_tokens=64)
    engine.set_default_max_tokens(config, {"count": 100})
    assert config.max_tokens == 1
