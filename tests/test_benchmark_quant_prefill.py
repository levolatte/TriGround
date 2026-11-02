"""CPU：缓存复用的保守接受条件与失败报告。"""

from __future__ import annotations

import copy
import json

import pytest
import torch

from tools import benchmark_quant_prefill as benchmark


def pairs(full=(95.0, 94.0, 96.0), generation=(60.0, 61.0, 59.0)):
    return [{"baseline": {"full_batch_seconds": 100.0, "generation_seconds": 70.0},
             "reuse": {"full_batch_seconds": batch, "generation_seconds": gen}}
            for batch, gen in zip(full, generation)]


def test_accept_requires_full_step_gain_and_all_repeats_faster():
    result = benchmark.timing_decision(pairs())
    assert result["accepted"] is True
    assert result["full_batch_median_gain"] == pytest.approx(0.05)
    assert benchmark.timing_decision(pairs(full=(99, 98, 99)))["accepted"] is False
    assert benchmark.timing_decision(pairs(full=(94, 95, 101)))["accepted"] is False
    assert benchmark.timing_decision(pairs(generation=(71, 72, 73)))["accepted"] is False


def test_timing_rejects_missing_or_invalid_measurements():
    with pytest.raises(ValueError, match="三组"):
        benchmark.timing_decision(pairs()[:2])
    for invalid in (0.0, -1.0, float("nan"), float("inf")):
        values = pairs()
        values[0]["reuse"]["full_batch_seconds"] = invalid
        with pytest.raises(ValueError, match="无效计时"):
            benchmark.timing_decision(values)


def generation_result(reuse):
    ids = torch.tensor([[10, 11, 12]])
    diagnostics = []
    completions = []
    for index in range(2):
        tokens = [15 + index, 2]
        text = f"answer {index}"
        completions.append({"text": text, "token_ids": tokens, "finish_reason": "stop"})
        diagnostics.append({
            "sequences": torch.tensor([[10, 11, 12, *tokens]]), "token_ids": tokens, "text": text,
            "generation_config": {"do_sample": True, "top_k": 0},
            "rng_before": {"python": (3, (1, 2), None), "numpy": ("MT19937",),
                           "torch": torch.tensor([index], dtype=torch.uint8), "cuda": []},
            "rng_after": {"python": (3, (1, 2), None), "numpy": ("MT19937",),
                          "torch": torch.tensor([index + 1], dtype=torch.uint8), "cuda": []},
            "logits": (torch.zeros(1, 2048), torch.zeros(1, 2048)),
            "scores": (torch.zeros(1, 2048), torch.zeros(1, 2048)),
            "cache": {"seq_length": 4, "tensors": {f"0.{field}": torch.zeros(1, 4) for field in
                      ("keys", "values", "conv_states", "recurrent_states")},
                      "layers": [{"has_previous_state": True}]},
            "prefill": [{"cache_length": 3, "prompt_length": 3,
                         "native_prefills": int(not (reuse and index == 1)),
                         "prefill_replays": int(reuse and index == 1)}], "rope_deltas": torch.tensor([[0]]),
        })
    return {"completions": completions, "diagnostics": diagnostics,
            "audit": {"enabled": reuse, "native_prefills": 1 if reuse else 2, "prefill_replays": int(reuse)},
            "traces": [{"step": 0}, {"step": 1}, *([] if reuse else [{"step": 2}]), {"step": 3}],
            "inputs": {"input_ids": ids}, "prefix_length": 3, "vocab_size": 2048,
            "rng_after": {"torch": torch.tensor([5], dtype=torch.uint8)},
            "native_after": {"rope_deltas": torch.tensor([[0]])}}


def test_generation_checks_full_logits_and_native_cache_not_only_sampled_tokens():
    baseline, reuse = generation_result(False), generation_result(True)
    assert benchmark.compare_generation(baseline, reuse, greedy=False)["passed"] is True
    changed = copy.deepcopy(reuse)
    changed["diagnostics"][1]["logits"][0][0, 1900] = 1
    with pytest.raises(AssertionError, match="logits"):
        benchmark.compare_generation(baseline, changed, greedy=False)
    changed = copy.deepcopy(reuse)
    changed["diagnostics"][1]["cache"]["tensors"]["0.keys"][0, 1] = 1
    with pytest.raises(AssertionError, match="cache"):
        benchmark.compare_generation(baseline, changed, greedy=False)
    changed = copy.deepcopy(reuse)
    changed["diagnostics"][1]["rng_after"]["torch"][0] = 9
    with pytest.raises(AssertionError, match="rng_after"):
        benchmark.compare_generation(baseline, changed, greedy=False)


def test_generation_checks_preserve_absent_rope_state_and_reject_changed_state():
    baseline, reuse = generation_result(False), generation_result(True)
    for result in (baseline, reuse):
        result["native_after"]["rope_deltas"] = None
        for diagnostic in result["diagnostics"]:
            diagnostic["rope_deltas"] = None
    assert benchmark.compare_generation(baseline, reuse, greedy=False)["passed"] is True
    for state in (reuse["native_after"], reuse["diagnostics"][1]):
        state["rope_deltas"] = torch.tensor([[0]])
        with pytest.raises(AssertionError, match="rope_deltas"):
            benchmark.compare_generation(baseline, reuse, greedy=False)
        state["rope_deltas"] = None


def test_selection_is_reproducible_and_contains_two_real_rows_per_image_count():
    rows = [{"id": str(index), "images": ["image"] * (2 + index % 2)} for index in range(12)]
    selected = benchmark.select_rows(rows, 2026)
    assert selected == benchmark.select_rows(rows, 2026)
    assert [len(row["images"]) for row in selected] == [2, 2, 3, 3]
    assert len({row["id"] for row in selected}) == 4
    with pytest.raises(ValueError, match="至少两道3图"):
        benchmark.select_rows(rows[:3], 2026)


def test_runtime_exception_writes_failed_rejected_report(tmp_path):
    rows = [{"id": str(index), "images": ["image"] * (2 + index % 2), "messages": []}
            for index in range(4)]
    index_path = tmp_path / "index.jsonl"
    index_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def oom(*args, **kwargs):
        raise RuntimeError("CUDA out of memory")

    output = tmp_path / "evidence"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "optimizer.pt").touch()
    (checkpoint / "loop_state.json").write_text(json.dumps({
        "schema": "acc-correction-v1", "mode": "correction", "presentations": 800, "updates": 200}), encoding="utf-8")
    report = benchmark.benchmark({"train_index": str(index_path), "seed": 3}, checkpoint,
                                 output, seeds=[3, 4], runtime_factory=oom)
    assert report["status"] == "failed" and report["accepted"] is False
    assert report["correctness"]["passed"] is False
    assert "out of memory" in report["reason"]
    assert json.loads((output / "report.json").read_text(encoding="utf-8")) == report


def test_cli_bad_config_writes_report_and_returns_nonzero(tmp_path):
    output = tmp_path / "evidence"
    code = benchmark.main(["--config", str(tmp_path / "missing.json"), "--adapter", str(tmp_path / "checkpoint"),
                           "--output-dir", str(output)])
    assert code != 0
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed" and report["accepted"] is False
    assert report["correctness"]["passed"] is False
