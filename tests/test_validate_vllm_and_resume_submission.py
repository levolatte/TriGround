import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from tools import validate_vllm_and_resume_submission as worker


def results(city=313, external=84, seconds=2.0):
    return {key: {"queries": count, "hits": hits, "resource": {"wall_seconds": count * seconds}}
            for key, count, hits in (("city378", 378, city), ("external88", 88, external))}


@pytest.mark.parametrize("city,external,seconds,accepted", [
    (313, 84, 3.6, True), (312, 88, 2.0, False),
    (378, 83, 2.0, False), (378, 88, 3.61, False),
])
def test_both_complete_accuracy_thresholds_and_speed_required(city, external, seconds, accepted):
    decision = worker.make_decision(results(city, external, seconds), 313, 84, 4.0, 0.10)
    assert decision["accepted"] is accepted


def test_incomplete_split_rejected_even_with_high_hits():
    partial = results()
    partial["city378"]["queries"] = 377
    assert worker.make_decision(partial, 313, 84, 4.0, 0.10)["accepted"] is False


@pytest.mark.parametrize("accept", [True, False])
def test_worker_reuses_one_engine_and_only_appends_official_after_acceptance(tmp_path, monkeypatch, accept):
    full = tmp_path / "full"
    full.mkdir()
    source = {key: {"query": key, "visible": "r.png", "infrared": "i.png", "depth": "d.png"}
              for key in ("a", "b", "c")}
    queries = tmp_path / "queries.json"
    worker.write(queries, source)
    config = dict(engine="hf", precision="nf4", answer_mode="short", format_policy="punctuation",
                  model="base", adapter="adapter", queries=str(queries), expected_queries=3,
                  data_root=str(tmp_path), min_pixels=200704, max_pixels=1204224,
                  max_length=8192, max_tokens=64, seed=2026)
    worker.write(full / "run_config.json", config)
    worker.write_jsonl(full / "unlabeled.jsonl", [{"id": key} for key in source])
    good = '{"bbox_2d":[100,100,500,500]}'
    bad = '{"bbox_2d":[600,600,900,900]}'
    saved = {"id": "a", "samples": [{"text": good, "finish_reason": "stop", "seconds": 4.0}]}
    worker.write_jsonl(full / "predictions_raw.jsonl", [saved])
    original = (full / "predictions_raw.jsonl").read_bytes()
    original_config = (full / "run_config.json").read_bytes()
    queue = dict(model="base", media_root=str(tmp_path))
    for key, count in (("city", 378), ("external", 88)):
        dataset = [{"id": f"{key}{i}", "source": key, "group_id": str(i),
                    "messages": [{"role": "user", "content": "find"}],
                    "solution": [0.1, 0.1, 0.5, 0.5]} for i in range(count)]
        path = tmp_path / f"{key}.jsonl"
        worker.write_jsonl(path, dataset)
        queue[f"{key}_dataset"] = str(path)
    queue_path = tmp_path / "queue.json"
    worker.write(queue_path, queue)
    transformers = ModuleType("transformers")
    transformers.AutoProcessor = SimpleNamespace(from_pretrained=lambda model: object())
    vllm = ModuleType("vllm")
    engine = object()
    loads = []
    vllm.LLM = lambda **options: loads.append(options) or engine
    vllm.SamplingParams = lambda **options: options
    lora_module = ModuleType("vllm.lora.request")
    lora_module.LoRARequest = lambda *args: args
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.lora", ModuleType("vllm.lora"))
    monkeypatch.setitem(sys.modules, "vllm.lora.request", lora_module)
    calls = []

    def run_chunks(args, dataset, completed, llm, processor, sampling, lora):
        assert llm is engine
        calls.append((args.output_dir.name, len(dataset), len(completed)))
        with (args.output_dir / "predictions_raw.jsonl").open("a" if args.resume else "x", encoding="utf-8") as output:
            for row in dataset[len(completed):]:
                text = good if accept or args.output_dir.name != "city378" else bad
                output.write(json.dumps({"id": row["id"], "samples": [{
                    "text": text, "finish_reason": "stop", "tokens": 10, "seconds": 2.0}]}) + "\n")
        return {"wall_seconds": 2.0 * len(dataset), "generation_seconds": 2.0 * len(dataset),
                "new_queries": len(dataset) - len(completed), "resumed_queries": len(completed),
                "completed_queries": len(dataset)}

    monkeypatch.setattr(worker, "run_chunks", run_chunks)
    code = worker.main(["--run-dir", str(tmp_path), "--queue-config", str(queue_path)])
    assert len(loads) == 1
    assert worker.read(tmp_path / "vllm_validation/decision.json")["accepted"] is accept
    assert (full / "predictions_raw.jsonl").read_bytes().startswith(original)
    if accept:
        assert code == 0
        assert calls == [("city378", 378, 0), ("external88", 88, 0), ("full", 3, 1)]
        assert (full / "submission.zip").is_file()
        assert worker.read(tmp_path / "state.json")["status"] == "completed"
        updated = worker.read(full / "run_config.json")
        assert updated["engine"] == "hf" and updated["resume_history"][0]["engine"] == "vllm"
        assert worker.read(full / "summary.json")["resource_segments"][0]["queries"] == 1
    else:
        assert code == 3 and len(calls) == 2
        assert (full / "predictions_raw.jsonl").read_bytes() == original
        assert (full / "run_config.json").read_bytes() == original_config
        assert not (full / "submission.zip").exists()
