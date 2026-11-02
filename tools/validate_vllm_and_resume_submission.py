"""Validate vLLM NF4 on both complete development splits, then resume the official run."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.eval_quant_grounding import completed_prediction_prefix, positive_int, read_jsonl, score_rows, summarize, write_jsonl
from tools.predict_quant_submission import load_queries, package_predictions
from tools.predict_vllm_quant_submission import build_llm_options, run_chunks


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_decision(results: dict, city_min_hits: int, external_min_hits: int,
                  baseline_seconds: float, min_time_reduction: float) -> dict:
    counts = {"city378": 378, "external88": 88}
    minimums = {"city378": city_min_hits, "external88": external_min_hits}
    complete = all(results[key]["queries"] == size for key, size in counts.items())
    hits_pass = all(results[key]["hits"] >= minimums[key] for key in counts)
    wall = sum(results[key]["resource"]["wall_seconds"] for key in counts)
    seconds_per_query = wall / sum(counts.values())
    ceiling = baseline_seconds * (1 - min_time_reduction)
    return {
        "accepted": complete and hits_pass and seconds_per_query <= ceiling,
        "complete": complete, "hits_pass": hits_pass,
        "speed_pass": seconds_per_query <= ceiling,
        "minimum_hits": minimums, "hits": {key: results[key]["hits"] for key in counts},
        "development_queries": sum(counts.values()), "development_wall_seconds": wall,
        "seconds_per_query": seconds_per_query, "max_seconds_per_query": ceiling,
        "baseline_seconds_per_query": baseline_seconds,
        "min_time_reduction": min_time_reduction,
        "speed_measurement": "request preparation plus complete generate calls; model loading excluded",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adopt-existing-validation", action="store_true",
                        help="Use completed validation and explicit user adoption without rerunning evaluation")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--code-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--queue-config", type=Path)
    parser.add_argument("--chunk-size", type=positive_int, default=64)
    parser.add_argument("--max-num-seqs", type=positive_int, default=8)
    parser.add_argument("--city-min-hits", type=int, default=313)
    parser.add_argument("--external-min-hits", type=int, default=84)
    parser.add_argument("--baseline-seconds-per-query", type=float, default=4.0)
    parser.add_argument("--min-time-reduction", type=float, default=0.10)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (not 0 <= args.city_min_hits <= 378 or not 0 <= args.external_min_hits <= 88
            or args.baseline_seconds_per_query <= 0 or not 0 <= args.min_time_reduction < 1):
        raise ValueError("invalid validation thresholds")
    root = args.run_dir.resolve()
    full = root / "full"
    validation = root / "vllm_validation"
    validation.mkdir(exist_ok=True)
    queue = read(args.queue_config or args.code_dir / "configs/quant_grounding/queue.json")
    config = read(full / "run_config.json")
    if queue["model"] != config["model"]:
        raise ValueError("validation and submission base models differ")
    if (config["engine"], config["precision"], config["answer_mode"], config["format_policy"]) != (
            "hf", "nf4", "short", "punctuation"):
        raise ValueError("expected the original HF NF4 punctuation submission configuration")
    options = build_llm_options(config, args.max_num_seqs)
    state_path = root / "state.json"
    state = read(state_path) if state_path.exists() else {}
    report = {"status": "loading", "model": config["model"], "adapter": config["adapter"],
              "options": options, "results": {}}

    if args.adopt_existing_validation:
        previous_report = read(validation / "report.json")
        previous_decision = read(validation / "decision.json")
        if not previous_decision["complete"]:
            raise ValueError("existing validation is incomplete")
        if (previous_report["model"], previous_report["adapter"], previous_report["options"]) != (config["model"], config["adapter"], options):
            raise ValueError("existing validation model or options differ")
        report["results"] = previous_report["results"]
        validation = root / "vllm_user_adoption"
        validation.mkdir(exist_ok=True)

    def stage(name: str, **extra):
        state.update(status="running", stage=name, worker_pid=os.getpid(),
                     updated_at=datetime.now(timezone.utc).isoformat(), **extra)
        write(state_path, state)
        report["status"] = name
        write(validation / "report.json", report)
        print(json.dumps({"stage": name, **extra}, ensure_ascii=False), flush=True)

    try:
        datasets = {"city378": read_jsonl(Path(queue["city_dataset"])),
                    "external88": read_jsonl(Path(queue["external_dataset"]))}
        for key, count in (("city378", 378), ("external88", 88)):
            if len(datasets[key]) != count or len({row["id"] for row in datasets[key]}) != count:
                raise ValueError(f"{key}: expected {count} unique complete development queries")
        stage("vllm_validation_loading")
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        processor = AutoProcessor.from_pretrained(config["model"])
        before = time.monotonic()
        llm = LLM(**options)
        report["loading_seconds"] = time.monotonic() - before
        sampling = SamplingParams(temperature=0, max_tokens=config["max_tokens"], seed=config["seed"])
        lora = LoRARequest("grounding", 1, config["adapter"])
        if args.adopt_existing_validation:
            decision = {**previous_decision, "accepted": True, "user_authorized": True,
                        "reason": "Explicit user adoption after reviewing per-query differences",
                        "original_decision": str(root / "vllm_validation/decision.json")}
        else:
            for key, dataset in datasets.items():
                output = validation / key
                output.mkdir(exist_ok=True)
                stage(f"vllm_validation_{key}")
                run_args = argparse.Namespace(output_dir=output, data_root=Path(queue["media_root"]),
                                              resume=False, chunk_size=args.chunk_size)
                resource = run_chunks(run_args, dataset, [], llm, processor, sampling, lora)
                predictions = read_jsonl(output / "predictions_raw.jsonl")
                scored = score_rows(dataset, predictions, "short", "acc05", "punctuation")
                write_jsonl(output / "predictions.jsonl", scored)
                summary = summarize(scored)
                summary.update(answer_mode="short", reward_mode="acc05", format_policy="punctuation",
                               samples_per_query=1, resource=resource, engine="vllm", precision="nf4")
                write(output / "summary.json", summary)
                report["results"][key] = {"queries": len(scored),
                                          "hits": sum(row["samples"][0]["hit"] for row in scored),
                                          "resource": resource, "output_dir": str(output)}
                write(validation / "report.json", report)
            decision = make_decision(report["results"], args.city_min_hits, args.external_min_hits,
                                     args.baseline_seconds_per_query, args.min_time_reduction)
        write(validation / "decision.json", decision)
        report["decision"] = decision
        if not decision["accepted"]:
            stage("vllm_validation_rejected", decision=decision)
            state["status"] = "validation_rejected"
            write(state_path, state)
            return 3

        dataset = read_jsonl(full / "unlabeled.jsonl")
        source = load_queries(Path(config["queries"]), config["expected_queries"])
        if [row["id"] for row in dataset] != list(source):
            raise ValueError("official prepared IDs differ from original query order")
        completed = completed_prediction_prefix(full / "predictions_raw.jsonl", dataset)
        stage("vllm_submission_resuming", completed_queries=len(completed), expected_queries=len(dataset))
        history = {"engine": "vllm", "options": options, "completed_queries": len(completed),
                   "chunk_size": args.chunk_size, "validation": str(validation / "decision.json"),
                   "started_at": datetime.now(timezone.utc).isoformat()}
        config.setdefault("resume_history", []).append(history)
        write(full / "run_config.json", config)
        run_args = argparse.Namespace(output_dir=full, data_root=Path(config["data_root"]),
                                      resume=True, chunk_size=args.chunk_size)
        resource = run_chunks(run_args, dataset, completed, llm, processor, sampling, lora)
        predictions = read_jsonl(full / "predictions_raw.jsonl")
        completed_prediction_prefix(full / "predictions_raw.jsonl", dataset)
        if len(predictions) != len(source):
            raise ValueError("official generation did not complete every query")
        stage("vllm_submission_packaging", completed_queries=len(predictions), expected_queries=len(source))
        packaged = package_predictions(source, predictions, full, "punctuation")
        hf_seconds = sum(row["samples"][0].get("seconds", 0.0) for row in completed)
        segments = [{"engine": "hf", "queries": len(completed), "timed_seconds": hf_seconds},
                    {"engine": "vllm", "queries": resource["new_queries"], "resource": resource}]
        summary = {"completed_queries": len(predictions), "expected_queries": len(source),
                   "format_policy": "punctuation", "engine": "hf_then_vllm", "resource": resource,
                   "resource_segments": segments, "validation_decision": decision, **packaged}
        if (full / "summary.json").exists():
            summary["previous_summary"] = read(full / "summary.json")
        write(full / "summary.json", summary)
        report["submission"] = summary
        stage("completed", completed_queries=len(predictions), expected_queries=len(source), **packaged)
        state["status"] = "completed"
        write(state_path, state)
        return 0
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        stage("failed", error=report["error"])
        state["status"] = "failed"
        write(state_path, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
