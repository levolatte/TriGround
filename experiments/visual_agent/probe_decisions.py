"""One-shot Qwen probes of exported SFT decisions; no tools or GT scoring.

This diagnoses action imitation on training inputs. It is not an autonomous
Agent rollout or a generalization metric.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

from .controller import parse_action
from .model import InputBudgetExceeded, QwenBackend
from .profiles import PROFILES
from .train import read_decisions


def _canonical(action):
    return json.dumps(action, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _arguments(action):
    return {key: value for key, value in action.items() if key != "evidence_note"} if isinstance(
        action.get("evidence_note"), str) else action


def probe_rows(rows, backend, *, started=None, max_run_seconds=None):
    """Yield one record per input row, including time- and budget-limited rows."""
    started = time.perf_counter() if started is None else started
    profile = PROFILES["sft"]
    for index, row in enumerate(rows):
        target = json.loads(row["target_action"])
        if not isinstance(target, dict) or not isinstance(target.get("action"), str):
            raise ValueError(f"decision {index}: invalid target_action")
        record = {"row_index": index, "id": str(row["id"]),
                  "decision_index": row.get("decision_index"),
                  "origin": row.get("origin", row.get("trajectory_origin", "unknown")),
                  "label_source": row.get("label_source", "unknown"),
                  "target_action": row["target_action"], "target_action_name": target["action"],
                  "status": "NOT_RUN_TIME_LIMIT", "raw_output": None, "usage": None,
                  "predicted_action": None, "exact_canonical_match": False,
                  "action_arguments_match": False,
                  "action_name_match": False,
                  "finish_id_match": False if target["action"] == "finish" else None}
        if max_run_seconds is not None and time.perf_counter() - started >= max_run_seconds:
            yield record
            continue
        backend.begin_sample()
        try:
            generated = backend.generate(row["messages"], 128, profile,
                                         profile.cumulative_visual_tokens)
        except InputBudgetExceeded as error:
            record.update(status="INPUT_BUDGET_EXCEEDED", usage=error.usage,
                          error=str(error))
            yield record
            continue
        record.update(status="GENERATED", raw_output=generated["raw_output"],
                      usage=generated["usage"])
        try:
            predicted = parse_action(generated["raw_output"])
        except ValueError as error:
            record.update(status="PARSE_ERROR", error=str(error))
            yield record
            continue
        record["predicted_action"] = predicted
        record["exact_canonical_match"] = _canonical(predicted) == _canonical(target)
        record["action_arguments_match"] = _canonical(_arguments(predicted)) == _canonical(_arguments(target))
        record["action_name_match"] = predicted["action"] == target["action"]
        if target["action"] == "finish":
            record["finish_id_match"] = (predicted["action"] == "finish" and
                                         str(predicted.get("candidate_id")) == str(target.get("candidate_id")))
        yield record


def summarize(records):
    """Use every requested row as denominator, including unrun and failed rows."""
    def counts(items):
        n = len(items)
        finishes = [item for item in items if item["target_action_name"] == "finish"]
        exact = sum(item["exact_canonical_match"] for item in items)
        arguments = sum(item["action_arguments_match"] for item in items)
        name = sum(item["action_name_match"] for item in items)
        finish_id = sum(item["finish_id_match"] is True for item in finishes)
        return {"rows": n, "exact_canonical_matches": exact,
                "exact_canonical_rate": exact / n if n else None,
                "action_arguments_matches": arguments,
                "action_arguments_rate": arguments / n if n else None,
                "action_name_matches": name, "action_name_rate": name / n if n else None,
                "finish_targets": len(finishes), "finish_id_matches": finish_id,
                "finish_id_rate": finish_id / len(finishes) if finishes else None,
                "input_budget_exceeded": sum(item["status"] == "INPUT_BUDGET_EXCEEDED" for item in items),
                "parse_errors": sum(item["status"] == "PARSE_ERROR" for item in items),
                "not_run_time_limit": sum(item["status"] == "NOT_RUN_TIME_LIMIT" for item in items)}

    result = {"all": counts(records)}
    for field, title in (("target_action_name", "by_target_action"),
                         ("origin", "by_origin"), ("label_source", "by_label_source")):
        groups = defaultdict(list)
        for record in records:
            groups[record[field]].append(record)
        result[title] = {key: counts(items) for key, items in sorted(groups.items())}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-run-seconds", type=float)
    args = parser.parse_args(argv)
    rows = read_decisions(args.decisions)
    if not rows:
        raise ValueError("decisions file is empty")
    if args.output_dir.exists():
        raise FileExistsError(f"probe output already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    backend = QwenBackend(args.model, args.adapter)
    records = []
    with (args.output_dir / "decisions.jsonl").open("w", encoding="utf-8") as handle:
        for record in probe_rows(rows, backend, started=started,
                                 max_run_seconds=args.max_run_seconds):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            records.append(record)
    report = {"diagnostic_only": True, "autonomous_agent_result": False,
              "generalization_metric": False, "input_decisions": str(args.decisions.resolve()),
              "model": args.model, "adapter": args.adapter,
              "model_load_seconds": backend.load_seconds,
              "elapsed_seconds": time.perf_counter() - started,
              "summary": summarize(records)}
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"]["all"], ensure_ascii=False))


if __name__ == "__main__":
    main()
