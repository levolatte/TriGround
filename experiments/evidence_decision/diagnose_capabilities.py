"""Offline capability-slice and evidence-recovery diagnostics.

Capability labels are supplied by a GT-free evaluation manifest; this command
uses GT only through an already-frozen offline evaluation bundle.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from . import evaluate, report


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_capability_slices(path: Path, expected_ids: set[str]) -> dict[str, list[str]]:
    rows = evaluate.read_jsonl(path)
    result = {}
    for row in rows:
        sample_id = str(row["id"])
        if sample_id in result:
            raise ValueError(f"{path}: duplicate sample ID {sample_id}")
        if not isinstance(row.get("capability_slices"), list) or not row["capability_slices"]:
            raise ValueError(f"{sample_id}: capability_slices must be a nonempty list")
        tags = [str(tag) for tag in row["capability_slices"]]
        if len(set(tags)) != len(tags):
            raise ValueError(f"{sample_id}: duplicate capability slice")
        result[sample_id] = tags
    missing, extra = expected_ids - result.keys(), result.keys() - expected_ids
    if missing or extra:
        raise ValueError(f"{path}: slice denominator mismatch: missing={len(missing)}, extra={len(extra)}")
    return result


def _action(event):
    value = event.get("action") or event.get("executed_action") or {}
    if isinstance(value, dict):
        if isinstance(value.get("name"), str):
            return value["name"], value.get("arguments") or {}
        if isinstance(value.get("action"), str):
            return value["action"], {key: item for key, item in value.items() if key != "action"}
    return None, {}


def evidence_transitions(traces: list[dict], scored_rows: dict[str, dict], gt: dict[str, dict]) -> dict:
    triggers = {"UNKNOWN": Counter(), "EMPTY": Counter(), "ERROR": Counter(),
                "no_new_evidence": Counter()}
    affected = {key: set() for key in triggers}
    uninformative_retries = Counter()
    for trace in traces:
        events = trace["events"]
        history = set()
        for index, event in enumerate(events):
            name, args = _action(event)
            signature = json.dumps([name, args], sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            observation = event.get("observation") or {}
            data = observation.get("data") or {}
            status = observation.get("status")
            repeat_signal = bool(data.get("no_new_evidence") or data.get("cached")) or signature in history
            if name:
                history.add(signature)
            next_name, next_args = _action(events[index + 1]) if index + 1 < len(events) else ("NO_NEXT_ACTION", {})
            next_signature = json.dumps([next_name, next_args], sort_keys=True, ensure_ascii=False,
                                        separators=(",", ":"))
            final_hit = False
            if trace["id"] in scored_rows and trace["id"] in gt:
                final_hit = evaluate.iou(evaluate.prediction_box(scored_rows[trace["id"]]),
                                         gt[trace["id"]]["bbox"]) >= .5
            if status in triggers:
                triggers[status][next_name] += 1
                affected[status].add((trace["id"], final_hit))
            if repeat_signal:
                triggers["no_new_evidence"][next_name] += 1
                affected["no_new_evidence"].add((trace["id"], final_hit))
                if next_name == name and next_signature == signature:
                    uninformative_retries[status or "UNKNOWN_STATUS"] += 1
    return {
        "trigger_next_action_counts": {key: dict(value) for key, value in triggers.items()},
        "affected_episode_final_accuracy": {
            key: {"n": len(items), "hits_0.5": sum(hit for _, hit in items),
                  "acc_0.5": (sum(hit for _, hit in items) / len(items)) if items else None}
            for key, items in affected.items()},
        "same_action_retries_after_repeat_or_no_new_evidence": dict(uninformative_retries),
        "scope": "Descriptive episode outcomes after tool states; action counts alone are not capability scores.",
    }


def _probe_sensitivity(path: Path | None) -> dict | None:
    if path is None:
        return None
    pairs = evaluate.read_jsonl(path)
    action_name_changes = action_argument_changes = generated_pairs = 0
    by_branch = defaultdict(Counter)
    for row in pairs:
        full, masked = row.get("full") or {}, row.get("masked") or {}
        full_action, masked_action = full.get("predicted_action"), masked.get("predicted_action")
        if full_action is not None and masked_action is not None:
            generated_pairs += 1
            action_name_changes += full_action.get("name") != masked_action.get("name")
            action_argument_changes += full_action.get("arguments") != masked_action.get("arguments")
            by_branch[full_action.get("name") or "<missing>"]["full"] += 1
            by_branch[masked_action.get("name") or "<missing>"]["masked"] += 1
    return {"pairs": len(pairs), "both_generated": generated_pairs,
            "action_name_changed": action_name_changes,
            "action_arguments_changed": action_argument_changes,
            "full_vs_masked_action_counts": {key: dict(value) for key, value in by_branch.items()},
            "interpretation": "Paired sensitivity to withholding listed tool-image pixels only; it does not prove image use or task success."}


def diagnose(evaluation_path: Path, capability_manifest: Path, output_dir: Path,
             traces_path: Path | None = None, trace_run: str | None = None,
             paired_results: Path | None = None) -> dict:
    summary = _read_json(evaluation_path)
    gt = evaluate.load_gt(Path(summary["gt"]))
    if summary.get("manifest"):
        manifest_ids = [str(row["id"]) for row in evaluate.read_jsonl(Path(summary["manifest"]))]
        if len(set(manifest_ids)) != len(manifest_ids) or not set(manifest_ids) <= set(gt):
            raise ValueError("evaluation manifest has duplicate IDs or IDs absent from GT")
        gt = {sample_id: gt[sample_id] for sample_id in manifest_ids}
    expected_ids = set(gt)
    slices = load_capability_slices(capability_manifest, expected_ids)
    run_rows = {name: evaluate.load_run(Path(path), expected_ids)
                for name, path in summary["run_sources"].items()}
    if set(run_rows) != set(summary["runs"]):
        raise ValueError("evaluation run sources differ from evaluated run names")
    slice_ids = defaultdict(list)
    for sample_id, tags in slices.items():
        for tag in tags:
            slice_ids[tag].append(sample_id)
    result = {
        "diagnostic_only": True, "official_score_bundle": str(evaluation_path),
        "capability_manifest": str(capability_manifest), "denominator": len(expected_ids),
        "overlapping_slices_allowed": True,
        "capability_slices": {
            name: {run: evaluate.score_subset(rows, gt, ids)
                   for run, rows in run_rows.items()}
            for name, ids in sorted(slice_ids.items())},
        "evidence_transitions": None, "paired_image_mask_sensitivity": _probe_sensitivity(paired_results),
    }
    if traces_path is not None:
        if trace_run not in run_rows:
            raise ValueError("--trace-run must name a run in the evaluation bundle")
        traces = report.load_traces(traces_path)
        trace_ids = {str(trace["id"]) for trace in traces}
        extra = trace_ids - expected_ids
        if extra:
            raise ValueError(f"trace IDs absent from evaluation: {len(extra)}")
        result["evidence_transitions"] = evidence_transitions(traces, run_rows[trace_run], gt)
        result["trace_run"] = trace_run
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "capability_diagnostics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "capability_diagnostics.md").write_text(render(result), encoding="utf-8")
    return result


def render(result: dict) -> str:
    lines = ["# 能力切片与证据依赖诊断", "",
             f"固定评分分母：{result['denominator']} 题。切片允许重叠；下表不是新的官方总分。", "",
             "| 能力切片 | 运行 | n | ACC@0.5 | mIoU | 初始池覆盖 | 最终池覆盖 | 搜索新增覆盖 | BBox独有救回 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for slice_name, runs in result["capability_slices"].items():
        for run, score in runs.items():
            coverage = score.get("search_and_bbox_coverage", {})
            lines.append(f"| {slice_name} | {run} | {score['n']} | {score['acc_0.5']:.4f} | "
                         f"{score['mean_iou']:.4f} | {score.get('initial_coverage_hits_0.5', '—')} | "
                         f"{score.get('final_coverage_hits_0.5', '—')} | "
                         f"{coverage.get('newly_covered_by_search', '—')} | "
                         f"{coverage.get('predicted_bbox_unique_rescues_without_final_candidate_coverage', '—')} |")
    if result.get("evidence_transitions"):
        transitions = result["evidence_transitions"]
        lines += ["", "## 工具状态后的行为（描述统计）", "",
                  "工具调用次数不作为能力分；最终命中率仅描述发生该状态的轨迹，不作因果解释。", "",
                  "| 触发状态 | 触发数 | 后续动作计数 | 相关轨迹ACC@0.5 |",
                  "|---|---:|---|---:|"]
        for key, actions in transitions["trigger_next_action_counts"].items():
            affected = transitions["affected_episode_final_accuracy"][key]
            count = sum(actions.values())
            lines.append(f"| {key} | {count} | {json.dumps(actions, ensure_ascii=False)} | "
                         f"{affected['acc_0.5'] if affected['acc_0.5'] is not None else '—'} |")
        lines.append("相同动作在无新证据后重试：" + json.dumps(
            transitions["same_action_retries_after_repeat_or_no_new_evidence"], ensure_ascii=False) + "。")
    if result.get("paired_image_mask_sensitivity"):
        probe = result["paired_image_mask_sensitivity"]
        lines += ["", "## 局部工具图遮蔽配对探针", "",
                  f"共有{probe['pairs']}对状态，两侧均成功生成{probe['both_generated']}对；"
                  f"动作名变化{probe['action_name_changed']}对，参数变化{probe['action_arguments_changed']}对。",
                  probe["interpretation"], ""]
    else:
        lines.append("")
    lines.append("这是离线诊断产物，不与正式评分文件合并。")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True,
                        help="evaluate.py full-denominator summary.json")
    parser.add_argument("--capability-manifest", type=Path, required=True,
                        help="same scored IDs with a nonempty capability_slices list")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--traces", type=Path, help="optional complete traces for recovery-state audit")
    parser.add_argument("--trace-run", help="evaluated run name corresponding to --traces")
    parser.add_argument("--paired-results", type=Path,
                        help="optional probe_decisions.py full/masked paired_results.jsonl")
    args = parser.parse_args(argv)
    if bool(args.traces) != bool(args.trace_run):
        parser.error("--traces and --trace-run must be provided together")
    result = diagnose(args.evaluation, args.capability_manifest, args.output_dir,
                      args.traces, args.trace_run, args.paired_results)
    print(json.dumps({"denominator": result["denominator"],
                      "capability_slice_counts": {key: next(iter(runs.values()))["n"]
                                                  for key, runs in result["capability_slices"].items()}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
