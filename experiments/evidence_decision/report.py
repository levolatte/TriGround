"""Render a scored visual-agent summary without inventing unrun comparisons."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


INSPECT_VIEWS = {
    "single": "single", "single_high_res": "single",
    "pair": "pair", "two_objects": "pair",
    "cross": "cross", "cross_modal": "cross", "one_object_cross_modal": "cross",
}


def load_traces(path: Path) -> list[dict]:
    """Read complete trace.json records, never predictions or GT as evidence."""
    if path.is_dir():
        files = sorted(path.rglob("trace.json"))
        if not files:
            raise ValueError(f"no trace.json files under {path}")
    elif path.name == "trace.json":
        files = [path]
    elif path.suffix == ".jsonl":
        predictions = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        files = [Path(row["trace_path"]) for row in predictions]
    else:
        raise ValueError("--traces requires trace.json, a directory of trace.json files, or predictions.jsonl")
    if len(set(files)) != len(files):
        raise ValueError("duplicate trace path")
    traces = [json.loads(file.read_text(encoding="utf-8-sig")) for file in files]
    ids = [str(trace["id"]) for trace in traces]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate trace ID")
    if not traces:
        raise ValueError("empty trace input")
    for trace in traces:
        if not isinstance(trace.get("events"), list):
            raise ValueError(f"{trace['id']}: complete trace events are required")
    return traces


def _path(value) -> str:
    # Compare the full path, never only a basename or the existence of a PNG.
    return str(value).replace("\\", "/")


def _stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "min": None, "mean": None, "max": None}
    return {"count": len(values), "min": min(values), "mean": sum(values) / len(values), "max": max(values)}


def audit_traces(traces: list[dict]) -> dict:
    """Audit observed tool behavior against actual later Processor geometry."""
    requested, selected_tools, invoked_tools, statuses = Counter(), Counter(), Counter(), Counter()
    error_reasons, error_kinds, inspect_views = Counter(), Counter(), Counter()
    inspect_view_statuses: dict[str, Counter] = {}
    returned = consumed = processor_matched = 0
    unconsumed_ids = set()
    widths, heights, areas, ratios = [], [], [], []
    unknown_ratios = ineffective = 0
    per_trace, processed_targets = [], []
    for trace in traces:
        trace_id = str(trace["id"])
        events = trace["events"]
        image_returns: dict[str, int] = {}
        for index, event in enumerate(events):
            action = event.get("action") or {}
            executed_action = event.get("executed_action") or action
            if action.get("action"):
                requested[action["action"]] += 1
            if executed_action.get("action") and "executed_action" in event:
                selected_tools[executed_action["action"]] += 1
                if "tool_seconds" in event:
                    invoked_tools[executed_action["action"]] += 1
            observation = event.get("observation") or {}
            status = observation.get("status")
            if status:
                statuses[status] += 1
            if status == "ERROR":
                error_reasons[observation.get("text") or "<empty>"] += 1
                error_kinds[(observation.get("data") or {}).get("kind") or "<unknown>"] += 1
            if executed_action.get("action") == "inspect_regions":
                view = INSPECT_VIEWS.get(executed_action.get("view"), executed_action.get("view") or "<missing>")
                inspect_views[view] += 1
                inspect_view_statuses.setdefault(view, Counter())[status or "<no observation>"] += 1
            for image in observation.get("images") or []:
                image_path = image.get("path")
                if not image_path:
                    raise ValueError(f"{trace_id}: tool image without path")
                image_returns.setdefault(_path(image_path), index)
        returned += len(image_returns)
        missing, seen = [], []
        for path, origin in image_returns.items():
            # The next decision must receive the observation; a much later
            # reappearance cannot explain the intervening controller action.
            matches = [(event, item) for event in events[origin + 1:origin + 2]
                       for item in (event.get("usage") or {}).get("image_geometry", [])
                       if item.get("path") is not None and _path(item["path"]) == path]
            processor_matched += bool(matches)
            geometry = next((item for event, item in matches
                             if (event.get("usage") or {}).get("model_seconds", 0) > 0), None)
            if geometry is None:
                missing.append(path)
                continue
            seen.append(path)
            consumed += 1
            for target in geometry.get("targets") or []:
                record = {"id": trace_id, "path": path, "modality": geometry.get("modality"),
                          "candidate_id": target.get("id"), "width_px": target.get("width_px"),
                          "height_px": target.get("height_px"), "area_px": target.get("area_px"),
                          "area_ratio_to_global": target.get("area_ratio_to_global"),
                          "effective_zoom": target.get("effective_zoom")}
                processed_targets.append(record)
                for key, values in (("width_px", widths), ("height_px", heights), ("area_px", areas)):
                    if record[key] is not None:
                        values.append(float(record[key]))
                ratio = record["area_ratio_to_global"]
                if ratio is None:
                    unknown_ratios += 1
                else:
                    ratios.append(float(ratio))
                    ineffective += ratio <= 1
        if missing:
            unconsumed_ids.add(trace_id)
        per_trace.append({"id": trace_id, "returned_images": len(image_returns),
                          "consumed_images": len(seen), "unconsumed_images": missing})
    return {
        "traces": len(traces),
        "requested_action_counts": dict(requested),
        "selected_tool_action_counts": dict(selected_tools), "tool_invocation_counts": dict(invoked_tools),
        "tool_status_counts": dict(statuses), "error_reasons": dict(error_reasons),
        "error_kinds": dict(error_kinds),
        "inspect_view_counts": dict(inspect_views),
        "inspect_view_status_counts": {view: dict(counts) for view, counts in inspect_view_statuses.items()},
        "returned_tool_images": returned, "processor_matched_tool_images": processor_matched,
        "processor_only_rejected_images": processor_matched - consumed,
        "consumed_tool_images": consumed,
        "unconsumed_tool_images": returned - consumed,
        "trace_ids_with_unconsumed_images": sorted(unconsumed_ids),
        "processed_target_geometry": {
            "count": len(processed_targets), "width_px": _stats(widths), "height_px": _stats(heights),
            "area_px": _stats(areas), "area_ratio_to_global": _stats(ratios),
            "unknown_area_ratio_count": unknown_ratios,
            "effective_zoom_le_1_count": ineffective,
            "effective_zoom_le_1_fraction_of_known": ineffective / len(ratios) if ratios else None,
        },
        "processed_targets": processed_targets, "per_trace": per_trace,
    }


def fmt(value, digits=4):
    return f"{value:.{digits}f}" if value is not None else "—"


def render(summary: dict) -> str:
    lines = ["# 视觉 Agent 实验结果", "",
             f"离线评分分母：{summary['denominator']} 题、{summary['image_groups']} 图组。",
             "缺少合法最终框的题仍在分母中，IoU 记 0。", "",
             "## 定位成绩", "",
             "| 运行 | 集合 | 题数 | ACC@0.5 | mIoU | ACC@0.7 | 初始候选覆盖 | 最终候选覆盖 | 覆盖仍选错 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, splits in summary["runs"].items():
        for split, score in splits.items():
            lines.append(f"| {name} | {split} | {score['n']} | {score['hits_0.5']}/{score['n']} "
                         f"({fmt(score['acc_0.5'])}) | {fmt(score['mean_iou'])} | "
                         f"{score['hits_0.7']}/{score['n']} ({fmt(score['acc_0.7'])}) | "
                         f"{score.get('initial_coverage_hits_0.5', '—')} | "
                         f"{score.get('final_coverage_hits_0.5', '—')} | "
                         f"{score.get('covered_but_final_wrong', '—')} |")
    lines += ["", "## 配对差异（同题同图组）", "",
              "| 对比：右减左 | 集合 | 纠正 | 破坏 | 净命中 | ΔACC@0.5 | 图组重采样 95% 区间 | ΔmIoU |",
              "|---|---|---:|---:|---:|---:|---|---:|"]
    for name, splits in summary["pairs"].items():
        for split, item in splits.items():
            lo, hi = item["group_bootstrap"]["delta_95ci"]["acc_0.5"]
            lines.append(f"| {name} | {split} | {item['corrected']} | {item['damaged']} | "
                         f"{item['net_hits_0.5']:+d} | {fmt(item['delta']['acc_0.5'])} | "
                         f"[{fmt(lo)}, {fmt(hi)}] | {fmt(item['delta']['mean_iou'])} |")
    lines += ["", "## 代价与协议状态", "",
              "| 运行 | 模型调用/题 | 工具调用/题 | 搜索/题 | 输入总 token/题 | 输入文本 token/题 | 输入视觉 token/题 | 输出 token/题 | 模型秒/题 | 工具秒/题 | Processor秒/题 | 初始图准备秒/题 | 总秒/题 | 峰值已分配显存字节 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, splits in summary["runs"].items():
        score = splits["all"]
        cost = score["cost_mean_per_query"]
        if score.get("cost_observed_queries") == 0:
            lines.append(f"| {name}（未记录成本） | " + " | ".join(["—"] * 13) + " |")
            continue
        lines.append(f"| {name} | {fmt(cost.get('model_calls'))} | {fmt(cost.get('tool_calls'))} | "
                     f"{fmt(cost.get('search_calls'))} | {fmt(cost.get('input_tokens'))} | "
                     f"{fmt(cost.get('input_text_tokens'))} | {fmt(cost.get('visual_tokens'))} | "
                     f"{fmt(cost.get('output_tokens'))} | {fmt(cost.get('model_seconds'))} | "
                     f"{fmt(cost.get('tool_seconds'))} | {fmt(cost.get('preprocess_seconds'))} | "
                     f"{fmt(cost.get('preparation_seconds'))} | "
                     f"{fmt(cost.get('elapsed_seconds'))} | {score['peak_memory_bytes']} |")
    lines += [""]
    for name, splits in summary["runs"].items():
        score = splits["all"]
        lines.append(f"- **{name}**：最终状态 {json.dumps(score['final_status_counts'], ensure_ascii=False)}；"
                     f"工具状态 {json.dumps(score['tool_status_counts'], ensure_ascii=False)}；"
                     f"非法动作 {score['cost_total'].get('invalid_actions', 0)}。")
        if "initial_uncovered" in score:
            lines.append(f"  - 失败线索：初始池未覆盖 {score['initial_uncovered']}、最终池未覆盖 "
                         f"{score['final_uncovered']}、搜索新补覆盖 {score['newly_covered_by_search']}、"
                         f"最终池覆盖却选错 {score['covered_but_final_wrong']}、无合法框 "
                         f"{score['invalid_final_bbox']}；KEEP {score['keep_count']} 次，命中 "
                         f"{score['keep_hits_0.5']} 次。")
    lines += ["", "## 终局来源与候选覆盖", "",
              "工具调用次数只描述执行量和成本，不代表能力。搜索新增候选覆盖与模型直接输出框分别统计。", "",
              "| 运行 | 终局来源 | 数量 | ACC@0.5 | mIoU |",
              "|---|---|---:|---:|---:|"]
    for name, splits in summary["runs"].items():
        for source, score in sorted(splits["all"].get("finish_source_scores", {}).items()):
            lines.append(f"| {name} | {source} | {score['n']} | {fmt(score['acc_0.5'])} | "
                         f"{fmt(score['mean_iou'])} |")
    lines += ["", "| 运行 | 初始池覆盖 | 初始池缺失 | 搜索新增覆盖 | BBox命中初始缺池 | BBox独有救回 |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, splits in summary["runs"].items():
        score = splits["all"]
        strata = score.get("initial_pool_strata", {})
        coverage_data = score.get("search_and_bbox_coverage", {})
        covered = strata.get("covered", {})
        uncovered = strata.get("uncovered", {})
        lines.append(f"| {name} | {covered.get('n', '—')} ({fmt(covered.get('acc_0.5'))}) | "
                     f"{uncovered.get('n', '—')} ({fmt(uncovered.get('acc_0.5'))}) | "
                     f"{coverage_data.get('newly_covered_by_search', '—')} | "
                     f"{coverage_data.get('predicted_bbox_hits_on_initially_uncovered', '—')} | "
                     f"{coverage_data.get('predicted_bbox_unique_rescues_without_final_candidate_coverage', '—')} |")
    lines += ["", "## 代表性纠正与破坏", ""]
    for name, splits in summary["pairs"].items():
        cases = splits["all"]
        lines += [f"### {name}", ""]
        for title, key in (("纠正", "representative_corrections"), ("破坏", "representative_damages")):
            examples = cases[key]
            if not examples:
                lines.append(f"- {title}：无。")
            for case in examples:
                lines.append(f"- {title} `{case['id']}`：IoU {fmt(case['before_iou'])} → "
                             f"{fmt(case['after_iou'])}；Query：{case['query'] or '未提供'}。")
        lines.append("")
    lines += ["## 解释边界", ""]
    lines += [f"- {item}" for item in summary["interpretation"]]
    lines += ["", "此报告只汇总已经执行并完整评分的运行。工具图像是否进入模型，需同时核对逐题 trace 的处理器视觉网格记录。", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, help="evaluate.py output summary.json; omit for GT-free audit only")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--traces", type=Path, help="trace.json, trace directory, or predictions.jsonl")
    args = parser.parse_args(argv)
    if not args.evaluation and not args.traces:
        parser.error("--evaluation or --traces is required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.evaluation:
        summary = json.loads(args.evaluation.read_text(encoding="utf-8-sig"))
        args.output.write_text(render(summary), encoding="utf-8")
        print(args.output)
    if args.traces:
        audit_path = args.output.parent / "behavior_audit.json" if args.evaluation else args.output
        audit_path.write_text(json.dumps(audit_traces(load_traces(args.traces)), ensure_ascii=False, indent=2) + "\n",
                              encoding="utf-8")
        print(audit_path)


if __name__ == "__main__":
    main()
