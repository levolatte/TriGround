"""Export real Agent traces into GT-audited, one-decision SFT rows.

GT is read only here, after online traces have been written. The exported
``original_action`` is never overwritten when the final label is corrected.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from .state import build_state_messages


def iou(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection)


def _rows(path):
    path = Path(path)
    if path.is_dir():
        # Runner layout is traces/<sample_id>/trace.json. Also accept older
        # flat per-sample JSON files, but never ingest run config or event files.
        nested = sorted(path.rglob("trace.json"))
        flat = sorted(entry for entry in path.glob("*.json") if entry.name != "trace.json")
        for entry in [*nested, *flat]:
            payload = json.loads(entry.read_text(encoding="utf-8-sig"))
            if isinstance(payload, dict) and {"id", "events", "manifest_row"} <= set(payload):
                yield payload
            elif entry in nested:
                raise ValueError(f"{entry}: trace.json lacks required trace fields")
    elif path.suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                yield json.loads(line)
    else:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        yield from payload if isinstance(payload, list) else [payload]


def load_frozen_manifest(path):
    rows = {}
    for row in _rows(path):
        sample_id = str(row["id"])
        if sample_id in rows:
            raise ValueError(f"{path}: duplicate frozen sample ID {sample_id}")
        rows[sample_id] = row
    if not rows:
        raise ValueError(f"{path}: empty frozen manifest")
    return rows


def verify_frozen_splits(train, holdout, debug):
    partitions = {"train": train, "holdout": holdout, "Z debug": debug}
    for name, rows in partitions.items():
        groups = [_manifest_group(row) for row in rows.values()]
        if len(groups) != len(set(groups)):
            raise ValueError(f"{name}: frozen starts must be from different image groups")
    for left, right in (("train", "holdout"), ("train", "Z debug"), ("holdout", "Z debug")):
        shared_ids = set(partitions[left]) & set(partitions[right])
        shared_groups = {_manifest_group(row) for row in partitions[left].values()} & {
            _manifest_group(row) for row in partitions[right].values()}
        if shared_ids or shared_groups:
            raise ValueError(f"frozen {left}/{right} overlap: ids={len(shared_ids)}, groups={len(shared_groups)}")


def _manifest_group(row):
    return str(row["images"]["rgb"])


def _gt_boxes(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        return {str(key): value["bbox"] for key, value in payload.items()}
    return {str(row["id"]): row["bbox"] for row in payload}


def _candidate_pool(trace, event_index):
    events = trace["events"]
    event = events[event_index] if event_index < len(events) else None
    if event and "candidates_before" in event:
        return event["candidates_before"]
    if event_index and "candidates_after" in events[event_index - 1]:
        return events[event_index - 1]["candidates_after"]
    if event_index == len(events):
        return trace.get("final_candidates", trace["initial_candidates"])
    # The runner should record snapshots for dynamic search. It is an input
    # boundary error to train against a future candidate pool.
    if any((e.get("action") or {}).get("action") == "search_candidates" for e in events[:event_index]):
        raise ValueError(f"{trace['id']}: dynamic candidate snapshot missing at step {event_index}")
    return trace["initial_candidates"]


def _keep_box(trace):
    candidates = trace["initial_candidates"]
    keep = next((c for c in candidates if str(c.get("id")) == "KEEP"), None)
    if keep is None:
        keep = next((c for c in candidates if c.get("is_baseline")), None)
    if keep is None:
        raise ValueError(f"{trace['id']}: immutable KEEP box missing")
    return keep["bbox"]


def revised_final_label(trace, candidates, gt_box):
    """GT changes only the last label; it never supplies a nonexistent box."""
    if iou(_keep_box(trace), gt_box) >= 0.5:
        return {"action": "finish", "candidate_id": "KEEP"}, "gt_keep"
    hits = [(iou(candidate["bbox"], gt_box), candidate) for candidate in candidates
            if candidate.get("role") == "target" and str(candidate.get("id")) != "KEEP"
            and iou(candidate["bbox"], gt_box) >= 0.5]
    if not hits:
        return None, "uncovered"
    hits.sort(key=lambda pair: (-pair[0], str(pair[1]["id"])))
    return {"action": "finish", "candidate_id": hits[0][1]["id"]}, "gt_candidate"


def _canonical_action(action):
    if isinstance(action, dict) and action.get("action"):
        return json.dumps(action, ensure_ascii=False, separators=(",", ":"))
    return None


def decision_rows(trace, gt_box):
    events = trace["events"]
    rows = []
    online_status = trace.get("final_status", "FINISHED")
    final_index = next((index for index, event in enumerate(events)
                        if (event.get("action") or {}).get("action") == "finish"
                        and event.get("observation") is None
                        and online_status == "FINISHED"), None)
    repaired_invalid_final = False
    if final_index is None and online_status == "INVALID_FINAL_ACTION" and events:
        last = events[-1]
        if (isinstance(last.get("raw_output"), str) and last["raw_output"].strip()
                and (last.get("usage") or {}).get("output_tokens", 0) > 0):
            final_index = len(events) - 1
            repaired_invalid_final = True
    end = final_index + 1 if final_index is not None else len(events)
    for index in range(end):
        is_final = index == final_index
        pool = _candidate_pool(trace, index)
        history = events[:index]
        original = events[index].get("action")
        if is_final:
            label, source = revised_final_label(trace, pool, gt_box)
            if label is None:
                break
            if repaired_invalid_final:
                source += "_repaired_invalid_final"
        else:
            label, source = original, "observed_action"
            if _canonical_action(label) is None:
                continue
            if label["action"] not in {"inspect_regions", "measure_depth", "search_candidates"}:
                continue
            if (events[index].get("observation") or {}).get("status") not in {"OK", "UNKNOWN", "EMPTY"}:
                continue
        actual_messages = events[index].get("messages")
        if not actual_messages:
            raise ValueError(f"{trace['id']} step {index}: actual visible messages missing")
        if trace.get("memory") == "latest":
            visible_messages = actual_messages
        else:
            hint = actual_messages[-1]
            if hint.get("role") != "user" or not isinstance(hint.get("content"), list) or any(
                    part.get("type") != "text" for part in hint["content"]):
                raise ValueError(f"{trace['id']} step {index}: decision hint missing")
            visible_messages = build_state_messages(trace["initial_messages"], history, pool)
            visible_messages.append(hint)
        rows.append({
            "id": str(trace["id"]), "decision_index": index,
            "image_group": _image_group(trace),
            "origin": trace.get("trajectory_origin", trace.get("origin", "autonomous")),
            "messages": visible_messages,
            "target_action": _canonical_action(label),
            "original_action": original,
            "original_raw_output": events[index].get("raw_output"),
            "label_source": source,
            "online_final_status": online_status,
            "is_final": is_final,
        })
        if is_final:
            break
    return rows


def _image_group(trace):
    row = trace["manifest_row"]
    return str(row.get("known_location_group") or row.get("image_group") or row["images"]["rgb"])


def _origin(trace):
    return trace.get("trajectory_origin", trace.get("origin", "autonomous"))


def _terminal_revised(row):
    original = row["original_action"]
    return (row.get("label_source", "").endswith("_repaired_invalid_final")
            or not isinstance(original, dict)
            or original.get("action") != "finish"
            or str(original.get("candidate_id")) != str(json.loads(row["target_action"])["candidate_id"]))


def trajectory_composition(trace, rows, gt_box):
    """Count observed behavior and offline corrections for one training start."""
    tool_kinds = {"inspect_regions", "measure_depth", "search_candidates"}
    unknown_tools = set()
    switched_after_unknown = search_added = False
    for event in trace["events"]:
        if "tool_seconds" not in event:
            continue
        kind = (event.get("executed_action") or {}).get("action")
        if kind not in tool_kinds:
            continue
        switched_after_unknown |= any(previous != kind for previous in unknown_tools)
        if (event.get("observation") or {}).get("status") in {"UNKNOWN", "EMPTY"}:
            unknown_tools.add(kind)
        if kind == "search_candidates":
            before = {str(candidate["id"]) for candidate in event["candidates_before"]}
            after = {str(candidate["id"]) for candidate in event["candidates_after"]}
            search_added |= bool(after - before)
    initial_correct = iou(_keep_box(trace), gt_box) >= 0.5
    online_finished = trace.get("final_status") == "FINISHED"
    online_box = trace.get("bbox")
    terminal_revised = any(row["is_final"] and _terminal_revised(row) for row in rows)
    return {
        "unknown_empty_then_other_tool": switched_after_unknown,
        "search_added_candidates": search_added,
        "initial_c_correct_online_keep": initial_correct and online_finished and str(trace.get("selected_id")) == "KEEP",
        "initial_c_wrong_online_final_correct": not initial_correct and online_finished and
            online_box is not None and iou(online_box, gt_box) >= 0.5,
        "offline_terminal_revised": terminal_revised,
    }


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def export(args):
    frozen_train = load_frozen_manifest(args.train_manifest)
    frozen_holdout = load_frozen_manifest(args.holdout_manifest)
    frozen_debug = load_frozen_manifest(args.debug_manifest)
    verify_frozen_splits(frozen_train, frozen_holdout, frozen_debug)
    eligible = {**frozen_train, **frozen_holdout}
    boxes = _gt_boxes(args.gt)
    traces = {}
    input_trajectories = ignored_debug = ignored_outside = 0
    for trace in _rows(args.traces):
        input_trajectories += 1
        sample_id = str(trace["id"])
        if sample_id in frozen_debug:
            ignored_debug += 1
            continue
        if sample_id not in eligible:
            ignored_outside += 1
            continue
        if trace.get("mechanism") in {"A", "B"}:
            continue
        if sample_id in traces:
            raise ValueError(f"duplicate trajectory start {sample_id}; select one attempt explicitly")
        if sample_id not in boxes:
            raise ValueError(f"GT missing for training trace {sample_id}")
        frozen = eligible[sample_id]
        if _manifest_group(trace["manifest_row"]) != _manifest_group(frozen) or trace["query"] != frozen["query"]:
            raise ValueError(f"{sample_id}: trace scene/query differs from frozen manifest")
        traces[sample_id] = trace
    if not traces:
        raise ValueError("no eligible real traces")
    selected = list(traces.values())
    training, holdout, skipped_keep = [], [], 0
    prepared = [(trace, decision_rows(trace, boxes[str(trace["id"])])) for trace in selected]
    for trace, rows in prepared:
        if str(trace["id"]) in frozen_holdout:
            holdout.extend(rows)
    train_items = []
    no_legal_final = terminal_only = excluded_tool_decisions = 0
    for trace, rows in prepared:
        if str(trace["id"]) not in frozen_train:
            continue
        final = next((row for row in rows if row["is_final"]), None)
        if final is None:
            no_legal_final += 1
            excluded_tool_decisions += sum(not row["is_final"] for row in rows)
            continue
        online_box = trace.get("bbox")
        online_correct = (trace.get("final_status") == "FINISHED" and online_box is not None
                          and iou(online_box, boxes[str(trace["id"])]) >= 0.5)
        if not online_correct and _terminal_revised(final):
            terminal_only += 1
            excluded_tool_decisions += sum(not row["is_final"] for row in rows)
            rows = [final]
        train_items.append((trace, rows))
    # Keep useful non-KEEP trajectories first; then admit KEEP-final
    # trajectories up to half of terminal supervision targets.
    train_items.sort(key=lambda item: any(row["target_action"] == '{"action":"finish","candidate_id":"KEEP"}'
                                               for row in item[1]))
    for trace, rows in train_items:
        keep_count = sum(row["is_final"] and row["target_action"] == '{"action":"finish","candidate_id":"KEEP"}'
                         for row in rows)
        new_keep = sum(row["is_final"] and row["target_action"] == '{"action":"finish","candidate_id":"KEEP"}'
                       for row in training) + keep_count
        new_finals = sum(row["is_final"] for row in training) + sum(row["is_final"] for row in rows)
        if new_finals and new_keep * 2 > new_finals:
            skipped_keep += 1
            continue
        training.extend(rows)
    if not training or not holdout:
        raise ValueError("group split or KEEP cap yielded an empty train/holdout set")
    collected_train = [trace for trace in selected if str(trace["id"]) in frozen_train]
    collected_holdout = [trace for trace in selected if str(trace["id"]) in frozen_holdout]
    train_ids = {row["id"] for row in training}
    holdout_ids = {row["id"] for row in holdout}
    exported_train = [trace for trace in collected_train if str(trace["id"]) in train_ids]
    exported_holdout = [trace for trace in collected_holdout if str(trace["id"]) in holdout_ids]
    prepared_by_id = {str(trace["id"]): rows for trace, rows in prepared}
    first_id = str(collected_train[0]["id"])
    composition_names = tuple(trajectory_composition(
        collected_train[0], prepared_by_id[first_id], boxes[first_id]))
    def composition_counts(traces):
        counts = Counter()
        for trace in traces:
            sample_id = str(trace["id"])
            values = trajectory_composition(trace, prepared_by_id[sample_id], boxes[sample_id])
            counts.update(name for name, present in values.items() if present)
        return {name: counts[name] for name in composition_names}

    collected_composition = composition_counts(collected_train)
    exported_composition = composition_counts(exported_train)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "train.jsonl", training)
    _write_jsonl(args.output_dir / "holdout.jsonl", holdout)
    inventory = {"partition": "explicit_frozen_manifests",
                 "input_trajectories": input_trajectories,
                 "collected_trajectories": len(selected),
                 "collected_train_trajectories": len(collected_train),
                 "collected_holdout_trajectories": len(collected_holdout),
                 "frozen_train_starts": len(frozen_train), "frozen_holdout_starts": len(frozen_holdout),
                 "missing_train_starts": len(set(frozen_train) - set(traces)),
                 "missing_holdout_starts": len(set(frozen_holdout) - set(traces)),
                 "ignored_Z_debug_traces": ignored_debug, "ignored_outside_traces": ignored_outside,
                 "train_trajectories": len(exported_train),
                 "holdout_trajectories": len(exported_holdout),
                 "train_decisions": len(training), "holdout_decisions": len(holdout),
                 "train_groups": len({_image_group(traces[row["id"]]) for row in training}),
                 "holdout_groups": len({_image_group(traces[row["id"]]) for row in holdout}),
                 "collected_train_without_legal_final_trajectories": no_legal_final,
                 "collected_train_terminal_only_revised_trajectories": terminal_only,
                 "collected_train_excluded_tool_decisions": excluded_tool_decisions,
                 "train_label_policy": "Require a legal final label; when the online final is wrong or invalid and GT revises it, supervise only that final decision. Preserve tool decisions from online-correct trajectories, including UNKNOWN/EMPTY observations. Counts above are before the KEEP cap; holdout rows are unfiltered.",
                 "skipped_for_keep_cap": skipped_keep,
                 "train_keep_fraction": sum(row["target_action"] == '{"action":"finish","candidate_id":"KEEP"}'
                                            for row in training) / len(training),
                 "train_final_keep_fraction": sum(row["is_final"] and row["target_action"] == '{"action":"finish","candidate_id":"KEEP"}'
                                                  for row in training) / max(1, sum(row["is_final"] for row in training)),
                 "collected_train_origin_trajectories": dict(Counter(_origin(trace) for trace in collected_train)),
                 "collected_holdout_origin_trajectories": dict(Counter(_origin(trace) for trace in collected_holdout)),
                 "collected_train_online_final_status_trajectories": dict(Counter(
                     trace.get("final_status", "UNKNOWN") for trace in collected_train)),
                 "collected_holdout_online_final_status_trajectories": dict(Counter(
                     trace.get("final_status", "UNKNOWN") for trace in collected_holdout)),
                 "train_origin_trajectories": dict(Counter(_origin(trace) for trace in exported_train)),
                 "holdout_origin_trajectories": dict(Counter(_origin(trace) for trace in exported_holdout)),
                 "train_online_final_status_trajectories": dict(Counter(
                     trace.get("final_status", "UNKNOWN") for trace in exported_train)),
                 "holdout_online_final_status_trajectories": dict(Counter(
                     trace.get("final_status", "UNKNOWN") for trace in exported_holdout)),
                 "train_origin_decisions": dict(Counter(row["origin"] for row in training)),
                 "collected_train_composition_trajectories": collected_composition,
                 "train_composition_trajectories": exported_composition,
                 "selected_without_supervised_decisions": sum(not rows for _, rows in prepared),
                 "label_sources_train": {name: sum(row["label_source"] == name for row in training)
                                         for name in sorted({row["label_source"] for row in training})}}
    (args.output_dir / "inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--holdout-manifest", type=Path, required=True)
    parser.add_argument("--debug-manifest", type=Path, required=True, help="Frozen Z debug starts to exclude")
    parser.add_argument("--gt", type=Path, required=True, help="Offline label construction only")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
