"""Offline SFT rows from real scripted recovery probes and frozen T starts.

GT chooses only an already visible legal target branch. ERROR and EMPTY probe
actions are kept as history when replayed, never supervised as target actions.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

from .controller import text_message
from .demonstrate import DECISION_INSTRUCTION, FINAL_INSTRUCTION
from .export_trajectories import _gt_boxes, iou, load_frozen_manifest, verify_frozen_splits
from .profiles import PROFILES
from .state import build_state_messages


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _records(root, filename):
    records = []
    for path in sorted(Path(root).rglob(filename)):
        for record in _jsonl(path):
            # search_events.jsonl stores per-attempt events; its sample ID is
            # encoded by the collector's traces/<id>/ directory.
            records.append(({"id": path.parent.name, **record} if "id" not in record else record, str(path)))
    return records


def _action_text(action):
    return json.dumps({key: value for key, value in action.items() if key != "evidence_note"},
                      ensure_ascii=False, separators=(",", ":"))


def _target(candidates, candidate_id):
    return next((candidate for candidate in candidates
                 if str(candidate["id"]) == str(candidate_id) and candidate["role"] == "target"), None)


def _hint(initial, history, candidates, *, final_only=False):
    messages = build_state_messages(initial, history, candidates)
    messages.append(text_message(FINAL_INSTRUCTION if final_only else DECISION_INSTRUCTION))
    return messages


def _row(sample_id, image_group, messages, action, *, index, origin, label_source,
         source_path, is_final=False, replay=None):
    return {
        "id": sample_id, "decision_index": index, "image_group": image_group,
        "origin": origin, "messages": messages, "target_action": _action_text(action),
        "original_action": None if is_final else action, "original_raw_output": None,
        "label_source": label_source, "online_final_status": "OFFLINE_SCRIPTED_REPLAY",
        "is_final": is_final, "source_path": source_path, "replay": replay,
    }


def _base_rows(path, train):
    rows = _jsonl(path)
    if not rows:
        raise ValueError("base training rows are empty")
    finals = {}
    for row in rows:
        sample_id = str(row["id"])
        if sample_id not in train or row["image_group"] != train[sample_id]["images"]["rgb"]:
            raise ValueError(f"base row outside frozen training scene: {sample_id}")
        if row.get("is_final"):
            action = json.loads(row["target_action"])
            if action.get("action") != "finish" or sample_id in finals:
                raise ValueError(f"base needs one legal final per start: {sample_id}")
            finals[sample_id] = action
    if {str(row["id"]) for row in rows} != set(finals):
        raise ValueError("base contains a start without exactly one final label")
    keep = sum(str(action["candidate_id"]) == "KEEP" for action in finals.values())
    if keep * 2 > len(finals):
        raise ValueError("base unique-start KEEP fraction exceeds 50%")
    for row in rows:
        row["target_action"] = _action_text(json.loads(row["target_action"]))
    return rows, finals


def _validate_scene(sample_id, record, frozen):
    if sample_id not in frozen:
        raise ValueError(f"nontraining branch: {sample_id}")
    manifest = record.get("manifest_row")
    if manifest and (manifest["query"] != frozen[sample_id]["query"] or
                     manifest["images"]["rgb"] != frozen[sample_id]["images"]["rgb"]):
        raise ValueError(f"branch Query/scene differs from frozen train: {sample_id}")


def _valid_search_branch(branch):
    search, inspect = branch["search_event"], branch["inspect_event"]
    candidate_id = str(branch["candidate_id"])
    return (branch.get("trajectory_origin") == "scripted_search_branch"
            and search["action"]["action"] == "search_candidates"
            and search["observation"]["status"] == "OK"
            and candidate_id in {str(value) for value in search["observation"]["data"]["appended_ids"]}
            and inspect["action"]["action"] == "inspect_regions"
            and candidate_id in {str(value) for value in inspect["action"]["candidate_ids"]}
            and inspect["observation"]["status"] == "OK"
            and bool(inspect["observation"]["images"])
            and search["candidates_after"] == inspect["candidates_before"] == branch["candidates"]
            and _target(branch["candidates"], candidate_id) is not None)


def _search_rows(branch, source, frozen, *, empty_event=None, empty_source=None):
    sample_id = str(branch["id"])
    image_group = frozen[sample_id]["images"]["rgb"]
    search = deepcopy(branch["search_event"])
    inspect = deepcopy(branch["inspect_event"])
    initial = branch["initial_messages"]
    replay = None
    if empty_event is None:
        search_messages = search["messages"]
        inspect_messages = inspect["messages"]
        final_messages = branch["terminal_messages"]
        origin = "scripted_search_branch"
        prefix = []
    else:
        if (empty_event["observation"]["status"] != "EMPTY" or
                empty_event["action"]["action"] != "search_candidates" or
                empty_event["candidates_before"] != empty_event["candidates_after"] or
                empty_event["candidates_after"] != search["candidates_before"] or
                _action_text(empty_event["action"]) == _action_text(search["action"])):
            raise ValueError(f"{sample_id}: EMPTY probe cannot precede selected search without a pool change")
        prefix = [deepcopy(empty_event)]
        search["step"] = 1
        inspect["step"] = 2
        search_messages = _hint(initial, prefix, search["candidates_before"])
        inspect_messages = _hint(initial, [*prefix, search], inspect["candidates_before"])
        final_messages = _hint(initial, [*prefix, search, inspect], branch["candidates"])
        origin = "scripted_off_policy_replay"
        replay = {"kind": "empty_then_successful_search", "empty_source_path": empty_source,
                  "empty_action": empty_event["action"], "empty_status": "EMPTY"}
    final = {"action": "finish", "candidate_id": str(branch["candidate_id"])}
    rows = [
        _row(sample_id, image_group, search_messages, search["action"], index=len(prefix),
             origin=origin, label_source="offline_selected_actual_search", source_path=source, replay=replay),
        _row(sample_id, image_group, inspect_messages, inspect["action"], index=len(prefix) + 1,
             origin=origin, label_source="offline_selected_actual_inspect", source_path=source, replay=replay),
        _row(sample_id, image_group, final_messages, final, index=len(prefix) + 2,
             origin=origin, label_source="offline_gt_existing_target_after_search", source_path=source,
             is_final=True, replay=replay),
    ]
    for row in rows:
        row["source_branch_key"] = branch["branch_key"]
        row["source_candidate_id"] = str(branch["candidate_id"])
    return rows


def _protocol_rows(record, source, branch, branch_source, frozen):
    sample_id = str(record["id"])
    error = record.get("event")
    prefix = record.get("prefix_events") or []
    inspect = branch["event"]
    candidate_id = str(branch["candidate_id"])
    if (record.get("status") != "ERROR" or not record.get("intentional_invalid_action") or
            not error or not prefix or prefix[-1] != error or
            error["observation"]["status"] != "ERROR" or
            error["observation"].get("data", {}).get("kind") != "protocol" or
            error["action"].get("action") != "inspect_regions" or
            error["action"].get("view") != "cross" or
            len(error["action"].get("candidate_ids", [])) != 3 or
            record["pool_before"] != record["pool_after"] or
            error["candidates_before"] != error["candidates_after"] or
            error["candidates_after"] != record["pool_after"] or
            record["pool_after"] != inspect["candidates_before"] or
            inspect["action"]["action"] != "inspect_regions" or
            candidate_id not in {str(value) for value in inspect["action"]["candidate_ids"]} or
            inspect["observation"]["status"] != "OK" or
            not inspect["observation"]["images"] or
            len(prefix) + 2 > PROFILES["sft"].evidence_calls + 1):
        raise ValueError(f"{sample_id}: incompatible protocol recovery state")
    if _target(branch["candidates"], candidate_id) is None:
        raise ValueError(f"{sample_id}: protocol branch target is absent")
    if len(prefix) - 1 != len(branch["prefix_events"]):
        raise ValueError(f"{sample_id}: protocol and branch prefix lengths differ")
    for actual, expected in zip(prefix[:-1], branch["prefix_events"], strict=True):
        if (actual["action"] != expected["action"] or
                actual["observation"] != expected["observation"] or
                actual["candidates_after"] != expected["candidates_after"]):
            raise ValueError(f"{sample_id}: protocol and branch prefix differ")
    inspect = deepcopy(inspect)
    inspect["step"] = len(prefix)
    initial = record["initial_messages"] if "initial_messages" in record else branch.get("initial_messages")
    if initial is None:
        initial = json.loads(Path(branch["parent_trace_path"]).read_text(encoding="utf-8"))["initial_messages"]
    messages = _hint(initial, prefix, inspect["candidates_before"])
    terminal = _hint(initial, [*prefix, inspect], branch["candidates"],
                     final_only=len(prefix) + 1 >= PROFILES["sft"].evidence_calls)
    image_group = frozen[sample_id]["images"]["rgb"]
    replay = {"kind": "protocol_error_then_actual_inspect", "error_source_path": source,
              "branch_source_path": branch_source, "error_status": "ERROR"}
    rows = [
        _row(sample_id, image_group, messages, inspect["action"], index=len(prefix),
             origin="scripted_off_policy_replay", label_source="offline_protocol_replay_actual_inspect",
             source_path=branch_source, replay=replay),
        _row(sample_id, image_group, terminal, {"action": "finish", "candidate_id": candidate_id},
             index=len(prefix) + 1, origin="scripted_off_policy_replay",
             label_source="offline_gt_existing_target_protocol_replay", source_path=branch_source,
             is_final=True, replay=replay),
    ]
    for row in rows:
        row["source_candidate_id"] = candidate_id
    return rows


def export(args):
    train = load_frozen_manifest(args.train_manifest)
    holdout = load_frozen_manifest(args.holdout_manifest)
    debug = load_frozen_manifest(args.debug_manifest)
    verify_frozen_splits(train, holdout, debug)
    boxes = _gt_boxes(args.gt)
    base, base_finals = _base_rows(args.base, train)
    for root in (args.candidate_branches, args.protocol_feedback, args.search_branches):
        if not Path(root).is_dir():
            raise ValueError(f"branch input directory missing: {root}")
    candidate_branches = _records(args.candidate_branches, "branches.jsonl")
    protocol_records = _records(args.protocol_feedback, "protocol_feedback.jsonl")
    search_branches = _records(args.search_branches, "branches.jsonl")
    search_events = _records(args.search_branches, "search_events.jsonl")
    for record, _ in [*candidate_branches, *protocol_records, *search_branches, *search_events]:
        _validate_scene(str(record["id"]), record, train)
    by_candidate = {}
    for branch, path in candidate_branches:
        by_candidate.setdefault((str(branch["id"]), str(branch["candidate_id"])), []).append((branch, path))
    by_protocol = {}
    for record, path in protocol_records:
        sample_id = str(record["id"])
        if sample_id in by_protocol:
            raise ValueError(f"duplicate protocol record: {sample_id}")
        by_protocol[sample_id] = (record, path)
    additions = []
    inventory = Counter()
    rejected = []
    protocol_ids = set()
    for sample_id, final in sorted(base_finals.items()):
        candidate_id = str(final["candidate_id"])
        if candidate_id == "KEEP" or sample_id not in by_protocol:
            continue
        record, source = by_protocol[sample_id]
        if record.get("status") == "SKIP":
            rejected.append({"id": sample_id, "kind": "protocol", "reason": "probe_skipped"})
            continue
        choices = by_candidate.get((sample_id, candidate_id), [])
        if not choices:
            rejected.append({"id": sample_id, "kind": "protocol", "reason": "no_real_candidate_branch"})
            continue
        branch, branch_source = choices[0]
        if (_target(branch["candidates"], candidate_id) is None or
                iou(_target(branch["candidates"], candidate_id)["bbox"], boxes[sample_id]) < 0.5):
            rejected.append({"id": sample_id, "kind": "protocol", "reason": "selected_target_not_gt_supported"})
            continue
        try:
            recovery_rows = _protocol_rows(record, source, branch, branch_source, train)
        except ValueError as error:
            rejected.append({"id": sample_id, "kind": "protocol", "reason": str(error)})
            continue
        additions.extend(recovery_rows)
        protocol_ids.add(sample_id)
        inventory["protocol_replay_rows"] += 2
    by_search = {}
    for branch, source in search_branches:
        sample_id = str(branch["id"])
        if not _valid_search_branch(branch):
            rejected.append({"id": sample_id, "kind": "search", "reason": "invalid_real_search_branch",
                             "source_path": source})
            continue
        candidate = _target(branch["candidates"], branch["candidate_id"])
        overlap = iou(candidate["bbox"], boxes[sample_id])
        if overlap >= 0.5:
            by_search.setdefault(sample_id, []).append((overlap, branch, source))
    empty_by_id = {}
    for event, source in search_events:
        sample_id = str(event["id"])
        if (event["observation"]["status"] == "EMPTY" and
                event["candidates_before"] == event["candidates_after"]):
            empty_by_id.setdefault(sample_id, []).append((event, source))
    search_ids = set()
    selected_search = []
    for sample_id, choices in sorted(by_search.items()):
        choices.sort(key=lambda item: (-item[0], item[1]["branch_key"], str(item[1]["candidate_id"])))
        overlap, branch, source = choices[0]
        search_action = branch["search_event"]["action"]
        selected_search.append({
            "id": sample_id, "query": train[sample_id]["query"],
            "category": search_action["category"], "modality": search_action["modality"],
            "region": search_action["region"], "candidate_id": str(branch["candidate_id"]),
            "candidate_iou_for_offline_selection": overlap, "source_path": source,
        })
        additions.extend(_search_rows(branch, source, train))
        search_ids.add(sample_id)
        inventory["search_branch_rows"] += 3
        for empty, empty_source in empty_by_id.get(sample_id, []):
            try:
                replay = _search_rows(branch, source, train, empty_event=empty["event"] if "event" in empty else empty,
                                      empty_source=empty_source)
            except ValueError as error:
                rejected.append({"id": sample_id, "kind": "empty_replay", "reason": str(error),
                                 "source_path": empty_source})
                continue
            additions.extend(replay)
            inventory["empty_replay_rows"] += 3
            inventory["empty_replays"] += 1
            break
    if not additions:
        raise ValueError("no valid real recovery/search additions; inspect rejected branches")
    all_rows = [*base, *additions]
    unique_ids = set(base_finals) | search_ids
    keep_ids = {sample_id for sample_id, final in base_finals.items()
                if str(final["candidate_id"]) == "KEEP"}
    if len(keep_ids) * 2 > len(unique_ids):
        raise ValueError("unique-start KEEP fraction exceeds 50% after additions")
    result = {
        "schema_version": "visual-agent-recovery-demonstrations-v1",
        "base_unique_starts": len(base_finals), "base_rows": len(base),
        "base_nonkeep_starts": sum(str(final["candidate_id"]) != "KEEP" for final in base_finals.values()),
        "input_candidate_branches": len(candidate_branches),
        "input_protocol_records": len(protocol_records),
        "input_search_branches": len(search_branches),
        "input_search_events": len(search_events),
        "search_branch_starts": len(search_ids),
        "new_search_unique_starts": len(search_ids - set(base_finals)),
        "search_branch_replays_on_base": len(search_ids & set(base_finals)),
        "all_unique_starts": len(unique_ids), "unique_keep_starts": len(keep_ids),
        "unique_keep_fraction": len(keep_ids) / len(unique_ids),
        "protocol_replay_starts": len(protocol_ids), "empty_replays": inventory["empty_replays"],
        "extra_replays": len(protocol_ids) + inventory["empty_replays"] + len(search_ids & set(base_finals)),
        "protocol_replay_rows": inventory["protocol_replay_rows"],
        "search_branch_rows": inventory["search_branch_rows"],
        "empty_replay_rows": inventory["empty_replay_rows"],
        "total_rows": len(all_rows), "rejected": rejected,
        "selected_search_branches_for_semantic_review": selected_search,
        "label_format": "minimal_action_json_without_optional_evidence_note",
        "action_counts": dict(Counter(json.loads(row["target_action"])["action"] for row in all_rows)),
        "provenance": "Actual tool observations only. Protocol ERROR and search EMPTY are unsupervised context in explicitly offline replays; GT chooses only extant target IDs.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "train.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    (args.output_dir / "inventory.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                                    encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="aligned base train.jsonl")
    parser.add_argument("--candidate-branches", type=Path, required=True)
    parser.add_argument("--protocol-feedback", type=Path, required=True)
    parser.add_argument("--search-branches", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--holdout-manifest", type=Path, required=True)
    parser.add_argument("--debug-manifest", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(export(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
