"""One frozen 170-row action-balance control for recovery demonstrations.

Drop duplicate replay finishes, then repeat existing real Depth/Search decisions.
No observation, action label, GT, or training code is changed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import random


EXPECTED_BEFORE = {"finish": 69, "inspect_regions": 69, "measure_depth": 11,
                   "search_candidates": 21}
EXPECTED_AFTER = {"finish": 46, "inspect_regions": 69, "measure_depth": 22,
                  "search_candidates": 33}


def _kind(row):
    return json.loads(row["target_action"])["action"]


def _drop_replay_finish(row):
    if not row.get("is_final") or _kind(row) != "finish":
        return None
    if row.get("label_source") == "offline_gt_existing_target_protocol_replay":
        return "protocol_replay_finish"
    if ((row.get("replay") or {}).get("kind") == "empty_then_successful_search" and
            row.get("label_source") == "offline_gt_existing_target_after_search"):
        return "empty_replay_finish"
    return None


def rebalance(rows):
    if len(rows) != 170 or dict(Counter(_kind(row) for row in rows)) != EXPECTED_BEFORE:
        raise ValueError("input must be the frozen 170-row recovery decisions and action distribution")
    source_ids = {str(row["id"]) for row in rows}
    kept = []
    removed = Counter()
    removed_ids = {"protocol_replay_finish": [], "empty_replay_finish": []}
    for index, row in enumerate(rows):
        reason = _drop_replay_finish(row)
        if reason:
            removed[reason] += 1
            removed_ids[reason].append(str(row["id"]))
        else:
            kept.append((index, row))
    if removed != Counter({"protocol_replay_finish": 20, "empty_replay_finish": 3}):
        raise ValueError(f"unexpected replay finishes: {dict(removed)}")
    finals = Counter(str(row["id"]) for _, row in kept if row.get("is_final"))
    if set(finals) != source_ids or any(count != 1 for count in finals.values()) or len(source_ids) != 46:
        raise ValueError("every one of the 46 original starts must retain exactly one final decision")

    depth = [(index, row) for index, row in kept if _kind(row) == "measure_depth"]
    search = [(index, row) for index, row in kept if _kind(row) == "search_candidates"]
    if len(depth) != 11 or len(search) != 21:
        raise ValueError("minority action source counts differ from the frozen control")
    rng = random.Random(2026)
    shuffled = search.copy()
    rng.shuffle(shuffled)
    chosen = []
    chosen_ids = set()
    for item in shuffled:
        sample_id = str(item[1]["id"])
        if sample_id not in chosen_ids:
            chosen.append(item)
            chosen_ids.add(sample_id)
            if len(chosen) == 12:
                break
    if len(chosen) < 12:
        chosen_indexes = {index for index, _ in chosen}
        chosen.extend(item for item in shuffled if item[0] not in chosen_indexes)
        chosen = chosen[:12]
    if len(chosen) != 12:
        raise ValueError("cannot select twelve real search decisions")

    output = [deepcopy(row) for _, row in kept]
    for kind, repeated in (("measure_depth", depth), ("search_candidates", chosen)):
        for original_index, row in repeated:
            copy = deepcopy(row)
            copy["resampling"] = {"source_row_index": original_index, "reason": "minority_action_balance",
                                  "action": kind, "seed": 2026}
            output.append(copy)
    actual = dict(Counter(_kind(row) for row in output))
    if len(output) != 170 or actual != EXPECTED_AFTER or {str(row["id"]) for row in output} != source_ids:
        raise ValueError("balanced control lost the frozen row/start counts")
    inventory = {
        "schema_version": "visual-agent-balanced-decisions-v1",
        "input_rows": len(rows), "output_rows": len(output),
        "unique_starts_before": len(source_ids), "unique_starts_after": len(source_ids),
        "removed_finish_counts": dict(removed), "removed_finish_source_ids": removed_ids,
        "repeated_depth_rows": len(depth),
        "repeated_depth_source_ids": [str(row["id"]) for _, row in depth],
        "repeated_search_rows": len(chosen),
        "repeated_search_source_ids": [str(row["id"]) for _, row in chosen],
        "repeated_search_unique_starts": len(chosen_ids),
        "action_counts_before": EXPECTED_BEFORE, "action_counts_after": actual,
        "seed": 2026,
        "interpretation": "Same real labels and 46 starts; repeats change decision weights, not evidence or coverage.",
    }
    return output, inventory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    output, inventory = rebalance(rows)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "train.jsonl").open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    (args.output_dir / "inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n",
                                                    encoding="utf-8")
    print(json.dumps(inventory, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
