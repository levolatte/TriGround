"""GT-free CPU search probes for frozen T training failures.

Each RGB/IR half-scene search starts from the same initial candidate pool.
Only real newly appended targets receive a two-step search -> inspect branch.
These scripted branches are offline teacher material, not Agent successes.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

from PIL import Image

from tools.predict_aux_selection import index_rows_by_id, load_manifest, read_jsonl
from .controller import text_message
from .demonstrate import DECISION_INSTRUCTION
from .export_trajectories import load_frozen_manifest, verify_frozen_splits
from .profiles import PROFILES
from .run import build_initial_messages, map_row_paths, write_json
from .state import build_state_messages
from .vision_tools import CandidatePool, VisualTools


REGIONS = ("left", "right", "top", "bottom")
MODALITIES = ("rgb", "ir")


def collect_one(row, candidate_row, sample_dir, dino_model=None):
    """Execute eight independent real searches, then inspect each new target."""
    semantic = candidate_row["query_info"]
    if semantic["scope"] != "single":
        return [], [], dino_model, "non_single_scope"
    category = semantic["target_category"]
    sample_dir = Path(sample_dir)
    with Image.open(row["images"]["rgb"]) as image:
        rgb_size = image.size
    initial_pool = CandidatePool(candidate_row, rgb_size)
    initial = build_initial_messages(row, initial_pool.public_candidates(), sample_dir / "global", PROFILES["sft"])
    search_events, branches = [], []
    for modality in MODALITIES:
        for region in REGIONS:
            # A new pool and tool instance make sibling attempts independent.
            pool = CandidatePool(candidate_row, rgb_size)
            tools = VisualTools(row, pool, sample_dir / "attempts" / f"{modality}_{region}",
                                PROFILES["sft"].tool_pixels, dino_model=dino_model,
                                max_searches=1, max_new_candidates=4, allow_search=True)
            before = pool.public_candidates()
            action = {"action": "search_candidates", "category": category,
                      "region": region, "modality": modality, "role": "target",
                      "evidence_note": "Probe this scene half for an omitted target category."}
            messages = build_state_messages(initial, [], before)
            messages.append(text_message(DECISION_INSTRUCTION))
            started = time.perf_counter()
            observation = tools.execute(action)
            branch_key = f"{modality}/{region}"
            search_event = {"step": 0, "origin": "scripted_search_branch", "branch_key": branch_key,
                            "messages": messages,
                            "candidates_before": before, "action": action, "executed_action": action,
                            "observation": observation, "candidates_after": pool.public_candidates(),
                            "tool_seconds": time.perf_counter() - started}
            search_events.append(search_event)
            dino_model = tools.dino_model
            if observation["status"] in {"ERROR", "LIMIT"}:
                raise ValueError(f"{row['id']} {modality}/{region}: search failed: {observation}")
            for candidate_id in observation.get("data", {}).get("appended_ids", []):
                candidate_id = str(candidate_id)
                candidate = pool.require_public(candidate_id)
                if candidate["role"] != "target":
                    continue
                inspect_action = {"action": "inspect_regions", "candidate_ids": ["KEEP", candidate_id],
                                  "modalities": ["rgb", "ir"], "view": "cross",
                                  "evidence_note": "Compare the newly found target with KEEP in RGB and IR."}
                inspect_messages = build_state_messages(initial, [search_event], pool.public_candidates())
                inspect_messages.append(text_message(DECISION_INSTRUCTION))
                inspect_started = time.perf_counter()
                inspected = tools.execute(inspect_action)
                if inspected["status"] != "OK" or len(inspected["images"]) != 4:
                    raise ValueError(f"{row['id']} {modality}/{region} {candidate_id}: real cross-view missing")
                inspect_event = {"step": 1, "origin": "scripted_search_branch", "branch_key": branch_key,
                                 "messages": inspect_messages,
                                 "candidates_before": pool.public_candidates(),
                                 "action": inspect_action, "executed_action": inspect_action,
                                 "observation": inspected, "candidates_after": pool.public_candidates(),
                                 "tool_seconds": time.perf_counter() - inspect_started}
                terminal = build_state_messages(initial, [search_event, inspect_event], pool.public_candidates())
                terminal.append(text_message(DECISION_INSTRUCTION))
                branches.append({"id": str(row["id"]), "candidate_id": candidate_id,
                                 "trajectory_origin": "scripted_search_branch",
                                 "branch_key": branch_key, "branch_replay": "offline_only",
                                 "manifest_row": row, "initial_messages": initial,
                                 "events": [search_event, inspect_event],
                                 "search_event": search_event, "inspect_event": inspect_event,
                                 "terminal_messages": terminal, "final_pool_snapshot": pool.snapshot(),
                                 "candidates": pool.public_candidates()})
    return search_events, branches, dino_model, None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failures-manifest", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--holdout-manifest", type=Path, required=True)
    parser.add_argument("--candidate-cache", type=Path, required=True)
    parser.add_argument("--dino-model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    failures = load_manifest(args.failures_manifest, require_images=True)
    train = load_frozen_manifest(args.train_manifest)
    holdout = load_frozen_manifest(args.holdout_manifest)
    verify_frozen_splits(train, holdout, {})
    failure_ids = [str(row["id"]) for row in failures]
    if len(failure_ids) != len(set(failure_ids)) or not set(failure_ids) <= set(train):
        raise ValueError("failure IDs must be unique frozen T training starts; heldout IDs are forbidden")
    for row in failures:
        frozen = train[str(row["id"])]
        if row["query"] != frozen["query"] or row["images"]["rgb"] != frozen["images"]["rgb"]:
            raise ValueError(f"{row['id']}: failure Query/scene differs from frozen training start")
    candidates = index_rows_by_id(read_jsonl(args.candidate_cache), source=str(args.candidate_cache))
    if not set(failure_ids) <= set(candidates):
        raise ValueError("candidate cache misses failure IDs")
    import torch
    torch.set_num_threads(4)
    config = {"failures_manifest": str(args.failures_manifest.resolve()),
              "train_manifest": str(args.train_manifest.resolve()),
              "holdout_manifest": str(args.holdout_manifest.resolve()),
              "candidate_cache": str(args.candidate_cache.resolve()),
              "dino_model": args.dino_model, "path_map": args.path_map,
              "device": "cpu", "origin": "scripted_search_branch", "gt_used": False}
    config_path = args.output_dir / "run_config.json"
    if args.output_dir.exists():
        if not args.resume or not config_path.exists() or json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise FileExistsError(f"search branch output exists or resume config differs: {args.output_dir}")
    else:
        args.output_dir.mkdir(parents=True)
        write_json(config_path, config)
    mappings = [tuple(value.split("=", 1)) for value in args.path_map]
    dino_model = None
    counts = Counter()
    started = time.perf_counter()
    for source_row in failures:
        sample_id = str(source_row["id"])
        sample_dir = args.output_dir / "traces" / sample_id
        branch_file = sample_dir / "branches.jsonl"
        search_file = sample_dir / "search_events.jsonl"
        if branch_file.exists() and search_file.exists():
            counts["resumed_samples"] += 1
            continue
        row = map_row_paths(source_row, mappings, args.failures_manifest.parent)
        row["dino_model_path"] = args.dino_model
        candidate_row = map_row_paths(candidates[sample_id], mappings, args.candidate_cache.parent)
        row["ir_rgb_registration"] = candidate_row.get("ir_rgb_registration")
        search_events, branches, dino_model, skipped = collect_one(row, candidate_row, sample_dir, dino_model)
        sample_dir.mkdir(parents=True, exist_ok=True)
        search_file.write_text("".join(json.dumps(event, ensure_ascii=False) + "\n" for event in search_events),
                               encoding="utf-8")
        branch_file.write_text("".join(json.dumps(branch, ensure_ascii=False) + "\n" for branch in branches),
                               encoding="utf-8")
        counts["processed_samples"] += 1
        counts["search_attempts"] += len(search_events)
        counts["new_target_branches"] += len(branches)
        counts.update("search_" + event["observation"]["status"].lower()
                      for event in search_events)
        if skipped:
            counts["skipped_" + skipped] += 1
        print(json.dumps({"id": sample_id, "attempts": len(search_events),
                          "branches": len(branches)}, ensure_ascii=False), flush=True)
    inventory = {"origin": "scripted_search_branch", "gt_used": False,
                 "frozen_failure_starts": len(failures), **dict(counts)}
    write_json(args.output_dir / "inventory.json", inventory)
    write_json(args.output_dir / "execution.json", {
        "expected": len(failures), "completed": counts["processed_samples"] + counts["resumed_samples"],
        "complete": counts["processed_samples"] + counts["resumed_samples"] == len(failures),
        "device": "cpu", "invocation_wall_seconds": time.perf_counter() - started,
    })
    print(json.dumps(inventory, ensure_ascii=False))


if __name__ == "__main__":
    main()
