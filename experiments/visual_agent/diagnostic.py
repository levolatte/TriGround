"""One GT-free scripted six-observation memory and context diagnostic.

The script forces real tools, measures each hypothetical decision input with the
same Qwen processor, and performs only one final model generation.  Its output
is a capacity diagnostic, never an autonomous-agent prediction or ACC result.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

from PIL import Image

from tools.predict_aux_selection import index_rows_by_id, load_manifest, read_jsonl

from .controller import observation_message, text_message
from .model import InputBudgetExceeded, QwenBackend, geometry_after_processor, prepare_image_messages
from .profiles import PROFILES
from .presentation import compact_json, model_candidates, PROMPT_VERSION
from .run import build_initial_messages, map_row_paths, write_json
from .vision_tools import CandidatePool, VisualTools


def _input_usage(backend, messages, profile, output_cap):
    """Exact processor grids without a Qwen forward pass."""
    prepared, metadata = prepare_image_messages(messages, profile.global_pixels)
    inputs = backend.processor.apply_chat_template(
        prepared, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    grids = inputs["image_grid_thw"].tolist() if "image_grid_thw" in inputs else []
    processor = backend.processor.image_processor
    geometry = geometry_after_processor(metadata, grids, processor.patch_size, processor.merge_size)
    visual = sum(image["visual_tokens"] for image in geometry)
    length = int(inputs["input_ids"].shape[-1])
    return {
        "input_tokens": length, "input_text_tokens": length - visual,
        "visual_tokens": visual, "output_cap": output_cap,
        "image_grid_thw": grids, "image_geometry": geometry,
        "single_visual_within_profile": visual <= profile.visual_tokens,
        "context_within_profile": length + output_cap <= profile.context_tokens,
    }


def _available(row):
    result = []
    for modality, key in (("rgb", "rgb"), ("ir", "ir"), ("depth", "depth_visual")):
        path = row["images"].get(key)
        if path and Path(path).exists():
            result.append(modality)
    return result


def _competitor(pool):
    choices = [candidate for candidate in pool.public_candidates()
               if candidate["id"] != "KEEP"]
    def score(candidate):
        return max((float(source["score"]) for source in candidate.get("sources", [])
                    if source.get("score") is not None), default=-1.0)
    choices.sort(key=lambda candidate: (candidate["role"] != "target", -score(candidate), tuple(candidate["bbox"])))
    return choices[0]["id"] if choices else None


def _scripted_actions(row, candidate_row, pool, tools, available):
    """Yield six slots, with legitimate skips when the sample lacks evidence."""
    competitor = _competitor(pool)
    yield "single_rgb", ({"action": "inspect_regions", "candidate_ids": ["KEEP"],
                          "modalities": ["rgb"], "view": "single"} if "rgb" in available else None), None
    yield "pair_rgb", ({"action": "inspect_regions", "candidate_ids": ["KEEP", competitor],
                        "modalities": ["rgb"], "view": "pair"}
                       if competitor and "rgb" in available else None), "no competing target or RGB" if not competitor or "rgb" not in available else None
    cross = [modality for modality in ("rgb", "ir", "depth") if modality in available]
    yield "cross_modal", ({"action": "inspect_regions", "candidate_ids": ["KEEP"],
                           "modalities": cross, "view": "cross"} if len(cross) >= 2 else None), "fewer than two available modalities" if len(cross) < 2 else None
    depth_ids = ["KEEP", competitor] if competitor else ["KEEP"]
    yield "depth", {"action": "measure_depth", "candidate_ids": depth_ids}, None
    category = candidate_row.get("query_info", {}).get("target_category")
    yield "search_rgb", ({"action": "search_candidates", "category": category,
                          "region": "full", "modality": "rgb", "role": "target"}
                         if category and "rgb" in available else None), "no target category or RGB" if not category or "rgb" not in available else None
    # The sixth slot is decided after the search has really changed the pool.
    current = pool.public_candidates()
    newest = next((candidate["id"] for candidate in reversed(current)
                   if candidate["id"] != "KEEP" and candidate["role"] == "target"
                   and candidate["id"] not in {initial["id"] for initial in candidate_row.get("_public_initial", [])}), None)
    inspected_id = newest or competitor or "KEEP"
    last_modality = "ir" if "ir" in available else "rgb"
    yield "post_search_inspect", ({"action": "inspect_regions", "candidate_ids": [inspected_id],
                                   "modalities": [last_modality], "view": "single"}
                                  if last_modality in available else None), "no visual modality for final inspect" if last_modality not in available else None


def diagnose(row, candidate_row, evidence_row, backend, output_dir, profile,
             *, dino_model=None, sam_model=None, tool_factory=VisualTools):
    """Execute a single scripted sample; backend/tool factory injection is for CPU tests."""
    if profile.name not in {"capacity24", "capacity48"}:
        raise ValueError("scripted full-history diagnostic supports capacity24/48 only")
    if profile.evidence_calls < 6:
        raise ValueError("six-observation diagnostic requires a six-call profile")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"diagnostic output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(row["images"]["rgb"]) as source:
        pool = CandidatePool(candidate_row, source.size)
    available = _available(row)
    if "rgb" not in available:
        raise ValueError("diagnostic requires RGB global image")
    initial = build_initial_messages(row, pool.public_candidates(), output_dir / "images", profile)
    history = copy.deepcopy(initial)
    histories = [copy.deepcopy(history)]
    candidate_stages = [pool.public_candidates()]
    candidate_row = copy.deepcopy(candidate_row)
    candidate_row["_public_initial"] = pool.public_candidates()
    tools = tool_factory(row, pool, output_dir / "images" / "observations", profile.tool_pixels,
                         dino_model=dino_model, sam_model=sam_model,
                         evidence_row=evidence_row, max_searches=profile.search_calls,
                         allow_search=True)
    backend.begin_sample()
    started = time.perf_counter()
    observations = []
    tool_seconds = 0.0
    for slot, action, skip_reason in _scripted_actions(row, candidate_row, pool, tools, available):
        if action is None:
            observations.append({"slot": slot, "status": "SKIPPED", "reason": skip_reason})
            continue
        history.append({"role": "assistant", "content": json.dumps(action, ensure_ascii=False)})
        tool_started = time.perf_counter()
        observation = tools.execute(action)
        tool_seconds += time.perf_counter() - tool_started
        if observation["status"] == "ERROR" and observation.get("data", {}).get("kind") == "protocol":
            raise ValueError(f"diagnostic scripted action is invalid: {action}: {observation['text']}")
        history.append(observation_message(observation, profile.tool_pixels))
        histories.append(copy.deepcopy(history))
        candidate_stages.append(pool.public_candidates())
        observations.append({"slot": slot, "action": action, "observation": observation})

    usages = []
    for index, (messages, candidates) in enumerate(zip(histories, candidate_stages, strict=True)):
        final = index == len(histories) - 1
        cap = 128 if final else profile.action_tokens
        instruction = ("Diagnostic final decision. Only finish with KEEP or a current target ID. "
                       if final else "Diagnostic next action would be selected from the current candidates. ")
        prepared_messages = copy.deepcopy(messages)
        prepared_messages.append(text_message(instruction + compact_json(model_candidates(candidates))))
        usage = _input_usage(backend, prepared_messages, profile, cap)
        usage["decision_index"] = index
        usages.append(usage)

    cumulative_visual = sum(usage["visual_tokens"] for usage in usages)
    stage_limit = any(not usage["single_visual_within_profile"] or not usage["context_within_profile"] for usage in usages)
    cumulative_limit = cumulative_visual > profile.cumulative_visual_tokens
    final_messages = copy.deepcopy(histories[-1])
    final_messages.append(text_message("Diagnostic final decision. Only finish with KEEP or a current target ID. " +
                                       compact_json(model_candidates(candidate_stages[-1]))))
    final = {"status": "LIMIT", "reason": "processor_budget", "raw_output": None}
    if not stage_limit and not cumulative_limit:
        try:
            result = backend.generate(final_messages, 128, profile,
                                      profile.cumulative_visual_tokens - sum(usage["visual_tokens"] for usage in usages[:-1]))
            final = {"status": "GENERATED", "raw_output": result["raw_output"], "usage": result["usage"]}
        except InputBudgetExceeded as error:
            final = {"status": "LIMIT", "reason": str(error), "usage": error.usage, "raw_output": None}
    executed = [entry for entry in observations if entry.get("status") != "SKIPPED"]
    report = {
        "schema_version": "visual-agent-capacity-diagnostic-v1",
        "prompt_version": PROMPT_VERSION,
        "autonomous_agent_result": False, "sample_id": str(row["id"]),
        "profile": profile.to_dict(), "available_modalities": available,
        "coverage": {"planned_slots": 6, "real_tool_calls": len(executed),
                     "all_six_executed": len(executed) == 6,
                     "skipped": [{"slot": item["slot"], "reason": item["reason"]}
                                 for item in observations if item.get("status") == "SKIPPED"],
                     "search_nonempty": any(item.get("slot") == "search_rgb" and
                                            item["observation"]["status"] == "OK" for item in executed)},
        "observations": observations, "decision_inputs": usages,
        "cumulative_visual_tokens_if_each_decision_generated": cumulative_visual,
        "cumulative_visual_within_profile": not cumulative_limit,
        "final_generation": final, "final_candidates": pool.public_candidates(),
        "final_source_candidates": pool.snapshot()["source_candidates"],
        "cost": {"tool_seconds": tool_seconds,
                 "elapsed_seconds": time.perf_counter() - started,
                 "model_load_seconds": getattr(backend, "load_seconds", None),
                 "peak_memory_bytes": backend.peak_memory()},
    }
    write_json(output_dir / "diagnostic.json", report)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--candidate-cache", type=Path, required=True)
    p.add_argument("--evidence-cache", type=Path)
    p.add_argument("--sample-id")
    p.add_argument("--model", required=True)
    p.add_argument("--adapter")
    p.add_argument("--dino-model")
    p.add_argument("--sam-model")
    p.add_argument("--profile", choices=("capacity24", "capacity48"), default="capacity24")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    rows = load_manifest(args.manifest, require_images=True)
    if args.sample_id:
        rows = [row for row in rows if str(row["id"]) == args.sample_id]
    if len(rows) != 1:
        raise ValueError("diagnostic needs exactly one manifest sample; pass --sample-id")
    mappings = [tuple(value.split("=", 1)) for value in args.path_map]
    row = map_row_paths(rows[0], mappings, args.manifest.parent)
    sample_id = str(row["id"])
    candidates = index_rows_by_id(read_jsonl(args.candidate_cache), source=str(args.candidate_cache))
    candidate_row = map_row_paths(candidates[sample_id], mappings, args.candidate_cache.parent)
    evidence_row = None
    if args.evidence_cache:
        evidence = index_rows_by_id(read_jsonl(args.evidence_cache), source=str(args.evidence_cache))
        evidence_row = map_row_paths(evidence[sample_id], mappings, args.evidence_cache.parent)
    if args.dino_model:
        row["dino_model_path"] = args.dino_model
    if args.sam_model:
        row["sam_model_path"] = args.sam_model
    backend = QwenBackend(args.model, args.adapter, device=args.device)
    report = diagnose(row, candidate_row, evidence_row, backend, args.output_dir, PROFILES[args.profile])
    print(json.dumps({"sample_id": sample_id, "coverage": report["coverage"],
                      "final_status": report["final_generation"]["status"],
                      "diagnostic": str((args.output_dir / "diagnostic.json").resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
