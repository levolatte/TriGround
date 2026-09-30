"""Export real-query final choices and fixed cross-modal observation lessons.

The paired-observation first step is a prescribed operation warmup, not learned
planning. Its actual CPU tool result enters the follow-up input. Final-choice
labels use GT only offline; the online messages keep the original query,
candidate proposals, and real rendered images.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import time

from PIL import Image

from .evaluate import iou
from .object_controller import (
    FINAL_INSTRUCTION,
    FINISH_SCHEMA,
    TOOL_SCHEMAS,
    _assistant_tool_message,
    _remember_observation_images,
    _tool_message,
    build_initial_messages,
    build_messages,
)
from .object_tools import ObjectTools
from .prepare_object_assets import public_rows, read_rows
from .presentation import compact_json
from .vision_tools import CandidatePool


def _load_labels(path: Path) -> dict[str, dict]:
    if path.suffix.lower() == ".jsonl":
        return {str(row["id"]): row for row in read_rows(path)}
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(value, list):
        return {str(row["id"]): row for row in value}
    return {str(key): item for key, item in value.items()}


def _action(name: str, arguments: dict) -> str:
    return "<tool_call>" + compact_json({"name": name, "arguments": arguments}) + "</tool_call>"


def _decision(sample_id: str, image_group: str, messages: list[dict], action: str,
              category: str, decision_index: int, *,
              tool_schemas: list[dict] | None = None, **extra) -> dict:
    return {
        "id": sample_id,
        "image_group": image_group,
        "decision_index": decision_index,
        "messages": messages,
        "tools": tool_schemas or TOOL_SCHEMAS,
        "target_action": action,
        "sample_category": category,
        **extra,
    }


def _candidate_bbox_for_target(candidate_row: dict, gt: list[float]) -> tuple[dict | None, bool, float]:
    baseline = next(item for item in candidate_row["candidates"] if item.get("is_baseline"))
    baseline_iou = iou(baseline["bbox"], gt)
    if baseline_iou >= 0.5:
        # Preserve a correct C prediction even when another proposal has a
        # slightly larger offline IoU.
        return baseline, True, baseline_iou

    eligible = []
    for item in candidate_row["candidates"]:
        frame = item.get("coordinate_frame")
        if frame is None:
            modalities = {source.get("modality") for source in item.get("sources", [])}
            frame = "ir" if not item.get("is_baseline") and modalities == {"ir"} else "rgb"
        if frame != "rgb":
            continue
        score = iou(item["bbox"], gt)
        if score >= 0.5:
            eligible.append((score, -int(item["id"]), item))
    if not eligible:
        return None, False, baseline_iou
    best = max(eligible, key=lambda item: (item[0], item[1]))
    return best[2], False, baseline_iou


def _tool_for_row(row: dict, candidate_row: dict, output_dir: Path, seed: int):
    with Image.open(row["images"]["rgb"]) as image:
        pool = CandidatePool(candidate_row, image.size)
    tool = ObjectTools(row, pool, output_dir, seed=seed)
    public = tool.public_candidates()
    atlas = tool.atlas(modalities=["rgb"])
    if atlas["status"] != "OK" or not atlas.get("images"):
        raise RuntimeError(f"{row['id']}: RGB atlas generation failed: {atlas['status']}")
    initial = build_initial_messages(row, public, atlas)
    return tool, public, atlas, initial


def _frame(item: dict) -> str:
    frame = item.get("coordinate_frame")
    if frame:
        return str(frame)
    modalities = {source.get("modality") for source in item.get("sources", [])}
    return "ir" if not item.get("is_baseline") and modalities == {"ir"} else "rgb"


def _top_scored_rgb_competitor(candidate_row: dict) -> tuple[dict | None, float | None]:
    baseline = next(item for item in candidate_row["candidates"] if item.get("is_baseline"))
    ranked = []
    for item in candidate_row["candidates"]:
        if item.get("is_baseline") or _frame(item) != "rgb":
            continue
        scores = [float(source["score"]) for source in item.get("sources", [])
                  if source.get("score") is not None and math.isfinite(float(source["score"]))]
        if scores:
            ranked.append((max(scores), -int(item["id"]), item))
    if not ranked:
        return None, None
    score, _, item = max(ranked, key=lambda value: (value[0], value[1]))
    return item, score


def _paired_evidence_decisions(row: dict, candidate_row: dict, target_raw: dict,
                               initial_iou: float, c_hit: bool, output_dir: Path,
                               seed: int) -> tuple[list[dict], dict]:
    competitor_raw, competitor_score = _top_scored_rgb_competitor(candidate_row)
    if competitor_raw is None:
        raise ValueError(f"{row['id']}: paired sample has no scored non-C RGB candidate")
    tool, public, atlas, initial = _tool_for_row(row, candidate_row, output_dir, seed)
    c_id = tool._public("KEEP")
    competitor_id = tool._public(competitor_raw["id"])
    target_id = tool._public("KEEP" if target_raw.get("is_baseline") else target_raw["id"])
    legal = {item["id"] for item in public if item.get("finish_eligible")}
    if c_id not in legal or competitor_id not in legal or target_id not in legal:
        raise ValueError(f"{row['id']}: C, competitor, and GT-labeled candidate must all be legal RGB IDs")
    modes = ["rgb"]
    ir_path = row.get("images", {}).get("ir")
    if ir_path and Path(ir_path).is_file():
        modes.append("ir")
    ids = [c_id, competitor_id]
    inspect_arguments = {"ids": ids, "modalities": modes}
    group = str(row.get("image_group", row.get("scene_id", row["images"]["rgb"])))
    first_instruction = (
        f"To ground the original query, call inspect to compare {c_id} and {competitor_id} "
        f"using {', '.join(modes).upper()}."
    )
    first_messages = [*initial, {"role": "user", "content": [{"type": "text", "text": first_instruction}]}]
    first = _decision(
        f"paired_evidence:{row['id']}", group, first_messages,
        _action("inspect", inspect_arguments), "prescribed_operation_warmup", 0,
        source_sample_id=str(row["id"]), inspected_candidate_ids=ids,
        c_candidate_id=c_id, competitor_candidate_id=competitor_id,
        competitor_score=round(float(competitor_score), 7),
        pair_selection_rule="highest finite source detection score among non-C RGB-frame candidates",
        requested_modalities=modes, label_source="prescribed_tool_operation",
        autonomous_tool_choice_claimed=False, actual_initial_atlas=True,
    )

    observation = tool.execute("inspect", inspect_arguments)
    if observation["status"] not in {"OK", "UNKNOWN"} or not observation.get("images"):
        raise RuntimeError(f"{row['id']}: actual paired inspect failed: {observation['status']} {observation['text']}")
    assistant = _assistant_tool_message(
        "object_call_1", {"name": "inspect", "arguments": inspect_arguments}, "")
    tool_message = _tool_message("object_call_1", "inspect", observation, row)
    retained, _ = _remember_observation_images(
        observation, 0, {}, 0, 602112, active_candidate_ids=ids)
    turns = [{"assistant_message": assistant, "tool_message": tool_message}]
    final_instruction = (
        "Based on the original query and the actual evidence, choose the final RGB candidate and call finish."
    )
    second_messages = build_messages(initial, turns, public, [], retained, final_instruction)
    second = _decision(
        f"paired_evidence:{row['id']}", group, second_messages,
        _action("finish", {"id": target_id}), "gt_supervised_final_choice_after_prescribed_observation", 1,
        tool_schemas=[FINISH_SCHEMA],
        source_sample_id=str(row["id"]), selected_candidate_id=target_id,
        selected_is_c=bool(c_hit), initial_c_iou=round(initial_iou, 6),
        label_source="offline_gt_candidate_coverage_with_c_hit_priority",
        tool_result_status=observation["status"], requested_modalities=modes,
        actual_tool_images_by_modality=dict(Counter(image.get("modality", "unknown")
                                                    for image in observation["images"])),
        evidence_images_added_to_messages=sum(
            1 for message in second_messages if isinstance(message.get("content"), list)
            for block in message["content"] if isinstance(block, dict) and block.get("type") == "image"
            and block.get("view") == "tool"),
        autonomous_tool_choice_claimed=False, visual_causality_claimed=False,
    )
    return [first, second], {
        "status": observation["status"],
        "images_by_modality": dict(Counter(image.get("modality", "unknown")
                                            for image in observation["images"])),
    }


def _final_choice_decision(row: dict, candidate_row: dict, target_raw: dict, initial_iou: float,
                           c_hit: bool, output_dir: Path, seed: int) -> tuple[dict, str]:
    tool, public, atlas, initial = _tool_for_row(row, candidate_row, output_dir, seed)
    target_id = tool._public("KEEP" if target_raw.get("is_baseline") else target_raw["id"])
    chosen = next(item for item in public if item["id"] == target_id)
    if not chosen.get("finish_eligible"):
        raise ValueError(f"{row['id']}: offline target label points to a non-finish-eligible box")
    group = str(row.get("image_group", row.get("scene_id", row["images"]["rgb"])))
    # Ground truth is used by the caller only to derive the label and audit
    # selection; it is deliberately absent from this input and output row.
    decision = _decision(
        f"final_choice:{row['id']}", group,
        [*initial, {"role": "user", "content": [{"type": "text", "text": FINAL_INSTRUCTION}]}],
        _action("finish", {"id": target_id}),
        "offline_gt_final_candidate_selection", 0,
        tool_schemas=[FINISH_SCHEMA],
        source_sample_id=str(row["id"]), selected_candidate_id=target_id,
        selected_is_c=bool(c_hit), initial_c_iou=round(initial_iou, 6),
        label_source="offline_gt_candidate_coverage_with_c_hit_priority",
        actual_initial_atlas=True,
    )
    return decision, target_id


def _teacher_jobs(rows: list[dict], candidates: dict[str, dict], candidate_cache_path: Path,
                  count_per_bucket: int, seed: int) -> list[dict]:
    rng = random.Random(seed + 41)
    buckets: dict[str, list[tuple[int, dict]]] = {
        "depth_relation_tool_choice": [],
        "registered_rgb_ir_evidence": [],
        "candidate_source_ambiguity": [],
    }
    for row in rows:
        sample_id = str(row["id"])
        candidate = candidates.get(sample_id)
        if not candidate:
            continue
        items = candidate.get("candidates", [])
        qi = candidate.get("query_info", {})
        modes = {str(src.get("modality")) for item in items for src in item.get("sources", [])}
        if (row.get("images", {}).get("depth_raw") or row.get("images", {}).get("depth_visual")) and len(items) >= 2 and qi.get("relation_type") not in (None, "none"):
            buckets["depth_relation_tool_choice"].append((len(items), row))
        if (row.get("ir_rgb_registration") == "normalized_shared_frame" and "ir" in row.get("images", {})
                and "ir" in modes and "rgb" in modes and len(items) >= 2):
            buckets["registered_rgb_ir_evidence"].append((len(items), row))
        if len(modes) >= 2 and len(items) >= 3:
            buckets["candidate_source_ambiguity"].append((len(modes) * 100 + len(items), row))

    selected, seen = [], set()
    for bucket, pool in buckets.items():
        rng.shuffle(pool)
        pool.sort(key=lambda item: item[0], reverse=True)
        added = 0
        for _, row in pool:
            sample_id = str(row["id"])
            if sample_id in seen:
                continue
            seen.add(sample_id)
            selected.append((bucket, row))
            added += 1
            if added == count_per_bucket:
                break

    jobs = []
    for index, (bucket, row) in enumerate(selected, 1):
        jobs.append({
            "job_id": f"blind_tool_causality_{index:03d}",
            "sample_id": str(row["id"]),
            "image_group": row.get("image_group", row.get("scene_id")),
            "selection_bucket": bucket,
            "query": row["query"],
            "images": row["images"],
            "candidate_cache_path": str(candidate_cache_path),
            "candidate_count": len(candidates[str(row["id"])].get("candidates", [])),
            "teacher_policy": (
                "Blind to offline GT and all labels. Read the real query, images, and candidate boxes; "
                "choose finish or one useful real inspect/depth/search call. If choosing a tool, state "
                "which observable fact it can return and how that fact could change the target choice. "
                "Do not infer target/reference identity, same-object identity, or image registration from IDs or GT."
            ),
            "label_status": "pending_blind_luna_teacher",
            "counts_as_training_sample": False,
        })
    return jobs


def export(args) -> dict:
    rows = public_rows(args.manifest)
    manifest_raw = {str(row["id"]): row for row in read_rows(args.manifest)}
    for row in rows:
        source = manifest_raw[str(row["id"])]
        for key in ("image_group", "scene_id", "source", "ir_rgb_registration",
                    "registration_source", "depth_encoding", "depth_visual_encoding"):
            if key in source:
                row[key] = source[key]

    candidate_rows = {str(row["id"]): row for row in read_rows(args.candidate_cache)}
    labels = _load_labels(args.labels)
    rows_by_id = {str(row["id"]): row for row in rows}

    eligible_final, final_skips = [], Counter()
    for sample_id, candidate_row in candidate_rows.items():
        row = rows_by_id.get(sample_id)
        label = labels.get(sample_id)
        if row is None or label is None:
            final_skips["missing_manifest_or_label"] += 1
            continue
        gt = label.get("bbox", label.get("bbox_xyxy_normalized"))
        if gt is None:
            final_skips["missing_gt_bbox"] += 1
            continue
        target, c_hit, initial_iou = _candidate_bbox_for_target(candidate_row, gt)
        if target is None:
            final_skips["no_gt_covered_finish_eligible_target_candidate"] += 1
            continue
        eligible_final.append((row, candidate_row, target, initial_iou, c_hit))

    rng = random.Random(args.seed)
    c_hits = [item for item in eligible_final if item[4]]
    c_misses = [item for item in eligible_final if not item[4]]
    rng.shuffle(c_hits)
    rng.shuffle(c_misses)
    selected_final = c_misses[:args.final_limit]
    remaining = max(0, args.final_limit - len(selected_final))
    selected_final.extend(c_hits[:remaining])
    rng.shuffle(selected_final)
    used_ids = {str(item[0]["id"]) for item in selected_final}

    paired_pool, paired_skips = [], Counter()
    for sample_id, candidate_row in candidate_rows.items():
        if sample_id in used_ids or sample_id not in rows_by_id:
            continue
        label = labels.get(sample_id)
        if label is None:
            paired_skips["missing_label"] += 1
            continue
        gt = label.get("bbox", label.get("bbox_xyxy_normalized"))
        if gt is None:
            paired_skips["missing_gt_bbox"] += 1
            continue
        target, c_hit, initial_iou = _candidate_bbox_for_target(candidate_row, gt)
        if target is None:
            paired_skips["no_gt_covered_legal_rgb_candidate"] += 1
            continue
        competitor, score = _top_scored_rgb_competitor(candidate_row)
        if competitor is None:
            paired_skips["no_scored_non_c_rgb_candidate"] += 1
            continue
        paired_pool.append((rows_by_id[sample_id], candidate_row, target, initial_iou, c_hit, score))
    rng.shuffle(paired_pool)
    selected_pairs = paired_pool[:args.paired_limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "foundation_decisions.jsonl"
    if output.exists():
        raise FileExistsError(output)

    decisions, tool_statuses, observed_images = [], Counter(), Counter()
    started = time.monotonic()
    for index, (row, candidate_row, target, initial_iou, c_hit, _) in enumerate(selected_pairs):
        pair, actual = _paired_evidence_decisions(
            row, candidate_row, target, initial_iou, c_hit,
            args.output_dir / "observations" / "paired" / str(row["id"]),
            args.seed + index)
        decisions.extend(pair)
        tool_statuses[actual["status"]] += 1
        observed_images.update(actual["images_by_modality"])

    for index, item in enumerate(selected_final):
        row, candidate_row, target, initial_iou, c_hit = item
        decision, target_id = _final_choice_decision(
            row, candidate_row, target, initial_iou, c_hit,
            args.output_dir / "observations" / "final" / str(row["id"]), args.seed + 100000 + index)
        decisions.append(decision)

    with output.open("w", encoding="utf-8") as handle:
        for row in decisions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    teacher_jobs = _teacher_jobs(rows, candidate_rows, args.candidate_cache,
                                 args.teacher_per_bucket, args.seed)
    teacher_path = args.output_dir / "blind_teacher_needed.jsonl"
    with teacher_path.open("w", encoding="utf-8") as handle:
        for job in teacher_jobs:
            handle.write(json.dumps(job, ensure_ascii=False) + "\n")

    final_rows = [row for row in decisions if row["sample_category"] == "offline_gt_final_candidate_selection"]
    static_final_rows = [row for row in final_rows if row["sample_category"] == "offline_gt_final_candidate_selection"]
    observed_final_rows = [row for row in decisions if row["sample_category"] == "gt_supervised_final_choice_after_prescribed_observation"]
    summary = {
        "schema": "visual_agent_capability_foundation_v1",
        "exported_decision_rows": len(decisions),
        "unique_training_start_ids": len({str(row["source_sample_id"]) for row in decisions}),
        "unique_image_groups": len({str(row["image_group"]) for row in decisions}),
        "decision_rows_by_category": dict(Counter(row["sample_category"] for row in decisions)),
        "prescribed_observation_starts": len(selected_pairs),
        "static_offline_final_choice_rows": len(static_final_rows),
        "post_observation_offline_final_choice_rows": len(observed_final_rows),
        "offline_final_choice_c_hit": sum(bool(row["selected_is_c"]) for row in [*final_rows, *observed_final_rows]),
        "offline_final_choice_c_miss_corrected_by_candidate": sum(not bool(row["selected_is_c"]) for row in [*final_rows, *observed_final_rows]),
        "offline_final_choice_policy": "When C IoU>=0.5, label the C baseline candidate. Otherwise label the highest-IoU RGB target candidate with IoU>=0.5. No candidate is created from GT.",
        "prescribed_observation_policy": "Original Query stays in input. C and the highest finite-score non-C RGB-frame proposal are selected with a GT-free rule; paired RGB+IR inspect is actually executed on CPU. The first action is prescribed_operation_warmup, not an autonomous tool-choice label. Follow-up finish is offline-GT supervision over any current finish-eligible RGB box, preserving C when IoU>=0.5. No tool-benefit or visual-causality claim.",
        "actual_cpu_inspect_results": dict(tool_statuses),
        "actual_tool_images_by_modality": dict(observed_images),
        "actual_tool_result_images_added_to_followup_messages": sum(row.get("evidence_images_added_to_messages", 0) for row in observed_final_rows),
        "blind_teacher_needed_jobs": len(teacher_jobs),
        "blind_teacher_jobs_counted_as_samples": 0,
        "blind_teacher_job_buckets": dict(Counter(job["selection_bucket"] for job in teacher_jobs)),
        "candidate_final_choice_eligible_total": len(eligible_final),
        "candidate_final_choice_skip_counts": dict(final_skips),
        "available_paired_observation_pool_before_limit": len(paired_pool),
        "paired_observation_skip_counts": dict(paired_skips),
        "elapsed_seconds_cpu_export": round(time.monotonic() - started, 3),
        "files": {"training_decisions": str(output), "pending_blind_teacher_jobs": str(teacher_path)},
        "reproduction_command": (
            "python -m experiments.visual_agent.export_capability_foundation "
            f"--manifest {args.manifest} --candidate-cache {args.candidate_cache} "
            f"--labels {args.labels} --output-dir {args.output_dir} "
            f"--final-limit {args.final_limit} --paired-limit {args.paired_limit} "
            f"--teacher-per-bucket {args.teacher_per_bucket} --seed {args.seed}"
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def rebalance_existing(args) -> dict:
    """Move up to 150 C-miss starts from the static pool into observed states.

    Reuses the first export's 900 static atlases and 150 retained C-hit
    observation trajectories; only the moved C-miss observations and same-size
    replacement C-hit static set need new CPU renders.
    """
    output = args.output_dir / "foundation_decisions.jsonl"
    decisions = read_rows(output)
    static = [row for row in decisions if row["sample_category"] == "offline_gt_final_candidate_selection"]
    pair_rows = [row for row in decisions if row["sample_category"] in {
        "prescribed_operation_warmup", "gt_supervised_final_choice_after_prescribed_observation"}]
    pairs_by_id: dict[str, list[dict]] = defaultdict(list)
    for row in pair_rows:
        pairs_by_id[str(row["source_sample_id"])].append(row)

    manifest_raw = {str(row["id"]): row for row in read_rows(args.manifest)}
    rows_by_id = {str(row["id"]): row for row in public_rows(args.manifest)}
    for sample_id, row in rows_by_id.items():
        source = manifest_raw[sample_id]
        for key in ("image_group", "scene_id", "source", "ir_rgb_registration",
                    "registration_source", "depth_encoding", "depth_visual_encoding"):
            if key in source:
                row[key] = source[key]
    candidate_rows = {str(row["id"]): row for row in read_rows(args.candidate_cache)}
    labels = _load_labels(args.labels)

    rng = random.Random(args.seed + 97)
    eligible_misses = []
    for row in static:
        if row["selected_is_c"]:
            continue
        sample_id = str(row["source_sample_id"])
        candidate_row = candidate_rows[sample_id]
        if _top_scored_rgb_competitor(candidate_row)[0] is not None:
            eligible_misses.append(row)
    rng.shuffle(eligible_misses)
    moved_misses = eligible_misses[:min(args.rebalance_misses, args.paired_limit)]
    moved_ids = {str(row["source_sample_id"]) for row in moved_misses}

    old_pair_hit_ids = [sample_id for sample_id, rows in pairs_by_id.items()
                        if len(rows) == 2 and any(row.get("selected_is_c") is True for row in rows)]
    rng.shuffle(old_pair_hit_ids)
    keep_old_pair_ids = set(old_pair_hit_ids[:args.paired_limit - len(moved_ids)])
    if len(keep_old_pair_ids) != args.paired_limit - len(moved_ids):
        raise ValueError("not enough existing C-hit paired observations to balance the requested batch")

    already_used = {str(row["source_sample_id"]) for row in decisions}
    new_static_hits = []
    for sample_id, candidate_row in candidate_rows.items():
        if sample_id in already_used or sample_id not in rows_by_id or sample_id not in labels:
            continue
        label = labels[sample_id]
        gt = label.get("bbox", label.get("bbox_xyxy_normalized"))
        if gt is None:
            continue
        target, c_hit, initial_iou = _candidate_bbox_for_target(candidate_row, gt)
        if target is not None and c_hit:
            new_static_hits.append((rows_by_id[sample_id], candidate_row, target, initial_iou))
    rng.shuffle(new_static_hits)
    new_static_hits = new_static_hits[:len(moved_ids)]
    if len(new_static_hits) != len(moved_ids):
        raise ValueError("not enough unused C-hit rows to refill the static choice pool")

    generated_pairs = []
    generated_images = Counter()
    generated_statuses = Counter()
    for index, row in enumerate(moved_misses):
        sample_id = str(row["source_sample_id"])
        manifest_row = rows_by_id[sample_id]
        candidate_row = candidate_rows[sample_id]
        gt = labels[sample_id].get("bbox", labels[sample_id].get("bbox_xyxy_normalized"))
        target, c_hit, initial_iou = _candidate_bbox_for_target(candidate_row, gt)
        if target is None or c_hit:
            raise ValueError(f"{sample_id}: moved row no longer matches its C-miss label")
        pair, actual = _paired_evidence_decisions(
            manifest_row, candidate_row, target, initial_iou, c_hit,
            args.output_dir / "observations" / "paired" / sample_id,
            args.seed + 200000 + index)
        generated_pairs.extend(pair)
        generated_statuses[actual["status"]] += 1
        generated_images.update(actual["images_by_modality"])

    generated_static = []
    for index, (row, candidate_row, target, initial_iou) in enumerate(new_static_hits):
        decision, _ = _final_choice_decision(
            row, candidate_row, target, initial_iou, True,
            args.output_dir / "observations" / "final" / str(row["id"]), args.seed + 300000 + index)
        generated_static.append(decision)

    kept_static = [row for row in static if str(row["source_sample_id"]) not in moved_ids]
    kept_pairs = [row for sample_id in keep_old_pair_ids for row in pairs_by_id[sample_id]]
    balanced = [*kept_static, *generated_static, *kept_pairs, *generated_pairs]
    if len(balanced) != args.final_limit + 2 * args.paired_limit:
        raise ValueError(f"balanced batch row count mismatch: {len(balanced)}")

    c_hit_final = [row for row in balanced if row["sample_category"] in {
        "offline_gt_final_candidate_selection", "gt_supervised_final_choice_after_prescribed_observation"}]
    prescribed = [row for row in balanced if row["sample_category"] == "prescribed_operation_warmup"]
    observed_final = [row for row in balanced if row["sample_category"] == "gt_supervised_final_choice_after_prescribed_observation"]
    active_tool_statuses = Counter(row.get("tool_result_status", "unknown") for row in observed_final)
    teacher_path = args.output_dir / "blind_teacher_needed.jsonl"
    teacher_jobs = read_rows(teacher_path)
    backup = args.output_dir / "foundation_decisions_prebalance.jsonl"
    if backup.exists():
        raise FileExistsError(backup)
    temp = args.output_dir / "foundation_decisions.rebalanced.tmp"
    with temp.open("w", encoding="utf-8") as handle:
        for row in balanced:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    output.replace(backup)
    temp.replace(output)

    summary = {
        "schema": "visual_agent_capability_foundation_v1",
        "exported_decision_rows": len(balanced),
        "unique_training_start_ids": len({str(row["source_sample_id"]) for row in balanced}),
        "unique_image_groups": len({str(row["image_group"]) for row in balanced}),
        "decision_rows_by_category": dict(Counter(row["sample_category"] for row in balanced)),
        "static_offline_final_choice_rows": sum(row["sample_category"] == "offline_gt_final_candidate_selection" for row in balanced),
        "prescribed_observation_starts": len(prescribed),
        "post_observation_offline_final_choice_rows": len(observed_final),
        "offline_final_choice_c_hit": sum(bool(row["selected_is_c"]) for row in c_hit_final),
        "offline_final_choice_c_miss_corrected_by_candidate": sum(not bool(row["selected_is_c"]) for row in c_hit_final),
        "c_miss_final_labels_by_category": {
            "static": sum(row["sample_category"] == "offline_gt_final_candidate_selection" and not row["selected_is_c"] for row in balanced),
            "after_observation": sum(row["sample_category"] == "gt_supervised_final_choice_after_prescribed_observation" and not row["selected_is_c"] for row in balanced),
        },
        "c_hit_final_labels_by_category": {
            "static": sum(row["sample_category"] == "offline_gt_final_candidate_selection" and row["selected_is_c"] for row in balanced),
            "after_observation": sum(row["sample_category"] == "gt_supervised_final_choice_after_prescribed_observation" and row["selected_is_c"] for row in balanced),
        },
        "actual_cpu_inspect_results": dict(active_tool_statuses),
        "actual_tool_images_by_modality": dict(Counter(
            modality for row in observed_final for modality, count in row.get("actual_tool_images_by_modality", {}).items()
            for _ in range(count))),
        "actual_tool_result_images_added_to_followup_messages": sum(
            row.get("evidence_images_added_to_messages", 0) for row in observed_final),
        "prescribed_observation_policy": "Original Query stays in input. C and highest finite-score non-C RGB proposal use a GT-free pair rule; RGB+IR (or RGB only when IR is absent) inspect was executed on CPU and the returned images enter the follow-up input. The inspect is prescribed-operation warmup, not autonomous tool choice or causal evidence. Final IDs use offline GT with C-hit priority and any current legal RGB candidate.",
        "blind_teacher_needed_jobs": len(teacher_jobs),
        "blind_teacher_jobs_counted_as_samples": 0,
        "blind_teacher_job_buckets": dict(Counter(job["selection_bucket"] for job in teacher_jobs)),
        "balanced_by_reusing_existing_starts": {
            "retained_existing_c_hit_paired_starts": len(keep_old_pair_ids),
            "moved_c_miss_static_starts_into_real_paired_observations": len(moved_ids),
            "added_unused_c_hit_static_starts": len(generated_static),
            "prebalance_file": str(backup),
        },
        "new_rebalance_tool_results": dict(generated_statuses),
        "new_rebalance_tool_images_by_modality": dict(generated_images),
        "files": {"training_decisions": str(output), "pending_blind_teacher_jobs": str(teacher_path)},
        "reproduction_command": (
            "python -m experiments.visual_agent.export_capability_foundation --rebalance-existing "
            f"--manifest {args.manifest} --candidate-cache {args.candidate_cache} --labels {args.labels} "
            f"--output-dir {args.output_dir} --final-limit {args.final_limit} "
            f"--paired-limit {args.paired_limit} --rebalance-misses {args.rebalance_misses} --seed {args.seed}"
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-cache", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--final-limit", type=int, default=900)
    parser.add_argument("--paired-limit", type=int, default=300)
    parser.add_argument("--rebalance-existing", action="store_true")
    parser.add_argument("--rebalance-misses", type=int, default=150)
    parser.add_argument("--teacher-per-bucket", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2031)
    args = parser.parse_args()
    summary = rebalance_existing(args) if args.rebalance_existing else export(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
