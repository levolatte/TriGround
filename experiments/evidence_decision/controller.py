"""Bounded single-controller loop; no ground truth and no implicit KEEP."""
from __future__ import annotations

import copy
import json
import time
from collections import Counter

from .model import InputBudgetExceeded
from .vision_tools import ProtocolError
from .presentation import model_candidates, model_observation, compact_json, PROMPT_VERSION


SYSTEM_PROMPT = """You locate a queried object using complementary RGB, infrared (IR), and depth evidence.
The final box belongs to the original RGB frame, but evidence may come from ANY of the three modalities.
First identify which query condition is still uncertain and which observation could distinguish the competing objects.
RGB resolves visible color, texture, text, parts and global left/right/ordinal relations.
IR can distinguish object silhouettes and relative thermal contrast when RGB is blurred, dim,
low-contrast or partially obscured. Examine the actual IR scene even when the query never mentions IR.
For uncertain identity use inspect_regions with one ID and modalities=["rgb","ir"], view="cross",
or compare two competing IDs with modalities=["ir"], view="pair". Check the same scene location across modalities.
Depth can separate foreground/middle/background objects and near/far competitors relative to the CAMERA.
For a camera-distance condition use measure_depth on the competing IDs; inspect a depth crop for layer boundaries
when useful, then ground any numerical or ordered conclusion in reliable measurements.
If RGB does not expose the target and the IR scene suggests an omitted object, search_candidates with
modality="ir", role="target" can add it. Search a reference only if that reference is needed to resolve the query.
Do not spend successive calls repeating the same RGB view without a new question it can answer.
After an observation, compare what was learned with the query condition. If it was uninformative,
switch to a relevant complementary modality/tool or finish; do not claim it resolved the condition.
Use only supplied images, candidates and actual tool observations. KEEP is the exact initial box.
Candidate IDs are immutable. A reference-role object cannot be the final target.
All left/right/ordinal relations refer to the ORIGINAL GLOBAL scene, never the crop display order.
IR brightness is relative appearance, not a universal target rule or temperature measurement.
Depth evidence only supports camera-depth ordering when the tool says supported. It does not
measure 2D proximity or distance to a reference object. UNKNOWN and EMPTY are valid observations.
Keep a correct initial answer when no supported evidence justifies changing it. You may finish early.
Return ONE JSON object, no markdown or free coordinates. Optional evidence_note is a STRING FIELD INSIDE
that same action JSON object, never text after the closing brace. Keep it under 30 words: name the unresolved
condition, the observation sought, or the actual evidence supporting finish. No text outside the JSON.
Action schemas (all listed fields are required):
inspect_regions: action="inspect_regions", candidate_ids=array of current ID strings,
modalities=array drawn from rgb/ir/depth, view=single/pair/cross.
Views: single (one ID/one modality), pair (two distinct IDs/one modality),
cross (one or two distinct IDs, each shown in two or three requested modalities, sharing one image budget).
Inspect crops help identify objects; their coordinates are never answers.
measure_depth: action="measure_depth", candidate_ids=array of one or two current ID strings.
search_candidates: action="search_candidates", category=object category from the query,
region=one allowed region, modality=rgb/ir, role=target/reference.
Search regions: full,left,right,top,bottom,grid:R:C (R,C zero-based 0..2), context:ID.
Search can also find role=reference. Search is available ONLY when explicitly enabled.
finish: action="finish", candidate_id=KEEP or one current target ID string.
Copy IDs only from this question's Current candidates. An old numeric alias for KEEP is not legal.
For a fixed schedule ONLY: action="continue" with no other fields executes its next displayed step.
"""


def parse_action(text):
    try:
        action = json.loads(text)
    except json.JSONDecodeError as error:
        raise ProtocolError("expected one complete JSON action") from error
    if not isinstance(action, dict) or not isinstance(action.get("action"), str):
        raise ProtocolError("action JSON needs a string action field")
    return action


def text_message(text):
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def observation_message(observation, tool_pixels):
    content = [{"type": "text", "text": compact_json(model_observation(observation))}]
    images = observation.get("images", [])
    for picture in images:
        content.append({"type": "text", "text":
                        f"Observation image: modality={picture.get('modality')}, ID roles={picture.get('candidate_roles', {})}, "
                        f"IDs={picture.get('candidate_ids')}, original boxes={picture.get('source_bbox', {})}, "
                        f"crop pixels={picture.get('crop_xyxy')} in source size={picture.get('source_size')}, "
                        f"interpolated={picture.get('interpolated')}. Original-scene relations still apply."})
        content.append({"type": "image", "image": picture["path"],
                        "max_pixels": max(1024, tool_pixels // max(1, len(images))),
                        "modality": picture.get("modality"), "view": "tool",
                        "candidate_ids": picture.get("candidate_ids", []),
                        "target_boxes": picture.get("target_boxes", []),
                        "source_bbox": picture.get("source_bbox", {})})
    return {"role": "user", "content": content}


def fixed_schedule(candidates, modalities):
    def score(candidate):
        scores = [source.get("score") for source in candidate.get("sources", [])]
        return candidate.get("score") or max([float(x) for x in scores if x is not None] or [0.0])
    competitors = sorted([c for c in candidates if str(c["id"]) != "KEEP" and c["role"] == "target"],
                         key=lambda c: (-score(c), tuple(c["bbox"])))
    first = ["KEEP"] + ([str(competitors[0]["id"])] if competitors else [])
    second = ["KEEP", str(competitors[1]["id"])] if len(competitors) > 1 else None
    def inspect(ids, modality):
        return {"action": "inspect_regions", "candidate_ids": ids, "modalities": [modality],
                "view": "pair" if len(ids) == 2 else "single"}
    # v5 exposes auxiliary evidence within the fast two-call budget.
    steps = [inspect(first, "ir" if "ir" in modalities else "rgb")]
    if "depth" in modalities:
        steps.append({"action": "measure_depth", "candidate_ids": first})
    if "ir" in modalities:
        steps.append(inspect(first, "rgb"))
    if len(modalities) > 1:
        steps.append({"action": "inspect_regions", "candidate_ids": ["KEEP"],
                      "modalities": modalities, "view": "cross"})
    if second:
        steps.append(inspect(second, "ir" if "ir" in modalities else "rgb"))
        if "depth" in modalities:
            steps.append({"action": "measure_depth", "candidate_ids": second})
    return steps


def run_episode(row, pool, tools, backend, initial_messages, *, mechanism, profile, memory="full", on_event=None):
    """Backend injection permits protocol tests; model and scripted runs stay distinct."""
    started = time.perf_counter()
    backend.begin_sample()
    initial = copy.deepcopy(initial_messages)
    history = copy.deepcopy(initial)
    events = []
    def record_event(event):
        events.append(event)
        if on_event:
            on_event({"phase": "completed", "event": event})
    initial_candidates = pool.public_candidates()
    initial_pool_snapshot = pool.snapshot()
    available = [name for name, key in (("rgb", "rgb"), ("ir", "ir"), ("depth", "depth_visual"))
                 if row["images"].get(key)]
    schedule = fixed_schedule(initial_candidates, available)[:profile.evidence_calls]
    metrics = {key: 0 for key in ("model_calls", "input_tokens", "input_text_tokens", "visual_tokens",
               "output_tokens", "tool_calls", "search_calls", "invalid_actions", "model_seconds",
               "tool_seconds", "preprocess_seconds")}
    status_counts = Counter()
    final_status, selected, bbox = "NO_FINAL_ACTION", None, None
    protocol_feedback_used = False
    force_finish = mechanism == "A"
    schedule_index = 0
    max_decisions = 1 if mechanism == "A" else (2 if mechanism == "B" else profile.evidence_calls + 1)

    for step in range(max_decisions):
        candidates_before = pool.public_candidates()
        if memory == "latest":
            from .state import build_state_messages
            messages = build_state_messages(initial, events, candidates_before)
        else:
            messages = copy.deepcopy(history)
        final_only = force_finish or step == max_decisions - 1
        remaining = profile.output_tokens - metrics["output_tokens"]
        if remaining <= profile.finish_tokens:
            final_only = True
        if mechanism == "B" and step == 0:
            instruction = "Compare the supplied RGB appearance, IR silhouettes/contrast and depth layers for the query conditions and competing IDs. State which evidence is decisive or insufficient. No tools are available. Return JSON with concise evidence_notes; the next turn will request your final action. Stop when sufficient."
            cap = remaining - profile.finish_tokens
        elif final_only:
            instruction = "Final decision now. Only action=finish with a current target candidate_id or KEEP is legal. No tools or coordinates."
            cap = min(128 if mechanism == "A" else profile.finish_tokens, remaining)
        else:
            instruction = "Identify the unresolved query condition in evidence_note and choose one observation that can distinguish the competing IDs, or finish using observed evidence. Consider IR for uncertain object identity/visibility and measured Depth for camera-distance layers; RGB-only zoom is useful only if it can resolve the remaining condition. "
            cap = min(profile.action_tokens, remaining - profile.finish_tokens)
            if mechanism == "S":
                if schedule_index >= len(schedule):
                    instruction = "Fixed schedule exhausted. Only action=finish is legal."
                    final_only = True
                    cap = min(profile.finish_tokens, remaining)
                else:
                    instruction += "Only action=continue or finish is legal. The next fixed tool is: " + json.dumps(schedule[schedule_index])
            else:
                instruction += "search_candidates is " + ("enabled." if mechanism == "D" else "disabled; use inspect_regions, measure_depth or finish.")
        if cap <= 0:
            final_status = "OUTPUT_BUDGET_EXHAUSTED"
            break
        messages.append(text_message(instruction + ("" if memory == "latest" else
                        "\nCurrent candidates: " + compact_json(model_candidates(candidates_before)))))
        event = {"step": step, "messages": messages, "candidates_before": candidates_before}
        remaining_visual = profile.cumulative_visual_tokens - metrics["visual_tokens"]
        try:
            result = backend.generate(messages, cap, profile, remaining_visual)
        except InputBudgetExceeded as error:
            # Do not silently discard images/Query to squeeze a sample through.
            event.update(raw_output="", action=None, usage=error.usage,
                         observation={"status": "LIMIT", "text": str(error), "images": [], "data": {}})
            record_event(event)
            status_counts["LIMIT"] += 1
            metrics["preprocess_seconds"] += error.usage.get("preprocess_seconds", 0)
            metrics["rejected_input_tokens"] = metrics.get("rejected_input_tokens", 0) + error.usage.get("input_tokens", 0)
            final_status = "INPUT_BUDGET_EXHAUSTED"
            break
        raw, usage = result["raw_output"], result["usage"]
        event.update(raw_output=raw, usage=usage)
        if on_event:
            on_event({"phase": "decision", "event": event})
        metrics["model_calls"] += 1
        for key in ("input_tokens", "input_text_tokens", "visual_tokens", "output_tokens",
                    "model_seconds", "preprocess_seconds"):
            metrics[key] += usage.get(key, 0)
        history.append({"role": "assistant", "content": raw})
        if mechanism == "B" and step == 0:
            event.update(action={"action": "analyze"}, observation=None,
                         candidates_after=pool.public_candidates())
            record_event(event)
            continue
        protocol_error_handled = False
        try:
            action = parse_action(raw)
            event["action"] = action
            kind = action["action"]
            if kind == "finish":
                if "candidate_id" not in action:
                    raise ProtocolError("finish needs candidate_id")
                selected = str(action["candidate_id"])
                bbox = pool.finish(selected)
                final_status = "FINISHED"
                event.update(observation=None, candidates_after=pool.public_candidates())
                record_event(event)
                break
            if final_only:
                raise ProtocolError("only finish is allowed on the final decision")
            if mechanism == "S":
                scheduled = schedule[schedule_index]
                # Copying the displayed next step grants no planning freedom.
                exact_step = {key: value for key, value in action.items() if key != "evidence_note"} == scheduled
                if kind != "continue" and not exact_step:
                    raise ProtocolError("fixed schedule accepts continue or finish only")
                event["fixed_step_alias"] = kind != "continue"
                executed_action = scheduled
                schedule_index += 1
            else:
                allowed = {"inspect_regions", "measure_depth"} | ({"search_candidates"} if mechanism == "D" else set())
                if kind not in allowed:
                    raise ProtocolError(f"action {kind} is unavailable in VA-{mechanism}")
                executed_action = action
            # Reserve a final read of current evidence plus the largest next image observation.
            next_visual = usage.get("visual_tokens", 0)
            if memory == "full" and executed_action["action"] in {"inspect_regions", "search_candidates"}:
                next_visual += (profile.tool_pixels + 1023) // 1024
            if next_visual > profile.cumulative_visual_tokens - metrics["visual_tokens"]:
                observation = {"status": "LIMIT", "text": "Remaining visual budget is reserved for finish.", "images": [], "data": {}}
                force_finish = True
            else:
                counts_before = dict(getattr(tools, "cost_counts", {}))
                seconds_before = dict(getattr(tools, "cost_seconds", {}))
                tool_start = time.perf_counter()
                observation = tools.execute(executed_action)
                event["tool_seconds"] = time.perf_counter() - tool_start
                metrics["tool_seconds"] += event["tool_seconds"]
                event["tool_cost_counts"] = {key: value - counts_before.get(key, 0)
                                             for key, value in getattr(tools, "cost_counts", {}).items()}
                event["tool_cost_seconds"] = {key: value - seconds_before.get(key, 0)
                                              for key, value in getattr(tools, "cost_seconds", {}).items()}
                metrics["tool_calls"] += 1
                metrics["search_calls"] += int(executed_action["action"] == "search_candidates")
            event["executed_action"] = executed_action
        except ProtocolError as error:
            # Only protocol errors are recoverable here; runtime/model errors propagate.
            metrics["invalid_actions"] += 1
            protocol_error_handled = True
            selected, bbox = None, None
            observation = {"status": "ERROR", "text": f"Invalid action: {error}", "images": [], "data": {"kind": "protocol"}}
            event.setdefault("action", None)
            if protocol_feedback_used or final_only:
                final_status = "INVALID_FINAL_ACTION"
                event.update(observation=observation, candidates_after=pool.public_candidates())
                status_counts["ERROR"] += 1
                record_event(event)
                break
            protocol_feedback_used = True
        if observation["status"] == "ERROR" and observation.get("data", {}).get("kind") == "protocol" and not protocol_error_handled:
            metrics["invalid_actions"] += 1
            if protocol_feedback_used:
                final_status = "INVALID_FINAL_ACTION"
                event.update(observation=observation, candidates_after=pool.public_candidates())
                status_counts["ERROR"] += 1
                record_event(event)
                break
            protocol_feedback_used = True
        event.update(observation=observation, candidates_after=pool.public_candidates())
        status_counts[observation["status"]] += 1
        record_event(event)
        history.append(observation_message(observation, profile.tool_pixels))

    if final_status != "FINISHED":
        bbox, selected = None, None
    metrics.update(elapsed_seconds=time.perf_counter() - started, peak_memory_bytes=backend.peak_memory(),
                   tool_status_counts=dict(status_counts), legal_finish=final_status == "FINISHED")
    metrics.update({key: value for key, value in getattr(tools, "cost_counts", {}).items()})
    metrics.update({key + "_seconds": value for key, value in getattr(tools, "cost_seconds", {}).items()})
    return {"schema_version": "visual-agent-v1", "prompt_version": PROMPT_VERSION,
            "id": str(row["id"]), "query": row["query"],
            "manifest_row": row, "initial_messages": initial, "initial_candidates": initial_candidates,
            "initial_bbox": pool.finish("KEEP"), "image_group": row["images"]["rgb"],
            "events": events, "final_status": final_status, "selected_id": selected, "bbox": bbox,
            "initial_pool_snapshot": initial_pool_snapshot, "final_pool_snapshot": pool.snapshot(),
            "final_candidates": pool.public_candidates(), "metrics": metrics,
            "mechanism": mechanism, "profile": profile.to_dict(), "memory": memory}
