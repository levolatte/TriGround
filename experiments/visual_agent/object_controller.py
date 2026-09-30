"""Object-centered Qwen tool loop for multimodal visual grounding."""
from __future__ import annotations

import copy
import json
import re
import time
from collections import Counter
from dataclasses import dataclass

from .model import InputBudgetExceeded
from .presentation import compact_json


PROMPT_VERSION = "paired-evidence-capability-training"
MAX_EVIDENCE_CALLS = 6
MAX_RETAINED_TOOL_PIXELS = 602112
MAX_ACTIVE_CANDIDATES = 3
REGION_EVIDENCE_ID = "__latest_region__"
RECOVERY_INSTRUCTION = (
    "The previous response or finish was rejected. This is the one recovery attempt: "
    "use the error feedback and current evidence to make a valid decision. Do not assume a default target."
)
DECISION_INSTRUCTION = "Choose one informative inspect/depth/search call, or finish. Name the remaining condition and competitors briefly."
FINAL_INSTRUCTION = "Only finish is available; choose the supported target."


@dataclass(frozen=True)
class ObjectProfile:
    name: str = "object-centered"
    global_pixels: int = 602112
    global_rgb_pixels: int = 602112
    global_ir_pixels: int = 602112
    global_depth_pixels: int = 200704
    candidate_overview_pixels: int = 196608
    tool_pixels: int = 602112
    visual_tokens: int = 4096
    context_tokens: int = 4096
    output_tokens: int = 2048
    action_tokens: int = 256
    finish_tokens: int = 128


OBJECT_PROFILE = ObjectProfile()


SYSTEM_PROMPT = """You ground a natural-language query to one object candidate in the original RGB frame.
Public P labels are stable candidate IDs, not physical-object identities. Every bbox is a hypothesis in the candidate's stated coordinate_frame: different IDs can refer to the same physical object, and a physical object can have multiple boxes. Candidate role is only a hypothesis. Only finish_eligible=true boxes are valid RGB output boxes; other coordinate frames may guide search but cannot be returned. Never reject a finish solely because role says reference.
Use only the query, supplied global frames, RGB candidate overview, current candidate table, and actual tool results. Preserve the exact source categories and modalities shown with each ID; do not infer object identity from ID order, box overlap, or role.
Before a tool call, write one concise natural sentence stating the target conditions and naming up to three plausible competing IDs. On later turns name the remaining uncertainty. Do not expose long reasoning. Assistant prose is a model hypothesis, not verified evidence; tool-role messages contain tool facts. Keep those sources distinct and revise a hypothesis only when the returned evidence supports it.
RGB supports appearance, parts, text and global left/right or ordinal relations. IR can reveal silhouettes and relative thermal contrast; brightness alone is not a target rule. Compare or pair RGB/IR regions only when the supplied registration metadata permits it. Depth tools report camera-depth evidence only when supported. Follow depth_encoding and depth_visual_encoding metadata for direction; if encoding is unknown, image brightness cannot establish near/far.
Use inspect for visual evidence, depth for camera-distance comparisons, search only when the pool may have missed an object, and finish when evidence is sufficient. Choose observations that distinguish the stated competitors. Do not repeat an uninformative observation.
All spatial relations refer to the original global scene, never atlas or crop layout. Atlas pages show RGB candidate boxes and outside labels. Tool observations can be UNKNOWN or EMPTY. Do not claim they resolved a condition unless their returned evidence supports it.
Call finish with a current finish-eligible public candidate ID. There is no implicit default candidate.
Each response must contain exactly one <tool_call>{\"name\":...,\"arguments\":{...}}</tool_call> block. A short assistant sentence may precede it. Do not use legacy action JSON or return coordinates."""


def _function(name, description, properties, required=()):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(required),
                "additionalProperties": False,
            },
        },
    }


_ID_ARRAY = {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": MAX_ACTIVE_CANDIDATES}
_MODALITIES = {"type": "array", "items": {"type": "string", "enum": ["rgb", "ir", "depth"]},
               "minItems": 1, "maxItems": 3, "uniqueItems": True}
_REGION = {"type": "string", "description":
           "Optional: full, left, right, top, bottom, grid:row:col (row/col 0..2), or context:P1 using an existing ID. Omit for automatic framing. Near/far/background are not image regions."}

TOOL_SCHEMAS = [
    _function("inspect", "Inspect one or more candidate regions in selected modalities.", {
        "ids": _ID_ARRAY,
        "region": _REGION,
        "modalities": _MODALITIES,
    }),
    _function("depth", "Compare camera-depth measurements for one or more candidate IDs.", {
        "ids": _ID_ARRAY,
    }, required=("ids",)),
    _function("search", "Search a scene region for a target or reference object and add public candidate IDs.", {
        "category": {"type": "string"},
        "modality": {"type": "string", "enum": ["rgb", "ir"]},
        "role": {"type": "string", "enum": ["target", "reference"]},
        "region": _REGION,
    }, required=("category", "modality")),
    _function("finish", "Select the final target using its stable public candidate ID.", {
        "id": {"type": "string"},
    }, required=("id",)),
]
FINISH_SCHEMA = TOOL_SCHEMAS[-1]
TOOL_NAMES = {item["function"]["name"] for item in TOOL_SCHEMAS}


class ToolCallError(ValueError):
    pass


def parse_tool_call(raw_output: str):
    """Parse one native Qwen tool-call tag plus optional short assistant text."""
    matches = list(re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", raw_output, flags=re.DOTALL))
    if len(matches) != 1:
        raise ToolCallError("expected exactly one native <tool_call> block")
    match = matches[0]
    try:
        call = json.loads(match.group(1))
    except json.JSONDecodeError as error:
        raise ToolCallError("tool call body must be JSON") from error
    if not isinstance(call, dict) or call.get("name") not in TOOL_NAMES:
        raise ToolCallError("tool call needs a supported name")
    arguments = call.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as error:
            raise ToolCallError("tool call arguments string must be JSON") from error
    if not isinstance(arguments, dict):
        raise ToolCallError("tool call arguments must be an object")
    note = (raw_output[:match.start()] + raw_output[match.end():]).strip()
    return {"name": call["name"], "arguments": arguments}, note


def _image_size(picture):
    size = picture.get("saved_size") or picture.get("size")
    if size:
        return int(size[0]), int(size[1])
    from PIL import Image
    with Image.open(picture["path"]) as image:
        return image.size


def _picture_text(picture):
    modality = picture.get("modality", "unknown")
    ids = ",".join(str(value) for value in picture.get("candidate_ids", [])) or "none"
    registration = picture.get("registration", picture.get("ir_rgb_registration"))
    registration_note = f" Registration: {registration}." if registration else ""
    return (f"Tool image for candidate IDs {ids} in modality {modality}; it is a view of bbox hypotheses, "
            f"not a physical-object identity label. Positions refer to the original frame.{registration_note}")


def _candidate_text_record(candidate):
    sources = []
    seen_sources = set()
    for source in candidate.get("sources", []):
        summary = {key: source[key] for key in ("modality", "category") if key in source}
        signature = compact_json(summary)
        if summary and signature not in seen_sources:
            seen_sources.add(signature)
            sources.append(summary)
    item = {"id": str(candidate["id"]), "role": candidate.get("role", "unknown"),
            "bbox": [round(float(value), 3) for value in candidate["bbox"]],
            "sources": sources}
    if "coordinate_frame" in candidate:
        item["coordinate_frame"] = candidate["coordinate_frame"]
    if "finish_eligible" in candidate:
        item["finish_eligible"] = bool(candidate["finish_eligible"])
    overlap_ids = candidate.get("overlap_candidate_ids", candidate.get("possible_same_object_ids", []))
    if overlap_ids:
        item["bbox_overlap_ids"] = [str(value) for value in overlap_ids]
    return item


def _initial_candidate_text(candidates):
    return compact_json([_candidate_text_record(candidate) for candidate in candidates])


def _depth_metadata_text(row):
    depth_encoding = row.get("depth_encoding", "unknown")
    visual_encoding = row.get("depth_visual_encoding", "unknown")
    if visual_encoding == "city_native_depth_visual":
        visual_rule = "Valid brighter pixels mean nearer; black pixels are invalid."
    elif visual_encoding == "fixed_raw_p01_p99_v1":
        visual_rule = "This is a per-image P01-P99 stretch; brightness alone has no confirmed near/far direction."
    else:
        visual_rule = "Visual depth direction is unknown; do not infer near/far from brightness."
    registration = row.get("ir_rgb_registration", "unknown")
    registration_source = row.get("registration_source")
    source_note = f" Registration source: {registration_source}." if registration_source else ""
    return (f"Depth metadata: numeric encoding={depth_encoding}; visual encoding={visual_encoding}. {visual_rule} "
            f"RGB/IR registration={registration}.{source_note}")


def _atlas_page_text(picture, index, total):
    rows = {}
    for candidate_id, position in picture.get("tile_positions", {}).items():
        rows.setdefault(int(position["row"]), []).append((int(position["col"]), str(candidate_id)))
    if rows:
        layout = "; ".join(
            "row" + str(row + 1) + " " + " ".join(candidate_id for _, candidate_id in sorted(values))
            for row, values in sorted(rows.items())
        )
    else:
        layout = "IDs " + " ".join(str(value) for value in picture.get("candidate_ids", []))
    return f"Atlas page {index}/{total} {picture.get('modality', 'unknown')}: {layout}."


def build_initial_messages(row, candidates, atlas_observation, profile=OBJECT_PROFILE):
    """Build global frames, compact candidate facts and the RGB-only overview."""
    content = [{"type": "text", "text":
                f"Query: {row['query']}\nCurrent public candidates (global xyxy bbox hypotheses; see coordinate_frame): " +
                _initial_candidate_text(candidates) +
                "\nCandidate role is a hypothesis; bbox overlap is geometric only, not proof of object identity. "
                "The public source summary preserves modality and category without repeating source boxes or scores. "
                "Global frames below are original scene views."}]
    content.append({"type": "text", "text": _depth_metadata_text(row)})
    available = [("rgb", "rgb"), ("ir", "ir"), ("depth", "depth_visual")]
    global_budgets = {"rgb": profile.global_rgb_pixels, "ir": profile.global_ir_pixels,
                      "depth": profile.global_depth_pixels}
    for modality, key in available:
        path = row.get("images", {}).get(key)
        if not path:
            continue
        content.append({"type": "text", "text": f"Original global {modality} frame."})
        content.append({"type": "image", "image": str(path), "max_pixels": global_budgets[modality],
                        "modality": modality, "view": "global"})

    pages = (atlas_observation or {}).get("images", [])[:6]
    content.append({"type": "text", "text":
                    "The initial candidate overview contains RGB views with thin candidate boxes and labels outside each box."})
    per_page = max(1024, profile.candidate_overview_pixels // max(1, len(pages)))
    for index, picture in enumerate(pages, start=1):
        content.append({"type": "text", "text": _atlas_page_text(picture, index, len(pages))})
        content.append({"type": "image", "image": picture["path"],
                        "max_pixels": min(per_page, profile.candidate_overview_pixels),
                        "modality": picture.get("modality"), "view": "atlas",
                        "candidate_ids": picture.get("candidate_ids", []),
                        "target_boxes": picture.get("target_boxes", []),
                        "source_bbox": picture.get("source_bbox", {}),
                        "registration": picture.get("registration")})
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]


def _tool_facts(observation, name):
    source_data = observation.get("data") or {}
    data = {}
    if name == "inspect":
        data["regions"] = [
            {key: region[key] for key in ("modality", "candidate_ids", "region_px", "coordinate_frame",
                                           "registration") if key in region}
            for region in source_data.get("regions", [])
        ]
    elif name == "depth":
        data["measurements"] = source_data.get("measurements", {})
        if "reason" in source_data:
            data["reason"] = source_data["reason"]
        data["pair_results"] = [
            {key: pair[key] for key in ("candidate_a_id", "candidate_b_id", "status", "reason",
                                          "near_candidate_id", "far_candidate_id", "near_median_m",
                                          "far_median_m", "median_difference_m", "required_difference_m")
             if key in pair}
            for pair in source_data.get("pair_results", [])
        ]
    elif name == "search":
        for key in ("category", "modality", "role", "found_ids", "appended_ids"):
            if key in source_data:
                data[key] = source_data[key]
        appended = set(str(value) for value in source_data.get("appended_ids", []))
        if appended:
            data["new_candidates"] = [
            _candidate_text_record(candidate)
            for candidate in source_data.get("candidates", [])
            if str(candidate["id"]) in appended
            ]
    elif name == "finish":
        for key in ("candidate_id", "bbox", "finish_eligible", "reason"):
            if key in source_data:
                data[key] = source_data[key]
    else:
        for key in ("reason", "candidate_ids", "action_count"):
            if key in source_data:
                data[key] = source_data[key]
    return {"status": observation.get("status"), "text": observation.get("text", ""), "data": data}


def build_messages(initial_messages, turns, candidates, model_notes, retained_images, instruction, profile=OBJECT_PROFILE):
    """Compose tool facts, model hypotheses and active paired evidence as separate messages."""
    messages = copy.deepcopy(initial_messages)
    kept_by_event = {}
    for retained in retained_images.values():
        kept_by_event.setdefault(retained["event_index"], {})[retained["image_key"]] = retained["picture"]

    unique_images = {}
    for retained in retained_images.values():
        unique_images[retained["image_key"]] = retained
    ordered_images = sorted(unique_images.values(), key=lambda item: item["order"])
    per_image = max(1024, profile.tool_pixels // max(1, len(ordered_images)))

    for event_index, turn in enumerate(turns):
        assistant_message = copy.deepcopy(turn["assistant_message"])
        messages.append(assistant_message)
        tool_message = copy.deepcopy(turn["tool_message"])
        for image_key, picture in kept_by_event.get(event_index, {}).items():
            tool_message["content"].append({"type": "text", "text": _picture_text(picture)})
            tool_message["content"].append({"type": "image", "image": picture["path"],
                "max_pixels": min(per_image, profile.tool_pixels), "modality": picture.get("modality"),
                "view": "tool", "candidate_ids": picture.get("candidate_ids", []),
                "target_boxes": picture.get("target_boxes", []),
                "source_bbox": picture.get("source_bbox", {}),
                "coordinate_frame": picture.get("coordinate_frame"),
                "registration": picture.get("registration", picture.get("ir_rgb_registration"))})
        messages.append(tool_message)
        if turn.get("feedback_message"):
            messages.append(copy.deepcopy(turn["feedback_message"]))

    messages.append({"role": "user", "content": [{"type": "text", "text": instruction}]})
    return messages


def _assistant_tool_message(call_id, call, note):
    function_call = {
        "id": call_id,
        "type": "function",
        "function": {"name": call["name"], "arguments": compact_json(call["arguments"])},
    }
    return {"role": "assistant", "content": note or "", "tool_calls": [function_call]}


def _tool_message(call_id, name, observation, row=None):
    facts = _tool_facts(observation, name)
    if name == "depth" and row is not None:
        facts["data"]["depth_encoding"] = row.get("depth_encoding", "unknown")
        facts["data"]["depth_visual_encoding"] = row.get("depth_visual_encoding", "unknown")
    content = [{"type": "text", "text": compact_json(facts)}]
    return {"role": "tool", "name": name, "tool_call_id": call_id, "content": content}


def _active_candidate_ids(note, call, observation, candidates, retained):
    valid_ids = {str(candidate["id"]) for candidate in candidates}
    # Preserve the evidence explicitly requested/returned by this action before
    # historical competitors named in prose. In particular, a search can return
    # four proposal IDs even though only three remain active competitors.
    ordered = [str(value) for value in call.get("arguments", {}).get("ids", [])]
    data = observation.get("data") or {}
    ordered.extend(str(value) for value in data.get("appended_ids", []))
    ordered.extend(str(value) for value in data.get("found_ids", []))
    for picture in observation.get("images", []):
        ordered.extend(str(value) for value in picture.get("candidate_ids", []))
    ordered.extend(re.findall(r"(?<![A-Za-z0-9_])P\d+(?![A-Za-z0-9_])", note or ""))
    ordered.extend(str(candidate_id) for candidate_id, _ in retained
                   if candidate_id != REGION_EVIDENCE_ID)
    result = []
    for candidate_id in ordered:
        if candidate_id in valid_ids and candidate_id not in result:
            result.append(candidate_id)
        if len(result) == MAX_ACTIVE_CANDIDATES:
            break
    return result


def _remember_observation_images(observation, event_index, retained, next_order, pixel_budget,
                                 active_candidate_ids=None):
    """Keep paired evidence for three active IDs plus every image from the latest result."""
    active = list(dict.fromkeys(str(value) for value in (active_candidate_ids or [])))[:MAX_ACTIVE_CANDIDATES]
    if not active:
        active = list(dict.fromkeys(
            str(candidate_id) for candidate_id, _ in retained if candidate_id != REGION_EVIDENCE_ID
        ))[:MAX_ACTIVE_CANDIDATES]
    pictures = observation.get("images", [])
    # Candidate-keyed history remains only for the selected three competitors.
    # Latest-result images are separately retained until a later observation
    # supplies new images, so a four-proposal search is still visible next turn.
    keep = {key: item for key, item in retained.items()
            if key[0] in active and key[0] != REGION_EVIDENCE_ID}
    if not pictures:
        keep.update({key: item for key, item in retained.items() if key[0] == REGION_EVIDENCE_ID})

    for picture_index, picture in enumerate(observation.get("images", [])):
        image_key = f"{event_index}:{picture_index}"
        ids = [str(candidate_id) for candidate_id in picture.get("candidate_ids", [])]
        item = {"event_index": event_index, "image_key": image_key, "picture": picture, "order": next_order}
        next_order += 1
        modality = str(picture.get("modality", "unknown"))
        if ids:
            for candidate_id in ids:
                if candidate_id in active:
                    keep[(candidate_id, modality)] = item
        # An observation image is an actual returned fact even if more than
        # three proposals were found or it has no candidate ID (region view).
        keep[(REGION_EVIDENCE_ID, f"{modality}:{picture_index}")] = item
    return keep, next_order


def run_episode(row, tools, backend, initial_messages, *, profile=OBJECT_PROFILE, on_event=None):
    """Run up to six native tool invocations, with no implicit KEEP selection."""
    started = time.perf_counter()
    backend.begin_sample()
    initial = copy.deepcopy(initial_messages)
    initial_candidates = tools.public_candidates()
    initial_bbox = list(tools.pool.c_bbox)
    events, turns, model_notes = [], [], []
    retained_images = {}
    next_image_order = 0
    metrics = {key: 0 for key in ("model_calls", "input_tokens", "input_text_tokens", "visual_tokens",
               "output_tokens", "tool_calls", "finish_calls", "tool_invocations", "invalid_actions", "model_seconds", "tool_seconds",
               "preprocess_seconds")}
    status_counts = Counter()
    final_status, selected_id, selected_public_id, bbox = "NO_FINAL_ACTION", None, None, None
    step = 0
    recovery_feedback_used = False
    recovery_pending = False
    while step < MAX_EVIDENCE_CALLS + 2:
        tool_slots = MAX_EVIDENCE_CALLS - metrics["tool_calls"]
        remaining_output = profile.output_tokens - metrics["output_tokens"]
        finish_only = tool_slots <= 0 or remaining_output <= profile.finish_tokens
        if remaining_output <= 0:
            final_status = "OUTPUT_BUDGET_EXHAUSTED"
            break
        if finish_only:
            cap = min(profile.finish_tokens, remaining_output)
            allowed_tools = [FINISH_SCHEMA]
            instruction = RECOVERY_INSTRUCTION if recovery_pending else FINAL_INSTRUCTION
        else:
            cap = min(profile.action_tokens, remaining_output - profile.finish_tokens)
            if cap <= 0:
                final_status = "OUTPUT_BUDGET_EXHAUSTED"
                break
            allowed_tools = TOOL_SCHEMAS
            instruction = RECOVERY_INSTRUCTION if recovery_pending else DECISION_INSTRUCTION
        candidates = tools.public_candidates()
        messages = build_messages(initial, turns, candidates, model_notes, retained_images, instruction, profile)
        event = {"step": step, "messages": copy.deepcopy(messages), "candidates_before": candidates,
                 "available_tools": [item["function"]["name"] for item in allowed_tools],
                 "tools": copy.deepcopy(allowed_tools)}
        try:
            result = backend.generate(messages, cap, profile, profile.visual_tokens, tools=allowed_tools)
        except InputBudgetExceeded as error:
            metrics["preprocess_seconds"] += error.usage.get("preprocess_seconds", 0)
            event["budget_transition"] = {"reason": str(error), "usage": error.usage}
            # A budget limit permits one explicit final decision, never an
            # implicit answer. The smaller finish schema releases prompt space.
            allowed_tools = [FINISH_SCHEMA]
            event.update(available_tools=["finish"], tools=copy.deepcopy(allowed_tools))
            messages = build_messages(initial, turns, candidates, model_notes, retained_images,
                "The observation input budget is exhausted. Call finish now using the evidence already available.", profile)
            event["messages"] = copy.deepcopy(messages)
            try:
                if finish_only:
                    raise error
                result = backend.generate(messages, min(profile.finish_tokens, remaining_output),
                                          profile, profile.visual_tokens, tools=allowed_tools)
            except InputBudgetExceeded as final_error:
                event.update(raw_output="", action=None, usage=final_error.usage,
                             error={"type": type(final_error).__name__, "message": str(final_error)})
                events.append(event)
                if on_event:
                    on_event({"phase": "completed", "event": event})
                final_status = "INPUT_BUDGET_EXHAUSTED"
                break

        raw, usage = result["raw_output"], result["usage"]
        event.update(raw_output=raw, usage=usage)
        metrics["model_calls"] += 1
        for key in ("input_tokens", "input_text_tokens", "visual_tokens", "output_tokens",
                    "model_seconds", "preprocess_seconds"):
            metrics[key] += usage.get(key, 0)
        if on_event:
            on_event({"phase": "decision", "event": event})
        try:
            call, note = parse_tool_call(raw)
        except ToolCallError as error:
            metrics["invalid_actions"] += 1
            observation = {"status": "ERROR", "text": str(error), "images": [], "data": {}}
            event.update(action=None, observation=observation, protocol_valid=False)
            events.append(event)
            status_counts["ERROR"] += 1
            step += 1
            can_recover = (not recovery_feedback_used and
                           profile.output_tokens - metrics["output_tokens"] >= profile.finish_tokens)
            event["recoverable"] = can_recover
            if can_recover:
                recovery_feedback_used = True
                recovery_pending = True
                turns.append({
                    "assistant_message": {"role": "assistant", "content": raw},
                    "tool_message": {"role": "user", "content": [{"type": "text",
                        "text": f"Protocol error: {error}. {RECOVERY_INSTRUCTION}"}]},
                })
            if on_event:
                on_event({"phase": "completed", "event": event})
            if can_recover:
                continue
            final_status = "INVALID_MODEL_OUTPUT"
            break

        event["action"] = call
        event["model_note"] = note[:280]
        model_notes.append(note[:280] if note else "")
        if call["name"] not in {item["function"]["name"] for item in allowed_tools}:
            metrics["invalid_actions"] += 1
            observation = {"status": "ERROR", "text": "Only finish is allowed with the remaining budget.",
                           "images": [], "data": {}}
            event.update(observation=observation, candidates_after=candidates, protocol_valid=False)
            events.append(event)
            status_counts["ERROR"] += 1
            step += 1
            can_recover = (not recovery_feedback_used and
                           profile.output_tokens - metrics["output_tokens"] >= profile.finish_tokens)
            event["recoverable"] = can_recover
            if can_recover:
                recovery_feedback_used = True
                recovery_pending = True
                turns.append({
                    "assistant_message": _assistant_tool_message(f"object_call_{step}", call, note),
                    "tool_message": {"role": "user", "content": [{"type": "text",
                        "text": observation["text"] + " " + RECOVERY_INSTRUCTION}]},
                })
            if on_event:
                on_event({"phase": "completed", "event": event})
            if can_recover:
                continue
            final_status = "INVALID_FINAL_ACTION"
            break
        call_id = f"object_call_{step + 1}"
        assistant_message = _assistant_tool_message(call_id, call, note)
        tool_start = time.perf_counter()
        observation = tools.execute(call["name"], call["arguments"])
        tool_seconds = time.perf_counter() - tool_start
        metrics["tool_invocations"] += 1
        if call["name"] == "finish":
            metrics["finish_calls"] += 1
        else:
            metrics["tool_calls"] += 1
        metrics["tool_seconds"] += tool_seconds
        status_counts[observation["status"]] += 1
        protocol_valid = observation["status"] not in {"ERROR", "LIMIT"}
        event.update(tool_seconds=tool_seconds, observation=observation,
                     candidates_after=tools.public_candidates(), protocol_valid=protocol_valid)
        events.append(event)
        if on_event:
            on_event({"phase": "completed", "event": event})

        tool_message = _tool_message(call_id, call["name"], observation, row)
        event_index = len(turns)
        next_candidates = tools.public_candidates()
        active_ids = _active_candidate_ids(note, call, observation, next_candidates, retained_images)
        retained_images, next_image_order = _remember_observation_images(
            observation, event_index, retained_images, next_image_order, MAX_RETAINED_TOOL_PIXELS, active_ids)
        turn = {"assistant_message": assistant_message, "tool_message": tool_message}
        turns.append(turn)
        step += 1
        recovery_pending = False

        if call["name"] == "finish":
            data = observation.get("data") or {}
            if data.get("bbox") is not None and data.get("candidate_id") is not None:
                selected_public_id = str(data["candidate_id"])
                bbox = [float(value) for value in data["bbox"]]
                if selected_public_id not in {str(item["id"]) for item in tools.public_candidates()}:
                    raise ValueError(f"finish returned unknown public candidate ID {selected_public_id}")
                selected_is_initial = bbox == initial_bbox
                if selected_is_initial:
                    selected_id = "KEEP"
                else:
                    selected_id = selected_public_id
                final_status = "FINISHED"
                event["selected_public_id"] = selected_public_id
                event["selected_is_initial"] = selected_is_initial
                break
            metrics["invalid_actions"] += 1
            event["protocol_valid"] = False
            can_recover = (not recovery_feedback_used and
                           profile.output_tokens - metrics["output_tokens"] >= profile.finish_tokens)
            event["recoverable"] = can_recover
            if can_recover:
                recovery_feedback_used = True
                recovery_pending = True
                turn["feedback_message"] = {"role": "user", "content": [{"type": "text",
                    "text": RECOVERY_INSTRUCTION}]}
                continue
            final_status = observation.get("status", "FINISH_FAILED")
            break

    if final_status != "FINISHED":
        selected_id = None
        selected_public_id = None
        bbox = None
        selected_is_initial = None
    metrics.update(elapsed_seconds=time.perf_counter() - started,
                   peak_memory_bytes=backend.peak_memory(), tool_status_counts=dict(status_counts),
                   search_calls=sum((event.get("action") or {}).get("name") == "search" for event in events))
    for key, value in getattr(tools, "cost_counts", {}).items():
        metrics[key] = value
    for key, value in getattr(tools, "cost_seconds", {}).items():
        metrics[key + "_seconds"] = value
    return {
        "schema_version": "visual-agent-object-v1", "prompt_version": PROMPT_VERSION,
        "id": str(row["id"]), "query": row["query"], "image_group": row["images"]["rgb"],
        "initial_messages": initial, "initial_candidates": initial_candidates,
        "initial_bbox": initial_bbox, "events": events,
        "final_status": final_status, "selected_id": selected_id,
        "selected_public_id": selected_public_id, "selected_is_initial": selected_is_initial,
        "bbox": bbox, "final_candidates": tools.public_candidates(), "metrics": metrics,
        "profile": profile.__dict__.copy(),
    }
