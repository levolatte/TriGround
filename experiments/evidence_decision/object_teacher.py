"""One-episode, human-steered tool demonstrations for object-centered TriGround."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from PIL import Image

from tools.predict_aux_selection import index_rows_by_id, load_manifest, read_jsonl

from .object_controller import (
    DECISION_INSTRUCTION,
    FINAL_INSTRUCTION,
    FINISH_SCHEMA,
    MAX_EVIDENCE_CALLS,
    MAX_RETAINED_TOOL_PIXELS,
    OBJECT_PROFILE,
    PROMPT_VERSION,
    TRACE_SCHEMA_VERSION,
    RECOVERY_INSTRUCTION,
    TOOL_SCHEMAS,
    _active_candidate_ids,
    _assistant_tool_message,
    _remember_observation_images,
    _tool_message,
    build_initial_messages,
    build_messages,
    parse_tool_call,
)
from .object_tools import ObjectTools
from .presentation import compact_json
from .vision_tools import CandidatePool


STATE_SCHEMA = "object-teacher-episode-v3"
LEGACY_PREFIX_VERSION = ("visual-agent-object-v2", "paired-evidence-capability-training")
LEGACY_PREFIX_VERSIONS = {
    ("visual-agent-object-v1", "paired-evidence-capability-training"),
    LEGACY_PREFIX_VERSION,
}
NEXT_STEP_INSTRUCTION = DECISION_INSTRUCTION


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _serialize_episode(state: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(state)
    result["retained_images"] = [
        {"candidate_id": key[0], "modality": key[1], "item": item}
        for key, item in result.get("retained_images", {}).items()
    ]
    return result


def _deserialize_episode(state: dict[str, Any]) -> dict[str, Any]:
    retained = state.get("retained_images", {})
    if isinstance(retained, list):
        state["retained_images"] = {
            (item["candidate_id"], item["modality"]): item["item"]
            for item in retained
        }
    return state


def _resolve_paths(row: dict[str, Any], relative_to: Path) -> dict[str, Any]:
    result = copy.deepcopy(row)
    result["images"] = {
        key: str((relative_to / value).resolve()) if value and not Path(value).is_absolute()
        else (str(Path(value).resolve()) if value else None)
        for key, value in row["images"].items()
    }
    return result


def _manifest_row(manifest_path: Path, sample_id: str) -> dict[str, Any]:
    rows = load_manifest(manifest_path, require_images=True)
    matches = [row for row in rows if str(row["id"]) == sample_id]
    if len(matches) != 1:
        raise ValueError(f"sample ID {sample_id!r} must occur exactly once in the manifest")
    row = _resolve_paths(matches[0], manifest_path.parent)
    raw = index_rows_by_id(read_jsonl(manifest_path), source=str(manifest_path))[sample_id]
    for field in ("ir_rgb_registration", "registration_source", "depth_visual_encoding",
                  "depth_encoding", "source", "image_group", "scene_id"):
        if field in raw:
            row[field] = copy.deepcopy(raw[field])
    return row


def _candidate_row(candidate_cache: Path, sample_id: str) -> dict[str, Any]:
    indexed = index_rows_by_id(read_jsonl(candidate_cache), source=str(candidate_cache))
    if sample_id not in indexed:
        raise ValueError(f"sample ID {sample_id!r} is missing from candidate cache")
    row = _resolve_paths(indexed[sample_id], candidate_cache.parent)
    for candidate in row.get("candidates", []):
        if candidate.get("mask_path"):
            mask = Path(candidate["mask_path"])
            if not mask.is_absolute():
                candidate["mask_path"] = str((candidate_cache.parent / mask).resolve())
    return row


def _pool_row(pool: CandidatePool) -> dict[str, Any]:
    snapshot = pool.snapshot()
    return {
        "id": pool.source_identity["id"],
        "images": {
            "rgb": pool.source_identity["rgb"],
            "depth_raw": pool.source_identity["depth_raw"],
        },
        "depth_encoding": pool.source_identity["depth_encoding"],
        **{key: copy.deepcopy(pool.source_identity[key]) for key in (
            "depth_visual_encoding", "ir_rgb_registration", "registration_source",
            "source", "image_group", "scene_id") if key in pool.source_identity},
        "c_bbox": list(pool.c_bbox),
        "query_info": {"scope": pool.scope},
        "candidates": snapshot["source_candidates"],
    }


def _tool_state(tools: ObjectTools) -> dict[str, Any]:
    return {
        "public_to_raw": dict(tools._public_to_raw),
        "action_count": tools.action_count,
        "searches": tools.visual.searches,
        "new_candidates": tools.visual.new_candidates,
        "visual_calls": tools.visual.calls,
        "view_index": tools._view_index,
        "cost_counts": dict(tools.visual.cost_counts),
        "cost_seconds": dict(tools.visual.cost_seconds),
        "observation_cache": [
            {"key": key, "value": copy.deepcopy(value)}
            for key, value in tools._observation_cache.items()
        ],
        "search_cache": [
            {"key": key, "value": copy.deepcopy(value)}
            for key, value in tools._search_cache.items()
        ],
    }


def _tuple_tree(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuple_tree(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _tuple_tree(item)) for key, item in value.items()))
    return value


def _restore_tools(state: dict[str, Any]) -> tuple[ObjectTools, CandidatePool]:
    row = state["row"]
    pool_row = state["pool_row"]
    with Image.open(row["images"]["rgb"]) as source:
        pool = CandidatePool(pool_row, source.size)
    output_dir = Path(state["episode_dir"]) / "observations"
    tools = ObjectTools(row, pool, output_dir, tool_pixels=state["tool_pixels"], seed=state["seed"])
    saved = state.get("tool_state", {})
    mapping = saved.get("public_to_raw")
    if mapping:
        tools._public_to_raw = {str(public): str(raw) for public, raw in mapping.items()}
        tools._raw_to_public = {raw: public for public, raw in tools._public_to_raw.items()}
    tools.action_count = saved.get("action_count", 0)
    tools.visual.searches = saved.get("searches", 0)
    tools.visual.new_candidates = saved.get("new_candidates", 0)
    tools.visual.calls = saved.get("visual_calls", 0)
    tools._view_index = saved.get("view_index", 0)
    tools.visual.cost_counts.update(saved.get("cost_counts", {}))
    tools.visual.cost_seconds.update(saved.get("cost_seconds", {}))
    tools._observation_cache = {
        _tuple_tree(item["key"]): item["value"] for item in saved.get("observation_cache", [])
    }
    tools._search_cache = {
        _tuple_tree(item["key"]): item["value"] for item in saved.get("search_cache", [])
    }
    return tools, pool


def _visible_messages(state: dict[str, Any], tools: ObjectTools, instruction: str = NEXT_STEP_INSTRUCTION):
    if state.get("complete") or state.get("terminal_status"):
        instruction = "This episode is finished."
    elif state.get("recovery_pending"):
        instruction = RECOVERY_INSTRUCTION
    elif tools.action_count >= MAX_EVIDENCE_CALLS:
        instruction = FINAL_INSTRUCTION
    return build_messages(
        state["initial_messages"], state["turns"], tools.public_candidates(),
        state["model_notes"], state["retained_images"], instruction, OBJECT_PROFILE,
    )


def _available_tools(tools: ObjectTools) -> list[dict[str, Any]]:
    return [FINISH_SCHEMA] if tools.action_count >= MAX_EVIDENCE_CALLS else TOOL_SCHEMAS


def _current_tools(state: dict[str, Any], tools: ObjectTools) -> list[dict[str, Any]]:
    return [] if state.get("complete") or state.get("terminal_status") else _available_tools(tools)


def _sync_trace_candidates(tools: ObjectTools, candidates: list[dict[str, Any]]) -> None:
    known = set(tools._public_to_raw)
    for candidate in candidates:
        public_id = str(candidate["id"])
        if public_id in known:
            continue
        sources = candidate.get("sources", [])
        if not sources:
            raise ValueError(f"trace candidate {public_id} has no provenance for pool reconstruction")
        raw_id, is_new = tools.pool.append_detection({
            "bbox": candidate["bbox"], "role": candidate["role"], "source": sources[0],
        })
        if not is_new:
            raise ValueError(f"trace candidate {public_id} merges into an already mapped candidate")
        tools._bind(raw_id, public_id)
        item = tools.pool.get(raw_id)
        for source in sources[1:]:
            if source not in item.setdefault("sources", []):
                item["sources"].append(copy.deepcopy(source))
        known.add(public_id)


def _trace_note(event: dict[str, Any]) -> str:
    raw_output = event.get("raw_output", "")
    start = raw_output.find("<tool_call>")
    end = raw_output.find("</tool_call>", start + len("<tool_call>"))
    if start >= 0 and end >= 0:
        return (raw_output[:start] + raw_output[end + len("</tool_call>"):]).strip()
    return event.get("model_note", "")


def _resume_prefix(state: dict[str, Any], tools: ObjectTools, trace_path: Path,
                   prefix_events: int | None, *, allow_legacy_prefix_migration: bool = False) -> None:
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    trace_version = (trace.get("schema_version"), trace.get("prompt_version"))
    current_version = (TRACE_SCHEMA_VERSION, PROMPT_VERSION)
    is_legacy_migration = trace_version in LEGACY_PREFIX_VERSIONS and allow_legacy_prefix_migration
    if trace_version != current_version and not is_legacy_migration:
        raise ValueError("resume trace does not match the current object-decision history version")
    if str(trace.get("id")) != str(state["sample_id"]):
        raise ValueError("resume trace and manifest sample IDs differ")
    initial = trace.get("initial_candidates")
    if initial:
        raw_candidates = tools.pool.public_candidates()
        unmatched = list(raw_candidates)
        public_to_raw = {}
        for public in initial:
            match = next((item for item in unmatched
                          if item["role"] == public["role"] and item["bbox"] == public["bbox"]), None)
            if match is None:
                raise ValueError(f"trace initial candidate {public['id']} does not match the candidate cache")
            unmatched.remove(match)
            public_to_raw[str(public["id"])] = str(match["id"])
        if unmatched:
            raise ValueError("trace and candidate cache have different initial pool sizes")
        tools._public_to_raw = public_to_raw
        tools._raw_to_public = {raw: public for public, raw in public_to_raw.items()}
    if trace.get("initial_messages") and not is_legacy_migration:
        state["initial_messages"] = copy.deepcopy(trace["initial_messages"])

    events = trace.get("events", [])
    count = len(events) if prefix_events is None else min(max(prefix_events, 0), len(events))
    state["prefix_trace"] = str(trace_path.resolve())
    state["prefix_source_version"] = f"{trace_version[0]}/{trace_version[1]}"
    state["prefix_events"] = [] if is_legacy_migration else copy.deepcopy(events[:count])
    added_ids = set()
    searches = 0
    for index, event in enumerate(events[:count]):
        action = event.get("action")
        real_tool_event = bool(action and isinstance(action, dict) and action.get("name") and
                               "tool_seconds" in event and event.get("observation") is not None)
        has_parsed_action = bool(isinstance(action, dict) and action.get("name"))
        has_assistant_output = bool(event.get("raw_output"))
        if is_legacy_migration and not (real_tool_event or has_parsed_action or has_assistant_output):
            continue
        if is_legacy_migration:
            state["prefix_events"].append(copy.deepcopy(event))
        _sync_trace_candidates(tools, event.get("candidates_after", []))
        observation = event.get("observation") or {
            "status": "ERROR", "text": (event.get("error") or {}).get("message", "Invalid student action."),
            "images": [], "data": {},
        }
        raw_output = event.get("raw_output", "")
        if action and isinstance(action, dict) and action.get("name"):
            # The trace already stores the parsed action; recover its prose
            # without decoding the tool JSON a second time.
            note = _trace_note(event)
            call_id = f"object_call_{event.get('step', index) + 1}"
            assistant_message = _assistant_tool_message(call_id, action, note)
            if "tool_seconds" in event:
                tool_message = _tool_message(call_id, action["name"], observation, state["row"])
            else:
                tool_message = {"role": "user", "content": [{"type": "text",
                    "text": observation.get("text", "Invalid student action.") +
                            (" " + RECOVERY_INSTRUCTION if event.get("recoverable") else "")}]}
            if action["name"] != "finish" and "tool_seconds" in event:
                tools.action_count += 1
            if action["name"] == "search" and "tool_seconds" in event:
                searches += sum(1 for region in observation.get("data", {}).get("regions", [])
                                if region.get("status") in {"OK", "EMPTY"})
            if event.get("recoverable") and action["name"] == "finish":
                turn_feedback = {"role": "user", "content": [{"type": "text",
                    "text": RECOVERY_INSTRUCTION}]}
            else:
                turn_feedback = None
        else:
            note = ""
            assistant_message = {"role": "assistant", "content": raw_output or "Invalid student action."}
            error = observation.get("text", "Invalid student action.")
            feedback = f"Protocol error: {error}"
            if event.get("recoverable"):
                feedback += f". {RECOVERY_INSTRUCTION}"
            tool_message = {"role": "user", "content": [{"type": "text", "text": feedback}]}
            turn_feedback = None
        state["turns"].append({"assistant_message": assistant_message, "tool_message": tool_message})
        if turn_feedback:
            state["turns"][-1]["feedback_message"] = turn_feedback
        if action and isinstance(action, dict):
            state["model_notes"].append(note[:280] if note else "")
        if event.get("recoverable"):
            state["recovery_feedback_used"] = True
            state["recovery_pending"] = True
        elif event.get("action"):
            state["recovery_pending"] = False
            if "tool_seconds" not in event and event.get("recoverable") is False:
                state["terminal_status"] = "INVALID_FINAL_ACTION"
            elif action.get("name") == "finish":
                data = observation.get("data") or {}
                if observation.get("status") == "OK" and data.get("bbox") is not None:
                    state["complete"] = True
                elif event.get("recoverable") is False:
                    state["terminal_status"] = observation.get("status", "FINISH_FAILED")
        elif event.get("recoverable") is False:
            state["terminal_status"] = "INVALID_MODEL_OUTPUT"
        state["retained_images"], state["next_image_order"] = _remember_observation_images(
            observation, len(state["turns"]) - 1, state["retained_images"],
            state["next_image_order"], MAX_RETAINED_TOOL_PIXELS,
            _active_candidate_ids(note, action or {}, observation, tools.public_candidates(),
                                  state["retained_images"]),
        )
        added_ids.update(str(value) for value in observation.get("data", {}).get("appended_ids", []))
    tools.visual.searches = searches
    tools.visual.new_candidates = len(added_ids)
    if "metrics" in trace:
        tools.visual.cost_counts["dino_calls"] = searches
    state["initial_candidates"] = copy.deepcopy(initial or state["initial_candidates"])


def init_episode(
    manifest: Path,
    candidate_cache: Path,
    sample_id: str,
    output_dir: Path,
    *,
    seed: int = 2026,
    tool_pixels: int = OBJECT_PROFILE.tool_pixels,
    dino_model: str | None = None,
    sam_model: str | None = None,
    resume_prefix: Path | None = None,
    prefix_events: int | None = None,
    allow_legacy_prefix_migration: bool = False,
) -> Path:
    manifest, candidate_cache, output_dir = manifest.resolve(), candidate_cache.resolve(), output_dir.resolve()
    sample_id = str(sample_id)
    row = _manifest_row(manifest, sample_id)
    candidate_row = _candidate_row(candidate_cache, sample_id)
    if str(candidate_row["id"]) != sample_id:
        raise ValueError("candidate cache and manifest sample IDs differ")
    if dino_model:
        row["dino_model_path"] = dino_model
    if sam_model:
        row["sam_model_path"] = sam_model
    with Image.open(row["images"]["rgb"]) as image:
        pool = CandidatePool(candidate_row, image.size)

    episode_dir = output_dir
    episode_dir.mkdir(parents=True, exist_ok=True)
    tools = ObjectTools(row, pool, episode_dir / "observations", tool_pixels=tool_pixels, seed=seed)
    modalities = [
        modality for modality, key in (("rgb", "rgb"), ("ir", "ir"), ("depth", "depth_visual"))
        if row.get("images", {}).get(key) and Path(row["images"][key]).exists()
    ]
    atlas = tools.atlas(modalities=modalities or ["rgb"])
    initial = build_initial_messages(row, tools.public_candidates(), atlas, OBJECT_PROFILE)
    state = {
        "schema_version": STATE_SCHEMA,
        "sample_id": sample_id,
        "episode_dir": str(episode_dir),
        "seed": seed,
        "tool_pixels": tool_pixels,
        "row": row,
        "pool_row": _pool_row(pool),
        "initial_candidates": tools.public_candidates(),
        "initial_messages": initial,
        "turns": [],
        "model_notes": [],
        "retained_images": {},
        "next_image_order": 0,
        "prefix_events": [],
        "prefix_source_version": None,
        "steps": [],
        "complete": False,
        "recovery_feedback_used": False,
        "recovery_pending": False,
        "terminal_status": None,
        "finish_source": None,
        "tool_state": _tool_state(tools),
    }
    if resume_prefix:
        _resume_prefix(state, tools, resume_prefix.resolve(), prefix_events,
                       allow_legacy_prefix_migration=allow_legacy_prefix_migration)
        state["pool_row"] = _pool_row(pool)
        state["tool_state"] = _tool_state(tools)
    state["current_messages"] = _visible_messages(state, tools)
    state["current_tools"] = _current_tools(state, tools)
    state_path = episode_dir / "episode.json"
    save_episode(state, state_path)
    return state_path


def save_episode(state: dict[str, Any], episode_path: Path) -> None:
    _write_json(episode_path, _serialize_episode(state))
    _write_json(Path(state["episode_dir"]) / "current_messages.json", state["current_messages"])
    _write_json(Path(state["episode_dir"]) / "current_tools.json", state["current_tools"])
    if state.get("steps"):
        _write_json(Path(state["episode_dir"]) / "latest_observation.json", state["steps"][-1]["observation"])


def _append_recoverable_error(state: dict[str, Any], tools: ObjectTools, *, raw_output: str,
                              error: str, messages: list[dict[str, Any]],
                              candidates: list[dict[str, Any]], action: dict[str, Any] | None = None,
                              note: str = "", native_call: bool = False) -> dict[str, Any]:
    observation = {"status": "ERROR", "text": error, "images": [], "data": {}}
    recoverable = not state.get("recovery_feedback_used")
    if native_call and action:
        call_id = f"teacher_call_{len(state['steps']) + 1}"
        assistant_message = _assistant_tool_message(call_id, action, note)
        feedback = error + (" " + RECOVERY_INSTRUCTION if recoverable else "")
        turn = {"assistant_message": assistant_message,
                "tool_message": {"role": "user", "content": [{"type": "text", "text": feedback}]}}
        state["model_notes"].append(note[:280])
    else:
        feedback = f"Protocol error: {error}. {RECOVERY_INSTRUCTION}" if recoverable else error
        turn = {"assistant_message": {"role": "assistant", "content": raw_output or "Invalid teacher action."},
                "tool_message": {"role": "user", "content": [{"type": "text", "text": feedback}]}}
    state["turns"].append(turn)
    if recoverable:
        state["recovery_feedback_used"] = True
        state["recovery_pending"] = True
    else:
        state["terminal_status"] = "INVALID_FINAL_ACTION" if action else "INVALID_MODEL_OUTPUT"
    step = {
        "index": len(state["steps"]), "note": note, "call": action,
        "raw_output": raw_output, "target_action": raw_output,
        "messages": messages, "tools": copy.deepcopy(_available_tools(tools)),
        "observation": observation, "candidates_before": candidates,
        "candidates_after": tools.public_candidates(), "protocol_valid": False,
        "step_review": {"evidence_support": "pending", "acceptance": "pending"},
        "recoverable": recoverable,
    }
    state["steps"].append(step)
    state["current_messages"] = _visible_messages(state, tools)
    state["current_tools"] = _current_tools(state, tools)
    state["pool_row"] = _pool_row(tools.pool)
    state["tool_state"] = _tool_state(tools)
    save_episode(state, Path(state["episode_dir"]) / "episode.json")
    return step


def step_episode(episode_path: Path, note: str = "", name: str | None = None,
                 arguments: dict[str, Any] | None = None, *, raw_output: str | None = None) -> dict[str, Any]:
    episode_path = episode_path.resolve()
    state = _deserialize_episode(json.loads(episode_path.read_text(encoding="utf-8")))
    if state.get("schema_version") != STATE_SCHEMA:
        raise ValueError("unsupported teacher episode state")
    if state.get("complete") or state.get("terminal_status"):
        raise ValueError("teacher episode is already terminal")
    tools, pool = _restore_tools(state)
    visible_before = _visible_messages(state, tools)
    candidates_before = tools.public_candidates()
    available_tools = _available_tools(tools)
    if raw_output is not None:
        try:
            call, note = parse_tool_call(raw_output)
        except (ValueError, TypeError) as error:
            return _append_recoverable_error(state, tools, raw_output=raw_output, error=str(error),
                                             messages=visible_before, candidates=candidates_before)
    else:
        note = note.strip()
        supported_names = {item["function"]["name"] for item in TOOL_SCHEMAS}
        if name not in supported_names:
            return _append_recoverable_error(state, tools, raw_output="", error="tool call needs a supported name",
                                             messages=visible_before, candidates=candidates_before)
        if not isinstance(arguments, dict):
            return _append_recoverable_error(state, tools, raw_output="",
                                             error="tool call arguments must be an object",
                                             messages=visible_before, candidates=candidates_before)
        call = {"name": name, "arguments": arguments}

    if call["name"] not in {item["function"]["name"] for item in available_tools}:
        return _append_recoverable_error(
            state, tools, raw_output=raw_output or "", error="Only finish is allowed with the remaining budget.",
            messages=visible_before, candidates=candidates_before, action=call, note=note, native_call=True,
        )

    state["recovery_pending"] = False
    name = call["name"]
    arguments = call["arguments"]
    observation = tools.execute(name, arguments)
    call_id = f"teacher_call_{len(state['steps']) + 1}"
    assistant_message = _assistant_tool_message(call_id, call, note)
    tool_message = _tool_message(call_id, name, observation, state["row"])
    event_index = len(state["turns"])
    turn = {"assistant_message": assistant_message, "tool_message": tool_message}
    state["model_notes"].append(note[:280])
    data = observation.get("data") or {}
    active_ids = _active_candidate_ids(note, call, observation, tools.public_candidates(),
                                       state["retained_images"])
    state["retained_images"], state["next_image_order"] = _remember_observation_images(
        observation, event_index, state["retained_images"], state["next_image_order"],
        MAX_RETAINED_TOOL_PIXELS, active_ids,
    )
    target_action = f"{note}\n<tool_call>{compact_json(call)}</tool_call>".strip()
    valid_name = name in {item["function"]["name"] for item in TOOL_SCHEMAS}
    finish_source = data.get("finish_source")
    valid_finish = (
        observation.get("status") == "OK" and data.get("bbox") is not None and (
            (finish_source == "candidate" and data.get("candidate_id") is not None) or
            (finish_source == "predicted_bbox" and data.get("candidate_id") is None)
        )
    )
    protocol_valid = (valid_name and observation["status"] not in {"ERROR", "LIMIT"} and
                      (name != "finish" or valid_finish))
    failed_finish = name == "finish" and not valid_finish
    can_recover = failed_finish and not state.get("recovery_feedback_used")
    if can_recover:
        state["recovery_feedback_used"] = True
        state["recovery_pending"] = True
        turn["feedback_message"] = {"role": "user", "content": [{"type": "text",
            "text": RECOVERY_INSTRUCTION}]}
    elif failed_finish:
        state["terminal_status"] = observation.get("status", "FINISH_FAILED")
    elif protocol_valid:
        state["recovery_pending"] = False
    elif not state.get("recovery_feedback_used"):
        can_recover = True
        state["recovery_feedback_used"] = True
        state["recovery_pending"] = True
        turn["feedback_message"] = {"role": "user", "content": [{"type": "text",
            "text": f"Tool error: {observation.get('text', observation['status'])}. {RECOVERY_INSTRUCTION}"}]}
    else:
        state["terminal_status"] = observation.get("status", "TOOL_FAILED")
    state["turns"].append(turn)
    step = {
        "index": len(state["steps"]),
        "note": note,
        "call": call,
        "target_action": target_action,
        "messages": visible_before,
        "tools": copy.deepcopy(available_tools),
        "observation": observation,
        "candidates_before": candidates_before,
        "candidates_after": tools.public_candidates(),
        "protocol_valid": protocol_valid,
        "step_review": {"evidence_support": "pending", "acceptance": "pending"},
        "recoverable": can_recover,
    }
    state["steps"].append(step)
    if name == "finish" and valid_finish:
        state["complete"] = True
        state["finish_source"] = finish_source
    state["pool_row"] = _pool_row(pool)
    state["tool_state"] = _tool_state(tools)
    state["current_messages"] = _visible_messages(
        state, tools,
        "This episode is finished." if state["complete"] else NEXT_STEP_INSTRUCTION,
    )
    state["current_tools"] = _current_tools(state, tools)
    save_episode(state, episode_path)
    return step


def export_episode(episode_path: Path, output_path: Path | None = None) -> list[dict[str, Any]]:
    episode_path = episode_path.resolve()
    state = json.loads(episode_path.read_text(encoding="utf-8"))
    if state.get("schema_version") != STATE_SCHEMA:
        raise ValueError("unsupported teacher episode state")
    rows = []
    for step in state["steps"]:
        if not step.get("protocol_valid"):
            continue
        messages = copy.deepcopy(step["messages"])
        last_content = messages[-1].get("content")
        if isinstance(last_content, list):
            for part in last_content:
                if part.get("type") == "text" and part.get("text", "").startswith("Teacher: review the visible evidence"):
                    part["text"] = DECISION_INSTRUCTION
        rows.append({
            "schema_version": "visual-agent-object-teacher-v3",
            "id": state["sample_id"],
            "query": state["row"]["query"],
            "image_group": state["row"]["images"]["rgb"],
            "step": step["index"],
            "origin": "teacher",
            "label_source": "gpt-6-luna_max_teacher",
            "supervision_status": "candidate_teacher_supervision",
            "messages": messages,
            "target_action": step["target_action"],
            "tools": step["tools"],
            "is_final": step["call"]["name"] == "finish",
            "finish_source": ((step.get("observation", {}).get("data") or {}).get("finish_source")
                              if step["call"]["name"] == "finish" else None),
            "observation_status": step["observation"]["status"],
            "protocol_valid": bool(step.get("protocol_valid")),
            "step_review": copy.deepcopy(step.get("step_review", {
                "evidence_support": "pending", "acceptance": "pending"})),
        })
    output_path = output_path or (episode_path.parent / "supervision.jsonl")
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows),
                           encoding="utf-8")
    return rows


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="initialize one GT-free episode")
    init.add_argument("--manifest", type=Path, required=True)
    init.add_argument("--candidate-cache", type=Path, required=True)
    init.add_argument("--sample-id", required=True)
    init.add_argument("--output-dir", type=Path, required=True)
    init.add_argument("--seed", type=int, default=2026)
    init.add_argument("--tool-pixels", type=int, default=OBJECT_PROFILE.tool_pixels)
    init.add_argument("--dino-model")
    init.add_argument("--sam-model")
    init.add_argument("--resume-prefix", type=Path)
    init.add_argument("--prefix-events", type=int)
    init.add_argument("--migrate-legacy-prefix", action="store_true",
                      help="explicitly migrate a v2 paired-evidence-capability-training trace")

    step = commands.add_parser("step", help="execute one teacher-selected real tool action")
    step.add_argument("--episode", type=Path, required=True)
    step.add_argument("--raw-output", help="original assistant response containing one native tool call")
    step.add_argument("--note")
    step.add_argument("--name")
    step.add_argument("--arguments", help="JSON object")

    export = commands.add_parser("export", help="write only valid teacher actions as JSONL")
    export.add_argument("--episode", type=Path, required=True)
    export.add_argument("--output", type=Path)
    return root


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    if args.command == "init":
        path = init_episode(args.manifest, args.candidate_cache, args.sample_id, args.output_dir,
                            seed=args.seed, tool_pixels=args.tool_pixels, dino_model=args.dino_model,
                            sam_model=args.sam_model, resume_prefix=args.resume_prefix,
                            prefix_events=args.prefix_events,
                            allow_legacy_prefix_migration=args.migrate_legacy_prefix)
        state = json.loads(path.read_text(encoding="utf-8"))
        print(json.dumps({"status": "ready", "episode": str(path),
                          "messages": str(path.parent / "current_messages.json"),
                          "tools": str(path.parent / "current_tools.json"),
                          "candidate_ids": [item["id"] for item in state["initial_candidates"]]},
                         ensure_ascii=False))
    elif args.command == "step":
        if args.raw_output is not None:
            step = step_episode(args.episode, raw_output=args.raw_output)
        else:
            if args.arguments is None or args.name is None:
                raise ValueError("provide --raw-output or both --name and --arguments")
            arguments = json.loads(args.arguments)
            step = step_episode(args.episode, args.note or "", args.name, arguments)
        print(json.dumps({"status": step["observation"]["status"],
                          "protocol_valid": step["protocol_valid"],
                          "recoverable": step["recoverable"],
                          "observation": {"status": step["observation"]["status"],
                                          "text": step["observation"].get("text"),
                                          "images": [{key: image[key] for key in ("path", "modality", "candidate_ids")
                                                      if key in image}
                                                     for image in step["observation"].get("images", [])]},
                          "observation_file": str(args.episode.resolve().parent / "latest_observation.json"),
                          "messages": str(args.episode.resolve().parent / "current_messages.json"),
                          "tools": str(args.episode.resolve().parent / "current_tools.json")},
                         ensure_ascii=False))
    else:
        rows = export_episode(args.episode, args.output)
        target = args.output or args.episode.resolve().parent / "supervision.jsonl"
        print(json.dumps({"status": "exported", "rows": len(rows), "path": str(target.resolve())},
                         ensure_ascii=False))


if __name__ == "__main__":
    main()
