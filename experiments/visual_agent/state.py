"""The shared, deterministic short-history view for T training and inference.

The public messages contain paths. ``model.prepare_image_messages`` loads their
actual pixels immediately before the Qwen processor sees them.
"""

from __future__ import annotations

from copy import deepcopy
import json
from .presentation import model_candidates, model_observation


GLOBAL_MAX_PIXELS = 200704
TOOL_MAX_PIXELS = 602112


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _observation_summary(observation):
    if observation is None:
        return {"status": "NO_OBSERVATION", "text": "analysis or final decision", "data": {}, "images": []}
    data = model_observation(observation)["data"]
    # Paths and pixel arrays belong only to image blocks. Keep the actual
    # structured numeric observations and candidate changes in text history.
    return {
        "status": observation.get("status"),
        "text": observation.get("text", ""),
        "data": data,
        "images": [
            {key: image[key] for key in ("modality", "candidate_ids", "crop_xyxy", "source_size") if key in image}
            for image in observation.get("images", [])
        ],
    }


def _candidate_summary(candidate):
    result = {key: candidate[key] for key in ("id", "role", "bbox", "is_baseline") if key in candidate}
    if "sources" in candidate:
        result["sources"] = [{key: source[key] for key in ("modality", "score", "category") if key in source}
                             for source in candidate["sources"]]
    return result


def build_state_messages(initial_messages, events, candidates):
    """Keep global images, all short action/result summaries and latest real tool images.

    ``events`` are completed calls, each with ``action`` and ``observation``.
    ``candidates`` is the complete pool visible at the *next* decision. Both
    online T inference and decision-row export must call this same function.
    """
    messages = deepcopy(initial_messages)
    for message in messages:
        for block in message.get("content", []) if isinstance(message.get("content"), list) else []:
            if block.get("type") == "text" and "latest_memory_text" in block:
                block["text"] = block.pop("latest_memory_text")
            elif block.get("type") == "image":
                block["max_pixels"] = GLOBAL_MAX_PIXELS
    summaries = [
        {"step": event.get("step", index), "action": event.get("action"),
         "observation": _observation_summary(event["observation"])}
        for index, event in enumerate(events)
    ]
    content = [{"type": "text", "text":
                "T decision state. All positions and relations refer to the original scene.\n"
                "Completed actions and actual observations: " + _json(summaries) + "\n"
                "Current candidates: " + _json(model_candidates(candidates))}]
    recent = next(((index, event["observation"]) for index, event in reversed(list(enumerate(events)))
                   if event.get("observation") and event["observation"].get("images")), None)
    if recent:
        origin_step, observation = recent
        images = observation["images"]
        per_image_pixels = TOOL_MAX_PIXELS // max(1, len(images))
        for image in images:
            path = image.get("path", image.get("image"))
            if not isinstance(path, str):
                raise ValueError("tool observation image requires a path")
            content.append({"type": "text", "text": "Most recent actual tool image: " +
                            _json({**{key: image[key] for key in ("modality", "candidate_ids", "candidate_roles", "crop_xyxy", "source_size") if key in image},
                                   "origin_step": events[origin_step].get("step", origin_step)})})
            content.append({"type": "image", "image": path,
                            "max_pixels": min(image.get("max_pixels", per_image_pixels), per_image_pixels),
                            "modality": image.get("modality"),
                            "candidate_ids": image.get("candidate_ids"),
                            "view": "tool", "target_boxes": image.get("target_boxes", []),
                            "source_bbox": image.get("source_bbox", {})})
    messages.append({"role": "user", "content": content})
    return messages


def build_initial_messages(row, candidates):
    """Optional GT-free initial view for an independent T runner."""
    images = row["images"]
    if not all(key in images for key in ("rgb", "ir", "depth_visual")):
        raise ValueError("row needs rgb, ir and depth_visual image paths")
    content = [{"type": "text", "text":
                "Ground the full query in the original RGB scene. Use RGB, IR and Depth as aligned scene views. "
                "Return one JSON action. KEEP means the immutable initial box. "
                "Never output a freehand box. Query: " + row["query"] +
                "\nInitial candidates: " + _json(model_candidates(candidates))}]
    for modality, key in (("RGB", "rgb"), ("IR", "ir"), ("Depth", "depth_visual")):
        content.append({"type": "text", "text": f"Global {modality} image:"})
        content.append({"type": "image", "image": images[key],
                        "max_pixels": GLOBAL_MAX_PIXELS, "modality": modality.lower()})
    return [{"role": "user", "content": content}]
