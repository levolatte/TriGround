"""A frozen grounder proposes a box; a separate verifier may keep or replace it.

The proposal is a model prediction, never a target box. KEEP copies its original
text and coordinates exactly. Training this verifier does not update the grounder.
"""
from __future__ import annotations

import json
import math


def revision_prompt(original_prompt: str, proposal: list[float] | None) -> str:
    candidate = None if proposal is None else [v * 1000 for v in proposal]
    return (
        original_prompt
        + "\nReview a previous model prediction for the SAME complete query. "
        + "The proposed RGB box (0-1000 xyxy) is "
        + json.dumps(candidate, separators=(",", ":"))
        + ". It is a model prediction and can be wrong. "
        "Use the complete query and original views to check the target and box. "
        "An auxiliary view need not help; do not change a correct RGB box just because "
        "IR or depth looks different. Follow this review output format instead of the "
        "earlier bbox-only format: return exactly {\"action\":\"keep\"} to retain the "
        "proposal, or {\"action\":\"replace\",\"bbox_2d\":[x1,y1,x2,y2]} with a corrected "
        "RGB box normalized to 0-1000. If the proposal is null, provide a replacement."
    )


def parse_revision(text: str) -> tuple[str, list[float] | None]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return "parse_failure", None
    if value == {"action": "keep"}:
        return "keep", None
    if not isinstance(value, dict) or set(value) != {"action", "bbox_2d"} or value["action"] != "replace":
        return "parse_failure", None
    box = value["bbox_2d"]
    if not isinstance(box, list) or len(box) != 4 or any(
        isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v)
        or not 0 <= v <= 1000 for v in box
    ) or not (box[0] < box[2] and box[1] < box[3]):
        return "parse_failure", None
    return "replace", box


def ground_then_verify(record, prompt, images, ground, verify, parse_bbox, token_cap):
    first = ground(prompt, images, record["prompt_has_image_placeholders"], token_cap)
    proposal = parse_bbox(first["raw_text"])
    review_prompt = revision_prompt(prompt, proposal)
    review = verify(review_prompt, images, record["prompt_has_image_placeholders"], 128)
    action, box = parse_revision(review["raw_text"])
    final = dict(review)
    if action == "keep" and proposal is not None:
        final["raw_text"] = first["raw_text"]
    elif action == "replace":
        final["raw_text"] = json.dumps({"bbox_2d": box}, separators=(",", ":"))
    else:
        # A failed verification is scored as a failure, not silently counted as KEEP.
        action = "parse_failure"
        final["raw_text"] = ""
    return {
        "calls": 2, "final": final, "revision_action": action,
        "baseline_raw_text": first["raw_text"], "baseline_prediction": proposal,
        "baseline_input_tokens": first["input_tokens"],
        "baseline_image_grid_thw": first["image_grid_thw"],
        "baseline_generation_cap_hit": first["generation_cap_hit"],
        "revision_raw_text": review["raw_text"], "final_prompt": review_prompt,
        "first_generated_tokens": first["generated_tokens"],
        "total_latency_seconds": first["latency_seconds"] + review["latency_seconds"],
    }
