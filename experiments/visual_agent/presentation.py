"""Compact model-visible evidence; exact coordinates stay in the candidate table."""
import json


PROMPT_VERSION = "visual-agent-multimodal-v5.1"


def model_candidates(candidates):
    result = []
    for candidate in candidates:
        sources = candidate.get("sources", [])
        item = {"id": str(candidate["id"]), "role": candidate["role"],
                "bbox": [round(float(x), 4) for x in candidate["bbox"]],
                "source_modalities": sorted({s["modality"] for s in sources if "modality" in s})}
        scores = [s["score"] for s in sources if s.get("score") is not None]
        if scores:
            item["detector_score"] = round(max(scores), 3)
        if candidate.get("possible_same_object_ids"):
            item["possible_same_object_ids"] = candidate["possible_same_object_ids"]
        result.append(item)
    return result


def model_observation(observation):
    if observation is None:
        return {"status": "NO_OBSERVATION"}
    data = {key: value for key, value in (observation.get("data") or {}).items() if key != "candidates"}
    return {"status": observation["status"], "text": observation.get("text", ""), "data": data}


def compact_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
