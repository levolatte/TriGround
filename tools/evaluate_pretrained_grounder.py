from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path, PurePosixPath
from statistics import median
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mm_grounding.boxes import box_iou_aligned
from mm_grounding.engine import parse_bbox
from mm_grounding.metrics import grounding_metrics

from tools.native_model_loading import resolve_model_class


EGM_PROMPT = "Locate {query}, output its bbox coordinates using JSON format"
NATIVE_PROMPT = (
    'Locate the object described by this referring expression: "{query}". '
    'Return exactly one JSON object: {"bbox_2d":[x1,y1,x2,y2]}. '
    "Use integer coordinates normalized to the range 0 to 1000."
)
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
FOUR_NUMBERS = rf"({NUMBER})\s*,\s*({NUMBER})\s*,\s*({NUMBER})\s*,\s*({NUMBER})"
IR_FIRST_TOKENS = 64
IR_FINAL_TOKENS = 128
# 这些键允许在历史 run_config 中缺失（缺省即视为旧值），避免新增字段后无法再对旧产物做
# --package-only / --resume；其余键仍严格逐字段比较。
RESUME_OPTIONAL_KEYS = {
    "inference_mode": "direct",
    "auxiliary_adapter": None,
    "model_max_length": 4096,
    "enable_thinking": True,
}


def parse_generated_bbox(text: str) -> list[float] | None:
    """Parse the final EGM-style 0--1000 xyxy box without repairing invalid output."""
    keyed = re.findall(
        rf'["\']?bbox_2d["\']?\s*:\s*\[\s*{FOUR_NUMBERS}\s*\]',
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    arrays = re.findall(rf"\[\s*{FOUR_NUMBERS}\s*\]", text, flags=re.DOTALL)
    candidates = keyed if keyed else arrays
    if not candidates:
        return None
    values = candidates[-1]
    payload = json.dumps({"bbox_2d": [float(value) for value in values]})
    return parse_bbox(payload)


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError("message content must be a string or list")
    parts = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict) and item.get("type") in {"text", "input_text"}:
            parts.append(str(item.get("text", item.get("value", ""))))
    return "\n".join(part for part in parts if part)


def _message_image(message: dict[str, Any]) -> str | None:
    content = message.get("content", "")
    if not isinstance(content, list):
        return None
    for item in content:
        if not isinstance(item, dict) or item.get("type") not in {"image", "image_url"}:
            continue
        value = item.get("image") or item.get("path") or item.get("url")
        if value is None and isinstance(item.get("image_url"), dict):
            value = item["image_url"].get("url")
        if value:
            return str(value)
    return None


def extract_query(user_text: str) -> str:
    text = user_text.replace("<image>", " ").strip()
    patterns = (
        r"<ref>\s*[\"“]?(.+?)[\"”]?\s*</ref>",
        r"Locate the object described by this referring expression:\s*[\"“](.+?)[\"”]\."
        r"\s*Return",
        r"Locate\s+(.+?),\s*output its bbox coordinates using JSON format\s*$",
        r"Locate the object described by this query:\s*(.+?)(?:\nReturn|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    if not text:
        raise ValueError("user message contains no query text")
    return text


def _image_values(
    record: dict[str, Any], user_message: dict[str, Any] | None = None
) -> list[str]:
    for name in ("rgb", "visible", "image"):
        value = record.get(name)
        if isinstance(value, str) and value:
            return [value]
        if isinstance(value, list) and value:
            return [str(item) for item in value]
    images = record.get("images")
    if isinstance(images, str):
        return [images]
    if isinstance(images, list) and images:
        values = []
        for value in images:
            if isinstance(value, dict):
                value = value.get("image") or value.get("path") or value.get("url")
            if not value:
                raise ValueError("images contains an empty path")
            values.append(str(value))
        return values
    value = _message_image(user_message) if user_message is not None else None
    if value:
        return [value]
    raise KeyError("record has no RGB image path")


def _modalities(record: dict[str, Any], images: list[str]) -> list[str] | None:
    values = record.get("modalities", record.get("image_order"))
    if values is None and isinstance(record.get("images"), list):
        entries = record["images"]
        if entries and all(isinstance(item, dict) and "modality" in item for item in entries):
            values = [item["modality"] for item in entries]
    if values is None:
        return None
    if not isinstance(values, list) or len(values) != len(images):
        raise ValueError("modalities must match image order and count")
    aliases = {"rgb": "rgb", "visible": "rgb", "ir": "ir", "infrared": "ir", "depth": "depth"}
    modalities = [aliases[str(value).lower()] for value in values]
    return modalities


def _validated_target(values: Any) -> list[float]:
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError("target bbox must contain four values")
    box = [float(value) for value in values]
    if not bool(torch.isfinite(torch.tensor(box)).all()):
        raise ValueError("target bbox must contain finite values")
    if not all(0.0 <= value <= 1.0 for value in box):
        raise ValueError("target bbox must use normalized 0--1 coordinates")
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError("target bbox must be valid xyxy")
    return box


def normalize_record(sample_id: str, record: dict[str, Any]) -> dict[str, Any]:
    conversations = record.get("conversations")
    if conversations is not None:
        if not isinstance(conversations, list):
            raise ValueError("conversations must be a list")
        human = [message for message in conversations if message.get("from") == "human"]
        assistant = [message for message in conversations if message.get("from") == "gpt"]
        if not human or not assistant:
            raise ValueError("SFT record needs human and gpt conversations")
        prompt = str(human[-1]["value"])
        images = _image_values(record)
        if prompt.count("<image>") != len(images):
            raise ValueError("human prompt image placeholders do not match image paths")
        target = parse_generated_bbox(str(assistant[-1]["value"]))
        if target is None:
            raise ValueError("gpt conversation has no valid 0--1000 bbox")
        return {
            "id": sample_id,
            "images": images,
            "prompt": prompt,
            "prompt_has_image_placeholders": True,
            "images_from_data_root": True,
            "bbox": target,
            "query": str(record["query"]).strip() if "query" in record else extract_query(prompt),
            "modalities": _modalities(record, images),
            "missing_modalities_actual": record.get("missing_modalities_actual", []),
        }

    messages = record.get("messages")
    if messages is None:
        target = _validated_target(record["bbox"])
        query = str(record["query"]).strip()
        if not query:
            raise ValueError("query must not be empty")
        images = _image_values(record)
        return {
            "id": sample_id,
            "images": images,
            "query": query,
            "prompt_has_image_placeholders": False,
            "images_from_data_root": False,
            "bbox": target,
            "modalities": _modalities(record, images),
            "missing_modalities_actual": record.get("missing_modalities_actual", []),
        }

    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    user_messages = [message for message in messages if message.get("role") == "user"]
    assistant_messages = [
        message for message in messages if message.get("role") == "assistant"
    ]
    if not user_messages or not assistant_messages:
        raise ValueError("Qwen record needs user and assistant messages")
    user_message = user_messages[-1]
    assistant_text = _message_text(assistant_messages[-1])
    target = parse_generated_bbox(assistant_text)
    if target is None:
        raise ValueError("assistant message has no valid 0--1000 bbox")
    images = _image_values(record, user_message)
    return {
        "id": sample_id,
        "images": images,
        "query": extract_query(_message_text(user_message)),
        "prompt_has_image_placeholders": False,
        "images_from_data_root": False,
        "bbox": target,
        "modalities": _modalities(record, images),
        "missing_modalities_actual": record.get("missing_modalities_actual", []),
    }


def load_records(manifest: Path) -> list[dict[str, Any]]:
    text = manifest.read_text(encoding="utf-8-sig")
    if manifest.suffix.lower() == ".jsonl":
        payload = _read_jsonl(manifest, text)
    else:
        payload = json.loads(text)
    if isinstance(payload, dict):
        items = payload.items()
    elif isinstance(payload, list):
        items = ((str(record.get("id", index)), record) for index, record in enumerate(payload))
    else:
        raise ValueError("manifest must be a JSON object, JSON array, or JSONL records")
    records = []
    seen_ids: set[str] = set()
    for sample_id, record in items:
        sample_id = str(sample_id)
        if sample_id in seen_ids:
            raise ValueError(f"manifest contains duplicate sample ID: {sample_id}")
        if not isinstance(record, dict):
            raise ValueError(f"manifest record {sample_id} must be an object")
        seen_ids.add(sample_id)
        records.append(normalize_record(sample_id, record))
    if not records:
        raise ValueError("manifest is empty")
    return records


def _read_jsonl(path: Path, text: str) -> list[Any]:
    """Read JSONL without silently dropping a malformed or truncated record."""
    values: list[Any] = []
    lines = text.splitlines()
    nonempty_lines = [index for index, line in enumerate(lines) if line.strip()]
    last_nonempty = nonempty_lines[-1] if nonempty_lines else None
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            values.append(json.loads(line))
        except json.JSONDecodeError as exc:
            kind = "trailing/incomplete" if index == last_nonempty else "invalid"
            raise ValueError(
                f"{path}: {kind} JSONL record at line {index + 1}; "
                "partial records are not ignored"
            ) from exc
    return values


def load_target_manifest(manifest: Path) -> dict[str, list[float]]:
    """Load raw normalized boxes keyed by sample ID for a common evaluation GT."""
    text = manifest.read_text(encoding="utf-8-sig")
    if manifest.suffix.lower() == ".jsonl":
        payload: Any = _read_jsonl(manifest, text)
    else:
        payload = json.loads(text)

    if isinstance(payload, dict):
        items = payload.items()
    elif isinstance(payload, list):
        items = ((record.get("id", index), record) for index, record in enumerate(payload))
    else:
        raise ValueError("target manifest must be a JSON object, JSON array, or JSONL records")

    targets: dict[str, list[float]] = {}
    for sample_id, record in items:
        sample_id = str(sample_id)
        if sample_id in targets:
            raise ValueError(f"target manifest contains duplicate sample ID: {sample_id}")
        if not isinstance(record, dict) or "bbox" not in record:
            raise ValueError(f"target manifest record {sample_id} must contain raw normalized bbox")
        targets[sample_id] = _validated_target(record["bbox"])
    if not targets:
        raise ValueError("target manifest is empty")
    return targets


def apply_target_manifest(
    records: list[dict[str, Any]], target_manifest: Path
) -> list[dict[str, Any]]:
    targets = load_target_manifest(target_manifest)
    record_ids = {str(record["id"]) for record in records}
    missing = sorted(record_ids - set(targets))
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise ValueError(
            f"target manifest does not completely cover evaluation manifest; "
            f"missing IDs: {preview}{suffix}"
        )
    source = str(target_manifest.resolve())
    for record in records:
        record["bbox"] = targets[str(record["id"])]
        record["target_source"] = source
    return records


def effective_prompt_style(records: list[dict[str, Any]], prompt_style: str) -> str:
    return (
        "manifest_conversations"
        if all("prompt" in record for record in records)
        else prompt_style
    )


def build_run_config(
    *,
    model: str,
    adapter: Path | None,
    manifest: Path,
    target_manifest: Path | None,
    data_root: Path,
    min_pixels: int,
    max_pixels: int,
    max_new_tokens: int,
    prompt_style: str,
    limit: int = 0,
    inference_mode: str = "direct",
    ir_reader_adapter: Path | None = None,
) -> dict[str, Any]:
    config = {
        "model": model,
        "adapter": str(adapter) if adapter is not None else None,
        "manifest": str(manifest),
        "target_manifest": str(target_manifest) if target_manifest is not None else None,
        "data_root": str(data_root),
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "max_new_tokens": max_new_tokens,
        "limit": limit,
        "prompt_style": prompt_style,
        "prompt": prompt_style,
        "tf32": False,
        "allow_tf32": False,
        "matmul_precision": "highest",
        "torch_float32_matmul_precision": "highest",
    }
    if inference_mode != "direct":
        config["inference_mode"] = inference_mode
        if inference_mode == "ir_then_ground":
            config["ir_first_max_new_tokens"] = IR_FIRST_TOKENS
            config["ir_final_max_new_tokens"] = IR_FINAL_TOKENS
        else:
            config["revision_max_new_tokens"] = 128
        if ir_reader_adapter is not None:
            config["auxiliary_adapter"] = str(ir_reader_adapter)
            config["grounder_adapter"] = str(adapter)
    return config


def validate_resume_rows(
    rows: list[dict[str, Any]], records: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    expected = {str(record["id"]): record for record in records}
    existing: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"predictions.jsonl record at line {line_number} must be an object")
        if "id" not in row:
            raise ValueError(f"predictions.jsonl record at line {line_number} has no ID")
        sample_id = str(row["id"])
        if sample_id not in expected:
            raise ValueError(f"resume prediction has unknown sample ID: {sample_id}")
        if sample_id in existing:
            raise ValueError(f"resume predictions contain duplicate sample ID: {sample_id}")
        if "target" not in row:
            raise ValueError(f"resume prediction {sample_id} has no target bbox")
        actual_target = _validated_target(row["target"])
        expected_target = _validated_target(expected[sample_id]["bbox"])
        if actual_target != expected_target:
            raise ValueError(
                f"resume target mismatch for sample ID {sample_id}: "
                f"saved={actual_target}, current={expected_target}"
            )
        existing[sample_id] = row
    return existing


def configure_torch_precision() -> None:
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_prediction_rows(path: Path) -> list[dict[str, Any]]:
    values = _read_jsonl(path, path.read_text(encoding="utf-8-sig"))
    if any(not isinstance(value, dict) for value in values):
        raise ValueError(f"{path} must contain JSON object records")
    return values  # type: ignore[return-value]


def ensure_jsonl_append_separator(path: Path, handle: Any) -> None:
    if path.stat().st_size:
        last_byte = path.read_bytes()[-1:]
        if last_byte not in {b"\n", b"\r"}:
            handle.write("\n")


def verify_run_config(saved: dict[str, Any], current: dict[str, Any]) -> None:
    for key, default in RESUME_OPTIONAL_KEYS.items():
        if saved.get(key, default) != current.get(key, default):
            raise ValueError(f"resume run_config mismatch for {key}")
    for key, expected in current.items():
        if key in RESUME_OPTIONAL_KEYS:
            continue
        if key not in saved:
            raise ValueError(f"resume run_config is missing {key}")
        if saved[key] != expected:
            raise ValueError(
                f"resume run_config mismatch for {key}: "
                f"saved={saved[key]!r}, current={expected!r}"
            )


def resolve_image_path(
    value: str,
    data_root: Path,
    manifest: Path,
    *,
    from_data_root: bool = False,
) -> Path:
    normalized = value.replace("\\", "/")
    path = Path(normalized)
    if path.is_absolute():
        return path
    parts = PurePosixPath(normalized).parts
    anchored = parts[:2] == ("external_data", "city_detection_prepared")
    anchored = anchored or parts[:1] == ("city_detection_prepared",)
    base = data_root if from_data_root or anchored else manifest.parent
    return (base.joinpath(*parts)).resolve()


def build_prompt(query: str, style: str) -> str:
    template = {"egm": EGM_PROMPT, "native": NATIVE_PROMPT}[style]
    return template.replace("{query}", query)


def build_user_content(
    prompt: str,
    images: list[Image.Image],
    *,
    prompt_has_image_placeholders: bool,
) -> list[dict[str, Any]]:
    if not prompt_has_image_placeholders:
        return [
            *({"type": "image", "image": image} for image in images),
            {"type": "text", "text": prompt},
        ]

    chunks = prompt.split("<image>")
    if len(chunks) != len(images) + 1:
        raise ValueError("human prompt image placeholders do not match loaded images")
    content: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        if chunk:
            content.append({"type": "text", "text": chunk})
        if index < len(images):
            content.append({"type": "image", "image": images[index]})
    return content


def load_model_images(paths: list[Path]) -> list[Image.Image]:
    images = []
    for path in paths:
        with Image.open(path) as source:
            images.append(source.convert("RGB").copy())
    return images


def parse_ir_reading(text: str) -> tuple[str, list[float] | None]:
    """Strict first-call IR output; coordinates remain in the IR 0--1000 frame."""
    try:
        answer = json.loads(text)
    except json.JSONDecodeError:
        return "reading_parse_failure", None
    if not isinstance(answer, dict) or set(answer) != {"bbox_2d"}:
        return "reading_parse_failure", None
    value = answer["bbox_2d"]
    if value is None:
        return "unknown", None
    if not isinstance(value, list) or len(value) != 4 or any(
        isinstance(number, bool) or not isinstance(number, (int, float)) for number in value
    ):
        return "reading_parse_failure", None
    box = [float(number) for number in value]
    if not all(math.isfinite(number) and 0 <= number <= 1000 for number in box):
        return "reading_parse_failure", None
    if box[0] >= box[2] or box[1] >= box[3]:
        return "reading_parse_failure", None
    return "box", box


def build_ir_first_prompt(query: str) -> str:
    return (
        "Use only this infrared image. Locate the object described by the complete query: "
        + query
        + "\nReturn exactly one JSON object: {\"bbox_2d\":[x1,y1,x2,y2]} "
        "with coordinates in this infrared image normalized to 0-1000. "
        "Give a box only if infrared evidence uniquely identifies the object. "
        "If a necessary color, brand, or other attribute cannot be determined in infrared, "
        "or the object is ambiguous, return {\"bbox_2d\":null}."
    )


def append_ir_guidance(prompt: str, box_ir: list[float]) -> str:
    coordinates = json.dumps(box_ir, separators=(",", ":"))
    return (
        prompt
        + "\nIR model prediction may be wrong. Its suggested box in the infrared image "
        + f"(IR 0-1000 xyxy coordinates) is {coordinates}. "
        "These are infrared-image coordinates, not RGB coordinates. "
        "Use all original views and the complete query to return the final box "
        "in the RGB image, normalized to RGB 0-1000 coordinates."
    )


def ir_image_index(record: dict[str, Any]) -> int | None:
    modalities = record.get("modalities")
    missing = record.get("missing_modalities_actual", [])
    if any(value in {"ir", "infrared"} for value in missing):
        return None
    return modalities.index("ir") if modalities is not None and "ir" in modalities else None


def ground_with_optional_ir(
    record: dict[str, Any],
    prompt: str,
    images: list[Image.Image],
    inference_mode: str,
    generate: Any,
    direct_max_new_tokens: int,
    *,
    read_ir: Any = None,
) -> dict[str, Any]:
    """Generate at most once per stage; generate(prompt, images, placeholders, cap)."""
    if inference_mode == "ir_then_ground" and record.get("modalities") is None and len(images) > 1:
        raise ValueError("ir_then_ground requires explicit modalities for multi-image records")
    ir_index = ir_image_index(record) if inference_mode == "ir_then_ground" else None
    first_raw_text = None
    first_box_ir = None
    first_status = "no_ir" if inference_mode == "ir_then_ground" else "not_requested"
    first = None
    final_prompt = prompt
    final_cap = direct_max_new_tokens
    if ir_index is not None:
        first = (read_ir or generate)(build_ir_first_prompt(record["query"]), [images[ir_index]], False, IR_FIRST_TOKENS)
        first_raw_text = first["raw_text"]
        first_status, first_box_ir = parse_ir_reading(first_raw_text)
        if first_status == "box":
            final_prompt = append_ir_guidance(prompt, first_box_ir)
        final_cap = IR_FINAL_TOKENS
    final = generate(final_prompt, images, record["prompt_has_image_placeholders"], final_cap)
    return {
        "first_raw_text": first_raw_text,
        "first_box_ir": first_box_ir,
        "first_status": first_status,
        "calls": 1 + int(first is not None),
        "total_latency_seconds": final["latency_seconds"] + (first["latency_seconds"] if first is not None else 0.0),
        "first_generated_tokens": first["generated_tokens"] if first is not None else 0,
        "first_generation_cap_hit": first["generation_cap_hit"] if first is not None else False,
        "final_prompt": final_prompt,
        "final": final,
    }


def score_prediction(
    prediction: list[float] | None, target: list[float]
) -> tuple[float, bool]:
    if prediction is None:
        return 0.0, False
    predicted = torch.tensor([prediction], dtype=torch.float32)
    expected = torch.tensor([target], dtype=torch.float32)
    iou = float(box_iou_aligned(predicted, expected)[0].item())
    return iou, iou >= 0.5


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("cannot summarize an empty prediction set")
    predictions = [row["prediction"] or [0.0, 0.0, 0.0, 0.0] for row in rows]
    targets = [row["target"] for row in rows]
    metrics = grounding_metrics(
        torch.tensor(predictions, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
    )
    parsed = sum(row["parsed"] for row in rows)
    hits = 0
    for row in rows:
        if "hit" in row:
            hits += bool(row["hit"])
        elif "acc_0.5" in row:
            hits += bool(row["acc_0.5"])
        elif "iou" in row:
            hits += float(row["iou"]) >= 0.5
        else:
            hits += score_prediction(row["prediction"], row["target"])[1]
    latencies = [float(row["latency_seconds"]) for row in rows]
    tokens = [int(row["generated_tokens"]) for row in rows]
    summary = {
        "samples": len(rows),
        "parsed": parsed,
        "parse_failures": len(rows) - parsed,
        "parse_rate": parsed / len(rows),
        "hits": hits,
        **metrics,
        "total_generated_tokens": sum(tokens),
        "mean_generated_tokens": sum(tokens) / len(tokens),
        "total_generation_seconds": sum(latencies),
        "mean_generation_seconds": sum(latencies) / len(latencies),
        "median_generation_seconds": median(latencies),
        "generation_cap_hits": sum(row["generation_cap_hit"] for row in rows),
    }
    if any("first_status" in row for row in rows):
        summary.update({
            "ir_available": sum(row.get("first_status") != "no_ir" for row in rows),
            "ir_box": sum(row.get("first_status") == "box" for row in rows),
            "ir_unknown": sum(row.get("first_status") == "unknown" for row in rows),
            "reading_parse_failures": sum(row.get("first_status") == "reading_parse_failure" for row in rows),
            "first_generation_cap_hits": sum(bool(row.get("first_generation_cap_hit", False)) for row in rows),
            "total_calls": sum(int(row.get("calls", 1)) for row in rows),
            "total_latency_seconds": sum(float(row.get("total_latency_seconds", row["latency_seconds"])) for row in rows),
            "total_call_generated_tokens": sum(
                int(row["generated_tokens"]) + int(row.get("first_generated_tokens", 0)) for row in rows
            ),
        })
    if any("revision_action" in row for row in rows):
        baseline_hits = [score_prediction(row["baseline_prediction"], row["target"])[1] for row in rows]
        final_hits = [score_prediction(row["prediction"], row["target"])[1] for row in rows]
        rescued = sum(not old and new for old, new in zip(baseline_hits, final_hits, strict=True))
        harmed = sum(old and not new for old, new in zip(baseline_hits, final_hits, strict=True))
        summary.update({
            "revision_keep": sum(row["revision_action"] == "keep" for row in rows),
            "revision_replace": sum(row["revision_action"] == "replace" for row in rows),
            "revision_parse_failures": sum(row["revision_action"] == "parse_failure" for row in rows),
            "total_calls": sum(row["calls"] for row in rows),
            "total_latency_seconds": sum(row["total_latency_seconds"] for row in rows),
            "total_call_generated_tokens": sum(row["generated_tokens"] + row["first_generated_tokens"] for row in rows),
            "revision_retention": {
                "samples": len(rows), "baseline_hits": sum(baseline_hits),
                "final_hits": sum(final_hits), "rescued": rescued, "harmed": harmed,
                "net_hits": rescued-harmed,
                "correct_proposals_replaced": sum(hit and row["revision_action"] == "replace"
                                                  for hit, row in zip(baseline_hits, rows, strict=True)),
                "wrong_proposals_kept": sum(not hit and row["revision_action"] == "keep"
                                            for hit, row in zip(baseline_hits, rows, strict=True)),
            },
        })
    return summary


def attach_frozen_ir_reader(model: Any, adapter: Path) -> None:
    """A separate reader LoRA shares only the frozen base with the grounder."""
    model.load_adapter(str(adapter), adapter_name="ir_reader", is_trainable=False,
                       autocast_adapter_dtype=False, local_files_only=True)
    model.set_adapter("default")
    model.requires_grad_(False)
    model.eval()


def adapter_generator(model: Any, processor: Any, adapter_name: str):
    def generate(prompt, images, placeholders, cap):
        # PEFT set_adapter can enable gradients; both inference roles stay frozen.
        model.set_adapter(adapter_name)
        model.requires_grad_(False)
        model.eval()
        return generate_text(model, processor, prompt, images, placeholders, cap)
    return generate


def load_model_and_processor(model_name: str, adapter: Path | None, min_pixels: int,
                             max_pixels: int, *, ir_reader_adapter: Path | None = None,
                             model_max_length: int | None = None):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for pretrained grounder evaluation")
    configure_torch_precision()
    processor = AutoProcessor.from_pretrained(
        model_name, min_pixels=min_pixels, max_pixels=max_pixels, local_files_only=True,
    )
    if model_max_length is not None:
        processor.tokenizer.model_max_length = model_max_length
    model_class, _ = resolve_model_class(model_name, local_files_only=True)
    model = model_class.from_pretrained(
        model_name, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True,
    )
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model, adapter, is_trainable=False, autocast_adapter_dtype=False,
            local_files_only=True,
        )
    if ir_reader_adapter is not None:
        if adapter is None:
            raise ValueError("a separate IR reader requires an explicit retained grounder adapter")
        attach_frozen_ir_reader(model, ir_reader_adapter)
    model = model.eval().to(device="cuda", dtype=torch.bfloat16)
    torch.cuda.reset_peak_memory_stats()
    return model, processor


def generate_text(model: Any, processor: Any, prompt: str, images: list[Image.Image],
                  placeholders: bool, max_new_tokens: int) -> dict[str, Any]:
    messages = [{"role": "user", "content": build_user_content(
        prompt, images, prompt_has_image_placeholders=placeholders,
    )}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    inputs = inputs.to("cuda")
    torch.cuda.synchronize()
    started = time.perf_counter()
    generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    torch.cuda.synchronize()
    latency = time.perf_counter() - started
    new_tokens = generated[0, inputs["input_ids"].shape[1]:]
    raw_text = processor.decode(
        new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    ).strip()
    eos = processor.tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    return {
        "raw_text": raw_text,
        "generated_tokens": int(len(new_tokens)),
        "input_tokens": int(inputs["input_ids"].shape[1]),
        "image_grid_thw": inputs["image_grid_thw"].detach().cpu().tolist(),
        "generation_cap_hit": len(new_tokens) >= max_new_tokens and not any(
            int(token) in eos_ids for token in new_tokens
        ),
        "latency_seconds": latency,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a local Qwen3-VL RGB or multi-image grounding checkpoint."
    )
    parser.add_argument("--model", default="nvidia/EGM-8B")
    parser.add_argument("--adapter", type=Path, help="Local PEFT adapter directory")
    parser.add_argument("--aux-adapter", "--ir-reader-adapter", dest="ir_reader_adapter", type=Path,
                        help="Separate auxiliary LoRA; --adapter remains the retained grounder")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--target-manifest",
        type=Path,
        help="Raw ID-to-record manifest whose normalized bbox is the common evaluation GT",
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume predictions.jsonl by sample ID and preserve completed rows",
    )
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates the full manifest")
    parser.add_argument("--min-pixels", type=int, default=200704)
    parser.add_argument("--max-pixels", type=int, default=602112)
    parser.add_argument("--model-max-length", type=int, default=4096,
                        help="处理器 tokenizer 的最大长度；高分辨率档需相应提高（原生三图约 6075 token）")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--prompt-style", choices=("egm", "native"), default="egm")
    parser.add_argument("--inference-mode", choices=("direct", "ir_then_ground", "ground_then_verify"), default="direct")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")
    if args.min_pixels <= 0 or args.max_pixels <= 0 or args.max_new_tokens <= 0:
        raise ValueError("pixel and token limits must be positive")
    if args.min_pixels > args.max_pixels:
        raise ValueError("--min-pixels must not exceed --max-pixels")

    manifest = args.manifest.resolve()
    target_manifest = (
        args.target_manifest.resolve() if args.target_manifest is not None else None
    )
    data_root = args.data_root.resolve()
    adapter = args.adapter.resolve() if args.adapter is not None else None
    ir_reader_adapter = args.ir_reader_adapter.resolve() if args.ir_reader_adapter is not None else None
    if ir_reader_adapter is not None:
        if args.inference_mode == "direct" or adapter is None:
            raise ValueError("--aux-adapter requires a two-call mode and an explicit --adapter")
        if not (ir_reader_adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(f"IR reader adapter is missing: {ir_reader_adapter}")
    if args.inference_mode == "ground_then_verify" and ir_reader_adapter is None:
        raise ValueError("ground_then_verify requires a separate --aux-adapter")
    if adapter is not None and not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"PEFT adapter_config.json not found: {adapter}")
    all_records = load_records(manifest)
    records = all_records[: args.limit] if args.limit else all_records
    if target_manifest is not None:
        apply_target_manifest(records, target_manifest)
    target_source = str(target_manifest or manifest)
    if target_manifest is not None:
        target_source_kind = "target_manifest_bbox"
    elif all("prompt" in record for record in records):
        target_source_kind = "manifest_conversation_gpt"
    else:
        target_source_kind = "manifest_bbox"
    for record in records:
        record.setdefault("target_source", target_source)
        record.setdefault("target_source_kind", target_source_kind)
    prompt_style = effective_prompt_style(records, args.prompt_style)
    prepared = []
    for record in records:
        prompt = record.get("prompt") or build_prompt(record["query"], args.prompt_style)
        image_paths = [
            resolve_image_path(
                value,
                data_root,
                manifest,
                from_data_root=record["images_from_data_root"],
            )
            for value in record["images"]
        ]
        prepared.append((record, prompt, image_paths))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"
    config_path = output_dir / "run_config.json"
    run_config = build_run_config(
        model=args.model,
        adapter=adapter,
        manifest=manifest,
        target_manifest=target_manifest,
        data_root=data_root,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        max_new_tokens=args.max_new_tokens,
        prompt_style=prompt_style,
        limit=args.limit,
        inference_mode=args.inference_mode,
        ir_reader_adapter=ir_reader_adapter,
    )

    existing_by_id: dict[str, dict[str, Any]] = {}
    if args.resume:
        if rows_path.exists() and not config_path.exists():
            raise FileNotFoundError(
                f"cannot resume {rows_path}: run_config.json is missing"
            )
        if config_path.exists():
            saved_config = json.loads(config_path.read_text(encoding="utf-8-sig"))
            if not isinstance(saved_config, dict):
                raise ValueError(f"{config_path} must contain a JSON object")
            verify_run_config(saved_config, run_config)
        if rows_path.exists():
            existing_by_id = validate_resume_rows(
                load_prediction_rows(rows_path), records
            )
        elif summary_path.exists():
            raise FileNotFoundError(
                f"cannot resume {output_dir}: summary.json exists but predictions.jsonl is missing"
            )
    elif any(path.exists() for path in (rows_path, summary_path, config_path)):
        raise FileExistsError(
            f"output directory already contains evaluation results: {output_dir}; "
            "use --resume to continue"
        )

    if not config_path.exists():
        config_path.write_text(
            json.dumps(run_config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    missing = [record for record, _, _ in prepared if str(record["id"]) not in existing_by_id]
    if not missing:
        ordered_rows = [existing_by_id[str(record["id"])] for record in records]
        previous_summary = (
            json.loads(summary_path.read_text(encoding="utf-8-sig"))
            if summary_path.exists()
            else {}
        )
        summary: dict[str, Any] = {
            **run_config,
            "manifest_samples": len(all_records),
            "limit": args.limit,
            "target_source": target_source,
            "target_source_kind": target_source_kind,
            **summarize_rows(ordered_rows),
            "gpu_peak_allocated_bytes": int(
                previous_summary.get("gpu_peak_allocated_bytes", 0)
            ),
            "gpu_peak_reserved_bytes": int(
                previous_summary.get("gpu_peak_reserved_bytes", 0)
            ),
            "predictions": str(rows_path),
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print("FINAL " + json.dumps(summary, ensure_ascii=False), flush=True)
        return

    model, processor = load_model_and_processor(
        args.model, adapter, args.min_pixels, args.max_pixels,
        model_max_length=args.model_max_length,
        **({"ir_reader_adapter": ir_reader_adapter} if ir_reader_adapter is not None else {}),
    )
    ground = (adapter_generator(model, processor, "default") if ir_reader_adapter is not None
              else lambda text, views, placeholders, cap: generate_text(model, processor, text, views, placeholders, cap))
    reader = adapter_generator(model, processor, "ir_reader") if ir_reader_adapter is not None else None
    rows: list[dict[str, Any]] = list(existing_by_id.values())
    file_mode = "a" if rows_path.exists() else "w"
    with rows_path.open(file_mode, encoding="utf-8") as handle, torch.inference_mode():
        if file_mode == "a":
            ensure_jsonl_append_separator(rows_path, handle)
        for index, (record, prompt, image_paths) in enumerate(prepared, 1):
            sample_id = str(record["id"])
            if sample_id in existing_by_id:
                continue
            images = load_model_images(image_paths)
            if args.inference_mode == "ground_then_verify":
                from tools.grounding_revision import ground_then_verify
                result = ground_then_verify(record, prompt, images, ground, reader,
                                            parse_generated_bbox, args.max_new_tokens)
            else:
                result = ground_with_optional_ir(
                    record, prompt, images, args.inference_mode,
                    ground, args.max_new_tokens, read_ir=reader,
                )
            final = result["final"]
            raw_text = final["raw_text"]
            prediction = parse_generated_bbox(raw_text)
            iou, accurate = score_prediction(prediction, record["bbox"])
            row = {
                "index": index - 1,
                "id": record["id"],
                "image": (
                    str(image_paths[0])
                    if len(image_paths) == 1
                    else [str(path) for path in image_paths]
                ),
                "query": record.get("query"),
                "prompt": prompt,
                "target": record["bbox"],
                "target_source": record["target_source"],
                "target_source_kind": record["target_source_kind"],
                "prediction": prediction,
                "parsed": prediction is not None,
                "iou": iou,
                "acc_0.5": accurate,
                "hit": accurate,
                "generated_tokens": final["generated_tokens"],
                "input_tokens": final["input_tokens"],
                "image_grid_thw": final["image_grid_thw"],
                "generation_cap_hit": final["generation_cap_hit"],
                "latency_seconds": final["latency_seconds"],
                "raw_text": raw_text,
            }
            if args.inference_mode == "ir_then_ground":
                row.update({key: result[key] for key in (
                    "first_raw_text", "first_box_ir", "first_status", "calls",
                    "total_latency_seconds", "first_generated_tokens", "first_generation_cap_hit",
                    "final_prompt",
                )})
            elif args.inference_mode == "ground_then_verify":
                row.update({key: result[key] for key in (
                    "baseline_raw_text", "baseline_prediction", "revision_raw_text",
                    "revision_action", "calls", "total_latency_seconds", "first_generated_tokens",
                    "baseline_input_tokens", "baseline_image_grid_thw", "baseline_generation_cap_hit",
                    "final_prompt",
                )})
            rows.append(row)
            existing_by_id[sample_id] = row
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                json.dumps(
                    {
                        "done": len(existing_by_id),
                        "total": len(records),
                        "id": sample_id,
                        "parsed": row["parsed"],
                        "iou": iou,
                        "tokens": row["generated_tokens"],
                        "seconds": result["total_latency_seconds"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    ordered_rows = [existing_by_id[str(record["id"])] for record in records]
    summary: dict[str, Any] = {
        **run_config,
        "manifest_samples": len(all_records),
        "limit": args.limit,
        "target_source": target_source,
        "target_source_kind": target_source_kind,
        **summarize_rows(ordered_rows),
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "predictions": str(rows_path),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("FINAL " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
