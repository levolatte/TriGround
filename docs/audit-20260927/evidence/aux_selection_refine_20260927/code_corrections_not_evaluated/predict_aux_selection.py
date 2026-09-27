"""Run the frozen C auxiliary query parsing, candidate selection and refinement.

The module-level helpers use only Python and Pillow so the coordinate, ID and
prompt rules can be checked on CPU. PyTorch, Transformers and PEFT are imported
only when a GPU inference subcommand starts.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


MIN_PIXELS = 200704
MAX_PIXELS = 602112
MAX_NEW_TOKENS = 128
SCHEMA_VERSION = "aux-selection-v1"
BASELINE_SCHEMA_VERSION = "aux-baseline-v1"
RELATION_TYPES = {"camera_near", "camera_far", "other", "none"}
SCOPES = {"single", "group", "part", "unclear"}
IMAGE_FIELDS = ("rgb", "ir", "depth_visual", "depth_raw")
PROMPT_VERSION = "frozen-C-select-refine-v1"
QUERY_PARSE_PROMPT_VERSION = "query-parse-v2"
CROP_POLICY_VERSION = "centered-clip-v2-not-gpu-evaluated"
PALETTE = (
    "#ff3b30", "#00a6ff", "#34c759", "#ff9500", "#af52de", "#00c7be",
    "#ff2d55", "#8e8e93", "#ffd60a", "#5e5ce6", "#30d158", "#64d2ff",
)


def valid_box(value: object) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return False
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        return False
    if not all(math.isfinite(float(item)) for item in value):
        return False
    x1, y1, x2, y2 = (float(item) for item in value)
    return 0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0


def require_box(value: object, *, field: str = "bbox") -> list[float]:
    if not valid_box(value):
        raise ValueError(f"{field} must be finite normalized xyxy in [0, 1]")
    return [float(item) for item in value]  # type: ignore[arg-type]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: each JSONL record must be an object")
        rows.append(value)
    return rows


def write_jsonl_row(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    handle.flush()


def load_manifest(path: Path, *, require_images: bool) -> list[dict[str, Any]]:
    """Read only public query fields; never accept a target box or class label."""
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for line_number, source in enumerate(rows, 1):
        if "id" not in source or "query" not in source:
            raise ValueError(f"{path}:{line_number}: manifest row needs id and query")
        sample_id = str(source["id"])
        if sample_id in seen:
            raise ValueError(f"manifest contains duplicate ID: {sample_id}")
        seen.add(sample_id)
        if any(key in source for key in ("bbox", "gt", "target_bbox", "class_name")):
            raise ValueError(f"{path}:{line_number}: GT/class_name fields are forbidden")
        query = source["query"]
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"{path}:{line_number}: query must be nonempty text")
        row: dict[str, Any] = {"id": source["id"], "_id": sample_id, "query": query}
        if require_images:
            images = source.get("images")
            if not isinstance(images, dict):
                raise ValueError(f"{path}:{line_number}: images must be an object")
            row["images"] = {key: images.get(key) for key in IMAGE_FIELDS}
            row["depth_encoding"] = source.get("depth_encoding")
        result.append(row)
    return result


def index_rows_by_id(rows: list[dict[str, Any]], *, source: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(rows, 1):
        if "id" not in row:
            raise ValueError(f"{source}:{line_number}: row has no id")
        sample_id = str(row["id"])
        if sample_id in indexed:
            raise ValueError(f"{source}:{line_number}: duplicate id {sample_id}")
        indexed[sample_id] = row
    return indexed


def parse_query_json(raw_text: str) -> dict[str, Any] | None:
    """Parse the exact C text-only JSON schema without repairing model output."""
    try:
        value = json.loads(raw_text.strip())
    except json.JSONDecodeError:
        return None
    required = {
        "target_category", "reference_categories", "relation_type", "scope",
    }
    if not isinstance(value, dict) or set(value) != required:
        return None
    target = value["target_category"]
    references = value["reference_categories"]
    relation = value["relation_type"]
    scope = value["scope"]
    if not isinstance(target, str) or not target.strip():
        return None
    if not isinstance(references, list) or len(references) > 2:
        return None
    if any(not isinstance(item, str) or not item.strip() for item in references):
        return None
    if not isinstance(relation, str) or relation not in RELATION_TYPES:
        return None
    if not isinstance(scope, str) or scope not in SCOPES:
        return None
    target_category = target.strip()
    references = [item.strip() for item in references
                  if item.strip().casefold() != target_category.casefold()]
    return {
        "target_category": target_category,
        "reference_categories": references,
        "relation_type": relation,
        "scope": scope,
    }


def parse_selection(raw_text: str, candidate_ids: set[int]) -> tuple[str, int | None] | None:
    text = raw_text.strip()
    if text == "KEEP":
        return "keep", None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return ("candidate", value) if value in candidate_ids else None
    if not isinstance(value, dict) or set(value) != {"id"}:
        return None
    selected_id = value["id"]
    if isinstance(selected_id, bool) or not isinstance(selected_id, int):
        return None
    if selected_id not in candidate_ids:
        return None
    return "candidate", selected_id


def parse_crop_bbox(raw_text: str) -> list[float] | str | None:
    text = raw_text.strip()
    if text == "KEEP":
        return "KEEP"
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict) or set(value) != {"bbox_2d"}:
        return None
    values = value["bbox_2d"]
    if not isinstance(values, list) or len(values) != 4:
        return None
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in values):
        return None
    values = [float(item) for item in values]
    if not all(math.isfinite(item) and 0.0 <= item <= 1000.0 for item in values):
        return None
    x1, y1, x2, y2 = values
    if x1 >= x2 or y1 >= y2:
        return None
    return [item / 1000.0 for item in values]


def candidate_index(candidates: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for candidate in candidates:
        candidate_id = candidate.get("id")
        if isinstance(candidate_id, bool) or not isinstance(candidate_id, int):
            raise ValueError("candidate id must be an integer")
        if candidate_id in indexed:
            raise ValueError(f"duplicate candidate id: {candidate_id}")
        role = candidate.get("role")
        is_baseline = candidate.get("is_baseline")
        if not isinstance(role, str) or role not in {"target", "reference"}:
            raise ValueError(f"candidate {candidate_id}: role must be target or reference")
        if not isinstance(is_baseline, bool):
            raise ValueError(f"candidate {candidate_id}: is_baseline must be boolean")
        candidate_copy = {
            "id": candidate_id,
            "bbox": require_box(candidate.get("bbox"), field=f"candidate {candidate_id} bbox"),
            "role": role,
            "is_baseline": is_baseline,
        }
        indexed[candidate_id] = candidate_copy
    return indexed


def select_bbox(selection: tuple[str, int | None], candidates: list[dict[str, Any]], c_bbox: object) -> list[float]:
    kind, candidate_id = selection
    if kind == "keep":
        return require_box(c_bbox, field="c_bbox")
    if kind != "candidate" or candidate_id is None:
        raise ValueError(f"unknown selection kind: {kind}")
    # Bind by candidate ID; candidate list order is deliberately irrelevant.
    chosen = candidate_index(candidates)[candidate_id]
    if chosen["role"] != "target":
        raise ValueError(f"reference candidate {candidate_id} cannot be selected as the target")
    return chosen["bbox"]


def selection_changed_from_c(selection: tuple[str, int | None], candidates: list[dict[str, Any]]) -> bool:
    kind, candidate_id = selection
    if kind == "keep":
        return False
    if kind != "candidate" or candidate_id is None:
        raise ValueError(f"unknown selection kind: {kind}")
    candidate = candidate_index(candidates)[candidate_id]
    if candidate["role"] != "target":
        raise ValueError(f"reference candidate {candidate_id} cannot be selected as the target")
    return not candidate["is_baseline"]


def bbox_min_side(box: list[float]) -> float:
    return min(box[2] - box[0], box[3] - box[1])


def needs_refine(selection_kind: str, box: list[float], *, changed_from_c: bool = False) -> bool:
    return changed_from_c or bbox_min_side(box) < 0.06


def make_crop_transform(
    box: list[float], image_width: int, image_height: int,
    *, scale: float = 2.0, minimum_side: float = 0.12,
) -> tuple[tuple[int, int, int, int], list[float]]:
    """Return pixel crop and its exact normalized bounds in the source image."""
    box = require_box(box)
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    x1, y1, x2, y2 = box
    crop_width = max((x2 - x1) * scale, minimum_side)
    crop_height = max((y2 - y1) * scale, minimum_side)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    left = max(center_x - crop_width / 2, 0.0)
    top = max(center_y - crop_height / 2, 0.0)
    right = min(center_x + crop_width / 2, 1.0)
    bottom = min(center_y + crop_height / 2, 1.0)
    pixel_left = max(0, min(image_width - 1, math.floor(left * image_width)))
    pixel_top = max(0, min(image_height - 1, math.floor(top * image_height)))
    pixel_right = max(pixel_left + 1, min(image_width, math.ceil(right * image_width)))
    pixel_bottom = max(pixel_top + 1, min(image_height, math.ceil(bottom * image_height)))
    transform = [
        pixel_left / image_width,
        pixel_top / image_height,
        pixel_right / image_width,
        pixel_bottom / image_height,
    ]
    return (pixel_left, pixel_top, pixel_right, pixel_bottom), transform


def map_crop_bbox_to_original(crop_box: list[float], transform: list[float]) -> list[float]:
    crop_box = require_box(crop_box, field="crop bbox")
    transform = require_box(transform, field="crop transform")
    left, top, right, bottom = transform
    mapped = [
        left + crop_box[0] * (right - left),
        top + crop_box[1] * (bottom - top),
        left + crop_box[2] * (right - left),
        top + crop_box[3] * (bottom - top),
    ]
    return require_box(mapped, field="mapped bbox")


def _pixel_bbox(box: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, min(width - 1, round(x1 * width))),
        max(0, min(height - 1, round(y1 * height))),
        max(0, min(width, round(x2 * width))),
        max(0, min(height, round(y2 * height))),
    )


def annotate_candidates(
    image: Image.Image,
    candidates: list[dict[str, Any]],
    *,
    selected_id: int | None = None,
    keep_box: list[float] | None = None,
) -> Image.Image:
    """Draw only anonymized candidate numbers; source and baseline flags are ignored."""
    image = image.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    width, height = image.size
    indexed = candidate_index(candidates)
    font_size = max(18, round(max(width, height) * 0.016))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=font_size)
    except OSError:
        font = ImageFont.truetype("arial.ttf", size=font_size)
    line_width = max(1, round(max(width, height) * 0.0015))
    for candidate_id, candidate in sorted(indexed.items()):
        color = PALETTE[candidate_id % len(PALETTE)]
        coords = _pixel_bbox(candidate["bbox"], width, height)
        current_width = line_width * (2 if candidate_id == selected_id else 1)
        draw.rectangle(coords, outline=color, width=current_width)
        label = str(candidate_id)
        x = coords[0]
        text_box = draw.textbbox((0, 0), label, font=font)
        pad = max(2, line_width)
        label_width = text_box[2] + pad * 2
        x = min(x, max(0, width - label_width))
        label_height = text_box[3] - text_box[1] + pad * 2
        y = coords[1] - label_height if coords[1] >= label_height else coords[3]
        if y + label_height > height:
            y = max(0, coords[1])
        label_rect = (x, y, x + label_width, y + label_height)
        draw.rectangle(label_rect, fill=color)
        draw.text((x + pad, y + pad), label, fill="black", font=font)
    if keep_box is not None:
        coords = _pixel_bbox(require_box(keep_box, field="c_bbox"), width, height)
        color = "#ffffff"
        draw.rectangle(coords, outline=color, width=line_width * 2)
        label = "KEEP"
        x, y = coords[0], coords[1]
        text_box = draw.textbbox((0, 0), label, font=font)
        pad = max(2, line_width)
        label_width = text_box[2] + pad * 2
        x = min(x, max(0, width - label_width))
        label_height = text_box[3] - text_box[1] + pad * 2
        label_y = y - label_height if y >= label_height else coords[3]
        if label_y + label_height > height:
            label_y = max(0, y)
        draw.rectangle((x, label_y, x + label_width, label_y + label_height), fill=color)
        draw.text((x + pad, label_y + pad), label, fill="black", font=font)
    return image


def _rounded_coords(box: list[float]) -> list[int]:
    return [round(value * 1000) for value in box]


def _depth_table(evidence: dict[str, Any], candidates: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for candidate in sorted(candidates, key=lambda item: item["id"]):
        depth = candidate.get("depth") or {}
        if depth.get("status") != "reliable":
            continue
        stats = depth.get("full") or {}
        core = depth.get("core") or {}
        values = []
        for prefix, stat in (("full", stats), ("core", core)):
            if not isinstance(stat, dict):
                continue
            selected = [
                (name, stat.get(key))
                for name, key in (("q1", "q1_m"), ("median", "median_m"), ("q3", "q3_m"))
            ]
            display = [f"{name}={value:.3f}m" for name, value in selected
                       if isinstance(value, (int, float)) and not isinstance(value, bool)
                       and math.isfinite(float(value))]
            if display:
                values.append(f"{prefix}[" + ", ".join(display) + "]")
        if values:
            lines.append(f"candidate {candidate['id']}: " + "; ".join(values))
    pair_lines: list[str] = []
    for pair in evidence.get("depth_pairs", []):
        if pair.get("status") != "supported":
            continue
        near_id = pair.get("near_candidate_id")
        far_id = pair.get("far_candidate_id")
        if (isinstance(near_id, int) and not isinstance(near_id, bool)
                and isinstance(far_id, int) and not isinstance(far_id, bool)):
            details = []
            for key, label in (("near_q3_m", "near q3"), ("far_q1_m", "far q1"),
                               ("median_difference_m", "median difference"),
                               ("required_difference_m", "required difference")):
                value = pair.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                    details.append(f"{label}={float(value):.3f}m")
            suffix = "; " + ", ".join(details) if details else ""
            pair_lines.append(f"candidate {near_id} is nearer than candidate {far_id}{suffix}")
    if not lines and not pair_lines:
        return ["No reliable numeric depth measurements are available for this query."]
    return [*lines, *pair_lines]


def build_selection_prompt(
    query: str,
    query_info: dict[str, Any],
    candidates: list[dict[str, Any]],
    condition: str,
    evidence: dict[str, Any],
) -> str:
    target_candidates = [candidate for candidate in candidates if candidate.get("role") == "target"]
    reference_candidates = [candidate for candidate in candidates if candidate.get("role") == "reference"]
    candidate_lines = [
        f"{candidate['id']}: bbox_2d={_rounded_coords(candidate['bbox'])}"
        for candidate in sorted(target_candidates, key=lambda item: item["id"])
    ]
    reference_lines = [
        f"{candidate['id']}: bbox_2d={_rounded_coords(candidate['bbox'])}"
        for candidate in sorted(reference_candidates, key=lambda item: item["id"])
    ]
    modalities = {
        "trimodal": "RGB, infrared, and depth visualization",
        "rgb_ir": "RGB and infrared",
        "rgb_evidence": "RGB",
    }[condition]
    lines = [
        "Choose the candidate that matches the Query. The images show the same scene with numbered candidate boxes.",
        f"Image views: {modalities}.",
        f"Query: {query}",
        f"Target category: {query_info['target_category']}",
    ]
    references = query_info["reference_categories"]
    if references:
        lines.append("Reference categories: " + ", ".join(references))
    lines.extend([
        "Candidate coordinates use normalized [x1,y1,x2,y2] values on a 0-to-1000 scale.",
        "Target candidates (choose only from these):",
        *(candidate_lines or ["(none)"]),
        "Reference candidates (context only; never return their IDs):",
        *(reference_lines or ["(none)"]),
        "Choose KEEP to retain the existing prediction box, or choose a target candidate ID.",
        'Return exactly {"id": N} or KEEP.',
    ])
    if condition in {"trimodal", "rgb_ir"}:
        lines.append("Do not choose an object merely because it looks brightest in the infrared view.")
    if condition == "trimodal":
        relation = query_info["relation_type"]
        if relation in {"camera_near", "camera_far"}:
            lines.append(
                "The Query expresses an explicit camera-near/far relation. Use only the reliable metric depth table below for numerical distance comparison; depth image brightness is not a numeric scale."
            )
            lines.extend(_depth_table(evidence, candidates))
    return "\n".join(lines)


def build_refine_prompt(query: str, selection_kind: str, selected_id: int | None) -> str:
    marker = f"candidate {selected_id}" if selection_kind == "candidate" else "KEEP"
    return (
        f'The first image is the full RGB scene with the selected region ({marker}) marked. '
        "The second image is an unmarked crop around that region. Refine the box for the original Query.\n"
        f"Original Query: {query}\n"
        'Return exactly {"bbox_2d":[x1,y1,x2,y2]} using integer coordinates from 0 to 1000 '
        "in the second image, or return KEEP to preserve the selected box."
    )


def resolve_image_path(value: object, manifest_path: Path, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"manifest image path {field} must be a nonempty string")
    path = Path(value)
    return (path if path.is_absolute() else manifest_path.parent / path).resolve()


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def load_condition_images(row: dict[str, Any], manifest_path: Path, condition: str) -> list[Image.Image]:
    keys = {
        "trimodal": ("rgb", "ir", "depth_visual"),
        "rgb_ir": ("rgb", "ir"),
        "rgb_evidence": ("rgb",),
    }[condition]
    return [
        load_rgb(resolve_image_path(row["images"].get(key), manifest_path, field=f"{row['_id']}:{key}"))
        for key in keys
    ]


def load_baseline_images(row: dict[str, Any], manifest_path: Path) -> list[Image.Image]:
    return [
        load_rgb(resolve_image_path(row["images"].get(key), manifest_path, field=f"{row['_id']}:{key}"))
        for key in ("rgb", "ir", "depth_visual")
    ]


def _load_qwen(model_path: str, adapter_path: Path) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for auxiliary Qwen3-VL inference")
    if not (adapter_path / "adapter_config.json").is_file():
        raise FileNotFoundError(f"PEFT adapter_config.json not found: {adapter_path}")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.evaluate_pretrained_grounder import configure_torch_precision

    configure_torch_precision()
    processor = AutoProcessor.from_pretrained(
        model_path, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS,
        local_files_only=True,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="sdpa",
        local_files_only=True,
    )
    from peft import PeftModel

    model = PeftModel.from_pretrained(
        model, adapter_path, is_trainable=False,
        autocast_adapter_dtype=False, local_files_only=True,
    )
    model = model.eval().to(device="cuda", dtype=torch.bfloat16)
    torch.cuda.reset_peak_memory_stats()
    return torch, processor, model


def _generate(
    torch: Any,
    processor: Any,
    model: Any,
    prompt: str,
    images: list[Image.Image],
    *,
    prompt_has_image_placeholders: bool,
) -> dict[str, Any]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.evaluate_pretrained_grounder import build_user_content

    messages = [{
        "role": "user",
        "content": build_user_content(
            prompt, images,
            prompt_has_image_placeholders=prompt_has_image_placeholders,
        ),
    }]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    inputs = inputs.to("cuda")
    torch.cuda.synchronize()
    started = time.perf_counter()
    generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    torch.cuda.synchronize()
    latency = time.perf_counter() - started
    new_tokens = generated[0, inputs["input_ids"].shape[1]:]
    eos = processor.tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    raw_text = processor.decode(
        new_tokens, skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    grid = inputs.get("image_grid_thw")
    return {
        "raw_text": raw_text,
        "image_grid_thw": grid.detach().cpu().tolist() if grid is not None else None,
        "generated_tokens": int(len(new_tokens)),
        "input_tokens": int(inputs["input_ids"].shape[1]),
        "generation_cap_hit": len(new_tokens) >= MAX_NEW_TOKENS and not any(
            int(token) in eos_ids for token in new_tokens
        ),
        "latency_seconds": latency,
    }


def _base_run_config(model: str, adapter: Path, manifest: Path, *, stage: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "model": model,
        "adapter": str(adapter.resolve()),
        "manifest": str(manifest.resolve()),
        "min_pixels": MIN_PIXELS,
        "max_pixels": MAX_PIXELS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "model_dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "peft_autocast_adapter_dtype": False,
        "tf32": False,
        "matmul_precision": "highest",
        "do_sample": False,
    }


def verify_run_config(path: Path, current: dict[str, Any], *, resume: bool) -> None:
    if not resume:
        if path.exists():
            raise FileExistsError(f"run config already exists: {path}; use --resume")
        return
    if not path.exists():
        raise FileNotFoundError(f"cannot resume without run config: {path}")
    saved = json.loads(path.read_text(encoding="utf-8-sig"))
    if saved != current:
        mismatched = sorted(key for key in set(saved) | set(current) if saved.get(key) != current.get(key))
        raise ValueError(f"resume run_config mismatch for: {', '.join(mismatched)}")


def validate_resume_rows(path: Path, expected_ids: set[str]) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    indexed = index_rows_by_id(rows, source=str(path))
    unknown = set(indexed) - expected_ids
    if unknown:
        raise ValueError(f"resume file has unknown IDs: {sorted(unknown)[:5]}")
    return indexed


def check_deadline(deadline_epoch: float | None, sample_id: str) -> bool:
    if deadline_epoch is None:
        return True
    if time.time() >= deadline_epoch:
        print(json.dumps({"status": "deadline_reached", "next_id": sample_id}, ensure_ascii=False), flush=True)
        return False
    return True


def build_query_parse_prompt(query: str) -> str:
    instruction = (
        'Read only this referring Query; do not use images. Return open category nouns for the '
        'target and up to two reference objects. Strip color, direction, position, ordinal, size, '
        'and other selection modifiers from category names. A reference must be a separately named '
        'object of a different category, never a target modifier, a piece of a compound target '
        'category, or a repeated same-category competitor. Do not turn a direction or a color into '
        'a reference object. Use these text-only examples:\n'
        'Query: "the silver sedan parked at the far left"\n'
        'Answer: {"target_category":"sedan","reference_categories":[],"relation_type":"none","scope":"single"}\n'
        'Query: "a person beside a tree"\n'
        'Answer: {"target_category":"person","reference_categories":["tree"],"relation_type":"other","scope":"single"}\n'
        'Query: "the front surface of a cabinet"\n'
        'Answer: {"target_category":"cabinet","reference_categories":[],"relation_type":"none","scope":"part"}\n'
        'relation_type is camera_near or camera_far only for explicit distance to the camera or clear '
        'foreground/background camera-depth layers. Foreground/background must describe camera depth, '
        'not image position. Near another object is relation_type other; left/right/top/bottom alone is '
        'not a depth relation. Use none when there is no relation. scope is group for a combination/group '
        'target, part for a requested component, face, side, surface, edge, or other portion of an object '
        '(including its front/back/top/bottom), single for one whole object, or unclear if the text alone '
        'does not decide. For a part, target_category is the containing object category. Do not infer '
        'scope from images. Return exactly one JSON object with keys target_category, '
        'reference_categories, relation_type, scope.\nQuery: '
    )
    return instruction + query


def _summary(
    path: Path,
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    process_started: float | None = None,
    model_load_seconds: float | None = None,
    torch: Any | None = None,
) -> None:
    summary_path = path.parent / "summary.json"
    previous = json.loads(summary_path.read_text(encoding="utf-8-sig")) if summary_path.exists() else {}
    allocated = int(previous.get("gpu_peak_allocated_bytes", 0))
    reserved = int(previous.get("gpu_peak_reserved_bytes", 0))
    if torch is not None:
        allocated = int(torch.cuda.max_memory_allocated())
        reserved = int(torch.cuda.max_memory_reserved())
    summary = {
        **config,
        "completed_queries": len(rows),
        "parse_failures": sum(not row.get("parsed", False) for row in rows),
        "predictions": str(path.resolve()),
        "total_generation_seconds": sum(float(row.get("latency_seconds", 0.0)) for row in rows),
        "total_elapsed_seconds": time.perf_counter() - process_started if process_started is not None else previous.get("total_elapsed_seconds"),
        "model_load_seconds": model_load_seconds if model_load_seconds is not None else previous.get("model_load_seconds"),
        "gpu_peak_allocated_bytes": allocated,
        "gpu_peak_reserved_bytes": reserved,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_parse_query(args: argparse.Namespace) -> None:
    process_started = time.perf_counter()
    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(manifest_path, require_images=False)
    selected = manifest[:args.limit] if args.limit else manifest
    rows_path = output_dir / "query_info.jsonl"
    config_path = output_dir / "run_config.json"
    config = {
        **_base_run_config(args.model, args.adapter, manifest_path, stage="parse-query"),
        "query_info": str(rows_path),
        "limit": args.limit,
        "prompt_version": QUERY_PARSE_PROMPT_VERSION,
    }
    if not args.resume and rows_path.exists():
        raise FileExistsError(f"output already exists: {rows_path}; use --resume")
    verify_run_config(config_path, config, resume=args.resume)
    existing = validate_resume_rows(rows_path, {row["_id"] for row in selected}) if args.resume else {}
    if not config_path.exists():
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    missing = [row for row in selected if row["_id"] not in existing]
    if not missing:
        _summary(rows_path, config, list(existing.values()), process_started=process_started)
        return
    model_load_started = time.perf_counter()
    torch, processor, model = _load_qwen(args.model, args.adapter.resolve())
    model_load_seconds = time.perf_counter() - model_load_started
    mode = "a" if rows_path.exists() else "w"
    all_rows = dict(existing)
    with rows_path.open(mode, encoding="utf-8") as handle, torch.inference_mode():
        for source in missing:
            if not check_deadline(args.deadline_epoch, source["_id"]):
                break
            prompt = build_query_parse_prompt(source["query"])
            started = time.perf_counter()
            output = _generate(torch, processor, model, prompt, [], prompt_has_image_placeholders=False)
            parsed = parse_query_json(output["raw_text"])
            elapsed = time.perf_counter() - started
            row = {
                "id": source["id"],
                "query": source["query"],
                **(parsed or {
                    "target_category": None,
                    "reference_categories": [],
                    "relation_type": None,
                    "scope": None,
                }),
                "parsed": parsed is not None,
                "prompt": prompt,
                "raw_text": output["raw_text"],
                "image_grid_thw": output["image_grid_thw"],
                "generated_tokens": output["generated_tokens"],
                "input_tokens": output["input_tokens"],
                "generation_cap_hit": output["generation_cap_hit"],
                "latency_seconds": output["latency_seconds"],
                "elapsed_seconds": elapsed,
            }
            if parsed is None:
                row["parse_error"] = "invalid_query_json"
                print("INVALID_MODEL_OUTPUT " + json.dumps({"id": source["id"], "stage": "parse-query", "raw_text": output["raw_text"]}, ensure_ascii=False), flush=True)
            write_jsonl_row(handle, row)
            all_rows[source["_id"]] = row
    _summary(rows_path, config, list(all_rows.values()), process_started=process_started,
             model_load_seconds=model_load_seconds, torch=torch)


def _load_query_info(path: Path, expected_ids: set[str]) -> dict[str, dict[str, Any]]:
    indexed = index_rows_by_id(read_jsonl(path), source=str(path))
    extra = set(indexed) - expected_ids
    if extra:
        raise ValueError(f"query_info has IDs absent from manifest: {sorted(extra)[:5]}")
    for sample_id, row in indexed.items():
        parsed = parse_query_json(json.dumps({key: row.get(key) for key in (
            "target_category", "reference_categories", "relation_type", "scope",
        )}, ensure_ascii=False))
        if row.get("parsed") is not True or parsed is None:
            raise ValueError(f"query_info has invalid parse for ID {sample_id}")
    return indexed


def _load_evidence(path: Path, manifest: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    evidence_by_id = index_rows_by_id(rows, source=str(path))
    manifest_by_id = {row["_id"]: row for row in manifest}
    if set(evidence_by_id) != set(manifest_by_id):
        raise ValueError(f"evidence IDs differ from manifest: missing={sorted(set(manifest_by_id)-set(evidence_by_id))[:5]}, extra={sorted(set(evidence_by_id)-set(manifest_by_id))[:5]}")
    for sample_id, evidence in evidence_by_id.items():
        source = manifest_by_id[sample_id]
        if evidence.get("query") != source["query"]:
            raise ValueError(f"evidence Query mismatch for ID {sample_id}")
        evidence_images = evidence.get("images")
        if not isinstance(evidence_images, dict) or any(
            evidence_images.get(key) != source["images"].get(key) for key in IMAGE_FIELDS
        ):
            raise ValueError(f"evidence image path mismatch for ID {sample_id}")
        if evidence.get("depth_encoding") != source["depth_encoding"]:
            raise ValueError(f"evidence depth_encoding mismatch for ID {sample_id}")
        require_box(evidence.get("c_bbox"), field=f"{sample_id}.c_bbox")
        candidates = evidence.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError(f"{sample_id}: candidates must be a list")
        candidate_index(candidates)
        depth_pairs = evidence.get("depth_pairs")
        if not isinstance(depth_pairs, list) or any(not isinstance(pair, dict) for pair in depth_pairs):
            raise ValueError(f"{sample_id}: depth_pairs must be a list of objects")
        if evidence.get("depth_encoding") != "city_mm":
            raise ValueError(f"{sample_id}: expected depth_encoding=city_mm")
        if not evidence.get("images", {}).get("rgb"):
            raise ValueError(f"{sample_id}: RGB path is required")
    return evidence_by_id


def _make_selection_row(
    source: dict[str, Any], evidence: dict[str, Any], query_info: dict[str, Any],
    condition: str, output: dict[str, Any] | None, prompt: str | None,
    elapsed: float, candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    indexed_candidates = candidate_index(candidates)
    candidate_ids = sorted(indexed_candidates)
    target_candidate_ids = sorted(
        candidate_id for candidate_id, candidate in indexed_candidates.items()
        if candidate["role"] == "target"
    )
    reference_candidate_ids = sorted(
        candidate_id for candidate_id, candidate in indexed_candidates.items()
        if candidate["role"] == "reference"
    )
    common = {
        "id": source["id"],
        "query": source["query"],
        "candidate_ids": candidate_ids,
        "target_candidate_ids": target_candidate_ids,
        "reference_candidate_ids": reference_candidate_ids,
        "condition": condition,
        "selection_prompt": prompt,
        "selection_raw": output["raw_text"] if output else None,
        "selection_image_grid_thw": output["image_grid_thw"] if output else None,
        "selection_latency_seconds": output["latency_seconds"] if output else 0.0,
        "selection_generated_tokens": output["generated_tokens"] if output else 0,
        "selection_input_tokens": output["input_tokens"] if output else 0,
        "selection_generation_cap_hit": output["generation_cap_hit"] if output else False,
        "elapsed_seconds": elapsed,
        "target_category": query_info["target_category"],
        "reference_categories": query_info["reference_categories"],
    }
    if query_info["scope"] != "single":
        box = require_box(evidence["c_bbox"], field="c_bbox")
        return {
            **common,
            "prediction": box,
            "prediction_bbox": box,
            "parsed": True,
            "parse": True,
            "raw_text": "",
            "selection_kind": "scope_policy_keep",
            "selection_id": None,
            "selection_status": f"scope_{query_info['scope']}_kept_c_bbox",
            "selected_box": box,
            "changed_from_c": False,
        }
    assert output is not None and prompt is not None
    choice = parse_selection(output["raw_text"], set(target_candidate_ids))
    if choice is None:
        return {
            **common,
            "prediction": None,
            "prediction_bbox": None,
            "parsed": False,
            "parse": False,
            "raw_text": output["raw_text"],
            "selection_kind": "invalid",
            "selection_id": None,
            "selection_status": "invalid_model_format_or_unknown_id",
            "selected_box": None,
            "changed_from_c": False,
        }
    box = select_bbox(choice, candidates, evidence["c_bbox"])
    changed = selection_changed_from_c(choice, candidates)
    return {
        **common,
        "prediction": box,
        "prediction_bbox": box,
        "parsed": True,
        "parse": True,
        "raw_text": output["raw_text"],
        "selection_kind": choice[0],
        "selection_id": choice[1],
        "selection_status": "selected",
        "selected_box": box,
        "changed_from_c": changed,
    }


def _refine_row(
    torch: Any, processor: Any, model: Any,
    source: dict[str, Any], evidence: dict[str, Any], candidates: list[dict[str, Any]],
    selection_row: dict[str, Any], manifest_path: Path,
) -> dict[str, Any]:
    selected_box = selection_row.get("selected_box")
    selection_kind = selection_row.get("selection_kind")
    if selection_kind == "invalid" or selected_box is None:
        return {
            **selection_row,
            "prediction": None,
            "prediction_bbox": None,
            "parsed": False,
            "parse": False,
            "refine_status": "skipped_invalid_selection",
        }
    selected_box = require_box(selected_box, field="selected_box")
    if selection_row.get("selection_status", "").startswith("scope_"):
        return {
            **selection_row,
            "refine_status": "skipped_scope_policy",
            "refine_raw": None,
            "crop_transform": None,
        }
    if not needs_refine(
        selection_kind, selected_box,
        changed_from_c=bool(selection_row.get("changed_from_c", False)),
    ):
        return {
            **selection_row,
            "refine_status": "skipped_candidate_box_large_enough",
            "refine_raw": None,
            "crop_transform": None,
        }
    rgb_path = resolve_image_path(evidence["images"]["rgb"], manifest_path, field=f"{source['_id']}:rgb")
    rgb = load_rgb(rgb_path)
    pixel_crop, transform = make_crop_transform(selected_box, *rgb.size)
    crop = rgb.crop(pixel_crop)
    global_view = annotate_candidates(
        rgb, candidates,
        selected_id=selection_row.get("selection_id"),
        keep_box=selected_box if selection_kind == "keep" else None,
    )
    prompt = build_refine_prompt(source["query"], selection_kind, selection_row.get("selection_id"))
    started = time.perf_counter()
    output = _generate(torch, processor, model, prompt, [global_view, crop], prompt_has_image_placeholders=False)
    elapsed = time.perf_counter() - started
    parsed = parse_crop_bbox(output["raw_text"])
    if parsed == "KEEP":
        prediction = selected_box
        status = "explicit_keep"
        okay = True
    elif parsed is None:
        prediction = None
        status = "invalid_model_format"
        okay = False
    else:
        prediction = map_crop_bbox_to_original(parsed, transform)
        status = "refined"
        okay = True
    return {
        **selection_row,
        "prediction": prediction,
        "prediction_bbox": prediction,
        "parsed": okay,
        "parse": okay,
        "raw_text": output["raw_text"],
        "refine_raw": output["raw_text"],
        "refine_prompt": prompt,
        "refine_image_grid_thw": output["image_grid_thw"],
        "refine_latency_seconds": output["latency_seconds"],
        "refine_generated_tokens": output["generated_tokens"],
        "refine_input_tokens": output["input_tokens"],
        "refine_generation_cap_hit": output["generation_cap_hit"],
        "refine_elapsed_seconds": elapsed,
        "image_grid_thw": output["image_grid_thw"],
        "latency_seconds": float(selection_row.get("selection_latency_seconds", 0.0)) + output["latency_seconds"],
        "generated_tokens": int(selection_row.get("selection_generated_tokens", 0)) + output["generated_tokens"],
        "input_tokens": int(selection_row.get("selection_input_tokens", 0)) + output["input_tokens"],
        "generation_cap_hit": bool(selection_row.get("selection_generation_cap_hit", False) or output["generation_cap_hit"]),
        "crop_transform": transform,
        "crop_pixel_box": list(pixel_crop),
        "selected_box": selected_box,
        "refine_status": status,
    }


def run_select_refine(args: argparse.Namespace) -> None:
    process_started = time.perf_counter()
    manifest_path = args.manifest.resolve()
    evidence_path = args.evidence.resolve()
    query_info_path = args.query_info.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(manifest_path, require_images=True)
    selected_manifest = manifest[:args.limit] if args.limit else manifest
    selected_ids = {row["_id"] for row in selected_manifest}
    # The parser artifact must cover the same selected manifest IDs.
    all_manifest_ids = {row["_id"] for row in manifest}
    query_info_by_id = _load_query_info(query_info_path, all_manifest_ids)
    if not selected_ids <= set(query_info_by_id):
        raise ValueError(f"query_info is missing selected IDs: {sorted(selected_ids-set(query_info_by_id))[:5]}")
    evidence_by_id = _load_evidence(evidence_path, manifest)
    for source in selected_manifest:
        if query_info_by_id[source["_id"]].get("query") != source["query"]:
            raise ValueError(f"query_info Query mismatch for ID {source['_id']}")
    selected_path = output_dir / "selected_predictions.jsonl"
    final_path = output_dir / "final_predictions.jsonl"
    config_path = output_dir / "run_config.json"
    config = {
        **_base_run_config(args.model, args.adapter, manifest_path, stage="select-refine"),
        "evidence": str(evidence_path),
        "query_info": str(query_info_path),
        "condition": args.condition,
        "limit": args.limit,
        "prompt_version": PROMPT_VERSION,
        "crop_policy_version": CROP_POLICY_VERSION,
        "candidate_pool": "fixed_with_rgb_and_ir_sources",
        "candidate_prompt_fields": ["id", "bbox", "target_or_reference_role"],
        "candidate_hidden_metadata": ["is_baseline", "sources", "mask_path"],
        "depth_prompt_policy": "trimodal_only_explicit_camera_near_far_reliable_stats",
        "rgb_evidence_note": "candidate pool stays fixed and includes IR-sourced candidates",
    }
    if not args.resume and any(path.exists() for path in (selected_path, final_path)):
        raise FileExistsError(f"output already contains predictions in {output_dir}; use --resume")
    verify_run_config(config_path, config, resume=args.resume)
    selected_existing = validate_resume_rows(selected_path, selected_ids) if args.resume else {}
    final_existing = validate_resume_rows(final_path, selected_ids) if args.resume else {}
    if set(final_existing) - set(selected_existing):
        raise ValueError("final_predictions contains IDs missing from selected_predictions")
    if not config_path.exists():
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    missing_final = [row for row in selected_manifest if row["_id"] not in final_existing]
    if not missing_final:
        _summary(final_path, config, list(final_existing.values()), process_started=process_started)
        return
    needs_model = False
    for source in missing_final:
        query_info = query_info_by_id[source["_id"]]
        selection_row = selected_existing.get(source["_id"])
        if selection_row is None:
            needs_model |= query_info["scope"] == "single"
        elif not selection_row.get("selection_status", "").startswith("scope_"):
            selected_box = selection_row.get("selected_box")
            if selection_row.get("selection_kind") != "invalid" and selected_box is not None:
                needs_model |= needs_refine(
                    selection_row["selection_kind"], selected_box,
                    changed_from_c=bool(selection_row.get("changed_from_c", False)),
                )
    if needs_model:
        model_load_started = time.perf_counter()
        torch, processor, model = _load_qwen(args.model, args.adapter.resolve())
        model_load_seconds: float | None = time.perf_counter() - model_load_started
        inference_context = torch.inference_mode()
    else:
        torch = processor = model = None
        model_load_seconds = 0.0
        inference_context = contextlib.nullcontext()
    selected_mode = "a" if selected_path.exists() else "w"
    final_mode = "a" if final_path.exists() else "w"
    all_final = dict(final_existing)
    with selected_path.open(selected_mode, encoding="utf-8") as selected_handle, \
            final_path.open(final_mode, encoding="utf-8") as final_handle, \
            inference_context:
        for source in missing_final:
            if not check_deadline(args.deadline_epoch, source["_id"]):
                break
            evidence = evidence_by_id[source["_id"]]
            query_info = query_info_by_id[source["_id"]]
            candidates = evidence["candidates"]
            started = time.perf_counter()
            selection_row = selected_existing.get(source["_id"])
            if selection_row is None:
                if query_info["scope"] != "single":
                    output = None
                    prompt = None
                else:
                    view_images = load_condition_images(source, manifest_path, args.condition)
                    annotated = [annotate_candidates(image, candidates) for image in view_images]
                    prompt = build_selection_prompt(source["query"], query_info, candidates, args.condition, evidence)
                    output = _generate(torch, processor, model, prompt, annotated, prompt_has_image_placeholders=False)
                selection_row = _make_selection_row(
                    source, evidence, query_info, args.condition, output, prompt,
                    time.perf_counter() - started, candidates,
                )
                if not selection_row["parsed"]:
                    print("INVALID_MODEL_OUTPUT " + json.dumps({"id": source["id"], "stage": "select", "raw_text": selection_row["selection_raw"]}, ensure_ascii=False), flush=True)
                write_jsonl_row(selected_handle, selection_row)
                selected_existing[source["_id"]] = selection_row
            selection_kind = selection_row.get("selection_kind")
            selected_box = selection_row.get("selected_box")
            if selection_row.get("selection_status", "").startswith("scope_") or selection_kind == "invalid":
                final_row = _refine_row(torch, processor, model, source, evidence, candidates, selection_row, manifest_path)
            elif selected_box is not None and needs_refine(
                selection_kind, selected_box,
                changed_from_c=bool(selection_row.get("changed_from_c", False)),
            ):
                final_row = _refine_row(torch, processor, model, source, evidence, candidates, selection_row, manifest_path)
                if final_row.get("refine_status") == "invalid_model_format":
                    print("INVALID_MODEL_OUTPUT " + json.dumps({"id": source["id"], "stage": "refine", "raw_text": final_row.get("refine_raw")}, ensure_ascii=False), flush=True)
            else:
                final_row = {
                    **selection_row,
                    "refine_status": "skipped_candidate_box_large_enough",
                    "refine_raw": None,
                    "refine_prompt": None,
                    "refine_image_grid_thw": None,
                    "refine_latency_seconds": 0.0,
                    "crop_transform": None,
                }
            final_row["elapsed_seconds"] = time.perf_counter() - started
            final_row["latency_seconds"] = float(final_row.get("selection_latency_seconds", 0.0)) + float(final_row.get("refine_latency_seconds", 0.0))
            final_row["generated_tokens"] = int(final_row.get("selection_generated_tokens", 0)) + int(final_row.get("refine_generated_tokens", 0))
            final_row["input_tokens"] = int(final_row.get("selection_input_tokens", 0)) + int(final_row.get("refine_input_tokens", 0))
            final_row["generation_cap_hit"] = bool(final_row.get("selection_generation_cap_hit", False) or final_row.get("refine_generation_cap_hit", False))
            final_row.setdefault("image_grid_thw", final_row.get("selection_image_grid_thw"))
            write_jsonl_row(final_handle, final_row)
            all_final[source["_id"]] = final_row
    _summary(final_path, config, list(all_final.values()), process_started=process_started,
             model_load_seconds=model_load_seconds, torch=torch)


def run_baseline(args: argparse.Namespace) -> None:
    process_started = time.perf_counter()
    manifest_path = args.manifest.resolve()
    prompts_path = args.prompts.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(manifest_path, require_images=True)
    prompts = json.loads(prompts_path.read_text(encoding="utf-8-sig"))
    if not isinstance(prompts, dict):
        raise ValueError("--prompts must be a JSON object mapping IDs to original SFT human prompts")
    if any(not isinstance(prompt, str) or not prompt for prompt in prompts.values()):
        raise ValueError("every prompt value must be nonempty text")
    prompt_by_id = {str(key): value for key, value in prompts.items()}
    manifest_ids = {row["_id"] for row in manifest}
    if set(prompt_by_id) != manifest_ids:
        raise ValueError(f"prompt IDs differ from manifest: missing={sorted(manifest_ids-set(prompt_by_id))[:5]}, extra={sorted(set(prompt_by_id)-manifest_ids)[:5]}")
    selected = manifest[:args.limit] if args.limit else manifest
    config = {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "stage": "baseline",
        "model": args.model,
        "adapter": str(args.adapter.resolve()),
        "manifest": str(manifest_path),
        "prompts": str(prompts_path),
        "output": str(output_path),
        "limit": args.limit,
        "min_pixels": MIN_PIXELS,
        "max_pixels": MAX_PIXELS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "model_dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "peft_autocast_adapter_dtype": False,
        "tf32": False,
        "matmul_precision": "highest",
        "do_sample": False,
        "image_order": ["rgb", "ir", "depth_visual"],
    }
    config_path = output_path.parent / "run_config.json"
    if not args.resume and output_path.exists():
        raise FileExistsError(f"output already exists: {output_path}; use --resume")
    verify_run_config(config_path, config, resume=args.resume)
    existing = validate_resume_rows(output_path, {row["_id"] for row in selected}) if args.resume else {}
    if not config_path.exists():
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    missing = [row for row in selected if row["_id"] not in existing]
    if not missing:
        _summary(output_path, config, list(existing.values()), process_started=process_started)
        return
    model_load_started = time.perf_counter()
    torch, processor, model = _load_qwen(args.model, args.adapter.resolve())
    model_load_seconds = time.perf_counter() - model_load_started
    mode = "a" if output_path.exists() else "w"
    all_rows = dict(existing)
    with output_path.open(mode, encoding="utf-8") as handle, torch.inference_mode():
        for source in missing:
            if not check_deadline(args.deadline_epoch, source["_id"]):
                break
            prompt = prompt_by_id[source["_id"]]
            images = load_baseline_images(source, manifest_path)
            if prompt.count("<image>") != len(images):
                raise ValueError(f"{source['_id']}: SFT prompt image placeholders do not match RGB/IR/depth_visual")
            started = time.perf_counter()
            output = _generate(torch, processor, model, prompt, images, prompt_has_image_placeholders=True)
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from tools.evaluate_pretrained_grounder import parse_generated_bbox

            prediction = parse_generated_bbox(output["raw_text"])
            elapsed = time.perf_counter() - started
            row = {
                "id": source["id"],
                "query": source["query"],
                "prediction": prediction,
                "prediction_bbox": prediction,
                "parsed": prediction is not None,
                "parse": prediction is not None,
                "raw_text": output["raw_text"],
                "prompt": prompt,
                "image_grid_thw": output["image_grid_thw"],
                "generated_tokens": output["generated_tokens"],
                "input_tokens": output["input_tokens"],
                "generation_cap_hit": output["generation_cap_hit"],
                "latency_seconds": output["latency_seconds"],
                "elapsed_seconds": elapsed,
            }
            if prediction is None:
                row["parse_error"] = "invalid_bbox_output"
                print("INVALID_MODEL_OUTPUT " + json.dumps({"id": source["id"], "stage": "baseline", "raw_text": output["raw_text"]}, ensure_ascii=False), flush=True)
            write_jsonl_row(handle, row)
            all_rows[source["_id"]] = row
    _summary(output_path, config, list(all_rows.values()), process_started=process_started,
             model_load_seconds=model_load_seconds, torch=torch)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    query = commands.add_parser("parse-query", help="C text-only Query category and scope parsing")
    query.add_argument("--manifest", type=Path, required=True)
    query.add_argument("--output-dir", type=Path, required=True)
    query.add_argument("--model", required=True, help="Local Qwen3-VL model path")
    query.add_argument("--adapter", type=Path, required=True)
    query.add_argument("--resume", action="store_true")
    query.add_argument("--limit", type=int, default=0)
    query.add_argument("--deadline-epoch", type=float)
    query.set_defaults(func=run_parse_query)

    select = commands.add_parser("select-refine", help="Select a fixed candidate and refine locally when required")
    select.add_argument("--manifest", type=Path, required=True)
    select.add_argument("--evidence", type=Path, required=True)
    select.add_argument("--query-info", type=Path, required=True)
    select.add_argument("--output-dir", type=Path, required=True)
    select.add_argument("--condition", choices=("trimodal", "rgb_ir", "rgb_evidence"), required=True)
    select.add_argument("--model", required=True, help="Local Qwen3-VL model path")
    select.add_argument("--adapter", type=Path, required=True)
    select.add_argument("--resume", action="store_true")
    select.add_argument("--limit", type=int, default=0)
    select.add_argument("--deadline-epoch", type=float)
    select.set_defaults(func=run_select_refine)

    baseline = commands.add_parser("baseline", help="Reproduce a frozen SFT C baseline with original prompts")
    baseline.add_argument("--manifest", type=Path, required=True)
    baseline.add_argument("--prompts", type=Path, required=True, help="JSON object mapping ID to original SFT human prompt")
    baseline.add_argument("--model", required=True, help="Local Qwen3-VL model path")
    baseline.add_argument("--adapter", type=Path, required=True)
    baseline.add_argument("--output", type=Path, required=True)
    baseline.add_argument("--resume", action="store_true")
    baseline.add_argument("--limit", type=int, default=0)
    baseline.add_argument("--deadline-epoch", type=float)
    baseline.set_defaults(func=run_baseline)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.limit < 0:
        raise ValueError("--limit must be nonnegative")
    args.func(args)


if __name__ == "__main__":
    main()
