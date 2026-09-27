"""Build RGB/IR object proposals and attach SAM2/City depth evidence.

The candidate stage uses only the manifest, the C baseline, and query metadata.
It never opens a ground-truth annotation. Run the two stages in separate
processes so GroundingDINO and SAM2 do not occupy the GPU at the same time.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from itertools import combinations, product
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter


BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
DEDUP_IOU = 0.9
MAX_TARGET_CANDIDATES = 8
MAX_REFERENCE_CANDIDATES = 4
MASK_QUALITY_THRESHOLD = 0.8
CORE_SHRINK_PIXELS = 3
MIN_CORE_VALID_PIXELS = 32
MIN_CORE_VALID_RATIO = 0.7
ID_SEED = 2026

Box = tuple[float, float, float, float]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} must contain JSON object records")
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _config_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".config.json")


def _input_config(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {"path": str(resolved), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _load_completed_rows(
    output_path: Path, config: dict[str, Any], expected_ids: set[str]
) -> dict[str, dict[str, Any]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config_path = _config_path(output_path)
    if config_path.exists():
        saved_config = json.loads(config_path.read_text(encoding="utf-8-sig"))
        if saved_config != config:
            raise ValueError(f"resume config mismatch: {config_path}")
    elif output_path.exists() and output_path.stat().st_size:
        raise ValueError(f"cannot resume {output_path}: stage config is missing")
    else:
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    existing: dict[str, dict[str, Any]] = {}
    if output_path.exists():
        for row in read_jsonl(output_path):
            sample_id = str(row["id"])
            if sample_id not in expected_ids:
                raise ValueError(f"resume output contains unknown ID: {sample_id}")
            if sample_id in existing:
                raise ValueError(f"resume output contains duplicate ID: {sample_id}")
            existing[sample_id] = row
    return existing


def _append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def _unique_by_id(rows: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row["id"])
        if sample_id in indexed:
            raise ValueError(f"duplicate {name} ID: {sample_id}")
        indexed[sample_id] = row
    return indexed


def resolve_data_path(value: str, data_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (data_root / path).resolve()


def _box(value: Any) -> Box:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("bbox must contain four normalized xyxy values")
    result = tuple(float(part) for part in value)
    x1, y1, x2, y2 = result
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError(f"invalid normalized xyxy bbox: {value!r}")
    return result


def box_iou(left: Box, right: Box) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / (left_area + right_area - intersection)


def _source_key(source: dict[str, Any]) -> tuple[str, tuple[float, ...]]:
    return source["modality"], tuple(source["bbox"])


def merge_candidate_boxes(
    c_bbox: Any, detections: list[dict[str, Any]], *, iou_threshold: float = DEDUP_IOU
) -> list[dict[str, Any]]:
    """Merge overlapping RGB/IR proposals while keeping the C box exact."""
    baseline_box = _box(c_bbox)
    candidates: list[dict[str, Any]] = [
        {
            "role": "target",
            "bbox": list(baseline_box),
            "is_baseline": True,
            "sources": [{"modality": "c", "bbox": list(baseline_box), "score": None}],
            "_primary_score": float("inf"),
        }
    ]
    ordered = sorted(
        detections,
        key=lambda item: (
            item["role"],
            -float(item["score"]),
            ("rgb", "ir").index(item["modality"]),
            tuple(item["bbox"]),
        ),
    )
    for detection in ordered:
        role = detection["role"]
        if role not in {"target", "reference"}:
            raise ValueError(f"invalid proposal role: {role!r}")
        modality = detection["modality"]
        if modality not in {"rgb", "ir"}:
            raise ValueError(f"invalid proposal modality: {modality!r}")
        bbox = _box(detection["bbox"])
        score = float(detection["score"])
        source = {"modality": modality, "bbox": list(bbox), "score": score}
        if modality == "ir":
            source["projection"] = "normalized_shared_frame"

        match = next(
            (
                candidate
                for candidate in candidates
                if candidate["role"] == role
                and box_iou(tuple(candidate["bbox"]), bbox) >= iou_threshold
            ),
            None,
        )
        if match is None:
            candidates.append(
                {
                    "role": role,
                    "bbox": list(bbox),
                    "is_baseline": False,
                    "sources": [source],
                    "_primary_score": score,
                }
            )
        else:
            if _source_key(source) not in {_source_key(item) for item in match["sources"]}:
                match["sources"].append(source)
            if not match["is_baseline"] and score > match["_primary_score"]:
                match["bbox"] = list(bbox)
                match["_primary_score"] = score

    for candidate in candidates:
        candidate.pop("_primary_score")
    return candidates


def _candidate_score(candidate: dict[str, Any], modality: str | None = None) -> float:
    scores = [
        float(source["score"])
        for source in candidate["sources"]
        if source["modality"] in ({modality} if modality else {"rgb", "ir"})
        and source["score"] is not None
    ]
    return max(scores, default=-1.0)


def select_candidate_cap(
    candidates: list[dict[str, Any]], role: str, cap: int, *, modality_top_k: int = 3
) -> list[dict[str, Any]]:
    """Prioritize each modality's top proposals, then fill the role cap by score."""
    pool = [candidate for candidate in candidates if candidate["role"] == role]
    selected = [candidate for candidate in pool if candidate["is_baseline"]]
    selected_ids = {id(candidate) for candidate in selected}
    for modality in ("rgb", "ir"):
        ranked = sorted(
            (candidate for candidate in pool if _candidate_score(candidate, modality) >= 0),
            key=lambda candidate: (-_candidate_score(candidate, modality), tuple(candidate["bbox"])),
        )
        for candidate in ranked[:modality_top_k]:
            if id(candidate) not in selected_ids and len(selected) < cap:
                selected.append(candidate)
                selected_ids.add(id(candidate))
    remainder = sorted(
        (candidate for candidate in pool if id(candidate) not in selected_ids),
        key=lambda candidate: (-_candidate_score(candidate), tuple(candidate["bbox"])),
    )
    selected.extend(remainder[: max(0, cap - len(selected))])
    return selected


def build_candidate_row(
    manifest_row: dict[str, Any],
    baseline_row: dict[str, Any],
    query_info: dict[str, Any],
    detections: list[dict[str, Any]],
    id_rng: random.Random,
) -> dict[str, Any]:
    sample_id = str(manifest_row["id"])
    scope = query_info["scope"]
    if scope not in {"single", "group", "part", "unclear"}:
        raise ValueError(f"{sample_id}: invalid query scope {scope!r}")
    if scope != "single":
        detections = []

    c_bbox = list(_box(baseline_row["bbox"]))
    candidates = merge_candidate_boxes(c_bbox, detections)
    selected = [
        *select_candidate_cap(candidates, "target", MAX_TARGET_CANDIDATES),
        *select_candidate_cap(
            candidates, "reference", MAX_REFERENCE_CANDIDATES, modality_top_k=2
        ),
    ]
    random_ids = id_rng.sample(range(1, len(selected) + 1), len(selected))
    output_candidates = []
    for candidate, candidate_id in zip(selected, random_ids, strict=True):
        output_candidates.append(
            {
                "id": candidate_id,
                "role": candidate["role"],
                "bbox": candidate["bbox"],
                "is_baseline": candidate["is_baseline"],
                "sources": candidate["sources"],
                "depth": {"status": "pending", "reason": "awaiting_depth_stage"},
                "mask_path": None,
            }
        )
    output_candidates.sort(key=lambda candidate: candidate["id"])

    info = {
        "target_category": query_info.get("target_category"),
        "reference_categories": query_info.get("reference_categories", []),
        "relation_type": query_info.get("relation_type"),
        "scope": scope,
    }
    return {
        "id": manifest_row["id"],
        "query": manifest_row["query"],
        "images": manifest_row["images"],
        "depth_encoding": manifest_row["depth_encoding"],
        "c_bbox": c_bbox,
        "query_info": info,
        "candidates": output_candidates,
    }


def _categories(query_info: dict[str, Any]) -> tuple[str, list[str]]:
    target = query_info.get("target_category")
    if not isinstance(target, str) or not target.strip():
        raise ValueError("single-scope query-info needs a target_category")
    references = query_info.get("reference_categories", [])
    if isinstance(references, str):
        references = [references]
    if not isinstance(references, list) or any(not isinstance(item, str) for item in references):
        raise ValueError("reference_categories must be a list of strings")
    return target.strip(), [item.strip() for item in references if item.strip()]


def _label_role(label: str, target: str, references: list[str]) -> str | None:
    normalized_label = re.sub(r"[^\w\s-]", " ", label.casefold()).strip()
    categories = [(target, "target"), *((item, "reference") for item in references)]
    matches = [
        (category, role)
        for category, role in categories
        if normalized_label
        and (
            normalized_label == category.casefold()
            or normalized_label in category.casefold()
            or category.casefold() in normalized_label
        )
    ]
    if not matches:
        return None
    matches.sort(key=lambda pair: (-len(pair[0]), pair[1] != "target"))
    return matches[0][1]


def _load_grounding_dino(model_name: str) -> tuple[Any, Any, str]:
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_name, local_files_only=True
    )
    return processor, model.eval().to(device=device, dtype=torch.float32), device


def detect_image_proposals(
    image: Image.Image,
    modality: str,
    query_info: dict[str, Any],
    processor: Any,
    model: Any,
    device: str,
) -> list[dict[str, Any]]:
    import torch

    target, references = _categories(query_info)
    categories = list(dict.fromkeys([target, *references]))
    caption = " . ".join(categories).casefold() + " ."
    inputs = processor(images=image, text=caption, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    result = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[(image.height, image.width)],
    )[0]

    proposals = []
    for box_tensor, score_tensor, label in aligned_detection_rows(result):
        role = _label_role(str(label), target, references)
        if role is None:
            continue
        x1, y1, x2, y2 = (float(value) for value in box_tensor.detach().cpu().tolist())
        normalized = (
            max(0.0, min(1.0, x1 / image.width)),
            max(0.0, min(1.0, y1 / image.height)),
            max(0.0, min(1.0, x2 / image.width)),
            max(0.0, min(1.0, y2 / image.height)),
        )
        if normalized[0] >= normalized[2] or normalized[1] >= normalized[3]:
            continue
        proposals.append(
            {
                "role": role,
                "modality": modality,
                "bbox": list(normalized),
                "score": float(score_tensor.detach().cpu()),
            }
        )
    return proposals


def aligned_detection_rows(result: dict[str, Any]) -> list[tuple[Any, Any, str]]:
    # Transformers 5.14 tokenizer.batch_decode([]) returns ['']; no box exists
    # when all detection scores are below threshold. This is a real empty pool.
    if len(result["boxes"]) == len(result["scores"]) == 0:
        if result["text_labels"] not in ([], [""]):
            raise ValueError("nonempty labels accompany zero detection boxes")
        return []
    return list(zip(result["boxes"], result["scores"], result["text_labels"], strict=True))


def build_candidates_stage(
    manifest_path: Path,
    baseline_path: Path,
    query_info_path: Path,
    data_root: Path,
    output_path: Path,
    *,
    model_name: str = "models/grounding-dino-tiny",
    deadline_epoch: float | None = None,
) -> list[dict[str, Any]]:
    manifests = read_jsonl(manifest_path)
    baselines = _unique_by_id(read_jsonl(baseline_path), "baseline")
    query_infos = _unique_by_id(read_jsonl(query_info_path), "query-info")
    manifest_by_id = _unique_by_id(manifests, "manifest")
    if set(manifest_by_id) - set(baselines):
        raise ValueError("baseline is missing manifest IDs")
    if set(manifest_by_id) - set(query_infos):
        raise ValueError("query-info is missing manifest IDs")

    input_paths = (manifest_path, baseline_path, query_info_path)
    if output_path.resolve() in {path.resolve() for path in input_paths}:
        raise ValueError("candidate output must be separate from all stage inputs")
    config = {
        "stage": "candidates",
        "inputs": {
            "manifest": _input_config(manifest_path),
            "baseline": _input_config(baseline_path),
            "query_info": _input_config(query_info_path),
        },
        "data_root": str(data_root.resolve()),
        "model": model_name,
        "box_threshold": BOX_THRESHOLD,
        "text_threshold": TEXT_THRESHOLD,
        "dedup_iou": DEDUP_IOU,
        "max_target_candidates": MAX_TARGET_CANDIDATES,
        "max_reference_candidates": MAX_REFERENCE_CANDIDATES,
        "target_modality_top_k": 3,
        "reference_modality_top_k": 2,
        "id_seed": ID_SEED,
    }
    existing = _load_completed_rows(
        output_path, config, set(manifest_by_id)
    )
    pending_rows = [row for row in manifests if str(row["id"]) not in existing]
    need_detector = any(
        query_infos[str(row["id"])]["scope"] == "single" for row in pending_rows
    )
    processor = model = device = None
    if need_detector:
        processor, model, device = _load_grounding_dino(model_name)

    for manifest_index, manifest_row in enumerate(manifests):
        sample_id = str(manifest_row["id"])
        if sample_id in existing:
            continue
        if deadline_epoch is not None and time.time() >= deadline_epoch:
            raise TimeoutError("candidate stage reached --deadline-epoch")
        query_info = query_infos[sample_id]
        detections: list[dict[str, Any]] = []
        if query_info["scope"] == "single":
            target, references = _categories(query_info)
            if not (target or references):
                raise ValueError(f"{sample_id}: no detection categories")
            for modality in ("rgb", "ir"):
                image_path = resolve_data_path(manifest_row["images"][modality], data_root)
                with Image.open(image_path) as source:
                    image = source.convert("RGB").copy()
                detections.extend(
                    detect_image_proposals(
                        image, modality, query_info, processor, model, device
                    )
                )
        row = build_candidate_row(
            manifest_row,
            baselines[sample_id],
            query_info,
            detections,
            random.Random(ID_SEED + manifest_index),
        )
        existing[sample_id] = row
        _append_jsonl_row(output_path, row)
    return [existing[str(row["id"])] for row in manifests]


def depth_statistics(depth_mm: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    """Summarize City-valid depth pixels inside one mask."""
    if depth_mm.shape != mask.shape:
        raise ValueError("raw depth and SAM mask dimensions differ")
    mask = np.asarray(mask, dtype=bool)
    valid = mask & (depth_mm >= 300) & (depth_mm <= 19999)
    values = depth_mm[valid].astype(np.float64) / 1000.0
    count = int(values.size)
    area = int(mask.sum())
    if count == 0:
        return {
            "valid_count": 0,
            "valid_ratio": 0.0,
            "q1_m": None,
            "median_m": None,
            "q3_m": None,
        }
    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
    return {
        "valid_count": count,
        "valid_ratio": count / area if area else 0.0,
        "q1_m": float(q1),
        "median_m": float(median),
        "q3_m": float(q3),
    }


def erode_mask(mask: np.ndarray, pixels: int = CORE_SHRINK_PIXELS) -> np.ndarray:
    """Shrink a binary mask by the requested pixel radius."""
    if pixels == 0:
        return np.asarray(mask, dtype=bool)
    if pixels < 0:
        raise ValueError("mask shrink must be non-negative")
    image = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255)
    eroded = image.filter(ImageFilter.MinFilter(2 * pixels + 1))
    return np.asarray(eroded) > 0


def assess_depth_mask(
    depth_mm: np.ndarray | None,
    mask: np.ndarray,
    mask_quality: float,
    *,
    depth_encoding: str,
) -> tuple[dict[str, Any], np.ndarray]:
    """Return full/core measurements and the eroded core mask."""
    core_mask = erode_mask(mask)
    if depth_encoding != "city_mm":
        return (
            {
                "status": "unknown_encoding",
                "reason": "depth_encoding_is_not_city_mm",
                "mask_quality": float(mask_quality),
                "full": None,
                "core": None,
                "median_delta_m": None,
                "stability_limit_m": None,
            },
            core_mask,
        )

    if depth_mm is None:
        raise ValueError("city_mm encoding requires a raw depth image")
    full = depth_statistics(depth_mm, mask)
    core = depth_statistics(depth_mm, core_mask)
    reasons = []
    if mask_quality < MASK_QUALITY_THRESHOLD:
        reasons.append("mask_quality_below_0.8")
    if core["valid_count"] < MIN_CORE_VALID_PIXELS:
        reasons.append("core_valid_pixels_below_32")
    if core["valid_ratio"] < MIN_CORE_VALID_RATIO:
        reasons.append("core_valid_ratio_below_0.7")

    delta = None
    stability_limit = None
    if full["median_m"] is None or core["median_m"] is None:
        reasons.append("mask_has_no_valid_core_depth")
    else:
        delta = abs(full["median_m"] - core["median_m"])
        stability_limit = max(0.25, 0.1 * core["median_m"])
        if delta > stability_limit:
            reasons.append("full_core_median_unstable")

    reliable = not reasons
    return (
        {
            "status": "reliable" if reliable else "unreliable",
            "reason": "reliable" if reliable else ";".join(reasons),
            "mask_quality": float(mask_quality),
            "full": full,
            "core": core,
            "median_delta_m": delta,
            "stability_limit_m": stability_limit,
        },
        core_mask,
    )


def compare_depth_pair(
    left: dict[str, Any], right: dict[str, Any], depth_encoding: str
) -> dict[str, Any]:
    """Test one unique pair of object candidates for separated near/far depth."""
    roles = {left["role"], right["role"]}
    if roles == {"target"}:
        pair_kind = "target_target"
    elif roles == {"target", "reference"}:
        pair_kind = "target_reference"
    else:
        raise ValueError(f"unsupported depth-pair roles: {sorted(roles)}")
    pair: dict[str, Any] = {
        "candidate_a_id": left["id"],
        "candidate_b_id": right["id"],
        "role_a": left["role"],
        "role_b": right["role"],
        "pair_kind": pair_kind,
    }
    if pair_kind == "target_reference":
        target = left if left["role"] == "target" else right
        reference = right if target is left else left
        pair["target_candidate_id"] = target["id"]
        pair["reference_candidate_id"] = reference["id"]
    if depth_encoding != "city_mm":
        return {**pair, "status": "unknown_encoding", "reason": "depth_encoding_is_not_city_mm"}
    if left["depth"]["status"] != "reliable" or right["depth"]["status"] != "reliable":
        return {**pair, "status": "unreliable", "reason": "candidate_depth_unreliable"}

    left_core = left["depth"]["core"]
    right_core = right["depth"]["core"]
    if left_core["median_m"] <= right_core["median_m"]:
        near, far = left, right
    else:
        near, far = right, left
    near_core = near["depth"]["core"]
    far_core = far["depth"]["core"]
    difference = far_core["median_m"] - near_core["median_m"]
    required_difference = max(0.5, 0.1 * near_core["median_m"])
    separated = (
        near_core["q3_m"] < far_core["q1_m"]
        and difference > required_difference
    )
    return {
        **pair,
        "status": "supported" if separated else "not_separated",
        "reason": "near_far_quantiles_and_median_gap_pass" if separated else "near_far_separation_failed",
        "near_candidate_id": near["id"],
        "far_candidate_id": far["id"],
        "near_median_m": near_core["median_m"],
        "near_q3_m": near_core["q3_m"],
        "far_median_m": far_core["median_m"],
        "far_q1_m": far_core["q1_m"],
        "median_difference_m": difference,
        "required_difference_m": required_difference,
    }


def build_depth_pairs(
    targets: list[dict[str, Any]],
    references: list[dict[str, Any]],
    depth_encoding: str,
) -> list[dict[str, Any]]:
    target_pairs = [
        compare_depth_pair(left, right, depth_encoding)
        for left, right in combinations(targets, 2)
    ]
    target_reference_pairs = [
        compare_depth_pair(target, reference, depth_encoding)
        for target, reference in product(targets, references)
    ]
    return [*target_pairs, *target_reference_pairs]


def _load_sam2(model_name: str) -> tuple[Any, Any, str]:
    import torch
    from transformers import Sam2Model, Sam2Processor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Sam2Processor.from_pretrained(model_name, local_files_only=True)
    model = Sam2Model.from_pretrained(model_name, local_files_only=True)
    return processor, model.eval().to(device=device, dtype=torch.float32), device


def segment_boxes(
    image: Image.Image,
    candidates: list[dict[str, Any]],
    processor: Any,
    model: Any,
    device: str,
) -> list[tuple[np.ndarray, float]]:
    import torch

    pixel_boxes = []
    for candidate in candidates:
        x1, y1, x2, y2 = candidate["bbox"]
        pixel_boxes.append(
            [x1 * image.width, y1 * image.height, x2 * image.width, y2 * image.height]
        )
    inputs = processor(
        images=image,
        input_boxes=[pixel_boxes],
        return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    original_sizes = inputs["original_sizes"].detach().cpu()
    masks = processor.post_process_masks(
        outputs.pred_masks.detach().cpu(), original_sizes
    )[0]
    qualities = outputs.iou_scores.detach().cpu()[0]
    if masks.ndim != 4 or masks.shape[0] != len(candidates):
        raise ValueError(
            f"SAM2 returned masks with shape {tuple(masks.shape)} for {len(candidates)} boxes"
        )
    if qualities.ndim != 2 or qualities.shape[0] != len(candidates):
        raise ValueError(
            f"SAM2 returned scores with shape {tuple(qualities.shape)} for {len(candidates)} boxes"
        )
    segmented = []
    for candidate_index in range(len(candidates)):
        best = int(qualities[candidate_index].argmax().item())
        mask = masks[candidate_index, best].numpy() > 0.5
        if mask.shape != (image.height, image.width):
            raise ValueError(f"SAM2 mask has unexpected image shape: {mask.shape}")
        segmented.append((mask, float(qualities[candidate_index, best].item())))
    return segmented


def build_depth_stage(
    candidates_path: Path,
    data_root: Path,
    output_path: Path,
    *,
    model_name: str = "models/sam2.1-hiera-tiny",
    deadline_epoch: float | None = None,
) -> list[dict[str, Any]]:
    rows = read_jsonl(candidates_path)
    if output_path.resolve() == candidates_path.resolve():
        raise ValueError("depth output must not replace candidates input")
    config = {
        "stage": "depth",
        "input": _input_config(candidates_path),
        "data_root": str(data_root.resolve()),
        "model": model_name,
        "depth_encoding_supported": "city_mm",
        "mask_quality_threshold": MASK_QUALITY_THRESHOLD,
        "core_shrink_pixels": CORE_SHRINK_PIXELS,
        "min_core_valid_pixels": MIN_CORE_VALID_PIXELS,
        "min_core_valid_ratio": MIN_CORE_VALID_RATIO,
        "depth_valid_min_mm": 300,
        "depth_valid_max_mm": 19999,
        "median_stability_floor_m": 0.25,
        "median_stability_fraction": 0.1,
        "near_far_min_gap_m": 0.5,
        "near_far_min_gap_fraction": 0.1,
    }
    existing = _load_completed_rows(output_path, config, {str(row["id"]) for row in rows})
    pending_rows = [row for row in rows if str(row["id"]) not in existing]
    needs_sam = any(row["query_info"]["scope"] == "single" for row in pending_rows)
    processor = model = device = None
    if needs_sam:
        processor, model, device = _load_sam2(model_name)

    mask_dir = output_path.parent / "masks"
    for row_index, row in enumerate(rows):
        sample_id = str(row["id"])
        if sample_id in existing:
            continue
        if deadline_epoch is not None and time.time() >= deadline_epoch:
            raise TimeoutError("depth stage reached --deadline-epoch")
        scope = row["query_info"]["scope"]
        if scope in {"group", "part", "unclear"}:
            for candidate in row["candidates"]:
                candidate["depth"] = {
                    "status": "skipped_scope",
                    "reason": "segmentation_skipped_for_non_single_scope",
                    "mask_quality": None,
                    "full": None,
                    "core": None,
                    "median_delta_m": None,
                    "stability_limit_m": None,
                }
                candidate["mask_path"] = None
            row["depth_pairs"] = []
        else:
            rgb_path = resolve_data_path(row["images"]["rgb"], data_root)
            with Image.open(rgb_path) as source:
                rgb = source.convert("RGB").copy()
            depth_mm = None
            if row["depth_encoding"] == "city_mm":
                depth_path = resolve_data_path(row["images"]["depth_raw"], data_root)
                with Image.open(depth_path) as source:
                    depth_mm = np.asarray(source).copy()
                if depth_mm.dtype != np.uint16:
                    raise ValueError(f"{sample_id}: city_mm depth_raw must decode as uint16")
                if depth_mm.ndim != 2 or depth_mm.shape != (rgb.height, rgb.width):
                    raise ValueError(
                        f"{sample_id}: city_mm depth_raw must be aligned single-channel image"
                    )

            mask_dir.mkdir(parents=True, exist_ok=True)
            masks = segment_boxes(rgb, row["candidates"], processor, model, device)
            for candidate, (mask, quality) in zip(row["candidates"], masks, strict=True):
                mask_path = mask_dir / f"{row_index:06d}_{candidate['id']}.png"
                Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
                candidate["mask_path"] = str(mask_path.resolve())
                depth_result, _ = assess_depth_mask(
                    depth_mm,
                    mask,
                    quality,
                    depth_encoding=row["depth_encoding"],
                )
                candidate["depth"] = depth_result

            targets = [candidate for candidate in row["candidates"] if candidate["role"] == "target"]
            references = [candidate for candidate in row["candidates"] if candidate["role"] == "reference"]
            row["depth_pairs"] = build_depth_pairs(
                targets, references, row["depth_encoding"]
            )

        existing[sample_id] = row
        _append_jsonl_row(output_path, row)
    return [existing[str(row["id"])] for row in rows]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)

    candidates = subparsers.add_parser("candidates", help="run RGB/IR GroundingDINO proposals")
    candidates.add_argument("--manifest", type=Path, required=True)
    candidates.add_argument("--baseline", type=Path, required=True)
    candidates.add_argument("--query-info", type=Path, required=True)
    candidates.add_argument("--data-root", type=Path, required=True)
    candidates.add_argument("--output", type=Path, required=True, help="write candidates.jsonl")
    candidates.add_argument("--model", default="models/grounding-dino-tiny")
    candidates.add_argument("--deadline-epoch", type=float)

    depth = subparsers.add_parser("depth", help="run SAM2.1 and raw16City depth statistics")
    depth.add_argument("--candidates", type=Path, required=True)
    depth.add_argument("--data-root", type=Path, required=True)
    depth.add_argument("--output", type=Path, required=True, help="write evidence.jsonl")
    depth.add_argument("--model", default="models/sam2.1-hiera-tiny")
    depth.add_argument("--deadline-epoch", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "candidates":
        build_candidates_stage(
            args.manifest,
            args.baseline,
            args.query_info,
            args.data_root,
            args.output,
            model_name=args.model,
            deadline_epoch=args.deadline_epoch,
        )
    else:
        build_depth_stage(
            args.candidates,
            args.data_root,
            args.output,
            model_name=args.model,
            deadline_epoch=args.deadline_epoch,
        )


if __name__ == "__main__":
    main()
