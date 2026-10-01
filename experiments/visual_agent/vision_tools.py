"""Real visual observations for the TriGround controller.

Each box proposal stays in its source coordinate frame (RGB or IR); a paired
view may use it as a region cue but does not change its frame. The baseline is
always named KEEP; the old numbered baseline is an internal alias.
The controller must open returned image paths and include the PIL images in its
next processor call.  No ground truth enters this module.
"""

from __future__ import annotations

from copy import deepcopy
from math import ceil, floor, sqrt
from pathlib import Path
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tools.aux_selection_evidence import (
    assess_depth_mask,
    compare_depth_pair,
    detect_image_proposals,
    segment_boxes,
)


class ProtocolError(ValueError):
    """Invalid controller action; the controller may recover after feedback."""


FORBIDDEN_CACHE_FIELDS = {"gt", "target_bbox", "ground_truth", "iou", "class_name"}


def _protocol(text: str) -> dict[str, Any]:
    return {"status": "ERROR", "text": text, "images": [], "data": {"kind": "protocol"}}


def _normalize_id(candidate_id: Any) -> int | str:
    if candidate_id == "KEEP":
        return "KEEP"
    if isinstance(candidate_id, str) and candidate_id.isdecimal():
        return int(candidate_id)
    if isinstance(candidate_id, int) and not isinstance(candidate_id, bool):
        return candidate_id
    raise ProtocolError(f"invalid candidate ID: {candidate_id}")


def _box(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("box must be normalized xyxy")
    box = [float(number) for number in value]
    if not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
        raise ValueError(f"invalid normalized xyxy: {value}")
    return box


def _pixel_bounds(box: list[float], size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = size
    return (
        max(0, min(width - 1, floor(box[0] * width))),
        max(0, min(height - 1, floor(box[1] * height))),
        max(1, min(width, ceil(box[2] * width))),
        max(1, min(height, ceil(box[3] * height))),
    )


def _context_bounds(
    box: list[float], size: tuple[int, int], scale: float, minimum_side: float
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    width = max((x2 - x1) * scale, minimum_side)
    height = max((y2 - y1) * scale, minimum_side)
    return _pixel_bounds(
        [max(0, cx - width / 2), max(0, cy - height / 2),
         min(1, cx + width / 2), min(1, cy + height / 2)],
        size,
    )


def _local_box(box: list[float], bounds: tuple[int, int, int, int], size: tuple[int, int]) -> list[float]:
    left, top, right, bottom = bounds
    width, height = size
    return [
        max(0.0, min(1.0, (box[0] * width - left) / (right - left))),
        max(0.0, min(1.0, (box[1] * height - top) / (bottom - top))),
        max(0.0, min(1.0, (box[2] * width - left) / (right - left))),
        max(0.0, min(1.0, (box[3] * height - top) / (bottom - top))),
    ]


def _fit_pixels(image: Image.Image, budget: int) -> Image.Image:
    # Upsampling makes a small target occupy real processor pixels, while the
    # caller can still distinguish interpolated pixels from source detail.
    target_scale = sqrt(budget / (image.width * image.height))
    width = max(1, floor(image.width * target_scale))
    height = max(1, floor(image.height * target_scale))
    while width * height > budget:
        if width >= height:
            width -= 1
        else:
            height -= 1
    return image.resize((width, height), Image.Resampling.LANCZOS)


class CandidatePool:
    def __init__(self, candidate_row: dict[str, Any], rgb_size: tuple[int, int]):
        forbidden = FORBIDDEN_CACHE_FIELDS & candidate_row.keys()
        if forbidden:
            raise ValueError(f"candidate cache contains forbidden fields: {sorted(forbidden)}")
        self.rgb_size = tuple(rgb_size)
        self.c_bbox = _box(candidate_row["c_bbox"])
        self.scope = candidate_row.get("query_info", {}).get("scope", "single")
        self.source_identity = {
            "id": candidate_row.get("id"),
            "rgb": candidate_row.get("images", {}).get("rgb"),
            "depth_raw": candidate_row.get("images", {}).get("depth_raw"),
            "depth_encoding": candidate_row.get("depth_encoding"),
        }
        self._candidates: dict[int, dict[str, Any]] = {}
        self.aliases: dict[int, str] = {}
        self._next_id = 1
        for original in candidate_row["candidates"]:
            forbidden = FORBIDDEN_CACHE_FIELDS & original.keys()
            forbidden |= set().union(*(FORBIDDEN_CACHE_FIELDS & source.keys() for source in original.get("sources", [])))
            if forbidden:
                raise ValueError(f"candidate cache contains forbidden fields: {sorted(forbidden)}")
            item = deepcopy(original)
            candidate_id = item["id"]
            if isinstance(candidate_id, bool) or not isinstance(candidate_id, int):
                raise ValueError("candidate ID must be an integer")
            item["bbox"] = _box(item["bbox"])
            if "coordinate_frame" not in item:
                source_modalities = {source.get("modality") for source in item.get("sources", [])}
                item["coordinate_frame"] = (
                    "ir" if not item.get("is_baseline") and source_modalities == {"ir"} else "rgb"
                )
            item.setdefault("coordinate_size", list(self.rgb_size))
            item.setdefault("possible_same_object_ids", [])
            if item["is_baseline"]:
                if item["bbox"] != self.c_bbox or item["role"] != "target":
                    raise ValueError("baseline candidate disagrees with c_bbox")
                self.aliases[candidate_id] = "KEEP"
                self._baseline = item
            else:
                self._candidates[candidate_id] = item
            self._next_id = max(self._next_id, candidate_id + 1)
        if len(self.aliases) != 1:
            raise ValueError("candidate row needs exactly one baseline alias")

    def get(self, candidate_id: int | str) -> dict[str, Any]:
        candidate_id = _normalize_id(candidate_id)
        if candidate_id == "KEEP" or candidate_id in self.aliases:
            return self._baseline
        if candidate_id not in self._candidates:
            raise ProtocolError(f"unknown candidate ID: {candidate_id}")
        return self._candidates[candidate_id]

    def require_public(self, candidate_id: int | str) -> dict[str, Any]:
        canonical = _normalize_id(candidate_id)
        if canonical in self.aliases:
            raise ProtocolError("baseline numeric alias is internal; use KEEP")
        return self.get(canonical)

    def public_candidates(self) -> list[dict[str, Any]]:
        def expose(item: dict[str, Any], candidate_id: int | str) -> dict[str, Any]:
            sources = [
                {key: deepcopy(source[key]) for key in (
                    "modality", "score", "category", "projection", "bbox", "region_px",
                    "coordinate_frame", "coordinate_size", "cue_modality", "cue_candidate_id",
                ) if key in source}
                for source in item.get("sources", [])
            ]
            return {
                "id": str(candidate_id),
                "role": item["role"],
                "role_is_hypothesis": True,
                "bbox": list(item["bbox"]),
                "coordinate_frame": item.get("coordinate_frame", "rgb"),
                "coordinate_size": list(item.get("coordinate_size", self.rgb_size)),
                "finish_eligible": item.get("coordinate_frame", "rgb") == "rgb",
                "sources": sources,
                # Box overlap is not an object-identity judgment.
                "possible_same_object_ids": [],
            }
        return [expose(self._baseline, "KEEP"), *(
            expose(item, key) for key, item in sorted(self._candidates.items())
        )]

    def snapshot(self) -> dict[str, Any]:
        return {
            "c_bbox": list(self.c_bbox),
            "candidates": self.public_candidates(),
            "source_candidates": [deepcopy(self._baseline), *(deepcopy(item) for _, item in sorted(self._candidates.items()))],
            "aliases": {str(key): value for key, value in self.aliases.items()},
            "next_id": self._next_id,
        }

    def finish(self, candidate_id: int | str) -> list[float]:
        candidate_id = _normalize_id(candidate_id)
        if candidate_id == "KEEP":
            return list(self.c_bbox)
        if candidate_id in self.aliases:
            raise ProtocolError("baseline numeric alias is internal; use KEEP")
        item = self.get(candidate_id)
        if item.get("coordinate_frame", "rgb") != "rgb":
            raise ProtocolError("IR candidate is a search-region cue; confirm an RGB box before finish")
        return list(item["bbox"])

    def append_detection(self, detection: dict[str, Any]) -> tuple[int, bool]:
        bbox = _box(detection["bbox"])
        role = detection["role"]
        if role not in {"target", "reference"}:
            raise ValueError("candidate role must be target or reference")
        coordinate_frame = detection.get("coordinate_frame", detection.get("source", {}).get("modality", "rgb"))
        if coordinate_frame not in {"rgb", "ir"}:
            raise ValueError(f"unsupported candidate coordinate frame: {coordinate_frame}")
        coordinate_size = tuple(detection.get("coordinate_size", self.rgb_size))
        key = _pixel_bounds(bbox, coordinate_size)
        for candidate_id, item in [("KEEP", self._baseline), *self._candidates.items()]:
            item_frame = item.get("coordinate_frame", "rgb")
            item_size = tuple(item.get("coordinate_size", self.rgb_size))
            if (item_frame == coordinate_frame and item_size == coordinate_size
                    and _pixel_bounds(item["bbox"], item_size) == key):
                source = deepcopy(detection["source"])
                if source not in item.setdefault("sources", []):
                    item["sources"].append(source)
                if role != item["role"]:
                    item.setdefault("role_hypotheses", [item["role"]])
                    if role not in item["role_hypotheses"]:
                        item["role_hypotheses"].append(role)
                return str(candidate_id), False
        candidate_id = self._next_id
        self._next_id += 1
        item = {
            "id": candidate_id, "role": role, "bbox": bbox,
            "role_hypotheses": [role],
            "coordinate_frame": coordinate_frame,
            "coordinate_size": list(coordinate_size),
            "is_baseline": False, "sources": [deepcopy(detection["source"])],
            "depth": {"status": "pending", "reason": "new_detection"},
            "mask_path": None, "possible_same_object_ids": [],
        }
        self._candidates[candidate_id] = item
        return str(candidate_id), True


class VisualTools:
    def __init__(
        self, manifest_row: dict[str, Any], pool: CandidatePool,
        output_dir: str | Path, tool_pixels: int,
        dino_model: Any = None, sam_model: Any = None,
        evidence_row: dict[str, Any] | None = None,
        max_searches: int = 1, max_new_candidates: int = 12,
        allow_search: bool = True,
    ):
        self.row = manifest_row
        self.pool = pool
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tool_pixels = int(tool_pixels)
        self.dino_model = dino_model
        self.sam_model = sam_model
        self.evidence_row = evidence_row
        self.max_searches = max_searches
        self.max_new_candidates = max_new_candidates
        self.allow_search = allow_search
        self.searches = 0
        self.new_candidates = 0
        self.calls = 0
        self._images: dict[str, Image.Image] = {}
        self._depth_raw: np.ndarray | None = None
        self._fresh_depth: dict[str, dict[str, Any]] = {}
        self.cost_counts = {key: 0 for key in (
            "depth_cache_hits", "depth_generated", "dino_calls", "sam_loads", "dino_loads"
        )}
        self.cost_seconds = {key: 0.0 for key in ("depth_cached", "depth_generated", "loads")}

    def _same_depth_identity(self, source: dict[str, Any]) -> bool:
        return (
            source.get("id") == self.row["id"]
            and source.get("images", {}).get("rgb") == self.row["images"]["rgb"]
            and source.get("images", {}).get("depth_raw") == self.row["images"].get("depth_raw")
            and source.get("depth_encoding") == self.row.get("depth_encoding")
        )

    def _image(self, modality: str) -> Image.Image:
        if modality not in self._images:
            key = "depth_visual" if modality == "depth" else modality
            with Image.open(self.row["images"][key]) as source:
                self._images[modality] = source.convert("RGB").copy()
        return self._images[modality]

    def _save_view(
        self, image: Image.Image, modality: str, bounds: tuple[int, int, int, int],
        candidate_ids: list[int | str], name: str, pixel_budget: int,
    ) -> dict[str, Any]:
        crop = image.crop(bounds)
        source_crop_size = crop.size
        crop = _fit_pixels(crop, pixel_budget)
        target_boxes = [
            {"id": str(candidate_id), "candidate_id": str(candidate_id),
             "bbox": _local_box(self.pool.get(candidate_id)["bbox"], bounds, image.size),
             "xyxy": _local_box(self.pool.get(candidate_id)["bbox"], bounds, image.size)}
            for candidate_id in candidate_ids
        ]
        draw = ImageDraw.Draw(crop)
        font = ImageFont.load_default(size=16)
        for target in target_boxes:
            x1, y1, x2, y2 = target["xyxy"]
            role = self.pool.get(target["candidate_id"])["role"]
            draw.rectangle(
                (round(x1 * crop.width), round(y1 * crop.height),
                 round(x2 * crop.width), round(y2 * crop.height)),
                outline="#ff3030" if role == "target" else "#00a6ff", width=max(2, round(crop.width / 200)),
            )
            draw.text((round(x1 * crop.width), round(y1 * crop.height)),
                      str(target["candidate_id"]), font=font, fill="#ffffff", stroke_width=2,
                      stroke_fill="#000000")
        path = self.output_dir / f"{self.calls:03d}_{name}.png"
        crop.save(path)
        return {
            "path": str(path.resolve()), "modality": modality,
            "candidate_ids": [str(value) for value in candidate_ids], "crop_xyxy": list(bounds),
            "candidate_roles": {str(value): self.pool.get(value)["role"] for value in candidate_ids},
            "source_size": list(image.size), "source_crop_size": list(source_crop_size),
            "saved_size": list(crop.size), "interpolated": crop.size[0] > source_crop_size[0],
            "target_boxes": target_boxes,
            "source_bbox": {str(i): list(self.pool.get(i)["bbox"]) for i in candidate_ids},
        }

    def execute(self, action: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        tool = action.get("action")
        if tool == "inspect_regions":
            return self.inspect_regions(action)
        if tool == "measure_depth":
            return self.measure_depth(action)
        if tool == "search_candidates":
            return self.search_candidates(action)
        if tool == "finish":
            candidate_id = action.get("candidate_id")
            try:
                bbox = self.pool.finish(candidate_id)
            except ProtocolError as exc:
                return _protocol(str(exc))
            return {"status": "OK", "text": f"finish {candidate_id}", "images": [],
                    "data": {"candidate_id": candidate_id, "bbox": bbox}}
        return _protocol(f"unknown action: {tool}")

    def inspect_regions(self, action: dict[str, Any]) -> dict[str, Any]:
        ids = action.get("candidate_ids", [])
        modalities = action.get("modalities", [])
        if isinstance(modalities, str):
            modalities = [part.strip() for part in modalities.split(",")]
        view = action.get("view")
        if not isinstance(ids, list) or not isinstance(modalities, list):
            return _protocol("candidate_ids and modalities must be lists")
        if len({str(candidate_id) for candidate_id in ids}) != len(ids):
            return _protocol("inspect_regions needs distinct candidate IDs")
        if view in {"single", "single_high_res"} and len(ids) == len(modalities) == 1:
            pairs = [(ids[0], modalities[0])]
            scale, minimum = 1.5, 0.06
        elif view in {"pair", "two_objects"} and len(ids) == 2 and len(modalities) == 1:
            pairs = [(candidate_id, modalities[0]) for candidate_id in ids]
            scale, minimum = 2.0, 0.12
        elif view in {"cross", "cross_modal", "one_object_cross_modal"} and 1 <= len(ids) <= 2 and 2 <= len(modalities) <= 3 and len(set(modalities)) == len(modalities):
            pairs = [(candidate_id, modality) for candidate_id in ids for modality in modalities]
            scale, minimum = 2.0, 0.12
        else:
            return _protocol("invalid inspect view, IDs, or modality list")
        if any(modality not in {"rgb", "ir", "depth"} for _, modality in pairs):
            return _protocol("unsupported modality")
        missing = [
            modality for _, modality in pairs
            if (key := "depth_visual" if modality == "depth" else modality) not in self.row["images"]
            or not Path(self.row["images"][key]).exists()
        ]
        if missing:
            return {"status": "UNKNOWN", "text": f"requested modality unavailable: {missing}",
                    "images": [], "data": {"missing_modalities": missing}}
        try:
            for candidate_id in ids:
                self.pool.require_public(candidate_id)
        except ProtocolError as exc:
            return _protocol(str(exc))
        views = []
        per_image = self.tool_pixels // len(pairs)
        for index, (candidate_id, modality) in enumerate(pairs):
            image = self._image(modality)
            bounds = _context_bounds(self.pool.get(candidate_id)["bbox"], image.size, scale, minimum)
            views.append(self._save_view(image, modality, bounds, [candidate_id],
                                         f"inspect_{index}_{modality}", per_image))
        return {
            "status": "OK", "text": "Real cropped views; positions refer to the original scene.",
            "images": views,
            "data": {"view": view, "total_saved_pixels": sum(v["saved_size"][0] * v["saved_size"][1] for v in views)},
        }

    def _cached_depth(self, candidate_id: int | str) -> dict[str, Any] | None:
        fresh = self._fresh_depth.get(str(candidate_id))
        if fresh is not None:
            return deepcopy(fresh)
        item = self.pool.get(candidate_id)
        pool_identity = {
            "id": self.pool.source_identity["id"],
            "images": {"rgb": self.pool.source_identity["rgb"],
                       "depth_raw": self.pool.source_identity["depth_raw"]},
            "depth_encoding": self.pool.source_identity["depth_encoding"],
        }
        if self._same_depth_identity(pool_identity) and item.get("depth", {}).get("status") in {"reliable", "unreliable", "unknown_encoding", "skipped_scope"}:
            return deepcopy(item["depth"])
        if self.evidence_row is None or not self._same_depth_identity(self.evidence_row):
            return None
        for cached in self.evidence_row.get("candidates", []):
            if cached["id"] == item["id"] and cached["bbox"] == item["bbox"]:
                return deepcopy(cached["depth"])
        return None

    def _sam(self) -> tuple[Any, Any, str]:
        if self.sam_model is None:
            started = time.perf_counter()
            import torch
            from transformers import Sam2Model, Sam2Processor

            model_path = self.row.get("sam_model_path", "models/sam2.1-hiera-tiny")
            processor = Sam2Processor.from_pretrained(model_path, local_files_only=True)
            model = Sam2Model.from_pretrained(model_path, local_files_only=True)
            self.sam_model = (processor, model.eval().to(device="cpu", dtype=torch.float32), "cpu")
            self.cost_counts["sam_loads"] += 1
            self.cost_seconds["loads"] += time.perf_counter() - started
        return self.sam_model

    def _dino(self) -> tuple[Any, Any, str]:
        if self.dino_model is None:
            started = time.perf_counter()
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            model_path = self.row.get("dino_model_path", "models/grounding-dino-tiny")
            processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(model_path, local_files_only=True)
            self.dino_model = (processor, model.eval().to(device="cpu", dtype=torch.float32), "cpu")
            self.cost_counts["dino_loads"] += 1
            self.cost_seconds["loads"] += time.perf_counter() - started
        return self.dino_model

    def _depth_for(self, candidate_id: int | str) -> dict[str, Any]:
        started = time.perf_counter()
        cached = self._cached_depth(candidate_id)
        if cached is not None:
            self.cost_counts["depth_cache_hits"] += 1
            self.cost_seconds["depth_cached"] += time.perf_counter() - started
            return cached
        loads_before = self.cost_seconds["loads"]
        if self._depth_raw is None:
            with Image.open(self.row["images"]["depth_raw"]) as source:
                self._depth_raw = np.asarray(source).copy()
            if self._depth_raw.dtype != np.uint16 or self._depth_raw.shape != self._image("rgb").size[::-1]:
                raise ValueError("city_mm depth must be aligned uint16")
        candidate = self.pool.get(candidate_id)
        processor, model, device = self._sam()
        (mask, quality), = segment_boxes(self._image("rgb"), [candidate], processor, model, device)
        measurement, _ = assess_depth_mask(self._depth_raw, mask, quality, depth_encoding="city_mm")
        candidate["depth"] = measurement
        self._fresh_depth[str(candidate_id)] = measurement
        self.cost_counts["depth_generated"] += 1
        self.cost_seconds["depth_generated"] += max(
            0.0, time.perf_counter() - started - (self.cost_seconds["loads"] - loads_before)
        )
        return deepcopy(measurement)

    def measure_depth(self, action: dict[str, Any]) -> dict[str, Any]:
        ids = action.get("candidate_ids", [])
        if not isinstance(ids, list) or not (1 <= len(ids) <= 2):
            return _protocol("measure_depth needs one or two distinct IDs")
        try:
            ids = [str(_normalize_id(candidate_id)) for candidate_id in ids]
            if len(set(ids)) != len(ids):
                raise ProtocolError("measure_depth needs distinct IDs")
            candidates = [self.pool.require_public(candidate_id) for candidate_id in ids]
        except ProtocolError as exc:
            return _protocol(str(exc))

        def unknown(reason: str, text: str) -> dict[str, Any]:
            measurements = {candidate_id: {"status": "UNKNOWN", "reason": reason}
                            for candidate_id in ids}
            pair = None
            if len(ids) == 2:
                pair = {"candidate_a_id": ids[0], "candidate_b_id": ids[1],
                        "status": "UNKNOWN", "reason": reason}
            return {"status": "UNKNOWN", "text": text, "images": [],
                    "data": {"reason": reason, "candidate_ids": ids,
                             "measurements": measurements, "pair": pair}}

        if any(candidate.get("coordinate_frame", "rgb") != "rgb" for candidate in candidates):
            return unknown("candidate_not_rgb_confirmed",
                           "Depth needs an RGB candidate box; IR proposals are not RGB coordinates.")
        if self.row.get("depth_encoding") != "city_mm":
            return unknown("unknown_encoding", "Raw depth encoding is not confirmed city_mm; no metric measurement.")
        depth_path = self.row["images"].get("depth_raw")
        if depth_path is None or not Path(depth_path).exists():
            return unknown("depth_raw_unavailable", "Raw depth modality unavailable.")
        if self.pool.scope != "single":
            return unknown("non_single_scope", "Depth segmentation is unsupported for this query scope.")
        measurements = {}
        for candidate_id, candidate in zip(ids, candidates, strict=True):
            result = self._depth_for(candidate_id)
            measurements[str(candidate_id)] = result
            candidate["depth"] = deepcopy(result)
        pair = None
        if len(ids) == 2:
            pair_candidates = [{**candidate, "id": str(candidate_id)} for candidate, candidate_id in zip(candidates, ids, strict=True)]
            pair = compare_depth_pair(pair_candidates[0], pair_candidates[1], "city_mm")
            pair.setdefault("candidate_a_id", ids[0])
            pair.setdefault("candidate_b_id", ids[1])
            if pair["status"] != "supported":
                pair = {"candidate_a_id": ids[0], "candidate_b_id": ids[1],
                        "status": pair["status"], "reason": pair.get("reason", "pair_not_supported")}
        reliable = all(result["status"] == "reliable" for result in measurements.values())
        supported = pair is None or pair["status"] == "supported"
        status = "OK" if reliable and supported else "UNKNOWN"
        return {
            "status": status,
            "text": "Camera-depth measurements in meters; only supported pairs establish near/far."
                    if status == "OK" else "Depth reliability or pair separation is insufficient; near/far UNKNOWN.",
            "images": [], "data": {"measurements": measurements, "pair": pair},
        }

    def _region_bounds(self, region: Any, size: tuple[int, int]) -> tuple[int, int, int, int]:
        width, height = size
        if region == "full":
            return 0, 0, width, height
        if region == "left":
            return 0, 0, ceil(width / 2), height
        if region == "right":
            return floor(width / 2), 0, width, height
        if region == "top":
            return 0, 0, width, ceil(height / 2)
        if region == "bottom":
            return 0, floor(height / 2), width, height
        if isinstance(region, str) and region.startswith("grid:"):
            parts = region.split(":")
            if len(parts) != 3 or not parts[1].isdecimal() or not parts[2].isdecimal():
                raise ProtocolError("grid region must be grid:row:col")
            region = {"kind": "grid", "row": int(parts[1]), "col": int(parts[2])}
        if isinstance(region, str) and region.startswith("context:"):
            region = {"kind": "context", "candidate_id": region.split(":", 1)[1]}
        if isinstance(region, dict) and region.get("kind") == "grid":
            try:
                col, row = int(region["col"]), int(region["row"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ProtocolError("grid region needs integer row and col") from exc
            if col not in range(3) or row not in range(3):
                raise ProtocolError("grid row and col must be 0..2")
            return floor(col * width / 3), floor(row * height / 3), ceil((col + 1) * width / 3), ceil((row + 1) * height / 3)
        if isinstance(region, dict) and region.get("kind") == "normalized_box":
            try:
                box = _box(region["bbox"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ProtocolError("normalized_box region needs normalized xyxy bbox") from exc
            return _context_bounds(box, size, float(region.get("scale", 1.6)),
                                   float(region.get("minimum_side", 0.06)))
        if isinstance(region, dict) and region.get("kind") == "context":
            if "candidate_id" not in region:
                raise ProtocolError("context region needs candidate_id")
            candidate = self.pool.require_public(region["candidate_id"])
            return _context_bounds(candidate["bbox"], size, 2.0, 0.12)
        raise ProtocolError(f"unsupported search region: {region}")

    def search_candidates(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.allow_search:
            return {"status": "LIMIT", "text": "search is unavailable for this fixed pool", "images": [], "data": {}}
        if self.searches >= self.max_searches or self.new_candidates >= self.max_new_candidates:
            return {"status": "LIMIT", "text": "search or new-candidate limit reached", "images": [], "data": {}}
        category = action.get("category")
        modality = action.get("modality")
        role = action.get("role")
        if not isinstance(category, str) or not category.strip() or modality not in {"rgb", "ir"} or role not in {"target", "reference"}:
            return _protocol("search needs category, RGB/IR modality and target/reference role")
        modality_path = self.row["images"].get(modality)
        if modality_path is None or not Path(modality_path).exists():
            return {"status": "UNKNOWN", "text": f"{modality} modality unavailable", "images": [],
                    "data": {"reason": "search_modality_unavailable", "modality": modality}}
        image = self._image(modality)
        try:
            bounds = self._region_bounds(action.get("region", "full"), image.size)
        except ProtocolError as exc:
            return _protocol(str(exc))
        self.searches += 1
        crop = image.crop(bounds)
        processor, model, device = self._dino()
        self.cost_counts["dino_calls"] += 1
        detections = detect_image_proposals(
            crop, modality, {"target_category": category.strip(), "reference_categories": []},
            processor, model, device,
        )
        detections.sort(key=lambda row: -row["score"])
        found = []
        appended = []
        new_source_ids = []
        for detection in detections[:4]:
            local = detection["bbox"]
            full_box = [
                (bounds[0] + local[0] * crop.width) / image.width,
                (bounds[1] + local[1] * crop.height) / image.height,
                (bounds[0] + local[2] * crop.width) / image.width,
                (bounds[1] + local[3] * crop.height) / image.height,
            ]
            source = {"modality": modality, "bbox": full_box, "score": detection["score"],
                      "category": category.strip(), "region_px": list(bounds),
                      "coordinate_frame": modality, "coordinate_size": list(image.size)}
            if modality == "ir":
                source["projection"] = (
                    "normalized_shared_frame_region_cue"
                    if self.row.get("ir_rgb_registration") == "normalized_shared_frame"
                    else "unregistered_ir_proposal"
                )
            cue_candidate_id = action.get("cue_candidate_id")
            if cue_candidate_id is not None:
                source["cue_modality"] = "ir"
                source["cue_candidate_id"] = str(cue_candidate_id)
            coordinate_size = list(image.size)
            frame_key = tuple(coordinate_size)
            pixel_key = _pixel_bounds(full_box, frame_key)
            matching = next((item for item in [("KEEP", self.pool._baseline), *self.pool._candidates.items()]
                             if item[1].get("coordinate_frame", "rgb") == modality
                             and tuple(item[1].get("coordinate_size", self.pool.rgb_size)) == frame_key
                             and _pixel_bounds(item[1]["bbox"], frame_key) == pixel_key), None)
            source_preexisting = bool(matching and source in matching[1].get("sources", []))
            detection_row = {"bbox": full_box, "role": role, "source": source,
                             "coordinate_frame": modality, "coordinate_size": coordinate_size}
            if self.new_candidates >= self.max_new_candidates:
                break
            candidate_id, is_new = self.pool.append_detection(detection_row)
            found.append(candidate_id)
            if is_new:
                self.new_candidates += 1
                appended.append(candidate_id)
            elif not source_preexisting:
                new_source_ids.append(candidate_id)
        if not found:
            return {"status": "EMPTY", "text": "DINO found no eligible candidates in the searched region.",
                    "images": [], "data": {"region_px": list(bounds), "category": category.strip(),
                                             "modality": modality, "found_ids": [], "appended_ids": [],
                                             "new_source_ids": [], "no_new_evidence": True}}
        view = self._save_view(image, modality, bounds, found, "search", self.tool_pixels)
        return {"status": "OK", "text": f"DINO found {len(found)} boxes; new IDs: {appended}.",
                "images": [view], "data": {"region_px": list(bounds), "category": category.strip(),
                                           "modality": modality, "role": role,
                                           "found_ids": found, "appended_ids": appended,
                                           "new_source_ids": new_source_ids,
                                           "no_new_evidence": not appended and not new_source_ids,
                                           "candidates": self.pool.public_candidates()}}
