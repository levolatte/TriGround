"""Public P-ID tools for the object-centric TriGround controller."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import random
from math import ceil, floor, isfinite, sqrt
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .vision_tools import (
    CandidatePool,
    ProtocolError,
    VisualTools,
    _context_bounds,
    _fit_pixels,
    _local_box,
)


MAX_TOOL_CANDIDATE_IDS = 6
MAX_DETAIL_CANDIDATES = 3


def _error(text: str) -> dict[str, Any]:
    return {"status": "ERROR", "text": text, "images": [], "data": {"kind": "protocol"}}


def _list_arg(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return list(value)


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _draw_candidate_marks(
    image: Image.Image,
    boxes: list[dict[str, Any]],
    *,
    dashed: bool = False,
    tile_bounds: tuple[int, int, int, int] | None = None,
) -> None:
    if not boxes:
        return
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=14)
    width, height = image.size
    left, top, right, bottom = tile_bounds or (0, 0, width, height)
    line_width = max(1, round(min(width, height) / 700))
    line_color = "#00e6ff" if dashed else "#ffe45c"

    for box in boxes:
        x1, y1, x2, y2 = box["xyxy"]
        x1, y1, x2, y2 = (round(x1 * width), round(y1 * height),
                           round(x2 * width), round(y2 * height))
        x1, x2 = max(left, x1), min(right - 1, x2)
        y1, y2 = max(top, y1), min(bottom - 1, y2)
        if x1 >= x2 or y1 >= y2:
            continue
        if dashed:
            dash = max(3, line_width * 3)
            for x in range(x1, x2 + 1, dash * 2):
                draw.line((x, y1, min(x + dash, x2), y1), fill=line_color, width=line_width)
                draw.line((x, y2, min(x + dash, x2), y2), fill=line_color, width=line_width)
            for y in range(y1, y2 + 1, dash * 2):
                draw.line((x1, y, x1, min(y + dash, y2)), fill=line_color, width=line_width)
                draw.line((x2, y, x2, min(y + dash, y2)), fill=line_color, width=line_width)
        else:
            draw.rectangle((x1, y1, x2, y2), outline=line_color, width=line_width)

        label = str(box["id"])
        text = draw.textbbox((0, 0), label, font=font)
        label_width, label_height = text[2] - text[0] + 6, text[3] - text[1] + 4
        label_x = min(max(left, x1), max(left, right - label_width))
        if y1 - label_height - 2 >= top:
            label_y = y1 - label_height - 2
        elif y2 + label_height + 2 <= bottom:
            label_y = y2 + 2
        else:
            label_y = top
            label_x = left
        draw.rectangle((label_x, label_y, label_x + label_width, label_y + label_height),
                       fill="#111111", outline=line_color, width=1)
        draw.text((label_x + 3, label_y + 1), label, font=font, fill="#ffffff")


class _ObjectVisualTools(VisualTools):
    def _save_view(self, image, modality, bounds, candidate_ids, name, pixel_budget):
        crop = image.crop(bounds)
        source_crop_size = crop.size
        crop = _fit_pixels(crop, max(1, pixel_budget))
        target_boxes = []
        for candidate_id in candidate_ids:
            candidate = self.pool.get(candidate_id)
            if candidate.get("coordinate_frame", "rgb") != modality:
                continue
            local = _local_box(candidate["bbox"], bounds, image.size)
            target_boxes.append({"id": str(candidate_id), "candidate_id": str(candidate_id),
                                 "bbox": local, "xyxy": local})
        _draw_candidate_marks(crop, target_boxes)
        path = self.output_dir / f"{self.calls:03d}_{name}.png"
        crop.save(path)
        return {
            "path": str(path.resolve()), "modality": modality,
            "candidate_ids": [str(value) for value in candidate_ids],
            "crop_xyxy": list(bounds),
            "candidate_roles": {str(value): self.pool.get(value)["role"] for value in candidate_ids},
            "candidate_coordinate_frames": {str(value): self.pool.get(value).get("coordinate_frame", "rgb")
                                             for value in candidate_ids},
            "source_size": list(image.size), "source_crop_size": list(source_crop_size),
            "saved_size": list(crop.size),
            "interpolated": crop.width > source_crop_size[0] or crop.height > source_crop_size[1],
            "target_boxes": target_boxes,
            "source_bbox": {str(i): list(self.pool.get(i)["bbox"]) for i in candidate_ids},
        }

    def _region_bounds(self, region: Any, size: tuple[int, int]) -> tuple[int, int, int, int]:
        if isinstance(region, dict) and region.get("kind") == "object_quadrant":
            width, height = size
            row, col = int(region["row"]), int(region["col"])
            middle_x, middle_y = (width + 1) // 2, (height + 1) // 2
            return (
                0 if col == 0 else middle_x,
                0 if row == 0 else middle_y,
                middle_x if col == 0 else width,
                middle_y if row == 0 else height,
            )
        return super()._region_bounds(region, size)


class ObjectTools:
    """Expose real scene evidence while keeping pool IDs private.

    Every observation has ``status``, ``text``, ``images`` and ``data`` keys.
    Candidate IDs name box proposals, not physical-object identities. Returned
    views use thin box marks with labels outside the candidate box.
    """

    max_actions = 6
    atlas_pixels = 196608

    def __init__(
        self,
        row: dict[str, Any],
        pool: CandidatePool,
        output_dir: str | Path,
        tool_pixels: int = 602112,
        dino_model: Any = None,
        sam_model: Any = None,
        evidence_row: dict[str, Any] | None = None,
        seed: int = 2026,
    ):
        self.row = row
        self.pool = pool
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tool_pixels = int(tool_pixels)
        self.visual = _ObjectVisualTools(
            row, pool, output_dir, tool_pixels, dino_model, sam_model, evidence_row,
            max_searches=self.max_actions * 5, max_new_candidates=12,
            allow_search=True,
        )
        self._raw_to_public: dict[str, str] = {}
        self._public_to_raw: dict[str, str] = {}
        self._view_index = 0
        self.action_count = 0
        self._observation_cache: dict[Any, dict[str, Any]] = {}
        self._search_cache: dict[Any, dict[str, Any]] = {}

        initial = [item["id"] for item in pool.public_candidates()]
        random.Random(seed).shuffle(initial)
        for index, raw_id in enumerate(initial, start=1):
            self._bind(raw_id, f"P{index}")

    @property
    def cost_counts(self) -> dict[str, int]:
        return self.visual.cost_counts

    @property
    def cost_seconds(self) -> dict[str, float]:
        return self.visual.cost_seconds

    @property
    def searches(self) -> int:
        return self.visual.searches

    def _bind(self, raw_id: Any, public_id: str | None = None) -> str:
        raw = str(raw_id)
        if raw in self._raw_to_public:
            return self._raw_to_public[raw]
        if public_id is None:
            public_id = f"P{len(self._public_to_raw) + 1}"
        self._raw_to_public[raw] = public_id
        self._public_to_raw[public_id] = raw
        return public_id

    def _raw(self, public_id: Any) -> str:
        if not isinstance(public_id, str) or public_id not in self._public_to_raw:
            raise ProtocolError(f"unknown candidate ID: {public_id}")
        return self._public_to_raw[public_id]

    def _public(self, raw_id: Any) -> str:
        return self._bind(raw_id)

    def public_candidates(self) -> list[dict[str, Any]]:
        result = []
        for item in self.pool.public_candidates():
            public_id = self._public(item["id"])
            raw = self.pool.get(item["id"])
            sources = []
            for source in raw.get("sources", []):
                item_source = deepcopy(source)
                if item_source.get("cue_candidate_id") is not None:
                    item_source["cue_candidate_id"] = self._public(item_source["cue_candidate_id"])
                sources.append(item_source)
            coordinate_frame = raw.get("coordinate_frame", "rgb")
            result.append({
                "id": public_id,
                "role": raw["role"],
                "role_is_hypothesis": True,
                "bbox": list(raw["bbox"]),
                "coordinate_frame": coordinate_frame,
                "coordinate_size": list(raw.get("coordinate_size", self.pool.rgb_size)),
                "finish_eligible": coordinate_frame == "rgb",
                "candidate_kind": "box_proposal",
                "sources": sources,
                "possible_same_object_ids": [],
            })
        return sorted(result, key=lambda item: int(item["id"][1:]))

    def _normalize_modalities(self, modalities: Any) -> list[str]:
        values = _list_arg(modalities)
        if not values:
            values = ["rgb"]
        if len(set(values)) != len(values) or any(value not in {"rgb", "ir", "depth"} for value in values):
            raise ProtocolError("modalities must be distinct values from rgb, ir, depth")
        return values

    def _normalize_ids(self, ids: Any, *, none_means_all: bool = False) -> list[str]:
        if ids is None and none_means_all:
            ids = [item["id"] for item in self.public_candidates()]
        values = _list_arg(ids)
        if len(set(values)) != len(values):
            raise ProtocolError("candidate IDs must be distinct")
        for value in values:
            self._raw(value)
        return values

    def _modality_path(self, modality: str) -> Path | None:
        key = "depth_visual" if modality == "depth" else modality
        value = self.row.get("images", {}).get(key)
        if value is None:
            return None
        path = Path(value)
        return path if path.exists() else None

    def _missing(self, modalities: list[str]) -> list[str]:
        return [modality for modality in modalities if self._modality_path(modality) is None]

    def _global_record(self, candidate_id: str) -> dict[str, Any]:
        item = next(row for row in self.public_candidates() if row["id"] == candidate_id)
        return {key: item[key] for key in (
            "id", "role", "role_is_hypothesis", "bbox", "coordinate_frame",
            "coordinate_size", "finish_eligible", "candidate_kind", "sources",
        )}

    def _registered_scene_pair(self) -> bool:
        return self.row.get("ir_rgb_registration") == "normalized_shared_frame"

    def _joint_candidate_ids(self, candidate_ids: list[str], modality: str) -> tuple[list[str], list[str]]:
        """Return IDs representable in the RGB-aligned joint view and those omitted."""
        if modality == "ir" and not self._registered_scene_pair():
            return [], list(candidate_ids)
        visible, unmapped = [], []
        for candidate_id in candidate_ids:
            candidate = self.pool.get(self._raw(candidate_id))
            frame = candidate.get("coordinate_frame", "rgb")
            mapping = self._candidate_mapping(candidate_id, modality)
            # Joint bounds are defined in the original RGB coordinate frame.
            # Unregistered IR boxes cannot contribute to a depth or RGB union.
            rgb_aligned = frame == "rgb" or (frame == "ir" and self._registered_scene_pair())
            if mapping is None or not rgb_aligned:
                unmapped.append(candidate_id)
            else:
                visible.append(candidate_id)
        return visible, unmapped

    def _joint_bounds(self, candidate_ids: list[str], size: tuple[int, int]) -> tuple[int, int, int, int]:
        if not candidate_ids:
            return (0, 0, size[0], size[1])
        boxes = [self.pool.get(self._raw(candidate_id))["bbox"] for candidate_id in candidate_ids]
        union = [min(box[0] for box in boxes), min(box[1] for box in boxes),
                 max(box[2] for box in boxes), max(box[3] for box in boxes)]
        return _context_bounds(union, size, 2.0, 0.12)

    def _candidate_mapping(self, candidate_id: str, modality: str) -> str | None:
        candidate = self.pool.get(self._raw(candidate_id))
        frame = candidate.get("coordinate_frame", "rgb")
        if frame == modality:
            return "direct"
        if modality in {"rgb", "ir"} and frame in {"rgb", "ir"} and self._registered_scene_pair():
            return "normalized_shared_frame_region_cue"
        if modality == "depth" and self.row.get("depth_encoding") == "city_mm":
            return "aligned_depth_region"
        return None

    def _plain_view(
        self,
        modality: str,
        bounds: tuple[int, int, int, int],
        candidate_ids: list[str],
        pixel_budget: int,
        name: str,
        *,
        requested_ids: list[str] | None = None,
        view: str = "crop",
    ) -> dict[str, Any]:
        image = self.visual._image(modality)
        crop = image.crop(bounds)
        source_crop_size = crop.size
        crop = _fit_pixels(crop, max(1, pixel_budget))
        targets = []
        source_bbox = {}
        source_coordinate_frame = {}
        roles = {}
        visible_ids = []
        target_marks = []
        region_marks = []
        mapping_status = {}
        for candidate_id in candidate_ids:
            raw_id = self._raw(candidate_id)
            item = self.pool.get(raw_id)
            mapping = self._candidate_mapping(candidate_id, modality)
            mapping_status[candidate_id] = mapping or "unregistered"
            if mapping is None:
                continue
            local = _local_box(item["bbox"], bounds, image.size)
            if local[0] >= local[2] or local[1] >= local[3]:
                continue
            target = {"id": candidate_id, "candidate_id": candidate_id,
                      "bbox": local, "xyxy": local}
            if mapping == "direct" or mapping == "aligned_depth_region":
                targets.append(target)
                target_marks.append(target)
            else:
                region_marks.append(target)
            visible_ids.append(candidate_id)
            source_bbox[candidate_id] = list(item["bbox"])
            source_coordinate_frame[candidate_id] = item.get("coordinate_frame", "rgb")
            roles[candidate_id] = item["role"]
        _draw_candidate_marks(crop, target_marks)
        _draw_candidate_marks(crop, region_marks, dashed=True)
        self._view_index += 1
        path = self.output_dir / f"object_{self._view_index:04d}_{name}.png"
        crop.save(path)
        return {
            "path": str(path.resolve()),
            "modality": modality,
            "view": view if visible_ids else "unregistered_global",
            "candidate_ids": visible_ids,
            "requested_candidate_ids": list(requested_ids if requested_ids is not None else candidate_ids),
            "labels": [self._global_record(candidate_id) for candidate_id in visible_ids],
            "crop_xyxy": list(bounds),
            "candidate_roles": roles,
            "candidate_coordinate_frames": source_coordinate_frame,
            "coordinate_mapping": mapping_status,
            "registration_source": self.row.get("registration_source"),
            "source_size": list(image.size),
            "source_crop_size": list(source_crop_size),
            "saved_size": list(crop.size),
            "interpolated": crop.width > source_crop_size[0] or crop.height > source_crop_size[1],
            "target_boxes": targets,
            "region_cues": region_marks,
            "source_bbox": source_bbox,
            "source_coordinate_frame": source_coordinate_frame,
        }

    def _atlas_board(
        self,
        modality: str,
        candidate_ids: list[str],
        pixel_budget: int,
        page_index: int,
    ) -> dict[str, Any]:
        if not candidate_ids:
            image = self.visual._image(modality)
            return self._plain_view(modality, (0, 0, image.width, image.height), [],
                                    pixel_budget, f"atlas_{modality}_{page_index}")

        image = self.visual._image(modality)
        columns = min(3, len(candidate_ids))
        rows = ceil(len(candidate_ids) / columns)
        aspect = image.width / image.height
        tile_width = max(1, floor(sqrt(max(1, pixel_budget) * aspect / (columns * rows))))
        tile_height = max(1, floor(tile_width / aspect))
        while columns * tile_width * rows * tile_height > pixel_budget:
            if tile_width / aspect >= tile_height and tile_width > 1:
                tile_width -= 1
            elif tile_height > 1:
                tile_height -= 1
            else:
                break
        board = Image.new("RGB", (columns * tile_width, rows * tile_height), (0, 0, 0))

        targets = []
        tile_positions = {}
        source_bbox = {}
        crop_bounds = {}
        crop_sizes = {}
        roles = {}
        overlay_tiles = {}
        resized_any = False
        for index, candidate_id in enumerate(candidate_ids):
            row_index, col_index = divmod(index, columns)
            raw_id = self._raw(candidate_id)
            candidate = self.pool.get(raw_id)
            bounds = _context_bounds(candidate["bbox"], image.size, 2.0, 0.12)
            crop = image.crop(bounds)
            crop_bounds[candidate_id] = list(bounds)
            crop_sizes[candidate_id] = list(crop.size)
            scale = min(tile_width / crop.width, tile_height / crop.height)
            resized_width = max(1, min(tile_width, round(crop.width * scale)))
            resized_height = max(1, min(tile_height, round(crop.height * scale)))
            resized = crop.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
            resized_any |= resized.size[0] > crop.width or resized.size[1] > crop.height
            pad_x = (tile_width - resized_width) // 2
            pad_y = (tile_height - resized_height) // 2
            tile_left, tile_top = col_index * tile_width, row_index * tile_height
            board.paste(resized, (tile_left + pad_x, tile_top + pad_y))

            local = _local_box(candidate["bbox"], bounds, image.size)
            box = [
                (tile_left + pad_x + local[0] * resized_width) / board.width,
                (tile_top + pad_y + local[1] * resized_height) / board.height,
                (tile_left + pad_x + local[2] * resized_width) / board.width,
                (tile_top + pad_y + local[3] * resized_height) / board.height,
            ]
            targets.append({"id": candidate_id, "candidate_id": candidate_id,
                            "bbox": box, "xyxy": box})
            tile_cell = [tile_left / board.width, tile_top / board.height,
                         (tile_left + tile_width) / board.width,
                         (tile_top + tile_height) / board.height]
            tile_positions[candidate_id] = {
                "row": row_index, "col": col_index, "cell_xyxy": tile_cell,
                "candidate_xyxy": box,
            }
            overlay_tiles[candidate_id] = (
                tile_left, tile_top, tile_left + tile_width, tile_top + tile_height,
            )
            source_bbox[candidate_id] = list(candidate["bbox"])
            roles[candidate_id] = candidate["role"]

        for target in targets:
            _draw_candidate_marks(board, [target], tile_bounds=overlay_tiles[target["candidate_id"]])

        self._view_index += 1
        path = self.output_dir / f"object_{self._view_index:04d}_atlas_{modality}_{page_index}.png"
        board.save(path)
        labels = []
        for candidate_id in candidate_ids:
            record = self._global_record(candidate_id)
            record["tile_position"] = tile_positions[candidate_id]
            labels.append(record)
        return {
            "path": str(path.resolve()), "modality": modality, "view": "atlas",
            "candidate_ids": list(candidate_ids), "labels": labels,
            "tile_positions": tile_positions,
            "candidate_roles": roles, "source_size": list(image.size),
            "source_crop_bounds": crop_bounds, "source_crop_sizes": crop_sizes,
            "saved_size": list(board.size), "interpolated": resized_any,
            "target_boxes": targets, "source_bbox": source_bbox,
            "source_coordinate_frame": {str(value): "rgb" for value in candidate_ids},
        }

    def _region_bounds(
        self,
        region: Any,
        size: tuple[int, int],
        candidate_id: str | None,
        args: dict[str, Any],
    ) -> tuple[int, int, int, int]:
        if region == "context":
            if candidate_id is None:
                raise ProtocolError("context region needs a candidate ID")
            return _context_bounds(self.pool.get(self._raw(candidate_id))["bbox"], size, 2.0, 0.12)
        if isinstance(region, str) and region.startswith("context:"):
            public_id = region.split(":", 1)[1]
            return _context_bounds(self.pool.get(self._raw(public_id))["bbox"], size, 2.0, 0.12)
        if region == "grid":
            region = {"kind": "grid", "row": args["row"], "col": args["col"]}
        if isinstance(region, dict) and region.get("kind") == "context":
            public_id = region.get("candidate_id", candidate_id)
            region = {**region, "candidate_id": self._raw(public_id)}
        return self.visual._region_bounds(region, size)

    def atlas(self, candidate_ids: Any = None, modalities: Any = None) -> dict[str, Any]:
        try:
            requested_ids = self._normalize_ids(candidate_ids, none_means_all=True)
            modes = self._normalize_modalities(modalities)
        except (ProtocolError, TypeError) as exc:
            return _error(str(exc))
        if "rgb" not in modes:
            return _error("atlas is RGB-only; inspect auxiliary modalities separately")
        skipped_ids = [value for value in requested_ids
                       if self.pool.get(self._raw(value)).get("coordinate_frame", "rgb") != "rgb"]
        ids = [value for value in requested_ids if value not in skipped_ids]
        if not ids:
            return {"status": "UNKNOWN", "text": "No RGB-frame candidate boxes are available for the atlas.",
                    "images": [], "data": {"skipped_non_rgb_candidate_ids": skipped_ids}}
        missing = self._missing(["rgb"])
        if missing:
            return {"status": "UNKNOWN", "text": f"Requested modality unavailable: {missing}.",
                    "images": [], "data": {"missing_modalities": missing}}

        pages = [ids[index:index + 6] for index in range(0, len(ids), 6)] or [[]]
        images = []
        page_data = []
        per_page_budget = self.atlas_pixels // len(pages)
        for page_index, page_ids in enumerate(pages):
            metadata = self._atlas_board("rgb", page_ids, per_page_budget, page_index + 1)
            images.append(metadata)
            page_data.append({"modality": "rgb", "candidate_ids": page_ids,
                              "candidates": metadata.get("labels", []),
                              "tile_positions": metadata.get("tile_positions", {})})
        return {
            "status": "OK",
            "text": "RGB candidate crops with thin box outlines and stable IDs placed outside each box; at most six candidates per page.",
            "images": images,
            "data": {"pages": page_data, "candidates": [self._global_record(value) for value in ids],
                     "skipped_non_rgb_candidate_ids": skipped_ids,
                     "pixel_budget": self.atlas_pixels,
                     "total_saved_pixels": sum(item["saved_size"][0] * item["saved_size"][1] for item in images)},
        }

    def inspect(
        self,
        candidate_ids: Any = None,
        region: Any = None,
        modalities: Any = None,
        **region_args: Any,
    ) -> dict[str, Any]:
        try:
            ids = self._normalize_ids(candidate_ids)
            modes = self._normalize_modalities(modalities)
        except (ProtocolError, TypeError) as exc:
            return _error(str(exc))
        if len(ids) > MAX_TOOL_CANDIDATE_IDS:
            return _error("inspect accepts at most six candidate IDs")
        missing = self._missing(modes)
        if missing:
            return {"status": "UNKNOWN", "text": f"Requested modality unavailable: {missing}.",
                    "images": [], "data": {"missing_modalities": missing}}

        if region is None:
            region = "full" if len(ids) > MAX_DETAIL_CANDIDATES else ("context" if ids else "full")
        if region == "joint" and not ids:
            return _error("joint region needs one to six candidate IDs")
        if region == "context" and len(ids) > MAX_DETAIL_CANDIDATES:
            return _error("context views accept at most three detailed candidates; use a shared scene region")
        context_views = region == "context" and bool(ids)
        if region == "context" and not ids:
            return _error("context region needs candidate IDs")
        cache_key = ("inspect", tuple(ids), _freeze(region), tuple(modes))
        if cache_key in self._observation_cache:
            cached = deepcopy(self._observation_cache[cache_key])
            cached["data"]["cached"] = True
            cached["data"]["new_evidence"] = False
            cached["text"] += " Repeated identical view; no new visual evidence."
            return cached

        requests = [(candidate_id, [candidate_id]) for candidate_id in ids] if context_views else [(None, ids)]
        tasks = []
        unregistered_full = set()
        for focus_id, page_ids in requests:
            for modality in modes:
                image = self.visual._image(modality)
                try:
                    if region == "joint":
                        visible_ids, unmapped_ids = self._joint_candidate_ids(page_ids, modality)
                        bounds = self._joint_bounds(visible_ids, image.size)
                        view_kind = "joint" if visible_ids else "unregistered_global"
                    elif focus_id is not None and context_views:
                        unmapped_ids = []
                        mapping = self._candidate_mapping(focus_id, modality)
                        if mapping is None:
                            if modality in unregistered_full:
                                continue
                            unregistered_full.add(modality)
                            bounds = (0, 0, image.width, image.height)
                            visible_ids = []
                            view_kind = "unregistered_global"
                        else:
                            bounds = _context_bounds(
                                self.pool.get(self._raw(focus_id))["bbox"], image.size, 2.0, 0.12
                            )
                            visible_ids = [focus_id]
                            view_kind = "crop" if mapping == "direct" else "region_cue"
                    else:
                        unmapped_ids = []
                        bounds = self._region_bounds(region, image.size, focus_id, region_args)
                        visible_ids = page_ids
                        view_kind = "crop"
                except (ProtocolError, KeyError, TypeError, ValueError) as exc:
                    return _error(str(exc))
                if region != "joint" and not context_views and len(page_ids) > 1:
                    view_kind = "shared_scene_region"
                tasks.append((focus_id, page_ids, modality, bounds, visible_ids, view_kind, unmapped_ids))

        image_count = max(1, len(tasks))
        images = []
        region_data = []
        for focus_id, page_ids, modality, bounds, visible_ids, view_kind, unmapped_ids in tasks:
            metadata = self._plain_view(
                modality, bounds, visible_ids,
                self.tool_pixels // image_count,
                f"inspect_{len(images) + 1}_{modality}",
                requested_ids=page_ids, view=view_kind,
            )
            images.append(metadata)
            region_data.append({"modality": modality, "candidate_ids": metadata["candidate_ids"],
                                "requested_candidate_ids": page_ids, "region_px": list(bounds),
                                "view": metadata["view"],
                                "unmapped_candidate_ids": unmapped_ids,
                                "coordinate_mapping": metadata["coordinate_mapping"]})
        result = {
            "status": "OK",
            "text": "All requested IDs share each joint/region view; context mode returns separate detail crops. Joint bounds use the union of selected RGB-coordinate candidates plus context. Boxes are outlined only in their native modality. Unregistered IR remains a full-frame view without projected RGB boxes.",
            "images": images,
            "data": {"regions": region_data,
                     "candidates": [self._global_record(value) for value in ids],
                     "cached": False, "new_evidence": bool(images),
                     "total_saved_pixels": sum(item["saved_size"][0] * item["saved_size"][1] for item in images)},
        }
        self._observation_cache[cache_key] = deepcopy(result)
        return result

    def depth(self, ids: Any = None) -> dict[str, Any]:
        try:
            public_ids = self._normalize_ids(ids)
            if not 1 <= len(public_ids) <= MAX_TOOL_CANDIDATE_IDS:
                raise ProtocolError("depth needs one to six candidate IDs")
        except (ProtocolError, TypeError) as exc:
            return _error(str(exc))
        raw_ids = [self._raw(value) for value in public_ids]
        cache_key = ("depth", tuple(
            (public_id, tuple(self.pool.get(raw_id)["bbox"]), self.pool.get(raw_id).get("mask_path"))
            for public_id, raw_id in zip(public_ids, raw_ids)
        ))
        if cache_key in self._observation_cache:
            cached = deepcopy(self._observation_cache[cache_key])
            cached["data"]["cached"] = True
            cached["data"]["new_evidence"] = False
            cached["data"]["no_new_evidence"] = True
            cached["text"] += " Repeated identical depth request; no new evidence."
            return cached
        compact = {}
        measurement_status = {}
        measurement_reason = {}
        for raw_id in raw_ids:
            observation = self.visual.execute({"action": "measure_depth", "candidate_ids": [raw_id]})
            raw_data = observation.get("data", {})
            for measured_id, measurement in raw_data.get("measurements", {}).items():
                public_id = self._public(measured_id)
                measurement_status[public_id] = measurement.get("status")
                measurement_reason[public_id] = measurement.get("reason", raw_data.get("reason"))
                compact[public_id] = {
                    "status": measurement.get("status"),
                    "reason": measurement.get("reason", raw_data.get("reason")),
                    "core": {key: measurement.get("core", {}).get(key)
                             for key in ("valid_count", "valid_ratio", "q1_m", "median_m", "q3_m")
                             if key in measurement.get("core", {})},
                    "full": {key: measurement.get("full", {}).get(key)
                             for key in ("valid_count", "valid_ratio", "median_m")
                             if key in measurement.get("full", {})},
                }

        pair_results = []
        for left_index in range(len(raw_ids)):
            for right_index in range(left_index + 1, len(raw_ids)):
                left_public, right_public = public_ids[left_index], public_ids[right_index]
                if (measurement_status.get(left_public) != "reliable"
                        or measurement_status.get(right_public) != "reliable"):
                    details = [f"{candidate_id}:{measurement_reason.get(candidate_id) or 'unavailable'}"
                               for candidate_id in (left_public, right_public)
                               if measurement_status.get(candidate_id) != "reliable"]
                    pair_results.append({
                        "candidate_a_id": left_public, "candidate_b_id": right_public,
                        "status": "UNKNOWN", "reason": "candidate_depth_unreliable",
                        "detail": "; ".join(details),
                    })
                    continue
                observation = self.visual.execute({
                    "action": "measure_depth", "candidate_ids": [raw_ids[left_index], raw_ids[right_index]],
                })
                pair = observation.get("data", {}).get("pair")
                if pair is None:
                    pair = {"status": "UNKNOWN", "reason": observation.get("data", {}).get(
                        "reason", "pair_measurement_unavailable")}
                else:
                    pair = dict(pair)
                    for key in ("candidate_a_id", "candidate_b_id", "target_candidate_id",
                                "reference_candidate_id", "near_candidate_id", "far_candidate_id"):
                        if key in pair:
                            pair[key] = self._public(pair[key])
                    if observation["status"] != "OK" and pair.get("status") == "supported":
                        pair["status"] = "UNKNOWN"
                        pair["reason"] = observation.get("data", {}).get(
                            "reason", "pair_measurement_unreliable")
                pair.setdefault("candidate_a_id", left_public)
                pair.setdefault("candidate_b_id", right_public)
                if pair.get("status") != "supported":
                    pair["status"] = "UNKNOWN"
                    pair.setdefault("reason", "pair_measurement_unsupported")
                    pair.pop("near_candidate_id", None)
                    pair.pop("far_candidate_id", None)
                    pair.pop("near_median_m", None)
                    pair.pop("far_median_m", None)
                pair_results.append(pair)

        supported_pairs = [pair for pair in pair_results if pair.get("status") == "supported"]
        all_reliable = len(compact) == len(public_ids) and all(
            value.get("status") == "reliable" for value in compact.values()
        )
        all_pairs_supported = len(pair_results) == 0 or len(supported_pairs) == len(pair_results)
        status = "OK" if all_reliable and all_pairs_supported else "UNKNOWN"
        result = {
            "status": status,
            "text": "Pairwise camera-depth evidence; only listed supported pairs establish a local near/far relation."
                    if status == "OK" else "Some depth measurements or pairs are UNKNOWN; supported pairs remain usable individually.",
            "images": [],
            "data": {"measurements": compact, "pair_results": pair_results,
                     "supported_pairs": supported_pairs, "cached": False,
                     "new_evidence": True, "no_new_evidence": False},
        }
        self._observation_cache[cache_key] = deepcopy(result)
        return result

    def search(
        self,
        category: str,
        modality: str,
        role: str = "target",
        region: Any = "full",
        **region_args: Any,
    ) -> dict[str, Any]:
        try:
            if modality not in {"rgb", "ir"}:
                raise ProtocolError("search modality must be rgb or ir")
            if not isinstance(category, str) or not category.strip():
                raise ProtocolError("search needs a non-empty category")
            if region == "context" or (isinstance(region, dict) and region.get("kind") == "context"):
                if isinstance(region, str):
                    raise ProtocolError("search context region needs an explicit candidate ID")
            if region == "grid":
                region = {"kind": "grid", "row": region_args["row"], "col": region_args["col"]}
            elif isinstance(region, str) and region.startswith("context:"):
                public_id = region.split(":", 1)[1]
                region = {"kind": "context", "candidate_id": self._raw(public_id)}
            elif isinstance(region, dict) and region.get("kind") == "context":
                public_id = region["candidate_id"]
                region = {**region, "candidate_id": self._raw(public_id)}
        except (ProtocolError, KeyError, TypeError, ValueError) as exc:
            return _error(str(exc))

        if self._modality_path(modality) is None:
            return {"status": "UNKNOWN", "text": f"{modality} modality unavailable", "images": [],
                    "data": {"reason": "search_modality_unavailable", "modality": modality}}
        if isinstance(region, dict) and region.get("kind") == "context":
            candidate = self.pool.get(region["candidate_id"])
            if candidate.get("coordinate_frame", "rgb") != modality and not self._registered_scene_pair():
                return {"status": "UNKNOWN", "text": "This candidate cannot define a region in the other modality because registration is unknown.",
                        "images": [], "data": {"reason": "unregistered_cross_modal_region",
                                                 "candidate_id": self._public(region["candidate_id"]),
                                                 "modality": modality}}
        image = self.visual._image(modality)
        try:
            requested_bounds = self.visual._region_bounds(region, image.size)
        except (ProtocolError, KeyError, TypeError, ValueError) as exc:
            return _error(str(exc))
        cache_key = ("search", category.strip().casefold(), modality, role, requested_bounds)
        if cache_key in self._search_cache:
            cached = deepcopy(self._search_cache[cache_key])
            cached["data"]["cached"] = True
            cached["data"]["new_evidence"] = False
            cached["data"]["no_new_evidence"] = True
            cached["text"] += " Repeated identical search; no new visual evidence."
            return cached

        attempts = []
        first = self.visual.execute({"action": "search_candidates", "category": category,
                                      "modality": modality, "role": role, "region": region})
        attempts.append(("full" if region == "full" else region, first))
        first_data = first.get("data", {})
        first_has_new = bool(first_data.get("appended_ids") or first_data.get("new_source_ids"))
        if region == "full" and (first["status"] == "EMPTY" or
                                  first["status"] == "OK" and not first_has_new):
            quadrants = (
                ("top_left", {"kind": "object_quadrant", "row": 0, "col": 0}),
                ("top_right", {"kind": "object_quadrant", "row": 0, "col": 1}),
                ("bottom_left", {"kind": "object_quadrant", "row": 1, "col": 0}),
                ("bottom_right", {"kind": "object_quadrant", "row": 1, "col": 1}),
            )
            for quadrant, quadrant_region in quadrants:
                result = self.visual.execute({"action": "search_candidates", "category": category,
                                              "modality": modality, "role": role,
                                              "region": quadrant_region})
                attempts.append((quadrant, result))

        ir_found_raw = []
        if modality == "ir" and self._registered_scene_pair():
            for _, result in attempts:
                ir_found_raw.extend(result.get("data", {}).get("found_ids", []))
            for raw_id in dict.fromkeys(ir_found_raw):
                proposal = self.pool.get(raw_id)
                rgb_result = self.visual.execute({
                    "action": "search_candidates", "category": category,
                    "modality": "rgb", "role": role,
                    "region": {"kind": "normalized_box", "bbox": proposal["bbox"],
                               "scale": 1.6, "minimum_side": 0.06},
                    "cue_candidate_id": raw_id,
                })
                attempts.append((f"rgb_confirm_{self._public(raw_id)}", rgb_result))

        statuses = [result["status"] for _, result in attempts]
        found_raw = []
        appended_raw = []
        new_source_raw = []
        for _, result in attempts:
            found_raw.extend(result.get("data", {}).get("found_ids", []))
            appended_raw.extend(result.get("data", {}).get("appended_ids", []))
            new_source_raw.extend(result.get("data", {}).get("new_source_ids", []))
        found = list(dict.fromkeys(self._public(value) for value in found_raw))
        appended = list(dict.fromkeys(self._public(value) for value in appended_raw))
        new_source_ids = list(dict.fromkeys(self._public(value) for value in new_source_raw))
        if found:
            status = "OK"
        elif any(value in {"UNKNOWN", "LIMIT", "ERROR"} for value in statuses):
            status = next(value for value in statuses if value in {"UNKNOWN", "LIMIT", "ERROR"})
        else:
            status = "EMPTY"

        images = []
        if found:
            for index, candidate_id in enumerate(found):
                candidate = self.pool.get(self._raw(candidate_id))
                view_modality = candidate.get("coordinate_frame", "rgb")
                view_image = self.visual._image(view_modality)
                bounds = _context_bounds(candidate["bbox"], view_image.size, 2.0, 0.12)
                images.append(self._plain_view(view_modality, bounds, [candidate_id],
                                               self.tool_pixels // len(found),
                                               f"search_{candidate_id}_{modality}"))
        regions = []
        for region_name, result in attempts:
            item = {"region": region_name, "status": result["status"]}
            if "region_px" in result.get("data", {}):
                item["region_px"] = result["data"]["region_px"]
            for key in ("found_ids", "appended_ids", "no_new_evidence"):
                if key in result.get("data", {}):
                    item[key] = [self._public(value) for value in result["data"][key]] if key.endswith("_ids") else result["data"][key]
            regions.append(item)
        no_new_evidence = not appended and not new_source_ids
        rgb_confirmed_ids = [candidate_id for candidate_id in found
                             if self.pool.get(self._raw(candidate_id)).get("coordinate_frame", "rgb") == "rgb"]
        ir_proposal_ids = [candidate_id for candidate_id in found
                           if self.pool.get(self._raw(candidate_id)).get("coordinate_frame", "rgb") == "ir"]
        if found and modality == "ir":
            text = (f"Found IR proposal IDs {ir_proposal_ids}; they are not final RGB boxes. "
                    f"RGB search-region confirmation added/found IDs {rgb_confirmed_ids}. "
                    f"New IDs: {appended}; new source evidence IDs: {new_source_ids}.")
        else:
            text = (f"Found {len(found)} box proposal(s); new IDs: {appended}; "
                    f"new source evidence IDs: {new_source_ids}." if found
                else "No candidate was found in the searched region(s)." if status == "EMPTY"
                else next((result["text"] for _, result in attempts if result["status"] == status), status))
        if no_new_evidence and found:
            text += " Search returned only existing box/source evidence."
        result = {"status": status, "text": text, "images": images,
                  "data": {"category": category, "modality": modality, "role": role,
                           "role_is_hypothesis": True,
                           "regions": regions, "found_ids": found, "appended_ids": appended,
                           "new_source_ids": new_source_ids,
                           "ir_proposal_ids": ir_proposal_ids,
                           "rgb_confirmed_ids": rgb_confirmed_ids,
                           "cross_modal_identity": "unknown",
                           "no_new_evidence": no_new_evidence,
                           "new_evidence": not no_new_evidence, "cached": False,
                           "candidates": [self._global_record(value) for value in found]}}
        self._search_cache[cache_key] = deepcopy(result)
        return result

    def finish(self, candidate_id: Any = None, bbox: Any = None) -> dict[str, Any]:
        if (candidate_id is None) == (bbox is None):
            return _error("finish requires exactly one of id or bbox")
        if candidate_id is not None:
            try:
                public_id = candidate_id
                raw_id = self._raw(public_id)
                candidate_bbox = self.pool.finish(raw_id)
            except ProtocolError as exc:
                return _error(str(exc))
            return {"status": "OK", "text": f"Finished with candidate {public_id}.", "images": [],
                    "data": {"candidate_id": public_id, "bbox": candidate_bbox,
                             "finish_source": "candidate"}}

        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4 or any(isinstance(value, bool) for value in bbox):
            return _error("bbox must be normalized xyxy in the original global RGB frame")
        try:
            predicted = [float(value) for value in bbox]
        except (TypeError, ValueError, OverflowError):
            return _error("bbox must contain four finite normalized xyxy values")
        if not all(isfinite(value) for value in predicted) or not (
            0 <= predicted[0] < predicted[2] <= 1 and 0 <= predicted[1] < predicted[3] <= 1
        ):
            return _error("bbox must be legal normalized xyxy in the original global RGB frame")
        return {"status": "OK", "text": "Finished with a predicted RGB bbox.", "images": [],
                "data": {"candidate_id": None, "bbox": predicted,
                         "finish_source": "predicted_bbox"}}

    def execute(self, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        is_finish = name == "finish"
        if not is_finish and self.action_count >= self.max_actions:
            return {"status": "LIMIT", "text": "The six-action limit has been reached.",
                    "images": [], "data": {"action_count": self.action_count}}
        if not is_finish:
            self.action_count += 1
        args = {} if args is None else args
        if not isinstance(args, dict):
            return _error("tool arguments must be an object")
        if name == "atlas":
            return self.atlas(args.get("candidate_ids", args.get("ids")), args.get("modalities"))
        if name in {"inspect", "inspect_regions"}:
            return self.inspect(args.get("candidate_ids", args.get("ids")), args.get("region"),
                                args.get("modalities"), **{key: args[key] for key in ("row", "col") if key in args})
        if name in {"depth", "measure_depth"}:
            return self.depth(args.get("candidate_ids", args.get("ids")))
        if name in {"search", "search_candidates"}:
            return self.search(args.get("category"), args.get("modality"), args.get("role", "target"),
                               args.get("region", "full"),
                               **{key: args[key] for key in ("row", "col") if key in args})
        if name == "finish":
            has_id, has_bbox = "id" in args, "bbox" in args
            if has_id == has_bbox:
                return _error("finish requires exactly one of id or bbox")
            return self.finish(args.get("id") if has_id else None,
                               args.get("bbox") if has_bbox else None)
        return _error(f"unknown tool: {name}")
