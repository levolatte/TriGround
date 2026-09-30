"""Local Qwen backend. Images, geometry and token costs enter the same path."""
from __future__ import annotations

import copy
import math
import time
from pathlib import Path

from PIL import Image


class InputBudgetExceeded(ValueError):
    def __init__(self, reason, usage):
        super().__init__(reason)
        self.usage = usage


def resize_for_budget(image, max_pixels, factor=32):
    """Resize once to Qwen3's patch/merge grid; do not assume content kwargs work."""
    if max_pixels < factor * factor:
        raise ValueError("image budget must contain at least one merged patch")
    width, height = image.size
    scale = min(1.0, math.sqrt(max_pixels / (width * height)))
    new_w = max(factor, round(width * scale / factor) * factor)
    new_h = max(factor, round(height * scale / factor) * factor)
    while new_w * new_h > max_pixels:
        if new_w >= new_h and new_w > factor:
            new_w -= factor
        elif new_h > factor:
            new_h -= factor
        else:
            raise ValueError("cannot fit image grid in pixel budget")
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def prepare_image_messages(messages, default_max_pixels=602112):
    """Materialize every image in every turn as PIL, preserving image order."""
    result = copy.deepcopy(messages)
    metadata = []
    for message in result:
        if not isinstance(message.get("content"), list):
            continue
        for part in message["content"]:
            if part.get("type") != "image":
                continue
            source = part["image"]
            if isinstance(source, Image.Image):
                original = source.convert("RGB")
                path = None
            else:
                path = str(source)
                with Image.open(path) as handle:
                    original = handle.convert("RGB")
            pixels = int(part.get("max_pixels", default_max_pixels))
            resized = resize_for_budget(original, pixels)
            metadata.append({
                "path": path, "modality": part.get("modality"),
                "view": part.get("view", "global"),
                "source_size": list(original.size), "prepared_size": list(resized.size),
                "max_pixels": pixels, "target_boxes": part.get("target_boxes", []),
                "candidate_ids": part.get("candidate_ids", []),
                "source_bbox": part.get("source_bbox", {}),
            })
            part["image"] = resized
    return result, metadata


def geometry_after_processor(metadata, grids, patch_size, merge_size):
    if len(metadata) != len(grids):
        raise ValueError("image contents and processor image_grid_thw have different counts")
    result = []
    global_areas = {}
    global_sizes = {}
    for meta, grid in zip(metadata, grids):
        t, h, w = map(int, grid)
        width, height = w * patch_size, h * patch_size
        if meta.get("view") == "global":
            global_sizes[meta.get("modality")] = (width, height)
        targets = []
        boxes = meta.get("target_boxes", [])
        if isinstance(boxes, dict):
            boxes = [{"id": key, "bbox": box} for key, box in boxes.items()]
        for target in boxes:
            box = target.get("bbox", target.get("xyxy", target.get("box")))
            if box is None:
                continue
            target_id = str(target.get("id", target.get("candidate_id")))
            tw, th = (box[2] - box[0]) * width, (box[3] - box[1]) * height
            area = tw * th
            key = (meta.get("modality"), target_id)
            if meta.get("view") == "global":
                global_areas[key] = area
            reference = global_areas.get(key)
            if reference is None and target_id in meta.get("source_bbox", {}) and meta.get("modality") in global_sizes:
                source_box = meta["source_bbox"][target_id]
                global_w, global_h = global_sizes[meta.get("modality")]
                reference = (source_box[2] - source_box[0]) * global_w * (source_box[3] - source_box[1]) * global_h
            targets.append({"id": target_id, "width_px": tw, "height_px": th,
                            "area_px": area, "merged_grid_area": area / (patch_size * merge_size) ** 2,
                            "area_ratio_to_global": area / reference if reference else None,
                            "effective_zoom": bool(reference and area > reference)})
        result.append({**meta, "processed_size": [width, height], "grid_thw": [t, h, w],
                       "visual_tokens": t * h * w // merge_size ** 2, "targets": targets})
    return result


class QwenBackend:
    def __init__(self, model_path, adapter_path=None, *, device="cuda:0"):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        from tools.evaluate_pretrained_grounder import configure_torch_precision

        if not torch.cuda.is_available():
            raise RuntimeError("Qwen experiment requires an explicitly allocated CUDA device")
        configure_torch_precision()
        self.torch, self.device = torch, device
        started = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(model_path, min_pixels=1024,
                                                       max_pixels=1204224, local_files_only=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True)
        if adapter_path:
            import json
            from peft import PeftModel
            config = json.loads((Path(adapter_path) / "adapter_config.json").read_text())
            saved_base = config["base_model_name_or_path"].replace("\\", "/").rstrip("/").split("/")[-1]
            if saved_base != Path(model_path).name:
                raise ValueError(f"adapter base {saved_base} differs from local model {Path(model_path).name}")
            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False,
                                              autocast_adapter_dtype=False, local_files_only=True)
        self.model = model.eval().to(device=device, dtype=torch.bfloat16)
        self.load_seconds = time.perf_counter() - started

    def begin_sample(self):
        self.torch.cuda.reset_peak_memory_stats(self.device)

    def peak_memory(self):
        return int(self.torch.cuda.max_memory_allocated(self.device))

    def generate(self, messages, max_new_tokens, profile, remaining_visual, *, tools=None):
        started = time.perf_counter()
        real_messages, image_metadata = prepare_image_messages(messages, profile.global_pixels)
        template_options = {"tools": tools} if tools is not None else {}
        inputs = self.processor.apply_chat_template(real_messages, tokenize=True,
            add_generation_prompt=True, return_dict=True, return_tensors="pt", **template_options)
        inputs.pop("token_type_ids", None)
        grids = inputs["image_grid_thw"].tolist() if "image_grid_thw" in inputs else []
        image_processor = self.processor.image_processor
        geometry = geometry_after_processor(image_metadata, grids, image_processor.patch_size,
                                             image_processor.merge_size)
        visual = sum(item["visual_tokens"] for item in geometry)
        length = int(inputs["input_ids"].shape[-1])
        usage = {"input_tokens": length, "input_text_tokens": length - visual,
                 "visual_tokens": visual, "output_tokens": 0, "model_seconds": 0.0,
                 "preprocess_seconds": time.perf_counter() - started,
                 "image_grid_thw": grids, "image_geometry": geometry}
        if visual > profile.visual_tokens or visual > remaining_visual:
            raise InputBudgetExceeded("visual_token_budget", usage)
        if length + max_new_tokens > profile.context_tokens:
            raise InputBudgetExceeded("context_token_budget", usage)
        inputs = inputs.to(self.device)
        self.torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        self.torch.cuda.synchronize(self.device)
        output = generated[0, length:]
        raw = self.processor.decode(output, skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False).strip()
        usage.update(output_tokens=int(output.numel()), model_seconds=time.perf_counter() - started)
        return {"raw_output": raw, "usage": usage}
