from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _open_image(path: Path, mode: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert(mode).copy()


def encode_depth_image(array: np.ndarray, unit_scale: float, clip_distance: float) -> Image.Image:
    if array.ndim == 3:
        array = array[..., 0]
    distance = array.astype(np.float32) / unit_scale
    valid = np.isfinite(distance) & (distance > 0) & (distance <= clip_distance)
    normalized = np.zeros_like(distance, dtype=np.float32)
    normalized[valid] = np.log1p(distance[valid]) / np.log1p(clip_distance)
    encoded = np.stack((
        np.round(normalized * 255).astype(np.uint8),
        valid.astype(np.uint8) * 255,
        np.zeros_like(valid, dtype=np.uint8),
    ), axis=-1)
    return Image.fromarray(encoded)


class GroundingDataset(Dataset):
    """Aligned RGB/IR/depth samples with English query and normalized xyxy bbox."""

    def __init__(self, manifest, stage="joint", depth_scale=1000.0, depth_clip=20.0, **_):
        if stage != "joint":
            raise ValueError("Only aligned RGB+IR+depth data is supported")
        self.manifest = Path(manifest)
        self.root = self.manifest.parent
        self.depth_scale = depth_scale
        self.depth_clip = depth_clip
        text = self.manifest.read_text(encoding="utf-8-sig")
        if self.manifest.suffix.lower() == ".json":
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError("JSON manifest must map sample IDs to records")
            self.records = [{"id": sample_id, **record} for sample_id, record in payload.items()]
        else:
            self.records = [json.loads(line) for line in text.splitlines() if line.strip()]
        if not self.records:
            raise ValueError("Dataset is empty")

    def __len__(self):
        return len(self.records)

    def _path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    @staticmethod
    def _field(record, *names):
        for name in names:
            if record.get(name):
                return str(record[name])
        raise KeyError(f"Missing modality field; accepted names: {names}")

    def __getitem__(self, index):
        record = self.records[index]
        rgb = _open_image(self._path(self._field(record, "rgb", "visible")), "RGB")
        infrared = _open_image(self._path(self._field(record, "ir", "infrared")), "L")
        with Image.open(self._path(self._field(record, "depth"))) as depth_image:
            depth_array = np.asarray(depth_image).copy()
        bbox = torch.tensor(record["bbox"], dtype=torch.float32)
        if bbox.shape != (4,) or not torch.isfinite(bbox).all():
            raise ValueError("bbox must contain four finite values")
        if (bbox < 0).any() or (bbox > 1).any() or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            raise ValueError("bbox must be normalized valid xyxy")
        query = str(record.get("query", "")).strip()
        if not query:
            raise ValueError("query must not be empty")
        return {
            "rgb": rgb,
            "ir": infrared.resize(rgb.size, Image.Resampling.BILINEAR).convert("RGB"),
            "depth": encode_depth_image(depth_array, self.depth_scale, self.depth_clip).resize(
                rgb.size, Image.Resampling.NEAREST
            ),
            "query": query,
            "bbox": bbox,
            "sample_id": str(record.get("id", index)),
        }


class NativeGroundingCollator:
    """Create Qwen causal-LM supervision for a bbox_2d JSON answer."""

    def __init__(self, processor, stage="joint"):
        if stage != "joint":
            raise ValueError("Only joint multimodal batches are supported")
        self.processor = processor
        self.processor.tokenizer.padding_side = "right"

    @staticmethod
    def _prompt(query: str) -> str:
        return (
            f'Locate the object described by this referring expression: "{query}". '
            'Return exactly one JSON object: {"bbox_2d":[x1,y1,x2,y2]}. '
            'Use integer coordinates normalized to the range 0 to 1000.'
        )

    def _coordinate_token_mask(
        self, input_ids: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Mark answer tokens containing coordinates between '[' and ']'."""
        mask = torch.zeros_like(labels, dtype=torch.bool)
        for row in range(labels.shape[0]):
            inside_coordinates = False
            for column in torch.nonzero(labels[row].ne(-100), as_tuple=False).flatten().tolist():
                piece = self.processor.tokenizer.decode(
                    [int(input_ids[row, column])],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                content = piece
                if not inside_coordinates:
                    if "[" not in piece:
                        continue
                    content = piece.split("[", 1)[1]
                    inside_coordinates = True
                before_close = content.split("]", 1)[0]
                if any(character.isdigit() for character in before_close):
                    mask[row, column] = True
                if "]" in content:
                    inside_coordinates = False
                    break
        return mask

    def __call__(self, samples):
        prompts, full = [], []
        for sample in samples:
            user = {"role": "user", "content": [
                {"type": "image", "image": sample["rgb"]},
                {"type": "text", "text": self._prompt(sample["query"])},
            ]}
            answer = json.dumps(
                {"bbox_2d": [round(float(v) * 1000) for v in sample["bbox"]]},
                separators=(",", ":"),
            )
            prompts.append(self.processor.apply_chat_template([user], tokenize=False, add_generation_prompt=True))
            full.append(self.processor.apply_chat_template(
                [user, {"role": "assistant", "content": answer}],
                tokenize=False, add_generation_prompt=False,
            ))
        images = [sample["rgb"] for sample in samples]
        encoded = self.processor(text=full, images=images, padding=True, return_tensors="pt")
        generated = self.processor(text=prompts, images=images, padding=True, return_tensors="pt")
        labels = encoded["input_ids"].clone()
        for index, length in enumerate(generated["attention_mask"].sum(1).tolist()):
            labels[index, :int(length)] = -100
        labels[encoded["attention_mask"].eq(0)] = -100
        coordinate_mask = self._coordinate_token_mask(encoded["input_ids"], labels)
        if not coordinate_mask.any(dim=1).all():
            raise ValueError("failed to identify coordinate tokens in bbox JSON answer")
        batch = {
            "pixel_values": encoded["pixel_values"],
            "image_grid_thw": encoded["image_grid_thw"],
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
            "coordinate_mask": coordinate_mask,
            "generation_input_ids": generated["input_ids"],
            "generation_attention_mask": generated["attention_mask"],
            "bbox": torch.stack([sample["bbox"] for sample in samples]),
            "sample_id": [sample["sample_id"] for sample in samples],
        }
        for name in ("ir", "depth"):
            auxiliary = self.processor.image_processor(
                images=[sample[name] for sample in samples], return_tensors="pt"
            )
            if not torch.equal(encoded["image_grid_thw"], auxiliary["image_grid_thw"]):
                raise ValueError(f"RGB and {name} patch grids differ")
            batch[f"{name}_pixel_values"] = auxiliary["pixel_values"]
        return batch
