from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoProcessor

from mm_grounding.checkpoint import load_model_checkpoint
from mm_grounding.config import load_config
from mm_grounding.data import GroundingDataset, NativeGroundingCollator
from mm_grounding.engine import _generation_inputs, _move, parse_bbox
from mm_grounding.model import build_grounder


def valid_box(value: object) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    if not all(isinstance(item, (int, float)) and math.isfinite(item) for item in value):
        return False
    x1, y1, x2, y2 = (float(item) for item in value)
    return 0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0


def load_progress(path: Path, expected_ids: set[str]) -> dict[str, list[float]]:
    predictions: dict[str, list[float]] = {}
    if not path.exists():
        return predictions
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        sample_id = str(row["id"])
        box = row["bbox"]
        if sample_id not in expected_ids:
            raise ValueError(f"progress line {line_number} has unknown id {sample_id!r}")
        if sample_id in predictions:
            raise ValueError(f"progress contains duplicate id {sample_id!r}")
        if not valid_box(box):
            raise ValueError(f"progress line {line_number} has invalid bbox")
        predictions[sample_id] = [float(item) for item in box]
    return predictions


def validate_submission(source: dict, submission: dict) -> None:
    if list(source) != list(submission):
        raise ValueError("submission query IDs or their order differ from the source")
    for sample_id, original in source.items():
        predicted = submission[sample_id]
        if set(predicted) != set(original) | {"bbox"}:
            raise ValueError(f"{sample_id}: fields differ from source plus bbox")
        for field, value in original.items():
            if field != "bbox" and predicted[field] != value:
                raise ValueError(f"{sample_id}: source field {field!r} was modified")
        if not valid_box(predicted.get("bbox")):
            raise ValueError(f"{sample_id}: invalid bbox {predicted.get('bbox')!r}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict normalized xyxy boxes and build competition ZIP")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Root used to resolve image paths (defaults to the queries file directory)",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--fallback-box", type=float, nargs=4, default=(0.0, 0.0, 1.0, 1.0))
    args = parser.parse_args()

    source = json.loads(args.queries.read_text(encoding="utf-8-sig"))
    if not isinstance(source, dict) or not source:
        raise ValueError("queries must be a non-empty object keyed by Query ID")
    if not valid_box(list(args.fallback_box)):
        raise ValueError("fallback-box must be a valid normalized xyxy box")

    config = load_config(args.config)
    workers = args.workers if args.workers is not None else config.data.workers
    max_new_tokens = args.max_new_tokens or config.train.max_new_tokens
    device = torch.device(args.device)
    processor = AutoProcessor.from_pretrained(
        config.model.backbone,
        min_pixels=config.data.min_pixels,
        max_pixels=config.data.max_pixels,
    )
    model = build_grounder(config.model, processor).to(device)
    load_model_checkpoint(args.checkpoint, model)
    model.eval()

    # GroundingDataset validates training targets. Add an in-memory-only dummy box;
    # the source query JSON is never rewritten during inference.
    dataset = GroundingDataset(
        args.queries,
        stage=config.stage,
        depth_scale=config.data.depth_scale,
        depth_clip=config.data.depth_clip,
    )
    if args.data_root is not None:
        dataset.root = args.data_root.resolve()
    for record in dataset.records:
        record["bbox"] = [0.0, 0.0, 1.0, 1.0]

    predictions = load_progress(args.progress, set(source))
    remaining_indices = [
        index for index, record in enumerate(dataset.records) if record["id"] not in predictions
    ]
    loader = DataLoader(
        Subset(dataset, remaining_indices),
        batch_size=config.train.val_batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=NativeGroundingCollator(processor, config.stage),
    )
    args.progress.parent.mkdir(parents=True, exist_ok=True)
    fallback_count = 0
    with args.progress.open("a", encoding="utf-8", buffering=1) as progress, torch.inference_mode():
        for batch in tqdm(loader, desc="competition inference", dynamic_ncols=True):
            batch = _move(batch, device)
            generated = model.generate(
                **_generation_inputs(batch, modalities={"ir", "depth"}),
                max_new_tokens=max_new_tokens,
            )
            prompt_length = batch["generation_input_ids"].shape[1]
            answers = processor.batch_decode(
                generated[:, prompt_length:], skip_special_tokens=True
            )
            for sample_id, answer in zip(batch["sample_id"], answers):
                box = parse_bbox(answer)
                if box is None:
                    box = list(args.fallback_box)
                    fallback_count += 1
                box = [float(item) for item in box]
                predictions[sample_id] = box
                progress.write(json.dumps({"id": sample_id, "bbox": box}) + "\n")

    missing = set(source) - set(predictions)
    if missing:
        raise RuntimeError(f"inference incomplete: {len(missing)} query IDs missing")
    submission = {
        sample_id: {**record, "bbox": predictions[sample_id]}
        for sample_id, record in source.items()
    }
    validate_submission(source, submission)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(submission, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(args.output_json, arcname=args.output_json.name)
    print(json.dumps({
        "queries": len(submission),
        "new_predictions": len(remaining_indices),
        "resumed_predictions": len(submission) - len(remaining_indices),
        "fallback_predictions_this_run": fallback_count,
        "json": str(args.output_json.resolve()),
        "json_sha256": sha256(args.output_json),
        "zip": str(args.output_zip.resolve()),
        "zip_sha256": sha256(args.output_zip),
    }, indent=2))


if __name__ == "__main__":
    main()
