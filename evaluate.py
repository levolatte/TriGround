from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor

from mm_grounding.checkpoint import load_model_checkpoint
from mm_grounding.config import load_config
from mm_grounding.data import GroundingDataset, NativeGroundingCollator
from mm_grounding.engine import evaluate
from mm_grounding.model import build_grounder


CLASS_FIELDS = ("class_name", "category", "category_name", "object_class")


def _target_size(target: list[float]) -> tuple[float, str]:
    area = max(target[2] - target[0], 0.0) * max(target[3] - target[1], 0.0)
    if area < 0.01:
        return area, "small"
    if area < 0.05:
        return area, "medium"
    return area, "large"


def _existing_query_type(record: dict):
    if record.get("query_type") is not None:
        return record["query_type"]
    metadata = record.get("candidate_metadata")
    if isinstance(metadata, dict):
        return metadata.get("query_type")
    return None


def _scene_group(record: dict) -> tuple[str, str]:
    for field in (
        "scene_id",
        "sequence_id",
        "video_id",
        "original_image_id",
        "rgb",
        "visible",
    ):
        if record.get(field) is not None:
            return field, str(record[field])
    return "id", str(record["id"])


def _evidence_rows(
    mode: str, result: dict, metadata_by_id: dict[str, dict]
) -> list[dict]:
    output = []
    for row in result["rows"]:
        record = metadata_by_id[row["id"]]
        group_source, group = _scene_group(record)
        target_area, target_size_bin = _target_size(row["target"])
        target_class = next(
            (record[field] for field in CLASS_FIELDS if record.get(field) is not None),
            None,
        )
        query_type = _existing_query_type(record)
        output.append(
            {
                "mode": mode,
                **row,
                "scene_group_source": group_source,
                "scene_group": group,
                "target_area": target_area,
                "target_size_bin": target_size_bin,
                **({"target_class": target_class} if target_class is not None else {}),
                **({"scale_bin": record["scale_bin"]} if record.get("scale_bin") is not None else {}),
                **({"query_type": query_type} if query_type is not None else {}),
            }
        )
    return output


def _stratified_subset(dataset, limit: int, seed: int):
    """Deterministic proportional sample across class and target scale."""
    if limit >= len(dataset):
        return dataset
    groups = {}
    for index, record in enumerate(dataset.records):
        key = (record.get("class_name", ""), record.get("scale_bin", ""))
        groups.setdefault(key, []).append(index)
    generator = torch.Generator().manual_seed(seed)
    for indices in groups.values():
        order = torch.randperm(len(indices), generator=generator).tolist()
        indices[:] = [indices[i] for i in order]
    exact = {key: len(indices) * limit / len(dataset) for key, indices in groups.items()}
    quotas = {key: int(value) for key, value in exact.items()}
    remaining = limit - sum(quotas.values())
    priority = sorted(
        groups,
        key=lambda key: (exact[key] - quotas[key], len(groups[key])),
        reverse=True,
    )
    for key in priority[:remaining]:
        quotas[key] += 1
    selected = []
    for key in sorted(groups):
        selected.extend(groups[key][:quotas[key]])
    return Subset(dataset, selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help=(
            "Checkpoint to load. Repeat the option to compose independently trained "
            "modality checkpoints; omit it for the native frozen backbone baseline."
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--manifest",
        help="Optional validation manifest override for paired/gated evaluation",
    )
    parser.add_argument("--output", help="Optional path for the JSON evaluation report")
    parser.add_argument(
        "--rows-output",
        help="Optional JSONL path for per-sample predictions and evidence metadata",
    )
    parser.add_argument(
        "--rgb-only",
        action="store_true",
        help="Evaluate only the RGB baseline (useful for the native Qwen baseline).",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        help="Evaluate a deterministic stratified subset of at most this many records.",
    )
    parser.add_argument("--subset-seed", type=int, default=2026)
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device(args.device)
    processor = AutoProcessor.from_pretrained(
        config.model.backbone, min_pixels=config.data.min_pixels, max_pixels=config.data.max_pixels
    )
    model = build_grounder(config.model, processor).to(device)
    loaded_checkpoints = []
    for checkpoint_path in args.checkpoint:
        load_model_checkpoint(checkpoint_path, model)
        loaded_checkpoints.append(checkpoint_path)
    manifest = args.manifest or config.data.val_manifest
    dataset = GroundingDataset(
        manifest, stage=config.stage, depth_scale=config.data.depth_scale,
        depth_clip=config.data.depth_clip,
    )
    eval_dataset = (
        _stratified_subset(dataset, args.subset_size, args.subset_seed)
        if args.subset_size is not None
        else dataset
    )
    loader = DataLoader(
        eval_dataset, batch_size=config.train.val_batch_size, shuffle=False,
        num_workers=config.data.workers,
        collate_fn=NativeGroundingCollator(processor, config.stage),
    )
    modes = {
        "ir": {"rgb_ir": {"ir"}},
        "depth": {"rgb_depth": {"depth"}},
        "joint": {
            "rgb_ir_depth": {"ir", "depth"},
            "rgb_ir": {"ir"},
            "rgb_depth": {"depth"},
        },
    }[config.stage]
    if config.stage == "joint" and config.model.fusion_type != "parallel_backbone":
        # Legacy/post-embed/deep fusion variants were defined for a complete
        # RGB+IR+Depth tuple and cannot perform auxiliary-modality ablations.
        modes = {"rgb_ir_depth": {"ir", "depth"}}
    report = {}
    evidence_rows = []
    metadata_by_id = {str(record["id"]): record for record in dataset.records}
    if not args.rgb_only:
        for name, modalities in modes.items():
            result = evaluate(
                model, loader, processor, device, config.train.max_new_tokens,
                modalities=modalities,
                return_rows=bool(args.rows_output),
            )
            report[name] = result["metrics"] if args.rows_output else result
            if args.rows_output:
                evidence_rows.extend(_evidence_rows(name, result, metadata_by_id))
    rgb_result = evaluate(
        model,
        loader,
        processor,
        device,
        config.train.max_new_tokens,
        rgb_only=True,
        return_rows=bool(args.rows_output),
    )
    report["rgb_baseline"] = rgb_result["metrics"] if args.rows_output else rgb_result
    if args.rows_output:
        evidence_rows.extend(
            _evidence_rows("rgb_baseline", rgb_result, metadata_by_id)
        )
    if not args.rgb_only:
        primary = next(iter(modes))
        report[f"{primary}_gain_over_rgb"] = {
            key: report[primary][key] - report["rgb_baseline"][key]
            for key in ("mean_iou", "acc_0.5", "acc_0.7")
        }
    report["diagnostic_metadata"] = {
        "config": args.config,
        "manifest": manifest,
        "checkpoints": loaded_checkpoints,
        "rgb_only": args.rgb_only,
        "samples": len(eval_dataset),
        "source_samples": len(dataset),
        "subset_seed": args.subset_seed if args.subset_size is not None else None,
    }
    report_text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report_text + "\n", encoding="utf-8")
    if args.rows_output:
        rows_output = Path(args.rows_output)
        rows_output.parent.mkdir(parents=True, exist_ok=True)
        rows_output.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False) + "\n" for row in evidence_rows
            ),
            encoding="utf-8",
        )
    print(report_text)


if __name__ == "__main__":
    main()
