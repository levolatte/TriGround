from __future__ import annotations

import argparse
import json
import random
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoProcessor

from mm_grounding.boxes import box_iou_aligned
from mm_grounding.checkpoint import load_model_checkpoint
from mm_grounding.config import load_config
from mm_grounding.data import GroundingDataset, NativeGroundingCollator
from mm_grounding.engine import _generation_inputs, _move, parse_bbox
from mm_grounding.model import build_grounder


class MismatchedModalityDataset(Dataset):
    """Keep RGB/query/target fixed while replacing selected auxiliary images."""

    def __init__(self, base: Dataset, modalities: set[str], seed: int = 2026):
        self.base = base
        self.modalities = modalities
        indices = list(range(len(base)))
        random.Random(seed).shuffle(indices)
        rotated = indices[1:] + indices[:1]
        self.donor_by_index = dict(zip(indices, rotated))

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict:
        target = self.base[index]
        donor = self.base[self.donor_by_index[index]]
        for modality in self.modalities:
            target[modality] = donor[modality]
        return target


@contextmanager
def temporary_modality_scales(model, ir_scale: float, depth_scale: float):
    if model.fusion_type != "parallel_backbone":
        raise ValueError("modality interventions require parallel_backbone fusion")
    saved = []
    for stage in model.fusion.stage_fusions.values():
        for branch, scale in ((stage.ir, ir_scale), (stage.depth, depth_scale)):
            saved.append((branch.residual_scale, branch.residual_scale.detach().clone()))
            with torch.no_grad():
                branch.residual_scale.mul_(scale)
    try:
        yield
    finally:
        with torch.no_grad():
            for parameter, value in saved:
                parameter.copy_(value)


def mode_definitions() -> list[dict]:
    return [
        {"name": "rgb", "rgb_only": True, "modalities": set(), "ir_scale": 0.0, "depth_scale": 0.0},
        *[
            {"name": f"ir_scale_{scale:g}", "modalities": {"ir"}, "ir_scale": scale, "depth_scale": 0.0}
            for scale in (0.0, 0.25, 0.5, 1.0)
        ],
        *[
            {"name": f"depth_scale_{scale:g}", "modalities": {"depth"}, "ir_scale": 0.0, "depth_scale": scale}
            for scale in (0.0, 0.25, 0.5, 1.0)
        ],
        *[
            {"name": f"triple_ir_{ir:g}_depth_{depth:g}", "modalities": {"ir", "depth"}, "ir_scale": ir, "depth_scale": depth}
            for ir, depth in (
                (0.0, 0.0), (0.0, 1.0), (0.25, 1.0), (0.5, 1.0),
                (1.0, 0.0), (1.0, 0.25), (1.0, 0.5), (1.0, 1.0),
            )
        ],
        {"name": "ir_mismatched", "modalities": {"ir"}, "mismatch": {"ir"}, "ir_scale": 1.0, "depth_scale": 0.0},
        {"name": "depth_mismatched", "modalities": {"depth"}, "mismatch": {"depth"}, "ir_scale": 0.0, "depth_scale": 1.0},
        {"name": "triple_ir_mismatched", "modalities": {"ir", "depth"}, "mismatch": {"ir"}, "ir_scale": 1.0, "depth_scale": 1.0},
        {"name": "triple_depth_mismatched", "modalities": {"ir", "depth"}, "mismatch": {"depth"}, "ir_scale": 1.0, "depth_scale": 1.0},
        {"name": "triple_both_mismatched", "modalities": {"ir", "depth"}, "mismatch": {"ir", "depth"}, "ir_scale": 1.0, "depth_scale": 1.0},
    ]


def subset_indices(length: int, limit: int | None, seed: int) -> list[int]:
    if limit is None or limit >= length:
        return list(range(length))
    indices = list(range(length))
    random.Random(seed).shuffle(indices)
    return sorted(indices[:limit])


@torch.no_grad()
def evaluate_rows(model, loader, processor, device, max_new_tokens, mode: dict) -> dict:
    model.eval()
    rows = []
    cap_hits = 0
    eos = processor.tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    with temporary_modality_scales(model, mode["ir_scale"], mode["depth_scale"]):
        for batch in loader:
            batch = _move(batch, device)
            generated = model.generate(
                **_generation_inputs(
                    batch,
                    rgb_only=bool(mode.get("rgb_only")),
                    modalities=set(mode["modalities"]),
                ),
                max_new_tokens=max_new_tokens,
            )
            prompt_length = batch["generation_input_ids"].shape[1]
            new_tokens = generated[:, prompt_length:]
            answers = processor.batch_decode(new_tokens, skip_special_tokens=True)
            for index, (sample_id, answer) in enumerate(zip(batch["sample_id"], answers)):
                prediction = parse_bbox(answer)
                predicted = torch.tensor(
                    prediction or [0.0, 0.0, 0.0, 0.0],
                    device=device,
                    dtype=batch["bbox"].dtype,
                ).unsqueeze(0)
                target = batch["bbox"][index : index + 1]
                iou = float(box_iou_aligned(predicted, target).item())
                capped = (
                    new_tokens.shape[1] >= max_new_tokens
                    and not any(int(token) in eos_ids for token in new_tokens[index])
                )
                cap_hits += int(capped)
                rows.append({
                    "id": sample_id,
                    "iou": iou,
                    "gt": [float(value) for value in target[0].tolist()],
                    "prediction": prediction,
                    "parsed": prediction is not None,
                    "generation_cap_hit": capped,
                })
    count = max(len(rows), 1)
    parsed_rows = [row for row in rows if row["prediction"] is not None]
    l1 = sum(
        sum(abs(p - g) for p, g in zip(row["prediction"], row["gt"])) / 4
        if row["prediction"] is not None else sum(row["gt"]) / 4
        for row in rows
    ) / count
    return {
        "metrics": {
            "mean_iou": sum(row["iou"] for row in rows) / count,
            "acc_0.5": sum(row["iou"] >= 0.5 for row in rows) / count,
            "acc_0.7": sum(row["iou"] >= 0.7 for row in rows) / count,
            "l1_coordinate_error": l1,
            "parse_rate": len(parsed_rows) / count,
            "generation_cap_hit_rate": cap_hits / count,
        },
        "rows": rows,
    }


def paired_summary(modes: dict[str, dict], baseline: str = "rgb") -> dict:
    base = {row["id"]: row["iou"] for row in modes[baseline]["rows"]}
    report = {}
    for name, result in modes.items():
        if name == baseline:
            continue
        deltas = [row["iou"] - base[row["id"]] for row in result["rows"]]
        report[name] = {
            "mean_iou_delta_vs_rgb": sum(deltas) / max(len(deltas), 1),
            "improved_samples": sum(delta > 1e-6 for delta in deltas),
            "degraded_samples": sum(delta < -1e-6 for delta in deltas),
            "unchanged_samples": sum(abs(delta) <= 1e-6 for delta in deltas),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Causal modality intervention diagnostics")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config = load_config(args.config)
    if config.stage != "joint":
        raise ValueError("diagnostic config must use joint stage")
    device = torch.device(args.device)
    processor = AutoProcessor.from_pretrained(
        config.model.backbone,
        min_pixels=config.data.min_pixels,
        max_pixels=config.data.max_pixels,
    )
    model = build_grounder(config.model, processor).to(device)
    for checkpoint in args.checkpoint:
        load_model_checkpoint(checkpoint, model)
    base = GroundingDataset(
        config.data.val_manifest,
        stage="joint",
        depth_scale=config.data.depth_scale,
        depth_clip=config.data.depth_clip,
    )
    indices = subset_indices(len(base), args.samples, args.seed)
    output = {
        "metadata": {
            "config": args.config,
            "checkpoints": args.checkpoint,
            "manifest": config.data.val_manifest,
            "samples": len(indices),
            "seed": args.seed,
        },
        "modes": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for mode in mode_definitions():
        dataset: Dataset = base
        mismatch = set(mode.get("mismatch", set()))
        if mismatch:
            dataset = MismatchedModalityDataset(base, mismatch, args.seed + 17)
        dataset = Subset(dataset, indices)
        loader = DataLoader(
            dataset,
            batch_size=config.train.val_batch_size,
            shuffle=False,
            num_workers=config.data.workers,
            collate_fn=NativeGroundingCollator(processor, "joint"),
        )
        result = evaluate_rows(
            model, loader, processor, device, config.train.max_new_tokens, mode
        )
        output["modes"][mode["name"]] = {
            "definition": {
                "rgb_only": bool(mode.get("rgb_only")),
                "modalities": sorted(mode["modalities"]),
                "mismatch": sorted(mismatch),
                "ir_scale": mode["ir_scale"],
                "depth_scale": mode["depth_scale"],
            },
            **result,
        }
        output["paired_vs_rgb"] = paired_summary(output["modes"])
        args.output.write_text(
            json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"mode": mode["name"], **result["metrics"]}), flush=True)


if __name__ == "__main__":
    main()
