from __future__ import annotations

import argparse
import json
import random
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


def scene_group(record: dict) -> tuple[str, str]:
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


def query_mode_definitions() -> list[dict]:
    common = {
        "modalities": {"ir", "depth"},
        "ir_scale": 1.0,
        "depth_scale": 1.0,
    }
    return [
        {"name": "query_correct", **common, "query_scale": 1.0},
        {"name": "query_zero", **common, "query_scale": 0.0},
        {
            "name": "query_shuffled",
            **common,
            "query_scale": 1.0,
            "query_mismatch": True,
        },
    ]


def mismatched_queries(records: list[dict], seed: int) -> dict[str, str]:
    """Assign a different-scene, different-text fusion query when possible."""
    if len(records) < 2:
        raise ValueError("query mismatch requires at least two validation samples")
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    output = {}
    for position, index in enumerate(order):
        target = records[index]
        target_scene = scene_group(target)[1]
        candidates = [
            records[order[(position + offset) % len(order)]]
            for offset in range(1, len(order))
        ]
        donor = next(
            (
                candidate
                for candidate in candidates
                if candidate["query"] != target["query"]
                and scene_group(candidate)[1] != target_scene
            ),
            None,
        )
        if donor is None:
            donor = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate["query"] != target["query"]
                ),
                candidates[0],
            )
        output[str(target["id"])] = str(donor["query"])
    return output


def subset_indices(length: int, limit: int | None, seed: int) -> list[int]:
    if limit is None or limit >= length:
        return list(range(length))
    indices = list(range(length))
    random.Random(seed).shuffle(indices)
    return sorted(indices[:limit])


@torch.no_grad()
def evaluate_rows(
    model,
    loader,
    processor,
    device,
    max_new_tokens,
    mode: dict,
    query_overrides: dict[str, str] | None = None,
) -> dict:
    model.eval()
    rows = []
    cap_hits = 0
    eos = processor.tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    for batch in loader:
        if query_overrides is not None:
            fusion_queries = [query_overrides[str(value)] for value in batch["sample_id"]]
            encoded_queries = processor.tokenizer(
                fusion_queries,
                padding=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            batch["query_input_ids"] = encoded_queries["input_ids"]
            batch["query_attention_mask"] = encoded_queries["attention_mask"]
        batch = _move(batch, device)
        generated = model.generate(
            **_generation_inputs(
                batch,
                rgb_only=bool(mode.get("rgb_only")),
                modalities=set(mode["modalities"]),
                ir_fusion_scale=float(mode["ir_scale"]),
                depth_fusion_scale=float(mode["depth_scale"]),
                query_fusion_scale=float(mode.get("query_scale", 1.0)),
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
        incorrect_to_correct = sum(
            base[row["id"]] < 0.5 and row["iou"] >= 0.5
            for row in result["rows"]
        )
        correct_to_incorrect = sum(
            base[row["id"]] >= 0.5 and row["iou"] < 0.5
            for row in result["rows"]
        )
        mean_iou_delta = sum(deltas) / max(len(deltas), 1)
        report[name] = {
            "mean_iou_delta": mean_iou_delta,
            "acc_0.5_delta": (
                result["metrics"]["acc_0.5"]
                - modes[baseline]["metrics"]["acc_0.5"]
            ),
            "incorrect_to_correct": incorrect_to_correct,
            "correct_to_incorrect": correct_to_incorrect,
            "improved_samples": sum(delta > 1e-6 for delta in deltas),
            "degraded_samples": sum(delta < -1e-6 for delta in deltas),
            "unchanged_samples": sum(abs(delta) <= 1e-6 for delta in deltas),
        }
        if baseline == "rgb":
            report[name]["mean_iou_delta_vs_rgb"] = mean_iou_delta
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Causal modality intervention diagnostics")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int)
    parser.add_argument(
        "--family",
        choices=("modality", "query", "all"),
        default="modality",
        help="Run modality scaling/mismatch, query interventions, or both.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config = load_config(args.config)
    if config.stage != "joint":
        raise ValueError("diagnostic config must use joint stage")
    if config.model.fusion_type != "parallel_backbone":
        raise ValueError("query interventions require parallel_backbone fusion")
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
    metadata_by_id = {str(record["id"]): record for record in base.records}
    indices = subset_indices(len(base), args.samples, args.seed)
    output = {
        "metadata": {
            "config": args.config,
            "checkpoints": args.checkpoint,
            "manifest": config.data.val_manifest,
            "samples": len(indices),
            "seed": args.seed,
            "family": args.family,
        },
        "modes": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    modes = []
    if args.family in {"modality", "all"}:
        modes.extend(mode_definitions())
    if args.family in {"query", "all"}:
        modes.extend(query_mode_definitions())
    query_donors = (
        mismatched_queries(base.records, args.seed + 31)
        if args.family in {"query", "all"}
        else None
    )
    for mode in modes:
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
            model,
            loader,
            processor,
            device,
            config.train.max_new_tokens,
            mode,
            query_donors if mode.get("query_mismatch") else None,
        )
        for row in result["rows"]:
            record = metadata_by_id[str(row["id"])]
            group_source, group = scene_group(record)
            row["scene_group_source"] = group_source
            row["scene_group"] = group
            row["main_query"] = str(record["query"])
            if mode.get("query_mismatch"):
                row["fusion_query"] = query_donors[str(row["id"])]
            elif mode.get("query_scale", 1.0) == 0:
                row["fusion_query"] = None
            else:
                row["fusion_query"] = str(record["query"])
        output["modes"][mode["name"]] = {
            "definition": {
                "rgb_only": bool(mode.get("rgb_only")),
                "modalities": sorted(mode["modalities"]),
                "mismatch": sorted(mismatch),
                "ir_scale": mode["ir_scale"],
                "depth_scale": mode["depth_scale"],
                "query_scale": mode.get("query_scale", 1.0),
                "query_mismatch": bool(mode.get("query_mismatch")),
            },
            **result,
        }
        if "rgb" in output["modes"]:
            output["paired_vs_rgb"] = paired_summary(output["modes"])
        if "query_correct" in output["modes"]:
            output["paired_vs_query_correct"] = paired_summary(
                output["modes"], baseline="query_correct"
            )
        args.output.write_text(
            json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"mode": mode["name"], **result["metrics"]}), flush=True)


if __name__ == "__main__":
    main()
