"""Create RGBDT500 Query-writing, blind-review, and human-review jobs.

The raw input comes from acquire_rgbdt500.py. This tool never invents a Query
or promotes a model agreement to a human-approved training label.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from tools.prepare_qwen3vl_native_sft import depth_mm_to_grayscale_rgb


FOCI = ("appearance", "instance_ordinal", "depth_relation", "infrared_evidence")
FOCUS_ALIASES = {
    "appearance": "appearance", "instance": "instance_ordinal", "instance_ordinal": "instance_ordinal",
    "depth": "depth_relation", "depth_relation": "depth_relation",
    "infrared": "infrared_evidence", "infrared_evidence": "infrared_evidence",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [row.get("query_id", row.get("id")) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate IDs in {path}")
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _images(group: dict[str, Any], root: Path, preview_dir: Path) -> dict[str, str]:
    paths = {modality: str((root / group[modality]).resolve()) for modality in ("rgb", "infrared", "depth")}
    for path in paths.values():
        if not Path(path).is_file():
            raise FileNotFoundError(f"selected RGBDT500 image is missing: {path}")
    if group.get("depth_visual"):
        preview = (root / group["depth_visual"]).resolve()
        if not preview.is_file():
            raise FileNotFoundError(f"selected RGBDT500 depth preview is missing: {preview}")
        paths["depth_preview"] = str(preview)
    elif group.get("depth_units") == "unverified" and group.get("depth_mode") == "I;16":
        preview_dir.mkdir(parents=True, exist_ok=True)
        preview = preview_dir / f"{group['id']}.png"
        if not preview.is_file():
            with Image.open(paths["depth"]) as image:
                raw = np.asarray(image)
            Image.fromarray(depth_mm_to_grayscale_rgb(raw)).save(preview)
        paths["depth_preview"] = str(preview.resolve())
    return paths


def _groups(path: Path) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(path)
    return {row["id"]: row for row in rows}


def prepare_generator(manifest: Path, data_root: Path, output_dir: Path, limit: int | None = None) -> dict[str, Any]:
    groups = list(_groups(manifest).values())
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        groups = groups[:limit]
    jobs = []
    for index, group in enumerate(groups):
        if group.get("query") is not None:
            raise ValueError(f"raw acquisition row unexpectedly already contains Query: {group['id']}")
        jobs.append({
            "image_group_id": group["id"],
            "images": _images(group, data_root, output_dir / "depth_previews"),
            "target_bbox_xyxy_normalized": group["bbox_xyxy_normalized"],
            "requested_focus": FOCI[index % len(FOCI)],
            "depth_units": group["depth_units"],
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    for batch_index in range(0, len(jobs), 10):
        (output_dir / f"generator_batch_{batch_index // 10:03d}.json").write_text(
            json.dumps({
                "instructions": (
                    "Inspect all three original views and the specified target box. Write 1-3 natural English "
                    "Queries that uniquely identify the target in this single frame. The requested focus is a probe: "
                    "if that evidence is absent, mark unsupported; do not invent relations, ordinal rank, thermal "
                    "properties or metric distance. Never mention the annotation box or markings. Depth units are "
                    "unverified unless independently established. Save each candidate as JSONL using the documented schema."
                ),
                "jobs": jobs[batch_index:batch_index + 10],
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    summary = {"groups": len(jobs), "batches": (len(jobs) + 9) // 10,
               "requested_focus": dict(Counter(row["requested_focus"] for row in jobs)),
               "generator_may_see_target_box": True, "queries_created": 0}
    (output_dir / "generator_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _generated(rows: list[dict[str, Any]], groups: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    valid = []
    for row in rows:
        if row["image_group_id"] not in groups:
            raise ValueError(f"unknown image group for {row['query_id']}")
        if row["generation_status"] in {"unsupported", "skipped"}:
            continue
        if row["generation_status"] not in {"proposed", "generated"}:
            raise ValueError(f"invalid generation_status for {row['query_id']}")
        if not isinstance(row.get("query"), str) or not row["query"].strip():
            raise ValueError(f"empty Query for {row['query_id']}")
        if row["focus"] not in FOCUS_ALIASES:
            raise ValueError(f"invalid focus for {row['query_id']}")
        if row.get("rewrite_count", 0) not in (0, 1):
            raise ValueError(f"at most one rewrite is allowed: {row['query_id']}")
        if not isinstance(row.get("evidence"), (dict, str)) or not row["evidence"]:
            raise ValueError(f"missing private evidence record: {row['query_id']}")
        valid.append({**row, "focus": FOCUS_ALIASES[row["focus"]]})
    return valid


def prepare_reviewer(manifest: Path, data_root: Path, generated: Path, output_dir: Path) -> dict[str, Any]:
    groups = _groups(manifest)
    candidates = _generated(_read_jsonl(generated), groups)
    jobs = [{
        "query_id": row["query_id"],
        "images": _images(groups[row["image_group_id"]], data_root, output_dir / "depth_previews"),
        "query": row["query"],
    } for row in candidates]
    output_dir.mkdir(parents=True, exist_ok=True)
    for batch_index in range(0, len(jobs), 10):
        (output_dir / f"blind_batch_{batch_index // 10:03d}.json").write_text(
            json.dumps({
                "instructions": (
                    "You are an independent reviewer. See only these unmarked original views and the Query. "
                    "Predict one normalized RGB xyxy box without access to target annotations or the generator's "
                    "reasoning. Mark ambiguous=true when more than one target fits; evidence_confirmed=true only "
                    "when the stated visual or modality cue is observable. Do not infer metric distances from "
                    "unverified depth units. Return one JSONL record per query_id."
                ),
                "jobs": jobs[batch_index:batch_index + 10],
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    summary = {"questions": len(jobs), "batches": (len(jobs) + 9) // 10,
            "blind_fields": ["query_id", "images", "query"],
            "target_box_disclosed": False, "generator_evidence_disclosed": False}
    (output_dir / "reviewer_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _box(value: Any) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"expected normalized xyxy box, got {value}")
    box = tuple(float(number) for number in value)
    if not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
        raise ValueError(f"invalid normalized xyxy box: {value}")
    return box


def _iou(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    x1, y1, x2, y2 = max(left[0], right[0]), max(left[1], right[1]), min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_left = (left[2] - left[0]) * (left[3] - left[1])
    area_right = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / (area_left + area_right - intersection)


def merge_reviews(manifest: Path, data_root: Path, generated: Path, reviews: Path, output_dir: Path) -> dict[str, Any]:
    groups = _groups(manifest)
    candidates = _generated(_read_jsonl(generated), groups)
    reviewed = {row["query_id"]: row for row in _read_jsonl(reviews)}
    unknown = set(reviewed) - {row["query_id"] for row in candidates}
    if unknown:
        raise ValueError(f"reviewed Query IDs not in generated file: {sorted(unknown)[:5]}")
    joined, provisional = [], []
    for row in candidates:
        group = groups[row["image_group_id"]]
        images = _images(group, data_root, output_dir / "depth_previews")
        review = reviewed.get(row["query_id"])
        prediction = review.get("predicted_bbox_xyxy_normalized") if review else None
        iou = _iou(_box(group["bbox_xyxy_normalized"]), _box(prediction)) if prediction is not None else None
        ambiguous = review.get("ambiguous") if review else None
        evidence_confirmed = review.get("evidence_confirmed") if review else None
        if review and (not isinstance(ambiguous, bool) or not isinstance(evidence_confirmed, bool)):
            raise ValueError(f"review flags must be boolean for {row['query_id']}")
        passed = iou is not None and iou >= 0.5 and ambiguous is False and evidence_confirmed is True
        base = {
            "query_id": row["query_id"], "image_group_id": row["image_group_id"],
            "split": group["split"], "query": row["query"], "focus": row["focus"],
            "rgb": images["rgb"], "infrared": images["infrared"], "depth": images["depth"],
            "depth_preview": images.get("depth_preview"),
            "bbox": group["bbox_xyxy_normalized"],
            "sequence_id": group["sequence"], "scene_id": group["sequence"],
            "depth_units": group["depth_units"],
            "depth_policy": (
                "sensor_linear_20000" if group["depth_units"] == "unverified" and group.get("depth_mode") == "I;16"
                else "visual" if group["depth_units"] == "unverified" else group["depth_units"]
            ),
            "rewrite_count": row.get("rewrite_count", 0),
            "review_predicted_bbox": prediction, "blind_iou": iou,
            "ambiguous": ambiguous, "evidence_confirmed": evidence_confirmed,
            "generation_evidence": row["evidence"],
            "review_status": "provisional" if passed else "needs_review",
        }
        joined.append(base)
        if passed and group["split"] == "pilot":
            provisional.append({
                "id": row["query_id"], "source": "rgbdt500", "split": "train",
                "rgb": base["rgb"], "infrared": base["infrared"], "depth": base["depth"],
                "bbox": base["bbox"], "query": base["query"],
                "sequence_id": base["sequence_id"], "scene_id": base["scene_id"],
                "depth_policy": base["depth_policy"], "review_status": "provisional",
                "annotation_focus": base["focus"],
                **({"depth_visual": base["depth_preview"]} if base["depth_preview"] else {}),
            })
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "joined_reviews.jsonl", joined)
    _write_jsonl(output_dir / "provisional_train_candidates.jsonl", provisional)
    summary = {
        "generated_questions": len(candidates), "reviewed_questions": len(reviewed),
        "missing_reviews": len(candidates) - len(reviewed), "provisional_train_questions": len(provisional),
        "provisional_by_focus": dict(Counter(row["focus"] for row in joined if row["review_status"] == "provisional")),
        "human_approved_questions": 0,
        "automatic_promotion_to_training": False,
    }
    (output_dir / "review_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _html_row(row: dict[str, Any]) -> str:
    box = _box(row["bbox"])
    left, top = box[0] * 100, box[1] * 100
    width, height = (box[2] - box[0]) * 100, (box[3] - box[1]) * 100
    def image(path: str, marked: bool = False) -> str:
        marker = (f'<span class="mark" style="left:{left:.3f}%;top:{top:.3f}%;width:{width:.3f}%;height:{height:.3f}%"></span>' if marked else "")
        return f'<div class="view"><img src="{html.escape(Path(path).as_uri())}">{marker}</div>'
    return (
        "<article><h3>" + html.escape(row["query_id"]) + "</h3>"
        + "<p>" + html.escape(row["query"]) + "</p>"
        + "<div class=" + '"views">' + image(row["rgb"], True)
        + image(row["infrared"]) + image(row.get("depth_preview") or row["depth"]) + "</div>"
        + f'<p>来源：{html.escape(row["split"])}；类型：{html.escape(row["focus"])}；独立复核 IoU：{row["blind_iou"]}</p></article>'
    )


def prepare_human_pack(pilot_joined: Path, external_joined: Path, output_dir: Path, seed: int = 2026) -> dict[str, Any]:
    pilot = [row for row in _read_jsonl(pilot_joined) if row["review_status"] == "provisional"]
    external = _read_jsonl(external_joined)
    by_focus = defaultdict(list)
    for row in pilot:
        by_focus[row["focus"]].append(row)
    rng = random.Random(seed)
    selected = []
    for focus in FOCI:
        rows = by_focus[focus]
        if len(rows) < 25:
            raise ValueError(f"pilot needs 25 provisional {focus} Queries for human pack; got {len(rows)}")
        selected.extend(rng.sample(rows, 25))
    external_groups = defaultdict(list)
    for row in external:
        external_groups[row["image_group_id"]].append(row)
    eligible_groups = sorted(group for group, rows in external_groups.items() if len(rows) >= 2)
    if len(eligible_groups) < 50:
        raise ValueError("external human pack needs two real Queries in each of 50 image groups")
    for group in eligible_groups[:50]:
        selected.extend(sorted(external_groups[group], key=lambda row: row["query_id"])[:2])
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["query_id", "split", "image_group_id", "focus", "query", "rgb", "infrared", "depth", "depth_preview", "bbox", "blind_iou", "review_status", "human_decision", "human_note"]
    with (output_dir / "human_review_200.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in selected:
            writer.writerow({field: json.dumps(row[field]) if field == "bbox" else row.get(field, "") for field in fields})
    style = "body{font:15px sans-serif;max-width:1350px;margin:auto}article{border-bottom:1px solid #aaa;padding:14px}.views{display:flex;gap:8px}.view{position:relative;width:32%}.view img{width:100%;height:auto;display:block}.mark{position:absolute;box-sizing:border-box;border:3px solid red}"
    (output_dir / "human_review_200.html").write_text(
        "<!doctype html><meta charset='utf-8'><style>" + style + "</style>"
        + "<h1>RGBDT500 200条人工复核</h1><p>在 CSV 中填写 human_decision 与 human_note。红框为原始跟踪目标；核对三图、Query是否唯一以及模态证据。</p>"
        + "\n".join(_html_row(row) for row in selected), encoding="utf-8"
    )
    return {"total": len(selected), "pilot": 100, "external": 100,
            "csv": str(output_dir / "human_review_200.csv"),
            "html": str(output_dir / "human_review_200.html")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare-generator", "prepare-reviewer", "merge-reviews"):
        command = sub.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--data-root", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
        if name != "prepare-generator":
            command.add_argument("--generated", type=Path, required=True)
        if name == "prepare-generator":
            command.add_argument("--limit", type=int)
        if name == "merge-reviews":
            command.add_argument("--reviews", type=Path, required=True)
    human = sub.add_parser("prepare-human-pack")
    human.add_argument("--pilot-joined", type=Path, required=True)
    human.add_argument("--external-joined", type=Path, required=True)
    human.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare-generator":
        result = prepare_generator(args.manifest, args.data_root, args.output_dir, args.limit)
    elif args.command == "prepare-reviewer":
        result = prepare_reviewer(args.manifest, args.data_root, args.generated, args.output_dir)
    elif args.command == "merge-reviews":
        result = merge_reviews(args.manifest, args.data_root, args.generated, args.reviews, args.output_dir)
    else:
        result = prepare_human_pack(args.pilot_joined, args.external_joined, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
