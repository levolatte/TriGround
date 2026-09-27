"""Build a read-only City error atlas from complete C/G/U prediction files."""

from __future__ import annotations

import argparse
import html
import json
import os
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw

from tools.report_rematch_experiment import box_iou, load_run, parse_run_spec


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OLD = ROOT / "results/failure_analysis_20260926"
DEFAULT_C = ROOT / "results/next_stage_20260925/c_step1500/predictions.jsonl"
DEFAULT_M2 = ROOT / "results/next_stage_20260925/baseline_m2/predictions.jsonl"
COLORS = {"C": "#2463eb", "M2": "#ee9b00"}


def original_gt(path: Path) -> dict:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            return json.loads(archive.read("qwen_generation_val.json"))
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def transition(left: dict, right: dict, sample_id: str, target: list[float]) -> str | None:
    left_box = left["rows"][sample_id]["primary"]
    right_box = right["rows"][sample_id]["primary"]
    left_hit = left_box is not None and box_iou(left_box, target) >= 0.5
    right_hit = right_box is not None and box_iou(right_box, target) >= 0.5
    return "wrong_to_right" if right_hit and not left_hit else "right_to_wrong" if left_hit and not right_hit else None


def image_paths(row: dict, image_root: Path) -> dict[str, Path]:
    stem = Path(row["paths"]["remote_visible"]).name
    return {
        "visible": image_root / "visible" / stem,
        "infrared": image_root / "infrared" / stem,
        "depth": image_root / "depth" / stem,
        "depth_visual": image_root / "depth_rgb" / stem,
    }


def mark(image: Image.Image, box: list[float], color: str, width: int = 4) -> None:
    w, h = image.size
    xy = [round(box[0] * w), round(box[1] * h), round(box[2] * w), round(box[3] * h)]
    ImageDraw.Draw(image).rectangle(xy, outline=color, width=width)


def relative(path: Path, html_dir: Path) -> str:
    return Path(os.path.relpath(path, html_dir)).as_posix()


def build(gt_path: Path, c_path: Path, m2_path: Path, specs: list[str], old_dir: Path, image_root: Path, out: Path) -> dict:
    gt = original_gt(gt_path)
    if len(gt) != 412:
        raise ValueError(f"original City GT needs 412 samples, got {len(gt)}")
    inventory = {row["id"]: row for row in read_jsonl(old_dir / "inventory_all.jsonl")}
    reviewed = {row["id"]: row for row in read_jsonl(old_dir / "case_manifest.jsonl")}
    if set(inventory) != set(gt) or len(reviewed) != 16:
        raise ValueError("existing 412-case inventory or 16 reviewed cases disagree with original GT")
    if any(inventory[sample_id]["target"] != row["bbox"] for sample_id, row in gt.items()):
        raise ValueError("inventory GT differs from original qwen_generation_val.json")
    paths = {"C": c_path, "M2": m2_path}
    for spec in specs:
        name, path = parse_run_spec(spec)
        if name in paths:
            raise ValueError(f"duplicate run name: {name}")
        paths[name] = path
    runs = {name: load_run(name, path, set(gt), allow_partial=False) for name, path in paths.items()}
    pairs = [("M2", "C")]
    pairs += [("C", name) for name in paths if name not in ("C", "M2")]
    for name in paths:
        if name.startswith("G") and "U" + name[1:] in paths:
            pairs.append((name, "U" + name[1:]))
    changes: dict[str, dict[str, list[str]]] = {}
    selected = set(reviewed)
    for left, right in pairs:
        key = f"{left}_vs_{right}"
        changes[key] = {"wrong_to_right": [], "right_to_wrong": []}
        for sample_id, record in gt.items():
            kind = transition(runs[left], runs[right], sample_id, record["bbox"])
            if kind:
                changes[key][kind].append(sample_id)
                selected.add(sample_id)

    out.mkdir(parents=True, exist_ok=True)
    visual_dir = out / "visuals"
    visual_dir.mkdir(exist_ok=True)
    missing: list[dict] = []
    cards: list[str] = []
    case_rows: list[dict] = []
    for sample_id in gt:
        if sample_id not in selected:
            continue
        row = inventory[sample_id]
        target = gt[sample_id]["bbox"]
        paths_for_case = image_paths(row, image_root)
        for modality, path in paths_for_case.items():
            if not path.is_file():
                missing.append({"id": sample_id, "group": row["group"], "modality": modality,
                                "local_path": str(path), "remote_path": row["paths"]["remote_" + modality]})
        area = (target[2] - target[0]) * (target[3] - target[1])
        case_changes = [f"{pair}:{kind}" for pair, kinds in changes.items() for kind, ids in kinds.items() if sample_id in ids]
        review = reviewed.get(sample_id, {})
        case_rows.append({"id": sample_id, "group": row["group"], "target": target,
                          "target_area": area, "small_target_area_lt_0_001": area < 0.001,
                          "query": row["query"], "changes": case_changes,
                          "existing_human_review": bool(review),
                          "target_depth_valid_fraction": review.get("target_depth_valid_fraction"),
                          "depth_validity_status": "reviewed_16" if review else "not_checked",
                          "visible_available": paths_for_case["visible"].is_file()})
        figures = []
        visible = paths_for_case["visible"]
        if visible.is_file():
            base = Image.open(visible).convert("RGB")
            for name, run in runs.items():
                marked = base.copy()
                mark(marked, target, "#00ca73")
                prediction = run["rows"][sample_id]["primary"]
                if prediction is not None:
                    mark(marked, prediction, COLORS.get(name, "#ef3340"))
                image_name = f"{sample_id}_{name}.jpg"
                marked.save(visual_dir / image_name, quality=88)
                iou = box_iou(prediction, target) if prediction is not None else 0.0
                figures.append(f'<figure><img src="visuals/{html.escape(image_name)}"><figcaption>{html.escape(name)} · IoU {iou:.3f}</figcaption></figure>')
        for modality, title in (("infrared", "IR 原图"), ("depth_visual", "模型深度输入"), ("depth", "原始深度文件")):
            path = paths_for_case[modality]
            if path.is_file():
                figures.append(f'<figure><img src="{html.escape(relative(path, out))}"><figcaption>{title}（不叠 RGB 框）</figcaption></figure>')
        note = html.escape(review.get("visual_finding", "未做人工目视分类"))
        old_link = f'<a href="{html.escape(relative(old_dir / "cases.html", out))}#{html.escape(sample_id)}">既有16例详细图册</a>' if review else ""
        cards.append(f'<section id="{html.escape(sample_id)}"><h2>{html.escape(sample_id)}</h2>'
                     f'<p>{html.escape(row["query"])} · GT面积 {area:.4%} · {html.escape(", ".join(case_changes) or "既有16例固定纳入")}</p>'
                     f'<p>{note} {old_link}</p><div class="grid">{"".join(figures)}</div></section>')

    (out / "transitions.json").write_text(json.dumps(changes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for filename, records in (("missing_images.jsonl", missing), ("case_rows.jsonl", case_rows)):
        (out / filename).write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records), encoding="utf-8")
    title = "G/U 与 C 错变对及退化图册" if specs else "历史结果 dry run：G/U 预测尚未提供"
    page = '<!doctype html><html lang="zh"><meta charset="utf-8"><title>' + title + '</title>'
    page += '<style>body{font:16px/1.5 system-ui;max-width:1700px;margin:24px auto;background:#f6f7f8;color:#17202a}section{background:white;padding:20px;margin:20px 0;border:1px solid #ddd}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}figure{margin:0;border:1px solid #ddd}img{width:100%}figcaption{padding:7px}@media(max-width:900px){.grid{grid-template-columns:1fr}}</style>'
    page += f'<h1>{title}</h1><p>原始 412 条 GT 重新计算 IoU≥0.5。目标绿框、预测彩框仅叠在 RGB 展示图；IR/Depth 不叠 RGB 框。GT 只用于离线展示和评分，未用于模型输入裁剪。既有 16 例固定保留；其余人工类别与深度有效率未核。</p>'
    page += ''.join(cards) + '</html>'
    (out / "cases.html").write_text(page, encoding="utf-8")
    stats = {"gt_samples": len(gt), "runs": list(paths), "pairs": {key: {kind: len(ids) for kind, ids in kinds.items()} for key, kinds in changes.items()},
             "cases": len(case_rows), "fixed_reviewed_cases": len(reviewed), "missing_image_files": len(missing),
             "small_target_cases_area_lt_0_001": sum(row["small_target_area_lt_0_001"] for row in case_rows),
             "depth_validity_checked_cases": len(reviewed), "bootstrap": "see report_rematch_experiment.py full-run report (10000 image-group replicates)"}
    (out / "summary.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, default=Path("F:/Downloads/qwen_generation_train_val_manifests.zip"))
    parser.add_argument("--c", type=Path, default=DEFAULT_C)
    parser.add_argument("--m2", type=Path, default=DEFAULT_M2)
    parser.add_argument("--run", action="append", default=[], metavar="NAME=predictions.jsonl")
    parser.add_argument("--old-dir", type=Path, default=DEFAULT_OLD)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_OLD / "source_images")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.gt, args.c, args.m2, args.run, args.old_dir, args.image_root, args.output_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
