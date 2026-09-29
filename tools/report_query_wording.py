"""Compare long and short wording on a fixed set of visual grounding tasks.

The report scores every task against its original candidate bbox. Missing and
invalid predictions count as failures; missing rows make the report partial,
so a partial run cannot support a wording preference claim.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from tools.prepare_triground_abv_data import _iou
from tools.report_triground_abv import _valid_prediction


ARMS = ("long", "short")
ARM_LABELS = {"long": "完整版", "short": "简洁版"}
COLORS = {"gt": (40, 190, 75), "long": (40, 115, 245), "short": (245, 145, 35)}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(row)
    return rows


def _load_pairs(path: Path) -> list[dict[str, Any]]:
    pairs = json.loads(path.read_text(encoding="utf-8-sig"))
    required = {"task_id", "long_query", "long_query_zh", "short_query", "short_query_zh"}
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("--pairs must contain a non-empty JSON list")
    seen: set[str] = set()
    for index, pair in enumerate(pairs):
        if not isinstance(pair, dict) or not required <= pair.keys():
            raise ValueError(f"--pairs item {index} must contain {sorted(required)}")
        task_id = str(pair["task_id"])
        if not task_id or task_id in seen:
            raise ValueError(f"--pairs has an empty or duplicate task_id: {task_id!r}")
        seen.add(task_id)
        pair["task_id"] = task_id
        for field in required - {"task_id"}:
            if not isinstance(pair[field], str):
                raise ValueError(f"--pairs {task_id}: {field} must be a string")
    return pairs


def _index_candidates(path: Path, pairs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    candidate_rows = _read_jsonl(path)
    all_candidates: dict[str, dict[str, Any]] = {}
    for line_number, candidate in enumerate(candidate_rows, 1):
        if "task_id" not in candidate:
            raise ValueError(f"{path}: row {line_number} has no task_id")
        task_id = str(candidate["task_id"])
        if task_id in all_candidates:
            raise ValueError(f"{path}: duplicate candidate task_id {task_id!r}")
        all_candidates[task_id] = candidate

    selected = {}
    for pair in pairs:
        task_id = pair["task_id"]
        if task_id not in all_candidates:
            raise ValueError(f"candidate task_id not found: {task_id}")
        candidate = all_candidates[task_id]
        if not _valid_prediction(candidate.get("bbox")):
            raise ValueError(f"candidate {task_id}: bbox must be a valid normalized xyxy box")
        images = candidate.get("images")
        if not isinstance(images, dict) or not images.get("rgb"):
            raise ValueError(f"candidate {task_id}: images.rgb is required")
        selected[task_id] = candidate
    return selected


def _prediction_rows(path: Path, pairs: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    expected = {f"{pair['task_id']}::{arm}" for pair in pairs for arm in ARMS}
    rows: dict[str, dict[str, Any]] = {}
    invalid: list[str] = []
    for line_number, row in enumerate(_read_jsonl(path), 1):
        if "id" not in row:
            raise ValueError(f"{path}: row {line_number} has no id")
        sample_id = str(row["id"])
        if sample_id not in expected:
            raise ValueError(f"{path}: unknown prediction id {sample_id!r}")
        if sample_id in rows:
            raise ValueError(f"{path}: duplicate prediction id {sample_id!r}")
        prediction = row.get("prediction")
        if _valid_prediction(prediction):
            row["_score_box"] = [float(value) for value in prediction]
        else:
            row["_score_box"] = None
            row["_failure"] = "parse_failure" if prediction is None else "invalid_box"
            invalid.append(sample_id)
        rows[sample_id] = row
    missing = sorted(expected - rows.keys())
    return rows, missing


def _iou_or_zero(prediction: list[float] | None, target: list[float]) -> float:
    return _iou(prediction, target) if prediction is not None else 0.0


def _arm_metrics(entries: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    count = len(entries)
    ious = [entry[f"{arm}_iou"] for entry in entries]
    hits_05 = sum(value >= 0.5 for value in ious)
    hits_07 = sum(value >= 0.7 for value in ious)
    return {
        "samples": count,
        "denominator": count,
        "hits_0.5": hits_05,
        "acc_0.5": hits_05 / count,
        "mean_iou": sum(ious) / count,
        "hits_0.7": hits_07,
        "acc_0.7": hits_07 / count,
        "valid_predictions": sum(entry[f"{arm}_box"] is not None for entry in entries),
    }


def _group_metrics(entries: list[dict[str, Any]], field: str) -> dict[str, Any]:
    labels = sorted({str(entry[field]) for entry in entries})
    return {
        label: {
            "samples": sum(str(entry[field]) == label for entry in entries),
            "long": _arm_metrics([entry for entry in entries if str(entry[field]) == label], "long"),
            "short": _arm_metrics([entry for entry in entries if str(entry[field]) == label], "short"),
        }
        for label in labels
    }


def build_report(
    pairs: list[dict[str, Any]],
    candidates: dict[str, dict[str, Any]],
    prediction_rows: dict[str, dict[str, Any]],
    missing_ids: list[str],
    invalid_ids: list[str],
) -> dict[str, Any]:
    entries = []
    for pair in pairs:
        task_id = pair["task_id"]
        candidate = candidates[task_id]
        entry: dict[str, Any] = {
            **pair,
            "scene_id": str(candidate.get("scene_id", "unknown")),
            "relation_type": str(candidate.get("relation_type", "unknown")),
            "rgb_path": str(candidate["images"]["rgb"]),
            "gt_bbox": [float(value) for value in candidate["bbox"]],
        }
        for arm in ARMS:
            sample_id = f"{task_id}::{arm}"
            row = prediction_rows.get(sample_id)
            box = row["_score_box"] if row is not None else None
            entry[f"{arm}_box"] = box
            entry[f"{arm}_iou"] = _iou_or_zero(box, entry["gt_bbox"])
            entry[f"{arm}_failure"] = row.get("_failure") if row else "missing_prediction"
            entry[f"{arm}_raw_output"] = _raw_output(row) if row else None
        left, right = entry["long_iou"], entry["short_iou"]
        entry["short_hit_0.5_rescue"] = left < 0.5 <= right
        entry["short_hit_0.5_harm"] = right < 0.5 <= left
        entry["iou_winner"] = "short" if right > left else "long" if left > right else "tie"
        entry["prediction_box_iou"] = (
            _iou(entry["long_box"], entry["short_box"])
            if entry["long_box"] is not None and entry["short_box"] is not None else None
        )
        entries.append(entry)

    long_metrics = _arm_metrics(entries, "long")
    short_metrics = _arm_metrics(entries, "short")
    rescued_ids = [entry["task_id"] for entry in entries if entry["short_hit_0.5_rescue"]]
    harmed_ids = [entry["task_id"] for entry in entries if entry["short_hit_0.5_harm"]]
    iou_wins = {arm: [entry["task_id"] for entry in entries if entry["iou_winner"] == arm]
                for arm in ARMS}
    iou_wins["tie"] = [entry["task_id"] for entry in entries if entry["iou_winner"] == "tie"]
    stable_ids = [entry["task_id"] for entry in entries
                  if entry["prediction_box_iou"] is not None and entry["prediction_box_iou"] >= 0.5]
    partial = bool(missing_ids)
    return {
        "status": "partial" if partial else "complete",
        "partial": partial,
        "task_count": len(entries),
        "expected_prediction_rows": len(entries) * 2,
        "observed_prediction_rows": len(prediction_rows),
        "missing_prediction_ids": missing_ids,
        "invalid_prediction_ids": invalid_ids,
        "denominator_policy": "all paired tasks; missing or invalid boxes score IoU 0",
        "overall": {"long": long_metrics, "short": short_metrics},
        "paired": {
            "short_rescued_ids_at_0.5": rescued_ids,
            "short_harmed_ids_at_0.5": harmed_ids,
            "short_net_hits_at_0.5": len(rescued_ids) - len(harmed_ids),
            "per_task_iou_wins": iou_wins,
            "box_overlap_stability": {
                "threshold": 0.5,
                "samples_with_both_valid_boxes": sum(entry["prediction_box_iou"] is not None for entry in entries),
                "overlap_ids": stable_ids,
                "interpretation": "prediction boxes overlap at IoU >= 0.5; this does not establish object-choice agreement",
            },
            "interpretation": ("partial run: paired outcomes are descriptive and do not support a wording preference"
                               if partial else "paired descriptive comparison on this fixed task set"),
        },
        "subsets": {
            "relation_type": _group_metrics(entries, "relation_type"),
            "scene_id": _group_metrics(entries, "scene_id"),
        },
        "scope_limitations": [
            "本次比较仅适用于给定题目与预测，不外推为模型整体能力。",
            "当前14题来自5个场景，结果不能证明Depth互补收益。",
            "两预测框IoU不低于0.5只说明几何重叠，不等于已确认对象选择一致。",
        ],
        "entries": entries,
    }


def _raw_output(row: dict[str, Any] | None) -> Any:
    if row is None:
        return None
    for field in ("raw_text", "raw_output", "output", "response"):
        if field in row:
            return row[field]
    return {key: value for key, value in row.items() if not key.startswith("_")}


def _fmt_box(value: Any) -> str:
    return "—" if value is None else json.dumps(value, ensure_ascii=False)


def _write_overlay(entry: dict[str, Any], destination: Path) -> None:
    source = Path(entry["rgb_path"])
    if not source.is_file():
        raise FileNotFoundError(f"{entry['task_id']}: RGB image not found: {source}")
    with Image.open(source) as image:
        canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    boxes = [("gt_bbox", "GT", COLORS["gt"]),
             ("long_box", "long", COLORS["long"]),
             ("short_box", "short", COLORS["short"])]
    def dashed_line(start: tuple[int, int], end: tuple[int, int],
                    color: tuple[int, int, int], phase: int) -> None:
        x1, y1 = start
        x2, y2 = end
        length = abs(x2 - x1) + abs(y2 - y1)
        direction = (1 if x2 >= x1 else -1, 1 if y2 >= y1 else -1)
        position = -phase
        while position < length:
            begin = max(0, position)
            finish = min(length, position + 7)
            if begin < finish:
                first = (x1 + direction[0] * begin, y1 + direction[1] * begin)
                last = (x1 + direction[0] * finish, y1 + direction[1] * finish)
                draw.line((first, last), fill=color, width=3)
            position += 12

    for index, (field, label, color) in enumerate(boxes):
        box = entry[field]
        if box is None:
            continue
        x1, y1, x2, y2 = box
        rect = (round(x1 * width), round(y1 * height),
                round(x2 * width), round(y2 * height))
        left, top, right, bottom = rect
        phase = index * 3
        for start, end in (((left, top), (right, top)),
                           ((right, top), (right, bottom)),
                           ((right, bottom), (left, bottom)),
                           ((left, bottom), (left, top))):
            dashed_line(start, end, color, phase)
        draw.text((left + 2, top + 2 + index * 12), label, fill=color,
                  stroke_width=2, stroke_fill=(0, 0, 0))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG")


def write_outputs(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps({key: value for key, value in report.items() if key != "entries"},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    csv_fields = [
        "task_id", "scene_id", "relation_type", "gt_bbox",
        "long_query", "long_query_zh", "long_bbox", "long_iou", "long_hit_0.5",
        "long_raw_output", "long_failure",
        "short_query", "short_query_zh", "short_bbox", "short_iou", "short_hit_0.5",
        "short_raw_output", "short_failure", "short_rescued", "short_harmed",
        "prediction_box_iou", "box_overlap_ge_0.5",
    ]
    with (output_dir / "paired.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        for entry in report["entries"]:
            row = {
                "task_id": entry["task_id"], "scene_id": entry["scene_id"],
                "relation_type": entry["relation_type"],
                "gt_bbox": _fmt_box(entry["gt_bbox"]),
                "long_query": entry["long_query"], "long_query_zh": entry["long_query_zh"],
                "long_bbox": _fmt_box(entry["long_box"]), "long_iou": entry["long_iou"],
                "long_hit_0.5": entry["long_iou"] >= 0.5,
                "long_raw_output": json.dumps(entry["long_raw_output"], ensure_ascii=False),
                "long_failure": entry["long_failure"],
                "short_query": entry["short_query"], "short_query_zh": entry["short_query_zh"],
                "short_bbox": _fmt_box(entry["short_box"]), "short_iou": entry["short_iou"],
                "short_hit_0.5": entry["short_iou"] >= 0.5,
                "short_raw_output": json.dumps(entry["short_raw_output"], ensure_ascii=False),
                "short_failure": entry["short_failure"],
                "short_rescued": entry["short_hit_0.5_rescue"],
                "short_harmed": entry["short_hit_0.5_harm"],
                "prediction_box_iou": entry["prediction_box_iou"],
                "box_overlap_ge_0.5": (entry["prediction_box_iou"] is not None
                                         and entry["prediction_box_iou"] >= 0.5),
            }
            writer.writerow(row)

    lines = [
        "# Query 长短表达配对比较", "",
        f"状态：**{report['status']}**；题目 {report['task_count']} 道，预测记录 "
        f"{report['observed_prediction_rows']}/{report['expected_prediction_rows']} 条；所有分数均以全部题目的原始 bbox 为分母。", "",
        "缺失或非法预测按失败（IoU=0）计。partial 结果不用于宣称长短表达优劣。", "",
        "## 总体分数", "",
        "| 版本 | ACC@0.5 | mIoU | ACC@0.7 | 有效框 | 分母 |", "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        metrics = report["overall"][arm]
        lines.append(f"| {ARM_LABELS[arm]} | {metrics['acc_0.5']:.4f} | {metrics['mean_iou']:.4f} | "
                     f"{metrics['acc_0.7']:.4f} | {metrics['valid_predictions']} | {metrics['denominator']} |")
    paired = report["paired"]
    lines.extend([
        "", f"short 在 ACC@0.5 救回 {len(paired['short_rescued_ids_at_0.5'])} 道、损害 "
        f"{len(paired['short_harmed_ids_at_0.5'])} 道；净命中变化 {paired['short_net_hits_at_0.5']:+d}。",
        f"救回 IDs：{', '.join(paired['short_rescued_ids_at_0.5']) or '无'}。",
        f"损害 IDs：{', '.join(paired['short_harmed_ids_at_0.5']) or '无'}。", "",
        "预测框稳定性按两框 IoU ≥ 0.5 统计，只表示几何框重叠，不表示模型确定选择了同一对象。", "",
        "## relation_type 子集", "",
        "| relation_type | n | 长 ACC@0.5 | 长 mIoU | 长 ACC@0.7 | 短 ACC@0.5 | 短 mIoU | 短 ACC@0.7 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for label, group in report["subsets"]["relation_type"].items():
        lines.append(f"| {label} | {group['samples']} | {group['long']['acc_0.5']:.4f} | "
                     f"{group['long']['mean_iou']:.4f} | {group['long']['acc_0.7']:.4f} | "
                     f"{group['short']['acc_0.5']:.4f} | {group['short']['mean_iou']:.4f} | "
                     f"{group['short']['acc_0.7']:.4f} |")
    lines.extend([
        "", "## scene 子集", "",
        "| scene_id | n | 长 ACC@0.5 | 长 mIoU | 长 ACC@0.7 | 短 ACC@0.5 | 短 mIoU | 短 ACC@0.7 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for label, group in report["subsets"]["scene_id"].items():
        lines.append(f"| {label} | {group['samples']} | {group['long']['acc_0.5']:.4f} | "
                     f"{group['long']['mean_iou']:.4f} | {group['long']['acc_0.7']:.4f} | "
                     f"{group['short']['acc_0.5']:.4f} | {group['short']['mean_iou']:.4f} | "
                     f"{group['short']['acc_0.7']:.4f} |")
    lines.extend(["", "## 适用范围", ""])
    lines.extend(f"- {item}" for item in report["scope_limitations"])
    if report["partial"]:
        lines.extend(["", f"缺失预测 IDs：{', '.join(report['missing_prediction_ids'])}。",
                      "本轮为 partial；上面的数值将缺失和非法框按失败计入，只供排查进度，不据此判断哪种措辞更好。"])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    gallery = [
        "<!doctype html>", "<html lang=\"zh-CN\"><meta charset=\"utf-8\"><title>Query wording paired report</title>",
        "<style>body{font:16px system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#18212b}"
        ".task{border-top:1px solid #ccd2d8;padding:1.5rem 0;display:grid;grid-template-columns:minmax(260px,1fr) minmax(300px,1.1fr);gap:1.5rem}"
        "img{max-width:100%;height:auto}h2{font-size:17px;overflow-wrap:anywhere}table{border-collapse:collapse}td,th{padding:.5rem 1rem;border-bottom:1px solid #ccd2d8}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f8;padding:.7rem}"
        ".legend span{margin-right:1rem}.gt{color:#168b35}.long{color:#2465de}.short{color:#be6c00}"
        "@media(max-width:720px){.task{grid-template-columns:1fr}}</style><body>",
        "<h1>Query 长短表达配对比较</h1>",
        f"<p>状态：<b>{html.escape(report['status'])}</b>；{report['task_count']} 道题，分母固定为全部原始 bbox。缺失或非法框按 IoU=0 计。</p>",
        "<p class=\"legend\"><span class=\"gt\">绿：原始标注框</span><span class=\"long\">蓝：完整版</span><span class=\"short\">橙：简洁版</span></p>",
    ]
    gallery.append("<table><tr><th>版本</th><th>ACC@0.5 命中</th><th>mIoU</th><th>ACC@0.7 命中</th></tr>")
    for arm in ARMS:
        metrics = report["overall"][arm]
        gallery.append(f"<tr><td>{ARM_LABELS[arm]}</td><td>{metrics['hits_0.5']}/{metrics['samples']}</td>"
                       f"<td>{metrics['mean_iou']:.4f}</td><td>{metrics['hits_0.7']}/{metrics['samples']}</td></tr>")
    gallery.append(f"</table><p>简洁版救回 {len(paired['short_rescued_ids_at_0.5'])} 题，新增错误 {len(paired['short_harmed_ids_at_0.5'])} 题。以下保留全部题目，不拼接两版最优答案。</p>")
    if report["partial"]:
        gallery.append("<p><strong>partial：结果只作进度诊断，不据此宣称措辞优劣。</strong></p>")
    for index, entry in enumerate(report["entries"]):
        image_name = f"{index + 1:02d}_{re.sub(r'[^A-Za-z0-9_-]+', '_', entry['task_id']).strip('_')}.png"
        _write_overlay(entry, output_dir / "images" / image_name)
        gallery.extend([
            "<section class=\"task\"><div>",
            f"<h2>{html.escape(entry['task_id'])}</h2>",
            f"<p>scene: {html.escape(entry['scene_id'])} · relation_type: {html.escape(entry['relation_type'])}</p>",
            f"<img src=\"images/{html.escape(image_name, quote=True)}\" alt=\"RGB with GT, long and short boxes\">",
            f"<p>原始 GT: {html.escape(_fmt_box(entry['gt_bbox']))}</p>",
            "</div><div>",
            f"<h3>完整版 · IoU {entry['long_iou']:.4f}</h3>",
            f"<p>{html.escape(entry['long_query'])}</p><p>{html.escape(entry['long_query_zh'])}</p>",
            f"<details><summary>long 原始输出 / 框</summary><pre>{html.escape(json.dumps(entry['long_raw_output'], ensure_ascii=False, indent=2))}</pre><p>{html.escape(_fmt_box(entry['long_box']))}</p></details>",
            f"<h3>简洁版 · IoU {entry['short_iou']:.4f}</h3>",
            f"<p>{html.escape(entry['short_query'])}</p><p>{html.escape(entry['short_query_zh'])}</p>",
            f"<details><summary>short 原始输出 / 框</summary><pre>{html.escape(json.dumps(entry['short_raw_output'], ensure_ascii=False, indent=2))}</pre><p>{html.escape(_fmt_box(entry['short_box']))}</p></details>",
            f"<p>short 相对 long：{'救回' if entry['short_hit_0.5_rescue'] else '损害' if entry['short_hit_0.5_harm'] else '未改变 ACC@0.5'}；"
            f"两预测框 IoU：{_fmt_box(entry['prediction_box_iou'])}</p>",
            "</div></section>",
        ])
    gallery.append("</body></html>")
    (output_dir / "index.html").write_text("\n".join(gallery) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True, help="JSON list of paired wording variants")
    parser.add_argument("--candidates", type=Path, required=True, help="original candidate JSONL")
    parser.add_argument("--predictions", type=Path, required=True, help="evaluator predictions JSONL")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    pairs = _load_pairs(args.pairs)
    candidates = _index_candidates(args.candidates, pairs)
    predictions, missing = _prediction_rows(args.predictions, pairs)
    invalid = sorted(sample_id for sample_id, row in predictions.items()
                     if row.get("_failure") is not None)
    report = build_report(pairs, candidates, predictions, missing, invalid)
    write_outputs(report, args.output_dir)
    print(json.dumps({key: report[key] for key in
                      ("status", "task_count", "expected_prediction_rows", "observed_prediction_rows",
                       "missing_prediction_ids", "overall", "paired")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
