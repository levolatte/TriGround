"""Render every reported rescue/harm as a small human-review HTML atlas.

Normal prediction rows provide the preferred image paths. The native manifest,
raw GT, and an optional local image root are fallbacks for copied cloud reports.
Reason labels are deliberately left for a human and can be exported from HTML.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image

from tools.build_gu_error_atlas import mark
from tools.report_rematch_experiment import box_iou, load_manifest, parse_run_spec
from tools.report_triground_abv import _load_optional


COLORS = {"C0": "#2463eb", "A": "#f59e0b", "B": "#ef4444", "V": "#8b5cf6"}
REASONS = (
    ("unknown", "未知／待人工核对"),
    ("wrong_object", "对象识别错误"),
    ("same_object_bad_box", "同一对象但框不准"),
    ("evidence_failure", "IR／Depth 证据失效"),
    ("parse_failure", "解析失败"),
)


def _prediction_rows(path: Path, manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    run = _load_optional(path.stem, path, manifest, require_run_config=False)
    return run["rows"]


def _native_rows(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        return {str(key): value for key, value in payload.items()}
    if isinstance(payload, list):
        return {str(row["id"]): row for row in payload}
    raise ValueError("native manifest must be an ID-keyed object or list of ID records")


def _image_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _native_images(row: dict[str, Any]) -> list[str]:
    if "image" in row:
        return _image_list(row["image"])
    images = row.get("images")
    if isinstance(images, dict):
        return [str(images[key]) for key in ("rgb", "infrared", "depth") if images.get(key)]
    return _image_list(images)


def _modality_from_sequence(images: list[str], modality: str, gt: dict[str, Any]) -> str | None:
    if not images:
        return None
    if modality == "visible":
        return images[0]
    if len(images) == 3:
        return images[1 if modality == "infrared" else 2]
    if len(images) == 2:
        if modality == "infrared" and gt.get("infrared") and not gt.get("depth"):
            return images[1]
        if modality == "depth" and gt.get("depth") and not gt.get("infrared"):
            return images[1]
    return None


def _candidate(path_text: str | None, base: Path | None) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    if path.is_file():
        return path
    if base is not None and not path.is_absolute() and (base / path).is_file():
        return base / path
    return None


def _resolve_images(
    sample_id: str, gt: dict[str, Any], rows: dict[str, dict[str, Any]],
    native: dict[str, dict[str, Any]], native_path: Path | None, image_root: Path | None,
) -> dict[str, Path | None]:
    normal_sequences = [_image_list(row["raw"].get("image"))
                        for name, row in rows.items() if row and ":" not in name]
    native_sequence = _native_images(native.get(sample_id, {}))
    selected: dict[str, Path | None] = {}
    for modality in ("visible", "infrared", "depth"):
        candidates = [
            _modality_from_sequence(sequence, modality, gt) for sequence in normal_sequences
        ] + [
            _modality_from_sequence(native_sequence, modality, gt),
            str(gt.get(modality) or ""),
        ]
        path = next(
            (resolved for value in candidates
             if (resolved := _candidate(value, native_path.parent if native_path else None)) is not None),
            None,
        )
        if path is None and image_root is not None:
            basename = Path(str(gt.get(modality) or gt.get("visible") or "")).name
            folders = ("visible", "rgb") if modality == "visible" else (
                ("infrared", "ir") if modality == "infrared" else ("depth", "depth_visual", "depth_rgb")
            )
            path = next((image_root / folder / basename for folder in folders
                         if (image_root / folder / basename).is_file()), None)
        selected[modality] = path
    return selected


def _query(gt: dict[str, Any], rows: dict[str, dict[str, Any]], native: dict[str, Any]) -> str:
    for value in (gt.get("query"), native.get("query"),
                  *(row["raw"].get("query") for row in rows.values() if row is not None)):
        if isinstance(value, str) and value.strip():
            return value.strip()
    prompts = [row["raw"].get("prompt") for row in rows.values() if row is not None]
    if native.get("conversations"):
        prompts.append(native["conversations"][0].get("value"))
    for prompt in prompts:
        if not isinstance(prompt, str):
            continue
        match = re.search(r"Locate the object described by this query:\s*(.*?)\s*Return only JSON",
                          prompt, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    return "Query 未在输入材料中单列；请核对原生清单。"


def _selected_flips(report: dict[str, Any], manifest: dict[str, Any]) -> dict[str, list[str]]:
    selected: dict[str, list[str]] = {}
    for pair_name, pair in report["pairs"].items():
        for kind, label in (("wrong_to_right", "救回"), ("right_to_wrong", "伤害")):
            for sample_id in pair[kind]["ids"]:
                if sample_id not in manifest:
                    raise ValueError(f"report flip {sample_id!r} is absent from GT")
                selected.setdefault(sample_id, []).append(f"{pair_name}：{label}")
    return selected


def _save_preview(source: Path, destination: Path, boxes: list[tuple[list[float], str]] = ()) -> None:
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    image.thumbnail((900, 900))
    for box, color in boxes:
        mark(image, box, color, width=4)
    image.save(destination, quality=88)


def build(
    report_path: Path, gt_path: Path, specs: list[str], output_dir: Path,
    *, native_manifest: Path | None = None, image_root: Path | None = None,
) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    gt = load_manifest(gt_path)
    if report["manifest_samples"] != len(gt):
        raise ValueError("report and GT have different sample denominators")
    parsed = [parse_run_spec(spec) for spec in specs]
    if len({name for name, _ in parsed}) != len(parsed):
        raise ValueError("normal run names must be unique")
    runs = {name: _prediction_rows(path, gt) for name, path in parsed}
    native = _native_rows(native_manifest)
    selected = _selected_flips(report, gt)
    if output_dir.resolve() == report_path.parent.resolve():
        raise ValueError("atlas output directory must differ from report directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    visuals = output_dir / "visuals"
    visuals.mkdir(exist_ok=True)
    cards = []
    cases = []
    missing_images = []
    for index, sample_id in enumerate(gt, 1):
        if sample_id not in selected:
            continue
        record = gt[sample_id]
        rows = {name: row.get(sample_id) for name, row in runs.items()}
        paths = _resolve_images(sample_id, record, rows, native, native_manifest, image_root)
        query = _query(record, rows, native.get(sample_id, {}))
        figures = []
        rgb = paths["visible"]
        if rgb is not None:
            for run_index, (name, row) in enumerate(rows.items()):
                if row is None:
                    continue
                prediction = row["primary"]
                overlays = [(record["bbox"], "#00bb73")]
                if prediction is not None:
                    overlays.append((prediction, COLORS.get(name, "#e11d48")))
                filename = f"{index:04d}_run{run_index:02d}.jpg"
                _save_preview(rgb, visuals / filename, overlays)
                iou = box_iou(prediction, record["bbox"]) if prediction is not None else 0.0
                figures.append(
                    f'<figure><img src="visuals/{html.escape(filename)}" alt="RGB {html.escape(name)}">'
                    f'<figcaption>RGB：GT 绿框、{html.escape(name)} {html.escape(COLORS.get(name, "#e11d48"))} 框；'
                    f'IoU {iou:.3f}；解析{"成功" if prediction is not None else "失败"}</figcaption></figure>'
                )
        for modality, title in (("infrared", "IR 原图"), ("depth", "Depth 原图／可视化")):
            source = paths[modality]
            if source is None:
                missing_images.append({"id": sample_id, "modality": modality,
                                       "gt_path": record.get(modality)})
                continue
            filename = f"{index:04d}_{modality}.jpg"
            _save_preview(source, visuals / filename)
            figures.append(f'<figure><img src="visuals/{filename}" alt="{title}">'
                           f'<figcaption>{title}；不叠加 RGB 坐标框</figcaption></figure>')
        if rgb is None:
            missing_images.append({"id": sample_id, "modality": "visible",
                                   "gt_path": record.get("visible")})
            figures.insert(0, "<p>RGB 原图未找到；预测数值仍列于下方。</p>")
        predictions = {}
        for name, row in rows.items():
            prediction = row["primary"] if row is not None else None
            predictions[name] = {
                "box": prediction,
                "iou": box_iou(prediction, record["bbox"]) if prediction is not None else 0.0,
                "raw_text": row["raw"].get("raw_text") if row is not None else None,
            }
        cases.append({"id": sample_id, "query": query, "flips": selected[sample_id],
                      "gt": record["bbox"], "predictions": predictions,
                      "images": {key: str(value) if value else None for key, value in paths.items()},
                      "human_reason": "unknown", "human_note": ""})
        options = "".join(f'<option value="{value}">{html.escape(label)}</option>'
                          for value, label in REASONS)
        flip_text = "；".join(selected[sample_id])
        prediction_text = "；".join(
            f"{name}: {item['box']} / IoU {item['iou']:.3f} / raw {item['raw_text']}"
            for name, item in predictions.items()
        )
        cards.append(
            f'<section class="case" data-id="{html.escape(sample_id, quote=True)}">'
            f'<h2>{html.escape(sample_id)}</h2>'
            f'<p><b>原 Query：</b>{html.escape(query)}</p>'
            f'<p><b>翻转：</b>{html.escape(flip_text)}</p>'
            f'<div class="grid">{"".join(figures)}</div>'
            f'<p><b>原始数值：</b>{html.escape(prediction_text)}</p>'
            f'<label>人工原因 <select class="reason">{options}</select></label> '
            f'<label>人工备注 <textarea class="note" rows="2"></textarea></label>'
            f'</section>'
        )
    (output_dir / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n",
                                             encoding="utf-8")
    (output_dir / "missing_images.json").write_text(
        json.dumps(missing_images, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    page = """<!doctype html><html lang="zh"><meta charset="utf-8"><title>A/B/V 翻转人工图册</title>
<style>body{font:16px/1.55 system-ui;max-width:1700px;margin:24px auto;background:#f6f7f8;color:#17202a}
.case{background:white;padding:20px;margin:20px 0;border:1px solid #ddd}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}
figure{margin:0;border:1px solid #ddd}img{width:100%}figcaption{padding:7px}textarea{vertical-align:top;min-width:360px}
@media(max-width:900px){.grid{grid-template-columns:1fr}}</style>
<h1>A/B/V 胜负翻转人工图册</h1><p>只收录报告中的全部救回与伤害 ID。GT 绿框和各组预测彩框仅叠在 RGB；IR/Depth 保留原图，不假定同坐标对齐。
原因字段默认“未知”，需要人工观察后填写。点击导出可下载 review.json。</p>
<button id="export">导出人工原因 JSON</button>
"""
    page += "\n".join(cards)
    page += """<script>
document.getElementById('export').addEventListener('click', () => {
  const rows = [...document.querySelectorAll('section.case')].map(s => ({
    id: s.dataset.id, reason: s.querySelector('select.reason').value,
    note: s.querySelector('textarea.note').value
  }));
  const blob = new Blob([JSON.stringify(rows, null, 2)], {type:'application/json'});
  const link = document.createElement('a'); link.href = URL.createObjectURL(blob);
  link.download = 'review.json'; link.click(); URL.revokeObjectURL(link.href);
});
</script></html>"""
    (output_dir / "cases.html").write_text(page, encoding="utf-8")
    result = {"report": str(report_path.resolve()), "gt": str(gt_path.resolve()),
              "flip_cases": len(cases), "normal_runs": list(runs),
              "missing_image_entries": len(missing_images)}
    (output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                              encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True, help="report_triground_abv summary.json")
    parser.add_argument("--gt", type=Path, required=True, help="raw GT ID-to-record JSON")
    parser.add_argument("--run", action="append", required=True, metavar="NAME=predictions.jsonl")
    parser.add_argument("--native-manifest", type=Path, help="actual original image list and prompt")
    parser.add_argument("--image-root", type=Path, help="local RGB/IR/Depth root when paths moved")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.report, args.gt, args.run, args.output_dir,
                           native_manifest=args.native_manifest,
                           image_root=args.image_root), ensure_ascii=False))


if __name__ == "__main__":
    main()
