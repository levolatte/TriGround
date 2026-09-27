"""Thin local evaluator for NVIDIA LocateAnything-3B.

This module deliberately keeps the heavy LocateAnything worker and torch imports
inside ``run_inference``.  Parsing, manifest handling, resume validation, and
the CPU tests therefore work without the model environment.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

# ``python tools/evaluate_locateanything.py`` puts only ``tools/`` on
# sys.path; add the project root so the pure-Python report helper remains
# importable without importing the heavy model stack.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.report_rematch_experiment import box_iou, load_manifest


DEFAULT_WORKER_CONFIG = {
    "use_batch_runtime": True,
    "attn": "la_flash",
    "vision_attn": "flash_attention_2",
    "scheduler": "pipeline",
    "strict_attn": True,
    "device": "cuda",
}
DEFAULT_GROUND_KWARGS = {
    "generation_mode": "hybrid",
    "max_new_tokens": 2048,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 0,
    "repetition_penalty": 1.0,
    "verbose": True,
}
MINIMUM_DEPENDENCIES = {
    "python": ">=3.10",
    "runtime": ["torch with CUDA", "Pillow", "transformers"],
    "official_worker": ["locateanything_worker.py", "nvidia/LocateAnything-3B local files"],
    "batch_runtime": ["batch_utils/", "kernel_utils/", "FlashAttention runtime"],
    "official_pins_reported_by_README": {
        "transformers": "4.57.1",
        "tokenizers": "0.22.0",
        "deepspeed": "0.15.4",
        "accelerate": "1.5.2",
        "timm": ">=1.0.11",
        "liger_kernel": "0.3.1",
        "peft": "0.12.0",
        "decord": "unversioned in README line",
    },
    "installed_by_this_tool": False,
}


BOX_PATTERN = re.compile(
    r"<box>\s*<([0-9]+)>\s*<([0-9]+)>\s*<([0-9]+)>\s*<([0-9]+)>\s*</box>",
    flags=re.IGNORECASE,
)


def parse_boxes(answer: Any) -> list[list[float]]:
    """Parse all valid LocateAnything ``<box><x>...</box>`` outputs.

    The worker's coordinates are integers in 0--1000.  Invalid boxes are
    ignored, so the first remaining box is the model's single-box prediction;
    ``<box>none</box>`` naturally yields an empty list.
    """
    if not isinstance(answer, str):
        return []
    boxes: list[list[float]] = []
    for match in BOX_PATTERN.finditer(answer):
        values = [int(value) for value in match.groups()]
        if any(value > 1000 for value in values):
            continue
        if values[0] >= values[2] or values[1] >= values[3]:
            continue
        boxes.append([value / 1000.0 for value in values])
    return boxes


def parse_first_box(answer: Any) -> list[float] | None:
    boxes = parse_boxes(answer)
    return boxes[0] if boxes else None


def resolve_rgb_path(manifest_path: Path, value: Any) -> Path:
    """Resolve a manifest ``visible`` path relative to the manifest directory."""
    if not isinstance(value, str) or not value:
        raise ValueError("manifest visible must be a non-empty path string")
    path = Path(value.replace("\\", "/"))
    return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()


def selected_records(
    manifest: dict[str, dict[str, Any]], limit: int
) -> list[tuple[str, dict[str, Any]]]:
    if limit < 0:
        raise ValueError("--limit must be non-negative")
    records = list(manifest.items())
    return records[:limit] if limit else records


def build_run_config(
    *,
    model: str,
    manifest: Path,
    output_dir: Path,
    limit: int,
    worker_dir: Path | None,
) -> dict[str, Any]:
    return {
        "model": model,
        "manifest": str(manifest.resolve()),
        "output_dir": str(output_dir.resolve()),
        "limit": limit,
        "worker_dir": str(worker_dir.resolve()) if worker_dir is not None else None,
        "worker_constructor_kwargs": dict(DEFAULT_WORKER_CONFIG),
        "ground_multi_kwargs": dict(DEFAULT_GROUND_KWARGS),
        "image_processing": {
            "rgb_field": "visible",
            "relative_to": "manifest.parent",
            "convert": "PIL.Image.convert('RGB')",
            "resize": False,
        },
        "bbox_format": "normalized_xyxy_0_1",
        "model_output_format": "<box><x1><y1><x2><y2></box>, integer coordinates 0..1000",
        "minimum_dependencies": MINIMUM_DEPENDENCIES,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: row must be a JSON object")
        rows.append(row)
    return rows


def validate_resume_rows(
    rows: list[dict[str, Any]], records: list[tuple[str, dict[str, Any]]]
) -> dict[str, dict[str, Any]]:
    expected = dict(records)
    existing: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows, 1):
        if "id" not in row:
            raise ValueError(f"resume prediction at row {index} has no id")
        sample_id = str(row["id"])
        if sample_id not in expected:
            raise ValueError(f"resume prediction has unknown selected ID: {sample_id}")
        if sample_id in existing:
            raise ValueError(f"resume predictions contain duplicate ID: {sample_id}")
        if row.get("target") != expected[sample_id]["bbox"]:
            raise ValueError(f"resume target mismatch for sample ID {sample_id}")
        existing[sample_id] = row
    return existing


def rescore_rows(
    rows_by_id: dict[str, dict[str, Any]], records: list[tuple[str, dict[str, Any]]]
) -> dict[str, Any]:
    ordered_rows = [rows_by_id[sample_id] for sample_id, _ in records]
    ious: list[float] = []
    parsed = 0
    for row, (_, record) in zip(ordered_rows, records):
        prediction = row.get("prediction")
        if prediction is None:
            iou = 0.0
        else:
            if not isinstance(prediction, list) or len(prediction) != 4:
                raise ValueError(f"prediction for {row['id']} is not normalized xyxy")
            iou = box_iou(prediction, record["bbox"])
            parsed += 1
        ious.append(iou)
    count = len(ious)
    if count == 0:
        raise ValueError("cannot summarize an empty evaluation")
    return {
        "samples": count,
        "parsed": parsed,
        "parse_failures": count - parsed,
        "parse_rate": parsed / count,
        "mean_iou": sum(ious) / count,
        "hits": sum(iou >= 0.5 for iou in ious),
        "acc_0.5": sum(iou >= 0.5 for iou in ious) / count,
        "acc_0.7": sum(iou >= 0.7 for iou in ious) / count,
    }


def _summary(
    *,
    run_config: dict[str, Any],
    manifest_samples: int,
    records: list[tuple[str, dict[str, Any]]],
    rows_by_id: dict[str, dict[str, Any]],
    predictions_path: Path,
) -> dict[str, Any]:
    return {
        **run_config,
        "manifest_samples": manifest_samples,
        "evaluated_samples": len(records),
        "predictions": str(predictions_path.resolve()),
        **rescore_rows(rows_by_id, records),
        "total_sample_seconds": sum(row.get("latency_seconds", 0.0) for row in rows_by_id.values()),
        "gpu_peak_allocated_bytes": max((row.get("gpu_peak_allocated_bytes", 0) for row in rows_by_id.values()), default=0),
        "gpu_peak_reserved_bytes": max((row.get("gpu_peak_reserved_bytes", 0) for row in rows_by_id.values()), default=0),
    }


def write_report(summary: dict[str, Any], path: Path) -> None:
    constructor = json.dumps(summary["worker_constructor_kwargs"], ensure_ascii=False, indent=2)
    generation = json.dumps(summary["ground_multi_kwargs"], ensure_ascii=False, indent=2)
    dependencies = json.dumps(summary["minimum_dependencies"], ensure_ascii=False, indent=2)
    text = "\n".join(
        [
            "# LocateAnything-3B 评估报告",
            "",
            f"模型：`{summary['model']}`；样本：{summary['evaluated_samples']}/{summary['manifest_samples']}。",
            "预测中的 target、IoU 与 ACC 均按原始 manifest bbox 重算。",
            "",
            "## 官方 worker 构造参数",
            "",
            "```json",
            constructor,
            "```",
            "",
            "## `ground_multi(image, phrase, **kwargs)` 实际参数",
            "",
            "```json",
            generation,
            "```",
            "",
            "## 图像与解析约定",
            "",
            "- 只读取 `visible`，相对路径基于 `manifest.parent`。",
            "- 仅转换为 RGB，不自行 resize，由官方 processor 保留原生输入处理。",
            "- 第一有效 `<box>` 作为单框 prediction，全部有效框写入 `candidates`；`<box>none</box>` 计为解析失败。",
            "- 本工具不安装依赖、不下载权重。",
            "",
            "## 最小依赖清单",
            "",
            "```json",
            dependencies,
            "```",
        ]
    )
    path.write_text(text + "\n", encoding="utf-8")


def _add_worker_dir(worker_dir: Path | None) -> None:
    if worker_dir is None:
        return
    resolved = str(worker_dir.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def run_inference(
    *,
    model: str,
    manifest_path: Path,
    output_dir: Path,
    limit: int = 0,
    resume: bool = False,
    worker_dir: Path | None = None,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    records = selected_records(manifest, limit)
    if not records:
        raise ValueError("selected manifest is empty")
    for _, record in records:
        if not isinstance(record.get("query"), str):
            raise ValueError("manifest query must be a string")

    manifest_path = manifest_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"
    config_path = output_dir / "run_config.json"
    report_path = output_dir / "report.md"
    run_config = build_run_config(
        model=model,
        manifest=manifest_path,
        output_dir=output_dir,
        limit=limit,
        worker_dir=worker_dir,
    )

    existing_by_id: dict[str, dict[str, Any]] = {}
    if resume:
        if rows_path.exists() and not config_path.exists():
            raise FileNotFoundError(f"cannot resume {rows_path}: run_config.json is missing")
        if config_path.exists():
            saved = json.loads(config_path.read_text(encoding="utf-8-sig"))
            if saved != run_config:
                raise ValueError("resume run_config differs from the current invocation")
        if rows_path.exists():
            existing_by_id = validate_resume_rows(_read_jsonl(rows_path), records)
        elif summary_path.exists():
            raise FileNotFoundError(
                f"cannot resume {output_dir}: summary.json exists but predictions.jsonl is missing"
            )
    elif any(path.exists() for path in (rows_path, summary_path, config_path)):
        raise FileExistsError(f"output directory already contains a run: {output_dir}; use --resume")

    if not config_path.exists():
        config_path.write_text(
            json.dumps(run_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    missing = [(sample_id, record) for sample_id, record in records if sample_id not in existing_by_id]
    if missing:
        for _, record in missing:
            image_path = resolve_rgb_path(manifest_path, record["visible"])
            if not image_path.is_file():
                raise FileNotFoundError(f"visible image does not exist: {image_path}")

        model_path = Path(model).expanduser()
        if not model_path.is_dir():
            raise FileNotFoundError(
                f"--model must point to an existing local model directory: {model_path}"
            )
        if worker_dir is not None and not worker_dir.is_dir():
            raise FileNotFoundError(f"--worker-dir does not exist: {worker_dir}")
        # The HF release ships batch_utils/kernel_utils beside the weights.
        # Keep the explicitly selected worker ahead of those runtime modules.
        _add_worker_dir(model_path)
        _add_worker_dir(worker_dir)
        import torch

        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        from PIL import Image
        from locateanything_worker import LocateAnythingWorker

        load_started = time.perf_counter()
        worker = LocateAnythingWorker(str(model_path.resolve()), **DEFAULT_WORKER_CONFIG)
        torch.cuda.synchronize()
        import transformers

        runtime = {
            "started_at_unix": time.time(), "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "transformers": transformers.__version__, "tf32": False,
            "model_load_seconds": time.perf_counter() - load_started,
            "resumed_samples": len(existing_by_id),
        }
        with (output_dir / "runtime_history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(runtime) + "\n")
        print(json.dumps({"event": "model_ready", **runtime}), flush=True)
        file_mode = "a" if rows_path.exists() else "w"
        with rows_path.open(file_mode, encoding="utf-8") as handle:
            if file_mode == "a" and rows_path.stat().st_size:
                last = rows_path.read_bytes()[-1:]
                if last not in {b"\n", b"\r"}:
                    handle.write("\n")
            for index, (sample_id, record) in enumerate(records):
                if sample_id in existing_by_id:
                    continue
                image_path = resolve_rgb_path(manifest_path, record["visible"])
                started = time.perf_counter()
                with Image.open(image_path) as source:
                    image = source.convert("RGB")
                    result = worker.ground_multi(image, record["query"], **DEFAULT_GROUND_KWARGS)
                if not isinstance(result, dict):
                    raise TypeError("LocateAnything ground_multi must return a result dict")
                answer = result.get("answer")
                candidates = parse_boxes(answer)
                prediction = candidates[0] if candidates else None
                iou = box_iou(prediction, record["bbox"]) if prediction is not None else 0.0
                row = {
                    "index": index,
                    "id": sample_id,
                    "image": str(image_path),
                    "query": record["query"],
                    "target": list(record["bbox"]),
                    "prediction": prediction,
                    "candidates": candidates,
                    "parsed": prediction is not None,
                    "iou": iou,
                    "acc_0.5": iou >= 0.5,
                    "hit": iou >= 0.5,
                    "latency_seconds": time.perf_counter() - started,
                    "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "raw_text": answer,
                    "failure_reason": None if candidates else "no_valid_box",
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                existing_by_id[sample_id] = row
                print(json.dumps({"done": len(existing_by_id), "total": len(records),
                                  "id": sample_id, "parsed": row["parsed"],
                                  "iou": iou, "seconds": row["latency_seconds"]}), flush=True)

    if len(existing_by_id) != len(records):
        raise RuntimeError("evaluation ended without one row per selected manifest ID")
    summary = _summary(
        run_config=run_config,
        manifest_samples=len(manifest),
        records=records,
        rows_by_id=existing_by_id,
        predictions_path=rows_path,
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(summary, report_path)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local LocateAnything-3B model path")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates the full manifest")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker-dir", type=Path, help="Directory containing locateanything_worker.py")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_inference(
        model=args.model,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        limit=args.limit,
        resume=args.resume,
        worker_dir=args.worker_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
