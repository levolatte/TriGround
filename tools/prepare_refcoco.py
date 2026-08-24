from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from PIL import Image


DATASETS = {
    "refcoco": "jxu124/refcoco",
    "refcocoplus": "jxu124/refcocoplus",
    "refcocog": "jxu124/refcocog",
}


def _mapping(value):
    if isinstance(value, str):
        return json.loads(value)
    return value


def _sentences(row: dict) -> list[dict]:
    values = row.get("sentences") or _mapping(row.get("raw_sentences")) or []
    return values if isinstance(values, list) else []


def _image_info(row: dict) -> dict:
    value = _mapping(row.get("raw_image_info"))
    return value if isinstance(value, dict) else {}


def _normalized_xyxy(row: dict, width: int, height: int) -> list[float]:
    box = [float(value) for value in row["bbox"]]
    if len(box) != 4:
        raise ValueError(f"bbox 不是四元组: {box}")
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"bbox 不是合法 xyxy: {box}, image={width}x{height}")
    return [round(x1 / width, 7), round(y1 / height, 7), round(x2 / width, 7), round(y2 / height, 7)]


def _parquet_files(snapshot: Path) -> list[Path]:
    files = sorted(snapshot.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"Hugging Face snapshot 中没有 parquet: {snapshot}")
    return files


def convert_dataset(name: str, snapshot: Path, coco_dir: Path, output: Path) -> dict[str, int]:
    split_rows: dict[str, list[dict]] = {"train": [], "val": []}
    missing_images: list[str] = []
    for parquet in _parquet_files(snapshot):
        parquet_file = pq.ParquetFile(parquet)
        for batch in parquet_file.iter_batches(batch_size=512):
            for row in batch.to_pylist():
                raw_split = str(row.get("split", "")).lower()
                split = "train" if raw_split == "train" else "val" if raw_split in {"val", "validation"} else None
                if split is None:
                    continue
                info = _image_info(row)
                # HF rows may expose a cropped-image name such as
                # COCO_train2014_..._2.jpg. Stage 1 uses the original COCO
                # archive, whose canonical name is stored in raw_image_info.
                file_name = str(info.get("file_name") or row.get("file_name") or "")
                image_path = coco_dir / file_name
                if not image_path.is_file():
                    if len(missing_images) < 10:
                        missing_images.append(str(image_path))
                    continue
                width = int(info.get("width", 0))
                height = int(info.get("height", 0))
                if width <= 0 or height <= 0:
                    with Image.open(image_path) as image:
                        width, height = image.size
                bbox = _normalized_xyxy(row, width, height)
                for sentence_index, sentence in enumerate(_sentences(row)):
                    query = str(sentence.get("raw") or sentence.get("sent") or "").strip()
                    if not query:
                        continue
                    sent_id = sentence.get("sent_id", sentence_index)
                    split_rows[split].append({
                        "id": f"{name}_{row.get('ref_id', row.get('ann_id'))}_{sent_id}",
                        "rgb": image_path.resolve().as_posix(),
                        "query": query,
                        "bbox": bbox,
                    })
    if missing_images:
        raise FileNotFoundError(
            "COCO train2014 图像不完整；示例缺失文件:\n" + "\n".join(missing_images)
        )
    output.mkdir(parents=True, exist_ok=True)
    for split, rows in split_rows.items():
        with (output / f"{name}_{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {split: len(rows) for split, rows in split_rows.items()}


def merge_manifests(output: Path) -> dict[str, int]:
    counts = {}
    for split in ("train", "val"):
        sources = [output / f"{name}_{split}.jsonl" for name in DATASETS]
        merged = output / f"{split}.jsonl"
        count = 0
        with merged.open("w", encoding="utf-8") as destination:
            for source in sources:
                for line in source.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        destination.write(line + "\n")
                        count += 1
        counts[split] = count
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Download RefCOCO metadata and build Stage-1 manifests")
    parser.add_argument("--coco-dir", required=True, help="Extracted COCO train2014 directory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hf-cache", required=True)
    parser.add_argument("--token", default=None)
    args = parser.parse_args()

    coco_dir = Path(args.coco_dir).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    cache = Path(args.hf_cache).expanduser().resolve()
    if not coco_dir.is_dir():
        raise FileNotFoundError(f"COCO train2014 目录不存在: {coco_dir}")
    report = {"datasets": {}}
    for name, repo_id in DATASETS.items():
        snapshot = Path(snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            cache_dir=cache,
            token=args.token,
            allow_patterns=["*.parquet", "**/*.parquet", "README.md"],
        ))
        report["datasets"][name] = convert_dataset(name, snapshot, coco_dir, output)
    report["merged"] = merge_manifests(output)
    report["output_dir"] = str(output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
