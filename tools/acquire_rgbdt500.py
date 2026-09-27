"""Acquire annotated RGBDT500 train frames from the official ZIP.

HTTP Range transfers only selected entries; --local-zip reads a completed local archive.
The plan always reserves 20 sequences for an external review set.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import shutil
import struct
import subprocess
import time
import zlib
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image


ARCHIVE_URL = (
    "https://drive.usercontent.google.com/download"
    "?id=1DGH4YNRpiJcNG6VnbWuzV1pP5-ZYYlwq&export=download&confirm=t"
)
ARCHIVE_SIZE = 26_447_029_453
SEED = 2026
DEPTH_RAW_MAX = 19_999
CURL = shutil.which("curl") or shutil.which("curl.exe")


def http_range(start: int, end: int) -> bytes:
    if CURL is None:
        raise RuntimeError("curl is required for HTTP Range acquisition")
    for attempt in range(3):
        result = subprocess.run(
            [CURL, "-fsSL", "--max-time", "120", "-r", f"{start}-{end}", ARCHIVE_URL],
            capture_output=True,
        )
        data = result.stdout
        if result.returncode == 0 and len(data) == end - start + 1:
            return data
        if attempt < 2:
            time.sleep(2**attempt)
    raise RuntimeError(
        f"HTTP Range {start}-{end} failed after 3 attempts: "
        f"curl exit {result.returncode}, {len(data)} bytes"
    )


class RangeReader(io.RawIOBase):
    def __init__(self) -> None:
        self.position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_END:
            self.position = ARCHIVE_SIZE + offset
        elif whence == io.SEEK_CUR:
            self.position += offset
        else:
            self.position = offset
        return self.position

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = ARCHIVE_SIZE - self.position
        if not size:
            return b""
        end = min(self.position + size, ARCHIVE_SIZE) - 1
        data = http_range(self.position, end)
        self.position += len(data)
        return data


def load_index(root: Path) -> tuple[dict[str, tuple[zipfile.ZipInfo, int]], int]:
    cache = root / "zip_index.json"
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        index = {
            item["filename"]: (
                SimpleNamespace(
                    filename=item["filename"],
                    header_offset=item["header_offset"],
                    compress_size=item["compress_size"],
                    file_size=item["file_size"],
                    compress_type=item["compress_type"],
                ),
                item["next_offset"],
            )
            for item in data
        }
        return index, len(index)
    with zipfile.ZipFile(RangeReader()) as archive:
        infos = sorted(archive.infolist(), key=lambda item: item.header_offset)
        end_of_files = archive.start_dir
    index = {
        info.filename: (info, infos[i + 1].header_offset if i + 1 < len(infos) else end_of_files)
        for i, info in enumerate(infos)
    }
    cache.write_text(
        json.dumps(
            [
                {
                    "filename": name,
                    "header_offset": info.header_offset,
                    "compress_size": info.compress_size,
                    "file_size": info.file_size,
                    "compress_type": info.compress_type,
                    "next_offset": next_offset,
                }
                for name, (info, next_offset) in index.items()
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return index, len(infos)


def read_entry(entry: tuple[zipfile.ZipInfo, int]) -> bytes:
    info, next_offset = entry
    region = http_range(info.header_offset, next_offset - 1)
    if region[:4] != b"PK\x03\x04":
        raise ValueError(f"invalid ZIP local header for {info.filename}")
    name_len, extra_len = struct.unpack_from("<HH", region, 26)
    start = 30 + name_len + extra_len
    compressed = region[start : start + info.compress_size]
    if len(compressed) != info.compress_size:
        raise ValueError(f"truncated ZIP entry {info.filename}")
    if info.compress_type == zipfile.ZIP_STORED:
        content = compressed
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        content = zlib.decompress(compressed, -15)
    else:
        raise ValueError(f"unsupported ZIP method {info.compress_type} for {info.filename}")
    if len(content) != info.file_size:
        raise ValueError(f"uncompressed size mismatch for {info.filename}")
    return content


def make_plan(index: dict[str, tuple[zipfile.ZipInfo, int]], root: Path, read_source=read_entry) -> dict:
    sequences = sorted(
        {
            name.split("/")[0]
            for name in index
            if name.endswith("/groundtruth.txt")
        }
    )
    if sequences != [f"{i:03d}" for i in range(1, 401)]:
        raise ValueError(f"expected 400 train sequences; got {len(sequences)}")
    rng = random.Random(SEED)
    shuffled = list(sequences)
    rng.shuffle(shuffled)
    review_sequences = shuffled[:20]
    pilot_sequences = shuffled[20:120]

    def frames_for(sequence: str) -> list[str]:
        return sorted(
            name.rsplit("/", 1)[-1]
            for name in index
            if name.startswith(f"{sequence}/color/") and name.endswith(".png")
        )

    pilot = []
    for sequence in pilot_sequences:
        frames = frames_for(sequence)
        if len(frames) != 10:
            raise ValueError(f"expected 10 annotated frames for {sequence}")
        pilot.append({"sequence": sequence, "frame": frames[rng.randrange(10)]})
    review = []
    for i, sequence in enumerate(review_sequences):
        frames = frames_for(sequence)
        count = 3 if i < 10 else 2
        for frame in rng.sample(frames, count):
            review.append({"sequence": sequence, "frame": frame})

    def annotated_rows(sequence: str) -> dict[str, tuple[float, float, float, float]]:
        for subset in ("pilot", "external_review"):
            cached = root / subset / sequence / "groundtruth.txt"
            if cached.is_file():
                return parse_groundtruth(cached.read_bytes())
        metadata = root / "metadata" / "groundtruth" / f"{sequence}.txt"
        if not metadata.exists():
            metadata.parent.mkdir(parents=True, exist_ok=True)
            metadata.write_bytes(read_source(index[f"{sequence}/groundtruth.txt"]))
        return parse_groundtruth(metadata.read_bytes())

    for group in (pilot, review):
        original_valid: dict[str, set[str]] = {}
        for item in group:
            rows = annotated_rows(item["sequence"])
            _, _, width, height = rows[item["frame"]]
            if width > 0 and height > 0:
                original_valid.setdefault(item["sequence"], set()).add(item["frame"])
        used: dict[str, set[str]] = {}
        for item in group:
            sequence = item["sequence"]
            already_used = used.setdefault(sequence, set())
            rows = annotated_rows(sequence)
            valid = sorted(frame for frame, (_, _, width, height) in rows.items() if width > 0 and height > 0)
            if not valid:
                raise ValueError(f"no valid annotated target in train sequence {sequence}")
            selected = item["frame"]
            if selected not in valid or selected in already_used:
                candidates = [frame for frame in valid if frame > selected and frame not in already_used and frame not in original_valid.get(sequence, set())]
                if not candidates:
                    candidates = [frame for frame in valid if frame not in already_used and frame not in original_valid.get(sequence, set())]
                if not candidates:
                    raise ValueError(f"not enough unique valid frames in train sequence {sequence}")
                selected = candidates[0]
                item["frame"] = selected
            already_used.add(selected)
    return {
        "source": "RGBDT500 official Train.zip",
        "seed": SEED,
        "pilot": pilot,
        "external_review": review,
        "external_review_sequences": review_sequences,
        "remaining_train_sequences": shuffled[120:],
        "notes": "All selected items are annotated frames from author Train.zip. No Query is present.",
    }


def parse_groundtruth(contents: bytes) -> dict[str, tuple[float, float, float, float]]:
    rows = csv.reader(contents.decode("utf-8-sig").splitlines())
    parsed = {}
    for row in rows:
        if len(row) != 5:
            raise ValueError(f"invalid groundtruth row {row}")
        parsed[row[0]] = tuple(map(float, row[1:]))
    if len(parsed) != 10:
        raise ValueError(f"expected 10 groundtruth rows, got {len(parsed)}")
    return parsed


def write_depth_visualization(raw_path: Path, visual_path: Path, overwrite: bool = False) -> None:
    """Fixed raw-value display mapping; raw units remain unknown."""
    if visual_path.exists() and not overwrite:
        return
    with Image.open(raw_path) as image:
        depth = np.asarray(image)
    if depth.dtype != np.uint16:
        raise ValueError(f"expected uint16 RGBDT depth: {raw_path} has {depth.dtype}")
    visual = np.zeros(depth.shape, dtype=np.uint8)
    valid = depth > 0
    clipped = np.minimum(depth[valid].astype(np.float32), DEPTH_RAW_MAX)
    visual[valid] = np.rint(255 - (clipped - 1) * 254 / (DEPTH_RAW_MAX - 1)).astype(np.uint8)
    visual_path.parent.mkdir(exist_ok=True)
    partial = visual_path.with_suffix(".png.part")
    Image.fromarray(visual, mode="L").convert("RGB").save(partial, format="PNG")
    partial.replace(visual_path)


def acquire_one(
    item: dict[str, str],
    subset: str,
    root: Path,
    index: dict[str, tuple[zipfile.ZipInfo, int]],
    read_source=read_entry,
) -> dict:
    sequence, frame = item["sequence"], item["frame"]
    target = root / subset / sequence
    target.mkdir(parents=True, exist_ok=True)
    annotation_path = target / "groundtruth.txt"
    if not annotation_path.exists():
        partial = annotation_path.with_suffix(".txt.part")
        partial.write_bytes(read_source(index[f"{sequence}/groundtruth.txt"]))
        partial.replace(annotation_path)
    groundtruth = parse_groundtruth(annotation_path.read_bytes())
    if frame not in groundtruth:
        raise ValueError(f"{sequence}/{frame} missing from groundtruth")
    paths = {}
    for modality in ("color", "depth", "infrared"):
        output = target / modality / frame
        output.parent.mkdir(exist_ok=True)
        if not output.exists():
            partial = output.with_suffix(".png.part")
            partial.write_bytes(read_source(index[f"{sequence}/{modality}/{frame}"]))
            partial.replace(output)
        paths[modality] = output
    depth_visual = target / "depth_visual" / frame
    write_depth_visualization(paths["depth"], depth_visual)
    with Image.open(paths["color"]) as rgb, Image.open(paths["depth"]) as depth, Image.open(paths["infrared"]) as ir:
        width, height = rgb.size
        depth_mode, depth_size = depth.mode, depth.size
        ir_mode, ir_size = ir.mode, ir.size
    x, y, w, h = groundtruth[frame]
    if not (0 <= x < x + w <= width and 0 <= y < y + h <= height):
        raise ValueError(f"invalid {sequence}/{frame} box {[x, y, w, h]} for {width}x{height}")
    if depth_size != (width, height) or ir_size != (width, height):
        raise ValueError(f"unaligned image sizes for {sequence}/{frame}")
    return {
        "id": f"rgbdt500_{sequence}_{Path(frame).stem}",
        "source": "rgbdt500",
        "split": subset,
        "sequence": sequence,
        "frame": frame,
        "rgb": paths["color"].relative_to(root).as_posix(),
        "depth": paths["depth"].relative_to(root).as_posix(),
        "depth_visual": depth_visual.relative_to(root).as_posix(),
        "infrared": paths["infrared"].relative_to(root).as_posix(),
        "source_paths": {modality: f"{sequence}/{modality}/{frame}" for modality in paths},
        "image_size": [width, height],
        "bbox_xywh_pixels": [x, y, w, h],
        "bbox_xyxy_normalized": [x / width, y / height, (x + w) / width, (y + h) / height],
        "depth_mode": depth_mode,
        "ir_mode": ir_mode,
        "depth_units": "unverified",
        "depth_visual_policy": "fixed_raw_1_to_19999_same_numeric_map_as_city_zero_invalid; no metric-unit claim",
        "query": None,
    }


def acquire_selected(args: argparse.Namespace, index: dict, count: int, read_source=read_entry) -> None:
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "acquisition_plan.json"
    if args.local_zip is not None and plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    else:
        plan = make_plan(index, root, read_source)
    if args.local_zip is None:
        if plan_path.exists():
            old = json.loads(plan_path.read_text(encoding="utf-8"))
            if old != plan:
                for subset in ("pilot", "external_review"):
                    if [item["sequence"] for item in old[subset]] != [item["sequence"] for item in plan[subset]]:
                        raise ValueError("existing acquisition sequence split differs from current source")
                    for original, revised in zip(old[subset], plan[subset], strict=True):
                        if original["frame"] == revised["frame"]:
                            continue
                        cached_gt = root / subset / original["sequence"] / "groundtruth.txt"
                        if not cached_gt.exists():
                            cached_gt = root / "metadata" / "groundtruth" / f"{original['sequence']}.txt"
                        if not cached_gt.exists():
                            raise ValueError(f"missing cached groundtruth for {original['sequence']}")
                        rows = parse_groundtruth(cached_gt.read_bytes())
                        _, _, width, height = rows[original["frame"]]
                        if width > 0 and height > 0:
                            raise ValueError("acquisition plan would replace a valid annotated frame")
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    elif not plan_path.exists():
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    selected = [
        (item, "pilot") for item in plan["pilot"][: args.pilot_groups]
    ] + [
        (item, "external_review") for item in plan["external_review"][: args.external_groups]
    ]
    records: dict[str, dict] = {}
    for subset in ("pilot", "external_review"):
        manifest = root / f"{subset}_manifest.jsonl"
        if manifest.exists():
            for line in manifest.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                records[row["id"]] = row

    def save_manifests() -> None:
        for subset in ("pilot", "external_review"):
            rows = sorted(
                (row for row in records.values() if row["split"] == subset),
                key=lambda row: (row["sequence"], row["frame"]),
            )
            manifest = root / f"{subset}_manifest.jsonl"
            partial = manifest.with_suffix(".jsonl.part")
            partial.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
            )
            partial.replace(manifest)

    progress_path = root / "acquisition_progress.jsonl"
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {
            pool.submit(acquire_one, item, subset, root, index, read_source): (subset, item)
            for item, subset in selected
            if f"rgbdt500_{item['sequence']}_{Path(item['frame']).stem}" not in records
        }
        for future in as_completed(pending):
            record = future.result()
            records[record["id"]] = record
            with progress_path.open("a", encoding="utf-8") as progress:
                progress.write(json.dumps(record, ensure_ascii=False) + "\n")
            save_manifests()
            print(f"acquired {record['split']} {record['sequence']}/{record['frame']}", flush=True)
    save_manifests()
    print(json.dumps({"indexed_archive_entries": count, "acquired": len(records), "output": str(root)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pilot-groups", type=int, default=0)
    parser.add_argument("--external-groups", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--local-zip", type=Path)
    args = parser.parse_args()
    if not (0 <= args.pilot_groups <= 100 and 0 <= args.external_groups <= 50):
        raise ValueError("pilot-groups must be 0..100 and external-groups 0..50")
    if args.local_zip is None:
        root = args.output_dir.resolve()
        root.mkdir(parents=True, exist_ok=True)
        index, count = load_index(root)
        acquire_selected(args, index, count)
    else:
        if args.local_zip.name.lower().endswith(".downloading"):
            raise ValueError("refusing incomplete .downloading ZIP")
        with zipfile.ZipFile(args.local_zip) as archive:
            index = {info.filename: (info, 0) for info in archive.infolist()}
            acquire_selected(args, index, len(index), lambda entry: archive.read(entry[0]))


if __name__ == "__main__":
    main()
