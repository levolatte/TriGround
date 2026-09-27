"""Extract a scene-balanced RoboRefIt train subset from the author archive."""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


def normalized_path(raw: str) -> str:
    path = str(raw).replace("\\", "/")
    if not path.startswith("final_dataset/train/") or ".." in Path(path).parts:
        raise ValueError(f"unexpected train path {raw!r}")
    return path


def valid_bbox(box: list, width: int, height: int) -> bool:
    return (
        len(box) == 4
        and all(isinstance(v, (float, int)) and math.isfinite(v) for v in box)
        and 0 <= box[0] < box[2] <= width
        and 0 <= box[1] < box[3] <= height
    )


def extract_pairs(archive: Path, root: Path, groups: list[tuple[str, list[tuple[int, dict]]]]) -> None:
    required = []
    for _, rows in groups:
        for key in ("rgb_path", "depth_path"):
            relative = normalized_path(rows[0][1][key])
            if not (root / relative).is_file():
                required.append(relative)
    if not required:
        return
    file_list = root / "extract_files.txt"
    file_list.write_text("\n".join(dict.fromkeys(required)) + "\n", encoding="utf-8")
    subprocess.run(["tar", "-xf", str(archive), "-C", str(root), "-T", str(file_list)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows = json.loads(args.source_json.read_text(encoding="utf-8-sig"))
    by_scene: dict[str, dict[str, list[tuple[int, dict]]]] = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        rgb = normalized_path(row["rgb_path"])
        normalized_path(row["depth_path"])
        by_scene[str(row["scene"])][rgb].append((index, row))
    rng = random.Random(args.seed)
    scenes = sorted(by_scene)
    groups = {}
    for scene in scenes:
        candidates = list(by_scene[scene].items())
        rng.shuffle(candidates)
        for _, image_rows in candidates:
            rng.shuffle(image_rows)
        groups[scene] = candidates
    quotas = dict.fromkeys(scenes, 0)
    for _ in range(args.count):
        eligible = [scene for scene in scenes if quotas[scene] < len(groups[scene])]
        if not eligible:
            raise ValueError("not enough unique train images")
        scene = min(eligible, key=lambda name: (quotas[name], name))
        quotas[scene] += 1

    selected: list[tuple[str, int, dict]] = []
    rejects = []
    positions = dict.fromkeys(scenes, 0)
    extract_pairs(args.archive, root, [group for scene in scenes for group in groups[scene][: quotas[scene]]])
    for scene in scenes:
        while sum(s == scene for s, _, _ in selected) < quotas[scene]:
            pos = positions[scene]
            if pos >= len(groups[scene]):
                raise ValueError(f"insufficient valid train images for {scene}")
            positions[scene] += 1
            image_group = groups[scene][pos]
            if pos >= quotas[scene]:
                extract_pairs(args.archive, root, [image_group])
            _, image_rows = image_group
            rgb = root / normalized_path(image_rows[0][1]["rgb_path"])
            depth = root / normalized_path(image_rows[0][1]["depth_path"])
            with Image.open(rgb) as image:
                width, height = image.size
            with Image.open(depth) as image:
                if image.size != (width, height):
                    raise ValueError(f"RGB/depth size mismatch: {rgb}, {depth}")
            valid = [(index, row) for index, row in image_rows if valid_bbox(row["bbox"], width, height) and str(row.get("text", "")).strip()]
            if not valid:
                rejects.append({"scene": scene, "rgb": rgb.relative_to(root).as_posix(), "reason": "no valid original bbox/query"})
                continue
            selected.append((scene, *valid[0]))

    # Preserve the author's original row schema for the existing converter.
    selected.sort(key=lambda item: item[1])
    subset_json = root / "final_dataset" / "train" / "roborefit_train.json"
    subset_json.parent.mkdir(parents=True, exist_ok=True)
    subset_json.write_text(json.dumps([row for _, _, row in selected], ensure_ascii=False), encoding="utf-8")
    index_path = root / "selection_index.jsonl"
    index_path.write_text(
        "".join(json.dumps({"subset_index": i, "original_index": original, "scene": scene, "rgb": normalized_path(row["rgb_path"]), "depth": normalized_path(row["depth_path"])}, ensure_ascii=False) + "\n" for i, (scene, original, row) in enumerate(selected)),
        encoding="utf-8",
    )
    report = {
        "author_train_rows": len(rows),
        "author_train_unique_images": len({normalized_path(row["rgb_path"]) for row in rows}),
        "selected_rows": len(selected),
        "selected_unique_images": len({normalized_path(row["rgb_path"]) for _, _, row in selected}),
        "seed": args.seed,
        "scene_quota": quotas,
        "selected_by_scene": dict(Counter(scene for scene, _, _ in selected)),
        "rejected_groups": rejects,
        "selection_index": str(index_path),
        "subset_json": str(subset_json),
    }
    (root / "selection_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
