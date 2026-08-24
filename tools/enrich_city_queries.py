from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path


ORDINAL_QUERY = re.compile(r"^The \d+(?:st|nd|rd|th) (.+) from the left$")
EDGE_QUERY = re.compile(r"^The (?:leftmost|rightmost) (.+)$")
PLAIN_QUERY = re.compile(r"^The (.+)$")


def object_name(query: str) -> str:
    for pattern in (ORDINAL_QUERY, EDGE_QUERY, PLAIN_QUERY):
        match = pattern.match(query)
        if match:
            return match.group(1)
    raise ValueError(f"无法从弱 Query 提取类别: {query!r}")


def center(box: list[float]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def region_phrase(box: list[float]) -> str:
    x, y = center(box)
    horizontal = "left" if x < 1 / 3 else "right" if x > 2 / 3 else "center"
    vertical = "upper" if y < 1 / 3 else "lower" if y > 2 / 3 else "middle"
    if horizontal == "center" and vertical == "middle":
        return "near the center of the image"
    if horizontal == "center":
        return f"in the {vertical} part of the image"
    if vertical == "middle":
        return f"on the {horizontal} side of the image"
    return f"in the {vertical}-{horizontal} part of the image"


def nearest_other(target: dict, records: list[dict]) -> dict | None:
    tx, ty = center(target["bbox"])
    candidates = [item for item in records if item["name"] != target["name"]]
    if not candidates:
        return None
    return min(candidates, key=lambda item: math.dist((tx, ty), center(item["bbox"])))


def relation_phrase(target: dict, reference: dict) -> str:
    tx, ty = center(target["bbox"])
    rx, ry = center(reference["bbox"])
    dx, dy = tx - rx, ty - ry
    if abs(dx) >= abs(dy):
        relation = "to the right of" if dx > 0 else "to the left of"
    else:
        relation = "below" if dy > 0 else "above"
    return f"{relation} the nearby {reference['name']}"


def base_identity(target: dict, same_class: list[dict]) -> str:
    name = target["name"]
    count = len(same_class)
    if count == 1:
        return f"The {name}"
    ordered = sorted(same_class, key=lambda item: (center(item["bbox"])[0], center(item["bbox"])[1]))
    rank = ordered.index(target)
    if rank == 0:
        return f"The leftmost {name}"
    if rank == count - 1:
        return f"The rightmost {name}"
    # Keep an exact ordinal even in crowded scenes: geometry-only descriptions
    # must remain unique and cannot invent appearance details.
    number = rank + 1
    suffix = "th" if 10 <= number % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"The {number}{suffix} {name} from the left"


def enrich_scene(scene_records: list[tuple[str, dict]]) -> dict[str, dict]:
    parsed = []
    for sample_id, record in scene_records:
        parsed.append({"sample_id": sample_id, "record": record, "bbox": record["bbox"], "name": object_name(record["query"])})
    by_class: dict[str, list[dict]] = defaultdict(list)
    for item in parsed:
        by_class[item["name"]].append(item)

    output = {}
    for item in parsed:
        identity = base_identity(item, by_class[item["name"]])
        reference = nearest_other(item, parsed)
        # Competition queries average about ten words. Add one verifiable clue:
        # a relation for a sole class instance, otherwise an image-region clue.
        parts = [identity]
        if len(by_class[item["name"]]) == 1 and reference is not None:
            parts.append(relation_phrase(item, reference))
        else:
            parts.append(region_phrase(item["bbox"]))
        query = ", ".join(parts)
        output[item["sample_id"]] = {
            "visible": item["record"]["visible"],
            "infrared": item["record"]["infrared"],
            "depth": item["record"]["depth"],
            "query": query,
            "bbox": item["bbox"],
        }
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich city weak queries in competition style")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = json.loads(args.source.read_text(encoding="utf-8-sig"))
    scenes: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for sample_id, record in records.items():
        scenes[record["visible"]].append((sample_id, record))
    output = {}
    for scene_records in scenes.values():
        output.update(enrich_scene(scene_records))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"scenes": len(scenes), "queries": len(output), "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
