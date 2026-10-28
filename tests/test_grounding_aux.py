import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from tools.prepare_grounding_aux import build, read_pairs


def _rows(tmp_path: Path) -> list[dict]:
    Image.new("RGB", (8, 8), (10, 20, 30)).save(tmp_path / "rgb.png")
    Image.new("RGB", (8, 8), (50, 50, 50)).save(tmp_path / "infrared.png")
    Image.new("RGB", (8, 8), (80, 80, 80)).save(tmp_path / "depth.png")

    def pair(source: str, category: str, index: int, choice: str, split: str = "train") -> dict:
        return {
            "id": f"{category}-{index}-{split}", "source": source,
            "scene_id": f"{source}-{category}-{index}-{split}",
            "image_id": f"image-{index}", "query": "the full visible target",
            "images": (["rgb.png", "infrared.png", "depth.png"] if source == "rgbdt500"
                       else ["rgb.png", "depth.png"]),
            "modalities": (["rgb", "infrared", "depth"] if source == "rgbdt500" else ["rgb", "depth"]),
            "depth_policy": "sensor_linear_20000", "bbox": [0.1, 0.2, 0.6, 0.8],
            "candidates": [[0.1, 0.2, 0.6, 0.8], [0.3, 0.4, 0.8, 0.9]],
            "choice": choice, "reason": "short_reason", "category": category,
            "split": split, "evidence": "SECRET_EVIDENCE",
            "review_status": "accepted", "reviewer": "SECRET_REVIEWER",
            "review_note": "SECRET_NOTE",
        }

    return [pair("rgbdt500", "wrong_object", 0, "A"),
            pair("rgbdt500", "wrong_object", 1, "B"),
            pair("roborefit", "wrong_object", 0, "both"),
            pair("rgbdt500", "extent", 0, "neither"),
            pair("rgbdt500", "extent", 1, "unknown"),
            pair("rgbdt500", "extent", 2, "A", "eval")]


def _manifest(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "pairs.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_paired_schedule_quota_swap_and_private_prompt(tmp_path: Path):
    rows = _rows(tmp_path)
    source = _manifest(tmp_path, rows)
    output = tmp_path / "out"
    summary = build([source], output, image_root=tmp_path)
    a, b, audit = (_read(output / f"{name}.jsonl") for name in ("A", "B", "paired_slots"))
    by_id = {f"{row['source']}:{row['id']}": row for row in rows}

    assert len(a) == len(b) == len(audit) == 1920
    assert summary["arm_a"] == {"rows": 1920, "box": 1920, "aux": 0}
    assert summary["arm_b"] == {"rows": 1920, "box": 1440, "aux": 480}
    assert summary["eval_pairs"] == 1
    assert summary["pools"]["rgbdt500/wrong_object"]["selected_distinct_pairs"] == 2
    assert summary["pools"]["roborefit/wrong_object"]["repeated_presentations"] == 159
    for start in range(0, 1920, 4):
        assert len({row["pair_id"] for row in a[start:start + 4]}) == 1
        assert [row["channel"] for row in b[start:start + 4]] == ["box", "box", "box", "aux"]
    for start in range(0, 1920, 8):
        assert sum(row["channel"] == "aux" for row in b[start:start + 8]) == 2
    for left, right, record in zip(a, b, audit):
        assert (left["slot_id"], left["pair_id"], left["images"], left["query"]) == (
            right["slot_id"], right["pair_id"], right["images"], right["query"])
        assert left["slot_id"] == record["slot_id"]
        if right["channel"] == "box":
            assert left["messages"] == right["messages"]
        else:
            original = by_id[right["pair_id"]]
            answer = json.loads(right["messages"][1]["content"])
            expected = original["choice"]
            if record["swapped"] and expected in {"A", "B"}:
                expected = "B" if expected == "A" else "A"
            assert answer == {"choice": expected, "reason": original["reason"]}
            candidates = original["candidates"][::-1] if record["swapped"] else original["candidates"]
            prompt = right["messages"][0]["content"]
            assert f"A: {[round(x * 1000) for x in candidates[0]]}" in prompt
            assert f"B: {[round(x * 1000) for x in candidates[1]]}" in prompt
            assert "complete visible target" in prompt
    assert sum(row["swapped"] for row in audit) == 240
    prompts = "\n".join(row["messages"][0]["content"] for row in a + b)
    for secret in ("SECRET_EVIDENCE", "SECRET_REVIEWER", "SECRET_NOTE", "rgbdt500", "roborefit"):
        assert secret not in prompts
    evaluation = _read(output / "aux_eval.jsonl")
    assert len(evaluation) == 1 and len(evaluation[0]["messages"]) == 1
    assert evaluation[0]["category"] == "extent"
    assert evaluation[0]["expected"] == {"choice": "A", "reason": "short_reason"}


def test_rejects_scene_leakage_and_raw_16bit_depth(tmp_path: Path):
    rows = _rows(tmp_path)
    rows[-1]["scene_id"] = rows[0]["scene_id"]
    path = _manifest(tmp_path, rows)
    with pytest.raises(ValueError, match="scene overlap"):
        read_pairs([path], tmp_path)

    rows[-1]["scene_id"] = "eval-only"
    Image.fromarray(np.full((8, 8), 500, dtype=np.uint16)).save(tmp_path / "raw.png")
    rows[0]["images"][-1] = "raw.png"
    path = _manifest(tmp_path, rows)
    with pytest.raises(ValueError, match="8-bit RGB"):
        read_pairs([path], tmp_path)


def test_training_requires_accepted_review_and_quantized_box(tmp_path: Path):
    rows = _rows(tmp_path)
    rows[0]["review_status"] = "draft"
    with pytest.raises(ValueError, match="accepted"):
        read_pairs([_manifest(tmp_path, rows)], tmp_path)
    rows[0]["review_status"] = "accepted"
    rows[0]["candidates"][0] = [0.0001, 0.1, 0.0002, 0.2]
    with pytest.raises(ValueError, match="collapsed"):
        read_pairs([_manifest(tmp_path, rows)], tmp_path)
