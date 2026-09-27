from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

from tools.prepare_gu_same_day_pilot import build


def _prompt(query: str, modality: str) -> str:
    return (
        "<image>\n<image>\n<image>\nViews are in this order: rgb, infrared, depth. "
        f"Locate the target described by: {query}\nReturn its box in the {modality} image."
    )


def _sources(tmp_path: Path) -> tuple[Path, Path, Path, Path, list[dict], list[dict]]:
    released = []
    metadata = []
    for index in range(98):
        source_id = f"rgbdt_{index:03d}"
        query = f"the distinct object {index}"
        images = [f"remote/images/{source_id}_{modality}.png" for modality in ("rgb", "ir", "depth")]
        modalities = ["rgb"]
        if index < 58:
            modalities.append("infrared")
        if index < 25:
            modalities.append("depth")
        for modality in modalities:
            sample_id = f"{source_id}::{modality}"
            released.append({
                "id": sample_id,
                "image": images,
                "conversations": [
                    {"from": "human", "value": _prompt(query, modality)},
                    {"from": "gpt", "value": json.dumps({"bbox_2d": [index, 20, 700, 900]})},
                ],
            })
            metadata.append({
                "id": sample_id,
                "source": "rgbdt",
                "source_id": source_id,
                "group_id": f"rgbdt_group_{index % 7}",
                "task": "bbox",
                "target_modality": modality,
                "review_status": "human_accepted",
            })

    city = []
    for index in range(24):
        group = f"frame_{index // 3:03d}"
        query = f"the City object {index}"
        city.append({
            "id": f"city_{index:03d}",
            "split": "train",
            "class_name": "object",
            "image": [f"visible/{group}.png", f"infrared/{group}.png", f"depth/{group}.png"],
            "conversations": [
                {"from": "human", "value": _prompt(query, "rgb")},
                {"from": "gpt", "value": json.dumps({"bbox_2d": [1, 2, 300, 400]})},
            ],
        })

    released_path = tmp_path / "released.jsonl"
    metadata_path = tmp_path / "metadata.jsonl"
    city_path = tmp_path / "city.json"
    released_path.write_text("".join(json.dumps(row) + "\n" for row in released), encoding="utf-8")
    metadata_path.write_text("".join(json.dumps(row) + "\n" for row in metadata), encoding="utf-8")
    city_path.write_text(json.dumps(city), encoding="utf-8")
    return released_path, metadata_path, city_path, tmp_path / "city_root", released, city


def _args(paths: tuple, output_dir: Path, steps: int) -> argparse.Namespace:
    released_path, metadata_path, city_path, city_root = paths[:4]
    return argparse.Namespace(
        released_jsonl=released_path,
        metadata_jsonl=metadata_path,
        city_native=city_path,
        city_root=city_root,
        output_dir=output_dir,
        seed=2026,
        steps=steps,
    )


def test_200_step_pairing_preserves_identity_paths_and_query(tmp_path: Path) -> None:
    paths = _sources(tmp_path)
    released_path, metadata_path, city_path = paths[:3]
    released_bytes = released_path.read_bytes()
    metadata_bytes = metadata_path.read_bytes()
    city_bytes = city_path.read_bytes()
    original_rows = {row["id"]: row for row in paths[4]}
    output_dir = tmp_path / "out200"

    report = build(_args(paths, output_dir, 200))
    g_rows = json.loads((output_dir / "g_train.json").read_text(encoding="utf-8"))
    u_rows = json.loads((output_dir / "u_train.json").read_text(encoding="utf-8"))
    schedule = [json.loads(line) for line in (output_dir / "paired_schedule.jsonl").read_text(encoding="utf-8").splitlines()]
    metadata_out = [json.loads(line) for line in (output_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines()]

    assert report["presentations"] == 1600
    assert report["city_presentations"] == 1200
    assert report["new_presentations"] == 400
    assert report["new_presentations_by_output_modality"] == {"depth": 55, "infrared": 128, "rgb": 217}
    assert report["micro_batch_size"] == 1 and report["gradient_accumulation_steps"] == 8
    assert report["pressure16"]["source_counts"] == {"city": 2, "rgbdt": 6}
    assert report["pressure16"]["new_task_output_modality_counts"] == {
        "rgb": 2, "infrared": 2, "depth": 2,
    }
    assert len(g_rows) == len(u_rows) == len(schedule) == len(metadata_out) == 1600
    assert len({row["id"] for row in g_rows}) == len(g_rows)
    assert len({row["id"] for row in u_rows}) == len(u_rows)
    assert [row["id"] for row in g_rows] == [row["id"] for row in u_rows]
    assert [row["ordinal"] for row in schedule] == list(range(1, 1601))
    for start in range(0, len(schedule), 8):
        assert Counter(row["source"] for row in schedule[start:start + 8]) == {"city": 6, "rgbdt": 2}

    # Each paired presentation keeps the original source image paths and Query.
    for g_row, u_row, schedule_row in zip(g_rows, u_rows, schedule):
        assert g_row["id"] == u_row["id"] == schedule_row["presentation_id"]
        source_id = schedule_row["source_task_id"]
        if schedule_row["source"] == "city":
            city_source = next(row for row in paths[5] if row["id"] == source_id)
            expected_images = [str((paths[3] / value).resolve()) for value in city_source["image"]]
            assert g_row["image"] == u_row["image"] == expected_images
            assert g_row["conversations"] == u_row["conversations"] == city_source["conversations"]
        else:
            source_row = original_rows[source_id]
            assert g_row["image"] == u_row["image"] == source_row["image"]
            query = schedule_row["query"]
            assert query in source_row["conversations"][0]["value"]
            if schedule_row["output_modality"] == "rgb":
                assert g_row["conversations"] == u_row["conversations"] == source_row["conversations"]
            else:
                rgb_source = next(
                    row for row in paths[4]
                    if row["id"] == source_id.rsplit("::", 1)[0] + "::rgb"
                )
                assert u_row["conversations"] == source_row["conversations"]
                assert g_row["conversations"] == rgb_source["conversations"]

    pressure = json.loads((output_dir / "pressure16.json").read_text(encoding="utf-8"))
    pressure_city = json.loads((output_dir / "pressure2city.json").read_text(encoding="utf-8"))
    assert len(pressure) == 16 and len(pressure_city) == 2
    assert [row["id"] for row in pressure_city] == [row["id"] for row in pressure[:2]]
    assert pressure[0]["image"][0] != pressure[1]["image"][0]
    assert len({row["id"] for row in pressure}) == 16
    assert released_path.read_bytes() == released_bytes
    assert metadata_path.read_bytes() == metadata_bytes
    assert city_path.read_bytes() == city_bytes


def test_600_steps_repeat_three_200_step_quotas(tmp_path: Path) -> None:
    paths = _sources(tmp_path)
    output_dir = tmp_path / "out600"
    report = build(_args(paths, output_dir, 600))
    schedule = [json.loads(line) for line in (output_dir / "paired_schedule.jsonl").read_text(encoding="utf-8").splitlines()]

    assert report["presentations"] == 4800
    assert report["city_presentations"] == 3600
    assert report["new_presentations"] == 1200
    assert report["new_presentations_by_output_modality"] == {"depth": 165, "infrared": 384, "rgb": 651}
    assert len({row["presentation_id"] for row in schedule}) == 4800
    for start in range(0, len(schedule), 8):
        assert Counter(row["source"] for row in schedule[start:start + 8]) == {"city": 6, "rgbdt": 2}


def test_released_source_must_have_human_acceptance(tmp_path: Path) -> None:
    paths = _sources(tmp_path)
    metadata_path = paths[1]
    metadata = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()]
    metadata[0]["review_status"] = "pending_human"
    metadata_path.write_text("".join(json.dumps(row) + "\n" for row in metadata), encoding="utf-8")

    try:
        build(_args(paths, tmp_path / "must_not_write", 200))
    except ValueError as exc:
        assert "human review" in str(exc)
    else:
        raise AssertionError("pending released sample was accepted")
