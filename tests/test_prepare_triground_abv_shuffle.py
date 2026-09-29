"""Fixed cross-scene donor matching on tiny local diagnostic images."""

import json
from pathlib import Path

from PIL import Image
import pytest

from tools.prepare_triground_abv_shuffle import prepare


def _image(path: Path, size=(8, 8)) -> str:
    Image.new("RGB", size, (32, 64, 96)).save(path)
    return str(path)


def _row(sample_id, modalities, images, *, placeholder=False):
    prompt = ("<image>\n" * len(images)
              + "These are views of the same scene in this order: "
              + ", ".join("RGB" if name == "rgb" else name for name in modalities)
              + ". Locate the object described by this query: find the car.\n"
              + 'Return only JSON: {"bbox_2d":[x1,y1,x2,y2]}')
    return {"id": sample_id, "image": images,
            "source": "fixture", "scene_id": sample_id,
            "missing_modalities_actual": ["infrared"] if placeholder else [],
            "intervention_applied": placeholder,
            "conversations": [{"from": "human", "value": prompt},
                              {"from": "gpt", "value": '{"bbox_2d":[1,2,3,4]}'}]}


def _fixture(tmp_path):
    rows = [
        _row("ir-a", ("rgb", "infrared"),
             [_image(tmp_path / "a_rgb.png"), _image(tmp_path / "a_ir.png")]),
        _row("ir-b", ("rgb", "infrared"),
             [_image(tmp_path / "b_rgb.png"), _image(tmp_path / "b_ir.png")]),
        _row("depth-c", ("rgb", "depth"),
             [_image(tmp_path / "c_rgb.png"), _image(tmp_path / "c_depth.png")]),
        _row("tri-d", ("rgb", "infrared", "depth"),
             [_image(tmp_path / "d_rgb.png"), _image(tmp_path / "d_ir.png"),
              _image(tmp_path / "d_depth.png")]),
        _row("ir-large", ("rgb", "infrared"),
             [_image(tmp_path / "e_rgb.png", (16, 16)),
              _image(tmp_path / "e_ir.png", (16, 16))]),
        _row("placeholder", ("rgb", "infrared"),
             [_image(tmp_path / "f_rgb.png"), _image(tmp_path / "f_ir.png")],
             placeholder=True),
    ]
    normal = tmp_path / "normal.json"
    normal.write_text(json.dumps(rows), encoding="utf-8")
    scene = tmp_path / "scene_map.json"
    scene.write_text(json.dumps({row["id"]: f"scene-{row['id']}" for row in rows}),
                     encoding="utf-8")
    return normal, scene, rows


def test_shuffle_only_replaces_true_modality_slot_and_records_eligible_ids(tmp_path):
    normal, scene, rows = _fixture(tmp_path)
    out = tmp_path / "shuffle"
    summary = prepare(normal, scene, out, seed=17)
    source = {row["id"]: row for row in rows}
    for prefix, target in (("ir", "infrared"), ("depth", "depth")):
        manifest = json.loads((out / f"{prefix}_shuffle.json").read_text())
        donor_map = json.loads((out / f"{prefix}_donors.json").read_text())
        eligible = donor_map["eligible_recipient_ids"]
        assert [row["id"] for row in manifest] == eligible == summary["diagnostics"][prefix]["eligible_recipient_ids"]
        assert "exactly eligible_recipient_ids" in donor_map["comparison_rule"]
        for row in manifest:
            original = source[row["id"]]
            assignment = donor_map["assignments"][row["id"]]
            assert row["conversations"] == original["conversations"]
            assert row["image"][0] == original["image"][0]
            assert len(row["image"]) == len(original["image"])
            assert assignment["recipient_scene"] != assignment["donor_scene"]
            assert assignment["same_size"]
            for index, path in enumerate(row["image"]):
                if index == assignment["slot"]:
                    assert path == assignment["donor_path"] != original["image"][index]
                else:
                    assert path == original["image"][index]
        assert target == donor_map["modality"]
    ir_donors = json.loads((out / "ir_donors.json").read_text())
    depth_donors = json.loads((out / "depth_donors.json").read_text())
    assert ir_donors["skipped"]["depth-c"]["reason"] == "modality_not_present"
    assert ir_donors["skipped"]["ir-large"]["reason"] == "no_cross_scene_same_size_donor"
    assert ir_donors["skipped"]["placeholder"]["reason"] == "not_a_real_normal_modality"
    assert depth_donors["skipped"]["ir-a"]["reason"] == "modality_not_present"
    assert "full normal denominator is invalid" in summary["comparison_rule"]


def test_seed_is_fixed_and_size_mismatch_is_explicit(tmp_path):
    normal, scene, _ = _fixture(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    prepare(normal, scene, first, seed=2026, allow_size_mismatch=True)
    prepare(normal, scene, second, seed=2026, allow_size_mismatch=True)
    assert (first / "ir_donors.json").read_bytes() == (second / "ir_donors.json").read_bytes()
    donor_map = json.loads((first / "ir_donors.json").read_text())
    assert donor_map["assignments"]["ir-large"]["same_size"] is False
    assert donor_map["assignments"]["ir-large"]["recipient_size"] == [16, 16]
    assert donor_map["assignments"]["ir-large"]["donor_size"] == [8, 8]


def test_shuffle_refuses_existing_output_and_bad_scene_map(tmp_path):
    normal, scene, _ = _fixture(tmp_path)
    output = tmp_path / "shuffle"
    output.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        prepare(normal, scene, output)
    scene.write_text(json.dumps({"ir-a": "only-one"}), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly match"):
        prepare(normal, scene, tmp_path / "new")
