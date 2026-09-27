from __future__ import annotations

import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _load_acquirer():
    path = Path(__file__).resolve().parents[1] / "tools" / "acquire_rgbdt500.py"
    spec = importlib.util.spec_from_file_location("acquire_rgbdt500_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plan_reselects_same_sequence_when_selected_bbox_is_invalid(tmp_path: Path) -> None:
    acquirer = _load_acquirer()
    sequences = [f"{i:03d}" for i in range(1, 401)]
    frames = [f"{i:08d}.png" for i in range(1, 11)]
    index = {}
    metadata = tmp_path / "metadata" / "groundtruth"
    metadata.mkdir(parents=True)

    for sequence in sequences:
        index[f"{sequence}/groundtruth.txt"] = (None, 0)
        for frame in frames:
            index[f"{sequence}/color/{frame}"] = (None, 0)
        (metadata / f"{sequence}.txt").write_text(
            "".join(f"{frame},0,0,1,1\n" for frame in frames),
            encoding="utf-8",
        )

    def forbid_network(_entry):
        raise AssertionError("offline plan regression must not read ZIP entries")

    acquirer.read_entry = forbid_network
    original = acquirer.make_plan(index, tmp_path)
    selected = next(item for item in original["pilot"] if item["sequence"] == "325")
    invalid_frame = selected["frame"]
    rows = [
        f"{frame},0,0,{0 if frame == invalid_frame else 1},{0 if frame == invalid_frame else 1}\n"
        for frame in frames
    ]
    (metadata / "325.txt").write_text("".join(rows), encoding="utf-8")

    repaired = acquirer.make_plan(index, tmp_path)
    replacement = next(item for item in repaired["pilot"] if item["sequence"] == "325")
    valid_alternatives = [frame for frame in frames if frame > invalid_frame]
    if not valid_alternatives:
        valid_alternatives = [frame for frame in frames if frame != invalid_frame]

    assert replacement["frame"] == valid_alternatives[0]
    assert [item["sequence"] for item in repaired["pilot"]] == [
        item["sequence"] for item in original["pilot"]
    ]
    assert repaired["external_review_sequences"] == original["external_review_sequences"]


def test_local_zip_uses_existing_plan_and_skips_completed_group(tmp_path: Path, monkeypatch) -> None:
    acquirer = _load_acquirer()
    frame = "00000001.png"
    root = tmp_path / "data"
    root.mkdir()
    plan = {"pilot": [{"sequence": "001", "frame": frame}], "external_review": []}
    plan_path = root / "acquisition_plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    def png(array: np.ndarray) -> bytes:
        output = io.BytesIO()
        Image.fromarray(array).save(output, format="PNG")
        return output.getvalue()

    contents = {
        "001/groundtruth.txt": "".join(
            f"{i:08d}.png,1,1,3,3\n" for i in range(1, 11)
        ).encode(),
        f"001/color/{frame}": png(np.full((6, 8, 3), 100, dtype=np.uint8)),
        f"001/depth/{frame}": png(np.full((6, 8), 500, dtype=np.uint16)),
        f"001/infrared/{frame}": png(np.full((6, 8), 70, dtype=np.uint8)),
    }
    local_zip = tmp_path / "Train.zip"
    with zipfile.ZipFile(local_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in contents.items():
            archive.writestr(name, data)

    def forbid_network(*_args):
        raise AssertionError("local ZIP path must not use HTTP")

    monkeypatch.setattr(acquirer, "http_range", forbid_network)
    monkeypatch.setattr(acquirer, "load_index", forbid_network)
    monkeypatch.setattr(
        sys, "argv", ["acquire_rgbdt500.py", "--output-dir", str(root), "--local-zip", str(local_zip), "--pilot-groups", "1"]
    )
    acquirer.main()
    assert json.loads(plan_path.read_text(encoding="utf-8")) == plan
    for name, data in contents.items():
        relative = name.split("/", 1)[1]
        output = root / "pilot" / "001" / relative
        assert output.read_bytes() == data
    records = (root / "pilot_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(records) == 1
    assert json.loads(records[0])["bbox_xywh_pixels"] == [1.0, 1.0, 3.0, 3.0]

    def forbid_second_read(*_args):
        raise AssertionError("completed group must not be re-extracted")

    monkeypatch.setattr(zipfile.ZipFile, "read", forbid_second_read)
    acquirer.main()
    assert len((root / "acquisition_progress.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_local_zip_rejects_downloading_file_before_open(tmp_path: Path, monkeypatch) -> None:
    acquirer = _load_acquirer()
    monkeypatch.setattr(
        sys, "argv", ["acquire_rgbdt500.py", "--output-dir", str(tmp_path), "--local-zip", str(tmp_path / "Train.zip.downloading")]
    )
    with pytest.raises(ValueError, match="downloading"):
        acquirer.main()
