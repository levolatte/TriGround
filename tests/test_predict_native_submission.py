from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tools.evaluate_pretrained_grounder import build_user_content
from tools.predict_native_submission import (
    MAX_NEW_TOKENS,
    MAX_PIXELS,
    MIN_PIXELS,
    PROMPTS,
    build_run_config,
    image_paths,
    load_modal_images,
    main,
    validate_rows,
)
from tools.prepare_qwen3vl_native_sft import _rgb_prompt, _trimodal_prompt, render_depth


def _source() -> dict[str, dict[str, str]]:
    return {
        "000001_001": {
            "visible": "Images/visible/000001.png",
            "infrared": "Images/infrared/000001.png",
            "depth": "Images/depth/000001.png",
            "query": "A squatting person",
            "extra": "preserved",
        },
        "000001_002": {
            "visible": "Images/visible/000001.png",
            "infrared": "Images/infrared/000001.png",
            "depth": "Images/depth/000001.png",
            "query": "The white street light on the left",
            "extra": "also preserved",
        },
    }


def _row(sample_id: str, record: dict[str, str], data_root: Path,
         queries: Path, modality: str = "trimodal") -> dict:
    return {
        "id": sample_id,
        "query": record["query"],
        "image_paths": image_paths(record, data_root, queries),
        "prompt": PROMPTS[modality](record["query"]),
        "bbox": [0.1, 0.2, 0.8, 0.9],
        "parsed": True,
        "fallback": False,
        "raw_text": '{"bbox_2d":[100,200,800,900]}',
        "generated_tokens": 12,
        "latency_seconds": 1.25,
        "generation_cap_hit": False,
    }


def _prepared_run(tmp_path: Path, modality: str = "trimodal") -> tuple[Path, Path, dict]:
    queries = tmp_path / "queries.json"
    source = _source()
    queries.write_text(json.dumps(source), encoding="utf-8")
    output = tmp_path / "run"
    output.mkdir()
    config = build_run_config("local-model", tmp_path / "adapter", queries.resolve(),
                              tmp_path.resolve(), modality)
    (output / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    return queries, output, source


def test_prompts_match_native_sft_templates_and_image_positions():
    query = "A squatting person"
    assert PROMPTS["rgb"] is _rgb_prompt
    assert PROMPTS["trimodal"] is _trimodal_prompt
    assert _rgb_prompt(query) == (
        "<image>\nThe image is the RGB view. Locate the object described by this query: "
        "A squatting person\nReturn only JSON in this exact form with coordinates "
        'normalized to 0-1000: {"bbox_2d":[x1,y1,x2,y2]}'
    )
    assert _trimodal_prompt(query).startswith("<image>\n<image>\n<image>\n")
    images = ["RGB", "IR", "Depth"]
    content = build_user_content(_trimodal_prompt(query), images,
                                 prompt_has_image_placeholders=True)
    assert [part["image"] for part in content if part["type"] == "image"] == images
    assert "".join(part["text"] for part in content if part["type"] == "text") == (
        _trimodal_prompt(query).replace("<image>", "")
    )
    config = build_run_config("model", Path("adapter"), Path("queries"), Path("data"),
                              "trimodal")
    assert (config["min_pixels"], config["max_pixels"], config["max_new_tokens"]) == (
        MIN_PIXELS, MAX_PIXELS, MAX_NEW_TOKENS) == (200704, 602112, 128)
    assert config["modalities"] == ["visible", "infrared", "depth"]
    assert config["peft_autocast_adapter_dtype"] is False
    assert config["do_sample"] is False
    assert config["tf32"] is False


def test_live_depth_mapping_matches_pre_rendered_sft_pixels(tmp_path):
    depth = np.array([[0, 1, 999, 19999, 25000]], dtype=np.uint16)
    paths = {}
    for key in ("visible", "infrared", "depth"):
        paths[key] = str(tmp_path / f"{key}.png")
    Image.new("RGB", (5, 1), color=(10, 20, 30)).save(paths["visible"])
    Image.new("RGB", (5, 1), color=(40, 50, 60)).save(paths["infrared"])
    Image.fromarray(depth).save(paths["depth"])
    rendered = tmp_path / "depth_rgb.png"
    render_depth(Path(paths["depth"]), rendered)
    images = load_modal_images(paths, "trimodal")
    assert len(images) == 3
    assert images[0].getpixel((0, 0)) == (10, 20, 30)
    assert images[1].getpixel((0, 0)) == (40, 50, 60)
    with Image.open(rendered) as expected:
        np.testing.assert_array_equal(np.asarray(images[2]), np.asarray(expected.convert("RGB")))
    assert len(load_modal_images(paths, "rgb")) == 1


def test_official_rendered_depth_is_preserved_without_mm_conversion(tmp_path):
    path = tmp_path / "visualization.jpg"
    Image.fromarray(np.array([[[0, 0, 0], [80, 80, 80], [255, 255, 255]]],
                             dtype=np.uint8)).save(path)
    paths = {key: str(path) for key in ("visible", "infrared", "depth")}
    with Image.open(path) as original:
        expected = np.asarray(original).copy()
    actual = np.asarray(load_modal_images(paths, "trimodal")[2])
    np.testing.assert_array_equal(actual, expected)


def test_package_only_preserves_all_fields_and_zip_root_without_cuda(tmp_path, monkeypatch):
    queries, output, source = _prepared_run(tmp_path)
    rows = [_row(sample_id, record, tmp_path, queries) for sample_id, record in source.items()]
    (output / "predictions.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["predict_native_submission.py", "--queries", str(queries),
                                      "--data-root", str(tmp_path), "--output-dir", str(output),
                                      "--expected-query-count", "2", "--package-only"])
    main()
    predicted = json.loads((output / "predictions.json").read_text(encoding="utf-8"))
    assert list(predicted) == list(source)
    for sample_id, record in source.items():
        assert predicted[sample_id] == {**record, "bbox": [0.1, 0.2, 0.8, 0.9]}
    with zipfile.ZipFile(output / "submission.zip") as archive:
        assert archive.namelist() == ["predictions.json"]
        assert json.loads(archive.read("predictions.json")) == predicted
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["completed_queries"] == 2
    assert "acc_0.5" not in summary and "iou" not in summary


def test_limit_resume_stays_partial_and_package_only_rejects_missing_id(tmp_path, monkeypatch):
    queries, output, source = _prepared_run(tmp_path)
    first_id = next(iter(source))
    (output / "predictions.jsonl").write_text(
        json.dumps(_row(first_id, source[first_id], tmp_path, queries)) + "\n", encoding="utf-8")
    common = ["predict_native_submission.py", "--model", "local-model", "--adapter",
              str(tmp_path / "adapter"), "--modality", "trimodal", "--queries", str(queries),
              "--data-root", str(tmp_path), "--output-dir", str(output),
              "--expected-query-count", "2"]
    (tmp_path / "adapter").mkdir()
    (tmp_path / "adapter" / "adapter_config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", common + ["--limit", "1", "--resume"])
    main()
    assert not (output / "submission.zip").exists()
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["completed_queries"] == 1 and summary["total_queries"] == 2
    monkeypatch.setattr(sys, "argv", common + ["--package-only"])
    with pytest.raises(ValueError, match="incomplete predictions"):
        main()


def test_resume_rejects_changed_query_path_modality_and_raw_box(tmp_path, monkeypatch):
    queries, output, source = _prepared_run(tmp_path)
    sample_id = next(iter(source))
    row = _row(sample_id, source[sample_id], tmp_path, queries)
    assert list(validate_rows([row], source, tmp_path, queries, "trimodal")) == [sample_id]
    changed = json.loads(json.dumps(source))
    changed[sample_id]["query"] = "different query"
    with pytest.raises(ValueError, match="query mismatch"):
        validate_rows([row], changed, tmp_path, queries, "trimodal")
    changed = json.loads(json.dumps(source))
    changed[sample_id]["depth"] = "Images/depth/other.png"
    with pytest.raises(ValueError, match="image_paths mismatch"):
        validate_rows([row], changed, tmp_path, queries, "trimodal")
    with pytest.raises(ValueError, match="prompt mismatch"):
        validate_rows([row], source, tmp_path, queries, "rgb")
    broken = dict(row, bbox=[0.0, 0.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="bbox differs from raw text"):
        validate_rows([broken], source, tmp_path, queries, "trimodal")
    (output / "predictions.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["predict_native_submission.py", "--queries", str(queries),
                                      "--data-root", str(tmp_path), "--output-dir", str(output),
                                      "--expected-query-count", "2", "--package-only",
                                      "--modality", "rgb"])
    with pytest.raises(ValueError, match="run_config mismatch for modality"):
        main()


def test_only_unparseable_generated_box_uses_fixed_fallback(tmp_path):
    queries, _, source = _prepared_run(tmp_path)
    sample_id = next(iter(source))
    row = _row(sample_id, source[sample_id], tmp_path, queries)
    row.update(raw_text='{"bbox_2d":[900,200,100,800]}', bbox=[0.0, 0.0, 1.0, 1.0],
               parsed=False, fallback=True)
    assert list(validate_rows([row], source, tmp_path, queries, "trimodal")) == [sample_id]
    row["raw_text"] = '{"bbox_2d":[100,200,800,900]}'
    with pytest.raises(ValueError, match="raw text and parse status differ"):
        validate_rows([row], source, tmp_path, queries, "trimodal")
