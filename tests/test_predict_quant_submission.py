from copy import deepcopy
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import pytest

from tools.predict_quant_submission import build_parser, load_queries, main, package_predictions, prepare_dataset
from tools.prepare_qwen3vl_native_sft import depth_mm_to_grayscale_rgb, native_prompt


def test_prepare_uses_training_prompt_and_depth_mapping_without_labels(tmp_path):
    Image.new("RGB", (4, 3)).save(tmp_path / "rgb.png")
    Image.new("RGB", (4, 3)).save(tmp_path / "ir.png")
    depth = np.array([[0, 1, 20000, 65535]] * 3, dtype=np.uint16)
    Image.fromarray(depth).save(tmp_path / "depth.png")
    record = {"visible": "rgb.png", "infrared": "ir.png", "depth": "depth.png", "query": "target"}
    source = {"b": record, "a": {**record, "query": "second", "extra": {"keep": 1}}}
    before = deepcopy(source)
    rows = prepare_dataset(source, tmp_path, tmp_path / "out")
    assert [row["id"] for row in rows] == ["b", "a"]
    assert source == before
    assert rows[0]["images"][2] == rows[1]["images"][2]
    assert rows[0]["messages"] == [{"role": "user", "content": native_prompt(
        "target", ("rgb", "infrared", "depth"), "millimeter")}]
    with Image.open(rows[0]["images"][2]) as rendered:
        np.testing.assert_array_equal(np.asarray(rendered), depth_mm_to_grayscale_rgb(depth))
    with Image.open(tmp_path / "depth.png") as original:
        np.testing.assert_array_equal(np.asarray(original), depth)
    assert "solution" not in rows[0]


def prepare_resume_run(tmp_path):
    for name in ("rgb", "ir", "depth"):
        Image.new("RGB", (4, 3)).save(tmp_path / f"{name}.png")
    record = {"visible": "rgb.png", "infrared": "ir.png", "depth": "depth.png", "query": "q"}
    source = {key: {**record, "query": key} for key in ("b", "a", "c")}
    queries = tmp_path / "queries.json"
    queries.write_text(json.dumps(source), encoding="utf-8")
    out = tmp_path / "out"
    argv = ["--queries", str(queries), "--data-root", str(tmp_path), "--output-dir", str(out),
            "--model", "model", "--expected-query-count", "3", "--hf-max-batch-size", "2"]
    main([*argv, "--prepare-only"])
    return argv, out


def test_completed_resume_packages_all_rows_without_repreparing_images(tmp_path, monkeypatch):
    import tools.predict_quant_submission as predictor

    argv, out = prepare_resume_run(tmp_path)
    rows = [{"id": key, "samples": [{"text": '{"bbox_2d":[100,100,500,500]}',
                                     "finish_reason": "stop"}]} for key in ("b", "a", "c")]
    raw = out / "predictions_raw.jsonl"
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    original = raw.read_bytes()
    monkeypatch.setattr(predictor, "prepare_dataset", lambda *args: pytest.fail("reuse prepared dataset"))
    result = main([*argv, "--resume", "--hf-max-batch-size", "8"])
    assert result["completed_queries"] == 3
    assert raw.read_bytes() == original
    assert list(json.loads((out / "predictions.json").read_text(encoding="utf-8"))) == ["b", "a", "c"]
    config = json.loads((out / "run_config.json").read_text(encoding="utf-8"))
    assert config["hf_max_batch_size"] == 2
    assert config["resume_history"] == [{"completed_queries": 3, "hf_max_batch_size": 8}]


def test_resume_rejects_changed_generation_protocol_before_writing(tmp_path):
    argv, out = prepare_resume_run(tmp_path)
    (out / "predictions_raw.jsonl").write_text("", encoding="utf-8")
    config = (out / "run_config.json").read_bytes()
    with pytest.raises(ValueError, match="resume configuration differs: max_tokens"):
        main([*argv, "--resume", "--max-tokens", "32"])
    assert (out / "run_config.json").read_bytes() == config


@pytest.mark.parametrize("value", ["0", "-1"])
def test_submission_rejects_nonpositive_batch_size(value):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--queries", "q", "--data-root", "d", "--output-dir", "o",
                                   "--hf-max-batch-size", value])


def test_official_count_and_missing_images_fail_at_boundary(tmp_path):
    path = tmp_path / "queries.json"
    source = {"a": {"visible": "missing.png", "infrared": "i.png", "depth": "d.png", "query": "q"}}
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="5691"):
        load_queries(path, 5691)
    with pytest.raises(FileNotFoundError, match="a:"):
        prepare_dataset(load_queries(path, 1), tmp_path, tmp_path / "out")


def test_already_rendered_rgb_depth_keeps_original_pixels_and_city_prompt(tmp_path):
    for name in ("rgb", "ir"):
        Image.new("RGB", (4, 3)).save(tmp_path / f"{name}.png")
    pixels = np.array([[[0, 0, 0], [42, 42, 42], [255, 255, 255], [1, 2, 3]]] * 3, dtype=np.uint8)
    Image.fromarray(pixels).save(tmp_path / "depth.png")
    source = {"a": {"visible": "rgb.png", "infrared": "ir.png", "depth": "depth.png", "query": "q"}}
    row = prepare_dataset(source, tmp_path, tmp_path / "out")[0]
    assert row["depth_policy"] == "visual"
    assert row["images"][2] == str((tmp_path / "depth.png").resolve())
    with Image.open(row["images"][2]) as image:
        np.testing.assert_array_equal(np.asarray(image), pixels)
    assert row["messages"][0]["content"] == native_prompt("q", ("rgb", "infrared", "depth"), "millimeter")


def test_unsupported_depth_encoding_fails_fast(tmp_path):
    for name in ("rgb", "ir"):
        Image.new("RGB", (4, 3)).save(tmp_path / f"{name}.png")
    Image.new("L", (4, 3)).save(tmp_path / "depth.png")
    source = {"a": {"visible": "rgb.png", "infrared": "ir.png", "depth": "depth.png", "query": "q"}}
    with pytest.raises(ValueError, match="unsupported official depth encoding"):
        prepare_dataset(source, tmp_path, tmp_path / "out")


def test_full_5691_cpu_packaging_preserves_non_bbox_fields_and_zip(tmp_path):
    source = {str(i): {"query": f"query {i}", "visible": f"Images/{i}.png",
                       "bbox": [0, 0, 0, 0], "extra": {"keep": [i]}} for i in range(5691)}
    predictions = [{"id": key, "samples": [{"text": '{"bbox_2d":[100,200,900,800]}',
                                             "finish_reason": "stop"}]} for key in reversed(source)]
    before = deepcopy(source)
    result = package_predictions(source, predictions, tmp_path, "strict")
    assert result["queries"] == 5691 and source == before
    actual = json.loads(Path(result["json"]).read_text(encoding="utf-8"))
    assert list(actual) == list(source)
    for key, row in source.items():
        assert actual[key] == {**row, "bbox": [0.1, 0.2, 0.9, 0.8]}
    with zipfile.ZipFile(result["zip"]) as archive:
        assert archive.namelist() == ["predictions.json"]
        assert json.loads(archive.read("predictions.json")) == actual


def test_prepare_only_cli_needs_no_gpu_and_limit_never_packages(tmp_path):
    for name in ("rgb", "ir"):
        Image.new("RGB", (4, 3)).save(tmp_path / f"{name}.png")
    Image.fromarray(np.ones((3, 4), dtype=np.uint16)).save(tmp_path / "depth.png")
    source = {"a": {"visible": "rgb.png", "infrared": "ir.png", "depth": "depth.png", "query": "q"}}
    queries = tmp_path / "queries.json"
    queries.write_text(json.dumps(source), encoding="utf-8")
    out = tmp_path / "out"
    summary = main(["--queries", str(queries), "--data-root", str(tmp_path), "--output-dir", str(out),
                    "--prepare-only", "--expected-query-count", "1", "--limit", "1"])
    assert summary["prepared_queries"] == 1
    assert not (out / "submission.zip").exists()
    config = json.loads((out / "run_config.json").read_text(encoding="utf-8"))
    assert (config["engine"], config["precision"], config["answer_mode"]) == ("hf", "nf4", "short")
