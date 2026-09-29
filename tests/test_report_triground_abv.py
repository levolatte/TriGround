from __future__ import annotations

import json
import sys

import pytest

from tools import report_triground_abv
from tools.report_triground_abv import build_report, write_outputs


def _manifest(count=20):
    return {
        f"id-{index}": {
            "visible": f"scene-{index // 2}/rgb.png",
            "infrared": f"scene-{index // 2}/ir.png",
            "depth": f"scene-{index // 2}/depth.png",
            "bbox": [0.1, 0.1, 0.4, 0.4],
        }
        for index in range(count)
    }


def _predictions(path, manifest, hits):
    path.write_text(
        "\n".join(
            json.dumps({
                "id": sample_id,
                "prediction": [0.1, 0.1, 0.4, 0.4] if index in hits else None,
                "target": [0.0, 0.0, 1.0, 1.0],
                "iou": 1.0 if index not in hits else 0.0,
            })
            for index, sample_id in enumerate(manifest)
        ) + "\n",
        encoding="utf-8",
    )


def test_full_denominator_raw_gt_gates_and_no_partial_ranking(tmp_path):
    manifest = _manifest()
    paths = {name: tmp_path / f"{name}.jsonl" for name in ("C0", "A", "B", "V", "diagnostic")}
    _predictions(paths["C0"], manifest, {0, 1})
    _predictions(paths["A"], manifest, {0, 1, 2, 4, 6, 8})
    _predictions(paths["B"], manifest, {0, 1, 2, 4, 6, 8, 10})
    _predictions(paths["V"], dict(list(manifest.items())[:3]), {0, 1, 2})
    report = build_report(manifest, paths, bootstrap_replicates=20)

    assert report["runs"]["C0"]["hits_0.5"] == 2  # evaluator iou fields are stale
    assert report["runs"]["A"]["hits_0.5"] == 6
    assert report["gates_vs_C0"]["A"]["candidate"] is True
    assert report["gates_vs_C0"]["A"]["rescued_image_groups"] == 4
    assert report["ranking"] == ["B", "A", "C0"]
    assert report["runs"]["V"]["status"] == "partial"
    assert report["runs"]["V"]["metrics"] is None
    assert report["runs"]["V"]["preview_metrics"]["samples"] == 3
    assert report["runs"]["diagnostic"]["status"] == "pending"
    assert "C0_to_V" not in report["pairs"]
    assert report["modal_utility"] == {}
    assert report["pairs"]["C0_to_A"]["wrong_to_right"]["ids"] == [
        "id-2", "id-4", "id-6", "id-8"
    ]
    output = tmp_path / "out"
    write_outputs(report, output)
    assert (output / "summary.csv").exists()
    assert "partial/pending" in (output / "report.md").read_text(encoding="utf-8")


def test_modal_rescue_harm_and_class_minimum(tmp_path):
    manifest = _manifest()
    paths = {name: tmp_path / f"{name}.jsonl" for name in ("C0", "B", "V")}
    _predictions(paths["C0"], manifest, {0, 1})
    _predictions(paths["B"], manifest, {0, 1, 2, 4})
    _predictions(paths["V"], manifest, {0, 2, 4, 6})
    classes = {sample_id: "large" if index < 16 else "small"
               for index, sample_id in enumerate(manifest)}
    report = build_report(manifest, paths, class_map=classes, scene_map=classes,
                          bootstrap_replicates=20)
    adaptation = report["pairs"]["B_to_V"]
    assert adaptation["wrong_to_right"]["ids"] == ["id-6"]
    assert adaptation["right_to_wrong"]["ids"] == ["id-1"]
    assert adaptation["scene_cluster_bootstrap"]["scene_groups"] == 2
    assert report["class_slices"]["large"]["runs"]["V"]["samples"] == 16
    assert report["class_slices"]["small"]["runs"]["V"] is None
    assert report["class_slices"]["small"]["interpretation"] == "insufficient_n_no_class_conclusion"
    assert report["scene_clusters"] == 2


def test_modal_utility_same_checkpoint_and_class_gate(tmp_path):
    manifest = _manifest()
    paths = {name: tmp_path / f"{name}.jsonl" for name in ("C0", "A", "B", "V")}
    _predictions(paths["C0"], manifest, {0})
    _predictions(paths["A"], manifest, {0, 1, 2})
    _predictions(paths["B"], manifest, {0, 1, 2, 4})
    _predictions(paths["V"], manifest, {0, 2, 4, 6})
    modal_paths = {}
    for arm, hits in (("A", {0, 2}), ("B", {0, 2, 6}), ("V", {0, 1, 2, 6})):
        path = tmp_path / f"{arm}_ir_missing.jsonl"
        _predictions(path, manifest, hits)
        modal_paths[(arm, "ir_missing")] = path
    classes = {sample_id: "IR_needed" if index < 16 else "RGB_sufficient"
               for index, sample_id in enumerate(manifest)}
    report = build_report(manifest, paths, modal_paths=modal_paths,
                          class_map=classes, bootstrap_replicates=20)
    b = report["modal_utility"]["B:ir_missing"]
    assert (b["rescued"], b["harmed"], b["net"]) == (2, 1, 1)
    assert b["rescued_ids"] == ["id-1", "id-4"]
    assert b["harmed_ids"] == ["id-6"]
    assert b["class_slices"]["IR_needed"]["samples"] == 16
    assert b["class_slices"]["RGB_sufficient"]["status"] == "insufficient_n_no_class_conclusion"
    assert report["modal_utility_delta_vs_A"]["B:ir_missing"]["rescued_delta"] == 1
    assert "A_to_B" in report["pairs"]


def test_required_run_config_checks_source_and_summary(tmp_path):
    manifest = _manifest(2)
    run_dir = tmp_path / "c0"
    run_dir.mkdir()
    path = run_dir / "predictions.jsonl"
    _predictions(path, manifest, {0})
    with pytest.raises(ValueError, match="run_config.json"):
        build_report(manifest, {"C0": path}, require_run_config=True, bootstrap_replicates=2)
    (run_dir / "run_config.json").write_text(json.dumps({"adapter": "c0"}), encoding="utf-8")
    with pytest.raises(ValueError, match="summary.json"):
        build_report(manifest, {"C0": path}, require_run_config=True, bootstrap_replicates=2)
    (run_dir / "summary.json").write_text(
        json.dumps({"adapter": "different", "predictions": str(path)}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="adapter differ"):
        build_report(manifest, {"C0": path}, require_run_config=True, bootstrap_replicates=2)
    (run_dir / "summary.json").write_text(
        json.dumps({"adapter": "c0", "predictions": "/root/cloud/eval/predictions.jsonl",
                    "samples": 2}), encoding="utf-8"
    )
    assert build_report(manifest, {"C0": path}, require_run_config=True,
                        bootstrap_replicates=2)["ranking"] == ["C0"]


def test_class_map_must_cover_same_ids(tmp_path):
    manifest = _manifest(2)
    path = tmp_path / "C0.jsonl"
    _predictions(path, manifest, {0})
    with pytest.raises(ValueError, match="exactly match"):
        build_report(manifest, {"C0": path}, class_map={"id-0": "x"}, bootstrap_replicates=2)


def test_city412_partition_mismatch_fails(tmp_path):
    manifest = _manifest(412)
    path = tmp_path / "C0.jsonl"
    _predictions(path, manifest, set())
    with pytest.raises(ValueError, match="47/365"):
        build_report(manifest, {"C0": path}, bootstrap_replicates=2)


def test_invalid_boxes_count_as_zero_without_aborting_full_denominator(tmp_path):
    manifest = _manifest(5)
    path = tmp_path / "C0.jsonl"
    boxes = (
        [0.1, 0.1, 0.4, 0.4],
        [0.6, 0.1, 0.4, 0.4],  # reversed
        [-0.1, 0.1, 0.4, 0.4],  # out of bounds
        [float("nan"), 0.1, 0.4, 0.4],
        None,  # parse failure
    )
    path.write_text(
        "\n".join(json.dumps({"id": sample_id, "prediction": box})
                  for sample_id, box in zip(manifest, boxes)) + "\n",
        encoding="utf-8",
    )
    report = build_report(manifest, {"C0": path}, bootstrap_replicates=2)
    item = report["runs"]["C0"]
    assert item["status"] == "complete"
    assert item["denominator"] == 5
    assert item["hits_0.5"] == 1
    assert item["metrics"]["parsed"] == 1
    assert item["invalid_predictions"] == {
        "id-1": "invalid_box", "id-2": "invalid_box", "id-3": "invalid_box",
        "id-4": "parse_failure",
    }


def test_cli_modal_run_uses_same_diagnostic_gt_ids(tmp_path, monkeypatch):
    manifest = _manifest(4)
    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps(manifest), encoding="utf-8")
    normal = tmp_path / "normal.jsonl"
    missing = tmp_path / "missing.jsonl"
    _predictions(normal, manifest, {0, 1})
    _predictions(missing, manifest, {0})
    output = tmp_path / "report"
    monkeypatch.setattr(sys, "argv", [
        "report_triground_abv", "--manifest", str(gt),
        "--run", f"C0={normal}",
        "--modal-run", f"C0:ir_missing={missing}",
        "--bootstrap-replicates", "2", "--output-dir", str(output),
    ])
    report_triground_abv.main()
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["modal_utility"]["C0:ir_missing"]["net"] == 1
