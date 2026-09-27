from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.report_modality_interventions import build_report, write_outputs


def _manifest():
    return {
        "a": {
            "visible": "rgb/scene-1.png",
            "infrared": "ir/scene-1.png",
            "depth": "depth/scene-1.png",
            "query": "Near the nearest curb",
            "bbox": [0.0, 0.0, 0.4, 0.4],
        },
        "b": {
            "visible": "rgb/scene-1.png",
            "infrared": "ir/scene-1.png",
            "depth": "depth/scene-1.png",
            "query": "the second person",
            "bbox": [0.0, 0.0, 0.4, 0.4],
        },
        "c": {
            "visible": "rgb/scene-2.png",
            "infrared": "ir/scene-2.png",
            "depth": "depth/scene-2.png",
            "query": "a person beside a tree",
            "bbox": [0.0, 0.0, 0.4, 0.4],
        },
    }


def _write_run(path, predictions):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps({"id": sample_id, "prediction": box, "raw_text": raw_text}) + "\n"
            for sample_id, box, raw_text in predictions
        ),
        encoding="utf-8",
    )


def test_full_pairs_are_scored_and_partial_arms_are_progress_only(tmp_path):
    manifest = _manifest()
    wrong = [0.6, 0.6, 0.9, 0.9]
    exact = manifest["a"]["bbox"]
    experiment_dir = tmp_path / "experiment"
    normal = [("a", wrong, "normal-a"), ("b", exact, "normal-b"), ("c", exact, "same")]
    intervention = [("a", exact, "changed-a"), ("b", wrong, "changed-b"), ("c", exact, "same")]
    _write_run(experiment_dir / "normal" / "predictions.jsonl", normal)
    _write_run(experiment_dir / "ir_black" / "predictions.jsonl", intervention)
    _write_run(experiment_dir / "depth_black" / "predictions.jsonl", intervention[:2])
    old_m2_path = tmp_path / "old_m2_4090.jsonl"
    rgb_path = tmp_path / "rgb_4090.jsonl"
    _write_run(old_m2_path, intervention)
    _write_run(rgb_path, normal)

    report = build_report(
        manifest,
        experiment_dir,
        old_m2_path,
        rgb_path,
        bootstrap_replicates=40,
        bootstrap_seed=7,
    )

    arms = {run["name"]: run for run in report["runs"]}
    assert arms["normal"]["status"] == "complete"
    assert arms["normal"]["metrics"]["samples"] == 3
    assert arms["depth_black"]["status"] == "partial"
    assert arms["depth_black"]["observed_metrics"]["samples"] == 2
    assert arms["depth_black"]["comparison_eligible"] is False
    assert arms["both_shuffle"]["status"] == "not_started"

    by_right = {pair["right"]: pair for pair in report["intervention_comparisons"]}
    paired = by_right["ir_black"]
    assert paired["status"] == "complete"
    assert paired["samples"] == 3
    assert paired["wrong_to_right"]["ids"] == ["a"]
    assert paired["right_to_wrong"]["ids"] == ["b"]
    assert paired["same_valid_box_count"] == 1
    assert paired["both_valid_box_count"] == 3
    assert paired["mean_coordinate_l1_when_both_valid"] == pytest.approx((0.55 + 0.55) / 3)
    assert paired["raw_text_compared_count"] == 3
    assert paired["raw_text_changed_count"] == 2
    assert paired["image_group_bootstrap"]["image_groups"] == 2
    assert paired["exploratory_query_slices"]["distance_keyword"]["ids"] == ["a"]
    assert paired["exploratory_query_slices"]["ordinal_keyword"]["ids"] == ["b"]

    assert by_right["depth_black"]["status"] == "excluded_incomplete_run"
    assert by_right["depth_black"]["incomplete_runs"] == ["depth_black"]
    assert report["cross_hardware_comparison"]["status"] == "complete"
    historical = report["historical_4090_rgb_vs_old_m2"]
    assert historical["wrong_to_right"]["ids"] == ["a"]
    assert historical["right_to_wrong"]["ids"] == ["b"]
    assert "candidate_union" not in report

    output_dir = tmp_path / "report"
    write_outputs(report, output_dir)
    assert json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))[
        "manifest_samples"
    ] == 3
    assert "干预" in (output_dir / "report.md").read_text(encoding="utf-8")
    assert (output_dir / "summary.csv").exists()


def test_direct_pair_comparison_rejects_partial_run():
    from tools.report_modality_interventions import compare_complete_runs, load_optional_run

    manifest = _manifest()
    partial_path = Path("not-created.jsonl")
    partial = load_optional_run("partial", partial_path, manifest)
    with pytest.raises(ValueError, match="complete 412-ID"):
        compare_complete_runs(partial, partial, manifest, bootstrap_replicates=2)
