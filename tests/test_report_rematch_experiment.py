import json

import pytest

from tools.report_rematch_experiment import (
    box_iou,
    build_candidate_pool,
    build_report,
    candidate_coverage,
    compare_runs,
    load_manifest,
    load_run,
    write_outputs,
)


def _manifest(records):
    return {
        sample_id: {
            "visible": f"visible/{group}.png",
            "infrared": f"infrared/{group}.png",
            "depth": f"depth/{group}.png",
            "query": "object",
            "bbox": bbox,
        }
        for sample_id, group, bbox in records
    }


def _run(name, rows):
    return {
        "name": name,
        "rows": {
            sample_id: {
                "id": sample_id,
                "primary": primary,
                "candidates": candidates,
                "raw": {},
            }
            for sample_id, primary, candidates in rows
        },
        "partial": False,
        "missing_ids": [],
        "path": f"{name}.jsonl",
        "metadata": None,
    }


def test_iou_threshold_is_real_and_controls_candidate_recall():
    target = [0.25, 0.0, 0.75, 1.0]
    proposal = [0.0, 0.0, 0.5, 1.0]
    assert box_iou(proposal, target) == pytest.approx(1 / 3)
    manifest = _manifest([("a", "scene-1", target)])
    run = _run("source", [("a", proposal, [proposal])])
    pool = build_candidate_pool([run], manifest)
    assert candidate_coverage(pool, manifest, threshold=0.3)["recall@1"] == 1.0
    assert candidate_coverage(pool, manifest, threshold=0.5)["recall@1"] == 0.0


def test_pairwise_transitions_and_image_group_bootstrap():
    manifest = _manifest(
        [
            ("a", "scene-1", [0.0, 0.0, 0.4, 0.4]),
            ("b", "scene-1", [0.0, 0.0, 0.4, 0.4]),
            ("c", "scene-2", [0.0, 0.0, 0.4, 0.4]),
        ]
    )
    wrong = [0.6, 0.6, 0.9, 0.9]
    exact = [0.0, 0.0, 0.4, 0.4]
    left = _run("left", [("a", wrong, [wrong]), ("b", exact, [exact]), ("c", exact, [exact])])
    right = _run("right", [("a", exact, [exact]), ("b", exact, [exact]), ("c", wrong, [wrong])])
    report = compare_runs(left, right, manifest, replicates=40, seed=2026)
    assert report["wrong_to_right"]["ids"] == ["a"]
    assert report["wrong_to_right"]["image_count"] == 1
    assert report["right_to_wrong"]["ids"] == ["c"]
    assert report["right_to_wrong"]["image_count"] == 1
    assert report["image_group_bootstrap"]["replicates"] == 40
    assert len(report["image_group_bootstrap"]["acc_0.5_delta_95ci"]) == 2


def test_candidate_order_is_round_robin_and_never_gt_ranked():
    manifest = _manifest([("a", "scene-1", [0.75, 0.0, 1.0, 1.0])])
    first_box = [0.0, 0.0, 0.5, 1.0]
    gt_box = manifest["a"]["bbox"]
    first = _run("first", [("a", first_box, [first_box])])
    second = _run("second", [("a", gt_box, [gt_box])])
    pool = build_candidate_pool([first, second], manifest)
    assert [item["source"] for item in pool["a"]] == ["first", "second"]
    assert candidate_coverage(pool, manifest)["recall"]["1"]["covered_samples"] == 0
    assert candidate_coverage(pool, manifest)["recall"]["4"]["covered_samples"] == 1


def test_ninth_candidate_is_only_in_full_pool():
    wrong = [[index / 10, 0.0, (index + 0.5) / 10, 0.4] for index in range(8)]
    target = [0.8, 0.0, 0.9, 0.4]
    manifest = _manifest([("a", "scene-1", target)])
    run = _run("source", [("a", wrong[0], wrong + [target])])

    full = build_candidate_pool([run], manifest)
    capped = build_candidate_pool([run], manifest, cap=8)
    assert len(full["a"]) == 9
    assert len(capped["a"]) == 8
    assert candidate_coverage(full, manifest)["recall@8"] == 0.0
    assert candidate_coverage(full, manifest)["recall@all"] == 1.0
    assert candidate_coverage(capped, manifest)["recall@all"] == 0.0

    report = build_report(manifest, [run], bootstrap_replicates=10)["candidate_union"]
    assert len(report["pool"]["a"]) == 9
    assert len(report["pool_top8"]["a"]) == 8
    assert report["new_cover_samples_vs_best"] == 0
    assert report["new_cover_samples_vs_best_primary"] == 1
    assert report["C"] == 0


def test_missing_ids_fail_without_explicit_partial_flag(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            _manifest(
                [
                    ("a", "scene-1", [0.0, 0.0, 0.4, 0.4]),
                    ("b", "scene-2", [0.0, 0.0, 0.4, 0.4]),
                ]
            )
        ),
        encoding="utf-8",
    )
    prediction_path = tmp_path / "predictions.jsonl"
    prediction_path.write_text(
        json.dumps({"id": "a", "prediction": [0.0, 0.0, 0.4, 0.4]}) + "\n",
        encoding="utf-8",
    )
    manifest = load_manifest(manifest_path)
    with pytest.raises(ValueError, match="allow-partial"):
        load_run("partial", prediction_path, set(manifest), allow_partial=False)
    run = load_run("partial", prediction_path, set(manifest), allow_partial=True)
    assert run["partial"] is True
    report = build_report(manifest, [run], bootstrap_replicates=10)
    assert report["pairs"] == []
    assert report["candidate_union"]["status"] == "excluded_no_complete_runs"


def test_manifest_gt_recomputes_stale_evaluator_fields_and_reports_metadata(tmp_path):
    manifest_payload = _manifest(
        [
            ("a", "scene-1", [0.0, 0.0, 0.4, 0.4]),
            ("b", "scene-2", [0.0, 0.0, 0.4, 0.4]),
        ]
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    predictions = run_dir / "predictions.jsonl"
    predictions.write_text(
        "\n".join(
            [
                json.dumps({"id": "a", "prediction": [0.0, 0.0, 0.4, 0.4], "iou": 0.0}),
                json.dumps({"id": "b", "prediction": [0.6, 0.6, 0.9, 0.9], "iou": 1.0}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    metadata = {"model": "cpu-test", "max_new_tokens": 17}
    (run_dir / "summary.json").write_text(json.dumps(metadata), encoding="utf-8")
    manifest = load_manifest(manifest_path)
    run = load_run("model", predictions, set(manifest), allow_partial=False)
    report = build_report(manifest, [run], bootstrap_replicates=10)
    assert report["runs"][0]["metrics"]["acc_0.5"] == 0.5
    assert report["runs"][0]["metadata"]["data"] == metadata
    output_dir = tmp_path / "report"
    write_outputs(report, output_dir)
    assert (output_dir / "summary.csv").exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "report.md").exists()
    assert json.loads((run_dir / "summary.json").read_text(encoding="utf-8")) == metadata
