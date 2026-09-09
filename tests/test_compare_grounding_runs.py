from tools.compare_grounding_runs import compare


def row(sample_id, iou, scene, parsed=True, scale="small"):
    return {
        "id": sample_id,
        "target": [0.1, 0.1, 0.5, 0.5],
        "iou": iou,
        "parsed": parsed,
        "scene_group": scene,
        "scale_bin": scale,
    }


def test_paired_comparison_reports_transitions_groups_and_bootstrap():
    control = {
        "a": row("a", 0.4, "scene-1"),
        "b": row("b", 0.8, "scene-1"),
        "c": row("c", 0.6, "scene-2"),
    }
    treatment = {
        "a": row("a", 0.7, "scene-1"),
        "b": row("b", 0.9, "scene-1"),
        "c": row("c", 0.3, "scene-2", parsed=False),
    }
    report = compare(control, treatment, bootstrap_replicates=20, seed=2026)
    assert report["transitions"]["incorrect_to_correct"] == 1
    assert report["transitions"]["correct_to_incorrect"] == 1
    assert report["delta"]["acc_0.5"] == 0.0
    assert report["treatment"]["parse_rate"] == 2 / 3
    assert report["scene_bootstrap"]["replicates"] == 20
    assert "acc_0.7_delta_95ci" in report["scene_bootstrap"]
    assert "parse_rate_delta_95ci" in report["scene_bootstrap"]
    assert report["groups"]["scale_bin"]["small"]["control"]["samples"] == 3
