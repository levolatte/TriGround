import json

from tools.report_8b_experiment import screening_decision, summarize


def test_three_hits_is_only_a_replication_screen():
    result = screening_decision({"hits": 70}, {"hits": 69, "parse_rate": 1}, {"hits": 72, "parse_rate": 1})
    assert result["next"] == "paired_c4_t4_seed2027"
    assert result["significance_claim"] is False
    assert screening_decision({"hits": 73}, {"hits": 69, "parse_rate": 1}, {"hits": 72, "parse_rate": 1})["next"] == "w4"


def test_best_does_not_substitute_for_missing_fixed4(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    report = {"diagnostic_metadata": {"samples": 119}, "rgb_ir_depth": {
        "acc_0.5": 1.0, "mean_iou": 0.9, "parse_rate": 1.0,
    }}
    (results / "t4_best.json").write_text(json.dumps(report))
    decision = summarize(tmp_path)
    assert decision["status"] == "pending"
    assert decision["next"] == "complete_mandatory"
    assert decision["best_candidates"][0]["experiment"] == "t4_best"
