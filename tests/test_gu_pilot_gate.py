"""Small CPU checks for the G/U decision gates."""

import json
from pathlib import Path

import pytest

from tools.gate_gu_pilot import seed_gate, winner_gate, external_gate


def write_json(path: Path, value):
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    return path


def write_predictions(path: Path, count: int, correct: int):
    path.write_text("".join(json.dumps({"id": str(i), "prediction": [0, 0, 1, 1] if i < correct else [2, 2, 3, 3]}) + "\n" for i in range(count)), encoding="utf-8")
    return path


def test_seed_gate_recomputes_412_cases_and_corrected_groups(tmp_path):
    gt = write_json(tmp_path / "gt.json", {
        str(i): {"visible": f"rgb{i}", "infrared": f"ir{i}", "depth": f"d{i}", "bbox": [0, 0, 1, 1]}
        for i in range(412)
    })
    result = seed_gate(gt, write_predictions(tmp_path / "c.jsonl", 412, 292),
                       write_predictions(tmp_path / "g.jsonl", 412, 301),
                       write_predictions(tmp_path / "u.jsonl", 412, 300))
    assert result["status"] == "continue"
    assert result["branches"]["G"]["repaired_image_groups_vs_C"] == 9
    assert result["branches"]["U"]["passed"]


def test_winner_requires_four_cases_mean_for_u_and_allows_small_consistent_u_edge_for_g():
    def seed(g_hits, u_hits):
        return {"status": "continue", "branches": {
            "G": {"passed": True, "metrics": {"acc_0.5": g_hits / 412}},
            "U": {"passed": True, "metrics": {"acc_0.5": u_hits / 412}},
        }}
    assert winner_gate(seed(300, 305), seed(300, 303))["winner"] == "U"
    assert winner_gate(seed(300, 302), seed(300, 303))["winner"] == "G"
    assert winner_gate(seed(300, 302), seed(301, 300))["status"] == "stop"


def test_external_gate_requires_100_human_reviewed_unique_ids(tmp_path):
    gt = write_json(tmp_path / "gt.json", {
        str(i): {"visible": f"rgb{i}", "infrared": f"ir{i}", "depth": f"d{i}",
                 "bbox": [0, 0, 1, 1], "source": "rgbdt", "split": "external_review"}
        for i in range(100)
    })
    native = write_json(tmp_path / "native.json", [{"id": str(i)} for i in range(100)])
    review = tmp_path / "review.jsonl"
    review.write_text("".join(json.dumps({"id": str(i), "source": "rgbdt", "split": "external_review",
                                          "review_status": "human_accepted"}) + "\n" for i in range(100)), encoding="utf-8")
    m2_dir, winner_dir = tmp_path / "m2", tmp_path / "winner"
    m2_dir.mkdir(); winner_dir.mkdir()
    for directory in (m2_dir, winner_dir):
        write_json(directory / "summary.json", {
            "model": "base", "target_manifest": str(gt), "manifest": str(native),
            "min_pixels": 200704, "max_pixels": 602112, "max_new_tokens": 128,
            "prompt_style": "native", "tf32": False,
        })
        write_json(directory / "evaluation_host.json", {"hostname": "same-host"})
    m2 = write_predictions(m2_dir / "predictions.jsonl", 100, 70)
    winner = write_predictions(winner_dir / "predictions.jsonl", 100, 68)
    assert external_gate(gt, native, review, m2, winner, "rgbdt")["status"] == "pass"
    lines = review.read_text().splitlines()
    review.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(ValueError, match="100 unique IDs"):
        external_gate(gt, native, review, m2, winner, "rgbdt")
