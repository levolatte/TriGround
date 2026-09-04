from __future__ import annotations

import json

from mm_grounding.engine import checkpoint_names, selection_improved, selection_key
from tools.select_fusion_checkpoint import best_epoch_key


def values(acc, miou, acc7, parsed):
    return {
        "acc_0.5": acc,
        "mean_iou": miou,
        "acc_0.7": acc7,
        "parse_rate": parsed,
    }


def test_acc_has_priority_over_higher_mean_iou():
    lower_acc = selection_key(values(0.6, 0.9, 0.8, 1.0))
    higher_acc = selection_key(values(0.7, 0.5, 0.4, 0.9))
    assert higher_acc > lower_acc


def test_selection_uses_all_tie_breakers_in_order():
    base = selection_key(values(0.7, 0.6, 0.5, 0.9))
    assert selection_key(values(0.7, 0.61, 0.1, 0.1)) > base
    assert selection_key(values(0.7, 0.6, 0.51, 0.1)) > base
    assert selection_key(values(0.7, 0.6, 0.5, 0.91)) > base
    assert selection_improved(base, None)
    assert not selection_improved(base, base)


def test_metric_log_selector_ignores_probe_and_uses_acc(tmp_path):
    path = tmp_path / "metrics.jsonl"
    records = [
        {"event": "early_probe", "eval_scope": "subset", **values(0.9, 0.9, 0.9, 1.0)},
        {"eval_scope": "subset", **values(0.6, 0.8, 0.7, 1.0)},
        {"eval_scope": "full", **values(0.7, 0.5, 0.4, 0.9)},
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    assert best_epoch_key(path) == (0.7, 0.5, 0.4, 0.9)


def test_each_phase_has_best_mean_iou_and_last_checkpoint_names():
    assert checkpoint_names("a") == (
        "best_phase_a.pt",
        "best_mean_iou_phase_a.pt",
        "last_phase_a.pt",
    )
    assert checkpoint_names("b") == (
        "best.pt",
        "best_mean_iou.pt",
        "last.pt",
    )
