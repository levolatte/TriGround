import json

import pytest

from tools.diagnose_quant_grounding_signal import analyze_signal, main


def reward_item(sample_id, occurrence_id, index, *, hit, reward, valid=True, truncated=False,
                source="city", negative_kind=None, sampling_pool="regular_city", iou=0.6,
                finish_reason="stop", generated_text=None):
    return {
        "id": sample_id,
        "occurrence_id": occurrence_id,
        "source": source,
        "negative_kind": negative_kind or [],
        "sampling_pool": sampling_pool,
        "finish_reason": finish_reason,
        "generated_text": generated_text,
        "generated_tokens": 24,
        "valid": valid,
        "format_valid": valid,
        "truncated": truncated,
        "iou": iou,
        "hit": hit,
        "reward": reward,
    }


def metrics_record(items):
    return {"count": len(items), "groups": [], "items": items}


def test_groups_by_occurrence_not_duplicate_query_id_and_keeps_partial_denominator():
    complete = [reward_item("duplicate", "occ-1", i, hit=True, reward=1.1) for i in range(8)]
    partial = [
        reward_item("duplicate", "occ-2", 0, hit=True, reward=1.1,
                    source="rgbdt", negative_kind=["extent"], sampling_pool="extent"),
        reward_item("duplicate", "occ-2", 1, hit=False, reward=0.1, valid=False,
                    truncated=True, source="rgbdt", negative_kind=["extent"], sampling_pool="extent"),
    ]

    report = analyze_signal([metrics_record(complete), metrics_record(partial)], 2.0)

    assert report["overall"]["groups"] == 2
    assert report["overall"]["unique_queries"] == 1
    assert report["overall"]["complete_groups"] == 1
    assert report["overall"]["incomplete_groups"] == 1
    assert report["overall"]["observed_answers"] == 10
    assert report["overall"]["expected_answer_slots"] == 16
    assert report["overall"]["unobserved_answer_slots"] == 6
    assert report["overall"]["outcomes"] == {"allhit": 1, "allwrong": 0, "mixed": 1}
    assert report["overall"]["differing_reward_groups"] == 1
    assert report["overall"]["differing_reward_groups_per_gpu_hour"] == 0.5
    assert report["overall"]["invalid_answers"] == 1
    assert report["overall"]["truncated_answers"] == 1
    assert report["by_source"]["rgbdt"]["incomplete_groups"] == 1
    assert report["by_negative_kind"]["extent"]["groups"] == 1
    assert report["by_sampling_pool"]["extent"]["groups"] == 1


def test_raw_text_missing_is_unknown_and_does_not_infer_d_reason_from_valid_false():
    items = [reward_item("q", "occ", i, hit=False, reward=-0.2, valid=False, iou=None)
             for i in range(8)]

    report = analyze_signal([metrics_record(items)], 1.0, answer_mode="thinking")

    assert report["answer_diversity"]["status"] == "unavailable"
    assert report["answer_diversity"]["reason"] == "raw_completions_not_provided"
    assert report["answer_diversity"]["mean_unique_parsed_bboxes_per_group"] is None
    assert report["answer_diversity"]["groups_with_bbox_variation"] is None
    assert report["d_failure_categories"]["status"] == "unknown"
    assert report["d_failure_categories"]["counts"]["unknown_raw_text"] == 8
    assert report["groups_detail"][0]["answer_diversity"]["reason"] == "raw_text_unknown"


def test_d_failure_categories_are_exclusive_and_use_raw_text():
    texts = [
        "reasoning that never finishes",
        "reasoning</think>not json",
        'reasoning</think>{"bbox_2d":[700,700,900,900]}',
        'reasoning</think>{"bbox_2d":[100,100,500,500]}',
        'reasoning</think>{"bbox_2d":[100,100,500,500]}',
        'reasoning</think>{"bbox_2d":[100,100,500,500]}',
        'reasoning</think>{"bbox_2d":[100,100,500,500]}',
        'reasoning</think>{"bbox_2d":[100,100,500,500]}',
    ]
    finish_reasons = ["length", "stop", "stop", "length", "stop", "stop", "stop", "stop"]
    items = [
        reward_item("q", "occ", i, hit=i >= 3, reward=1.1 if i >= 3 else -0.2,
                    valid=i >= 3, truncated=finish_reasons[i] == "length",
                    iou=0.1 if i == 2 else (1.0 if i >= 3 else None),
                    finish_reason=finish_reasons[i])
        for i in range(8)
    ]
    raw_samples = []
    for index, (text, reason) in enumerate(zip(texts, finish_reasons)):
        sample = {"text": text}
        if index != 3:  # Swift log_completions may not store finish reasons.
            sample["finish_reason"] = reason
        raw_samples.append(sample)
    raw = [{"id": "q", "samples": raw_samples}]

    report = analyze_signal([metrics_record(items)], 1.0, raw_rows=raw, answer_mode="thinking")

    assert report["d_failure_categories"]["status"] == "available"
    assert report["d_failure_categories"]["counts"]["unfinished_thinking"] == 1
    assert report["d_failure_categories"]["counts"]["token_limit_after_thinking"] == 1
    assert report["d_failure_categories"]["counts"]["invalid_final_json"] == 1
    assert report["d_failure_categories"]["counts"]["valid_bbox_wrong_target"] == 1
    assert report["answer_diversity"]["groups_with_parsed_bbox"] == 1
    assert report["answer_diversity"]["groups_with_bbox_variation"] == 1
    assert report["groups_detail"][0]["truncated_answers"] == 2


def test_generated_text_in_reward_metrics_is_preferred_without_raw_file():
    texts = [
        "reasoning never ends",
        "reasoning</think>not json",
        'reasoning</think>{"bbox_2d":[700,700,900,900]}',
        'reasoning</think>{"bbox_2d":[100,100,500,500]}',
        'reasoning</think>{"bbox_2d":[100,100,500,500]}',
        'reasoning</think>{"bbox_2d":[100,100,500,500]}',
        'reasoning</think>{"bbox_2d":[100,100,500,500]}',
        'reasoning</think>{"bbox_2d":[100,100,500,500]}',
    ]
    finish_reasons = ["length", "stop", "stop", "length", "stop", "stop", "stop", "stop"]
    items = [
        reward_item("q", "occ", i, hit=i >= 3, reward=1.1 if i >= 3 else -0.2,
                    valid=i >= 3, truncated=finish_reasons[i] == "length",
                    iou=0.1 if i == 2 else (1.0 if i >= 3 else None),
                    finish_reason=finish_reasons[i], generated_text=texts[i])
        for i in range(8)
    ]

    report = analyze_signal([metrics_record(items)], 1.0)

    assert report["answer_mode"] == "thinking"
    assert report["answer_diversity"]["status"] == "available"
    assert report["answer_diversity"]["raw_sample_count"] == 8
    assert report["answer_diversity"]["groups_with_bbox_variation"] == 1
    assert report["d_failure_categories"]["status"] == "available"
    assert report["d_failure_categories"]["counts"]["unfinished_thinking"] == 1
    assert report["d_failure_categories"]["counts"]["token_limit_after_thinking"] == 1
    assert report["d_failure_categories"]["counts"]["invalid_final_json"] == 1
    assert report["d_failure_categories"]["counts"]["valid_bbox_wrong_target"] == 1


def test_partial_external_raw_completions_remain_unmatched_and_unknown():
    items = [reward_item("q", "occ", i, hit=False, reward=-0.2) for i in range(8)]
    raw = [{"id": "q", "samples": [
        {"text": '{"bbox_2d":[100,100,500,500]}'} for _ in range(4)
    ]}]

    report = analyze_signal([metrics_record(items)], 1.0, raw_rows=raw, answer_mode="thinking")

    assert report["answer_diversity"]["raw_sample_count"] == 4
    assert report["answer_diversity"]["groups_with_raw_samples"] == 0
    assert report["answer_diversity"]["status"] == "unavailable"
    assert report["d_failure_categories"]["status"] == "unknown"
    assert report["d_failure_categories"]["counts"]["unknown_raw_text"] == 8


def test_kl_and_swift_columnar_raw_completions():
    items = [reward_item("q", "occ", i, hit=True, reward=1.1) for i in range(8)]
    raw = [{"step": ["1"] * 8,
            "completion": ['{"bbox_2d":[100,100,500,500]}'] * 8}]
    report = analyze_signal(
        [metrics_record(items)], 0.5,
        training_rows=[{"step": 1, "metrics": {"kl": 0.0}}, {"step": 2, "metrics": {"kl": 0.25}}],
        raw_rows=raw,
    )

    assert report["kl"] == {
        "status": "available", "steps": [{"step": 1, "kl": 0.0}, {"step": 2, "kl": 0.25}],
        "count": 2, "mean": 0.125, "min": 0.0, "max": 0.25, "first": 0.0, "last": 0.25,
    }
    assert report["answer_diversity"]["groups_with_raw_samples"] == 1
    assert report["answer_diversity"]["mean_unique_parsed_bboxes_per_group"] == 1


def test_cli_writes_json_and_markdown_reports(tmp_path):
    reward_path = tmp_path / "reward.jsonl"
    raw_path = tmp_path / "raw.jsonl"
    output_dir = tmp_path / "report"
    items = [reward_item("q", "occ", i, hit=True, reward=1.1) for i in range(8)]
    reward_path.write_text(json.dumps(metrics_record(items)) + "\n", encoding="utf-8")
    raw_path.write_text(json.dumps({"id": "q", "samples": [
        {"text": '{"bbox_2d":[100,100,500,500]}', "finish_reason": "stop"}
        for _ in range(8)
    ]}) + "\n", encoding="utf-8")

    assert main(["--reward-metrics", str(reward_path), "--raw-completions", str(raw_path),
                 "--gpu-hours", "1", "--output-dir", str(output_dir)]) == 0

    report = json.loads((output_dir / "learning_signal_report.json").read_text(encoding="utf-8"))
    assert report["overall"]["groups"] == 1
    assert "Unique queries: 1" in (output_dir / "learning_signal_report.md").read_text(encoding="utf-8")


def test_gpu_hours_must_be_positive_finite():
    with pytest.raises(ValueError, match="positive finite"):
        analyze_signal([], 0.0)
