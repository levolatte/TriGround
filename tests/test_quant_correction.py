"""ACC 纠错核心的 CPU 分流、恢复与实际梯度检查。"""

from __future__ import annotations

import copy
import json
import random
import statistics

import pytest
import torch

from tools.prepare_qwen3vl_native_sft import bbox_to_qwen1000
from tools.quant_grounding_reward import box_is_valid, iou_xyxy
from tools.quant_correction import (
    Completion,
    QuestionSampler,
    RouteDecision,
    ce_question_loss,
    classify,
    decide_route,
    jitter_solution,
    mixed_question_loss,
    random_state_from_json,
    random_state_to_json,
    serialize_gt,
)


def test_jitter_solution_stays_inside_the_acc05_tolerance_band():
    """AC C@0.5 的容忍带是线性 0.707~1.414 倍；抖动的监督目标必须始终落在带内。

    教出一个必然超出该带的框等于教模型做错，所以这条不变量比抖动本身更重要。
    """
    solution = [0.476042, 0.59537, 0.5375, 0.917592]
    base_area = (solution[2] - solution[0]) * (solution[3] - solution[1])
    rng = random.Random(2026)
    for low, high, min_iou in ((0.85, 1.20, 0.65), (1.00, 1.30, 0.60)):
        for _ in range(500):
            candidate = jitter_solution(solution, rng, min_iou, low, high)
            assert box_is_valid(candidate)
            assert iou_xyxy(candidate, solution) >= min_iou - 1e-9
            area = (candidate[2] - candidate[0]) * (candidate[3] - candidate[1])
            linear = (area / base_area) ** 0.5
            assert 0.707 <= linear <= 1.414, linear


def test_jitter_target_pulls_upwards_and_default_keeps_exact_gt():
    """偏大抖动要让监督目标整体上移（实测近失题尺寸比中位仅 0.79），
    而 rng 为 None 时必须逐字等于真值，保证旧运行可复现。"""
    solution = [0.476042, 0.59537, 0.5375, 0.917592]
    row = {"id": "q", "solution": solution}
    assert serialize_gt(row) == '{"bbox_2d":[476,595,538,918]}'
    assert serialize_gt(row, None, 0.60, 1.00, 1.30) == '{"bbox_2d":[476,595,538,918]}'

    rng = random.Random(2026)
    base_area = (solution[2] - solution[0]) * (solution[3] - solution[1])
    scales = []
    for _ in range(400):
        candidate = jitter_solution(solution, rng, 0.60, 1.00, 1.30)
        area = (candidate[2] - candidate[0]) * (candidate[3] - candidate[1])
        scales.append((area / base_area) ** 0.5)
    assert min(scales) >= 1.0 - 1e-9
    assert 1.08 <= statistics.mean(scales) <= 1.20


def test_jitter_survives_tiny_targets_where_it_matters():
    """真实失分集中在小目标（面积约 0.001）；那里也必须能抽到合格框，不能大量回退。"""
    tiny = [0.500, 0.500, 0.532, 0.532]
    rng = random.Random(11)
    accepted = sum(1 for _ in range(500)
                   if iou_xyxy(jitter_solution(tiny, rng, 0.60, 1.00, 1.30), tiny) >= 0.60)
    assert accepted >= 495


def row(sample_id="query_0", **extra):
    return {
        "id": sample_id,
        "solution": [0.0, 0.0, 1.0, 1.0],
        "messages": [{"role": "user", "content": "<image>Locate the person."}],
        **extra,
    }


def completion(width=1000, *, reason="stop", text=None, token_ids=None):
    return Completion(
        text if text is not None else f'{{"bbox_2d":[0,0,{width},1000]}}',
        [10, 20, 30] if token_ids is None else token_ids,
        reason,
    )


def rows(count=12):
    return [row(f"query_{index}") for index in range(count)]


@pytest.mark.parametrize(
    "widths,hits,route",
    [((1000, 500), [True, True], "all_hit"),
     ((500, 499), [True, False], "mixed"),
     ((499, 500), [False, True], "mixed"),
     ((499, 300), [False, False], "all_wrong")],
)
def test_classification_uses_binary_half_iou_boundary(widths, hits, route):
    result = classify(row(), [completion(widths[0]), completion(widths[1])])
    assert result["hits"] == hits
    assert result["route"] == route


@pytest.mark.parametrize("reason", ["stop", "eos"])
def test_punctuation_normalization_never_rewrites_rollout_tokens(reason):
    raw = '{"bbox_2d":[(0,0),(1000,1000)]}'
    tokens = [101, 902, 1001]
    first = completion(text=raw, reason=reason, token_ids=tokens)
    other = completion(499)
    original = copy.deepcopy(row())
    result = classify(original, [first, other])
    assert result == {
        "hits": [True, False],
        "normalized_texts": ['{"bbox_2d":[0,0,1000,1000]}', other.text],
        "route": "mixed",
        # 逐样本的解析框，供方向自适应的容忍带抖动判断模型偏大还是偏小。
        # 它必须来自与判分同一份 normalized_texts，否则「判分说命中、抖动方向却按另一个框算」。
        "predicted_boxes": [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.499, 1.0]],
    }
    assert first.text == raw
    assert first.token_ids is tokens
    assert first.token_ids == [101, 902, 1001]
    assert original == row()


@pytest.mark.parametrize("text", [
    '{"bbox_2d":[0,0,1000,1000],"label":"person"}',
    '[{"bbox_2d":[0,0,1000,1000]}]',
    '```json\n{"bbox_2d":[0,0,1000,1000]}\n```',
    '{"bbox_2d":[0,0,1000,1000]} extra',
    '{"bbox_2d":[0,0,1001,1000]}',
    '{"bbox_2d":[0,0,1000,1000],"bbox_2d":[0,0,500,1000]}',
    '<think>reason</think>{"bbox_2d":[0,0,1000,1000]}',
])
def test_classification_keeps_existing_strict_schema(text):
    result = classify(row(), [completion(text=text), completion(1000)])
    assert result["hits"] == [False, True]
    assert result["normalized_texts"][0] == text


def test_truncation_is_wrong_even_with_valid_box_and_terminal_wrapper():
    text = completion().text + "<|im_end|>"
    result = classify(row(), [completion(reason="length", text=text), completion()])
    assert result["hits"] == [False, True]
    unknown = classify(row(), [completion(reason="unknown"), completion()])
    assert unknown["hits"] == [False, True]
    terminated = classify(row(), [completion(reason="unknown", text=text), completion()])
    assert terminated["route"] == "all_hit"


def test_classification_does_not_use_reviewed_negatives_or_iou_auxiliary_rewards():
    example = row(negative_boxes=[[0, 0, 0.5, 1]], source="reviewed_pair")
    assert classify(example, [completion(500), completion(499)])["hits"] == [True, False]
    with pytest.raises(ValueError, match="两个"):
        classify(example, [completion()])


@pytest.mark.parametrize("route,expected", [("mixed", "mixed"), ("all_wrong", "gt")])
def test_nonretention_routes_do_not_advance_independent_rng(route, expected):
    rng = random.Random(99)
    state = rng.getstate()
    assert decide_route({"route": route}, rng) == RouteDecision(expected)
    assert rng.getstate() == state


def test_all_hit_p0_skips_and_p1_selects_both_successful_answers_uniformly():
    rng = random.Random(37)
    assert all(decide_route({"route": "all_hit"}, rng, 0) == RouteDecision("skip")
               for _ in range(20))
    rng = random.Random(37)
    chosen = [decide_route({"route": "all_hit"}, rng, 1) for _ in range(1000)]
    assert {item.route for item in chosen} == {"retain"}
    indices = [item.selected_index for item in chosen]
    assert 450 <= indices.count(0) <= 550
    assert set(indices) == {0, 1}


def test_default_retention_probability_and_rng_json_restore():
    first = random.Random(2026)
    second = random.Random(2026)
    for _ in range(100):
        expected = (RouteDecision("retain", second.randrange(2))
                    if second.random() < 0.2 else RouteDecision("skip"))
        assert decide_route({"route": "all_hit"}, first) == expected
    state = json.loads(json.dumps(random_state_to_json(first.getstate())))
    restored = random.Random()
    restored.setstate(random_state_from_json(state))
    assert [first.random() for _ in range(20)] == [restored.random() for _ in range(20)]
    with pytest.raises(ValueError, match="keep_probability"):
        decide_route({"route": "all_hit"}, first, 1.1)
    with pytest.raises(ValueError, match="未知分类"):
        decide_route({"route": "typo"}, first)


def test_gt_is_compact_quantized_target_and_never_enters_prompt():
    example = row("gt_query", solution=[0.0125, 0.2005, 0.6789, 0.9995])
    original = copy.deepcopy(example)
    text = serialize_gt(example)
    assert text == '{"bbox_2d":[13,201,679,1000]}'
    assert json.loads(text)["bbox_2d"] == bbox_to_qwen1000(example["solution"])
    assert example == original
    assert all(message["role"] != "assistant" for message in example["messages"])
    sampler = QuestionSampler([example, *rows(3)])
    assert all(all(message["role"] != "assistant" for message in item["messages"])
               for item in sampler.next_batch())


def test_sampler_rejects_duplicate_ids_small_pools_and_assistant_prompt():
    with pytest.raises(ValueError, match="唯一查询"):
        QuestionSampler([*rows(4), row()])
    with pytest.raises(ValueError, match="至少四个"):
        QuestionSampler(rows(3))
    unsafe = rows(4)
    unsafe[0]["messages"].append({"role": "assistant", "content": serialize_gt(unsafe[0])})
    with pytest.raises(ValueError, match="assistant"):
        QuestionSampler(unsafe)


def test_fresh_cycles_are_uniform_over_queries_without_source_oversampling():
    examples = rows(12)
    for index, example in enumerate(examples):
        example["source"] = "negative_pool" if index == 0 else "regular"
        example["negative_boxes"] = [[0, 0, 0.5, 1]] if index == 0 else []
    sampler = QuestionSampler(examples)
    unweighted = QuestionSampler(rows(12))
    for _ in range(4):
        cycle = []
        for _ in range(3):
            batch = sampler.next_batch()
            ids = [item["id"] for item in batch]
            assert len(set(ids)) == 4
            assert ids == [item["id"] for item in unweighted.next_batch()]
            cycle.extend(ids)
        assert len(set(cycle)) == len(examples)
        assert set(cycle) == {item["id"] for item in examples}
    assert sampler.fresh_count == 48
    assert sampler.revisit_count == 0


def test_latest_mixed_pool_revisits_at_most_two_and_removes_resolved_queries():
    sampler = QuestionSampler(rows(12))
    for example in rows(12):
        sampler.observe(example["id"], "mixed")
    for _ in range(10):
        before_fresh = sampler.fresh_count
        before_revisits = sampler.revisit_count
        batch = sampler.next_batch()
        assert len({item["id"] for item in batch}) == 4
        assert sampler.fresh_count - before_fresh == 2
        assert sampler.revisit_count - before_revisits == 2
    sampler.observe("query_0", "all_wrong")
    sampler.observe("query_1", "all_hit")
    assert set(sampler.mixed_pool) == {f"query_{index}" for index in range(2, 12)}
    for index in range(2, 12):
        sampler.observe(f"query_{index}", "all_hit")
    before = sampler.fresh_count
    sampler.next_batch()
    assert sampler.fresh_count - before == 4
    assert sampler.mixed_pool == {}
    with pytest.raises(KeyError):
        sampler.observe("unknown", "mixed")
    with pytest.raises(ValueError, match="未知分类"):
        sampler.observe("query_0", "retain")


def test_small_mixed_pool_fills_shortfall_with_fresh_and_never_duplicates():
    sampler = QuestionSampler(rows(4))
    sampler.observe("query_0", "mixed")
    for _ in range(20):
        fresh_before, revisit_before = sampler.fresh_count, sampler.revisit_count
        batch = sampler.next_batch()
        fresh_delta = sampler.fresh_count - fresh_before
        revisit_delta = sampler.revisit_count - revisit_before
        assert fresh_delta + revisit_delta == 4
        assert revisit_delta in {0, 1}
        assert len({item["id"] for item in batch}) == 4


def test_sampler_json_resume_continues_batches_observations_and_counters():
    original = QuestionSampler(rows(7), seed=4)
    for index in range(11):
        for example in original.next_batch():
            original.observe(example["id"], "mixed" if index % 2 else "all_wrong")
    saved = json.loads(json.dumps(original.state_dict()))
    restored = QuestionSampler(rows(7), seed=999)
    restored.load_state_dict(saved)
    assert restored.state_dict() == saved
    for index in range(35):
        first = original.next_batch()
        second = restored.next_batch()
        assert [item["id"] for item in first] == [item["id"] for item in second]
        assert len({item["id"] for item in first}) == 4
        for position, example in enumerate(first):
            route = ("mixed", "all_wrong", "all_hit")[(index + position) % 3]
            original.observe(example["id"], route)
            restored.observe(example["id"], route)
        assert original.state_dict() == restored.state_dict()
    saved["fresh_order"][0] = "different_dataset"
    with pytest.raises(ValueError, match="train_index"):
        restored.load_state_dict(saved)


def test_retention_rng_does_not_change_question_order():
    first, second = QuestionSampler(rows()), QuestionSampler(rows())
    retention_rng = random.Random(123)
    for _ in range(10):
        decide_route({"route": "all_hit"}, retention_rng)
        assert [item["id"] for item in first.next_batch()] == [item["id"] for item in second.next_batch()]


def test_binary_rloo_has_exact_plus_minus_one_direction_and_token_means():
    correct = torch.tensor([-1.0, -3.0], requires_grad=True)
    wrong = torch.tensor([-2.0, -4.0, -6.0], requires_grad=True)
    loss = mixed_question_loss([correct, wrong], [correct.detach(), wrong.detach()], [True, False], beta=0)
    assert loss.item() == pytest.approx((2.0 - 4.0) / 2)
    loss.backward()
    torch.testing.assert_close(correct.grad, torch.full_like(correct, -1 / 4))
    torch.testing.assert_close(wrong.grad, torch.full_like(wrong, 1 / 6))


def test_kl_uses_unclipped_reference_delta_and_never_backprops_reference():
    correct = torch.tensor([-5.0, -2.0], dtype=torch.float64, requires_grad=True)
    wrong = torch.tensor([-1.0], dtype=torch.float64, requires_grad=True)
    refs = [torch.tensor([-1.0, -1.5], dtype=torch.float64, requires_grad=True),
            torch.tensor([-2.5], dtype=torch.float64, requires_grad=True)]
    loss = mixed_question_loss([correct, wrong], refs, [True, False], beta=0.01)
    expected = sum(((-1 if hit else 1) * logp + 0.01 * (
        torch.exp(ref.detach() - logp) - (ref.detach() - logp) - 1)).mean()
        for logp, ref, hit in zip([correct, wrong], refs, [True, False])) / 2
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert correct.grad[0] < -0.25  # delta=4 保留完整 KL 梯度，没有 PPO 截断。
    assert all(reference.grad is None for reference in refs)


def test_ce_is_negative_mean_with_increasing_target_log_probability():
    logps = torch.tensor([-0.5, -2.5, -3.0], requires_grad=True)
    loss = ce_question_loss(logps)
    assert loss.item() == pytest.approx(2.0)
    loss.backward()
    torch.testing.assert_close(logps.grad, torch.full_like(logps, -1 / 3))


def test_mixed_gt_retain_and_skip_compose_with_fixed_four_question_denominator():
    torch.manual_seed(12)
    layer = torch.nn.Linear(3, 4, bias=False, dtype=torch.float64)
    inputs = [torch.randn(length, 3, dtype=torch.float64) for length in (2, 5, 3, 4)]
    targets = [torch.arange(len(features)) % 4 for features in inputs]

    def answer_logps():
        return [torch.log_softmax(layer(features), dim=-1).gather(1, labels[:, None]).squeeze(1)
                for features, labels in zip(inputs, targets)]

    logps = answer_logps()
    refs = [logps[0].detach() + 0.1, logps[1].detach() - 0.2]
    batch_loss = (mixed_question_loss(logps[:2], refs, [True, False])
                  + ce_question_loss(logps[2]) + ce_question_loss(logps[3])) / 4
    batch_loss.backward()
    actual_grad = layer.weight.grad.clone()
    assert torch.isfinite(actual_grad).all() and actual_grad.abs().sum() > 0

    layer.zero_grad(set_to_none=True)
    expected_logps = answer_logps()
    per_answer = []
    for logp, ref, advantage in zip(expected_logps[:2], refs, [1.0, -1.0]):
        delta = ref - logp
        per_answer.append((-advantage * logp + 0.01 * (torch.exp(delta) - delta - 1)).mean())
    expected = ((per_answer[0] + per_answer[1]) / 2
                - expected_logps[2].mean() - expected_logps[3].mean() + 0) / 4
    expected.backward()
    torch.testing.assert_close(batch_loss, expected)
    torch.testing.assert_close(actual_grad, layer.weight.grad)


def test_loss_contracts_fail_fast_on_wrong_routes_or_empty_tokens():
    valid = torch.tensor([-1.0])
    with pytest.raises(ValueError, match="一中一错"):
        mixed_question_loss([valid, valid], [valid, valid], [True, True])
    with pytest.raises(ValueError, match="非空"):
        mixed_question_loss([valid, torch.empty(0)], [valid, torch.empty(0)], [True, False])
    with pytest.raises(ValueError, match="reference"):
        mixed_question_loss([valid, valid], [valid, torch.ones(2)], [True, False])
    with pytest.raises(ValueError, match="非空"):
        ce_question_loss(torch.empty(0))
