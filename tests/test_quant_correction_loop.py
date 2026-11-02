"""用真实 CPU 模型检查四题循环、逐题分流及原生断点恢复。"""

from __future__ import annotations

import copy
import json
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import torch

from tools import train_quant_grounding as loop
from tools.quant_correction import Completion, serialize_gt
from tools.quant_grounding_config import DEFAULTS


def make_rows(count=8, all_hit=False):
    scenarios = ("mixed", "all_wrong", "all_hit", "all_hit")
    return [{
        "id": f"query_{index}",
        "solution": [0.0125, 0.2005, 0.6789, 0.9995],
        "images": [f"visible_{index}.png", f"infrared_{index}.png"],
        "messages": [{"role": "user", "content": f"<image><image>Locate object {index}."}],
        "scenario": "all_hit" if all_hit else scenarios[index % len(scenarios)],
    } for index in range(count)]


def make_config(tmp_path, rows=None, **overrides):
    examples = make_rows() if rows is None else rows
    index = tmp_path / "train_index.jsonl"
    index.write_text("".join(json.dumps(row) + "\n" for row in examples), encoding="utf-8")
    return DEFAULTS | {
        "train_index": str(index), "media_root": str(tmp_path),
        "init_adapter": "fake_initial_adapter", "seed": 3,
        "learning_rate": 0.03, "keep_probability": 0.5,
        "stop_presentations": 16, "checkpoint_every": 8,
    } | overrides


class FakeRuntime:
    """只替换模型边界；损失、反传、梯度裁剪、AdamW 与持久化均实际执行。"""

    def __init__(self, config, mode, adapter_path=None):
        self.mode = mode
        self.model = torch.nn.Linear(3, 11, bias=False, dtype=torch.float64)
        self.reference = copy.deepcopy(self.model).requires_grad_(False)
        self.initial_weights = copy.deepcopy(self.model.state_dict())
        if adapter_path:
            self.model.load_state_dict(torch.load(
                Path(adapter_path) / "fake_adapter.pt", weights_only=True))
        self.active = False
        self.events = []
        self.question_weights = []
        self.prompts = []
        self.supervised = []
        self.generated = []

    def trainable_parameters(self):
        return self.model.parameters()

    @contextmanager
    def question(self, row):
        assert not self.active
        assert all(message["role"] != "assistant" for message in row["messages"])
        self.active = True
        self.prompts.append(copy.deepcopy(row["messages"]))
        self.question_weights.append(copy.deepcopy(self.model.state_dict()))
        self.events.append(("question", row["id"]))
        try:
            yield FakeQuestion(self, row)
        finally:
            self.active = False
            self.events.append(("end_question", row["id"]))

    def save_adapter(self, path):
        torch.save(self.model.state_dict(), Path(path) / "fake_adapter.pt")

    @staticmethod
    def target_tokens(text):
        return [ord(character) % 11 for character in text]

    @staticmethod
    def token_logps(model, row, token_ids):
        index = int(row["id"].split("_")[-1])
        features = torch.tensor([
            [(token + 1) / 11, (position + 1) / 10, (index + 1) / 8]
            for position, token in enumerate(token_ids)
        ], dtype=torch.float64)
        targets = torch.tensor(token_ids, dtype=torch.long)
        return torch.log_softmax(model(features), dim=-1).gather(1, targets[:, None]).squeeze(1)


class FakeQuestion:
    def __init__(self, runtime, row):
        self.runtime, self.row = runtime, row

    def generate(self):
        assert self.runtime.active
        assert self.runtime.mode == "correction"
        self.runtime.events.append(("generate", self.row["id"]))
        scenario = self.row["scenario"]
        hits = {"mixed": [True, False], "all_wrong": [False, False], "all_hit": [True, True]}[scenario]
        result = []
        for index, hit in enumerate(hits):
            # 三种全局 RNG 都影响原始词元，断点恢复必须逐一恢复它们。
            tokens = [index, random.randrange(11), int(np.random.randint(11)),
                      int(torch.randint(11, ()).item())]
            text = ('{"bbox_2d":[(13,201),(679,1000)]}' if hit
                    else '{"bbox_2d":[800,0,1000,100]}')
            result.append(Completion(text, tokens, "stop"))
        self.runtime.generated.append((self.row["id"], copy.deepcopy(result)))
        return result

    def logps(self, token_ids, reference=False):
        assert self.runtime.active
        self.runtime.events.append(("reference" if reference else "policy", self.row["id"]))
        model = self.runtime.reference if reference else self.runtime.model
        logps = FakeRuntime.token_logps(model, self.row, token_ids)
        return logps.detach() if reference else logps

    def supervised_logps(self, answer):
        assert self.runtime.active
        self.runtime.events.append(("supervised", self.row["id"]))
        self.runtime.supervised.append({
            "id": self.row["id"], "answer": answer,
            "prompt": copy.deepcopy(self.row["messages"]),
        })
        return FakeRuntime.token_logps(
            self.runtime.model, self.row, FakeRuntime.target_tokens(answer))


def factory_collector():
    runtimes = []

    def factory(config, mode, adapter_path=None):
        runtime = FakeRuntime(config, mode, adapter_path)
        runtimes.append(runtime)
        return runtime

    return factory, runtimes


def load_checkpoint(result):
    checkpoint = Path(result["checkpoint"])
    return (
        json.loads((checkpoint / "loop_state.json").read_text(encoding="utf-8")),
        torch.load(checkpoint / "fake_adapter.pt", weights_only=True),
        torch.load(checkpoint / "optimizer.pt", weights_only=False),
        torch.load(checkpoint / "rng.pt", weights_only=False),
    )


def assert_tree_equal(first, second):
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, np.ndarray):
        assert np.array_equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_tree_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert type(first) is type(second)
        assert len(first) == len(second)
        for left, right in zip(first, second):
            assert_tree_equal(left, right)
    else:
        assert first == second


def test_one_four_question_batch_covers_all_routes_and_matches_weighted_adam_step(tmp_path):
    config = make_config(tmp_path, make_rows(4), stop_presentations=4, checkpoint_every=4)
    factory, runtimes = factory_collector()
    result = loop.run(config, "correction", tmp_path / "run", runtime_factory=factory)
    assert result["presentations"] == 4
    assert result["batches"] == result["updates"] == 1
    records = loop.read_jsonl(tmp_path / "run/order.jsonl")
    assert {record["route"] for record in records} == {"mixed", "gt", "retain", "skip"}
    assert [record["presentation"] for record in records] == [1, 2, 3, 4]
    assert len({record["id"] for record in records}) == 4
    runtime = runtimes[0]
    for weights in runtime.question_weights:
        assert_tree_equal(weights, runtime.initial_weights)

    # 独立按记录重建四题损失，核对 /2 回答权重、/4 固定题数与真正 AdamW 更新。
    expected = copy.deepcopy(runtime.model)
    expected.load_state_dict(runtime.initial_weights)
    optimizer = torch.optim.AdamW(expected.parameters(), lr=config["learning_rate"], weight_decay=0)
    by_id = {row["id"]: row for row in make_rows(4)}
    total = torch.zeros((), dtype=torch.float64)
    for record in records:
        row = by_id[record["id"]]
        if record["route"] == "mixed":
            for generated, hit in zip(record["completions"], record["hits"]):
                logp = FakeRuntime.token_logps(expected, row, generated["token_ids"])
                ref = FakeRuntime.token_logps(runtime.reference, row, generated["token_ids"])
                delta = ref - logp
                total = total + ((-1 if hit else 1) * logp + config["beta"] * (
                    torch.exp(delta) - delta - 1)).mean() / 8
        elif record["route"] == "gt":
            tokens = FakeRuntime.target_tokens(serialize_gt(row))
            total = total - FakeRuntime.token_logps(expected, row, tokens).mean() / 4
        elif record["route"] == "retain":
            tokens = record["completions"][record["selected_index"]]["token_ids"]
            total = total - FakeRuntime.token_logps(expected, row, tokens).mean() / 4
    total.backward()
    torch.nn.utils.clip_grad_norm_(expected.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    torch.testing.assert_close(runtime.model.weight, expected.weight, rtol=1e-12, atol=1e-12)
    assert not torch.equal(runtime.model.weight, runtime.initial_weights["weight"])
    metric = loop.read_jsonl(tmp_path / "run/batches.jsonl")[0]
    assert metric["active_questions"] == 3
    assert metric["loss"] == pytest.approx(total.item())


def test_all_skip_batches_do_not_initialize_or_advance_adam(tmp_path, monkeypatch):
    config = make_config(tmp_path, make_rows(4, all_hit=True), keep_probability=0,
                         stop_presentations=12, checkpoint_every=4)
    actual_adam = torch.optim.AdamW
    step_calls = []

    class ObservedAdam(actual_adam):
        def step(self, closure=None):
            step_calls.append(True)
            return super().step(closure)

    monkeypatch.setattr(torch.optim, "AdamW", ObservedAdam)
    factory, runtimes = factory_collector()
    result = loop.run(config, "correction", tmp_path / "skip", runtime_factory=factory)
    assert result["presentations"] == 12 and result["batches"] == 3
    assert result["updates"] == 0 and step_calls == []
    _, weights, optimizer, _ = load_checkpoint(result)
    assert_tree_equal(weights, runtimes[0].initial_weights)
    assert optimizer["state"] == {}
    assert all(record["route"] == "skip" for record in loop.read_jsonl(tmp_path / "skip/order.jsonl"))
    assert all(record["grad_norm"] is None and record["active_questions"] == 0
               for record in loop.read_jsonl(tmp_path / "skip/batches.jsonl"))
    assert not any(event[0] in {"policy", "reference", "supervised"} for event in runtimes[0].events)


def test_correction_resume_exactly_restores_weights_optimizer_rng_pool_and_order(tmp_path):
    config = make_config(tmp_path, stop_presentations=32)
    full_factory, _ = factory_collector()
    full = loop.run(config, "correction", tmp_path / "full", runtime_factory=full_factory)
    part_factory, _ = factory_collector()
    part = loop.run(config | {"stop_presentations": 16}, "correction", tmp_path / "resumed",
                    runtime_factory=part_factory)
    # 模拟已写入、但未纳入检查点的尾部日志；恢复须回到已提交前缀。
    loop.append_jsonl(tmp_path / "resumed/order.jsonl", [{"presentation": 17, "id": "stale"}])
    loop.append_jsonl(tmp_path / "resumed/batches.jsonl", [{"batches": 5, "seconds": -1}])
    resumed_factory, _ = factory_collector()
    resumed = loop.run(config, "correction", tmp_path / "resumed", Path(part["checkpoint"]),
                       runtime_factory=resumed_factory)
    full_saved, full_weights, full_optimizer, full_rng = load_checkpoint(full)
    resumed_saved, resumed_weights, resumed_optimizer, resumed_rng = load_checkpoint(resumed)
    assert_tree_equal(full_weights, resumed_weights)
    assert_tree_equal(full_optimizer, resumed_optimizer)
    assert_tree_equal(full_rng, resumed_rng)
    assert full_saved == resumed_saved
    assert full_saved["sampler"]["mixed_pool"]
    assert full_saved["sampler"]["revisit_count"] > 0
    assert full_saved["sampler"]["observation_count"] == 32
    assert loop.read_jsonl(tmp_path / "full/order.jsonl") == loop.read_jsonl(tmp_path / "resumed/order.jsonl")
    full_metrics = loop.read_jsonl(tmp_path / "full/batches.jsonl")
    resumed_metrics = loop.read_jsonl(tmp_path / "resumed/batches.jsonl")
    assert [{k: v for k, v in item.items() if k != "seconds"} for item in full_metrics] == [
        {k: v for k, v in item.items() if k != "seconds"} for item in resumed_metrics]
    assert {item["route"] for item in loop.read_jsonl(tmp_path / "full/order.jsonl")} == {
        "mixed", "gt", "retain", "skip"}


def test_initial_checkpoint_recovers_interruption_before_first_batch_log(tmp_path):
    config = make_config(tmp_path, stop_presentations=8)
    factory, _ = factory_collector()
    full = loop.run(config, "correction", tmp_path / "full_initial", runtime_factory=factory)

    class InterruptedRuntime(FakeRuntime):
        @contextmanager
        def question(self, row):
            with super().question(row) as question:
                question.generate()  # 中断前消耗三种 RNG，恢复必须回到 checkpoint-0。
                raise RuntimeError("simulated interruption")
                yield question

    output = tmp_path / "interrupted"
    with pytest.raises(RuntimeError, match="simulated interruption"):
        loop.run(config, "correction", output, runtime_factory=InterruptedRuntime)
    checkpoint = output / "checkpoint-0"
    assert (checkpoint / "loop_state.json").is_file()
    assert not (output / "order.jsonl").exists()
    assert not (output / "completed.json").exists()
    factory, _ = factory_collector()
    resumed = loop.run(config, "correction", output, checkpoint, runtime_factory=factory)
    for first, second in zip(load_checkpoint(full), load_checkpoint(resumed)):
        assert_tree_equal(first, second)
    assert loop.read_jsonl(tmp_path / "full_initial/order.jsonl") == loop.read_jsonl(output / "order.jsonl")


def test_sft_replays_actual_adaptive_order_and_resumes_exactly(tmp_path):
    config = make_config(tmp_path, stop_presentations=24)
    correction_factory, _ = factory_collector()
    loop.run(config, "correction", tmp_path / "correction", runtime_factory=correction_factory)
    sequence = tmp_path / "correction/order.jsonl"
    sft_config = config | {"sequence_path": str(sequence)}
    full_factory, full_runtimes = factory_collector()
    full = loop.run(sft_config, "sft", tmp_path / "sft_full", runtime_factory=full_factory)
    part_factory, _ = factory_collector()
    part = loop.run(sft_config | {"stop_presentations": 8}, "sft", tmp_path / "sft_resume",
                    runtime_factory=part_factory)
    resumed_factory, _ = factory_collector()
    resumed = loop.run(sft_config, "sft", tmp_path / "sft_resume", Path(part["checkpoint"]),
                       runtime_factory=resumed_factory)
    correction_ids = [item["id"] for item in loop.read_jsonl(sequence)]
    full_order = loop.read_jsonl(tmp_path / "sft_full/order.jsonl")
    assert [item["id"] for item in full_order] == correction_ids
    assert [item["id"] for item in full_runtimes[0].supervised] == correction_ids
    assert all(item["route"] == "gt" for item in full_order)
    assert not full_runtimes[0].generated
    assert full_order == loop.read_jsonl(tmp_path / "sft_resume/order.jsonl")
    for first, second in zip(load_checkpoint(full), load_checkpoint(resumed)):
        assert_tree_equal(first, second)
    assert load_checkpoint(full)[0]["sampler"] is None


def test_generation_compute_switch_can_resume_original_checkpoint_without_resetting_state(tmp_path):
    config = make_config(tmp_path, stop_presentations=16)
    # 模拟首段旧原生检查点：配置中尚无提示复用字段。
    config.pop("generation_prefill_reuse")
    full_factory, _ = factory_collector()
    full = loop.run(config, "correction", tmp_path / "full_switch", runtime_factory=full_factory)
    part_factory, _ = factory_collector()
    part = loop.run(config | {"stop_presentations": 8}, "correction", tmp_path / "switch",
                    runtime_factory=part_factory)
    resumed_factory, _ = factory_collector()
    resumed = loop.run(config | {"generation_prefill_reuse": True}, "correction", tmp_path / "switch",
                       Path(part["checkpoint"]), runtime_factory=resumed_factory)
    for first, second in zip(load_checkpoint(full)[1:], load_checkpoint(resumed)[1:]):
        assert_tree_equal(first, second)
    full_state, resumed_state = load_checkpoint(full)[0], load_checkpoint(resumed)[0]
    for key in ("presentations", "updates", "batches", "sampler", "retention_rng"):
        assert_tree_equal(full_state[key], resumed_state[key])
    assert loop.read_jsonl(tmp_path / "full_switch/order.jsonl") == loop.read_jsonl(tmp_path / "switch/order.jsonl")


def test_gt_is_only_a_supervised_target_and_original_prompts_stay_unchanged(tmp_path):
    examples = make_rows(4)
    config = make_config(tmp_path, examples, stop_presentations=8, checkpoint_every=4)
    original = Path(config["train_index"]).read_text(encoding="utf-8")
    factory, runtimes = factory_collector()
    loop.run(config, "correction", tmp_path / "boundary", runtime_factory=factory)
    assert Path(config["train_index"]).read_text(encoding="utf-8") == original
    runtime = runtimes[0]
    assert runtime.generated and runtime.supervised
    for prompt in runtime.prompts:
        assert all(message["role"] == "user" for message in prompt)
        visible = json.dumps(prompt)
        assert "solution" not in visible and "bbox_2d" not in visible
    by_id = {row["id"]: row for row in examples}
    for call in runtime.supervised:
        assert call["answer"] == serialize_gt(by_id[call["id"]])
        assert call["prompt"] == by_id[call["id"]]["messages"]


def test_assistant_in_input_is_rejected_before_runtime_or_training(tmp_path):
    examples = make_rows(4)
    examples[0]["messages"].append({"role": "assistant", "content": serialize_gt(examples[0])})
    config = make_config(tmp_path, examples)
    factory, runtimes = factory_collector()
    with pytest.raises(ValueError, match="标准答案不得进入生成提示"):
        loop.run(config, "correction", tmp_path / "invalid", runtime_factory=factory)
    assert runtimes == []


def test_sft_rejects_a_short_or_noncontinuous_actual_sequence(tmp_path):
    config = make_config(tmp_path, stop_presentations=8)
    sequence = tmp_path / "sequence.jsonl"
    loop.append_jsonl(sequence, [{"presentation": index + 1, "id": f"query_{index}"} for index in range(4)])
    factory, runtimes = factory_collector()
    with pytest.raises(ValueError, match="尚未覆盖"):
        loop.run(config | {"sequence_path": str(sequence)}, "sft", tmp_path / "short", runtime_factory=factory)
    loop.append_jsonl(sequence, [{"presentation": 99, "id": "query_0"}] * 4)
    with pytest.raises(ValueError, match="非连续"):
        loop.run(config | {"sequence_path": str(sequence)}, "sft", tmp_path / "discontinuous", runtime_factory=factory)
    assert runtimes == []
