"""仅覆盖 CPU 可证实的参数和数据边界；GPU 链路由 --probe 实测。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.quant_grounding_config import (configure_unrestricted_hf_sampling, load_config,
                                          prepare_dataset, swift_arguments)
from tools.train_quant_grounding import (audit_answers_per_optimizer_step, audit_base_config,
                                        budget_stop_signals,
                                        audit_gradients, audit_optimizer,
                                        configure_deterministic_training,
                                        install_language_gradient_checkpointing, install_runtime_probe,
                                        install_dapo_loss_normalizer, install_selected_logits_audit,
                                        normalize_runtime_optimization_config,
                                        sft_loss_scaling_contract_holds, storage_bytes)


def config_file(tmp_path: Path, arm="A", precision="nf4") -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"arm": arm, "precision": precision, "model": "/models/public-qwen",
                                "dataset": "/data/train.jsonl", "data_root": "/data/media",
                                "output_dir": "/runs/a", "max_gpu_hours": 8}), encoding="utf-8")
    return path


@pytest.mark.parametrize("arm,reward,length,thinking", [
    ("A", "quant_bbox_contrast", "64", "false"),
    ("B", "quant_bbox_acc05", "64", "false"),
    ("D", "quant_bbox_contrast_thinking", "384", "true"),
])
def test_rl_arms_keep_g8_separate_from_hf_cap(tmp_path, arm, reward, length, thinking):
    config = load_config(config_file(tmp_path, arm))
    args = swift_arguments(config, Path("prepared.jsonl"))
    pairs = dict(zip(args[::2], args[1::2]))
    assert pairs["--num_generations"] == pairs["--generation_batch_size"] == "8"
    assert pairs["--gradient_accumulation_steps"] == "32"
    assert pairs["--reward_funcs"] == reward
    assert pairs["--max_completion_length"] == length
    assert pairs["--enable_thinking"] == thinking
    assert pairs["--loss_type"] == "dapo"
    assert pairs["--add_version"] == "false"
    assert pairs["--dataset_shuffle"] == pairs["--train_dataloader_shuffle"] == "false"
    assert pairs["--scale_rewards"] == "none"
    assert pairs["--dynamic_sample"] == "false"
    assert "--adapters" not in args and "--ref_adapters" not in args and "--ref_model" not in args
    assert config.get("hf_max_batch_size", 1) == 1


def test_bf16_and_sft_control(tmp_path):
    config = load_config(config_file(tmp_path, arm="C", precision="bf16"))
    args = swift_arguments(config, Path("prepared.jsonl"))
    assert "--rlhf_type" not in args and "--reward_funcs" not in args
    assert dict(zip(args[::2], args[1::2]))["--enable_thinking"] == "false"
    assert dict(zip(args[::2], args[1::2]))["--dataset_shuffle"] == "false"
    assert dict(zip(args[::2], args[1::2]))["--train_dataloader_shuffle"] == "false"
    assert "--quant_method" not in args and "--quant_bits" not in args
    assert "--freeze_vit" in args and "--freeze_aligner" in args
    nf4 = swift_arguments(load_config(config_file(tmp_path, precision="nf4")), Path("prepared.jsonl"))
    pairs = dict(zip(nf4[::2], nf4[1::2]))
    assert pairs["--quant_method"] == "bnb"
    assert pairs["--quant_bits"] == "4"
    assert pairs["--bnb_4bit_quant_type"] == "nf4"
    assert pairs["--bnb_4bit_use_double_quant"] == "true"
    assert pairs["--bnb_4bit_compute_dtype"] == "bfloat16"


def test_repaired_d_uses_distinct_reward_and_shared_prompt(tmp_path):
    from tools.quant_grounding_thinking import apply_bounded_thinking_prompt
    path = config_file(tmp_path, arm="D")
    config = json.loads(path.read_text(encoding="utf-8"))
    config.update(bounded_thinking=True, format_policy="single_box")
    path.write_text(json.dumps(config), encoding="utf-8")
    loaded = load_config(path)
    args = swift_arguments(loaded, Path("prepared.jsonl"))
    assert args[args.index("--reward_funcs") + 1] == "quant_bbox_contrast_thinking_repaired"
    pairs = dict(zip(args[::2], args[1::2]))
    assert pairs['--rollout_importance_sampling_mode'] == 'token_truncate'
    assert pairs['--rollout_importance_sampling_threshold'] == '2.0'
    assert pairs['--log_rollout_offpolicy_metrics'] == 'true'
    assert loaded['sampling_replay_tokens'] == 64
    assert loaded['sampling_replay_atol'] == 0.05
    original = [{"role": "user", "content": "<image> locate the chair"}]
    prepared = apply_bounded_thinking_prompt(original)
    assert prepared[0]["role"] == "system"
    assert prepared[1:] == original
    assert original[0]["role"] == "user"
    config["arm"] = "A"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="仅用于"):
        load_config(path)


@pytest.mark.parametrize("thought_limit,completion", [(256, 322), (1024, 1090)])
def test_repaired_d_derives_token_window_and_can_disable_time_stop(tmp_path, thought_limit, completion):
    path = config_file(tmp_path, arm="D")
    config = json.loads(path.read_text(encoding="utf-8"))
    config.update(bounded_thinking=True, format_policy="single_box", thought_limit=thought_limit,
                  answer_limit=64, max_completion_length=384, max_gpu_hours=None)
    path.write_text(json.dumps(config), encoding="utf-8")
    loaded = load_config(path)
    assert loaded["max_completion_length"] == completion
    args = swift_arguments(loaded, Path("prepared.jsonl"))
    assert args[args.index("--max_completion_length") + 1] == str(completion)
    assert budget_stop_signals(100000, None, 1800) == (False, False)


@pytest.mark.parametrize("arm,answers", [("A", 32), ("B", 32), ("C", 8), ("D", 32)])
@pytest.mark.parametrize("micro_batch", [1, 2])
def test_training_micro_batch_keeps_answers_per_update(tmp_path, arm, answers, micro_batch):
    config = load_config(config_file(tmp_path, arm=arm))
    config["train_micro_batch_size"] = micro_batch
    if micro_batch == 2:
        config["runtime_optimization"] = True
    pairs = dict(zip(swift_arguments(config, Path("prepared.jsonl"))[::2],
                     swift_arguments(config, Path("prepared.jsonl"))[1::2]))
    assert int(pairs["--per_device_train_batch_size"]) * int(pairs["--gradient_accumulation_steps"]) == answers
    if arm != "C":
        assert pairs["--num_generations"] == pairs["--generation_batch_size"] == "8"


@pytest.mark.parametrize("name,value", [("train_micro_batch_size", 3),
                                        ("gradient_checkpointing_every_n_layers", 3),
                                        ("matched_query_exposures", 5)])
def test_optimization_config_rejects_unsupported_values(tmp_path, name, value):
    path = config_file(tmp_path)
    config = json.loads(path.read_text(encoding="utf-8"))
    config[name] = value
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_runtime_optimization_defaults_legacy_and_explicit_opt_in():
    config = {}
    assert normalize_runtime_optimization_config(config) is False
    assert config["runtime_optimization"] is False

    optimized = {"runtime_optimization": True, "train_micro_batch_size": 2}
    assert normalize_runtime_optimization_config(optimized) is True
    assert optimized["runtime_optimization"] is True
    with pytest.raises(ValueError, match="runtime_optimization=false"):
        normalize_runtime_optimization_config({"runtime_optimization": False,
                                               "train_micro_batch_size": 2})
    with pytest.raises(ValueError, match="runtime_optimization=false"):
        normalize_runtime_optimization_config({"runtime_optimization": False,
                                               "gradient_checkpointing_every_n_layers": 2})
    with pytest.raises(ValueError, match="布尔值"):
        normalize_runtime_optimization_config({"runtime_optimization": "false"})


def test_deterministic_training_is_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    cudnn = SimpleNamespace(deterministic=False, benchmark=True)
    calls = []
    fake_torch = SimpleNamespace(use_deterministic_algorithms=calls.append,
                                 backends=SimpleNamespace(cudnn=cudnn))

    default = {}
    assert configure_deterministic_training(default, fake_torch) is False
    assert default["deterministic_training"] is False
    assert calls == [] and cudnn.deterministic is False and cudnn.benchmark is True
    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ

    explicit = {"deterministic_training": True}
    assert configure_deterministic_training(explicit, fake_torch) is True
    assert calls == [True] and cudnn.deterministic is True and cudnn.benchmark is False
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    with pytest.raises(ValueError, match="布尔值"):
        configure_deterministic_training({"deterministic_training": "true"})


def test_optimizer_update_answer_gate_stays_32_for_rl_and_8_for_sft():
    assert audit_answers_per_optimizer_step("A", 1, 32)["answers_per_optimizer_step"] == 32
    assert audit_answers_per_optimizer_step("D", 2, 16)["answers_per_optimizer_step"] == 32
    assert audit_answers_per_optimizer_step("C", 2, 4)["answers_per_optimizer_step"] == 8
    with pytest.raises(RuntimeError, match="应保留 32 个回答"):
        audit_answers_per_optimizer_step("A", 2, 15)


@pytest.mark.parametrize("elapsed,limit,longest_step,expected", [
    (9686.85, 2.75 * 3600, 1275, True),
    (8000, 2.75 * 3600, 1275, False),
    (2.75 * 3600, 2.75 * 3600, None, True),
    (10000, 2.75 * 3600, None, True),
    (1000, 2.75 * 3600, None, False),
])
def test_budget_stop_requests_save_and_stop_at_step_boundary(elapsed, limit, longest_step, expected):
    should_save, should_stop = budget_stop_signals(elapsed, limit, longest_step)
    assert should_save is expected
    assert should_stop is expected


def test_matched_exposure_checkpoint_survives_rolling_limit(tmp_path):
    path = config_file(tmp_path, arm="C")
    config = json.loads(path.read_text(encoding="utf-8"))
    config.update({"matched_query_exposures": 120, "save_total_limit": 3})
    path.write_text(json.dumps(config), encoding="utf-8")
    loaded = load_config(path)
    pairs = dict(zip(swift_arguments(loaded, Path("prepared.jsonl"))[::2],
                     swift_arguments(loaded, Path("prepared.jsonl"))[1::2]))
    assert loaded["matched_query_exposures"] == 120
    assert pairs["--save_total_limit"] == "4"


def test_repeated_checkpoint_enable_preserves_language_mask_and_frozen_vision():
    import torch

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gradient_checkpointing = False

    class Base(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = torch.nn.Module()
            self.language_model.layers = torch.nn.ModuleList([Layer() for _ in range(5)])
            self.visual = Layer()
            self.calls = []

        def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
            self.calls.append(gradient_checkpointing_kwargs)
            for module in self.modules():
                if hasattr(module, "gradient_checkpointing"):
                    module.gradient_checkpointing = True

    class Outer(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.model = base

        def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    base = Base()
    model = Outer(base)
    audit, restore = install_language_gradient_checkpointing(model, base, 2)
    base.gradient_checkpointing_enable({"use_reentrant": False})
    model.gradient_checkpointing_enable({"use_reentrant": False})
    assert [layer.gradient_checkpointing for layer in base.language_model.layers] == [True, False, True, False, True]
    assert base.visual.gradient_checkpointing is True
    assert base.calls == [{"use_reentrant": False}] * 2
    assert audit["enabled_layer_indices"] == [0, 2, 4]
    assert audit["mask_applied"] is True
    assert audit["reenable_calls"] >= 2
    restore()


def test_legacy_gradient_checkpointing_leaves_swift_enable_unwrapped():
    import torch

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gradient_checkpointing = False

    class Base(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = torch.nn.Module()
            self.language_model.layers = torch.nn.ModuleList([Layer() for _ in range(3)])
            self.visual = Layer()

        def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
            for module in self.modules():
                if hasattr(module, "gradient_checkpointing"):
                    module.gradient_checkpointing = True

    class Outer(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.model = base

        def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    base = Base()
    model = Outer(base)
    audit, restore = install_language_gradient_checkpointing(model, base, 1, mask_language_layers=False)
    base.gradient_checkpointing_enable({"use_reentrant": False})
    assert [layer.gradient_checkpointing for layer in base.language_model.layers] == [True, True, True]
    assert base.visual.gradient_checkpointing is True
    assert audit["mask_applied"] is False
    assert audit["reenable_calls"] == 0
    restore()


def test_aggregated_gradient_audit_catches_nan_and_frozen_gradient():
    import torch

    model = torch.nn.Module()
    model.lora = torch.nn.Parameter(torch.tensor([1., 2.]))
    model.frozen = torch.nn.Parameter(torch.tensor([1.]), requires_grad=False)
    model.lora.grad = torch.zeros_like(model.lora)
    assert audit_gradients(model, torch) is False
    model.lora.grad[0] = 1
    assert audit_gradients(model, torch) is True
    model.lora.grad[0] = float("nan")
    with pytest.raises(RuntimeError, match="NaN/Inf"):
        audit_gradients(model, torch)
    model.lora.grad[0] = 1
    model.frozen.grad = torch.ones_like(model.frozen)
    with pytest.raises(RuntimeError, match="冻结底座"):
        audit_gradients(model, torch)


def test_legacy_gradient_audit_uses_per_tensor_item_checks():
    import torch

    model = torch.nn.Module()
    model.first_lora = torch.nn.Parameter(torch.ones(2))
    model.second_lora = torch.nn.Parameter(torch.ones(2))
    model.first_lora.grad = torch.zeros_like(model.first_lora)
    model.second_lora.grad = torch.tensor([0.0, 1.0])
    calls = {"finite_item": 0, "count_nonzero_item": 0}

    class ScalarCheck:
        def __init__(self, value, name):
            self.value = value
            self.name = name

        def all(self):
            return self

        def item(self):
            calls[self.name] += 1
            return self.value

    legacy_torch = SimpleNamespace(
        isfinite=lambda _gradient: ScalarCheck(True, "finite_item"),
        count_nonzero=lambda gradient: ScalarCheck(int(torch.count_nonzero(gradient)), "count_nonzero_item"),
    )

    assert audit_gradients(model, legacy_torch, runtime_optimization=False) is True
    assert calls == {"finite_item": 2, "count_nonzero_item": 2}


def test_mb2_dapo_normalizer_preserves_same_32_answer_gradient_and_excludes_sft():
    import torch

    shared_normalizer = torch.tensor(320.0)

    class Trainer:
        def __init__(self, micro_batch_size):
            self.model = SimpleNamespace(training=True)
            self.micro_batch_size = micro_batch_size

        def _prepare_batch_inputs(self, samples):
            return [{"grpo_batch": SimpleNamespace(num_items_in_batch=shared_normalizer),
                     "numerator": sum(samples[start:start + self.micro_batch_size])}
                    for start in range(0, len(samples), self.micro_batch_size)]

    samples = [float(index + 1) for index in range(32)]
    trainer = Trainer(2)
    audit, restore = install_dapo_loss_normalizer(
        trainer, arm="A", train_micro_batch_size=2, runtime_optimization=True)
    batches = trainer._prepare_batch_inputs(samples)
    assert len(batches) == 16
    assert audit == {"applied": True, "factor": 2, "arm": "A"}
    assert all(batch["grpo_batch"].num_items_in_batch.item() == 640 for batch in batches)
    assert shared_normalizer.item() == 320  # Swift reuses this scalar across its prepared batches.

    mb1_weight = torch.nn.Parameter(torch.tensor(1.0))
    mb2_weight = torch.nn.Parameter(torch.tensor(1.0))
    mb1_loss = sum(mb1_weight * value / shared_normalizer / 32 for value in samples)
    mb2_loss = sum(mb2_weight * batch["numerator"] / batch["grpo_batch"].num_items_in_batch / 16
                   for batch in batches)
    mb1_loss.backward()
    mb2_loss.backward()
    assert mb2_loss.item() == pytest.approx(mb1_loss.item())
    assert mb2_weight.grad.item() == pytest.approx(mb1_weight.grad.item())
    restore()

    sft = Trainer(2)
    sft_audit, _ = install_dapo_loss_normalizer(
        sft, arm="C", train_micro_batch_size=2, runtime_optimization=True)
    assert sft_audit == {"applied": False, "factor": 1, "arm": "C"}
    assert all(batch["grpo_batch"].num_items_in_batch.item() == 320
               for batch in sft._prepare_batch_inputs(samples))


def test_c_microbatch_two_requires_transformers_loss_item_normalization():
    assert sft_loss_scaling_contract_holds("C", 1, False)
    assert sft_loss_scaling_contract_holds("C", 2, True)
    assert not sft_loss_scaling_contract_holds("C", 2, False)
    assert sft_loss_scaling_contract_holds("A", 2, False)


@pytest.mark.parametrize("micro_batch_size", [1, 2, 8])
def test_swift_grpo_compute_loss_through_transformers_training_step_is_equivalent(micro_batch_size):
    import torch
    from accelerate.utils import DistributedType
    from transformers import Trainer

    samples = [float(index + 1) for index in range(32)]
    shared_normalizer = torch.tensor(320.0)

    def run_update(micro_batch_size, gas, runtime_optimization):
        model = torch.nn.Module()
        model.weight = torch.nn.Parameter(torch.tensor(1.0))
        class MiniDapoTrainer:
            model_accepts_loss_kwargs = False
            compute_loss_func = None
            use_apex = False

            def __init__(self):
                self.current_gradient_accumulation_steps = gas
                self.model = model
                self.optimizer = None
                self.state = SimpleNamespace(global_step=0)
                self.args = SimpleNamespace(optim="adamw_torch", n_gpu=1,
                                            torch_empty_cache_steps=None, device=torch.device("cpu"),
                                            gradient_accumulation_steps=gas)
                self.accelerator = SimpleNamespace(
                    distributed_type=DistributedType.NO,
                    backward=lambda loss, **_kwargs: loss.backward(),
                )
                self._prepare_context_parallel_inputs = lambda _model, inputs: (nullcontext, inputs)
                self.compute_loss_context_manager = nullcontext
                self._prepare_inputs = lambda inputs: inputs

            def compute_loss(self, _model, inputs, **_kwargs):
                # Mirrors Swift GRPO's DAPO reduction: per-token numerator divided by
                # GRPOBatch.num_items_in_batch; Trainer applies gradient-accumulation scaling.
                return model.weight * inputs["model_inputs"] / inputs["grpo_batch"].num_items_in_batch

        trainer = MiniDapoTrainer()

        def prepare(_samples):
            return [{"model_inputs": sum(samples[start:start + micro_batch_size]),
                     "grpo_batch": SimpleNamespace(num_items_in_batch=shared_normalizer)}
                    for start in range(0, len(samples), micro_batch_size)]

        trainer._prepare_batch_inputs = prepare
        _, restore = install_dapo_loss_normalizer(
            trainer, arm="A", train_micro_batch_size=micro_batch_size,
            runtime_optimization=runtime_optimization)
        batches = trainer._prepare_batch_inputs(samples)
        losses = [Trainer.training_step(trainer, model, batch).item() for batch in batches]
        gradient = model.weight.grad.item()
        restore()
        return sum(losses), gradient

    baseline_loss, baseline_gradient = run_update(1, 32, False)
    candidate_loss, candidate_gradient = run_update(micro_batch_size, 32 // micro_batch_size, True)
    assert candidate_loss == pytest.approx(baseline_loss)
    assert candidate_gradient == pytest.approx(baseline_gradient)


def test_real_swift_hf_sampling_clears_model_top_k():
    swift_engine = pytest.importorskip("swift.infer_engine.transformers_engine")
    from swift.infer_engine.protocol import RequestConfig
    from transformers import GenerationConfig, GPT2Config, GPT2LMHeadModel

    model = GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=1, n_embd=16, vocab_size=100,
                                      bos_token_id=0, eos_token_id=1, pad_token_id=0))
    model.generation_config = GenerationConfig(do_sample=True, top_k=20, top_p=0.8, temperature=0.7)
    engine = object.__new__(swift_engine.TransformersEngine)
    engine.generation_config = model.generation_config
    engine.processor = SimpleNamespace(eos_token_id=1, pad_token_id=0)
    request = RequestConfig(max_tokens=64, temperature=1, top_p=1, top_k=-1)
    assert engine._prepare_generation_config(request).top_k == 20
    assert configure_unrestricted_hf_sampling(engine, request) == {
        "temperature": 1, "top_p": 1, "top_k": 0, "do_sample": True,
    }
    final, _ = model._prepare_generation_config(engine._prepare_generation_config(request))
    assert final.top_k == 0 and final.do_sample is True


def test_peft_outer_hook_observes_grpo_logits_and_base_hook_misses_them():
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import GPT2Config, GPT2LMHeadModel

    base = GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=1, n_embd=16, vocab_size=100,
                                     bos_token_id=0, eos_token_id=1, pad_token_id=0))
    model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=["c_attn"],
                                            task_type=TaskType.CAUSAL_LM))
    base_seen = []
    base_hook = model.get_base_model().register_forward_pre_hook(
        lambda _module, _args, kwargs: base_seen.append(kwargs.get("logits_to_keep")), with_kwargs=True)

    class Trainer:
        def __init__(self):
            self.model = model

        def _get_logps_via_local_forward(self, input_ids):
            return self.model(input_ids=input_ids, logits_to_keep=25)

    trainer = Trainer()
    forward, logps, restore = install_selected_logits_audit(trainer, torch)
    input_ids = torch.randint(0, 100, (1, 32))
    assert trainer._get_logps_via_local_forward(input_ids).logits.shape[-2] == 25

    def replace_ids_with_embeds(_module, args, kwargs):
        kwargs = kwargs.copy()
        kwargs["inputs_embeds"] = model.get_input_embeddings()(kwargs.pop("input_ids"))
        return args, kwargs

    multimodal_hook = model.register_forward_pre_hook(replace_ids_with_embeds, with_kwargs=True)
    with torch.no_grad():
        assert trainer._get_logps_via_local_forward(input_ids).logits.shape[-2] == 25
        model(input_ids=input_ids, logits_to_keep=1)
    assert base_seen == []
    assert forward["calls"] == 3 and forward["minimum"] == 1 and forward["maximum"] == 25
    assert logps["with_grad"]["calls"] == logps["without_grad"]["calls"] == 1
    assert logps["with_grad"]["first"] == {
        "input_tokens": 32, "requested_logits_tokens": 25, "output_logits_tokens": 25}
    assert logps["without_grad"]["first"] == logps["with_grad"]["first"]
    assert logps["with_grad"]["output_lengths_match"]
    multimodal_hook.remove()
    restore()
    base_hook.remove()
    assert trainer._get_logps_via_local_forward(input_ids).logits.shape[-2] == 25
    assert forward["calls"] == 3


def test_shared_warmup_uses_same_peft_reference(tmp_path):
    config = load_config(config_file(tmp_path))
    config["adapters"] = "/warmup/adapter"
    args = swift_arguments(config, Path("prepared.jsonl"))
    pairs = dict(zip(args[::2], args[1::2]))
    assert pairs["--adapters"] == pairs["--ref_adapters"] == "/warmup/adapter"
    assert "--ref_model" not in args
    assert dict(zip(swift_arguments(config, Path("prepared.jsonl"), max_steps=2)[::2],
                    swift_arguments(config, Path("prepared.jsonl"), max_steps=2)[1::2]))["--max_steps"] == "2"
    assert dict(zip(swift_arguments(config, Path("prepared.jsonl"))[::2],
                    swift_arguments(config, Path("prepared.jsonl"))[1::2]))["--max_steps"] == "1000000"


def test_hidden_metadata_never_enters_sft_prompt(tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    for name in ("rgb.png", "ir.png", "depth.png"):
        (media / name).write_bytes(b"image")
    row = {"id": "city-1", "messages": [{"role": "user", "content": "<image><image><image> target?"}],
           "images": ["rgb.png", "ir.png", "depth.png"], "solution": [0.1012, 0.2, 0.751, 0.899],
           "negative_boxes": [[0.2, 0.2, 0.3, 0.3]], "negative_kind": ["wrong_object"],
           "source": "city", "group_id": "g1", "label_source": "manual"}
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    rl_path = tmp_path / "rl.jsonl"
    sft_path = tmp_path / "sft.jsonl"
    assert prepare_dataset(source, rl_path, media, "A") == 1
    assert prepare_dataset(source, sft_path, media, "C") == 1
    rl = json.loads(rl_path.read_text())
    sft = json.loads(sft_path.read_text())
    assert rl["solution"] == row["solution"]
    assert rl["grounding_negative_boxes"] == row["negative_boxes"]
    assert rl["grounding_negative_kind"] == row["negative_kind"]
    assert "negative_boxes" not in rl and "negative_kind" not in rl
    assert len(rl["images"]) == 3 and all(Path(image).is_absolute() for image in rl["images"])
    assert "solution" not in sft and "negative_boxes" not in sft
    assert sft["messages"][0] == row["messages"][0]
    assert json.loads(sft["messages"][1]["content"]) == {"bbox_2d": [101, 200, 751, 899]}


def test_swift_template_inputs_accept_grounding_negative_metadata():
    pytest.importorskip("swift.template.template_inputs")
    from swift.template.template_inputs import TemplateInputs

    row = {"messages": [{"role": "user", "content": "target?"}],
           "solution": [0, 0, 1, 1],
           "grounding_negative_boxes": [[0.2, 0.2, 0.3, 0.3]],
           "grounding_negative_kind": ["wrong_object"]}
    parsed = TemplateInputs.from_dict(row)
    assert parsed.negative == []
    assert parsed.chosen.extra_kwargs["grounding_negative_boxes"] == row["grounding_negative_boxes"]
    assert parsed.chosen.extra_kwargs["grounding_negative_kind"] == row["grounding_negative_kind"]


def test_probe_requires_real_three_image_records(tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    for name in ("rgb.png", "ir.png", "depth.png"):
        (media / name).write_bytes(b"image")
    source = tmp_path / "source.jsonl"
    rows = []
    for i in range(8):
        rows.append({"id": f"tri-{i}", "messages": [{"role": "user", "content": "<image>" * 3}],
                     "images": ["rgb.png", "ir.png", "depth.png"], "solution": [0, 0, 1, 1]})
    rows.insert(0, {"id": "rgb-only", "messages": [{"role": "user", "content": "<image>"}],
                    "images": ["rgb.png"], "solution": [0, 0, 1, 1]})
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    target = tmp_path / "probe.jsonl"
    assert prepare_dataset(source, target, media, "A", probe=True) == 8
    assert all(len(json.loads(line)["images"]) == 3 for line in target.read_text().splitlines())


def test_sft_bbox_uses_half_up_and_rejects_quantized_empty_box(tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    (media / "rgb.png").write_bytes(b"image")
    source = tmp_path / "source.jsonl"
    row = {"id": "half-up", "messages": [{"role": "user", "content": "<image>"}],
           "images": ["rgb.png"], "solution": [0.1005, 0.2, 0.7515, 0.9]}
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    target = tmp_path / "sft.jsonl"
    prepare_dataset(source, target, media, "C")
    answer = json.loads(json.loads(target.read_text())["messages"][-1]["content"])
    assert answer["bbox_2d"] == [101, 200, 752, 900]
    row["solution"] = [0.1001, 0.2, 0.1002, 0.9]
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="collapsed"):
        prepare_dataset(source, target, media, "C")


def test_dry_run_needs_no_swift_or_media(tmp_path):
    config = config_file(tmp_path)
    script = Path(__file__).resolve().parents[1] / "tools" / "train_quant_grounding.py"
    result = subprocess.run([sys.executable, str(script), "--config", str(config), "--dry-run"],
                            capture_output=True, text=True, check=True)
    output = json.loads(result.stdout)
    assert output["swift_arguments"][0:2] == ["--model", "/models/public-qwen"]
    assert not (tmp_path / "prepared_train.jsonl").exists()


@pytest.mark.parametrize("placement", ["model", "text_config"])
def test_nf4_source_rejects_prequantized_model(placement):
    class ConfigLoader:
        @staticmethod
        def from_pretrained(_name):
            return SimpleNamespace(
                model_type="qwen3_5",
                quantization_config={"quant_method": "fp8"} if placement == "model" else None,
                text_config=SimpleNamespace(
                    quantization_config={"quant_method": "fp8"} if placement == "text_config" else None),
            )

    with pytest.raises(RuntimeError, match="已有 quantization_config"):
        audit_base_config("/models/official-fp8", ConfigLoader)


def test_optimizer_audit_rejects_frozen_parameter_or_gradient():
    import torch

    model = torch.nn.Module()
    model.language_model = torch.nn.Module()
    model.language_model.lora_A = torch.nn.Parameter(torch.ones(2))
    model.base = torch.nn.Parameter(torch.ones(2), requires_grad=False)
    model.ref_adapter = torch.nn.Parameter(torch.ones(2), requires_grad=False)
    optimizer = torch.optim.SGD([model.language_model.lora_A], lr=0.1)
    assert audit_optimizer(model, optimizer)["optimizer_parameter_count"] == 1
    assert storage_bytes(model.parameters()) == 3 * 2 * 4

    model.base.grad = torch.ones_like(model.base)
    with pytest.raises(RuntimeError, match="冻结底座"):
        audit_optimizer(model, optimizer)
    model.base.grad = None
    bad_optimizer = torch.optim.SGD([model.language_model.lora_A, model.base], lr=0.1)
    with pytest.raises(RuntimeError, match="优化器参数组"):
        audit_optimizer(model, bad_optimizer)


@pytest.mark.parametrize("arm,budget", [("A", 64), ("D", 384)])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("runtime_optimization", [True, False])
def test_runtime_probe_observes_swift_method_boundaries(arm, budget, batch_size, runtime_optimization):
    import torch

    calls = []

    class Engine:
        @staticmethod
        def _get_num_tokens(inputs):
            return inputs["input_ids"].shape[-1]

        @staticmethod
        def set_default_max_tokens(request_config, inputs):
            calls.append(("budget", request_config.max_tokens, inputs["input_ids"].shape[-1]))

    engine = Engine()
    inputs = {"input_ids": torch.zeros((batch_size, 3501), dtype=torch.long),
              "image_grid_thw": torch.tensor([[1, 50, 90]] * (3 * batch_size)),
              "pixel_values": torch.zeros((13500 * batch_size, 1176))}

    class Trainer:
        def __init__(self):
            self.engine = engine
            self.accelerator = SimpleNamespace(backward=self.backward)
            self.template = SimpleNamespace(encode=lambda value: value)

        def _prepare_batch_inputs(self, samples):
            with torch.no_grad():
                self._get_per_token_logps_and_entropies()
                with self.null_ref_context():
                    self._get_per_token_logps_and_entropies()
            return samples

        @staticmethod
        def _get_per_token_logps_and_entropies(*_args, **_kwargs):
            return None

        def _generate_completions(self, _samples):
            self.engine.set_default_max_tokens(SimpleNamespace(max_tokens=budget), inputs)
            calls.append("generate")
            return [SimpleNamespace(extra={"id": "q1", "occurrence_id": "presentation:000001"}) for _ in range(8)]

        @contextmanager
        def null_ref_context(self):
            calls.append("enter-ref")
            yield
            calls.append("exit-ref")

        def compute_loss(self, _model, _inputs):
            calls.append("loss")
            return 1

        @staticmethod
        def backward(_loss):
            calls.append("backward")

    trainer = Trainer()
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: calls.append("sync")),
                                 Tensor=torch.Tensor, is_grad_enabled=torch.is_grad_enabled)
    prompts, sft_inputs, take_step, restore, _, _ = install_runtime_probe(
        trainer, fake_torch, {"arm": arm, "max_length": 8192,
                              "runtime_optimization": runtime_optimization}, probe=True)
    assert len(trainer._generate_completions([object()])) == 8
    with trainer.null_ref_context():
        calls.append("ref-forward")
    trainer._prepare_batch_inputs([])
    trainer.accelerator.backward(trainer.compute_loss(None, {}))
    step = take_step()
    assert step["rollout_groups"] == 1 and step["rollout_answers"] == 8
    assert step["rollout_occurrence_ids"] == ["presentation:000001"]
    assert step["prompt_count"] == batch_size and step["prompt_tokens_step_max"] == 3501
    assert step["generation_seconds"] >= 0 and step["ref_forward_seconds"] >= 0
    assert step["old_policy_forward_seconds"] >= 0 and step["encoding_wall_seconds"] >= 0
    assert step["training_forward_backward_seconds"] >= 0
    assert all(name in step for name in ("gradient_audit_seconds", "optimizer_seconds"))
    if not runtime_optimization:
        assert calls.count("sync") >= 6
    assert prompts["first"]["image_grid_thw"] == [[1, 50, 90]] * (3 * batch_size)
    assert prompts["first"]["pixel_values_shape"] == [13500 * batch_size, 1176]
    assert prompts["first"]["batch_size"] == batch_size
    assert sft_inputs["count"] == 0
    assert calls.index("enter-ref") < calls.index("ref-forward") < calls.index("exit-ref")
    assert take_step()["rollout_answers"] == 0
    inputs["image_grid_thw"] = inputs["image_grid_thw"][:-1]
    with pytest.raises(RuntimeError, match="three-image probe"):
        engine.set_default_max_tokens(SimpleNamespace(max_tokens=budget), inputs)
    restore()
    assert trainer.compute_loss(None, {}) == 1


def test_runtime_probe_rejects_silent_budget_reduction_and_wrong_g8_ownership():
    import torch

    class Engine:
        @staticmethod
        def _get_num_tokens(inputs):
            return inputs["input_ids"].shape[-1]

        @staticmethod
        def set_default_max_tokens(request_config, _inputs):
            request_config.max_tokens -= 1

    class Trainer:
        def __init__(self):
            self.engine = Engine()
            self.accelerator = SimpleNamespace(backward=lambda _: None)
            self.template = SimpleNamespace(encode=lambda value: value)

        @staticmethod
        def _prepare_batch_inputs(samples):
            return samples

        @staticmethod
        def _get_per_token_logps_and_entropies(*_args, **_kwargs):
            return None

        @staticmethod
        def _generate_completions(_samples):
            return [SimpleNamespace(extra={"id": "q1", "occurrence_id": f"occ:{i}"}) for i in range(8)]

        @contextmanager
        def null_ref_context(self):
            yield

        @staticmethod
        def compute_loss(_model, _inputs):
            return 1

    trainer = Trainer()
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None), Tensor=torch.Tensor)
    _, _, _, restore, _, _ = install_runtime_probe(trainer, fake_torch, {"arm": "A", "max_length": 8192})
    with pytest.raises(RuntimeError, match="shortened completion budget"):
        trainer.engine.set_default_max_tokens(SimpleNamespace(max_tokens=64),
                                              {"input_ids": torch.zeros((1, 3501), dtype=torch.long)})
    with pytest.raises(RuntimeError, match="G8 rollout ownership"):
        trainer._generate_completions([])
    restore()


def test_runtime_probe_records_sft_selected_logits_without_rollout():
    import torch

    class Trainer:
        def __init__(self):
            self.accelerator = SimpleNamespace(backward=lambda _: None)
            self.template = SimpleNamespace(encode=lambda value: value)

        @staticmethod
        def compute_loss(_model, inputs):
            return inputs["labels"].sum()

    trainer = Trainer()
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None), Tensor=torch.Tensor)
    prompts, sft_inputs, take_step, restore, _, _ = install_runtime_probe(trainer, fake_torch, {"arm": "C"})
    inputs = {"input_ids": torch.zeros((1, 3501), dtype=torch.long),
              "labels": torch.ones((1, 24)),
              "logits_to_keep": torch.tensor([False] * 3477 + [True] * 24),
              "image_grid_thw": torch.tensor([[1, 50, 90]] * 3),
              "pixel_values": torch.zeros((13500, 1176))}
    trainer.accelerator.backward(trainer.compute_loss(None, inputs))
    assert prompts["count"] == 0
    assert sft_inputs["first"]["tokens"] == 3501
    assert sft_inputs["first"]["selected_logits_tokens"] == 24
    assert sft_inputs["first"]["image_grid_thw"] == [[1, 50, 90]] * 3
    step = take_step()
    assert step["generation_seconds"] == step["ref_forward_seconds"] == 0
    assert step["rollout_answers"] == 0 and step["prompt_count"] == 0
    restore()
