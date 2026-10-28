"""ms-swift 4.5.3 原生 Qwen3.6-27B NF4/BF16 四臂训练入口。

CPU: python tools/train_quant_grounding.py --config CONFIG --dry-run
GPU: python tools/train_quant_grounding.py --config CONFIG [--probe]
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import shutil
import sys
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.quant_grounding_config import (  # noqa: E402
    configure_unrestricted_hf_sampling, load_config, prepare_dataset, swift_arguments)


def audit_base_config(model_name: str, auto_config) -> dict:
    """训练只接受原始 BF16 权重，避免在已有 FP8 等量化权重上叠 NF4。"""
    base = auto_config.from_pretrained(model_name)
    if base.model_type != "qwen3_5":
        raise RuntimeError(f"模型应为 Qwen3.6/Qwen3.5 架构，实际为 {base.model_type}")
    quantized = [name for name, part in (("model", base), ("text_config", getattr(base, "text_config", None)))
                 if part is not None and getattr(part, "quantization_config", None)]
    if quantized:
        raise RuntimeError(f"训练需要原始 BF16 权重；{model_name} 的 {quantized} 已有 quantization_config")
    return {"model_type": base.model_type, "source_quantization_config": None}


def selected_kernel(func) -> dict:
    """查看 Transformers 装饰器闭包中实际捕获的实现，而非 wrapper.__module__。"""
    while func is not None:
        implementation = inspect.getclosurevars(func).nonlocals.get("implementation")
        if implementation is not None:
            return {"module": implementation.__module__, "name": implementation.__name__}
        func = getattr(func, "__wrapped__", None)
    raise RuntimeError("无法解析 Transformers kernel 装饰器选中的实现")


def audit_kernels() -> dict:
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    names = ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn")
    return {name: selected_kernel(getattr(qwen, name)) for name in names}


def storage_bytes(parameters) -> int:
    """按唯一 PyTorch 参数 storage 计字节，不含 bitsandbytes quant_state 附加存储。"""
    storages = {}
    for parameter in parameters:
        storage = parameter.untyped_storage()
        storages[(parameter.device, storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())


def budget_stop_signals(elapsed_seconds: float, limit_seconds: float | None,
                        longest_step_seconds: float | None) -> tuple[bool, bool]:
    """在下一完整更新可能越过时限前，同时请求保存并停止训练。"""
    if limit_seconds is None:
        return False, False
    stop = elapsed_seconds >= limit_seconds or (
        longest_step_seconds is not None and elapsed_seconds + longest_step_seconds >= limit_seconds)
    return stop, stop


def audit_optimizer(model, optimizer) -> dict:
    """核对真实参数组只优化语言 LoRA，冻结底座和参考分支无梯度。"""
    named = dict(model.named_parameters())
    by_id = {id(parameter): name for name, parameter in named.items()}
    expected = {id(parameter) for parameter in named.values() if parameter.requires_grad}
    actual = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    actual_ids = [id(parameter) for parameter in actual]
    bad = [by_id.get(identifier, f"unregistered:{identifier}") for identifier in actual_ids
           if identifier not in expected or "language_model" not in by_id.get(identifier, "")
           or "lora_" not in by_id.get(identifier, "")]
    if bad or set(actual_ids) != expected or len(actual_ids) != len(expected):
        raise RuntimeError(f"优化器参数组不等于语言 LoRA 可训练集合：unexpected={bad[:8]}，"
                           f"actual={len(actual_ids)}，expected={len(expected)}")
    frozen_with_grad = [name for name, parameter in named.items()
                        if not parameter.requires_grad and parameter.grad is not None]
    if frozen_with_grad:
        raise RuntimeError(f"冻结底座或参考参数出现梯度：{frozen_with_grad[:8]}")
    return {"optimizer_type": type(optimizer).__name__, "optimizer_group_count": len(optimizer.param_groups),
            "optimizer_parameter_count": len(actual), "frozen_parameter_count": len(named) - len(expected),
            "frozen_grad_count": 0}


def audit_model(model, precision: str, torch) -> dict:
    """检查加载后的真实模块类型、dtype 与语言 LoRA 命中范围。"""
    modules = dict(model.named_modules())
    linear4 = [name for name, module in modules.items() if type(module).__name__ == "Linear4bit"]
    visual4 = [name for name in linear4 if ".visual." in name]
    gate_names = [name for name in modules if name.endswith(("linear_attn.in_proj_a", "linear_attn.in_proj_b"))]
    gate4 = [name for name in gate_names if name in linear4]
    gate_dtypes = {str(modules[name].get_base_layer().weight.dtype if hasattr(modules[name], "get_base_layer")
                       else modules[name].weight.dtype) for name in gate_names}
    head = next(module for name, module in modules.items() if name.endswith("lm_head") and hasattr(module, "weight"))
    embeddings = [module for name, module in modules.items() if name.endswith("embed_tokens") and hasattr(module, "weight")]
    visual_linear = [module for name, module in modules.items()
                     if ".visual." in name and isinstance(module, torch.nn.Linear)]
    visual_dtypes = sorted({str(module.weight.dtype) for module in visual_linear})
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    report = {
        "precision": precision,
        "linear4_count": len(linear4),
        "visual_linear_count": len(visual_linear),
        "visual_linear_dtypes": visual_dtypes,
        "gate_count": len(gate_names),
        "gate_dtypes": sorted(gate_dtypes),
        "head_dtype": str(head.weight.dtype),
        "embedding_dtypes": sorted({str(module.weight.dtype) for module in embeddings}),
        "trainable_count": len(trainable),
        "trainable_examples": trainable[:8],
        "parameter_storage_bytes": storage_bytes(model.parameters()),
        "trainable_parameter_storage_bytes": storage_bytes(parameter for parameter in model.parameters()
                                                     if parameter.requires_grad),
        "parameter_storage_note": "PyTorch parameter storage only; excludes bitsandbytes quant_state storage",
    }
    if precision == "nf4" and len(linear4) < 100:
        raise RuntimeError(f"NF4 未覆盖足够语言线性层：{report}")
    if precision == "bf16" and linear4:
        raise RuntimeError(f"BF16 对照出现 Linear4bit：{report}")
    if visual4 or not visual_linear or visual_dtypes != ["torch.bfloat16"] or not gate_names or gate4:
        raise RuntimeError(f"视觉或 GDN 门控量化范围错误：{report}")
    if gate_dtypes != {"torch.bfloat16"} or head.weight.dtype != torch.bfloat16:
        raise RuntimeError(f"保留模块精度错误：{report}")
    if not embeddings or any(module.weight.dtype != torch.bfloat16 for module in embeddings):
        raise RuntimeError(f"词嵌入精度错误：{report}")
    if not trainable or any("language_model" not in name or "lora_" not in name for name in trainable):
        raise RuntimeError(f"训练范围不是语言侧 LoRA：{report}")
    return report


def install_language_gradient_checkpointing(model, base_model, every_n_layers: int,
                                           *, mask_language_layers: bool = True):
    """Keep Swift's repeated PEFT/base enables on the same language-only layer mask."""
    matches = [(name, module) for name, module in model.named_modules()
               if name.endswith("language_model.layers")]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one language_model.layers, found {[name for name, _ in matches]}")
    _, layers = matches[0]
    if not layers or any(not hasattr(layer, "gradient_checkpointing") for layer in layers):
        raise RuntimeError("语言 decoder layers 未提供 gradient_checkpointing 标志")
    originals = []
    audit = {"every_n_layers": every_n_layers, "language_layer_count": len(layers),
             "enabled_layer_indices": list(range(0, len(layers), every_n_layers)),
             "mask_applied": mask_language_layers, "reenable_calls": 0}

    if not mask_language_layers:
        return audit, lambda: None

    def apply_mask():
        for index, layer in enumerate(layers):
            layer.gradient_checkpointing = index % every_n_layers == 0
        audit["reenable_calls"] += 1

    for target in (base_model, model):
        if any(target is previous for previous, _ in originals):
            continue
        original = target.gradient_checkpointing_enable

        @wraps(original)
        def enabled(*args, _original=original, **kwargs):
            result = _original(*args, **kwargs)
            apply_mask()
            return result

        target.gradient_checkpointing_enable = enabled
        originals.append((target, original))

    def restore():
        for target, original in reversed(originals):
            target.gradient_checkpointing_enable = original

    return audit, restore


def normalize_runtime_optimization_config(config: dict) -> bool:
    enabled = config.setdefault("runtime_optimization", False)
    if not isinstance(enabled, bool):
        raise ValueError("runtime_optimization 必须是布尔值")
    config["runtime_optimization"] = enabled
    if not enabled and (config.get("train_micro_batch_size", 1) != 1
                        or config.get("gradient_checkpointing_every_n_layers", 1) != 1):
        raise ValueError("runtime_optimization=false 要求 train_micro_batch_size=1 且 gradient_checkpointing_every_n_layers=1")
    return enabled


def configure_deterministic_training(config: dict, torch=None) -> bool:
    enabled = config.setdefault("deterministic_training", False)
    if not isinstance(enabled, bool):
        raise ValueError("deterministic_training 必须是布尔值")
    if enabled:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        if torch is not None:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    return enabled


def audit_gradients(model, torch, *, runtime_optimization: bool = True) -> bool:
    """Check every LoRA gradient on device, then transfer just two aggregate flags."""
    frozen_with_grad = [name for name, parameter in model.named_parameters()
                        if not parameter.requires_grad and parameter.grad is not None]
    if frozen_with_grad:
        raise RuntimeError(f"冻结底座或参考参数出现梯度：{frozen_with_grad[:8]}")
    gradients = [parameter.grad for parameter in model.parameters()
                 if parameter.requires_grad and parameter.grad is not None]
    if not gradients:
        return False
    if not runtime_optimization:
        if any(not torch.isfinite(gradient).all().item() for gradient in gradients):
            raise RuntimeError("LoRA 梯度含 NaN/Inf")
        return any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients)
    # Infinity norm is max(abs(grad)): it preserves NaN/Inf and detects any nonzero element.
    maxima = torch.stack(torch._foreach_norm(gradients, ord=float("inf")))
    finite = torch.isfinite(maxima).all()
    nonzero = (maxima > 0).any()
    finite_cpu, nonzero_cpu = torch.stack((finite, nonzero)).tolist()
    if not finite_cpu:
        raise RuntimeError("LoRA 梯度含 NaN/Inf")
    return bool(nonzero_cpu)


def install_dapo_loss_normalizer(trainer, *, arm: str, train_micro_batch_size: int,
                                 runtime_optimization: bool):
    """Compensate DAPO's per-rollout token denominator for larger RL microbatches."""
    factor = train_micro_batch_size if runtime_optimization and arm != "C" else 1
    audit = {"applied": factor > 1, "factor": factor, "arm": arm}
    if factor == 1:
        return audit, lambda: None

    original = trainer._prepare_batch_inputs

    def prepare_batch_inputs(samples):
        batches = original(samples)
        if trainer.model.training:
            for batch in batches:
                grpo_batch = batch["grpo_batch"]
                grpo_batch.num_items_in_batch = grpo_batch.num_items_in_batch * factor
        return batches

    trainer._prepare_batch_inputs = prepare_batch_inputs

    def restore():
        trainer._prepare_batch_inputs = original

    return audit, restore


def audit_answers_per_optimizer_step(arm: str, train_micro_batch_size: int,
                                    gradient_accumulation_steps: int, num_generations: int = 8) -> dict:
    expected = 8 if arm == "C" else 4 * num_generations
    actual = train_micro_batch_size * gradient_accumulation_steps
    if actual != expected:
        raise RuntimeError(f"{arm} 臂每次优化器更新应保留 {expected} 个回答，实际为 {actual}")
    return {"train_micro_batch_size": train_micro_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "answers_per_optimizer_step": actual}


def sft_loss_scaling_contract_holds(arm: str, train_micro_batch_size: int,
                                   model_accepts_loss_kwargs: bool | None) -> bool:
    return arm != "C" or train_micro_batch_size == 1 or model_accepts_loss_kwargs is True


def install_selected_logits_audit(trainer, torch):
    """Observe GRPO logits on the PEFT call boundary, including output sequence length."""
    forward = {"calls": 0, "first": None, "minimum": None, "maximum": None,
               "hook_target": "peft_outer"}
    logps = {name: {"calls": 0, "first": None, "minimum": None, "maximum": None,
                    "output_lengths_match": True}
             for name in ("with_grad", "without_grad")}
    active = False
    original = trainer._get_logps_via_local_forward

    def tracked_logps_forward(*args, **kwargs):
        nonlocal active
        active = True
        try:
            return original(*args, **kwargs)
        finally:
            active = False

    def observe_input(_module, _args, kwargs):
        value = kwargs.get("logits_to_keep")
        if value is None:
            return
        value = int(value)
        forward["calls"] += 1
        if forward["first"] is None:
            forward["first"] = value
        forward["minimum"] = value if forward["minimum"] is None else min(value, forward["minimum"])
        forward["maximum"] = value if forward["maximum"] is None else max(value, forward["maximum"])

    def observe_output(_module, _args, kwargs, output):
        if not active:
            return
        requested = int(kwargs["logits_to_keep"])
        token_inputs = kwargs.get("input_ids")
        if token_inputs is None:
            token_inputs = kwargs["inputs_embeds"]
        input_tokens = int(token_inputs.shape[1])
        output_tokens = int(output.logits.shape[-2])
        bucket = logps["with_grad" if torch.is_grad_enabled() else "without_grad"]
        bucket["calls"] += 1
        if bucket["first"] is None:
            bucket["first"] = {"input_tokens": input_tokens, "requested_logits_tokens": requested,
                               "output_logits_tokens": output_tokens}
        bucket["minimum"] = requested if bucket["minimum"] is None else min(requested, bucket["minimum"])
        bucket["maximum"] = requested if bucket["maximum"] is None else max(requested, bucket["maximum"])
        bucket["output_lengths_match"] &= output_tokens == min(requested, input_tokens)

    trainer._get_logps_via_local_forward = tracked_logps_forward
    input_handle = trainer.model.register_forward_pre_hook(observe_input, with_kwargs=True)
    output_handle = trainer.model.register_forward_hook(observe_output, with_kwargs=True)

    def restore():
        input_handle.remove()
        output_handle.remove()
        trainer._get_logps_via_local_forward = original

    return forward, logps, restore


def install_runtime_probe(trainer, torch, config: dict, *, probe: bool = False):
    """Observe Swift's actual HF rollout and training phases without changing their inputs."""
    runtime_optimization = normalize_runtime_optimization_config(config)
    timer_names = ("generation_seconds", "old_policy_forward_seconds", "ref_forward_seconds",
                   "training_forward_backward_seconds", "gradient_audit_seconds", "optimizer_seconds")
    event_pairs = {name: [] for name in timer_names}
    phase = {"encoding_wall_seconds": 0.0, "rollout_occurrence_ids": [],
             "rollout_answers": 0, "prompt_count": 0, "prompt_tokens_step_max": None}
    prompts = {"count": 0, "first": None, "maximum": None}
    sft_inputs = {"count": 0, "first": None, "maximum": None}
    originals = {"compute_loss": trainer.compute_loss,
                 "backward": trainer.accelerator.backward, "encode": trainer.template.encode}

    def start_event():
        if not runtime_optimization:
            torch.cuda.synchronize()
            return time.monotonic()
        if not hasattr(torch.cuda, "Event"):
            return time.monotonic()
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def stop_event(name, started):
        if runtime_optimization:
            finished = start_event()
        else:
            torch.cuda.synchronize()
            finished = time.monotonic()
        event_pairs[name].append((started, finished))

    def elapsed_events(name):
        pairs = event_pairs[name]
        seconds = sum((end - start) if isinstance(start, float) else start.elapsed_time(end) / 1000
                      for start, end in pairs)
        pairs.clear()
        return seconds

    def measured_encode(*args, **kwargs):
        started = time.monotonic()
        try:
            return originals["encode"](*args, **kwargs)
        finally:
            phase["encoding_wall_seconds"] += time.monotonic() - started

    trainer.template.encode = measured_encode

    def timed_call(original, key):
        def wrapped(*args, **kwargs):
            if config["arm"] == "C" and key == "training_forward_backward_seconds" and original is originals["compute_loss"]:
                inputs = args[1]
                input_ids = inputs["input_ids"]
                length = int(input_ids.shape[-1])
                sft_inputs["count"] += 1
                if sft_inputs["first"] is None or length > sft_inputs["maximum"]["tokens"]:
                    selected = inputs.get("logits_to_keep")
                    if isinstance(selected, torch.Tensor):
                        selected = int(selected.sum().item())
                    grid = inputs.get("image_grid_thw")
                    pixels = inputs.get("pixel_values")
                    record = {"tokens": length, "selected_logits_tokens": selected,
                              "image_grid_thw_shape": list(grid.shape) if grid is not None else None,
                              "image_grid_thw": grid.detach().cpu().tolist() if grid is not None else None,
                              "pixel_values_shape": list(pixels.shape) if pixels is not None else None}
                    if sft_inputs["first"] is None:
                        sft_inputs["first"] = record
                    if sft_inputs["maximum"] is None or length > sft_inputs["maximum"]["tokens"]:
                        sft_inputs["maximum"] = record
            started = start_event()
            try:
                return original(*args, **kwargs)
            finally:
                stop_event(key, started)
        return wrapped

    trainer.compute_loss = timed_call(originals["compute_loss"], "training_forward_backward_seconds")
    trainer.accelerator.backward = timed_call(originals["backward"], "training_forward_backward_seconds")

    if config["arm"] != "C":
        engine = trainer.engine
        originals["generate"] = trainer._generate_completions
        originals["reference"] = trainer.null_ref_context
        originals["set_default_max_tokens"] = engine.set_default_max_tokens
        originals["prepare_batch_inputs"] = trainer._prepare_batch_inputs
        originals["get_logps"] = trainer._get_per_token_logps_and_entropies
        preparing = False
        reference_active = False

        def measured_prepare_batch_inputs(*args, **kwargs):
            nonlocal preparing
            preparing = True
            try:
                return originals["prepare_batch_inputs"](*args, **kwargs)
            finally:
                preparing = False

        def measured_logps(*args, **kwargs):
            if preparing and not reference_active and not torch.is_grad_enabled():
                started = start_event()
                try:
                    return originals["get_logps"](*args, **kwargs)
                finally:
                    stop_event("old_policy_forward_seconds", started)
            return originals["get_logps"](*args, **kwargs)

        def observe_max_tokens(request_config, inputs):
            prompt_tokens = engine._get_num_tokens(inputs)
            requested = request_config.max_tokens
            if prompt_tokens + requested > int(config.get("max_length", 8192)):
                raise RuntimeError(f"prompt {prompt_tokens} + completion {requested} exceeds max_length")
            result = originals["set_default_max_tokens"](request_config, inputs)
            if request_config.max_tokens != requested:
                raise RuntimeError(f"Swift shortened completion budget from {requested} to {request_config.max_tokens}")
            token_inputs = inputs.get("input_ids")
            if token_inputs is None:
                token_inputs = inputs["inputs_embeds"]
            batch_size = int(token_inputs.shape[0])
            prompts["count"] += batch_size
            phase["prompt_count"] += batch_size
            phase["prompt_tokens_step_max"] = max(phase["prompt_tokens_step_max"] or 0, prompt_tokens)
            grid = inputs.get("image_grid_thw")
            pixels = inputs.get("pixel_values")
            image_count = int(grid.shape[0]) if grid is not None else 0
            if probe and (image_count != 3 * batch_size or pixels is None):
                raise RuntimeError(f"three-image probe encoded {image_count} image grids for {batch_size} requests; "
                                   f"pixel_values={pixels is not None}")
            if prompts["first"] is None or prompt_tokens > prompts["maximum"]["tokens"]:
                record = {"tokens": prompt_tokens, "completion_budget": requested,
                          "image_count": image_count, "batch_size": batch_size,
                          "image_grid_thw_shape": list(grid.shape) if grid is not None else None,
                          "image_grid_thw": grid.detach().cpu().tolist() if grid is not None else None,
                          "pixel_values_shape": list(pixels.shape) if pixels is not None else None}
                if prompts["first"] is None:
                    prompts["first"] = record
                if prompts["maximum"] is None or prompt_tokens > prompts["maximum"]["tokens"]:
                    prompts["maximum"] = record
            return result

        def measured_generate(samples):
            started = start_event()
            try:
                generated = originals["generate"](samples)
            finally:
                stop_event("generation_seconds", started)
            occurrences = [sample.extra.get("occurrence_id") for sample in generated]
            ids = [sample.extra.get("id") for sample in generated]
            generations = int(config.get("num_generations", 8))
            if (len(generated) != generations or None in occurrences or None in ids
                    or len(set(occurrences)) != 1 or len(set(ids)) != 1):
                raise RuntimeError(f"G{generations} rollout ownership differs: answers={len(generated)}, "
                                   f"occurrence_ids={occurrences}, ids={ids}")
            phase["rollout_occurrence_ids"].append(occurrences[0])
            phase["rollout_answers"] += len(generated)
            return generated

        @contextmanager
        def measured_reference():
            nonlocal reference_active
            with originals["reference"]():
                reference_active = True
                started = start_event()
                try:
                    yield
                finally:
                    stop_event("ref_forward_seconds", started)
                    reference_active = False

        engine.set_default_max_tokens = observe_max_tokens
        trainer._generate_completions = measured_generate
        trainer.null_ref_context = measured_reference
        trainer._prepare_batch_inputs = measured_prepare_batch_inputs
        trainer._get_per_token_logps_and_entropies = measured_logps

    def take_step():
        result = {name: elapsed_events(name) for name in timer_names}
        result.update({"encoding_wall_seconds": phase["encoding_wall_seconds"],
                  "rollout_groups": len(phase["rollout_occurrence_ids"]),
                  "rollout_answers": phase["rollout_answers"],
                  "rollout_occurrence_ids": phase["rollout_occurrence_ids"],
                  "prompt_count": phase["prompt_count"],
                  "prompt_tokens_step_max": phase["prompt_tokens_step_max"],
                  "first_prompt_tokens": prompts["first"]["tokens"] if prompts["first"] else None,
                  "max_prompt_tokens": prompts["maximum"]["tokens"] if prompts["maximum"] else None})
        phase.update({"encoding_wall_seconds": 0.0, "rollout_occurrence_ids": [],
                      "rollout_answers": 0, "prompt_count": 0, "prompt_tokens_step_max": None})
        return result

    def restore():
        trainer.compute_loss = originals["compute_loss"]
        trainer.accelerator.backward = originals["backward"]
        trainer.template.encode = originals["encode"]
        if config["arm"] != "C":
            trainer.engine.set_default_max_tokens = originals["set_default_max_tokens"]
            trainer._generate_completions = originals["generate"]
            trainer.null_ref_context = originals["reference"]
            trainer._prepare_batch_inputs = originals["prepare_batch_inputs"]
            trainer._get_per_token_logps_and_entropies = originals["get_logps"]

    return prompts, sft_inputs, take_step, restore, start_event, stop_event


def run_swift(config: dict, dataset: Path, *, max_steps: int | None = None,
              save_steps: int | None = None, resume: str | None = None,
              probe: bool = False, trainer_hook=None) -> dict:
    runtime_optimization = normalize_runtime_optimization_config(config)
    deterministic_training = configure_deterministic_training(config)
    import importlib.metadata

    import torch
    if deterministic_training:
        configure_deterministic_training(config, torch)
    from swift.arguments import RLHFArguments, SftArguments
    from swift.pipelines.train.rlhf import SwiftRLHF
    from swift.pipelines.train.sft import SwiftSft
    from transformers import AutoConfig, TrainerCallback

    version = importlib.metadata.version("ms-swift")
    if version != "4.5.3":
        raise RuntimeError(f"训练入口按 ms-swift 4.5.3 核对，当前为 {version}")
    if not torch.cuda.is_available():
        raise RuntimeError("真实训练需要 CUDA GPU；CPU 请用 --dry-run")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("本实验预算与 HF 并发检查限定单进程单卡")

    base_audit = audit_base_config(str(config["model"]), AutoConfig)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    segment_start = time.monotonic()
    state_path = output_dir / "budget_state.json"
    if state_path.exists() and not (resume or config.get("resume_from_checkpoint")):
        raise RuntimeError("输出目录已有训练预算记录，继续训练必须指定完整 checkpoint")
    prior_seconds = json.loads(state_path.read_text(encoding="utf-8"))["elapsed_seconds"] if state_path.exists() else 0.0
    limit_seconds = None if config.get("max_gpu_hours") is None else float(config["max_gpu_hours"]) * 3600
    if limit_seconds is not None and prior_seconds >= limit_seconds:
        raise RuntimeError("本臂累计 GPU 时限已用尽")

    class QuantRLHFArguments(RLHFArguments):
        def get_modules_to_not_convert(self):
            skip = super().get_modules_to_not_convert() or []
            return list(dict.fromkeys([*skip, "model.visual", "lm_head",
                                       "linear_attn.in_proj_a", "linear_attn.in_proj_b"]))

    class QuantSftArguments(SftArguments):
        def get_modules_to_not_convert(self):
            skip = super().get_modules_to_not_convert() or []
            return list(dict.fromkeys([*skip, "model.visual", "lm_head",
                                       "linear_attn.in_proj_a", "linear_attn.in_proj_b"]))

    class BudgetCallback(TrainerCallback):
        def __init__(self, trainer, take_runtime_step, start_event, stop_event):
            self.trainer = trainer
            self.take_runtime_step = take_runtime_step
            self.start_event = start_event
            self.stop_event = stop_event
            self.last_step_time = segment_start
            self.longest_step_seconds = None
            self.gradient_observed = False
            self.optimizer_audited = False
            self.peak_allocated = 0
            self.milestones = [float(hour) * 3600 for hour in config.get("save_hours", [2, 4, 8])]
            self.reached = {value for value in self.milestones if value <= prior_seconds}
            self.matched_checkpoint_step = None
            self.optimizer_started = None

        def elapsed(self):
            return prior_seconds + time.monotonic() - segment_start

        def persist(self, step: int):
            state_path.write_text(json.dumps({"elapsed_seconds": self.elapsed(), "global_step": step,
                                               "peak_allocated_bytes": self.peak_allocated}, indent=2), encoding="utf-8")

        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            if not self.optimizer_audited and not config.get("benchmark_mode", False):
                report = audit_optimizer(self.trainer.model, self.trainer.optimizer)
                audit_path = output_dir / "runtime_audit.json"
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
                audit["optimizer"] = report
                audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
                self.optimizer_audited = True
            if not config.get("benchmark_mode", False):
                started = self.start_event()
                try:
                    self.gradient_observed |= audit_gradients(
                        self.trainer.model, torch, runtime_optimization=runtime_optimization)
                finally:
                    self.stop_event("gradient_audit_seconds", started)
            self.optimizer_started = self.start_event()

        def on_optimizer_step(self, args, state, control, **kwargs):
            self.stop_event("optimizer_seconds", self.optimizer_started)
            self.optimizer_started = None

        def on_step_end(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            now = time.monotonic()
            step_seconds = now - self.last_step_time
            self.longest_step_seconds = (step_seconds if self.longest_step_seconds is None
                                         else max(self.longest_step_seconds, step_seconds))
            self.peak_allocated = max(self.peak_allocated, torch.cuda.max_memory_allocated())
            row = {"step": state.global_step, "step_seconds": step_seconds,
                   "elapsed_seconds": self.elapsed(),
                   "allocated_peak_bytes": torch.cuda.max_memory_allocated(),
                   "reserved_peak_bytes": torch.cuda.max_memory_reserved(),
                   "nonzero_finite_lora_gradient_seen": self.gradient_observed}
            row.update(self.take_runtime_step())
            if not config.get("benchmark_mode", False):
                with (output_dir / "resource_steps.jsonl").open("a", encoding="utf-8") as out:
                    out.write(json.dumps(row) + "\n")
            if not config.get("benchmark_mode", False):
                torch.cuda.reset_peak_memory_stats()
            self.last_step_time = now
            for milestone in self.milestones:
                if self.elapsed() >= milestone and milestone not in self.reached:
                    self.reached.add(milestone)
                    control.should_save = True
            matched = config.get("matched_query_exposures")
            exposure_per_step = 8 if config["arm"] == "C" else 4
            if matched and (state.global_step - 1) * exposure_per_step < matched <= state.global_step * exposure_per_step:
                self.matched_checkpoint_step = state.global_step
                control.should_save = True
            save_for_budget, stop_for_budget = budget_stop_signals(
                self.elapsed(), limit_seconds,
                self.longest_step_seconds)
            control.should_save = control.should_save or save_for_budget
            control.should_training_stop = control.should_training_stop or stop_for_budget
            self.persist(state.global_step)
            return control

        def on_save(self, args, state, control, **kwargs):
            if state.global_step == self.matched_checkpoint_step:
                source = output_dir / f"checkpoint-{state.global_step}"
                protected = output_dir / f"matched-exposure-checkpoint-{state.global_step}"
                if not protected.exists():
                    shutil.copytree(source, protected)

        def on_log(self, args, state, control, logs=None, **kwargs):
            if config.get("benchmark_mode", False):
                return
            with (output_dir / "training_metrics.jsonl").open("a", encoding="utf-8") as out:
                out.write(json.dumps({"step": state.global_step, "elapsed_seconds": self.elapsed(),
                                      "metrics": logs or {}}) + "\n")

    class QuantRunMixin:
        @classmethod
        def prepare_model(cls, args, model, *, template=None, train_dataset=None, task_type=None):
            model = super().prepare_model(args, model, template=template, train_dataset=train_dataset,
                                          task_type=task_type)
            if config["arm"] != "C" and config.get("adapters"):
                ref_parameters = [parameter for name, parameter in model.named_parameters()
                                  if ".ref_adapter." in name]
                if not ref_parameters:
                    raise RuntimeError("共享 warmup adapter 未加载为 ref_adapter")
                for parameter in ref_parameters:
                    parameter.requires_grad_(False)
            report = audit_model(model, config["precision"], torch)
            (output_dir / "model_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            return model

        def train(self, trainer):
            forward_observed = None
            logps_observed = None
            restore_logits_audit = None
            audit_path = output_dir / "runtime_audit.json"
            prior_audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
            runtime_audit = {
                "device_name": torch.cuda.get_device_name(),
                "base_model": base_audit,
                "kernel_implementations": audit_kernels(),
                "model": json.loads((output_dir / "model_audit.json").read_text(encoding="utf-8")),
                "runtime_optimization": runtime_optimization,
                "deterministic_training": deterministic_training,
                "determinism": {
                    "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                    "cudnn_deterministic": torch.backends.cudnn.deterministic,
                    "cudnn_benchmark": torch.backends.cudnn.benchmark,
                    "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                },
            }
            configured_micro_batch = int(config.get("train_micro_batch_size", 1))
            actual_micro_batch = int(trainer.args.per_device_train_batch_size)
            if actual_micro_batch != configured_micro_batch:
                raise RuntimeError(f"训练 micro batch 配置/Swift 参数不一致: {configured_micro_batch}/{actual_micro_batch}")
            model_accepts_loss_kwargs = getattr(trainer, "model_accepts_loss_kwargs", None)
            runtime_audit["model_accepts_loss_kwargs"] = model_accepts_loss_kwargs
            if not sft_loss_scaling_contract_holds(
                config["arm"], actual_micro_batch, model_accepts_loss_kwargs
            ):
                audit_path.write_text(json.dumps(runtime_audit, indent=2), encoding="utf-8")
                raise RuntimeError(
                    "C 臂 train_micro_batch_size>1 要求 Swift Trainer.model_accepts_loss_kwargs=True，"
                    f"实际为 {model_accepts_loss_kwargs!r}"
                )
            runtime_audit["training_batch"] = audit_answers_per_optimizer_step(
                config["arm"], actual_micro_batch, int(trainer.args.gradient_accumulation_steps),
                int(config.get("num_generations", 8)))
            base_model = trainer.template.get_base_model(trainer.accelerator.unwrap_model(trainer.model))
            checkpoint_audit, restore_checkpointing = install_language_gradient_checkpointing(
                trainer.model, base_model, int(config.get("gradient_checkpointing_every_n_layers", 1)),
                mask_language_layers=runtime_optimization)
            runtime_audit["gradient_checkpointing"] = checkpoint_audit
            with dataset.open(encoding="utf-8") as prepared_file:
                prepared_rows = sum(bool(line.strip()) for line in prepared_file)
            encoded_rows = len(trainer.train_dataset)
            runtime_audit["dataset_rows"] = {"prepared": prepared_rows, "encoded": encoded_rows}
            if encoded_rows != prepared_rows:
                raise RuntimeError(f"Swift encoded {encoded_rows}/{prepared_rows} rows; inspect context truncation")
            if config["arm"] != "C":
                from peft import PeftModel
                if trainer.ref_model is not None or not isinstance(trainer.model, PeftModel):
                    raise RuntimeError("GRPO 必须通过同一 PEFT 模型关闭 adapter 计算参考分布")
                if len(trainer.reward_funcs) != 1:
                    raise RuntimeError("本实验每臂应恰好有一个 grounding 奖励函数")
                trainer.reward_funcs[0].bind_template(trainer.template)
                expected_ref_adapter = "ref_adapter" if config.get("adapters") else None
                if trainer.ref_adapter_name != expected_ref_adapter:
                    raise RuntimeError("共享 warmup 的冻结参考 adapter 配置错误")
                if trainer.engine.max_batch_size != 0:
                    raise RuntimeError("ms-swift HF engine 默认值已变化，请复核分批语义")
                trainer.engine.max_batch_size = int(config.get("hf_max_batch_size", 1))
                sampling = configure_unrestricted_hf_sampling(trainer.engine, trainer.request_config)
                if not sampling["do_sample"]:
                    raise RuntimeError("GRPO rollout must sample completions")
                runtime_audit["effective_sampling"] = sampling
                generations = int(config.get("num_generations", 8))
                if trainer.num_generations != generations or trainer.args.generation_batch_size != generations:
                    raise RuntimeError(f"G{generations}/生成批次{generations} 与计划不符")
                if "logits_to_keep" not in trainer.model_kwarg_keys:
                    raise RuntimeError("当前模型前向不支持短 logits_to_keep")
                if not config.get("shared_prefix", False):
                    forward_observed, logps_observed, restore_logits_audit = install_selected_logits_audit(
                        trainer, torch)
                runtime_audit.update({
                    "hf_max_batch_size": trainer.engine.max_batch_size,
                    "num_generations": trainer.num_generations,
                    "generation_batch_size": trainer.args.generation_batch_size,
                    "gradient_accumulation_steps": trainer.args.gradient_accumulation_steps,
                    "ref_model_is_none": trainer.ref_model is None,
                    "logits_to_keep_supported": True,
                })
            normalizer_audit, restore_loss_normalizer = install_dapo_loss_normalizer(
                trainer, arm=config["arm"],
                train_micro_batch_size=int(config.get("train_micro_batch_size", 1)),
                runtime_optimization=runtime_optimization,
            )
            runtime_audit["grpo_loss_normalizer"] = normalizer_audit
            shared_prefix_audit = None
            restore_shared_prefix = None
            if config.get("shared_prefix", False):
                from tools.quant_grounding_shared_prefix import install_shared_prefix_training
                shared_prefix_audit, restore_shared_prefix = install_shared_prefix_training(trainer, config)
                runtime_audit["shared_prefix"] = shared_prefix_audit
            audit_path.write_text(json.dumps(runtime_audit, indent=2), encoding="utf-8")
            prompts, sft_inputs, take_runtime_step, restore_runtime_probe, start_event, stop_event = install_runtime_probe(
                trainer, torch, config, probe=probe)
            callback = BudgetCallback(trainer, take_runtime_step, start_event, stop_event)
            if limit_seconds is not None and callback.elapsed() >= limit_seconds:
                raise RuntimeError("模型加载后本臂 GPU 时限已用尽")
            trainer.add_callback(callback)
            hook_cleanup = None
            thinking_cleanup = None
            thinking_audit = None
            try:
                if config.get("bounded_thinking", False):
                    from tools.quant_grounding_thinking import install_bounded_thinking_training
                    thinking_audit, thinking_cleanup = install_bounded_thinking_training(trainer, config)
                    audit = json.loads(audit_path.read_text(encoding="utf-8"))
                    audit["bounded_thinking"] = thinking_audit
                    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
                if trainer_hook is not None:
                    hook_cleanup = trainer_hook(trainer)
                result = super().train(trainer)
            finally:
                if hook_cleanup is not None:
                    hook_cleanup()
                if thinking_cleanup is not None:
                    thinking_cleanup()
                if restore_logits_audit is not None:
                    restore_logits_audit()
                restore_checkpointing()
                callback.persist(trainer.state.global_step)
                restore_runtime_probe()
                if restore_shared_prefix is not None:
                    restore_shared_prefix()
                restore_loss_normalizer()
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
                if shared_prefix_audit is not None:
                    audit["shared_prefix"] = shared_prefix_audit
                if thinking_audit is not None:
                    audit["bounded_thinking"] = thinking_audit
                for key, observed in (("prompt_observations", prompts), ("sft_input_observations", sft_inputs)):
                    previous = prior_audit.get(key)
                    if previous:
                        observed["count"] += previous["count"]
                        observed["first"] = previous["first"] or observed["first"]
                        if previous["maximum"] and (
                            observed["maximum"] is None or previous["maximum"]["tokens"] > observed["maximum"]["tokens"]
                        ):
                            observed["maximum"] = previous["maximum"]
                audit["prompt_observations"] = prompts
                audit["sft_input_observations"] = sft_inputs
                audit["gradient_checkpointing"] = checkpoint_audit
                if runtime_optimization:
                    audit["phase_timing_note"] = (
                        "CUDA events with one synchronization per optimizer step; generation includes HF encoding "
                        "and postprocessing. encoding_wall_seconds uses CPU wall time for template.encode and "
                        "overlaps generation. Old-policy forward remains an independent Swift forward."
                    )
                else:
                    audit["phase_timing_note"] = (
                        "Legacy timing synchronizes and uses wall time before and after each measured phase. "
                        "encoding_wall_seconds uses CPU wall time for template.encode and overlaps generation."
                    )
                if restore_logits_audit is not None:
                    previous = prior_audit.get("selected_logits_forward")
                    if previous and previous.get("hook_target") == "peft_outer":
                        forward_observed["calls"] += previous["calls"]
                        if previous.get("first") is not None:
                            forward_observed["first"] = previous["first"]
                        values = [value for value in (previous["minimum"], forward_observed["minimum"])
                                  if value is not None]
                        forward_observed["minimum"] = min(values) if values else None
                        values = [value for value in (previous["maximum"], forward_observed["maximum"])
                                  if value is not None]
                        forward_observed["maximum"] = max(values) if values else None
                    audit["selected_logits_forward"] = forward_observed
                    audit["selected_logits_forward_note"] = (
                        "PEFT outer hook counts every call it observes; grpo_logps_forward "
                        "records only Swift's local logps method."
                    )
                    for phase_name, observed in logps_observed.items():
                        previous_phase = prior_audit.get("grpo_logps_forward", {}).get(phase_name)
                        if previous_phase:
                            observed["calls"] += previous_phase["calls"]
                            observed["first"] = previous_phase["first"] or observed["first"]
                            for bound, combine in (("minimum", min), ("maximum", max)):
                                values = [value for value in (previous_phase[bound], observed[bound])
                                          if value is not None]
                                observed[bound] = combine(values) if values else None
                            observed["output_lengths_match"] &= previous_phase["output_lengths_match"]
                    audit["grpo_logps_forward"] = logps_observed
                audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
            if probe and trainer.state.global_step >= 2 and not callback.gradient_observed:
                raise RuntimeError("真实 probe 未观察到非零有限 LoRA 梯度")
            if probe and config.get("shared_prefix", False) and not shared_prefix_audit["with_grad_calls"]:
                raise RuntimeError("真实 probe 未观察到可微共享前缀训练前向")
            if probe and config["arm"] != "C" and not config.get("shared_prefix", False) and not forward_observed["calls"]:
                raise RuntimeError("真实 probe 未观察到 logits_to_keep 进入模型前向")
            return result

    class QuantSwiftRLHF(QuantRunMixin, SwiftRLHF):
        args_class = QuantRLHFArguments

    class QuantSwiftSft(QuantRunMixin, SwiftSft):
        args_class = QuantSftArguments

    argv = swift_arguments(config, dataset, max_steps=max_steps, save_steps=save_steps, resume=resume)
    pipeline = QuantSwiftSft(argv) if config["arm"] == "C" else QuantSwiftRLHF(argv)
    return pipeline.main()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model")
    parser.add_argument("--dataset")
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--max-gpu-hours", type=float, help="本臂累计训练时限，含本进程模型加载")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--probe", action="store_true", help="真实三图 G8 两步训练，保存并重载后再更新一步")
    cli = parser.parse_args()
    config = load_config(cli.config, model=cli.model, dataset=cli.dataset, data_root=cli.data_root,
                         output_dir=cli.output_dir, resume_from_checkpoint=cli.resume_from_checkpoint)
    normalize_runtime_optimization_config(config)
    configure_deterministic_training(config)
    if cli.max_gpu_hours is not None:
        if cli.max_gpu_hours <= 0:
            raise ValueError("--max-gpu-hours 必须为正")
        config["max_gpu_hours"] = cli.max_gpu_hours
    if cli.probe:
        config["output_dir"] = str(Path(config["output_dir"]) / "probe")
        config.pop("matched_query_exposures", None)
        config["save_total_limit"] = 1
    destination = Path(config["output_dir"]) / ("probe_train.jsonl" if cli.probe else "prepared_train.jsonl")
    if cli.dry_run:
        dry = {"prepared_dataset": str(destination), "probe": cli.probe}
        if cli.probe:
            checkpoint = Path(config["output_dir"]) / "checkpoint-2"
            dry["initial_arguments"] = swift_arguments(config, destination, max_steps=2, save_steps=1)
            dry["reload_arguments"] = swift_arguments(config, destination, max_steps=3, save_steps=1,
                                                        resume=str(checkpoint))
        else:
            dry["swift_arguments"] = swift_arguments(config, destination)
        print(json.dumps(dry, ensure_ascii=False, indent=2))
        return
    if cli.probe and config.get("resume_from_checkpoint"):
        raise ValueError("probe 从独立空目录启动，不传 resume_from_checkpoint")
    source = Path(config["dataset"])
    summary = json.loads((source.parent / "summary.json").read_text(encoding="utf-8"))
    if not cli.probe and summary["mode"] != "formal":
        raise ValueError(f"正式训练只接受 summary.json mode=formal，当前为 {summary['mode']}")
    data_root = Path(config["data_root"])
    os.environ["MAX_PIXELS"] = str(config.get("max_pixels", 1204224))
    os.environ["MIN_PIXELS"] = str(config.get("min_pixels", 200704))
    os.environ["QUANT_REWARD_METRICS_JSONL"] = str(Path(config["output_dir"]) / "reward_metrics.jsonl")
    count = prepare_dataset(source, destination, data_root, config["arm"], probe=cli.probe,
                            bounded_thinking=config.get("bounded_thinking", False))
    print(f"prepared_samples={count}", flush=True)
    if cli.probe:
        run_swift(config, destination, max_steps=2, save_steps=1, probe=True)
        checkpoint = Path(config["output_dir"]) / "checkpoint-2"
        required = ["trainer_state.json", "optimizer.pt", "scheduler.pt"]
        missing = [name for name in required if not (checkpoint / name).is_file()]
        if missing:
            raise RuntimeError(f"无法真实续训，checkpoint-2 缺少 {missing}")
        gc.collect()
        import torch
        torch.cuda.empty_cache()
        run_swift(config, destination, max_steps=3, save_steps=1, resume=str(checkpoint), probe=True)
    else:
        run_swift(config, destination)


if __name__ == "__main__":
    main()
