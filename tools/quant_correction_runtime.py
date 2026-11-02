"""Qwen3.6 NF4 纠错训练：原生生成、逐回答计分与逐题冻结视觉缓存。"""

from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from pathlib import Path

import torch

from tools.quant_correction import Completion
from tools.quant_prefill_reuse import generation_cache_snapshot, native_prefill_reuse


SHORT_RESPONSE_PREFIX = "<think>\n\n</think>\n\n"
MODEL_INPUTS = {
    "input_ids", "attention_mask", "position_ids", "mm_token_type_ids",
    "pixel_values", "image_grid_thw",
}


def _load_components(config: dict, adapter_path: str | None):
    """只用 Swift 加载模型和模板；不建立 Trainer 或替换模型计算。"""
    from peft import PeftModel
    from swift.model import get_model_processor
    from swift.template import get_template
    from transformers import AutoConfig, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3.6-27B NF4 运行需要 CUDA")
    os.environ["MIN_PIXELS"] = str(config.get("min_pixels", 200704))
    os.environ["MAX_PIXELS"] = str(config.get("max_pixels", 1204224))
    source = AutoConfig.from_pretrained(config["model"], revision=config.get("model_revision"))
    if source.model_type != "qwen3_5" or any(
        getattr(part, "quantization_config", None)
        for part in (source, source.text_config)
    ):
        raise ValueError("纠错训练只接受 Qwen3.5/3.6 原始未量化底座")
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=[
            "model.visual", "lm_head", "linear_attn.in_proj_a", "linear_attn.in_proj_b",
        ],
    )
    base, processor = get_model_processor(
        config["model"], model_type=config.get("model_type", "qwen3_5"),
        revision=config.get("model_revision"), use_hf=True,
        torch_dtype=torch.bfloat16, quantization_config=quantization,
        device_map={"": torch.cuda.current_device()}, attn_impl="sdpa",
    )
    modules = dict(base.named_modules())
    quantized = [name for name, module in modules.items() if type(module).__name__ == "Linear4bit"]
    if len(quantized) < 100 or any(
        ".visual." in name or name.endswith(("lm_head", "linear_attn.in_proj_a", "linear_attn.in_proj_b"))
        for name in quantized
    ):
        raise RuntimeError("NF4 覆盖范围不符合语言线性层量化、视觉/门控/输出头保留的约定")
    sensitive = [module for name, module in modules.items()
                 if hasattr(module, "weight") and (
                     (".visual." in name and isinstance(module, torch.nn.Linear))
                     or name.endswith(("lm_head", "embed_tokens", "linear_attn.in_proj_a", "linear_attn.in_proj_b")))]
    if not sensitive or any(module.weight.dtype != torch.bfloat16 for module in sensitive):
        raise RuntimeError("视觉、嵌入、门控或输出头未保留 BF16")
    base.requires_grad_(False)
    # 不使用 prepare_model_for_kbit_training：它会把保留的 BF16 模块整体转为 FP32。
    model = PeftModel.from_pretrained(
        base, adapter_path or config["init_adapter"], adapter_name="default", is_trainable=True,
    )
    model.load_adapter(config["init_adapter"], adapter_name="reference", is_trainable=False)
    for name, adapter_config in model.peft_config.items():
        if (adapter_config.peft_type != "LORA" or adapter_config.modules_to_save
                or adapter_config.lora_dropout != 0):
            raise ValueError(f"{name}: 需要无 dropout、无额外保存模块的语言 LoRA")
    template = get_template(
        processor, template_type="qwen3_5", max_length=config.get("max_length", 8192),
        truncation_strategy="raise", max_pixels=config.get("max_pixels", 1204224),
        enable_thinking=False, response_prefix=SHORT_RESPONSE_PREFIX,
    )
    return model, template


class CorrectionRuntime:
    def __init__(self, config: dict, mode: str, adapter_path: str | None = None):
        if mode not in {"correction", "sft"}:
            raise ValueError("mode 必须为 correction 或 sft")
        self.config = dict(config)
        self.mode = mode
        self.model, self.template = _load_components(self.config, adapter_path)
        self.base = self.model.get_base_model()
        self.multimodal = self.base.model
        self.tokenizer = self.template.tokenizer
        self.device = self.base.get_input_embeddings().weight.device
        self.template.model = self.model
        self.template.set_mode("transformers")
        self.eos_token_ids = self.base.generation_config.eos_token_id
        if isinstance(self.eos_token_ids, int):
            self.eos_token_ids = [self.eos_token_ids]
        self.eos_token_ids = tuple(self.eos_token_ids)
        self.answer_eos_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if self.answer_eos_id not in self.eos_token_ids:
            raise ValueError("Qwen 短回答终止词元 <|im_end|> 不在 generation_config.eos_token_id 中")
        self._activate("default")
        self.model.train()
        self.multimodal.visual.eval()
        if config.get("gradient_checkpointing", True):
            # 非重入 checkpoint 允许冻结嵌入的输入无梯度，只重算原生语言 decoder。
            self.multimodal.language_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        self._question_active = False
        self.audit = {"questions": 0, "visual_encodes": 0,
                      "policy_forwards": 0, "reference_forwards": 0,
                      "trainable_parameters": sum(parameter.numel() for parameter in self.trainable_parameters())}
        trainable = [name for name, parameter in self.model.named_parameters() if parameter.requires_grad]
        if not trainable or any(
            "language_model" not in name or ".lora_" not in name or ".default." not in name
            for name in trainable
        ):
            raise RuntimeError("可训练参数必须恰为 default adapter 的语言 LoRA")
        if any(".lora_" in name and "language_model" not in name
               for name, _ in self.model.named_parameters()):
            raise RuntimeError("C550 adapter 包含语言模块之外的 LoRA 权重")

    def _activate(self, adapter: str):
        self.model.set_adapter(adapter)
        # PEFT set_adapter 会改变 requires_grad，必须每次明确恢复训练范围。
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_(
                adapter == "default" and "language_model" in name
                and ".lora_" in name and ".default." in name)

    @contextmanager
    def _model_context(self, *, reference: bool = False, generation: bool = False):
        training = self.model.training
        self._activate("reference" if reference else "default")
        self.model.train(not (reference or generation))
        self.multimodal.visual.eval()
        try:
            with torch.no_grad() if reference or generation else nullcontext():
                with self._autocast():
                    yield
        finally:
            self._activate("default")
            self.model.train(training)
            self.multimodal.visual.eval()

    def _autocast(self):
        return (torch.autocast("cuda", dtype=torch.bfloat16)
                if self.device.type == "cuda" else nullcontext())

    def trainable_parameters(self):
        return [parameter for parameter in self.model.parameters() if parameter.requires_grad]

    @contextmanager
    def question(self, row: dict):
        if self._question_active:
            raise RuntimeError("question 上下文不可嵌套")
        if any(message["role"] == "assistant" for message in row["messages"]):
            raise ValueError("生成输入不得包含 assistant 或 GT 回答")
        messages = deepcopy(row["messages"])
        images = [str((Path(self.config["media_root"]) / image).resolve()) for image in row["images"]]
        if sum(message["content"].count("<image>") for message in messages) != len(images):
            raise ValueError("生成输入的 image 标记与图像数量不一致")
        # 只传 messages/images，solution 等监督字段从未进入模板。
        encoded = self.template.encode({
            "messages": messages, "images": images,
            "chat_template_kwargs": {"enable_thinking": False},
        })
        with torch.inference_mode(False):
            batch = self.template.data_collator([encoded])
            # Swift encode 在 inference_mode 下运行；clone 后的普通张量可参与语言反传。
            inputs = {key: value.to(self.device).clone() for key, value in batch.items() if key in MODEL_INPUTS}
        if inputs["input_ids"].shape[1] + 64 > self.config.get("max_length", 8192):
            raise ValueError("输入与完整 64 词元回答预算超出 max_length")
        original = self.multimodal.get_image_features
        had_override = "get_image_features" in self.multimodal.__dict__
        with torch.inference_mode(False), torch.no_grad(), self._autocast():
            self.multimodal.visual.eval()
            features = original(inputs["pixel_values"], inputs["image_grid_thw"], return_dict=True)
        self.audit["visual_encodes"] += 1
        self.audit["questions"] += 1

        def cached_image_features(*args, **kwargs):
            return features

        self.multimodal.get_image_features = cached_image_features
        self._question_active = True
        question = _Question(self, inputs)
        try:
            yield question
        finally:
            question.active = False
            if had_override:
                self.multimodal.get_image_features = original
            else:
                del self.multimodal.get_image_features
            self._question_active = False

    def save_adapter(self, path: str | Path):
        self._activate("default")
        self.model.save_pretrained(
            path, selected_adapters=["default"], safe_serialization=True, save_embedding_layers=False)


class _Question:
    def __init__(self, runtime: CorrectionRuntime, inputs: dict):
        self.runtime = runtime
        self.inputs = inputs
        self.prefix_length = inputs["input_ids"].shape[1]
        self.active = True
        self.generation_diagnostics = []
        self.prefill_reuse_audit = {}

    def _check_active(self):
        if not self.active:
            raise RuntimeError("回答操作必须位于 question 上下文内")

    def generate(self, count: int = 2, *, greedy: bool = False,
                 record_generation_diagnostics: bool = False) -> list[Completion]:
        self._check_active()
        from transformers import GenerationConfig

        # 新建配置，避免继承模型仓库中的 top-k、重复惩罚、强制词元等生成默认值。
        generation_config = GenerationConfig(
            max_new_tokens=64, do_sample=not greedy, temperature=1.0, top_p=1.0, top_k=0,
            num_beams=1, num_return_sequences=1, use_cache=True,
            eos_token_id=list(self.runtime.eos_token_ids),
            pad_token_id=self.runtime.tokenizer.pad_token_id,
            bos_token_id=self.runtime.tokenizer.bos_token_id,
            return_dict_in_generate=True,
        )
        return self._generate(count, generation_config, record_generation_diagnostics)

    def _generate(self, count, generation_config, record_generation_diagnostics=False):
        """HF 原生配置与采样路径；诊断仅在显式请求时保存完整词元分布。"""
        self.generation_diagnostics = []
        reuse = bool(self.runtime.config.get("generation_prefill_reuse", False)
                     and count == 2 and generation_config.do_sample)
        self.prefill_reuse_audit = {
            "enabled": reuse, "native_prefills": 0, "prefill_replays": 0, "prefill_calls": 0,
        }
        if record_generation_diagnostics:
            generation_config.output_logits = True
            generation_config.output_scores = True
        completions = []
        with self.runtime._model_context(generation=True):
            context = (native_prefill_reuse(self.runtime.base, self.prefill_reuse_audit)
                       if reuse else nullcontext())
            with context:
                for _ in range(count):
                    if record_generation_diagnostics:
                        rng_before = _generation_rng_state()
                    diagnostics = (_generation_prefill_diagnostics(self.runtime.base, self.prefill_reuse_audit)
                                   if record_generation_diagnostics else nullcontext())
                    with diagnostics as prefill:
                        generated = self.runtime.model.generate(**self.inputs, generation_config=generation_config)
                    sequence = generated.sequences[0]
                    if not torch.equal(sequence[:self.prefix_length], self.inputs["input_ids"][0]):
                        raise RuntimeError("HF generate 返回的 prompt 词元前缀发生变化")
                    token_ids = sequence[self.prefix_length:].tolist()
                    stop = next((index for index, token in enumerate(token_ids)
                                 if token in self.runtime.eos_token_ids), None)
                    if stop is not None:
                        token_ids = token_ids[:stop + 1]
                    text = self.runtime.template.decode_generate_ids(token_ids, first_token=False)
                    completions.append(Completion(text, token_ids, "stop" if stop is not None else "length"))
                    if record_generation_diagnostics:
                        rope_deltas = self.runtime.multimodal.rope_deltas
                        self.generation_diagnostics.append({
                            "sequences": generated.sequences.detach().cpu().clone(),
                            "token_ids": list(token_ids), "text": text,
                            "generation_config": generation_config.to_dict(),
                            "logits": tuple(tensor.detach().cpu().clone() for tensor in generated.logits),
                            "scores": tuple(tensor.detach().cpu().clone() for tensor in generated.scores),
                            "rng_before": rng_before, "rng_after": _generation_rng_state(),
                            "prefill": prefill,
                            "cache": generation_cache_snapshot(generated.past_key_values),
                            "rope_deltas": None if rope_deltas is None else rope_deltas.detach().cpu().clone(),
                        })
                    del generated
        if reuse:
            for key in ("native_prefills", "prefill_replays"):
                self.runtime.audit[key] = self.runtime.audit.get(key, 0) + self.prefill_reuse_audit[key]
            self.runtime.audit["prefill_reuse_pairs"] = self.runtime.audit.get("prefill_reuse_pairs", 0) + 1
        return completions

    def logps(self, token_ids: list[int], reference: bool = False) -> torch.Tensor:
        self._check_active()
        if not token_ids:
            raise ValueError("回答词元不能为空")
        runtime = self.runtime
        if self.prefix_length + len(token_ids) > runtime.config.get("max_length", 8192):
            raise ValueError("计分输入超出 max_length")
        answer = torch.tensor([token_ids], dtype=torch.long, device=runtime.device)
        inputs = dict(self.inputs)
        inputs["input_ids"] = torch.cat((inputs["input_ids"], answer), dim=-1)
        inputs["attention_mask"] = torch.cat((inputs["attention_mask"], torch.ones_like(answer)), dim=-1)
        inputs["mm_token_type_ids"] = torch.cat((inputs["mm_token_type_ids"], torch.zeros_like(answer)), dim=-1)
        positions = inputs["position_ids"]
        # 与 HF _update_model_kwargs_for_generation 相同：从原生模板的每根位置轴递增。
        # 保留完整 text + M-RoPE 四轴 prompt，绝不从展平的图像占位词元推断坐标。
        increments = torch.arange(1, len(token_ids) + 1, device=runtime.device)
        inputs["position_ids"] = torch.cat((positions, positions[..., -1:] + increments), dim=-1)
        # 每个回答词元 y[t] 由其前一位置的 logits 预测；EOS 也计入同一个目标。
        selected = torch.arange(self.prefix_length - 1, inputs["input_ids"].shape[1] - 1,
                                device=runtime.device)
        with runtime._model_context(reference=reference):
            output = runtime.model(**inputs, logits_to_keep=selected, use_cache=False, return_dict=True)
            logits = output.logits.squeeze(0).float()
            if logits.shape[0] != len(token_ids):
                raise RuntimeError("原生模型未按 logits_to_keep 保留回答预测位置")
            logps = logits.gather(-1, answer[0, :, None]).squeeze(-1) - logits.logsumexp(-1)
        runtime.audit["reference_forwards" if reference else "policy_forwards"] += 1
        return logps

    def supervised_logps(self, text: str) -> torch.Tensor:
        # 固定空 think 已在 prompt 中，CE 只监督 JSON 及真实的回答结束词元。
        tokens = self.runtime.tokenizer.encode(text, add_special_tokens=False)
        return self.logps([*tokens, self.runtime.answer_eos_id])


def _generation_rng_state():
    import random

    import numpy as np

    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
        "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else [],
    }


@contextmanager
def _generation_prefill_diagnostics(native, audit):
    """仅诊断使用：同时观察基线与复用路径的真实预填充缓存长度。"""
    original = native._prefill
    had_override = "_prefill" in native.__dict__
    events = []

    def record(input_ids, generation_config, model_kwargs, is_first_iteration=True):
        native_before, replays_before = audit["native_prefills"], audit["prefill_replays"]
        output = original(input_ids, generation_config, model_kwargs, is_first_iteration)
        if not audit["enabled"]:
            audit["native_prefills"] += 1
            audit["prefill_calls"] += 1
        events.append({
            "prompt_length": input_ids.shape[1], "cache_length": output.past_key_values.get_seq_length(),
            "native_prefills": audit["native_prefills"] - native_before,
            "prefill_replays": audit["prefill_replays"] - replays_before,
        })
        return output

    native._prefill = record
    try:
        yield events
    finally:
        if had_override:
            native._prefill = original
        else:
            del native._prefill
