"""CPU：原始词元因果对齐、冻结参考分支和逐题视觉缓存。"""

from __future__ import annotations

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from tools.quant_correction_runtime import CorrectionRuntime, SHORT_RESPONSE_PREFIX


class TinyLanguage(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 5)
        self.lora_A = nn.ModuleDict({name: nn.Linear(5, 5, bias=False) for name in ("default", "reference")})
        self.checkpoint_kwargs = None

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs):
        self.checkpoint_kwargs = gradient_checkpointing_kwargs


class TinyMultimodal(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = TinyLanguage()
        self.visual = nn.Linear(3, 5, bias=False)
        self.visual_calls = 0
        self.last_features = None

    def get_image_features(self, pixel_values, image_grid_thw, return_dict):
        self.visual_calls += 1
        self.last_features = self.visual(pixel_values)
        return SimpleNamespace(pooler_output=(self.last_features,))


class TinyNative(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyMultimodal()
        self.lm_head = nn.Linear(5, 16, bias=False)
        self.generation_config = SimpleNamespace(eos_token_id=[2, 3])

    def get_input_embeddings(self):
        return self.model.language_model.embed_tokens


class TinyPeft(nn.Module):
    def __init__(self):
        super().__init__()
        self.native = TinyNative()
        self.active_adapter = "default"
        self.calls = []
        self.generations = []
        self.saved = None

    def get_base_model(self):
        return self.native

    def set_adapter(self, adapter):
        self.active_adapter = adapter
        # 仿真 PEFT 激活 adapter 时自动把它改成可训练，runtime 应再次冻结 reference。
        for name, module in self.native.model.language_model.lora_A.items():
            module.requires_grad_(name == adapter)

    def forward(self, **inputs):
        features = self.native.model.get_image_features(
            inputs["pixel_values"], inputs["image_grid_thw"], return_dict=True).pooler_output[0]
        language = self.native.model.language_model
        hidden = language.embed_tokens(inputs["input_ids"]) + features.mean(0)
        hidden = hidden + language.lora_A[self.active_adapter](hidden)
        full_logits = self.native.lm_head(hidden)
        selected_logits = full_logits[:, inputs["logits_to_keep"], :]
        self.calls.append({**inputs, "adapter": self.active_adapter,
                           "logits": full_logits.detach().clone(),
                           "reference_frozen": not any(p.requires_grad for p in language.lora_A["reference"].parameters())})
        return SimpleNamespace(logits=selected_logits)

    def generate(self, **inputs):
        self.native.model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"], return_dict=True)
        self.generations.append(inputs)
        # 包含 EOS 后的 padding，runtime 应只保留首个 EOS 及此前原始词元。
        suffix = torch.tensor([[7, 8, 2, 0]], device=inputs["input_ids"].device)
        return SimpleNamespace(sequences=torch.cat((inputs["input_ids"], suffix), dim=-1))

    def save_pretrained(self, path, **kwargs):
        self.saved = (path, kwargs)


class TinyTokenizer:
    pad_token_id = 0
    bos_token_id = 1

    def __init__(self):
        self.encoded = []

    def convert_tokens_to_ids(self, token):
        assert token == "<|im_end|>"
        return 2

    def encode(self, text, add_special_tokens):
        self.encoded.append((text, add_special_tokens))
        return [9, 10]


class TinyTemplate:
    def __init__(self):
        self.tokenizer = TinyTokenizer()
        self.encoded_prompt = None
        self.model = None

    def set_mode(self, mode):
        self.mode = mode

    @torch.inference_mode()
    def encode(self, row):
        self.encoded_prompt = row
        return {"input_ids": [4, 5, 6], "pixel_values": torch.ones(2, 3),
                "image_grid_thw": torch.tensor([[1, 1, 2]])}

    def data_collator(self, batch):
        row = batch[0]
        return {"input_ids": torch.tensor([row["input_ids"]]), "attention_mask": torch.ones(1, 3, dtype=torch.long),
                "mm_token_type_ids": torch.tensor([[0, 1, 0]]),
                # text + 3D 多模态坐标，不能用展平图像位置重建。
                "position_ids": torch.tensor([[[0, 1, 2]], [[0, 2, 3]], [[0, 4, 5]], [[0, 6, 7]]]),
                "pixel_values": row["pixel_values"], "image_grid_thw": row["image_grid_thw"],
                "labels": torch.tensor([[-100, -100, -100]])}

    def decode_generate_ids(self, tokens, first_token):
        assert not first_token
        return '{"bbox_2d":[1,2,3,4]}'


class CorrectionRuntimeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.model = TinyPeft()
        self.template = TinyTemplate()
        self.config = {"media_root": ".", "max_length": 128}
        with patch("tools.quant_correction_runtime._load_components", return_value=(self.model, self.template)):
            self.runtime = CorrectionRuntime(self.config, "correction")
        self.row = {"id": "q1", "messages": [{"role": "user", "content": "<image> locate"}],
                    "images": ["rgb.png"], "solution": [0.1, 0.2, 0.3, 0.4]}

    def test_selected_logits_predict_raw_tokens_and_eos_at_previous_positions(self):
        with self.runtime.question(self.row) as question:
            completion = question.generate()[0]
            actual = question.logps(completion.token_ids)
            call = self.model.calls[-1]
            self.assertEqual(completion.token_ids, [7, 8, 2])
            self.assertEqual(completion.finish_reason, "stop")
            self.assertEqual(call["input_ids"].tolist(), [[4, 5, 6, 7, 8, 2]])
            self.assertEqual(call["logits_to_keep"].tolist(), [2, 3, 4])
            expected = call["logits"][0, [2, 3, 4]].log_softmax(-1)[torch.arange(3), torch.tensor([7, 8, 2])]
            torch.testing.assert_close(actual, expected)
            self.assertEqual(actual.dtype, torch.float32)
            self.assertTrue(actual.requires_grad)
            self.assertEqual(self.template.tokenizer.encoded, [])  # 没有 decode 后重新分词。
            self.assertEqual(call["position_ids"].tolist(),
                             [[[0, 1, 2, 3, 4, 5]], [[0, 2, 3, 4, 5, 6]],
                              [[0, 4, 5, 6, 7, 8]], [[0, 6, 7, 8, 9, 10]]])
            self.assertEqual(call["mm_token_type_ids"].tolist(), [[0, 1, 0, 0, 0, 0]])

    def test_two_generations_policy_reference_and_gt_share_one_normal_visual_tensor(self):
        with self.runtime.question(self.row) as question:
            completions = question.generate()
            policy = question.logps(completions[0].token_ids)
            reference = question.logps(completions[0].token_ids, reference=True)
            supervised = question.supervised_logps('{"bbox_2d":[100,200,300,400]}')
            self.assertFalse(reference.requires_grad)
            self.assertTrue(self.model.calls[-2]["reference_frozen"])
            self.assertEqual(self.model.active_adapter, "default")
            self.assertFalse(torch.is_inference(self.model.native.model.last_features))
            (-policy.mean() - supervised.mean()).backward()
            named = dict(self.model.named_parameters())
            trainable = {name for name, parameter in named.items() if parameter.requires_grad}
            self.assertEqual(trainable, {"native.model.language_model.lora_A.default.weight"})
            self.assertTrue(torch.count_nonzero(named[next(iter(trainable))].grad) > 0)
            self.assertTrue(all(parameter.grad is None for name, parameter in named.items() if name not in trainable))
            self.assertEqual(self.model.native.model.visual_calls, 1)
            self.assertEqual(self.runtime.audit["visual_encodes"], 1)
            self.assertEqual(self.template.encoded_prompt.keys(), {"messages", "images", "chat_template_kwargs"})
            self.assertEqual(self.model.calls[-1]["input_ids"].tolist(), [[4, 5, 6, 9, 10, 2]])
        self.assertNotIn("get_image_features", self.model.native.model.__dict__)
        with self.runtime.question(self.row) as question:
            question.generate(count=1, greedy=True)
        self.assertEqual(self.model.native.model.visual_calls, 2)

    def test_generation_sampling_and_checkpoint_are_explicit(self):
        self.assertEqual(SHORT_RESPONSE_PREFIX, "<think>\n\n</think>\n\n")
        self.assertEqual(self.model.native.model.language_model.checkpoint_kwargs, {"use_reentrant": False})
        with self.runtime.question(self.row) as question:
            question.generate()
        self.assertEqual(len(self.model.generations), 2)
        for generated in self.model.generations:
            config = generated["generation_config"]
            self.assertEqual((config.max_new_tokens, config.temperature, config.top_p, config.top_k), (64, 1, 1, 0))
            self.assertTrue(config.do_sample)
            self.assertNotIn("labels", generated)

    def test_prefill_reuse_is_opt_in_and_only_for_two_sampled_answers(self):
        with patch("tools.quant_correction_runtime.native_prefill_reuse", return_value=nullcontext()) as reuse:
            with self.runtime.question(self.row) as question:
                question.generate()
                reuse.assert_not_called()
                self.runtime.config["generation_prefill_reuse"] = True
                for count, greedy in ((1, False), (3, False), (2, True)):
                    question.generate(count=count, greedy=greedy)
                reuse.assert_not_called()
                question.generate()
                reuse.assert_called_once_with(self.runtime.base, question.prefill_reuse_audit)
                self.assertTrue(question.prefill_reuse_audit["enabled"])
                self.assertEqual(question.generation_diagnostics, [])
        self.assertTrue(self.model.training)
        self.assertEqual(self.model.active_adapter, "default")

    def test_cache_restores_on_error_and_question_handle_expires(self):
        with self.assertRaisesRegex(RuntimeError, "expected"):
            with self.runtime.question(self.row) as question:
                raise RuntimeError("expected")
        self.assertNotIn("get_image_features", self.model.native.model.__dict__)
        with self.assertRaisesRegex(RuntimeError, "question"):
            question.logps([7, 2])

    def test_input_rejects_assistant_and_full_generation_budget_overflow(self):
        answered = {**self.row, "messages": [*self.row["messages"], {"role": "assistant", "content": "GT"}]}
        with self.assertRaisesRegex(ValueError, "assistant"):
            with self.runtime.question(answered):
                pass
        self.runtime.config["max_length"] = 66
        with self.assertRaisesRegex(ValueError, "64"):
            with self.runtime.question(self.row):
                pass

    def test_save_excludes_reference_adapter_and_embeddings(self):
        self.runtime.save_adapter("checkpoint-1")
        self.assertEqual(self.model.saved, ("checkpoint-1", {
            "selected_adapters": ["default"], "safe_serialization": True, "save_embedding_layers": False,
        }))


class NativeQwenRuntimeTests(unittest.TestCase):
    def test_native_qwen_tensor_logit_selection_checkpoint_and_frozen_reference(self):
        """真实小尺寸 Qwen3.5（含 GDN/注意力/视觉）验证 API 与反向传播。"""
        from peft import LoraConfig, get_peft_model
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

        torch.manual_seed(29)
        config = Qwen3_5Config(
            text_config=dict(
                vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                linear_key_head_dim=8, linear_value_head_dim=8,
                linear_num_key_heads=1, linear_num_value_heads=2,
                layer_types=["linear_attention", "full_attention"], partial_rotary_factor=1.0,
                rope_parameters=dict(rope_type="default", rope_theta=10000.0, mrope_section=[1, 1, 2])),
            vision_config=dict(
                depth=1, hidden_size=8, intermediate_size=16, num_heads=2, patch_size=2,
                temporal_patch_size=2, spatial_merge_size=2, out_hidden_size=16,
                num_position_embeddings=16),
            image_token_id=13, video_token_id=15, vision_start_token_id=12, vision_end_token_id=14,
        )
        native = Qwen3_5ForConditionalGeneration(config)
        native.generation_config.eos_token_id = [2, 3]
        lora = LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "in_proj_qkv"],
                          lora_dropout=0, task_type="CAUSAL_LM")
        model = get_peft_model(native, lora)
        model.add_adapter("reference", lora)

        class NativeTemplate(TinyTemplate):
            @torch.inference_mode()
            def encode(self, row):
                return {"input_ids": [12, 13, 14, 4, 5], "pixel_values": torch.randn(4, 24),
                        "image_grid_thw": torch.tensor([[1, 2, 2]])}

            def data_collator(self, batch):
                data = batch[0]
                ids = torch.tensor([data["input_ids"]])
                types = torch.tensor([[0, 1, 0, 0, 0]])
                positions, _ = native.model.get_rope_index(
                    ids, image_grid_thw=data["image_grid_thw"], mm_token_type_ids=types)
                return {**data, "input_ids": ids, "attention_mask": torch.ones_like(ids),
                        "mm_token_type_ids": types,
                        "position_ids": torch.cat((torch.arange(5).view(1, 1, 5), positions), dim=0)}

        template = NativeTemplate()
        with patch("tools.quant_correction_runtime._load_components", return_value=(model, template)):
            runtime = CorrectionRuntime({"media_root": ".", "max_length": 128}, "correction")
        row = {"messages": [{"role": "user", "content": "<image> locate"}], "images": ["rgb.png"]}
        vision_calls = []
        handle = native.model.visual.register_forward_hook(lambda *args: vision_calls.append(1))
        try:
            with runtime.question(row) as question:
                policy = question.logps([7, 8, 2])
                reference = question.logps([7, 8, 2], reference=True)
                # 独立全词表、全位置前向，使用原生位置计算作为因果对齐参照。
                full_inputs = {key: value for key, value in question.inputs.items() if key != "position_ids"}
                full_inputs["input_ids"] = torch.tensor([[12, 13, 14, 4, 5, 7, 8, 2]])
                full_inputs["attention_mask"] = torch.ones(1, 8, dtype=torch.long)
                full_inputs["mm_token_type_ids"] = torch.tensor([[0, 1, 0, 0, 0, 0, 0, 0]])
                with torch.no_grad():
                    all_logits = model(**full_inputs, use_cache=False).logits.float()
                    expected = all_logits[0, [4, 5, 6]].log_softmax(-1)[torch.arange(3), torch.tensor([7, 8, 2])]
                torch.testing.assert_close(policy, expected)
                self.assertFalse(reference.requires_grad)
                (-policy.mean()).backward()
                grads = [parameter.grad for parameter in runtime.trainable_parameters()]
                self.assertTrue(any(gradient is not None and gradient.count_nonzero() > 0 for gradient in grads))
                self.assertTrue(all(parameter.grad is None for name, parameter in model.named_parameters()
                                    if ".reference." in name or ".visual." in name))
                self.assertEqual(len(vision_calls), 1)
        finally:
            handle.remove()


if __name__ == "__main__":
    unittest.main()
