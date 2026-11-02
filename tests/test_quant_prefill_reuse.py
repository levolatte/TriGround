"""预填充复用：作用域隔离和真实多模态 Qwen3.5/PEFT CPU 对照。"""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import GenerationConfig

from tools.quant_prefill_reuse import cache_tensor_items, native_prefill_reuse


class PrefillScopeTests(unittest.TestCase):
    def setUp(self):
        class Native:
            def _prefill(self, ids, config, kwargs, is_first_iteration=True):
                self.calls += 1
                layer = SimpleNamespace(keys=torch.arange(6).view(1, 2, 3), values=torch.ones(1, 2, 3))
                cache = SimpleNamespace(layers=[layer], get_seq_length=lambda: ids.shape[1])
                return SimpleNamespace(logits=torch.tensor([[[1., 2.]]]), past_key_values=cache,
                                       extra_native_output={"tensor": torch.tensor([9.])})

        self.native = Native()
        self.native.calls = 0
        self.ids = torch.tensor([[4, 5, 6]])
        self.kwargs = {"attention_mask": torch.ones_like(self.ids), "pixel_values": torch.ones(2, 3)}
        self.config = GenerationConfig(use_cache=True, num_beams=1, num_return_sequences=1)

    def test_full_output_copied_once_and_independent_cache_transferred_once(self):
        rng_before = torch.get_rng_state().clone()
        with patch("tools.quant_prefill_reuse.deepcopy", wraps=deepcopy) as clone:
            with native_prefill_reuse(self.native) as audit:
                first = self.native._prefill(self.ids, self.config, self.kwargs)
                first.past_key_values.layers[0].keys.add_(20)
                first.extra_native_output["tensor"].add_(20)
                second = self.native._prefill(self.ids, self.config, self.kwargs)
                self.assertEqual(second.past_key_values.layers[0].keys.flatten().tolist(), list(range(6)))
                self.assertEqual(second.extra_native_output["tensor"].item(), 9)
                self.assertNotEqual(first.past_key_values.layers[0].keys.data_ptr(),
                                    second.past_key_values.layers[0].keys.data_ptr())
                self.assertEqual((audit["native_prefills"], audit["prefill_replays"]), (1, 1))
            self.assertEqual(clone.call_count, 1)
        self.assertEqual(self.native.calls, 1)
        self.assertNotIn("_prefill", self.native.__dict__)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))

    def test_scope_restores_existing_override_after_error(self):
        override = self.native._prefill
        self.native._prefill = override
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            with native_prefill_reuse(self.native):
                self.native._prefill(self.ids, self.config, self.kwargs)
                raise RuntimeError("interrupted")
        self.assertIs(self.native.__dict__["_prefill"], override)

    def test_exactly_two_calls_and_no_cross_prompt_or_changed_image(self):
        for changed in ("ids", "image", "third", "one"):
            with self.subTest(changed=changed):
                error = ValueError if changed in ("ids", "image") else RuntimeError
                with self.assertRaises(error):
                    with native_prefill_reuse(self.native):
                        self.native._prefill(self.ids, self.config, self.kwargs)
                        if changed == "one":
                            continue
                        ids = self.ids + 1 if changed == "ids" else self.ids
                        if changed == "image":
                            self.kwargs["pixel_values"].add_(1)
                        self.native._prefill(ids, self.config, self.kwargs)
                        if changed == "third":
                            self.native._prefill(ids, self.config, self.kwargs)
                self.assertNotIn("_prefill", self.native.__dict__)

    def test_nested_scopes_fail_and_restore(self):
        with self.assertRaisesRegex(RuntimeError, "嵌套"):
            with native_prefill_reuse(self.native):
                with native_prefill_reuse(self.native):
                    pass
        self.assertNotIn("_prefill", self.native.__dict__)


def tiny_model(dtype):
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    torch.manual_seed(35)
    config = Qwen3_5Config(
        text_config=dict(
            vocab_size=80, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            linear_key_head_dim=8, linear_value_head_dim=8,
            linear_num_key_heads=2, linear_num_value_heads=4,
            layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"],
            partial_rotary_factor=1.0,
            rope_parameters=dict(rope_type="default", rope_theta=10000., mrope_section=[1, 1, 2])),
        vision_config=dict(
            depth=1, hidden_size=16, intermediate_size=32, num_heads=2, patch_size=2,
            temporal_patch_size=2, spatial_merge_size=2, out_hidden_size=32,
            num_position_embeddings=64),
        image_token_id=61, video_token_id=63, vision_start_token_id=60, vision_end_token_id=62,
    )
    native = Qwen3_5ForConditionalGeneration(config).to(dtype=dtype)
    native.config._attn_implementation = "sdpa"
    native.generation_config.eos_token_id = [3]
    lora = LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "in_proj_qkv"],
                      lora_dropout=0., task_type="CAUSAL_LM")
    model = get_peft_model(native, lora).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.copy_(torch.randn_like(parameter) * .03)
    return model, native, lora


def multimodal_inputs(native, image_count, explicit_positions):
    ids, types = [2], [0]
    for _ in range(image_count):
        ids += [60, 61, 61, 61, 61, 62, 7]
        types += [0, 1, 1, 1, 1, 0, 0]
    ids += [9, 10, 11, 12, 13]
    types += [0] * 5
    ids, types = torch.tensor([ids]), torch.tensor([types])
    grid = torch.tensor([[1, 4, 4]] * image_count)
    inputs = dict(input_ids=ids, attention_mask=torch.ones_like(ids), mm_token_type_ids=types,
                  pixel_values=torch.randn(16 * image_count, 24, dtype=native.dtype), image_grid_thw=grid)
    positions, delta = native.model.get_rope_index(ids, image_grid_thw=grid, mm_token_type_ids=types)
    if explicit_positions:
        inputs["position_ids"] = torch.cat((torch.arange(ids.shape[1]).view(1, 1, -1), positions), dim=0)
        native.model.rope_deltas = delta
    return inputs


def cache_equal(test, first, second):
    test.assertEqual(first.get_seq_length(), second.get_seq_length())
    left, right = dict(cache_tensor_items(first)), dict(cache_tensor_items(second))
    test.assertEqual(left.keys(), right.keys())
    for key in left:
        test.assertTrue(torch.equal(left[key], right[key]), key)
        test.assertNotEqual(left[key].data_ptr(), right[key].data_ptr(), key)
    for first_layer, second_layer in zip(first.layers, second.layers):
        for key in ("has_previous_state", "is_conv_states_initialized", "is_recurrent_states_initialized"):
            test.assertEqual(getattr(first_layer, key, None), getattr(second_layer, key, None))


def two_generations(model, native, inputs, config, seed, reuse):
    from contextlib import nullcontext

    torch.manual_seed(seed)
    outputs, starts, traces = [], [], []
    trace = []

    def record(module, args, kwargs):
        cache = kwargs.get("past_key_values")
        trace.append({
            **{key: value.clone() for key, value in kwargs.items()
               if key in ("input_ids", "position_ids", "attention_mask", "mm_token_type_ids")},
            "past_length": cache.get_seq_length() if cache is not None else 0,
        })

    handle = native.register_forward_pre_hook(record, with_kwargs=True)
    audit = {}
    try:
        with torch.no_grad(), native_prefill_reuse(native, audit) if reuse else nullcontext():
            for _ in range(2):
                trace = []
                starts.append(torch.get_rng_state().clone())
                outputs.append(model.generate(**inputs, generation_config=config))
                traces.append(trace)
    finally:
        handle.remove()
    return outputs, starts, torch.get_rng_state().clone(), traces, audit, native.model.rope_deltas.clone()


class NativePrefillCPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_native_qwen_multimodal_fp32_bf16_32_cases(self):
        for dtype in (torch.float32, torch.bfloat16):
            model, native, _ = tiny_model(dtype)
            for explicit in (False, True):
                inputs = multimodal_inputs(native, 2 if explicit else 3, explicit)
                original_inputs = {key: value.clone() for key, value in inputs.items()}
                for greedy in (False, True):
                    config = GenerationConfig(
                        max_new_tokens=12, do_sample=not greedy, temperature=1., top_p=1., top_k=0,
                        num_beams=1, num_return_sequences=1, use_cache=True, eos_token_id=[3],
                        pad_token_id=0, bos_token_id=2, return_dict_in_generate=True,
                        output_logits=True, output_scores=True)
                    for seed in (2026, 2027, 17, 91):
                        with self.subTest(dtype=dtype, explicit=explicit, greedy=greedy, seed=seed):
                            baseline = two_generations(model, native, inputs, config, seed, False)
                            reused = two_generations(model, native, inputs, config, seed, True)
                            for index, (first, second) in enumerate(zip(baseline[0], reused[0])):
                                self.assertTrue(torch.equal(first.sequences, second.sequences))
                                cache_equal(self, first.past_key_values, second.past_key_values)
                                self.assertEqual(len(first.logits), len(second.logits))
                                for key in ("logits", "scores"):
                                    for left, right in zip(getattr(first, key), getattr(second, key)):
                                        self.assertTrue(torch.equal(left, right), key)
                                self.assertTrue(torch.equal(baseline[1][index], reused[1][index]))
                                if not greedy:
                                    saved_rng = torch.get_rng_state()
                                    torch.set_rng_state(baseline[1][index])
                                    sampled = torch.multinomial(first.scores[0].softmax(-1), 1).item()
                                    torch.set_rng_state(saved_rng)
                                    self.assertEqual(sampled, first.sequences[0, inputs["input_ids"].shape[1]].item())
                            self.assertTrue(torch.equal(baseline[2], reused[2]))
                            self.assertTrue(torch.equal(baseline[5], reused[5]))
                            expected_traces = [baseline[3][0], baseline[3][1][1:]]
                            for expected, actual in zip(expected_traces, reused[3]):
                                self.assertEqual(len(expected), len(actual))
                                for left, right in zip(expected, actual):
                                    self.assertEqual(left.keys(), right.keys())
                                    for key in left:
                                        if isinstance(left[key], torch.Tensor):
                                            self.assertTrue(torch.equal(left[key], right[key]), key)
                                        else:
                                            self.assertEqual(left[key], right[key])
                            self.assertEqual((reused[4]["native_prefills"], reused[4]["prefill_replays"]), (1, 1))
                            for key, value in inputs.items():
                                self.assertTrue(torch.equal(value, original_inputs[key]), key)
                            self.assertNotIn("_prefill", native.__dict__)

    def test_runtime_opt_in_diagnostics_preserves_raw_tokens_rng_and_native_caches(self):
        import numpy as np

        from tests.test_quant_correction_runtime import TinyTemplate
        from tools.quant_correction_runtime import CorrectionRuntime

        def equal(left, right):
            if isinstance(left, torch.Tensor):
                self.assertTrue(torch.equal(left, right))
            elif isinstance(left, np.ndarray):
                np.testing.assert_array_equal(left, right)
            elif isinstance(left, dict):
                self.assertEqual(left.keys(), right.keys())
                for key in left:
                    equal(left[key], right[key])
            elif isinstance(left, (tuple, list)):
                self.assertEqual(len(left), len(right))
                for first, second in zip(left, right):
                    equal(first, second)
            else:
                self.assertEqual(left, right)

        for dtype in (torch.float32, torch.bfloat16):
            model, native, lora = tiny_model(dtype)
            model.add_adapter("reference", lora)
            native.generation_config.eos_token_id = [2, 3]
            for image_count, rope_initialized in ((2, False), (3, False), (2, True), (3, True)):
                inputs = multimodal_inputs(native, image_count, explicit_positions=True)
                self.assertEqual(inputs["position_ids"].shape, (4, 1, inputs["input_ids"].shape[1]))
                # Swift 显式传入四轴位置时，原生模型可以从未建立 rope_deltas。
                if not rope_initialized:
                    native.model.rope_deltas = None

                class NativeTemplate(TinyTemplate):
                    def encode(self, row):
                        return inputs

                    def data_collator(self, batch):
                        return batch[0]

                with patch("tools.quant_correction_runtime._load_components", return_value=(model, NativeTemplate())):
                    runtime = CorrectionRuntime({"media_root": ".", "max_length": 128}, "correction")
                row = {"messages": [{"role": "user", "content": "<image>" * image_count + " locate"}],
                       "images": [f"image-{index}.png" for index in range(image_count)]}
                for seed in (2026, 2027):
                    with self.subTest(dtype=dtype, image_count=image_count, rope_initialized=rope_initialized, seed=seed):
                        with runtime.question(row) as question:
                            runtime.config["generation_prefill_reuse"] = False
                            torch.manual_seed(seed)
                            baseline = question.generate(record_generation_diagnostics=True)
                            original_diagnostics = question.generation_diagnostics
                            self.assertEqual(question.prefill_reuse_audit["native_prefills"], 2)
                            runtime.config["generation_prefill_reuse"] = True
                            torch.manual_seed(seed)
                            actual = question.generate(record_generation_diagnostics=True)
                            self.assertEqual(baseline, actual)
                            for index, (first, second) in enumerate(zip(original_diagnostics, question.generation_diagnostics)):
                                if rope_initialized:
                                    self.assertIsInstance(first["rope_deltas"], torch.Tensor)
                                else:
                                    self.assertIsNone(first["rope_deltas"])
                                    self.assertIsNone(second["rope_deltas"])
                                for key in first.keys() - {"prefill"}:
                                    equal(first[key], second[key])
                                self.assertEqual(second["generation_config"]["max_new_tokens"], 64)
                                self.assertEqual(first["prefill"][0]["cache_length"], question.prefix_length)
                                self.assertEqual(second["prefill"][0]["native_prefills"], int(index == 0))
                                self.assertEqual(second["prefill"][0]["prefill_replays"], int(index == 1))
                            self.assertEqual(question.prefill_reuse_audit["native_prefills"], 1)
                            self.assertEqual(question.prefill_reuse_audit["prefill_replays"], 1)
                            self.assertNotIn("_prefill", native.__dict__)
                            # 正式路径不保留诊断张量，也不会在调用之间保留 prompt 缓存。
                            question.generate()
                            self.assertEqual(question.generation_diagnostics, [])
                            self.assertNotIn("_prefill", native.__dict__)
                        self.assertEqual(model.active_adapter, "default")
                        self.assertTrue(model.training)


if __name__ == "__main__":
    unittest.main()
