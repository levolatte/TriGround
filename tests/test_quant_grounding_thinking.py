"""CPU-only trajectory and action-mask checks for bounded D generation."""

import unittest
from unittest.mock import patch
from types import SimpleNamespace
import sys
import types

import numpy as np

from tools.quant_grounding_thinking import (BOUNDED_THINKING_SYSTEM, BoundedHFGenerationHook,
                                             ThinkingBudgetProcessor, ThinkingTrace, apply_bounded_thinking_prompt,
                                             answer_state_mask, suppress_answer_close,
                                             install_bounded_thinking_training,
                                             overlay_grpo_action_masks, overlay_grpo_rollout_logps)


PAD, EOS, CLOSE = 0, 2, 9


def rollout(processor, schedules):
    """Simulate the HF one-token sampling loop with mixed finished rows."""
    ids = np.array([[PAD, 41, 42]] * len(schedules), dtype=np.int64)
    generated = [[] for _ in schedules]
    for step in range(processor.max_new_tokens):
        scores = np.zeros((len(schedules), 64), dtype=np.float32)
        scores = processor(ids, scores)
        next_tokens = []
        for row, schedule in enumerate(schedules):
            if EOS in generated[row]:
                token = PAD
            elif np.count_nonzero(np.isfinite(scores[row])) == 1:
                token = int(np.argmax(scores[row]))
            else:
                token = schedule[min(step, len(schedule) - 1)]
                if not np.isfinite(scores[row, token]):
                    token = 11
            generated[row].append(token)
            next_tokens.append(token)
        ids = np.concatenate((ids, np.array(next_tokens, dtype=np.int64)[:, None]), axis=1)
        if all(EOS in row for row in generated):
            break
    return [processor.trace(row, ids[row, 3:].tolist()) for row in range(len(schedules))]


def torch_schedule_model(schedules, *, extra_processor=None):
    """Small real-tensor sampler with the same causal processor call order as HF."""
    import torch

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.linspace(-0.7, 0.7, 16))

        @staticmethod
        def _get_logits_processor(*, logits_processor):
            if extra_processor is not None:
                logits_processor.append(extra_processor)
            return logits_processor

        def forward(self, input_ids, logits_to_keep=None, use_cache=None):
            width = min(input_ids.shape[1], logits_to_keep or input_ids.shape[1])
            return SimpleNamespace(logits=self.weight.expand(input_ids.shape[0], width, -1))

        def generate(self, *, input_ids, generation_config, logits_processor, stopping_criteria=None):
            processors = self._get_logits_processor(logits_processor=logits_processor)
            ids = input_ids
            generated = [[] for _ in schedules]
            for step in range(generation_config.max_new_tokens):
                # HF consumes RNG even when the next token is deterministically
                # forced; replay must restore that stream for later rollouts.
                torch.rand(())
                scores = self.weight.detach().expand(len(schedules), -1).clone()
                for processor in processors:
                    scores = processor(ids, scores)
                chosen = []
                for row, schedule in enumerate(schedules):
                    if EOS in generated[row]:
                        token = PAD
                    elif torch.isfinite(scores[row]).sum().item() == 1:
                        token = int(scores[row].argmax().item())
                    else:
                        token = schedule[min(step, len(schedule) - 1)]
                        if not torch.isfinite(scores[row, token]).item():
                            token = 11
                    generated[row].append(token)
                    chosen.append(token)
                ids = torch.cat((ids, torch.tensor(chosen)[:, None]), dim=1)
                if stopping_criteria is not None and stopping_criteria(ids, None).all().item():
                    break
                if all(EOS in row for row in generated):
                    break
            return SimpleNamespace(sequences=ids)

    return Model()


def tiny_tokenizer():
    return SimpleNamespace(eos_token_id=EOS, pad_token_id=PAD,
                           encode=lambda text, add_special_tokens=False: [CLOSE])


class BoundedGenerationTest(unittest.TestCase):
    def test_answer_state_shift_and_restricted_logprob_gradient(self):
        import torch

        ids = torch.tensor([[41, 7, CLOSE, 11, EOS, PAD],
                            [41, 7, 8, CLOSE, 12, EOS]])
        states = answer_state_mask(ids, logits_to_keep=5, close_token_id=CLOSE)
        self.assertEqual(states.tolist(), [[False, False, True, True, True],
                                          [False, False, False, True, True]])
        source = torch.linspace(-2, 2, 2 * 5 * 16, dtype=torch.float64).reshape(2, 5, 16)
        source.requires_grad_()
        logits = suppress_answer_close(source / 1, states, close_token_id=CLOSE)
        labels = ids[:, -5:]
        selected = logits.log_softmax(-1).gather(-1, labels[..., None]).squeeze(-1)
        gradients, = torch.autograd.grad(selected.sum(), source)
        for row in range(2):
            for position in range(5):
                allowed = [token for token in range(16)
                           if not states[row, position] or token != CLOSE]
                expected = source[row, position, labels[row, position]] - torch.logsumexp(
                    source[row, position, allowed], dim=-1)
                torch.testing.assert_close(selected[row, position], expected, atol=1e-12, rtol=1e-12)
                expected_gradient = torch.zeros(16, dtype=torch.float64)
                expected_gradient[allowed] = -source[row, position, allowed].softmax(-1)
                expected_gradient[labels[row, position]] += 1
                torch.testing.assert_close(gradients[row, position], expected_gradient,
                                           atol=1e-12, rtol=1e-12)
                if states[row, position]:
                    self.assertEqual(gradients[row, position, CLOSE].item(), 0.0)
        self.assertNotEqual(gradients[0, 0, CLOSE].item(), 0.0)

    def test_real_chosen_logprobs_include_last_eos_and_finished_row_padding(self):
        import torch

        model = torch_schedule_model([[7, CLOSE, CLOSE, EOS], [7, 8, 11, 12, 13, 14], [7, EOS]])
        engine = SimpleNamespace(model=model, tokenizer=tiny_tokenizer())
        config = SimpleNamespace(num_return_sequences=1, num_beams=1, max_new_tokens=6,
                                 eos_token_id=EOS, pad_token_id=PAD)
        with BoundedHFGenerationHook(engine, thought_limit=2, answer_limit=2,
                                     capture_logprobs=True) as hook:
            model.generate(input_ids=torch.tensor([[PAD, 41, 42]] * 3), generation_config=config)
            natural, forced, early = hook.pop_traces()
        self.assertEqual(natural.token_ids, (7, CLOSE, 11, EOS))
        self.assertEqual(forced.token_ids, (7, 8, CLOSE, 12, 13, EOS))
        self.assertEqual(forced.forced_positions, (2, 5))
        self.assertEqual(early.token_ids, (7, EOS))
        self.assertEqual(early.finish_reason, 'eos_in_thought')
        for trace in [natural, forced, early]:
            self.assertEqual(len(trace.sampled_logprobs), len(trace.token_ids))
            self.assertIn('ThinkingBudgetProcessor', trace.generation_processors)
            close_seen = False
            for position, token in enumerate(trace.token_ids):
                if position in trace.forced_positions:
                    expected = 0.0
                else:
                    logits = model.weight.detach().clone()
                    if close_seen:
                        logits[CLOSE] = -float('inf')
                    expected = logits.log_softmax(-1)[token].item()
                self.assertAlmostEqual(trace.sampled_logprobs[position], expected, places=6)
                close_seen |= token == CLOSE

    def test_actual_transformers_processor_order_and_chosen_logprobs(self):
        import torch
        from transformers import GPT2Config, GPT2LMHeadModel, GenerationConfig

        torch.manual_seed(2026)
        model = GPT2LMHeadModel(GPT2Config(vocab_size=64, n_positions=16, n_embd=8,
                                         n_layer=1, n_head=1, eos_token_id=EOS,
                                         pad_token_id=PAD, bos_token_id=1)).eval()
        engine = SimpleNamespace(model=model, tokenizer=tiny_tokenizer())
        config = GenerationConfig(do_sample=True, temperature=1, top_p=1, top_k=0,
                                  max_new_tokens=6, eos_token_id=EOS, pad_token_id=PAD,
                                  num_return_sequences=1, num_beams=1,
                                  suppress_tokens=[PAD], return_dict_in_generate=True)
        prompt = torch.tensor([[41, 42], [41, 42]])
        with BoundedHFGenerationHook(engine, thought_limit=2, answer_limit=2,
                                     capture_logprobs=True) as hook:
            model.generate(input_ids=prompt, generation_config=config)
            traces = hook.pop_traces()
        for row, trace in enumerate(traces):
            self.assertIn('SuppressTokensLogitsProcessor', trace.generation_processors)
            self.assertEqual(trace.generation_processors[-1], 'ThinkingBudgetProcessor')
            self.assertEqual(len(trace.sampled_logprobs), len(trace.token_ids))
            with torch.no_grad():
                for position, token in enumerate(trace.token_ids):
                    if position in trace.forced_positions:
                        expected = 0.0
                    else:
                        prefix = torch.tensor([prompt[row].tolist() + list(trace.token_ids[:position])])
                        logits = model(input_ids=prefix).logits[0, -1].float()
                        logits[PAD] = -float('inf')
                        if CLOSE in trace.token_ids[:position]:
                            logits[CLOSE] = -float('inf')
                        expected = logits.log_softmax(-1)[token].item()
                    self.assertAlmostEqual(trace.sampled_logprobs[position], expected, places=6)

    def test_repeat_close_is_blocked_after_natural_and_forced_boundaries(self):
        processor = ThinkingBudgetProcessor(prompt_width=3, batch_size=2, close_token_id=CLOSE,
                                            eos_token_id=EOS, pad_token_id=PAD,
                                            thought_limit=2, answer_limit=2)
        natural, forced = rollout(processor, [[7, CLOSE, CLOSE, EOS], [7, 8, 7, CLOSE, 12, 7]])
        self.assertEqual(natural.token_ids, (7, CLOSE, 11, EOS))
        self.assertEqual(forced.token_ids, (7, 8, CLOSE, 11, 12, EOS))
        for trace in [natural, forced]:
            self.assertEqual(trace.token_ids.count(CLOSE), 1)
        with self.assertRaisesRegex(ValueError, 'more than one closing delimiter'):
            processor.trace(0, [7, CLOSE, CLOSE, EOS])

    def test_same_context_replay_runs_once_and_restores_rng(self):
        import torch

        config = SimpleNamespace(num_return_sequences=1, num_beams=1, max_new_tokens=6,
                                 eos_token_id=EOS, pad_token_id=PAD)
        tails = []
        for replay in [0, 4]:
            torch.manual_seed(2026)
            model = torch_schedule_model([[7, CLOSE, 11, EOS], [7, 8, 11, 12, 13, 14]])
            engine = SimpleNamespace(model=model, tokenizer=tiny_tokenizer())
            with BoundedHFGenerationHook(engine, thought_limit=2, answer_limit=2,
                                         capture_logprobs=True, sampling_replay_tokens=replay) as hook:
                for _ in range(2):
                    model.generate(input_ids=torch.tensor([[PAD, 41, 42]] * 2), generation_config=config)
                    hook.pop_traces()
                if replay:
                    self.assertEqual(hook.replay_audit['status'], 'passed')
                    self.assertEqual(hook.replay_audit['tokens'], 7)
                    self.assertEqual(hook.replay_audit['context']['batch_size'], 2)
                    self.assertLess(hook.replay_audit['max_abs_gap'], 1e-6)
            tails.append(torch.rand(5))
        torch.testing.assert_close(*tails, atol=0, rtol=0)

    def test_real_swift_mutable_stop_state_replays_all_64_tokens(self):
        import torch
        from swift.template.utils import StopWordsCriteria
        from transformers import StoppingCriteriaList

        class Tokenizer:
            eos_token_id = EOS
            pad_token_id = PAD

            def encode(self, text, add_special_tokens=False):
                return [CLOSE]

            def batch_decode(self, rows, **kwargs):
                return [' '.join(map(str, row.tolist())) for row in rows]

            def __deepcopy__(self, memo):
                raise AssertionError('the tokenizer must be shared when cloning stopping criteria')

        tokenizer = Tokenizer()
        criteria = StopWordsCriteria(tokenizer, [[EOS]])
        model = torch_schedule_model([[7] * 64 + [7, EOS], [8] * 64 + [8, EOS]])
        engine = SimpleNamespace(model=model, tokenizer=tokenizer)
        config = SimpleNamespace(num_return_sequences=1, num_beams=1, max_new_tokens=68,
                                 eos_token_id=EOS, pad_token_id=PAD)
        with BoundedHFGenerationHook(engine, thought_limit=64, answer_limit=2,
                                     capture_logprobs=True, sampling_replay_tokens=64) as hook:
            model.generate(input_ids=torch.tensor([[PAD, 41, 42]] * 2), generation_config=config,
                           stopping_criteria=StoppingCriteriaList([criteria]))
            self.assertTrue(criteria.is_done.all().item())
            self.assertEqual(hook.replay_audit['status'], 'passed')
            self.assertEqual(hook.replay_audit['expected_width'], 64)
            self.assertEqual(hook.replay_audit['observed_width'], 64)
            self.assertEqual(hook.replay_audit['expected_action_tokens'], 128)
            self.assertEqual(hook.replay_audit['tokens'], 128)
        # The same real Swift criterion, reused after completion, reproduces
        # the original one-step replay failure without the fresh-state copy.
        stale = model.generate(input_ids=torch.tensor([[PAD, 41, 42]] * 2), generation_config=config,
                               logits_processor=[], stopping_criteria=StoppingCriteriaList([criteria]))
        self.assertEqual(stale.sequences.shape[1] - 3, 1)

    def test_language_amp_scope_preserves_visual_and_restores_forwards(self):
        import torch

        class Qwen3_5GatedDeltaNet(torch.nn.Linear):
            pass

        class Language(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gdn = Qwen3_5GatedDeltaNet(4, 4)

            def forward(self, value):
                return self.gdn(value)

        language = Language()
        visual = torch.nn.Linear(4, 4)
        head = torch.nn.Linear(4, 4)
        owner = SimpleNamespace(model=SimpleNamespace(language_model=language, visual=visual), lm_head=head)
        hook = BoundedHFGenerationHook(
            SimpleNamespace(tokenizer=tiny_tokenizer()),
            language_autocast_factory=lambda: torch.autocast('cpu', dtype=torch.bfloat16))
        original_language, original_head = language.forward, head.forward
        output = hook._run_generate(owner, lambda **kw: head(language(visual(torch.ones(1, 4)))),
                                    {'input_ids': torch.tensor([[7]])})
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(language.forward, original_language)
        self.assertEqual(head.forward, original_head)
        self.assertEqual(hook.amp_audit['status'], 'passed')
        self.assertTrue(hook.amp_audit['observations']['gdn_internal_autocast'])
        self.assertFalse(hook.amp_audit['observations']['visual_autocast'])
        self.assertEqual(hook.amp_audit['autocast_dtype'], 'torch.bfloat16')

    def test_long_thought_budget_keeps_answer_reservation(self):
        processor = ThinkingBudgetProcessor(prompt_width=3, batch_size=1, close_token_id=CLOSE,
                                            eos_token_id=EOS, pad_token_id=PAD,
                                            thought_limit=1024, answer_limit=64)
        trace = rollout(processor, [[7] * 1200])[0]
        self.assertEqual(processor.max_new_tokens, 1090)
        self.assertEqual(trace.forced_positions, (1024, 1089))
        self.assertEqual(trace.thought_tokens, 1024)
        self.assertEqual(trace.answer_tokens, 64)
        self.assertEqual(sum(trace.action_mask), 1088)
        self.assertTrue(trace.answer_truncated)

    def test_training_recomputes_same_support_for_old_current_ref_and_rejects_other_transforms(self):
        import torch

        class ExtraBias:
            def __call__(self, input_ids, scores):
                scores[:, 11] -= 1.0
                return scores

        for extra in [None, ExtraBias()]:
            with self.subTest(extra_processor=type(extra).__name__):
                model = torch_schedule_model([[7, CLOSE, CLOSE, EOS],
                                              [7, 8, 11, 12, 13, 14], [7, EOS]],
                                             extra_processor=extra)
                grpo_module = types.ModuleType('swift.rlhf_trainers.grpo_trainer')
                grpo_module.selective_log_softmax = lambda logits, labels: (
                    logits.log_softmax(-1).gather(-1, labels[..., None]).squeeze(-1))

                def collate(samples, *args, **kwargs):
                    rows = [sample.response_token_ids[0] for sample in samples]
                    width = max(map(len, rows))
                    ids = torch.tensor([[PAD, 41, 42] + row + [PAD] * (width - len(row)) for row in rows])
                    mask = torch.tensor([[True] * len(row) + [False] * (width - len(row)) for row in rows])
                    return {'input_ids': ids}, SimpleNamespace(completion_mask=mask)

                grpo_module.collate_to_grpo_micro_batch = collate
                swift = types.ModuleType('swift')
                swift.__path__ = []
                rlhf = types.ModuleType('swift.rlhf_trainers')
                rlhf.__path__ = []
                rlhf.grpo_trainer = grpo_module
                swift.rlhf_trainers = rlhf

                class Trainer:
                    use_fast_infer = False
                    async_generate = False
                    multi_turn_scheduler = None
                    template = SimpleNamespace(padding_free=False, sequence_parallel_size=1)
                    accelerator = SimpleNamespace(num_processes=1, autocast=None)
                    model_kwarg_keys = {'use_cache', 'logits_to_keep'}
                    temperature = 1.0
                    rollout_importance_sampling_mode = 'token_truncate'
                    rollout_importance_sampling_threshold = 2.0
                    disable_rollout_importance_sampling = False

                    @staticmethod
                    def _get_rollout_is_correction(old, behavior, mask):
                        gap = old-behavior
                        return gap, gap.clamp(-20, 20).exp().clamp(max=2)

                    def __init__(self):
                        self.model = model
                        self.engine = SimpleNamespace(model=model, tokenizer=tiny_tokenizer())

                    @staticmethod
                    def _get_logps_via_local_forward(*args, **kwargs):
                        raise AssertionError('unrestricted original logprob path was called')

                    def _generate_completions(self, samples):
                        generated = self.engine.model.generate(
                            input_ids=torch.tensor([[PAD, 41, 42]] * len(samples)),
                            generation_config=SimpleNamespace(
                                num_return_sequences=1, num_beams=1, max_new_tokens=6,
                                eos_token_id=EOS, pad_token_id=PAD)).sequences[:, 3:].tolist()
                        for sample, row in zip(samples, generated):
                            sample.response_token_ids = [row[:row.index(EOS) + 1]]
                        return samples

                    def _prepare_batch_inputs(self, samples):
                        inputs, batch = grpo_module.collate_to_grpo_micro_batch(samples)
                        width = batch.completion_mask.shape[1]
                        with torch.no_grad():
                            batch.old_per_token_logps = self._get_logps_via_local_forward(
                                self.model, inputs, width, inputs['input_ids'])[0]
                            batch.ref_per_token_logps = self._get_logps_via_local_forward(
                                self.model, inputs, width, inputs['input_ids'])[0]
                        batch.num_items_in_batch = batch.completion_mask.sum()
                        return [{'model_inputs': inputs, 'grpo_batch': batch}]

                samples = [SimpleNamespace(response_token_ids=[], rollout_infos={}, finish_reason='stop')
                           for _ in range(3)]
                with patch.dict(sys.modules, {'swift': swift, 'swift.rlhf_trainers': rlhf,
                                              'swift.rlhf_trainers.grpo_trainer': grpo_module}):
                    trainer = Trainer()
                    audit, restore = install_bounded_thinking_training(
                        trainer, {'arm': 'D', 'bounded_thinking': True, 'thought_limit': 2,
                                  'answer_limit': 2, 'max_completion_length': 6,
                                  'logprob_audit_answers': 3, 'logprob_audit_atol': 1e-6})
                    try:
                        trainer._generate_completions(samples)
                        prepared = trainer._prepare_batch_inputs(samples)[0]
                        batch = prepared['grpo_batch']
                        current, entropies = trainer._get_logps_via_local_forward(
                            model, prepared['model_inputs'], batch.completion_mask.shape[1],
                            prepared['model_inputs']['input_ids'], compute_entropy=True)
                        torch.testing.assert_close(current, batch.old_per_token_logps)
                        torch.testing.assert_close(current, batch.ref_per_token_logps)
                        self.assertTrue(torch.isfinite(entropies).all().item())
                        current[batch.completion_mask].sum().backward()
                        self.assertTrue(torch.isfinite(model.weight.grad).all().item())
                        self.assertGreater(model.weight.grad.abs().max().item(), 0)
                        self.assertEqual(audit['logprob_audit']['answers'], 3)
                        self.assertEqual(audit['logprob_audit']['tokens'], 10)
                        if extra is None:
                            self.assertLess(audit['logprob_audit']['max_abs_gap'], 1e-6)
                        else:
                            self.assertGreater(audit['logprob_audit']['max_abs_gap'], 0.05)
                            self.assertIn('ExtraBias', audit['logprob_audit']['generation_processors'])
                            self.assertGreater(audit['rollout_importance_sampling']['raw_weight_max'], 2.0)
                            self.assertGreater(audit['rollout_importance_sampling']['truncated_fraction'], 0.0)
                        self.assertEqual(audit['sampling_replay']['status'], 'passed')
                        self.assertLess(audit['sampling_replay']['max_abs_gap'], 1e-6)
                        self.assertEqual(audit['rollout_importance_sampling']['action_tokens'], 10)
                        self.assertEqual(batch.rollout_per_token_logps[~batch.completion_mask].tolist(), [0.0] * 8)
                        self.assertEqual(audit['global_action_tokens'], [10.0])
                        self.assertIn('sampled_logprobs', samples[0].rollout_infos['bounded_thinking'])
                        # Behavior probabilities remain complete beyond the old
                        # first-N audit window; every subsequent rollout needs IS.
                        trainer._generate_completions(samples)
                        trainer._prepare_batch_inputs(samples)
                        self.assertEqual(audit['logprob_audit']['answers'], 6)
                        self.assertEqual(audit['rollout_importance_sampling']['action_tokens'], 20)
                    finally:
                        restore()

    def test_training_installer_masks_before_old_logps_and_dapo_count(self):
        import torch

        trace = ThinkingTrace((7, 8, CLOSE, 11, 12, EOS),
                              (True, True, False, True, True, False), (2, 5),
                              2, 2, 'forced', 'answer_budget', (0.0,) * 6)
        holder = {}

        class FakeHook:
            pad_id = PAD
            close_id = CLOSE

            def __init__(self, engine, *, thought_limit, answer_limit, total_limit, **kwargs):
                self.traces = []
                self.thought_limit = thought_limit
                self.answer_limit = answer_limit
                self.replay_audit = {}
                self.amp_audit = {}
                holder['hook'] = self

            def install(self):
                return self

            def pop_traces(self):
                traces, self.traces = self.traces, []
                return traces

            def restore(self):
                pass

        grpo_module = types.ModuleType('swift.rlhf_trainers.grpo_trainer')

        def collate(samples, *args, **kwargs):
            ids = torch.tensor([[PAD, 41, 42, *trace.token_ids]])
            batch = SimpleNamespace(completion_mask=torch.ones((1, len(trace.token_ids)), dtype=torch.bool))
            return {'input_ids': ids}, batch

        grpo_module.collate_to_grpo_micro_batch = collate
        swift = types.ModuleType('swift')
        swift.__path__ = []
        rlhf = types.ModuleType('swift.rlhf_trainers')
        rlhf.__path__ = []
        rlhf.grpo_trainer = grpo_module
        swift.rlhf_trainers = rlhf

        class Trainer:
            use_fast_infer = False
            async_generate = False
            multi_turn_scheduler = None
            engine = object()
            model = SimpleNamespace(training=True)
            accelerator = SimpleNamespace(num_processes=1, autocast=None)
            template = SimpleNamespace(padding_free=False, sequence_parallel_size=1)
            rollout_importance_sampling_mode = 'token_truncate'
            rollout_importance_sampling_threshold = 2.0
            # Swift's HF early-return can leave the vLLM-version gate enabled.
            disable_rollout_importance_sampling = True

            @staticmethod
            def _get_rollout_is_correction(old, behavior, mask):
                return old-behavior, torch.ones_like(old)

            @staticmethod
            def _get_logps_via_local_forward(*args, **kwargs):
                raise AssertionError('unused fake logprob forward')

            @staticmethod
            def _generate_completions(samples):
                holder['hook'].traces.append(trace)
                return samples

            @staticmethod
            def _prepare_batch_inputs(samples):
                inputs, batch = grpo_module.collate_to_grpo_micro_batch(samples)
                # This is the position where Swift computes old/ref logps and global DAPO length.
                assert batch.completion_mask.tolist() == [[True, True, False, True, True, False]]
                batch.num_items_in_batch = batch.completion_mask.sum()
                batch.old_per_token_logps = torch.zeros_like(batch.rollout_per_token_logps)
                return [{'model_inputs': inputs, 'grpo_batch': batch}]

        sample = SimpleNamespace(response_token_ids=[list(trace.token_ids)], response_loss_mask=[],
                                 rollout_infos={}, finish_reason='stop')
        with patch.dict(sys.modules, {'swift': swift, 'swift.rlhf_trainers': rlhf,
                                      'swift.rlhf_trainers.grpo_trainer': grpo_module}), \
             patch('tools.quant_grounding_thinking.BoundedHFGenerationHook', FakeHook):
            trainer = Trainer()
            audit, restore = install_bounded_thinking_training(
                trainer, {'arm': 'D', 'bounded_thinking': True, 'logprob_audit_answers': 0})
            self.assertFalse(trainer.disable_rollout_importance_sampling)
            generated = trainer._generate_completions([sample])
            trainer._prepare_batch_inputs(generated)
            restore()
            self.assertTrue(trainer.disable_rollout_importance_sampling)
        self.assertEqual(sample.finish_reason, 'length')
        self.assertEqual(sample.response_loss_mask, [[1, 1, 0, 1, 1, 0]])
        self.assertEqual(audit['global_action_tokens'], [4.0])
        self.assertEqual(audit['effective_dapo_denominator'], [4.0])
        self.assertEqual(audit['effective_forced_positions'], 2)
        self.assertEqual(audit['newly_masked_positions'], 2)
        self.assertTrue(audit['rollout_importance_sampling']['native_disabled_before_hf_override'])

    def test_shared_prompt_preserves_caller_messages(self):
        original = [{'role': 'user', 'content': 'Find <image>'}]
        prompted = apply_bounded_thinking_prompt(original)
        self.assertEqual(original, [{'role': 'user', 'content': 'Find <image>'}])
        self.assertEqual(prompted[0], {'role': 'system', 'content': BOUNDED_THINKING_SYSTEM})
        self.assertIn('bbox_2d', prompted[0]['content'])
        existing = [{'role': 'system', 'content': 'Visual grounding.'}, *original]
        extended = apply_bounded_thinking_prompt(existing)
        self.assertEqual(len(extended), len(existing))
        self.assertTrue(extended[0]['content'].startswith('Visual grounding.\n'))

    def test_swift_hf_generate_hook_uses_same_state_machine(self):
        class Tokenizer:
            eos_token_id = EOS
            pad_token_id = PAD

            @staticmethod
            def encode(value, add_special_tokens=False):
                assert value == "</think>" and add_special_tokens is False
                return [CLOSE]

        class Model:
            calls = 0

            @staticmethod
            def _get_logits_processor(*, logits_processor):
                return logits_processor

            def generate(self, *, input_ids, generation_config, logits_processor):
                self.calls += 1
                logits_processor = self._get_logits_processor(logits_processor=logits_processor)
                ids = input_ids
                generated = [[], []]
                schedules = [[7, CLOSE, 11, EOS], [7] * 256 + [EOS]]
                for step in range(322):
                    scores = logits_processor[0](ids, np.zeros((2, 64), dtype=np.float32))
                    next_tokens = []
                    for row in range(2):
                        if EOS in generated[row]:
                            token = PAD
                        elif np.isfinite(scores[row]).sum() == 1:
                            token = int(np.argmax(scores[row]))
                        else:
                            token = schedules[row][min(step, len(schedules[row]) - 1)]
                        generated[row].append(token)
                        next_tokens.append(token)
                    ids = np.concatenate((ids, np.array(next_tokens, dtype=np.int64)[:, None]), axis=1)
                    if all(EOS in row for row in generated):
                        break
                return SimpleNamespace(sequences=ids)

        engine = SimpleNamespace(model=Model(), tokenizer=Tokenizer())
        config = SimpleNamespace(num_return_sequences=1, num_beams=1, max_new_tokens=384,
                                 eos_token_id=EOS, pad_token_id=PAD)
        transformers = types.ModuleType("transformers")
        transformers.LogitsProcessorList = list
        with patch.dict(sys.modules, {"transformers": transformers}):
            hook = BoundedHFGenerationHook(engine).install()
            output = engine.model.generate(input_ids=np.array([[PAD, 41, 42], [PAD, 51, 52]]),
                                           generation_config=config)
            natural, forced = hook.pop_traces()
            hook.restore()
        self.assertEqual(engine.model.calls, 1)
        self.assertEqual(output.sequences.shape[1], 3 + 258)
        self.assertEqual(natural.token_ids, (7, CLOSE, 11, EOS))
        self.assertEqual(natural.forced_positions, ())
        self.assertTrue(natural.complete)
        self.assertEqual(forced.forced_positions, (256,))
        self.assertTrue(forced.complete)
        self.assertEqual(hook.pop_traces(), [])

    def test_mixed_batch_natural_and_forced_boundaries(self):
        processor = ThinkingBudgetProcessor(prompt_width=3, batch_size=4,
                                            close_token_id=CLOSE, eos_token_id=EOS, pad_token_id=PAD)
        traces = rollout(processor, [
            [7, CLOSE, 11, EOS],                # natural separator and natural EOS
            [7] * 256 + [8, EOS],              # separator is forced after 256 sampled thoughts
            [7, EOS],                          # EOS in thought: incomplete, then padded
            [CLOSE] + [12] * 64 + [13, EOS],   # force EOS after 64 sampled answer tokens
        ])
        natural, forced_boundary, incomplete, forced_end = traces
        self.assertEqual(natural.token_ids, (7, CLOSE, 11, EOS))
        self.assertEqual(natural.action_mask, (True,) * 4)
        self.assertTrue(natural.complete)
        self.assertEqual(natural.thought_boundary, "natural")

        self.assertEqual(forced_boundary.token_ids[:258], (7,) * 256 + (CLOSE, EOS))
        self.assertEqual(forced_boundary.forced_positions, (256,))
        self.assertEqual(sum(forced_boundary.action_mask), 257)
        self.assertTrue(forced_boundary.complete)
        self.assertEqual(forced_boundary.thought_tokens, 256)
        self.assertEqual(forced_boundary.answer_tokens, 0)

        self.assertEqual(incomplete.token_ids, (7, EOS))
        self.assertEqual(incomplete.finish_reason, "eos_in_thought")
        self.assertFalse(incomplete.complete)

        self.assertEqual(forced_end.forced_positions, (65,))
        self.assertEqual(forced_end.answer_tokens, 64)
        self.assertTrue(forced_end.answer_truncated)
        self.assertFalse(forced_end.complete)
        self.assertEqual(sum(forced_end.action_mask), 65)
        self.assertLessEqual(len(forced_end.token_ids), 384)

    def test_masked_policy_kl_and_dapo_denominator_exclude_only_forced_tokens(self):
        processor = ThinkingBudgetProcessor(prompt_width=3, batch_size=2,
                                            close_token_id=CLOSE, eos_token_id=EOS, pad_token_id=PAD,
                                            thought_limit=2, answer_limit=2)
        natural, forced = rollout(processor, [[7, CLOSE, 11, EOS], [7, 8, 11, 11, 12, 13]])
        self.assertEqual(forced.token_ids, (7, 8, CLOSE, 11, 12, EOS))
        self.assertEqual(forced.forced_positions, (2, 5))
        self.assertEqual(forced.action_mask, (True, True, False, True, True, False))
        tokens = [natural.token_ids, forced.token_ids]
        width = max(map(len, tokens))
        input_ids = np.array([[PAD, 41, 42] + list(row) + [PAD] * (width - len(row)) for row in tokens])
        initial_mask = np.array([[True] * len(row) + [False] * (width - len(row)) for row in tokens])
        # Template loss_scale already masks the delimiter; dynamic EOS can revive suffix labels.
        initial_mask[1, 2] = False
        batch = SimpleNamespace(completion_mask=initial_mask.copy())
        audit = overlay_grpo_action_masks(batch, {"input_ids": input_ids}, [natural, forced], pad_token_id=PAD)
        self.assertEqual(audit, {'forced_positions': 2, 'newly_masked_positions': 1,
                                 'effective_action_tokens': 8})
        self.assertEqual(batch.completion_mask.tolist(),
                         [[True, True, True, True, False, False],
                          [True, True, False, True, True, False]])
        self.assertEqual(int(batch.completion_mask.sum()), 8)  # DAPO denominator
        token_loss = np.array([[1, 2, 3, 4, 0, 0], [1, 2, 999, 3, 4, 999]], dtype=float)
        self.assertEqual(float((token_loss * batch.completion_mask).sum() / batch.completion_mask.sum()), 2.5)
        token_kl = np.array([[0, 0, 0, 0, 0, 0], [0, 0, 999, 0, 0, 999]], dtype=float)
        self.assertEqual(float((token_kl * batch.completion_mask).sum()), 0.0)

    def test_raw_trajectory_and_budget_are_strict(self):
        processor = ThinkingBudgetProcessor(prompt_width=3, batch_size=1,
                                            close_token_id=CLOSE, eos_token_id=EOS, pad_token_id=PAD,
                                            thought_limit=1, answer_limit=1)
        trace = rollout(processor, [[7, 11, 12]])[0]
        self.assertEqual(trace.forced_positions, (1, 3))
        with self.assertRaisesRegex(ValueError, "forced token trajectory"):
            processor.trace(0, [7, 11, 11, EOS])
        with self.assertRaisesRegex(ValueError, "generation limit"):
            ThinkingBudgetProcessor(prompt_width=3, batch_size=1, close_token_id=CLOSE,
                                    eos_token_id=EOS, pad_token_id=PAD, total_limit=320)


if __name__ == "__main__":
    unittest.main()
