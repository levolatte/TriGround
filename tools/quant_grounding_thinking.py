"""Bounded Qwen thinking for HF generation and exact GRPO action masks.

The processor is installed on one ``model.generate`` call (sampling or greedy).
The caller must retain the raw generated token ids: decoded text is insufficient to
distinguish a sampled ``</think>`` from the same delimiter forced by this processor.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from dataclasses import dataclass
from functools import wraps
from typing import Sequence


BOUNDED_THINKING_SYSTEM = (
    'Reason briefly about the target and distinguishing visual evidence. '
    'End the thinking section promptly. In the final answer output exactly one bare '
    '{"bbox_2d":[x1,y1,x2,y2]} object with integer coordinates from 0 to 1000. '
    'Do not use Markdown fences, a list, labels, or multiple candidates.'
)


def apply_bounded_thinking_prompt(messages: list[dict]) -> list[dict]:
    """Return a fresh message list with the same instruction for train/eval."""
    result = deepcopy(messages)
    systems = [message for message in result if message['role'] == 'system']
    if len(systems) > 1:
        raise ValueError('expected at most one system message')
    if systems:
        systems[0]['content'] = systems[0]['content'].rstrip() + '\n' + BOUNDED_THINKING_SYSTEM
    else:
        result.insert(0, {'role': 'system', 'content': BOUNDED_THINKING_SYSTEM})
    return result


@dataclass(frozen=True)
class ThinkingTrace:
    token_ids: tuple[int, ...]
    action_mask: tuple[bool, ...]
    forced_positions: tuple[int, ...]
    thought_tokens: int
    answer_tokens: int
    thought_boundary: str
    finish_reason: str
    sampled_logprobs: tuple[float, ...] = ()
    generation_processors: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return self.thought_boundary in {"natural", "forced"} and self.finish_reason == "natural_eos"

    @property
    def answer_truncated(self) -> bool:
        return self.finish_reason == "answer_budget"


class ThinkingBudgetProcessor:
    """Bound thought/answer lengths and allow exactly one closing delimiter.

    ``prompt_width`` is the padded input width passed to ``generate``. The
    processor supports ``num_beams=1``; rows must keep their order throughout the
    call. Use a fresh instance for every call. Answer tokens are sampled from the
    distribution conditioned on not emitting another closing delimiter. All other
    naturally sampled tokens, including a natural delimiter, remain actions.
    """

    def __init__(self, *, prompt_width: int, batch_size: int, close_token_id: int,
                 eos_token_id: int, pad_token_id: int, thought_limit: int = 256,
                 answer_limit: int = 64, total_limit: int | None = None):
        if total_limit is None:
            total_limit = thought_limit + answer_limit + 2
        if not (0 < thought_limit and 0 < answer_limit and
                thought_limit + 1 + answer_limit + 1 <= total_limit):
            raise ValueError("thinking and answer budgets exceed the total generation limit")
        if close_token_id in {eos_token_id, pad_token_id} or eos_token_id == pad_token_id:
            raise ValueError("close, EOS and pad token ids must differ for an exact generation trace")
        self.prompt_width = prompt_width
        self.batch_size = batch_size
        self.close_token_id = close_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.thought_limit = thought_limit
        self.answer_limit = answer_limit
        self.total_limit = total_limit
        self.max_new_tokens = thought_limit + 1 + answer_limit + 1
        self._forced: list[dict[int, int]] = [{} for _ in range(batch_size)]

    def __call__(self, input_ids, scores):
        if input_ids.shape[0] != self.batch_size or scores.shape[0] != self.batch_size:
            raise ValueError("generation batch size changed; beam search is unsupported")
        step = input_ids.shape[1] - self.prompt_width
        if step < 0 or step >= self.max_new_tokens:
            raise ValueError("generation exceeded the bounded thinking token window")
        for row in range(self.batch_size):
            generated = input_ids[row, self.prompt_width:].tolist()
            # HF keeps finished rows in the batch and pads them on later steps.
            if self.eos_token_id in generated or self.pad_token_id in generated:
                continue
            if self.close_token_id not in generated:
                forced_token = self.close_token_id if len(generated) >= self.thought_limit else None
            else:
                close_at = generated.index(self.close_token_id)
                answer_length = len(generated) - close_at - 1
                forced_token = self.eos_token_id if answer_length >= self.answer_limit else None
                scores[row, self.close_token_id] = float("-inf")
            if forced_token is not None:
                previous = self._forced[row].setdefault(step, forced_token)
                if previous != forced_token:
                    raise RuntimeError("inconsistent forced generation token")
                scores[row] = float("-inf")
                scores[row, forced_token] = 0.0
        return scores

    def trace(self, row: int, generated_token_ids: Sequence[int]) -> ThinkingTrace:
        """Classify the raw suffix returned by generate and align its action mask.

        Padding after EOS is excluded from ``token_ids``. Forced token positions
        must match the emitted raw IDs; a transformed or decoded trajectory fails.
        """
        if not 0 <= row < self.batch_size:
            raise IndexError(row)
        tokens = [int(token) for token in generated_token_ids]
        for position, token in self._forced[row].items():
            if position >= len(tokens) or tokens[position] != token:
                raise ValueError("raw completion differs from the forced token trajectory")
        eos_at = tokens.index(self.eos_token_id) if self.eos_token_id in tokens else None
        if eos_at is not None:
            if any(token != self.pad_token_id for token in tokens[eos_at + 1:]):
                raise ValueError("non-padding token after EOS")
            tokens = tokens[:eos_at + 1]
        elif self.pad_token_id in tokens:
            raise ValueError("padding before EOS")
        if len(tokens) > self.total_limit:
            raise ValueError("completion exceeds total token budget")
        if tokens.count(self.close_token_id) > 1:
            raise ValueError("raw completion contains more than one closing delimiter")
        forced = tuple(sorted(self._forced[row]))
        actions = tuple(position not in self._forced[row] for position in range(len(tokens)))
        if self.close_token_id in tokens:
            close_at = tokens.index(self.close_token_id)
            boundary = "forced" if close_at in self._forced[row] else "natural"
            answer_tokens = len(tokens) - close_at - 1 - int(eos_at is not None)
            if answer_tokens > self.answer_limit:
                raise ValueError("answer exceeds token budget")
        else:
            close_at = len(tokens)
            boundary = "missing"
            answer_tokens = 0
        if eos_at is None:
            finish_reason = "missing_eos"
        elif eos_at in self._forced[row]:
            finish_reason = "answer_budget"
        elif boundary == "missing":
            finish_reason = "eos_in_thought"
        else:
            finish_reason = "natural_eos"
        thought_tokens = close_at if boundary != 'missing' else close_at - int(eos_at is not None)
        return ThinkingTrace(tuple(tokens), actions, forced, thought_tokens, answer_tokens,
                             boundary, finish_reason)


class _ChosenTokenLogprobRecorder:
    """Observe the final HF sampling scores without retaining a vocabulary history.

    The next processor call reveals the previous sampled token. Only that scalar
    logprob is kept; one pending score tensor is released on the following call.
    ``finish`` handles the last token, for which HF makes no further call.
    """

    def __init__(self, prompt_width: int):
        self.prompt_width = prompt_width
        self.pending = None
        self.pending_normalizer = None
        self.values = []

    def _take_previous(self, input_ids):
        if self.pending is None:
            return
        chosen = input_ids[:, -1:]
        self.values.append(self.pending.gather(-1, chosen).squeeze(-1) - self.pending_normalizer)
        self.pending = self.pending_normalizer = None

    def __call__(self, input_ids, scores):
        import torch

        self._take_previous(input_ids)
        if len(self.values) != input_ids.shape[1] - self.prompt_width:
            raise RuntimeError("HF sampler and chosen-token logprob positions differ")
        self.pending = scores.detach()
        self.pending_normalizer = torch.logsumexp(scores.detach().float(), dim=-1)
        return scores

    def finish(self, sequences) -> list[tuple[float, ...]]:
        import torch

        self._take_previous(sequences)
        if len(self.values) != sequences.shape[1] - self.prompt_width:
            raise RuntimeError("HF sampler omitted a generated token from the logprob trace")
        values = torch.stack(self.values, dim=1).float().cpu().tolist()
        self.values.clear()
        return [tuple(row) for row in values]


class _FixedTokenReplay:
    """Read the actual processed probability, then replay the existing action."""

    def __init__(self, traces, *, prompt_width, pad_token_id, token_limit, audit):
        self.traces = traces
        self.prompt_width = prompt_width
        self.pad_token_id = pad_token_id
        self.token_limit = token_limit
        self.audit = audit
        self.gap_sum = 0.0

    def __call__(self, input_ids, scores):
        import torch

        step = input_ids.shape[1] - self.prompt_width
        tokens = [trace.token_ids[step] if step < len(trace.token_ids) else self.pad_token_id
                  for trace in self.traces]
        selected = torch.tensor(tokens, device=scores.device)
        observed = scores.gather(-1, selected[:, None]).squeeze(-1) - torch.logsumexp(scores.float(), dim=-1)
        for row, trace in enumerate(self.traces):
            if step >= min(self.token_limit, len(trace.token_ids)) or not trace.action_mask[step]:
                continue
            actual = float(observed[row].item())
            expected = trace.sampled_logprobs[step]
            gap = abs(actual - expected)
            if not torch.isfinite(observed[row]).item():
                raise RuntimeError('same-context replay produced non-finite sampled-action logprob')
            if self.audit['max_gap_location'] is None or gap > self.audit['max_abs_gap']:
                self.audit['max_gap_location'] = {'row': row, 'response_position': step,
                                                  'token_id': tokens[row], 'sampled_logprob': expected,
                                                  'replayed_logprob': actual}
            self.audit['max_abs_gap'] = max(self.audit['max_abs_gap'], gap)
            self.audit['tokens'] += 1
            self.gap_sum += gap
            self.audit['mean_abs_gap'] = self.gap_sum / self.audit['tokens']
        scores[:] = -float('inf')
        scores[torch.arange(len(tokens), device=scores.device), selected] = 0.0
        return scores


def answer_state_mask(input_ids, *, logits_to_keep: int, close_token_id: int):
    """Whether each predicted suffix token follows the first closing delimiter.

    Swift's local logits are shifted by one: a token at input column j predicts
    column j+1. Count delimiters in that causal history, not in the target token.
    The supported single-turn Qwen prompt has no closing delimiter of its own.
    """
    history = input_ids[:, :-1]
    return (history.eq(close_token_id).cumsum(-1) > 0)[:, -logits_to_keep:]


def suppress_answer_close(logits, answer_states, *, close_token_id: int):
    """Apply the sampler's answer support before any policy log-softmax.

    Updating the one vocabulary column avoids cloning the full logits tensor.
    The masked column has zero gradient in answer states; the normalizer retains
    gradients through all allowed tokens.
    """
    logits[..., close_token_id] = logits[..., close_token_id].masked_fill(answer_states, float('-inf'))
    return logits


def overlay_grpo_action_masks(grpo_batch, model_inputs, traces: Sequence[ThinkingTrace], *, pad_token_id: int) -> dict:
    """Exclude forced tokens from Swift's already-collated HF GRPO token frame.

    Call immediately after ``collate_to_grpo_micro_batch`` and before old/ref
    logprob forwards or the global DAPO token count in ``_prepare_batch_inputs``.
    ``labels`` cannot be changed afterwards (Swift pops them), and Swift may
    re-enable a masked suffix token during template encoding.
    """
    input_ids = model_inputs["input_ids"]
    mask = grpo_batch.completion_mask
    if input_ids.shape[0] != len(traces) or mask.shape[0] != len(traces):
        raise ValueError("GRPO rows do not align with generation traces")
    frame_start = input_ids.shape[1] - mask.shape[1]
    newly_masked = 0
    forced_total = 0
    for row, trace in enumerate(traces):
        tokens = list(trace.token_ids)
        ids = input_ids[row].tolist()
        mask_row = mask[row].tolist()
        forced_cols = []
        valid_end = len(ids)
        while valid_end and ids[valid_end - 1] == pad_token_id:
            valid_end -= 1
        if ids[valid_end - len(tokens):valid_end] != tokens:
            raise ValueError("encoded GRPO response is not the raw sampled trajectory")
        begin = valid_end - len(tokens)
        for position, is_action in enumerate(trace.action_mask):
            col = begin + position - frame_start
            if not 0 <= col < mask.shape[1]:
                raise ValueError("generated token falls outside Swift's logprob frame")
            if is_action and not mask_row[col]:
                raise ValueError("sampled token unexpectedly absent from GRPO completion mask")
            if not is_action:
                newly_masked += bool(mask_row[col])
                forced_total += 1
                forced_cols.append(col)
        mask[row, forced_cols] = False
    return {'forced_positions': forced_total, 'newly_masked_positions': newly_masked,
            'effective_action_tokens': int(mask.sum().item() if hasattr(mask.sum(), 'item') else mask.sum())}


def overlay_grpo_rollout_logps(grpo_batch, model_inputs, traces: Sequence[ThinkingTrace], *, pad_token_id: int):
    """Align behavior logprobs by immutable raw positions after action masking.

    Swift's flat rollout-logprob builder runs before the forced-token mask overlay.
    Scatter again using the raw trajectory so an artificial EOS or injected prompt
    prefix cannot shift sampled probabilities. Forced/padding entries stay zero.
    """
    import torch

    mask = grpo_batch.completion_mask.bool()
    result = torch.zeros(mask.shape, device=mask.device, dtype=torch.float32)
    input_ids = model_inputs['input_ids']
    frame_start = input_ids.shape[1] - mask.shape[1]
    for row, trace in enumerate(traces):
        if len(trace.sampled_logprobs) != len(trace.token_ids):
            raise ValueError('bounded rollout requires a behavior logprob for every raw token')
        ids = input_ids[row].tolist()
        end = len(ids)
        while end and ids[end-1] == pad_token_id:
            end -= 1
        begin = end-len(trace.token_ids)
        if ids[begin:end] != list(trace.token_ids):
            raise ValueError('behavior logprobs differ from the raw sampled trajectory')
        start = begin-frame_start
        values = torch.tensor(trace.sampled_logprobs, device=mask.device, dtype=torch.float32)
        result[row, start:start+len(trace.token_ids)] = values
    result.masked_fill_(~mask, 0.0)
    if not torch.isfinite(result[mask]).all().item():
        raise ValueError('non-finite behavior logprob on a sampled action')
    grpo_batch.rollout_per_token_logps = result
    return result


class BoundedHFGenerationHook:
    """Install the same bounded decode on Swift training or evaluation HF engine.

    Swift 4.5.3 calls ``engine.model.generate`` from ``TransformersEngine`` via
    ``template.generate``. The engine's non-streaming ``infer`` executes calls in
    request order; ``pop_traces`` returns those raw trajectories in that order.
    The caller must align each trace with ``choice.token_ids`` before attaching
    it to a rollout sample or evaluation record. This hook assumes synchronous
    ``infer``, ``n=1`` and no beams.
    """

    def __init__(self, engine, *, thought_limit: int = 256, answer_limit: int = 64,
                 total_limit: int | None = None, capture_logprobs: bool = False,
                 sampling_replay_tokens: int = 0, sampling_replay_atol: float = 0.05,
                 language_autocast_factory=None):
        self.engine = engine
        self.close_id, self.eos_id, self.pad_id = qwen_thinking_token_ids(engine.tokenizer)
        self.thought_limit = thought_limit
        self.answer_limit = answer_limit
        self.total_limit = total_limit
        self.capture_logprobs = capture_logprobs
        self.sampling_replay_tokens = sampling_replay_tokens
        self.sampling_replay_atol = sampling_replay_atol
        self.language_autocast_factory = language_autocast_factory
        self.amp_audit = {'status': 'pending' if language_autocast_factory is not None else 'disabled',
                          'scope': 'language_model_and_lm_head', 'observations': {}}
        self.replay_audit = {'status': 'pending' if sampling_replay_tokens else 'disabled',
                             'token_limit': sampling_replay_tokens, 'atol': sampling_replay_atol,
                             'tokens': 0, 'max_abs_gap': 0.0, 'mean_abs_gap': 0.0,
                             'max_gap_location': None}
        self.traces: list[ThinkingTrace] = []
        self._original_generate = None

    def _run_generate(self, owner, original, kwargs):
        """Match training text AMP while leaving the vision encoder outside it."""
        if self.language_autocast_factory is None:
            return original(**kwargs)
        import torch

        # The actual supported Qwen3.5/3.6 owner shape; do not silently broaden
        # autocast to the outer model or fall back to another module path.
        language = owner.model.language_model
        head = owner.lm_head
        visual = owner.model.visual
        device_type = kwargs['input_ids'].device.type
        observations = self.amp_audit['observations']
        handles = []
        originals = [(language, language.forward), (head, head.forward)]

        def observe(name):
            def hook(_module, _args):
                observations.setdefault(name, torch.is_autocast_enabled(device_type))
            return hook

        if self.amp_audit['status'] == 'pending':
            gdn = next(module for module in language.modules()
                       if type(module).__name__ == 'Qwen3_5GatedDeltaNet')
            handles.append(gdn.register_forward_pre_hook(observe('gdn_internal_autocast')))
            handles.append(visual.register_forward_pre_hook(observe('visual_autocast')))
            fourbit = [module for module in language.modules() if type(module).__name__ == 'Linear4bit']
            if fourbit:
                handles.append(fourbit[0].register_forward_pre_hook(observe('linear4bit_internal_autocast')))
        for name, (module, forward) in zip(['language_body_autocast', 'head_body_autocast'], originals):
            @wraps(forward)
            def wrapped(*args, _forward=forward, _name=name, **kwargs):
                with self.language_autocast_factory():
                    observations.setdefault(_name, torch.is_autocast_enabled(device_type))
                    observations.setdefault(_name + '_dtype', str(torch.get_autocast_dtype(device_type)))
                    return _forward(*args, **kwargs)
            module.forward = wrapped
        try:
            output = original(**kwargs)
            if self.amp_audit['status'] == 'pending':
                required = ['gdn_internal_autocast', 'language_body_autocast', 'head_body_autocast']
                if (any(observations.get(name) is not True for name in required)
                        or observations.get('visual_autocast') is not False
                        or observations.get('linear4bit_internal_autocast', True) is not True
                        or observations.get('language_body_autocast_dtype') != 'torch.bfloat16'
                        or observations.get('head_body_autocast_dtype') != 'torch.bfloat16'):
                    raise RuntimeError('generation text/vision autocast scope differs from the training contract')
                self.amp_audit.update(status='passed', device_type=device_type,
                                      autocast_dtype=observations['language_body_autocast_dtype'])
            return output
        finally:
            for module, forward in originals:
                module.forward = forward
            for handle in handles:
                handle.remove()

    def install(self):
        if self._original_generate is not None:
            raise RuntimeError("bounded HF generation hook is already installed")
        from transformers import LogitsProcessorList

        model = self.engine.model
        original = model.generate
        self._original_generate = original

        def generate(*args, **kwargs):
            if args:
                raise ValueError("Swift HF rollout must call generate with keyword inputs")
            config = kwargs["generation_config"]
            if config.num_return_sequences != 1 or config.num_beams != 1:
                raise ValueError("bounded generation requires one return sequence and no beams")
            eos_ids = config.eos_token_id
            eos_ids = [eos_ids] if isinstance(eos_ids, int) else eos_ids
            if eos_ids is None or self.eos_id not in eos_ids:
                raise ValueError("HF generation EOS differs from the loaded tokenizer")
            if config.pad_token_id != self.pad_id:
                raise ValueError("HF generation pad token differs from the loaded tokenizer")
            input_ids = kwargs["input_ids"]
            # Swift StopWordsCriteria mutates start_idx/is_done during generate.
            # Snapshot it before the original rollout; share its tokenizer so a
            # fresh stopping state does not copy the vocabulary or processor.
            fresh_stopping_criteria = None
            if self.replay_audit['status'] == 'pending' and 'stopping_criteria' in kwargs:
                fresh_stopping_criteria = deepcopy(
                    kwargs['stopping_criteria'], {id(self.engine.tokenizer): self.engine.tokenizer})
            processor = ThinkingBudgetProcessor(
                prompt_width=input_ids.shape[1], batch_size=input_ids.shape[0],
                close_token_id=self.close_id, eos_token_id=self.eos_id, pad_token_id=self.pad_id,
                thought_limit=self.thought_limit, answer_limit=self.answer_limit,
                total_limit=self.total_limit)
            if config.max_new_tokens < processor.max_new_tokens:
                raise ValueError("HF max_new_tokens is smaller than the bounded generation window")
            existing = kwargs.get("logits_processor") or []
            kwargs["logits_processor"] = LogitsProcessorList([*existing, processor])
            # PEFT delegates generate to its underlying Transformers model. Append
            # the observer there, after HF has assembled custom processors, warpers
            # and optional LogitNormalization, rather than assuming they are absent.
            owner = model.get_base_model() if hasattr(model, 'get_base_model') else model
            original_processors = owner._get_logits_processor
            recorder = (_ChosenTokenLogprobRecorder(input_ids.shape[1])
                        if self.capture_logprobs else None)
            processor_names = []

            def final_processors(*args, **kwargs):
                processors = original_processors(*args, **kwargs)
                if processor_names:
                    raise RuntimeError('HF rebuilt logits processors during a single bounded rollout')
                processor_names.extend(type(item).__name__ for item in processors)
                if recorder is not None:
                    processors.append(recorder)
                return processors

            owner._get_logits_processor = final_processors
            try:
                output = self._run_generate(owner, original, kwargs)
            finally:
                owner._get_logits_processor = original_processors
            if not processor_names:
                raise RuntimeError('bounded rollout did not use the HF logits processor path')
            generated = output.sequences[:, input_ids.shape[1]:]
            logprobs = recorder.finish(output.sequences) if recorder is not None else None
            call_traces = []
            for row in range(generated.shape[0]):
                trace = processor.trace(row, generated[row].tolist())
                call_traces.append(ThinkingTrace(
                    **{**asdict(trace),
                       'sampled_logprobs': logprobs[row][:len(trace.token_ids)] if logprobs else (),
                       'generation_processors': tuple(processor_names)}))
            if self.replay_audit['status'] == 'pending':
                import torch

                if logprobs is None:
                    raise RuntimeError('same-context replay requires captured behavior logprobs')
                replay_kwargs = dict(kwargs)
                replay_config = deepcopy(config)
                replay_config.max_new_tokens = min(self.sampling_replay_tokens, generated.shape[1])
                replay_kwargs['generation_config'] = replay_config
                if 'stopping_criteria' in kwargs:
                    replay_kwargs['stopping_criteria'] = fresh_stopping_criteria
                replay_budget = ThinkingBudgetProcessor(
                    prompt_width=input_ids.shape[1], batch_size=input_ids.shape[0],
                    close_token_id=self.close_id, eos_token_id=self.eos_id, pad_token_id=self.pad_id,
                    thought_limit=self.thought_limit, answer_limit=self.answer_limit, total_limit=self.total_limit)
                replay_kwargs['logits_processor'] = LogitsProcessorList([*existing, replay_budget])
                replay_observer = _FixedTokenReplay(
                    call_traces, prompt_width=input_ids.shape[1], pad_token_id=self.pad_id,
                    token_limit=replay_config.max_new_tokens, audit=self.replay_audit)
                replay_names = []
                expected_width = replay_config.max_new_tokens
                expected_actions = sum(sum(trace.action_mask[:expected_width]) for trace in call_traces)
                self.replay_audit.update(expected_width=expected_width, expected_action_tokens=expected_actions)

                def replay_processors(*args, **kwargs):
                    processors = original_processors(*args, **kwargs)
                    replay_names.extend(type(item).__name__ for item in processors)
                    processors.append(replay_observer)
                    return processors

                self.replay_audit['context'] = {
                    'batch_size': input_ids.shape[0], 'prompt_width': input_ids.shape[1],
                    'model_training': model.training, 'cuda_autocast': torch.is_autocast_enabled(),
                    'use_cache': getattr(config, 'use_cache', None),
                    'cache_implementation': getattr(config, 'cache_implementation', None),
                    'generation_kwarg_keys': sorted(kwargs),
                    'processors': processor_names}
                devices = [input_ids.device.index] if input_ids.device.type == 'cuda' else []
                owner._get_logits_processor = replay_processors
                try:
                    # The deterministic replay must not consume the RNG stream of
                    # later on-policy rollouts. It neither scores nor trains a new answer.
                    with torch.random.fork_rng(devices=devices):
                        replay_output = self._run_generate(owner, original, replay_kwargs)
                    replay_ids = replay_output.sequences[:, input_ids.shape[1]:]
                    self.replay_audit['observed_width'] = replay_ids.shape[1]
                    if replay_ids.shape[1] != expected_width:
                        raise RuntimeError(f'same-context replay ended at {replay_ids.shape[1]} tokens; '
                                           f'expected exactly {expected_width}')
                    if self.replay_audit['tokens'] != expected_actions:
                        raise RuntimeError(f'same-context replay checked {self.replay_audit["tokens"]} actions; '
                                           f'expected exactly {expected_actions}')
                    expected_ids = generated[:, :replay_ids.shape[1]]
                    if not torch.equal(replay_ids, expected_ids):
                        raise RuntimeError('same-context replay changed saved raw token ids')
                    if replay_names != processor_names:
                        raise RuntimeError('same-context replay changed the HF score processors')
                    if not self.replay_audit['tokens'] or self.replay_audit['max_abs_gap'] > self.sampling_replay_atol:
                        raise RuntimeError('same-context replay logprob gap exceeds its fixed tolerance')
                    self.replay_audit['status'] = 'passed'
                except Exception:
                    self.replay_audit['status'] = 'failed'
                    raise
                finally:
                    owner._get_logits_processor = original_processors
            self.traces.extend(call_traces)
            return output

        model.generate = generate
        return self

    def pop_traces(self) -> list[ThinkingTrace]:
        traces, self.traces = self.traces, []
        return traces

    def restore(self):
        if self._original_generate is not None:
            self.engine.model.generate = self._original_generate
            self._original_generate = None

    def __enter__(self):
        return self.install()

    def __exit__(self, _exc_type, _exc, _tb):
        self.restore()


def install_bounded_thinking_training(trainer, config: dict):
    """Wire one-shot HF rollout, trace ownership and GRPO masking into Swift 4.5.3.

    Returns ``(audit, restore)``. Install after creating the trainer and before
    training. The hook is deliberately limited to synchronous HF GRPO rollout.
    ``restore`` must run when training exits, including on an exception.
    """
    if not config.get('bounded_thinking') or config.get('arm') != 'D':
        raise ValueError('bounded thinking training requires the repaired D config')
    if trainer.use_fast_infer or trainer.async_generate or trainer.multi_turn_scheduler:
        raise ValueError('bounded thinking supports synchronous single-turn HF rollout only')
    if trainer.template.padding_free or trainer.template.sequence_parallel_size != 1:
        raise ValueError('bounded policy logprobs require padding_free=false and sequence_parallel_size=1')
    correction_actual = {
        'mode': trainer.rollout_importance_sampling_mode,
        'mode_type': type(trainer.rollout_importance_sampling_mode).__name__,
        'threshold': trainer.rollout_importance_sampling_threshold,
        'threshold_type': type(trainer.rollout_importance_sampling_threshold).__name__,
        'native_disabled': trainer.disable_rollout_importance_sampling}
    if (trainer.rollout_importance_sampling_mode != 'token_truncate'
            or trainer.rollout_importance_sampling_threshold != 2.0):
        raise ValueError('bounded HF rollout requires Swift token_truncate importance correction at 2.0; '
                         f'actual={correction_actual!r}')
    import swift.rlhf_trainers.grpo_trainer as grpo_module

    hook = BoundedHFGenerationHook(
        trainer.engine, thought_limit=int(config.get('thought_limit', 256)),
        answer_limit=int(config.get('answer_limit', 64)),
        total_limit=config.get('max_completion_length'), capture_logprobs=True,
        sampling_replay_tokens=int(config.get('sampling_replay_tokens', 64)),
        sampling_replay_atol=float(config.get('sampling_replay_atol', 0.05)),
        language_autocast_factory=trainer.accelerator.autocast).install()
    original_generate = trainer._generate_completions
    original_prepare = trainer._prepare_batch_inputs
    original_collate = grpo_module.collate_to_grpo_micro_batch
    original_local_logps = trainer._get_logps_via_local_forward
    original_disable_correction = trainer.disable_rollout_importance_sampling
    # Swift 4.5.3 _prepare_vllm sets this from the installed vLLM version before
    # its HF early return. Here the synchronous HF hook itself supplies complete
    # behavior probabilities, so this trainer is independent of that vLLM gate.
    trainer.disable_rollout_importance_sampling = False
    audit = {'generated': 0, 'natural_boundaries': 0, 'forced_boundaries': 0,
             'answer_truncated': 0, 'masked_forced_tokens': 0, 'collated_responses': 0,
             'effective_forced_positions': 0, 'newly_masked_positions': 0,
             'local_action_tokens': 0, 'global_action_tokens': [],
             'effective_dapo_denominator': [],
             'thought_limit': hook.thought_limit, 'answer_limit': hook.answer_limit,
             'sampling_replay': hook.replay_audit,
             'generation_amp': hook.amp_audit,
             'logprob_audit': {'comparison': 'local_old_vs_behavior', 'fatal_tolerance': None,
                              'answers': 0, 'tokens': 0, 'max_abs_gap': 0.0,
                              'mean_abs_gap': 0.0, 'max_gap_location': None,
                              'generation_processors': []},
             'rollout_importance_sampling': {
                 'mode': 'token_truncate', 'threshold': 2.0, 'action_tokens': 0,
                 'native_disabled_before_hf_override': original_disable_correction,
                 'native_disabled_after_hf_override': False,
                 'raw_weight_mean': 0.0, 'raw_weight_min': None, 'raw_weight_max': None,
                 'corrected_weight_mean': 0.0, 'truncated_tokens': 0, 'truncated_fraction': 0.0}}
    gap_sum = 0.0
    raw_weight_sum = corrected_weight_sum = 0.0

    def local_logps(model, model_inputs, logits_to_keep, input_ids, compute_entropy=False):
        # This is Swift 4.5.3's sole local padded path, with the answer support
        # applied before selective_log_softmax. Old/current/ref all call it.
        if 'use_cache' in trainer.model_kwarg_keys:
            model_inputs['use_cache'] = False
        if 'logits_to_keep' in trainer.model_kwarg_keys:
            model_inputs['logits_to_keep'] = logits_to_keep + 1
        logits = model(**model_inputs).logits[:, -(logits_to_keep + 1):-1, :]
        # HF samples from float32 scores even when the model emits BF16 logits.
        logits = logits.float() / trainer.temperature
        states = answer_state_mask(input_ids, logits_to_keep=logits_to_keep,
                                   close_token_id=hook.close_id)
        logits = suppress_answer_close(logits, states, close_token_id=hook.close_id)
        labels = input_ids[:, -logits_to_keep:]
        logps = grpo_module.selective_log_softmax(logits, labels)
        entropies = None
        if compute_entropy:
            import torch

            # A suppressed logit is -inf with probability zero; avoid 0 * -inf.
            probabilities = torch.softmax(logits, dim=-1)
            safe_logits = logits.masked_fill(~torch.isfinite(logits), 0.0)
            entropies = torch.logsumexp(logits, dim=-1) - (probabilities * safe_logits).sum(-1)
        return logps, entropies

    def generate(samples):
        if hook.traces:
            raise RuntimeError('unclaimed bounded generation traces before rollout')
        result = original_generate(samples)
        traces = hook.pop_traces()
        if len(result) != len(traces):
            raise RuntimeError(f'HF generation traces {len(traces)} != samples {len(result)}')
        for sample, trace in zip(result, traces):
            response_ids = sample.response_token_ids
            if response_ids and isinstance(response_ids[0], list):
                if len(response_ids) != 1:
                    raise ValueError('bounded thinking expects one assistant response')
                response_ids = response_ids[0]
            if list(trace.token_ids) != list(response_ids):
                raise ValueError('Swift response tokens differ from raw bounded generation')
            sample.response_loss_mask = [[int(action) for action in trace.action_mask]]
            if len(trace.sampled_logprobs) != len(trace.token_ids):
                raise RuntimeError('HF rollout did not capture all behavior logprobs')
            # Swift's flat convention contains sampled actions only. Keep the
            # complete raw alignment separately in the immutable thinking trace.
            sample.rollout_logprobs = [[value for value, action in zip(trace.sampled_logprobs, trace.action_mask)
                                       if action]]
            sample.rollout_infos = {**sample.rollout_infos, 'bounded_thinking': asdict(trace)}
            if trace.answer_truncated or trace.finish_reason == 'missing_eos':
                sample.finish_reason = 'length'
            audit['generated'] += 1
            audit['natural_boundaries'] += trace.thought_boundary == 'natural'
            audit['forced_boundaries'] += trace.thought_boundary == 'forced'
            audit['answer_truncated'] += trace.answer_truncated
            audit['masked_forced_tokens'] += len(trace.forced_positions)
        return result

    def collate_with_actions(samples, *args, **kwargs):
        model_inputs, grpo_batch = original_collate(samples, *args, **kwargs)
        traces = [ThinkingTrace(**sample.rollout_infos['bounded_thinking']) for sample in samples]
        mask_audit = overlay_grpo_action_masks(grpo_batch, model_inputs, traces, pad_token_id=hook.pad_id)
        overlay_grpo_rollout_logps(grpo_batch, model_inputs, traces, pad_token_id=hook.pad_id)
        grpo_batch.bounded_thinking_traces = traces
        audit['effective_forced_positions'] += mask_audit['forced_positions']
        audit['newly_masked_positions'] += mask_audit['newly_masked_positions']
        audit['local_action_tokens'] += mask_audit['effective_action_tokens']
        audit['collated_responses'] += len(samples)
        return model_inputs, grpo_batch

    def prepare(samples):
        nonlocal gap_sum, raw_weight_sum, corrected_weight_sum
        if grpo_module.collate_to_grpo_micro_batch is not original_collate:
            raise RuntimeError('Swift GRPO collate hook changed unexpectedly')
        local_before = audit['local_action_tokens']
        grpo_module.collate_to_grpo_micro_batch = collate_with_actions
        try:
            result = original_prepare(samples)
            import torch

            check = audit['logprob_audit']
            for microbatch, batch in enumerate(result):
                signal = batch['grpo_batch']
                mask = signal.completion_mask.bool()
                if signal.rollout_per_token_logps is None:
                    raise RuntimeError('Swift batch lacks aligned behavior probabilities')
                # Observe Swift's native correction; the trainer applies it once
                # in its loss. Old/current/ref remain the local full-forward policy.
                _, native_weights = trainer._get_rollout_is_correction(
                    signal.old_per_token_logps, signal.rollout_per_token_logps, mask)
                raw_weights = torch.exp((signal.old_per_token_logps[mask].double()
                                         - signal.rollout_per_token_logps[mask].double()))
                if native_weights is None or not torch.isfinite(raw_weights).all().item():
                    raise RuntimeError('non-finite or disabled rollout importance correction')
                effective_weights = native_weights[mask]
                if not torch.isfinite(effective_weights).all().item():
                    raise RuntimeError('non-finite Swift rollout importance weights')
                sampling = audit['rollout_importance_sampling']
                count = int(mask.sum().item())
                sampling['action_tokens'] += count
                raw_weight_sum += float(raw_weights.sum().item())
                corrected_weight_sum += float(effective_weights.double().sum().item())
                sampling['raw_weight_mean'] = raw_weight_sum / sampling['action_tokens']
                sampling['corrected_weight_mean'] = corrected_weight_sum / sampling['action_tokens']
                low, high = float(raw_weights.min().item()), float(raw_weights.max().item())
                sampling['raw_weight_min'] = low if sampling['raw_weight_min'] is None else min(sampling['raw_weight_min'], low)
                sampling['raw_weight_max'] = high if sampling['raw_weight_max'] is None else max(sampling['raw_weight_max'], high)
                sampling['truncated_tokens'] += int((raw_weights > 2.0).sum().item())
                sampling['truncated_fraction'] = sampling['truncated_tokens'] / sampling['action_tokens']
                for row, trace in enumerate(signal.bounded_thinking_traces):
                    if not trace.sampled_logprobs:
                        continue
                    ids = batch['model_inputs']['input_ids'][row].tolist()
                    valid_end = len(ids)
                    while valid_end and ids[valid_end - 1] == hook.pad_id:
                        valid_end -= 1
                    begin = valid_end - len(trace.token_ids)
                    frame_start = len(ids) - signal.completion_mask.shape[1]
                    start = begin - frame_start
                    actions = signal.completion_mask[row, start:start + len(trace.token_ids)].bool()
                    rollout = torch.tensor(trace.sampled_logprobs,
                                           device=signal.old_per_token_logps.device, dtype=torch.float32)
                    old = signal.old_per_token_logps[row, start:start + len(trace.token_ids)].float()
                    gaps = (old - rollout).abs()
                    action_gaps = gaps[actions]
                    if not torch.isfinite(action_gaps).all().item() or not action_gaps.numel():
                        raise RuntimeError('sampled-action rollout/recomputed logprobs are non-finite or empty')
                    largest = float(action_gaps.max().item())
                    if check['max_gap_location'] is None or largest > check['max_abs_gap']:
                        position = int(gaps.masked_fill(~actions, -1).argmax().item())
                        check['max_gap_location'] = {
                            'checked_answer': check['answers'], 'microbatch': microbatch, 'row': row,
                            'response_position': position, 'token_id': int(trace.token_ids[position]),
                            'sampled_logprob': float(rollout[position].item()),
                            'recomputed_logprob': float(old[position].item())}
                    check['max_abs_gap'] = max(check['max_abs_gap'], largest)
                    check['tokens'] += int(action_gaps.numel())
                    gap_sum += float(action_gaps.sum().item())
                    check['mean_abs_gap'] = gap_sum / check['tokens']
                    check['answers'] += 1
                    for name in trace.generation_processors:
                        if name not in check['generation_processors']:
                            check['generation_processors'].append(name)
            denominators = {float(batch['grpo_batch'].num_items_in_batch.item()) for batch in result}
            if len(denominators) != 1:
                raise RuntimeError('DAPO global denominator differs across GRPO microbatches')
            denominator = denominators.pop()
            factor = (int(config.get('train_micro_batch_size', 1))
                      if config.get('runtime_optimization', False) and trainer.model.training else 1)
            global_actions = denominator / factor
            local_actions = audit['local_action_tokens'] - local_before
            if trainer.accelerator.num_processes == 1 and global_actions != local_actions:
                raise RuntimeError('DAPO denominator differs from the effective local action mask')
            audit['global_action_tokens'].append(global_actions)
            audit['effective_dapo_denominator'].append(denominator)
            return result
        finally:
            grpo_module.collate_to_grpo_micro_batch = original_collate

    trainer._generate_completions = generate
    trainer._prepare_batch_inputs = prepare
    trainer._get_logps_via_local_forward = local_logps

    def restore():
        trainer._generate_completions = original_generate
        trainer._prepare_batch_inputs = original_prepare
        trainer._get_logps_via_local_forward = original_local_logps
        trainer.disable_rollout_importance_sampling = original_disable_correction
        hook.restore()

    return audit, restore


def qwen_thinking_token_ids(tokenizer) -> tuple[int, int, int]:
    """Resolve atomic delimiter and distinct EOS/pad IDs from the loaded tokenizer."""
    close = tokenizer.encode("</think>", add_special_tokens=False)
    if len(close) != 1:
        raise ValueError("</think> must be one atomic tokenizer token")
    if not isinstance(tokenizer.eos_token_id, int) or not isinstance(tokenizer.pad_token_id, int):
        raise ValueError("tokenizer requires single EOS and pad token IDs")
    return close[0], tokenizer.eos_token_id, tokenizer.pad_token_id
