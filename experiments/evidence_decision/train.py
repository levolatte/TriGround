"""Independent Qwen3-VL controller LoRA SFT with current-action-only loss.

Use exported ``train.jsonl``. The input is checked at its full processor length
before Trainer sees any row. Prepared labels may continue a copied C adapter.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
import sys
import time

from .model import prepare_image_messages


MAX_LENGTH = 8192


def action_token_indices(tokenizer, action, supervised_ids):
    encoded = tokenizer(action, add_special_tokens=False, return_offsets_mapping=True)
    if supervised_ids[:len(encoded['input_ids'])] != encoded['input_ids']:
        raise ValueError('Action tokens differ from actual supervised prefix')
    match = re.search(r'"action"\s*:\s*"([^"\\]+)"', action)
    if not match:
        raise ValueError('Cannot locate action JSON value')
    start, end = match.span(1)
    indices = [i for i, (a, b) in enumerate(encoded['offset_mapping']) if a < end and b > start]
    if not indices:
        raise ValueError('Action value has no supervised tokens')
    return indices


def read_decisions(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _processor_inputs(processor, messages, *, add_generation_prompt, tools=None):
    prepared, _ = prepare_image_messages(messages)
    return processor.apply_chat_template(
        prepared, tokenize=True, add_generation_prompt=add_generation_prompt,
        return_dict=True, return_tensors="pt", **({"tools": tools} if tools is not None else {}),
    )


def supervised_token_range(prefix_ids, full_ids, end_id, *, max_length=MAX_LENGTH,
                           decode_tail=None):
    """Return label span from a complete role boundary, independent of text."""
    start, stop = len(prefix_ids), len(full_ids)
    if stop > max_length:
        raise ValueError(f"decision length {stop}>{max_length} tokens")
    if stop <= start or list(full_ids[:start]) != list(prefix_ids):
        raise ValueError("assistant prompt is not an exact prefix of the full message")
    ends = [index for index in range(start, stop) if full_ids[index] == end_id]
    if len(ends) != 1:
        raise ValueError("assistant message needs exactly one end token")
    end = ends[-1] + 1
    if end < stop and (decode_tail is None or decode_tail(full_ids[end:]).strip()):
        raise ValueError("non-whitespace content after assistant end token")
    return start, end


def encode_decision(processor, row, *, max_length=MAX_LENGTH, action_balanced=False):
    """Find assistant supervision by a verified message prefix, never token text.

    The prefix contains the entire user and image state plus the assistant
    header. Full input adds exactly one assistant action and its end marker.
    """
    import torch

    action = row["target_action"]
    if not isinstance(action, str) or not action:
        raise ValueError(f"{row.get('id')}: empty target action")
    prefix = _processor_inputs(processor, row["messages"], add_generation_prompt=True, tools=row.get("tools"))
    full = _processor_inputs(
        processor,
        [*row["messages"], {"role": "assistant", "content": [{"type": "text", "text": action}]}],
        add_generation_prompt=False, tools=row.get("tools"),
    )
    prefix_ids = prefix["input_ids"][0]
    full_ids = full["input_ids"][0]
    if ("image_grid_thw" in prefix) != ("image_grid_thw" in full) or (
        "image_grid_thw" in prefix and not torch.equal(prefix["image_grid_thw"], full["image_grid_thw"])
    ):
        raise ValueError(f"{row.get('id')}: prompt and target image grids differ")
    end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    try:
        prefix_length, label_end = supervised_token_range(
            prefix_ids.tolist(), full_ids.tolist(), end_id, max_length=max_length,
            decode_tail=lambda ids: processor.tokenizer.decode(ids, skip_special_tokens=False))
    except ValueError as error:
        raise ValueError(f"{row.get('id')} decision {row.get('decision_index')}: {error}") from error
    labels = torch.full_like(full["input_ids"], -100)
    labels[:, prefix_length:label_end] = full["input_ids"][:, prefix_length:label_end]
    full["labels"] = labels
    if action_balanced:
        offsets = action_token_indices(processor.tokenizer, action,
                                       full_ids[prefix_length:label_end].tolist())
        mask = torch.zeros_like(labels, dtype=torch.bool)
        mask[:, [prefix_length + index for index in offsets]] = True
        full['action_value_mask'] = mask
    full.pop("token_type_ids", None)
    return full


def preflight_all_lengths(processor, rows, *, max_length=MAX_LENGTH, action_balanced=False):
    """Check every example before batching; fail rather than truncate."""
    lengths = []
    for row in rows:
        encoded = encode_decision(processor, row, max_length=max_length, action_balanced=action_balanced)
        lengths.append(int(encoded["input_ids"].shape[-1]))
    return lengths


def _training_dataset(rows, processor, *, max_length=MAX_LENGTH, action_balanced=False):
    from torch.utils.data import Dataset

    class Decisions(Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            return encode_decision(processor, rows[index], max_length=max_length, action_balanced=action_balanced)

    return Decisions()


def _single_item_collator(batch):
    if len(batch) != 1:
        raise ValueError("T training requires micro-batch one")
    return batch[0]


def _place_capacity_probe(model, probe):
    """Put the model and probe on the allocated training device before backward."""
    model.to("cuda")
    return probe.to(model.device)


def action_only_loss(model, inputs):
    """Exact masked causal CE, projecting only positions that predict labels.

    Qwen's default full-sequence vocabulary projection allocates gigabytes for
    ignored image/user positions. The transformer still receives the full input.
    """
    import torch.nn.functional as functional

    values = dict(inputs)
    labels = values.pop('labels')
    action_mask = values.pop('action_value_mask', None)
    positions = (labels[0] != -100).nonzero(as_tuple=True)[0]
    outputs = model(**values, logits_to_keep=positions - 1)
    losses = functional.cross_entropy(outputs.logits[0].float(), labels[0, positions], reduction='none')
    if action_mask is None:
        loss = losses.mean()
    else:
        is_action = action_mask[0, positions]
        loss = 0.5 * losses[is_action].mean() + 0.5 * losses[~is_action].mean()
    return loss, outputs


def run_training(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    import peft
    import transformers
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    tools_dir = Path(__file__).resolve().parents[2] / "tools"
    sys.path.insert(0, str(tools_dir))
    import native_lora_training as native

    if args.epochs > 2 or args.epochs <= 0:
        raise ValueError("epochs must be in (0, 2]")
    if args.max_run_seconds <= 0:
        raise ValueError("max-run-seconds must be positive")
    if args.stop_after_step is not None and (args.stop_after_step < 1 or
                                              (args.max_steps > 0 and args.stop_after_step >= args.max_steps)):
        raise ValueError("stop-after-step must be positive and below max-steps")
    rows = read_decisions(args.train)
    if not rows:
        raise ValueError("empty training decisions")
    if args.save_steps <= 0:
        raise ValueError("save-steps must be positive")
    effective_save_steps = min(args.save_steps, max(1, math.ceil(len(rows) / 8)),
                               args.max_steps if args.max_steps > 0 else args.save_steps)
    preflight_start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model, min_pixels=1024,
                                              max_pixels=602112, local_files_only=True)
    action_balanced = args.loss_mode == 'action-balanced'
    max_length = getattr(args, 'max_length', MAX_LENGTH)
    validated_lengths = getattr(args, 'validated_lengths', None)
    if validated_lengths:
        validated = json.loads(validated_lengths.read_text(encoding='utf-8'))
        if (validated['train'] != str(args.train.resolve()) or validated['max_length'] != max_length
                or validated['row_ids'] != [row['id'] for row in rows]):
            raise ValueError('CPU length report does not match this frozen dataset/context')
        lengths = validated['lengths']
    else:
        lengths = preflight_all_lengths(processor, rows, max_length=max_length, action_balanced=action_balanced)
    preflight_seconds = time.perf_counter() - preflight_start
    args.output_dir.mkdir(parents=True, exist_ok=True)
    length_report = {'train': str(args.train.resolve()), 'max_length': max_length,
                     'row_ids': [row['id'] for row in rows], 'lengths': lengths,
                     'longest_row_id': rows[max(range(len(rows)), key=lengths.__getitem__)]['id']}
    (args.output_dir / 'preflight_lengths.json').write_text(
        json.dumps(length_report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    config = {"base_model": str(args.model), "train": str(args.train),
              "seed": 2026, "epochs": args.epochs, "max_length": max_length,
              "image_budget_source": "per-image max_pixels in prepared training messages",
              "lora": {"scope": "language", "r": 32, "alpha": 64,
                       "dropout": 0.05, "trainable_dtype": "float32"},
              "learning_rate": 1e-5, "micro_batch": 1, "gradient_accumulation": 8,
              "precision": "bf16", "attention": "sdpa", "max_input_tokens_observed": max(lengths),
              "save_steps": effective_save_steps, "max_steps": args.max_steps}
    if getattr(args, "init_adapter", None):
        config['init_adapter'] = str(args.init_adapter)
    config['loss'] = ('half_action_value_ce_half_remaining_supervised_ce' if action_balanced
                      else 'masked_causal_ce_supervised_positions_only_per_decision_mean')
    config_path = args.output_dir / "training_config.json"
    if config_path.exists():
        prior = json.loads(config_path.read_text(encoding="utf-8"))
        if prior != config:
            raise ValueError("existing training_config.json differs from this run")
    else:
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.preflight_only:
        return {**config, "preflight_seconds": preflight_seconds}

    resume_start_step = 0
    if args.resume:
        state_file = args.resume / "trainer_state.json"
        if not state_file.is_file():
            raise FileNotFoundError(f"resume checkpoint has no trainer_state.json: {args.resume}")
        resume_start_step = int(json.loads(state_file.read_text(encoding="utf-8"))["global_step"])
        if args.max_steps > 0 and resume_start_step >= args.max_steps:
            raise ValueError("resume checkpoint already reached max-steps")

    # Trainer seeds during its own initialization, which happens after model
    # and LoRA construction here. Seed both before allocating their weights.
    transformers.enable_full_determinism(2026)
    if not torch.cuda.is_available():
        raise RuntimeError("T training requires the separately allocated GPU")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    gpu_stage_start = time.perf_counter()
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True)
    base.config.use_cache = False
    lora_config = peft.LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05,
        target_modules=list(native.LANGUAGE_TARGETS), bias="none", task_type="CAUSAL_LM")
    model = native.initialize_lora(base, lora_config, scope="language",
                                   init_adapter=str(getattr(args, "init_adapter", None) or ""), peft=peft, get_peft_model=peft.get_peft_model)
    native.audit_lora_model(model, "language")
    model.enable_input_require_grads()
    model_context = base.config.text_config.max_position_embeddings
    if max_length > model_context:
        raise ValueError(f'Requested context {max_length} exceeds model context {model_context}')
    if getattr(args, 'capacity_probe', False) or getattr(args, 'capacity_only', False):
        # Real longest training state, backward included, with no optimizer update.
        model.gradient_checkpointing_enable()
        probe_index = max(range(len(rows)), key=lengths.__getitem__)
        probe = _place_capacity_probe(
            model,
            encode_decision(processor, rows[probe_index], max_length=max_length,
                            action_balanced=action_balanced),
        )
        torch.cuda.reset_peak_memory_stats()
        probe_started = time.perf_counter()
        model.train()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            probe_loss, probe_outputs = action_only_loss(model, probe)
        probe_loss.backward()
        torch.cuda.synchronize()
        capacity = {'status': 'passed', 'id': rows[probe_index]['id'],
                    'input_tokens': lengths[probe_index], 'max_length': max_length,
                    'elapsed_seconds': time.perf_counter() - probe_started,
                    'peak_cuda_allocated_bytes': int(torch.cuda.max_memory_allocated()),
                    'optimizer_updates': 0}
        (args.output_dir / 'capacity_probe.json').write_text(
            json.dumps(capacity, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        model.zero_grad(set_to_none=True)
        del probe, probe_loss, probe_outputs
        torch.cuda.empty_cache()
        if getattr(args, 'capacity_only', False):
            return capacity
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - gpu_stage_start
    training_args = transformers.TrainingArguments(
        output_dir=str(args.output_dir), per_device_train_batch_size=1,
        gradient_accumulation_steps=8, num_train_epochs=args.epochs,
        learning_rate=1e-5, lr_scheduler_type="linear", warmup_steps=0,
        optim="adamw_torch_fused", adam_beta1=0.9, adam_beta2=0.999,
        adam_epsilon=1e-8, weight_decay=0.0, max_grad_norm=1.0,
        bf16=True, tf32=False, gradient_checkpointing=True,
        seed=2026, data_seed=2026, full_determinism=True,
        save_strategy="steps", save_steps=effective_save_steps,
        save_total_limit=2, logging_steps=10, logging_first_step=True,
        logging_nan_inf_filter=False, remove_unused_columns=False,
        dataloader_num_workers=0, report_to="none",
        max_steps=args.max_steps,
    )

    class ControllerTrainer(transformers.Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Per-decision mean loss: Trainer handles accumulation normalization.
            self.model_accepts_loss_kwargs = False

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            loss, outputs = action_only_loss(model, inputs)
            return (loss, outputs) if return_outputs else loss

        def create_optimizer(self):
            if self.optimizer is None:
                groups = native.optimizer_parameter_groups(
                    self.model, scope="language", language_lr=1e-5, visual_lr=0.0)
                optimizer_cls, kwargs = self.get_optimizer_cls_and_kwargs(self.args)
                self.optimizer = optimizer_cls(groups, **kwargs)
                native.assert_fp32_optimizer(self.optimizer, self.model, require_state=False)
            return self.optimizer

    class WallClockBudget(transformers.TrainerCallback):
        def __init__(self, seconds, stop_after_step):
            self.seconds = seconds
            self.stop_after_step = stop_after_step
            self.started = None

        def on_train_begin(self, args, state, control, **kwargs):
            self.started = time.monotonic()

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step - resume_start_step == 20:
                elapsed = time.monotonic() - self.started
                throughput = {"optimizer_steps": 20, "elapsed_seconds": elapsed,
                              "seconds_per_step": elapsed / 20,
                              "estimated_remaining_seconds": (state.max_steps - state.global_step) * elapsed / 20}
                (Path(args.output_dir) / "throughput20.json").write_text(
                    json.dumps(throughput, indent=2) + "\n", encoding="utf-8")
                print(json.dumps({"throughput20": throughput}), flush=True)
            if (time.monotonic() - self.started >= self.seconds or
                    (self.stop_after_step is not None and state.global_step >= self.stop_after_step)):
                control.should_save = True
                control.should_training_stop = True
            return control

    trainer = ControllerTrainer(model=model, args=training_args,
                                train_dataset=_training_dataset(rows, processor, max_length=max_length,
                                                                action_balanced=action_balanced),
                                data_collator=_single_item_collator,
                                callbacks=[WallClockBudget(args.max_run_seconds, args.stop_after_step)])
    training_start = time.perf_counter()
    train_result = trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - training_start
    native.assert_fp32_optimizer(trainer.optimizer, trainer.model, require_state=True)
    save_start = time.perf_counter()
    trainer.save_model(str(args.output_dir / "adapter"))
    processor.save_pretrained(str(args.output_dir / "processor"))
    torch.cuda.synchronize()
    save_seconds = time.perf_counter() - save_start
    gpu_stage_seconds = time.perf_counter() - gpu_stage_start
    final_step = int(trainer.state.global_step)
    target_step = int(trainer.state.max_steps)
    summary = {"resume_from": str(args.resume) if args.resume else None,
               "resume_start_step": resume_start_step, "final_global_step": final_step,
               "target_global_step": target_step,
               "achieved_epoch": float(trainer.state.epoch or 0.0),
               "training_complete": final_step >= target_step,
               "optimizer_steps_this_invocation": final_step - resume_start_step,
               "max_steps": args.max_steps, "epochs": args.epochs,
               "stop_after_step": args.stop_after_step,
               "max_run_seconds": args.max_run_seconds,
               "train_loss": train_result.metrics.get("train_loss"),
               "preflight_seconds_cpu": preflight_seconds,
               "model_load_seconds": model_load_seconds,
               "trainer_seconds": train_seconds, "final_save_seconds": save_seconds,
               "gpu_stage_seconds": gpu_stage_seconds,
               "gpu_stage_hours": gpu_stage_seconds / 3600,
               "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
               "last_checkpoint": str(args.output_dir / f"checkpoint-{final_step}")
                   if (args.output_dir / f"checkpoint-{final_step}").is_dir() else None,
               "adapter": str(args.output_dir / "adapter")}
    summary_path = args.output_dir / "train_summary.json"
    history = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {"invocations": []}
    history["invocations"].append(summary)
    summary_path.write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-adapter", type=Path,
                        help="Continue the copied C or first-stage language adapter with a new optimizer")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument('--max-length', type=int, choices=(6144, 8192), default=MAX_LENGTH,
                        help='Must match the frozen inference context length')
    parser.add_argument('--capacity-probe', action='store_true',
                        help='Check longest real sample backward before training without an optimizer update')
    parser.add_argument('--capacity-only', action='store_true',
                        help='Run the longest-sample capacity check and exit without training')
    parser.add_argument('--validated-lengths', type=Path,
                        help='Reuse CPU preflight_lengths.json for the frozen dataset; rows are still encoded intact')
    parser.add_argument('--loss-mode', choices=('token-mean', 'action-balanced'), default='token-mean')
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-run-seconds", type=float, default=5400,
                        help="Stop and save after an optimizer step once training wall time reaches this limit")
    parser.add_argument("--stop-after-step", type=int,
                        help="For a save/resume smoke check: stop and save at this optimizer step")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_training(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
