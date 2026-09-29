#!/usr/bin/env bash
set -euo pipefail

: "${QWEN_FINETUNE_DIR:?Set QWEN_FINETUNE_DIR to Qwen3-VL/qwen-vl-finetune}"
: "${DATA_ROOT:?Set DATA_ROOT to the City train directory}"
: "${MODEL_PATH:?Set MODEL_PATH to the local Qwen3-VL-8B-Instruct directory}"

DATASET_VARIANT="${DATASET_VARIANT:-trimodal}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/outputs/qwen3vl_native_${DATASET_VARIANT}_lora}"
MAX_PIXELS="${MAX_PIXELS:-602112}"
MIN_PIXELS="${MIN_PIXELS:-200704}"
MAX_STEPS="${MAX_STEPS:--1}"
SEED="${SEED:-2026}"
EPOCHS="${EPOCHS:-2}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
VISUAL_LORA_LR="${VISUAL_LORA_LR:-2e-5}"
LORA_SCOPE="${LORA_SCOPE:-language}"
INIT_ONLY="${INIT_ONLY:-0}"
INIT_ADAPTER="${INIT_ADAPTER:-}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
SAVE_STEPS="${SAVE_STEPS:-}"
STOP_AFTER_STEP="${STOP_AFTER_STEP:-}"
PRESERVE_MANIFEST_ORDER="${PRESERVE_MANIFEST_ORDER:-0}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
AUDIT_SAMPLE_COUNT="${AUDIT_SAMPLE_COUNT:-1}"
CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${LORA_SCOPE}" != "language" && "${LORA_SCOPE}" != "language_merger" ]]; then
  echo "LORA_SCOPE must be language or language_merger" >&2
  exit 2
fi
if [[ "${INIT_ONLY}" != "0" && "${INIT_ONLY}" != "1" ]]; then
  echo "INIT_ONLY must be 0 or 1" >&2
  exit 2
fi
if [[ "${INIT_ONLY}" == "1" && -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  echo "INIT_ONLY cannot resume an optimizer checkpoint" >&2
  exit 2
fi
if [[ "${PRESERVE_MANIFEST_ORDER}" != "0" && "${PRESERVE_MANIFEST_ORDER}" != "1" ]]; then
  echo "PRESERVE_MANIFEST_ORDER must be 0 or 1" >&2
  exit 2
fi
if ! [[ "${SAVE_TOTAL_LIMIT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SAVE_TOTAL_LIMIT must be a positive integer" >&2
  exit 2
fi
if ! [[ "${AUDIT_SAMPLE_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "AUDIT_SAMPLE_COUNT must be a positive integer" >&2
  exit 2
fi

if [[ -n "${INIT_ADAPTER}" && -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  echo "INIT_ADAPTER and RESUME_FROM_CHECKPOINT are mutually exclusive" >&2
  exit 2
fi
if [[ -n "${SAVE_STEPS}" ]] && ! [[ "${SAVE_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SAVE_STEPS must be a positive integer" >&2
  exit 2
fi
if [[ -n "${STOP_AFTER_STEP}" ]]; then
  if ! [[ "${STOP_AFTER_STEP}" =~ ^[1-9][0-9]*$ ]] || [[ -z "${SAVE_STEPS}" ]] || (( STOP_AFTER_STEP % SAVE_STEPS != 0 )); then
    echo "STOP_AFTER_STEP must be a positive multiple of SAVE_STEPS" >&2
    exit 2
  fi
  if [[ "${MAX_STEPS}" == "-1" ]] || (( STOP_AFTER_STEP >= MAX_STEPS )); then
    echo "STOP_AFTER_STEP requires MAX_STEPS above the pause step" >&2
    exit 2
  fi
fi

case "${DATASET_VARIANT}" in
  rgb)
    DEFAULT_ANNOTATION_PATH="${DATA_ROOT}/target_v2/qwen3vl_native_sft/rgb_train.json"
    ;;
  trimodal)
    DEFAULT_ANNOTATION_PATH="${DATA_ROOT}/target_v2/qwen3vl_native_sft/trimodal_train.json"
    ;;
  *)
    echo "DATASET_VARIANT must be rgb or trimodal" >&2
    exit 2
    ;;
esac
ANNOTATION_PATH="${ANNOTATION_PATH:-${DEFAULT_ANNOTATION_PATH}}"

test -f "${ANNOTATION_PATH}"
test -f "${QWEN_FINETUNE_DIR}/qwenvl/train/train_qwen.py"
test -f "${CODE_ROOT}/tools/native_lora_training.py"

export CITY_DATA_ROOT="${DATA_ROOT}"
export CITY_ANNOTATION_PATH="${ANNOTATION_PATH}"
export CITY_DATASET_NAME="city_native_${DATASET_VARIANT}_train"
export CITY_MIN_PIXELS="${MIN_PIXELS}"
export CITY_MAX_PIXELS="${MAX_PIXELS}"
export CITY_MODEL_MAX_LENGTH="4096"
export CITY_MODEL_PATH="${MODEL_PATH}"
export CITY_SEED="${SEED}"
export CITY_EPOCHS="${EPOCHS}"
export CITY_LEARNING_RATE="${LEARNING_RATE}"
export CITY_VISUAL_LORA_LR="${VISUAL_LORA_LR}"
export CITY_LORA_SCOPE="${LORA_SCOPE}"
export CITY_INIT_ONLY="${INIT_ONLY}"
export CITY_CODE_ROOT="${CODE_ROOT}"
export CITY_MAX_STEPS="${MAX_STEPS}"
export CITY_OUTPUT_DIR="${OUTPUT_DIR}"
export CITY_INIT_ADAPTER="${INIT_ADAPTER}"
export CITY_RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT}"
export CITY_SAVE_STEPS="${SAVE_STEPS}"
export CITY_STOP_AFTER_STEP="${STOP_AFTER_STEP}"
export CITY_PRESERVE_MANIFEST_ORDER="${PRESERVE_MANIFEST_ORDER}"
export CITY_SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT}"
export CITY_AUDIT_SAMPLE_COUNT="${AUDIT_SAMPLE_COUNT}"
cd "${QWEN_FINETUNE_DIR}"

args=(
  --model_name_or_path "${MODEL_PATH}"
  --dataset_use "${CITY_DATASET_NAME}"
  --output_dir "${OUTPUT_DIR}"
  --data_flatten False
  --data_packing False
  --tune_mm_vision False
  --tune_mm_mlp False
  --tune_mm_llm False
  --lora_enable True
  --lora_r 32
  --lora_alpha 64
  --lora_dropout 0.05
  --learning_rate "${LEARNING_RATE}"
  --optim adamw_torch_fused
  --adam_beta1 0.9
  --adam_beta2 0.999
  --adam_epsilon 1e-8
  --weight_decay 0.0
  --lr_scheduler_type linear
  --warmup_steps 0
  --warmup_ratio 0.0
  --max_grad_norm 1.0
  --per_device_train_batch_size 1
  --gradient_accumulation_steps 8
  --num_train_epochs "${EPOCHS}"
  --seed "${SEED}"
  --data_seed "${SEED}"
  --full_determinism True
  --bf16
  --tf32 False
  --gradient_checkpointing True
  --max_pixels "${MAX_PIXELS}"
  --min_pixels "${MIN_PIXELS}"
  --model_max_length 4096
  --eval_strategy no
  --save_total_limit "${SAVE_TOTAL_LIMIT}"
  --logging_steps 10
  --logging_first_step True
  --logging_nan_inf_filter False
  --skip_memory_metrics False
  --dataloader_num_workers "$([[ "$PRESERVE_MANIFEST_ORDER" == 1 ]] && echo 0 || echo 2)"
  --remove_unused_columns False
  --report_to none
  --run_name "qwen3vl_native_${DATASET_VARIANT}_lora_r32"
)

if [[ -n "${SAVE_STEPS}" ]]; then
  args+=(--save_strategy steps --save_steps "${SAVE_STEPS}")
else
  args+=(--save_strategy epoch)
fi

if [[ "${MAX_STEPS}" != "-1" ]]; then
  args+=(--max_steps "${MAX_STEPS}")
fi

# The official entry point hard-codes flash_attention_2 in __main__ and exposes no
# attention CLI flag. Importing the same train() function lets this single-process
# launch select SDPA explicitly. The PEFT and Trainer hooks add the audited
# composite adapter and its two explicit optimizer groups when requested.
"${PYTHON_EXECUTABLE:-python}" - "${args[@]}" <<'PY'
import math
import os
import shutil
import sys
import types
import json
from pathlib import Path

import torch

qwen_root = Path.cwd()
sys.path.insert(0, str(Path(os.environ["CITY_CODE_ROOT"]) / "tools"))
import native_lora_training as native_lora
sys.path.insert(0, str(qwen_root))
sys.path.insert(0, str(qwen_root / "qwenvl" / "train"))

# train_qwen.py imports the sibling trainer.py unconditionally. That module in
# turn imports flash_attn unconditionally, even when SDPA and ordinary collation
# are requested. Supply only the symbol train_qwen.py needs. It must never be
# called because this launcher fixes both flattening and packing to False.
trainer_shim = types.ModuleType("trainer")


def reject_flash_attention_patch():
    raise RuntimeError("The FlashAttention packing patch is disabled for the SDPA launcher")


trainer_shim.replace_qwen2_vl_attention_class = reject_flash_attention_patch
sys.modules["trainer"] = trainer_shim

from qwenvl.data import data_dict

dataset_name = os.environ["CITY_DATASET_NAME"]
data_dict[dataset_name] = {
    "annotation_path": os.environ["CITY_ANNOTATION_PATH"],
    "data_path": os.environ["CITY_DATA_ROOT"],
}

import peft
import transformers

# Keep the numerical policy explicit and independent of the host defaults. These
# assignments do not initialize CUDA; they are also harmless in the CPU test
# environment used for this launcher.
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

output_dir = Path(os.environ["CITY_OUTPUT_DIR"])
output_dir.mkdir(parents=True, exist_ok=True)
init_adapter = os.environ["CITY_INIT_ADAPTER"]
resume_from_checkpoint = os.environ["CITY_RESUME_FROM_CHECKPOINT"]
save_steps = os.environ["CITY_SAVE_STEPS"]
stop_after_step = os.environ["CITY_STOP_AFTER_STEP"]
lora_scope = os.environ["CITY_LORA_SCOPE"]
visual_lora_lr = float(os.environ["CITY_VISUAL_LORA_LR"])
init_only = os.environ["CITY_INIT_ONLY"] == "1"
if visual_lora_lr <= 0:
    raise ValueError("VISUAL_LORA_LR must be positive")
preserve_manifest_order = os.environ["CITY_PRESERVE_MANIFEST_ORDER"] == "1"
save_total_limit = int(os.environ["CITY_SAVE_TOTAL_LIMIT"])
config_path = output_dir / "native_train_config.json"
resume_config_path = (
    Path(resume_from_checkpoint).expanduser().parent / "native_train_config.json"
    if resume_from_checkpoint else None
)
prior_config_path = config_path if config_path.is_file() else resume_config_path
previous_config = (
    json.loads(prior_config_path.read_text(encoding="utf-8"))
    if prior_config_path and prior_config_path.is_file() else {}
)
stage_init_adapter = (
    previous_config.get("init_adapter") if resume_from_checkpoint else init_adapter
)
if init_adapter and list(output_dir.glob("checkpoint-*")):
    raise RuntimeError(
        "INIT_ADAPTER starts a fresh training stage; output_dir already contains "
        f"checkpoints: {output_dir}. Use a new output_dir or RESUME_FROM_CHECKPOINT."
    )
resolved_config = {
    "launcher": "run_qwen3vl_native_lora.sh",
    "attention_implementation": "sdpa",
    "dataset_name": os.environ["CITY_DATASET_NAME"],
    "annotation_path": os.environ["CITY_ANNOTATION_PATH"],
    "data_root": os.environ["CITY_DATA_ROOT"],
    "model_path": os.environ.get("CITY_MODEL_PATH", ""),
    "output_dir": str(output_dir),
    "min_pixels": int(os.environ["CITY_MIN_PIXELS"]),
    "max_pixels": int(os.environ["CITY_MAX_PIXELS"]),
    "model_max_length": int(os.environ["CITY_MODEL_MAX_LENGTH"]),
    "seed": os.environ.get("CITY_SEED", ""),
    "epochs": os.environ.get("CITY_EPOCHS", ""),
    "learning_rate": os.environ.get("CITY_LEARNING_RATE", ""),
    "lora": {
        "scope": lora_scope,
        "target_modules": sorted(native_lora.expected_modules(lora_scope)),
        "dropout": 0.05,
        "language": {"rank": 32, "alpha": 64, "dtype": "float32"},
        "visual": {"rank": 8, "alpha": 16, "modules": 8, "dtype": "float32"}
        if lora_scope == "language_merger" else None,
    },
    "max_steps": os.environ.get("CITY_MAX_STEPS", "-1"),
    "init_adapter": stage_init_adapter or None,
    "save_strategy": "steps" if save_steps else "epoch",
    "save_steps": int(save_steps) if save_steps else None,
    "stop_after_step": int(stop_after_step) if stop_after_step else None,
    "preserve_manifest_order": preserve_manifest_order,
    "dataloader_num_workers": 0 if preserve_manifest_order else 2,
    "save_total_limit": save_total_limit,
    "audit_sample_count": int(os.environ["CITY_AUDIT_SAMPLE_COUNT"]),
    "optimizer": {
        "name": "adamw_torch_fused",
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-8,
        "weight_decay": 0.0,
        "language_lr": float(os.environ["CITY_LEARNING_RATE"]),
        "visual_lr": visual_lora_lr if lora_scope == "language_merger" else None,
        "lr_scheduler_type": "linear",
        "warmup_steps": 0,
        "max_grad_norm": 1.0,
    },
    "numerics": {
        "frozen_base_dtype": "bfloat16",
        "trainable_gradient_dtype": "float32",
        "adam_moments_dtype": "float32",
        "tf32": False,
        "float32_matmul_precision": "highest",
        "cudnn_allow_tf32": False,
        "logging_nan_inf_filter": False,
        "full_determinism": True,
    },
    "resume": {
        "requested": resume_from_checkpoint or None,
        "policy": (
            "explicit_path"
            if resume_from_checkpoint
            else "fresh_stage_from_adapter"
            if init_adapter
            else "official_auto_latest_checkpoint"
        ),
    },
    "fail_fast_dataset_getitem": True,
}
comparison_keys = (
    "attention_implementation",
    "dataset_name",
    "annotation_path",
    "data_root",
    "model_path",
    "min_pixels",
    "max_pixels",
    "model_max_length",
    "seed",
    "epochs",
    "learning_rate",
    "lora",
    "max_steps",
    "preserve_manifest_order",
    "dataloader_num_workers",
    "save_total_limit",
    "init_adapter",
    "optimizer",
    "numerics",
)


def check_existing_config(path):
    if not path.is_file():
        return
    previous = json.loads(path.read_text(encoding="utf-8"))
    mismatches = {
        key: (previous.get(key), resolved_config[key])
        for key in comparison_keys
        if previous.get(key, {"preserve_manifest_order": False, "dataloader_num_workers": 2, "save_total_limit": 2}.get(key)) != resolved_config[key]
    }
    if mismatches:
        raise RuntimeError(f"training config mismatch for resume/output directory {path}: {mismatches}")
    print(f"=== preserved compatible training config: {path} ===")


check_existing_config(config_path)
if resume_from_checkpoint:
    check_existing_config(Path(resume_from_checkpoint).expanduser() / "native_train_config.json")
    if resume_config_path != config_path:
        check_existing_config(resume_config_path)
if not config_path.is_file():
    config_path.write_text(
        json.dumps(resolved_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

original_get_peft_model = peft.get_peft_model
targets = ("q_proj", "k_proj", "v_proj", "o_proj")
trace_path = output_dir / "native_lora_audit.jsonl"


def record_lora_audit(event, **details):
    row = {"event": event, **details}
    with trace_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print("=== LoRA audit " + json.dumps(row, ensure_ascii=False, sort_keys=True) + " ===")


def check_adapter_checkpoint(model, checkpoint):
    saved = peft.load_peft_weights(str(checkpoint), device="cpu")
    loaded = peft.get_peft_model_state_dict(model)
    if set(saved) != set(loaded):
        raise RuntimeError(f"adapter checkpoint keys differ: {checkpoint}")
    changed = [key for key, value in saved.items()
               if value.dtype != torch.float32
               or not torch.equal(value, loaded[key].detach().cpu())]
    if changed:
        raise RuntimeError(f"adapter checkpoint values/dtype differ: {checkpoint}: {changed[:5]}")
    return len(saved)


class FiniteTrainingCallback(transformers.TrainerCallback):
    """Fail on an observed non-finite loss or gradient without re-walking the model."""

    def __init__(self, model):
        # Cache only trainable parameters once. In particular, do not require a
        # non-zero gradient for every LoRA tensor: lora_A is zero-gradient on
        # the first update when lora_B starts at zero.
        self._trainable_parameters = tuple(
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
        self._audited_targets = {
            target: tuple(
                (name, parameter)
                for name, parameter in self._trainable_parameters
                if f".{target}.lora_" in name
            )
            for target in targets
        }
        self._gradient_checks = 0
        self._first_step_probes = None

    def _probe(self, group):
        candidates = [
            (name, parameter) for name, parameter in self._trainable_parameters
            if (".language_model." if group == "language" else ".visual.") in name
            and ".lora_B." in name and parameter.grad is not None
            and bool(torch.count_nonzero(parameter.grad).item())
        ]
        if not candidates:
            raise RuntimeError(f"first optimizer step has no nonzero {group} LoRA B gradient")
        b_name, b_parameter = candidates[0]
        a_name = b_name.replace(".lora_B.", ".lora_A.")
        parameters = dict(self._trainable_parameters)
        a_parameter = parameters[a_name]
        return a_name, a_parameter, b_name, b_parameter, a_parameter.detach().float().cpu().clone(), b_parameter.detach().float().cpu().clone()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs and logs["loss"] is not None:
            loss = float(logs["loss"])
            if not math.isfinite(loss):
                raise FloatingPointError(f"non-finite logged loss at step {state.global_step}: {loss}")
        return control

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        for name, parameter in self._trainable_parameters:
            gradient = parameter.grad
            if gradient is not None and gradient.dtype != torch.float32:
                raise RuntimeError(f"LoRA gradient must be float32: {name}: {gradient.dtype}")
            if gradient is not None and not bool(torch.isfinite(gradient).all().item()):
                raise FloatingPointError(
                    f"non-finite gradient in trainable parameter {name} at step {state.global_step}"
                )
        if self._gradient_checks == 0:
            model = kwargs["model"]
            optimizer = kwargs["optimizer"]
            expected_count = 304 if lora_scope == "language_merger" else 288
            module_audit = native_lora.audit_lora_model(model, lora_scope)
            optimizer_audit = native_lora.assert_fp32_optimizer(
                optimizer, model, require_state=bool(resume_from_checkpoint)
            )
            if module_audit["tensors"] != expected_count:
                raise RuntimeError(f"LoRA count differs at first optimizer step: {module_audit}")
            if resume_from_checkpoint:
                checkpoint = Path(resume_from_checkpoint).expanduser()
                checkpoint_count = check_adapter_checkpoint(model, checkpoint)
                native_lora.compare_optimizer_checkpoint(optimizer, checkpoint)
                record_lora_audit("resume_restored", step=state.global_step,
                                  adapter_tensors=checkpoint_count,
                                  optimizer=optimizer_audit)
            missing_targets = [
                target
                for target, parameters in self._audited_targets.items()
                if not any(parameter.grad is not None for _, parameter in parameters)
            ]
            if missing_targets:
                raise RuntimeError(
                    "first optimizer step has no gradient in LoRA target groups: "
                    f"{missing_targets}; zero-valued lora_A gradients are allowed"
                )
            non_none = sum(
                parameter.grad is not None
                for _, parameter in self._trainable_parameters
            )
            disconnected = [name for name, parameter in self._trainable_parameters
                            if parameter.grad is None]
            if disconnected:
                raise RuntimeError(f"trainable LoRA tensors disconnected from loss: {disconnected[:8]}")
            print(
                "=== first optimizer gradient audit: "
                f"{non_none}/{len(self._trainable_parameters)} trainable tensors have gradients; "
                "zero-valued lora_A gradients are allowed ==="
            )
            self._first_step_probes = {group: self._probe(group) for group in
                                       (["language", "visual"] if lora_scope == "language_merger"
                                        else ["language"])}
            record_lora_audit("pre_first_step", step=state.global_step,
                              modules=module_audit, optimizer=optimizer_audit,
                              gradient_tensors=non_none,
                              probes={group: probe[2] for group, probe in self._first_step_probes.items()})
        self._gradient_checks += 1
        return control

    def on_optimizer_step(self, args, state, control, **kwargs):
        if self._first_step_probes is None:
            return control
        updates = {}
        for group, (a_name, a, b_name, b, old_a, old_b) in self._first_step_probes.items():
            new_a = a.detach().float().cpu()
            new_b = b.detach().float().cpu()
            b_change = float(torch.linalg.vector_norm(new_b - old_b))
            ba_change = native_lora.low_rank_delta_norm(old_a, old_b, new_a, new_b)
            if b_change <= 0 or ba_change <= 0:
                raise RuntimeError(f"first {group} LoRA B/BA did not update: B={b_change}, BA={ba_change}")
            updates[group] = {"A": a_name, "B": b_name,
                              "B_update_norm": b_change, "BA_update_norm": ba_change}
        optimizer_audit = native_lora.assert_fp32_optimizer(
            kwargs["optimizer"], kwargs["model"], require_state=True
        )
        record_lora_audit("post_first_step", step=state.global_step + 1,
                          updates=updates, optimizer=optimizer_audit)
        self._first_step_probes = None
        return control

    def on_save(self, args, state, control, **kwargs):
        checkpoint = output_dir / f"checkpoint-{state.global_step}"
        count = check_adapter_checkpoint(kwargs["model"], checkpoint)
        optimizer_audit = native_lora.assert_fp32_optimizer(
            kwargs["optimizer"], kwargs["model"], require_state=True
        )
        record_lora_audit("checkpoint_saved", step=state.global_step,
                          adapter_tensors=count, optimizer=optimizer_audit)
        free_bytes = shutil.disk_usage(output_dir).free
        free_gib = free_bytes / (1024**3)
        print(f"=== checkpoint disk audit: free_gib={free_gib:.3f} output_dir={output_dir} ===")
        if free_bytes < 3 * 1024**3:
            raise RuntimeError(f"free disk space below 3 GiB after checkpoint save: {free_gib:.3f} GiB")
        if stop_after_step and state.global_step == int(stop_after_step):
            print(f"=== planned pause after checkpoint at step {state.global_step} ===")
            control.should_training_stop = True
        return control


def audited_get_peft_model(*args, **kwargs):
    if kwargs:
        raise RuntimeError(f"unexpected get_peft_model keyword arguments: {sorted(kwargs)}")
    torch.manual_seed(int(os.environ["CITY_SEED"]))
    model = native_lora.initialize_lora(
        args[0], args[1], scope=lora_scope,
        init_adapter=init_adapter if not resume_from_checkpoint else "",
        peft=peft, get_peft_model=original_get_peft_model,
    )
    module_audit = native_lora.audit_lora_model(model, lora_scope)
    record_lora_audit("adapter_initialized", step=0, modules=module_audit,
                      source=init_adapter or None, scope=lora_scope)
    model.print_trainable_parameters()
    if init_only:
        if (output_dir / "adapter_config.json").exists():
            raise RuntimeError(f"INIT_ONLY output already contains an adapter: {output_dir}")
        model.save_pretrained(str(output_dir), safe_serialization=True)
        count = check_adapter_checkpoint(model, output_dir)
        record_lora_audit("init_only_saved", step=0, adapter_tensors=count)
        raise SystemExit(0)
    return model


peft.get_peft_model = audited_get_peft_model

original_trainer_init = transformers.Trainer.__init__


def audited_trainer_init(self, *args, **kwargs):
    original_trainer_init(self, *args, **kwargs)
    self.add_callback(FiniteTrainingCallback(self.model))
    print(
        "=== finite-value audit enabled: "
        f"{sum(parameter.requires_grad for parameter in self.model.parameters())} trainable tensors; "
        "zero LoRA A gradients are allowed ==="
    )


transformers.Trainer.__init__ = audited_trainer_init

def audited_create_optimizer(self):
    if self.optimizer is None:
        groups = native_lora.optimizer_parameter_groups(
            self.model, scope=lora_scope,
            language_lr=float(os.environ["CITY_LEARNING_RATE"]),
            visual_lr=visual_lora_lr,
        )
        optimizer_cls, optimizer_kwargs = transformers.Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(groups, **optimizer_kwargs)
        audit = native_lora.assert_fp32_optimizer(self.optimizer, self.model, require_state=False)
        record_lora_audit("optimizer_created", step=0, scope=lora_scope,
                          groups=[{"name": group["group_name"], "tensors": len(group["params"]),
                                   "lr": group["lr"]} for group in groups], audit=audit)
    return self.optimizer


transformers.Trainer.create_optimizer = audited_create_optimizer

if preserve_manifest_order:
    from torch.utils.data import SequentialSampler

    def sequential_train_sampler(self, *args, **kwargs):
        return SequentialSampler(self.train_dataset)

    # Trainer's ordinary checkpoint resume skips already consumed batches from
    # this same sampler. The trace is written after training_step, never in
    # __getitem__, because DataLoaderShard can fetch ahead even with 0 workers.
    transformers.Trainer._get_train_sampler = sequential_train_sampler
    original_training_step = transformers.Trainer.training_step

    def traced_training_step(self, model, inputs, *args, **kwargs):
        ids = inputs.pop("_sample_trace_ids")
        loss = original_training_step(self, model, inputs, *args, **kwargs)
        with (output_dir / "consumed_samples.jsonl").open("a", encoding="utf-8") as stream:
            for sample_id in ids:
                stream.write(json.dumps({"id": sample_id}, ensure_ascii=False) + "\n")
        return loss

    transformers.Trainer.training_step = traced_training_step
    print("=== manifest order: SequentialSampler, dataloader_num_workers=0 ===")

original_trainer_train = transformers.Trainer.train


def resume_aware_train(self, *args, **kwargs):
    if resume_from_checkpoint:
        checkpoint = Path(resume_from_checkpoint).expanduser()
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"RESUME_FROM_CHECKPOINT is not a directory: {checkpoint}")
        required_state = ("trainer_state.json", "optimizer.pt", "scheduler.pt")
        missing_state = [name for name in required_state if not (checkpoint / name).is_file()]
        if not list(checkpoint.glob("rng_state*.pth")):
            missing_state.append("rng_state*.pth")
        if missing_state:
            raise RuntimeError(
                f"RESUME_FROM_CHECKPOINT lacks full training state at {checkpoint}: {missing_state}"
            )
        if args and args[0] is True:
            args = (str(checkpoint), *args[1:])
        elif not args and kwargs.get("resume_from_checkpoint") in (None, True):
            kwargs["resume_from_checkpoint"] = str(checkpoint)
        print(f"=== resume policy: explicit checkpoint {checkpoint} ===")
    else:
        print(
            "=== resume policy: fresh adapter stage ==="
            if init_adapter
            else "=== resume policy: official output_dir checkpoint-* auto-resume behavior ==="
        )
    return original_trainer_train(self, *args, **kwargs)


transformers.Trainer.train = resume_aware_train

from qwenvl.train import train_qwen as train_qwen_module

original_auto_processor = transformers.AutoProcessor
min_pixels = int(os.environ["CITY_MIN_PIXELS"])
max_pixels = int(os.environ["CITY_MAX_PIXELS"])
model_max_length = int(os.environ["CITY_MODEL_MAX_LENGTH"])


class AutoProcessorWithPixelBounds:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        kwargs["min_pixels"] = min_pixels
        kwargs["max_pixels"] = max_pixels
        processor = original_auto_processor.from_pretrained(*args, **kwargs)
        processor.tokenizer.model_max_length = model_max_length
        return processor


train_qwen_module.AutoProcessor = AutoProcessorWithPixelBounds
original_make_data_module = train_qwen_module.make_supervised_data_module


def audited_make_data_module(processor, data_args):
    data_module = original_make_data_module(processor, data_args)
    dataset = data_module["train_dataset"]

    # The upstream LazySupervisedDataset retries a broken sample and then
    # silently substitutes a neighboring sample. Training data errors must be
    # visible for this bounded run, so call its original single-sample method
    # directly. The launcher fixes data_packing=False, hence _get_item is the
    # correct official code path here.
    dataset_class = type(dataset)

    def fail_fast_getitem(self, index):
        sources = self.list_data_dict[index]
        if isinstance(sources, dict):
            sources = [sources]
        return self._get_item(sources)

    dataset_class.__getitem__ = fail_fast_getitem
    print("=== dataset audit: LazySupervisedDataset.__getitem__ fail-fast _get_item ===")

    patch_size = int(processor.image_processor.patch_size)
    merge_size = int(processor.image_processor.merge_size)
    for index in range(int(os.environ["CITY_AUDIT_SAMPLE_COUNT"])):
        first_sample = dataset[index]
        first_batch = data_module["data_collator"]([first_sample])
        input_tokens = int(first_batch["input_ids"].shape[-1])
        labels = first_batch["labels"][0]
        supervised_ids = labels[labels.ne(-100)]
        supervised_text = processor.tokenizer.decode(supervised_ids.tolist())
        expected_answer = dataset.list_data_dict[index]["conversations"][-1]["value"]
        grid = first_batch["image_grid_thw"].tolist()
        resized_pixels = [int(t * h * w * patch_size * patch_size) for t, h, w in grid]
        merged_visual_tokens = [int(t * h * w // (merge_size * merge_size)) for t, h, w in grid]
        image_field = dataset.list_data_dict[index]["image"]
        expected_images = len(image_field) if isinstance(image_field, list) else 1
        audit = {
            "id": dataset.list_data_dict[index]["id"],
            "input_tokens": input_tokens,
            "model_max_length": processor.tokenizer.model_max_length,
            "grid": grid,
            "resized_pixels": resized_pixels,
            "merged_visual_tokens": merged_visual_tokens,
            "pixel_values_shape": list(first_batch["pixel_values"].shape),
            "supervised_token_count": int(supervised_ids.numel()),
            "supervised_text": supervised_text,
            "image_processor_size": str(processor.image_processor.size),
        }
        print(f"=== official training sample {index} audit ===")
        print(json.dumps(audit, ensure_ascii=False))
        if input_tokens > model_max_length:
            raise RuntimeError(f"sample {index} has {input_tokens} tokens, above {model_max_length}")
        if len(grid) != expected_images:
            raise RuntimeError(f"sample {index} expected {expected_images} image grids, got {len(grid)}")
        if any(pixels > max_pixels for pixels in resized_pixels):
            raise RuntimeError(f"sample {index} image pixel bound not applied: {resized_pixels} > {max_pixels}")
        if supervised_ids.numel() == 0 or expected_answer not in supervised_text:
            raise RuntimeError(
                f"sample {index} assistant bbox supervision is absent or truncated: "
                f"expected={expected_answer!r}, decoded={supervised_text!r}"
            )
    if preserve_manifest_order:
        sample_ids = [str(row["id"]) for row in dataset.list_data_dict]
        if len(sample_ids) != len(set(sample_ids)):
            raise RuntimeError("ordered training manifest contains duplicate sample IDs")
        trace_path = output_dir / "consumed_samples.jsonl"
        if resume_from_checkpoint:
            checkpoint_step = json.loads(
                (Path(resume_from_checkpoint) / "trainer_state.json").read_text(encoding="utf-8")
            )["global_step"]
            expected_count = checkpoint_step * 8
            recorded = trace_path.read_text(encoding="utf-8").splitlines()
            if len(recorded) < expected_count:
                raise RuntimeError(
                    f"sample trace has {len(recorded)} rows, expected at least {expected_count} "
                    f"at checkpoint step {checkpoint_step}"
                )
            if len(recorded) > expected_count:
                abandoned = output_dir / f"uncommitted_after_checkpoint_{checkpoint_step}.jsonl"
                if abandoned.exists():
                    raise RuntimeError(f"uncommitted trace backup already exists: {abandoned}")
                abandoned.write_text("\n".join(recorded[expected_count:]) + "\n", encoding="utf-8")
                trace_path.write_text("\n".join(recorded[:expected_count]) + "\n", encoding="utf-8")
                print(f"=== backed up and discarded {len(recorded) - expected_count} uncommitted sample IDs: {abandoned} ===")
        elif trace_path.exists():
            raise RuntimeError(f"fresh ordered stage already has sample trace: {trace_path}")
        original_getitem = dataset_class.__getitem__

        def traced_getitem(self, index):
            sample = original_getitem(self, index)
            sample["_sample_trace_id"] = sample_ids[index]
            return sample

        dataset_class.__getitem__ = traced_getitem
        original_collator = data_module["data_collator"]

        def traced_collator(features):
            ids = [feature.pop("_sample_trace_id") for feature in features]
            batch = original_collator(features)
            batch["_sample_trace_ids"] = ids
            return batch

        data_module["data_collator"] = traced_collator
        print(f"=== completed training-step sample trace: {trace_path} ===")
    return data_module


train_qwen_module.make_supervised_data_module = audited_make_data_module

train_qwen_module.train(attn_implementation="sdpa")
PY
