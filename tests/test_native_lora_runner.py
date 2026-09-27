"""CPU-only contract checks for the bounded native Qwen3-VL launcher."""

import os
from pathlib import Path
import ast
import json
import shutil
import subprocess
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_qwen3vl_native_lora.sh"
BASH = "C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash")
pytestmark = pytest.mark.skipif(not BASH or not Path(BASH).is_file(), reason="Bash unavailable")


def test_native_launcher_keeps_bounded_training_contract():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'ANNOTATION_PATH="${ANNOTATION_PATH:-${DEFAULT_ANNOTATION_PATH}}"' in text
    assert 'SEED="${SEED:-2026}"' in text
    assert 'EPOCHS="${EPOCHS:-2}"' in text
    assert 'LEARNING_RATE="${LEARNING_RATE:-1e-5}"' in text
    assert 'RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"' in text
    assert 'INIT_ADAPTER="${INIT_ADAPTER:-}"' in text
    assert 'SAVE_STEPS="${SAVE_STEPS:-}"' in text
    assert 'STOP_AFTER_STEP="${STOP_AFTER_STEP:-}"' in text
    assert "autocast_adapter_dtype=False" in text
    assert '"trainer_state.json", "optimizer.pt", "scheduler.pt"' in text
    assert "--optim adamw_torch_fused" in text
    assert "--adam_beta1 0.9" in text
    assert "--adam_beta2 0.999" in text
    assert "--adam_epsilon 1e-8" in text
    assert "--weight_decay 0.0" in text
    assert "--lr_scheduler_type linear" in text
    assert "--warmup_steps 0" in text
    assert "--max_grad_norm 1.0" in text
    assert "--tf32 False" in text
    assert "--logging_nan_inf_filter False" in text
    assert 'torch.set_float32_matmul_precision("highest")' in text
    assert "torch.backends.cuda.matmul.allow_tf32 = False" in text
    assert "torch.backends.cudnn.allow_tf32 = False" in text
    assert 'train_qwen_module.train(attn_implementation="sdpa")' in text
    assert "processor.tokenizer.model_max_length = model_max_length" in text
    assert "return self._get_item(sources)" in text
    assert "native_train_config.json" in text
    assert "on_pre_optimizer_step" in text
    assert "on_save" in text


def _bash_path(path: Path) -> str:
    value = path.as_posix()
    if os.name == "nt":
        return f"/{value[0].lower()}{value[2:]}"
    return value


def test_native_launcher_passes_annotation_and_max_steps_overrides(tmp_path):
    qwen_root = tmp_path / "qwen root"
    (qwen_root / "qwenvl" / "train").mkdir(parents=True)
    (qwen_root / "qwenvl" / "train" / "train_qwen.py").write_text("# fake\n", encoding="utf-8")
    data_root = tmp_path / "data root"
    annotation = data_root / "custom_16.json"
    data_root.mkdir()
    annotation.write_text(json.dumps([{"id": index} for index in range(16)]) + "\n", encoding="utf-8")
    model_path = tmp_path / "model"
    model_path.mkdir()
    output_dir = tmp_path / "output"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture = tmp_path / "capture.txt"
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'args=%s\\n' \"$*\" > \"$CAPTURE\"\n"
        "printf 'annotation=%s\\nmax_steps=%s\\nseed=%s\\nepochs=%s\\nlr=%s\\ninit=%s\\nresume=%s\\nsave_steps=%s\\n' "
        "\"$ANNOTATION_PATH\" \"$MAX_STEPS\" \"$SEED\" \"$EPOCHS\" \"$LEARNING_RATE\" \"$INIT_ADAPTER\" \"$RESUME_FROM_CHECKPOINT\" \"$SAVE_STEPS\" >> \"$CAPTURE\"\n",
        encoding="utf-8",
        newline="\n",
    )
    fake_python.chmod(0o755)

    env = dict(
        os.environ,
        PATH=str(fake_bin) + os.pathsep + os.environ.get("PATH", ""),
        QWEN_FINETUNE_DIR=_bash_path(qwen_root),
        DATA_ROOT=_bash_path(data_root),
        MODEL_PATH=_bash_path(model_path),
        ANNOTATION_PATH=_bash_path(annotation),
        OUTPUT_DIR=_bash_path(output_dir),
        MAX_STEPS="2",
        SEED="17",
        EPOCHS="3",
        LEARNING_RATE="2e-5",
        INIT_ADAPTER="",
        RESUME_FROM_CHECKPOINT="",
        SAVE_STEPS="500",
        STOP_AFTER_STEP="",
        CAPTURE=_bash_path(capture),
    )
    result = subprocess.run(
        [BASH, _bash_path(SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    captured = capture.read_text(encoding="utf-8")
    assert "--max_steps 2" in captured
    assert "--save_strategy steps --save_steps 500" in captured
    assert f"annotation={_bash_path(annotation)}" in captured
    assert "max_steps=2" in captured
    assert "seed=17" in captured
    assert "epochs=3" in captured
    assert "lr=2e-5" in captured


def test_native_launcher_rejects_initial_adapter_and_resume_together(tmp_path):
    qwen_root = tmp_path / "qwen"
    (qwen_root / "qwenvl" / "train").mkdir(parents=True)
    (qwen_root / "qwenvl" / "train" / "train_qwen.py").write_text("# fake\n")
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "target_v2" / "qwen3vl_native_sft").mkdir(parents=True)
    (data_root / "target_v2" / "qwen3vl_native_sft" / "trimodal_train.json").write_text("[]\n")
    env = dict(
        os.environ,
        QWEN_FINETUNE_DIR=_bash_path(qwen_root),
        DATA_ROOT=_bash_path(data_root),
        MODEL_PATH="unused",
        INIT_ADAPTER="m2/checkpoint-928",
        RESUME_FROM_CHECKPOINT="stage/checkpoint-500",
    )
    result = subprocess.run([BASH, _bash_path(SCRIPT)], env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert "mutually exclusive" in result.stderr


def _inline_function(name: str, scope: dict):
    source = SCRIPT.read_text(encoding="utf-8").split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    tree = ast.parse(source)
    definition = next(
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name
    )
    module = ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[]))
    exec(compile(module, str(SCRIPT), "exec"), scope)
    return scope[name]


def test_initial_adapter_is_loaded_trainable_without_creating_a_new_lora(tmp_path):
    called = []

    class FakeTensor:
        dtype = "bf16"

        def __init__(self, value):
            self.value = value

        def to(self, *, dtype):
            assert dtype == self.dtype
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

    config = SimpleNamespace(
        r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules={"q_proj", "k_proj", "v_proj", "o_proj"},
    )
    params = [
        (f"base_model.model.language_model.layers.0.self_attn.{target}.lora_A.default.weight",
         SimpleNamespace(requires_grad=True))
        for target in config.target_modules
    ]
    model = SimpleNamespace(
        named_parameters=lambda: params,
        print_trainable_parameters=lambda: None,
    )

    def load_adapter(base, path, **kwargs):
        called.append((base, path, kwargs))
        return model

    peft = SimpleNamespace(
        PeftConfig=SimpleNamespace(from_pretrained=lambda path, **kw: config),
        PeftModel=SimpleNamespace(from_pretrained=load_adapter),
        load_peft_weights=lambda path, **kw: {"lora_A.weight": FakeTensor(7)},
        get_peft_model_state_dict=lambda loaded_model: {"lora_A.weight": FakeTensor(7)},
    )
    scope = dict(
        init_adapter=str(tmp_path), Path=Path, peft=peft,
        torch=SimpleNamespace(equal=lambda a, b: a.value == b.value),
        original_get_peft_model=lambda *args, **kwargs: pytest.fail("new LoRA must not be created"),
        targets=("q_proj", "k_proj", "v_proj", "o_proj"),
    )
    hook = _inline_function("audited_get_peft_model", scope)
    base = object()
    assert hook(base, config) is model
    assert called == [(
        base, str(tmp_path),
        dict(is_trainable=True, autocast_adapter_dtype=False, local_files_only=True),
    )]


def test_resume_of_initialized_stage_recreates_lora_without_dtype_upcast():
    calls = []
    params = [
        (f"base_model.model.language_model.layers.0.self_attn.{target}.lora_A.default.weight",
         SimpleNamespace(requires_grad=True))
        for target in ("q_proj", "k_proj", "v_proj", "o_proj")
    ]
    model = SimpleNamespace(
        named_parameters=lambda: params,
        print_trainable_parameters=lambda: None,
    )

    def make_lora(*args, **kwargs):
        calls.append((args, kwargs))
        return model

    scope = dict(
        init_adapter="",
        resume_from_checkpoint="stage/checkpoint-500",
        stage_init_adapter="m2/checkpoint-928",
        original_get_peft_model=make_lora,
        targets=("q_proj", "k_proj", "v_proj", "o_proj"),
    )
    hook = _inline_function("audited_get_peft_model", scope)
    base, config = object(), object()
    assert hook(base, config) is model
    assert calls == [((base, config), {"autocast_adapter_dtype": False})]


def test_resume_requires_optimizer_scheduler_and_rng_state(tmp_path):
    checkpoint = tmp_path / "checkpoint-500"
    checkpoint.mkdir()
    for name in ("trainer_state.json", "scheduler.pt", "rng_state.pth"):
        (checkpoint / name).write_text("state")
    called = []
    scope = dict(
        resume_from_checkpoint=str(checkpoint), Path=Path,
        original_trainer_train=lambda self, *args, **kwargs: called.append((args, kwargs)),
    )
    hook = _inline_function("resume_aware_train", scope)
    with pytest.raises(RuntimeError, match="optimizer.pt"):
        hook(object(), resume_from_checkpoint=True)
    (checkpoint / "optimizer.pt").write_text("state")
    hook(object(), resume_from_checkpoint=True)
    assert called == [((), {"resume_from_checkpoint": str(checkpoint)})]


def test_planned_pause_occurs_after_step_checkpoint_is_saved(tmp_path):
    scope = dict(
        transformers=SimpleNamespace(TrainerCallback=object),
        targets=("q_proj", "k_proj", "v_proj", "o_proj"),
        shutil=SimpleNamespace(disk_usage=lambda path: SimpleNamespace(free=5 * 1024**3)),
        output_dir=tmp_path,
        stop_after_step="500",
    )
    callback_class = _inline_function("FiniteTrainingCallback", scope)
    callback = callback_class(SimpleNamespace(named_parameters=lambda: []))
    control = SimpleNamespace(should_training_stop=False)
    callback.on_save(None, SimpleNamespace(global_step=499), control)
    assert not control.should_training_stop
    callback.on_save(None, SimpleNamespace(global_step=500), control)
    assert control.should_training_stop
