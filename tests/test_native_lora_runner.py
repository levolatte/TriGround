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
    assert 'LORA_SCOPE="${LORA_SCOPE:-language}"' in text
    assert 'VISUAL_LORA_LR="${VISUAL_LORA_LR:-2e-5}"' in text
    assert 'INIT_ONLY="${INIT_ONLY:-0}"' in text
    assert 'import native_lora_training as native_lora' in text
    assert 'native_lora.initialize_lora(' in text
    assert 'native_lora.optimizer_parameter_groups(' in text
    assert 'native_lora.compare_optimizer_checkpoint(' in text
    assert 'model.save_pretrained(str(output_dir), safe_serialization=True)' in text
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
    assert "--full_determinism True" in text
    assert '"full_determinism": True' in text
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


def test_invalid_lora_scope_is_rejected_before_python(tmp_path):
    env = dict(os.environ, LORA_SCOPE="everything", QWEN_FINETUNE_DIR="unused",
               DATA_ROOT="unused", MODEL_PATH="unused")
    result = subprocess.run([BASH, _bash_path(SCRIPT)], env=env,
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert "LORA_SCOPE must be language or language_merger" in result.stderr
