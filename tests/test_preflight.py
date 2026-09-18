from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from mm_grounding.config import load_config
from tools import preflight


ROOT = Path(__file__).resolve().parents[1]


class _Dataset:
    def __init__(self, *_args, **_kwargs):
        pass

    def __len__(self):
        return 1

    def __getitem__(self, _index):
        return {}


class _Model:
    def __init__(self):
        visual = SimpleNamespace(
            deepstack_visual_indexes=[8, 16, 24],
            blocks=[None] * 27,
            config=SimpleNamespace(hidden_size=128),
        )
        self.backbone = SimpleNamespace(
            config=SimpleNamespace(use_cache=True),
            model=SimpleNamespace(visual=visual),
            gradient_checkpointing_enable=lambda: None,
            enable_input_require_grads=lambda: None,
        )
        self.fusion = SimpleNamespace(
            fusion_layer_indices=[8, 16, 24, 26],
            prompt_blocks=[None] * 4,
        )

    def to(self, _device):
        return self

    def train(self):
        self.backbone.training = True
        return self

    def set_phase_a_trainable(self, *_args, **_kwargs):
        pass

    def __call__(self, **_inputs):
        return {"loss": torch.tensor(1.0)}


def test_nonoffline_main_uses_dropout_copy_and_reports_both_values(monkeypatch, capsys):
    source_config = load_config(ROOT / "configs/qwen3_vl_8b_stage1a_ir.yaml")
    captured = {}

    def fake_processor(*args, **kwargs):
        captured["processor_args"] = args
        captured["processor_kwargs"] = kwargs
        return object()

    def fake_build_grounder(model_config, _processor):
        captured["model_config"] = model_config
        return _Model()

    monkeypatch.setattr(preflight, "load_config", lambda _path: source_config)
    monkeypatch.setattr(preflight, "GroundingDataset", _Dataset)
    monkeypatch.setattr(preflight, "NativeGroundingCollator", lambda *_args: object())
    monkeypatch.setattr(preflight, "_largest_visual_batch", lambda *_args: ({}, 1))
    monkeypatch.setattr(preflight, "_model_inputs", lambda *_args: {})
    monkeypatch.setattr(preflight.AutoProcessor, "from_pretrained", fake_processor)
    monkeypatch.setattr(preflight, "build_grounder", fake_build_grounder)
    monkeypatch.setattr(
        sys,
        "argv",
        ["preflight.py", "--config", "configs/qwen3_vl_8b_stage1a_ir.yaml", "--device", "cpu"],
    )

    preflight.main()

    report = json.loads(capsys.readouterr().out)
    assert source_config.model.modality_dropout == 0.2
    assert captured["model_config"] is not source_config.model
    assert captured["model_config"].modality_dropout == 0.0
    assert captured["processor_args"] == (source_config.model.backbone,)
    assert captured["processor_kwargs"]["revision"] == source_config.model.backbone_revision
    assert captured["processor_kwargs"]["local_files_only"] is True
    assert report["training_modality_dropout"] == 0.2
    assert report["preflight_modality_dropout"] == 0.0
    assert report["real_model"]["backbone_training"] is True
