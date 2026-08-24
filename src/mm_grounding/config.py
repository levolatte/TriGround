from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ModelConfig:
    backbone: str = "Qwen/Qwen3-VL-2B-Instruct"
    adapter_channels: int = 128
    orthogonal_channels: int = 8
    prompt_gate_init: float = -3.0
    fusion_type: str = "legacy_patch"
    modality_dropout: float = 0.1
    fusion_residual_scale_init: float = 0.001
    fusion_zero_init_prompt_restore: bool = False
    parallel_fusion_stages: int = 1
    parallel_adapter_scale_init: float = 0.01
    query_encoder_layers: int = 1
    query_attention_heads: int = 4
    query_dropout: float = 0.0
    freeze_parallel_adapters: bool = False
    auxiliary_bbox_enabled: bool = False
    auxiliary_bbox_l1_weight: float = 2.0
    auxiliary_bbox_giou_weight: float = 1.0
    vision_lora_enabled: bool = True
    vision_lora_rank: int = 8
    vision_lora_alpha: float = 16.0
    vision_lora_dropout: float = 0.05
    vision_lora_last_n_blocks: int = 4


@dataclass(frozen=True)
class DataConfig:
    stage: str = "joint"
    train_manifest: str = ""
    val_manifest: str = ""
    workers: int = 4
    min_pixels: int = 200704
    max_pixels: int = 802816
    depth_scale: float = 1000.0
    depth_clip: float = 20.0


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 8
    phase_a_epochs: int = 3
    phase_b_epochs: int = 5
    batch_size: int = 1
    grad_accumulation: int = 8
    val_batch_size: int = 1
    eval_subset_size: int = 512
    eval_every_n_epochs: int = 1
    max_new_tokens: int = 40
    early_stopping_patience: int = 2
    early_stopping_min_delta: float = 0.005
    phase_b_warmup_steps: int = 100
    auxiliary_head_only_steps: int = 100
    auxiliary_geometry_warmup_steps: int = 400
    auxiliary_gradient_diagnostics: bool = True
    early_probe_step: int = 0
    early_probe_steps: tuple[int, ...] = ()
    stop_after_last_probe: bool = False
    early_probe_subset_size: int = 128
    early_probe_abort_ratio: float = 0.0
    fusion_lr_a: float = 1e-4
    fusion_lr_b: float = 3e-5
    vision_lora_lr: float = 1e-5
    weight_decay: float = 0.01
    amp: bool = True
    seed: int = 2026
    gradient_checkpointing: bool = True
    initialization_checkpoints: tuple[str, ...] = ()
    init_checkpoint: str | None = None
    resume_epoch: int = 0
    override_resume_learning_rates: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    output_dir: str
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def stage(self) -> str:
        return self.data.stage

    def validate(self) -> None:
        if self.model.fusion_type not in {
            "legacy_patch",
            "safe_post_embed",
            "rdt_deep",
            "parallel_backbone",
        }:
            raise ValueError(
                "fusion_type must be 'legacy_patch', 'safe_post_embed', 'rdt_deep', "
                "or 'parallel_backbone'"
            )
        if not 0 <= self.model.modality_dropout < 1:
            raise ValueError("modality_dropout must be in [0, 1)")
        if self.model.fusion_residual_scale_init < 0:
            raise ValueError("fusion_residual_scale_init must be non-negative")
        if self.model.fusion_type in {"rdt_deep", "parallel_backbone"} and self.model.vision_lora_enabled:
            raise ValueError(f"{self.model.fusion_type} keeps the Qwen vision backbone frozen")
        if self.model.parallel_fusion_stages < 1:
            raise ValueError("parallel_fusion_stages must be positive")
        if self.model.parallel_adapter_scale_init <= 0:
            raise ValueError("parallel_adapter_scale_init must be positive")
        if self.model.query_encoder_layers < 1:
            raise ValueError("query_encoder_layers must be positive")
        if self.model.query_attention_heads < 1:
            raise ValueError("query_attention_heads must be positive")
        if self.model.adapter_channels % self.model.query_attention_heads:
            raise ValueError("adapter_channels must be divisible by query_attention_heads")
        if not 0 <= self.model.query_dropout < 1:
            raise ValueError("query_dropout must be in [0, 1)")
        if self.model.auxiliary_bbox_enabled and self.model.fusion_type != "safe_post_embed":
            raise ValueError("auxiliary bbox supervision requires safe_post_embed fusion")
        if (
            self.model.auxiliary_bbox_l1_weight < 0
            or self.model.auxiliary_bbox_giou_weight < 0
        ):
            raise ValueError("auxiliary bbox loss weights must be non-negative")
        if not 0 < self.data.min_pixels <= self.data.max_pixels:
            raise ValueError("invalid min_pixels/max_pixels")
        if self.data.stage not in {"ir", "depth", "joint"}:
            raise ValueError("data.stage must be 'ir', 'depth', or 'joint'")
        if self.data.stage != "joint" and self.model.fusion_type != "parallel_backbone":
            raise ValueError("single-auxiliary stages require parallel_backbone fusion")
        if self.data.depth_scale <= 0 or self.data.depth_clip <= 0:
            raise ValueError("depth_scale and depth_clip must be positive")
        if self.train.batch_size < 1 or self.train.grad_accumulation < 1 or self.train.val_batch_size < 1:
            raise ValueError("batch sizes must be positive")
        if self.train.eval_subset_size < 1 or self.train.eval_every_n_epochs < 1:
            raise ValueError("evaluation settings must be positive")
        if self.train.max_new_tokens < 16:
            raise ValueError("max_new_tokens is too small for bbox JSON")
        if self.train.early_stopping_patience < 1 or self.train.early_stopping_min_delta < 0:
            raise ValueError("invalid early stopping settings")
        if self.train.phase_b_warmup_steps < 0:
            raise ValueError("phase_b_warmup_steps must be non-negative")
        if (
            self.train.auxiliary_head_only_steps < 0
            or self.train.auxiliary_geometry_warmup_steps < 0
        ):
            raise ValueError("auxiliary bbox warmup steps must be non-negative")
        if self.train.early_probe_step < 0 or self.train.early_probe_subset_size < 1:
            raise ValueError("early probe settings must be positive")
        if self.train.early_probe_step and self.train.early_probe_steps:
            raise ValueError("use either early_probe_step or early_probe_steps, not both")
        probe_steps = tuple(self.train.early_probe_steps)
        if any(step <= 0 for step in probe_steps):
            raise ValueError("early_probe_steps must contain only positive steps")
        if tuple(sorted(set(probe_steps))) != probe_steps:
            raise ValueError("early_probe_steps must be strictly increasing and unique")
        if self.train.stop_after_last_probe and not (
            self.train.early_probe_step or probe_steps
        ):
            raise ValueError("stop_after_last_probe requires an early probe step")
        if not 0 <= self.train.early_probe_abort_ratio <= 1:
            raise ValueError("early_probe_abort_ratio must be in [0, 1]")
        if self.train.phase_a_epochs < 0 or self.train.phase_b_epochs < 0:
            raise ValueError("invalid Phase A/B epochs")
        if self.train.phase_a_epochs == 0 and self.train.phase_b_epochs == 0:
            raise ValueError("at least one training phase is required")
        if self.train.phase_a_epochs + self.train.phase_b_epochs != self.train.epochs:
            raise ValueError("phase_a_epochs + phase_b_epochs must equal epochs")
        if not 0 <= self.train.resume_epoch <= self.train.epochs:
            raise ValueError("resume_epoch must be within the configured training range")
        if self.train.initialization_checkpoints and self.train.init_checkpoint:
            raise ValueError(
                "initialization_checkpoints cannot be combined with init_checkpoint resume"
            )
        if self.train.phase_b_epochs and not self.model.vision_lora_enabled:
            raise ValueError("Phase B requires Vision LoRA")


def _section(cls, raw: dict, name: str):
    values = raw.get(name, {})
    known = set(cls.__dataclass_fields__)
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown {name} fields: {sorted(unknown)}")
    return cls(**values)


def load_config(path: str | Path) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("config must be a mapping")
    allowed = {"output_dir", "model", "data", "train"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown top-level fields: {sorted(unknown)}")
    config = ExperimentConfig(
        output_dir=str(raw.get("output_dir", "runs/multimodal")),
        model=_section(ModelConfig, raw, "model"),
        data=_section(DataConfig, raw, "data"),
        train=_section(TrainConfig, raw, "train"),
    )
    config.validate()
    return config
