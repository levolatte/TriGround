"""Read-only fixed-sample diagnostics for the 8B parallel fusion modules.

Hooks observe the tensors already produced by the model.  They return ``None``
and therefore do not replace any forward value or alter model parameters.  The
reported residual ratio is measured from the stage output after it has been
written back to the RGB token dtype (BF16 for the formal 8B run), while the
BF16 statistic compares the restore projection before and after that cast.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoProcessor

from mm_grounding.checkpoint import load_model_checkpoint
from mm_grounding.config import load_config
from mm_grounding.data import GroundingDataset, NativeGroundingCollator
from mm_grounding.engine import _training_inputs
from mm_grounding.model import build_grounder


def _resolve_path(value: str, config_path: Path, root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = (root / path, config_path.parent / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


class _Summary:
    """Streaming scalar summary with a bounded value sample for quantiles."""

    def __init__(self, limit: int = 20000) -> None:
        self.limit = limit
        self.count = 0
        self.sum = 0.0
        self.sum_sq = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf
        self.values: list[float] = []

    def add(self, value: torch.Tensor | float) -> None:
        values = torch.as_tensor(value).detach().float().reshape(-1).cpu()
        values = values[torch.isfinite(values)]
        if not values.numel():
            return
        self.count += int(values.numel())
        self.sum += float(values.sum())
        self.sum_sq += float(values.square().sum())
        self.minimum = min(self.minimum, float(values.min()))
        self.maximum = max(self.maximum, float(values.max()))
        remaining = self.limit - len(self.values)
        if remaining > 0:
            self.values.extend(values[:remaining].tolist())

    def as_dict(self) -> dict[str, float | int | None]:
        if not self.count:
            return {"count": 0, "mean": None, "std": None, "min": None, "p05": None,
                    "p50": None, "p95": None, "max": None}
        sample = torch.tensor(self.values, dtype=torch.float32)
        mean = self.sum / self.count
        variance = max(0.0, self.sum_sq / self.count - mean * mean)
        quantiles = torch.quantile(sample, torch.tensor((0.05, 0.5, 0.95)))
        return {
            "count": self.count,
            "mean": mean,
            "std": math.sqrt(variance),
            "min": self.minimum,
            "p05": float(quantiles[0]),
            "p50": float(quantiles[1]),
            "p95": float(quantiles[2]),
            "max": self.maximum,
        }


class _StageStats:
    def __init__(self) -> None:
        self.calls = 0
        self.delta_rms = _Summary()
        self.rgb_rms = _Summary()
        self.delta_to_rgb_rms = _Summary()
        self.token_gate = _Summary()
        self.sample_gate = _Summary()
        self.residual_precast_rms = _Summary()
        self.bf16_changed_fraction = _Summary()

    def as_dict(self) -> dict:
        return {
            "calls": self.calls,
            "delta_rms": self.delta_rms.as_dict(),
            "rgb_rms": self.rgb_rms.as_dict(),
            "delta_to_rgb_rms": self.delta_to_rgb_rms.as_dict(),
            "token_gate": self.token_gate.as_dict(),
            "sample_gate": self.sample_gate.as_dict(),
            "residual_precast_rms": self.residual_precast_rms.as_dict(),
            "bf16_changed_fraction": self.bf16_changed_fraction.as_dict(),
        }


def _stage_name(module_name: str, module) -> str:
    return module_name or module.__class__.__name__


class FusionProbe:
    """Install no-op-observing hooks and summarize all active fusion stages."""

    def __init__(self, model) -> None:
        self.model = model
        self.stats: dict[str, _StageStats] = {}
        self.handles = []
        fusion = model.fusion
        modules = []
        for prefix, collection in (
            ("stage", getattr(fusion, "stage_fusions", {})),
            ("joint", getattr(fusion, "joint_stage_fusions", {})),
        ):
            for key, module in collection.items():
                modules.append((f"{prefix}.{key}", module))
        if not modules:
            raise RuntimeError("checkpoint model has no stage or joint fusion modules")
        for name, module in modules:
            stage = self.stats.setdefault(name, _StageStats())
            self.handles.append(module.register_forward_hook(self._stage_hook(name, stage)))
            if hasattr(module, "token_reliability"):
                self.handles.append(
                    module.token_reliability.register_forward_hook(
                        self._gate_hook(stage.token_gate)
                    )
                )
            if hasattr(module, "sample_reliability"):
                self.handles.append(
                    module.sample_reliability.register_forward_hook(
                        self._gate_hook(stage.sample_gate)
                    )
                )
            if hasattr(module, "restore"):
                self.handles.append(
                    module.restore.register_forward_hook(
                        self._restore_hook(stage, model)
                    )
                )

    @staticmethod
    def _gate_hook(summary: _Summary):
        def hook(_module, _inputs, output):
            summary.add(torch.sigmoid(output))
            return None

        return hook

    @staticmethod
    def _stage_hook(name: str, stats: _StageStats):
        def hook(_module, inputs, output):
            # The first input is the RGB token stream.  ``output`` is the value
            # returned by the stage, including its actual dtype write-back.
            rgb = inputs[0]
            if not torch.is_tensor(output) or not torch.is_tensor(rgb):
                return None
            delta = output.float() - rgb.float()
            rgb_rms = float(rgb.float().square().mean().sqrt())
            delta_rms = float(delta.square().mean().sqrt())
            stats.calls += 1
            stats.rgb_rms.add(rgb_rms)
            stats.delta_rms.add(delta_rms)
            stats.delta_to_rgb_rms.add(delta_rms / max(rgb_rms, 1e-12))
            stats.bf16_changed_fraction.add(float((output != rgb).float().mean()))
            return None

        return hook

    @staticmethod
    def _restore_hook(stats: _StageStats, model):
        def hook(_module, _inputs, output):
            if not torch.is_tensor(output):
                return None
            work = output.float()
            stats.residual_precast_rms.add(float(work.square().mean().sqrt()))
            return None

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def report(self) -> dict[str, dict]:
        return {name: stats.as_dict() for name, stats in sorted(self.stats.items())}


def run_diagnostics(
    config_path: str | Path,
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
    maxsamples: int = 8,
    device_name: str = "cuda",
) -> dict:
    root = Path(__file__).resolve().parents[1]
    config_path = Path(config_path)
    config_path = config_path if config_path.is_absolute() else root / config_path
    config = load_config(config_path)
    manifest = _resolve_path(str(manifest_path), config_path, root)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA diagnostics requested but CUDA is unavailable")
    processor = AutoProcessor.from_pretrained(
        config.model.backbone,
        revision=config.model.backbone_revision,
        min_pixels=config.data.min_pixels,
        max_pixels=config.data.max_pixels,
        local_files_only=True,
    )
    model = build_grounder(config.model, processor).to(device)
    load_model_checkpoint(Path(checkpoint_path), model)
    if config.train.gradient_checkpointing:
        model.backbone.gradient_checkpointing_enable()
        model.backbone.enable_input_require_grads()
    model.eval()
    model.backbone.config.use_cache = False

    dataset = GroundingDataset(
        manifest,
        config.stage,
        config.data.depth_scale,
        config.data.depth_clip,
    )
    sample_count = min(int(maxsamples), len(dataset))
    if sample_count < 1:
        raise ValueError("diagnostic manifest is empty")
    collator = NativeGroundingCollator(processor, config.stage)
    probe = FusionProbe(model)
    losses = _Summary()
    parameter_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    try:
        with torch.no_grad():
            for index in range(sample_count):
                batch = {
                    key: value.to(device) if torch.is_tensor(value) else value
                    for key, value in collator([dataset[index]]).items()
                }
                output = model(**_training_inputs(batch))
                losses.add(float(output["loss"]))
    finally:
        probe.close()
    changed_parameters = [
        name
        for name, parameter in model.named_parameters()
        if name in parameter_before and not torch.equal(parameter.detach(), parameter_before[name])
    ]
    if changed_parameters:
        raise RuntimeError("diagnostic hooks changed model parameters: " + ", ".join(changed_parameters[:5]))
    report = {
        "event": "8b_fusion_diagnostics",
        "config": str(config_path.resolve()),
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "manifest": str(manifest),
        "stage": config.stage,
        "samples": sample_count,
        "sample_ids": [str(dataset.records[index].get("id", index)) for index in range(sample_count)],
        "model_dtype": str(next(model.backbone.parameters()).dtype),
        "loss": losses.as_dict(),
        "fusion": probe.report(),
        "hooks_return_replacement": False,
        "bf16_changed_fraction_definition": "fraction of actual fusion output elements unequal to incoming RGB; interpreted as BF16 writeback only when model_dtype is bfloat16",
        "weights_changed": False,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--maxsamples", "--max-samples", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = run_diagnostics(
        args.config,
        args.checkpoint,
        args.manifest,
        args.output,
        args.maxsamples,
        args.device,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
