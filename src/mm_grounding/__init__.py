"""Unified RGB/IR/depth referring-expression grounding package."""

from .config import ExperimentConfig, load_config
from .model import MultiModalGrounder, build_grounder

__all__ = [
    "ExperimentConfig",
    "MultiModalGrounder",
    "build_grounder",
    "load_config",
]
