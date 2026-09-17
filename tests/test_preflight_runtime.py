import sys

import pytest

from mm_grounding.config import ExperimentConfig, ModelConfig
from tools import preflight


def test_preflight_can_disable_dropout_without_mutating_frozen_config(monkeypatch):
    config = ExperimentConfig(output_dir="runs/test", model=ModelConfig(modality_dropout=0.2))
    monkeypatch.setattr(preflight, "load_config", lambda path: config)

    class Dataset:
        def __init__(self, *args):
            pass

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return {}

    class ProcessorReached(Exception):
        pass

    def processor(*args, **kwargs):
        raise ProcessorReached

    monkeypatch.setattr(preflight, "GroundingDataset", Dataset)
    monkeypatch.setattr(preflight.AutoProcessor, "from_pretrained", processor)
    monkeypatch.setattr(sys, "argv", ["preflight.py", "--config", "unused.yaml"])
    with pytest.raises(ProcessorReached):
        preflight.main()
    assert config.model.modality_dropout == 0.2
