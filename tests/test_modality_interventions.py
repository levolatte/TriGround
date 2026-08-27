from __future__ import annotations

import torch
from torch import nn

from tools.diagnose_modality_interventions import (
    MismatchedModalityDataset,
    temporary_modality_scales,
)


class TinyDataset:
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return {"rgb": index, "ir": f"ir-{index}", "depth": f"depth-{index}"}


class Branch(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.residual_scale = nn.Parameter(torch.tensor(value))


class Stage(nn.Module):
    def __init__(self):
        super().__init__()
        self.ir, self.depth = Branch(2.0), Branch(3.0)


class Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage_fusions = nn.ModuleDict({"0": Stage()})


class Model:
    fusion_type = "parallel_backbone"
    fusion = Fusion()


def test_mismatch_keeps_target_and_replaces_selected_auxiliary():
    dataset = MismatchedModalityDataset(TinyDataset(), {"ir"}, seed=2026)
    row = dataset[0]
    assert row["rgb"] == 0
    assert row["depth"] == "depth-0"
    assert row["ir"] != "ir-0"


def test_temporary_scales_are_applied_and_restored():
    model = Model()
    with temporary_modality_scales(model, 0.25, 0.5):
        assert model.fusion.stage_fusions["0"].ir.residual_scale.item() == 0.5
        assert model.fusion.stage_fusions["0"].depth.residual_scale.item() == 1.5
    assert model.fusion.stage_fusions["0"].ir.residual_scale.item() == 2.0
    assert model.fusion.stage_fusions["0"].depth.residual_scale.item() == 3.0
