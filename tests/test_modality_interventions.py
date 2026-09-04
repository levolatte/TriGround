from __future__ import annotations

from tools.diagnose_modality_interventions import (
    MismatchedModalityDataset,
    mode_definitions,
)


class TinyDataset:
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return {"rgb": index, "ir": f"ir-{index}", "depth": f"depth-{index}"}


def test_mismatch_keeps_target_and_replaces_selected_auxiliary():
    dataset = MismatchedModalityDataset(TinyDataset(), {"ir"}, seed=2026)
    row = dataset[0]
    assert row["rgb"] == 0
    assert row["depth"] == "depth-0"
    assert row["ir"] != "ir-0"


def test_scale_modes_include_zero_half_and_full_joint_interventions():
    modes = {mode["name"]: mode for mode in mode_definitions()}
    assert modes["triple_ir_0_depth_0"]["ir_scale"] == 0.0
    assert modes["triple_ir_0.5_depth_1"]["ir_scale"] == 0.5
    assert modes["triple_ir_1_depth_1"]["depth_scale"] == 1.0
