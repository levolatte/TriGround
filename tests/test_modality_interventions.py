from __future__ import annotations

from tools.diagnose_modality_interventions import (
    MismatchedModalityDataset,
    mismatched_queries,
    mode_definitions,
    query_mode_definitions,
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


def test_query_modes_keep_main_query_separate_from_fusion_intervention():
    modes = {mode["name"]: mode for mode in query_mode_definitions()}
    assert modes["query_correct"]["query_scale"] == 1.0
    assert modes["query_zero"]["query_scale"] == 0.0
    assert modes["query_shuffled"]["query_mismatch"] is True


def test_mismatched_queries_prefer_different_text_and_scene():
    records = [
        {"id": "a", "query": "left person", "visible": "scene-a.png"},
        {"id": "b", "query": "right car", "visible": "scene-b.png"},
        {"id": "c", "query": "near bicycle", "visible": "scene-c.png"},
    ]
    donors = mismatched_queries(records, seed=2026)
    by_id = {record["id"]: record for record in records}
    for sample_id, donor_query in donors.items():
        assert donor_query != by_id[sample_id]["query"]
