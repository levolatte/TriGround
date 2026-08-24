import json

from tools.build_grouped_subsets import grouped_split, write_manifest


def test_grouped_subsets_are_nested_and_validation_has_no_scene_leakage():
    rows = [
        (f"{scene}-{index}", {"scene_id": scene, "query": str(index)})
        for scene in ("a", "b", "c", "d", "e", "f", "g", "h")
        for index in range(2)
    ]
    subsets, validation = grouped_split(
        rows, (0.25, 0.5, 1.0), validation_fraction=0.25, seed=2026, group_key="scene_id"
    )
    ids = {fraction: {sample_id for sample_id, _ in subset} for fraction, subset in subsets.items()}
    assert ids[0.25] < ids[0.5] < ids[1.0]
    validation_scenes = {record["scene_id"] for _, record in validation}
    training_scenes = {record["scene_id"] for _, record in subsets[1.0]}
    assert validation_scenes.isdisjoint(training_scenes)


def test_grouped_subsets_fall_back_to_individual_sample_ids():
    rows = [(str(index), {"query": str(index)}) for index in range(8)]
    subsets, validation = grouped_split(
        rows, (0.5, 1.0), validation_fraction=0.25, seed=1, group_key=None
    )
    assert len(validation) == 2
    assert len(subsets[0.5]) == 3
    assert len(subsets[1.0]) == 6


def test_written_subsets_rebase_relative_modality_paths(tmp_path):
    source_root = tmp_path / "source"
    output = source_root / "subsets" / "train_50.json"
    write_manifest(
        output,
        [("a", {"rgb": "rgb/a.png", "aux": "thermal/a.png"})],
        "json",
        source_root,
    )
    record = json.loads(output.read_text(encoding="utf-8"))["a"]
    assert (output.parent / record["rgb"]).resolve() == source_root / "rgb/a.png"
    assert (output.parent / record["aux"]).resolve() == source_root / "thermal/a.png"
