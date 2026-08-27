from tools.select_weak_subset import select_scene_diverse_stratified


def test_subset_is_exact_deterministic_scene_diverse_and_unchanged():
    rows = [
        (
            f"scene-{scene}-{index}",
            {
                "visible": f"visible/{scene}.png",
                "class_name": "person" if index == 0 else "car",
                "scale_bin": "large" if scene % 2 else "small",
                "query": f"original query {scene} {index}",
            },
        )
        for scene in range(20)
        for index in range(2)
    ]
    first = select_scene_diverse_stratified(rows, size=12, seed=2026)
    second = select_scene_diverse_stratified(rows, size=12, seed=2026)
    assert first == second
    assert len(first) == 12
    assert len({record["visible"] for _, record in first}) == 12
    source = dict(rows)
    assert all(record == source[sample_id] for sample_id, record in first)
