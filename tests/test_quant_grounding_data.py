import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import prepare_quant_grounding as prep


def test_reviewed_sources_merge_and_hide_answers():
    missing = set()
    rgbdt, revisions = prep.make_rgbdt_and_pairs(prep.DATA, missing)
    robo = prep.make_robo(prep.DATA, missing)
    assert not missing
    assert len(rgbdt) == 176  # 98 old + 148 reviewed - 70 repeated Query texts
    assert len(robo) == 2070  # 2,000 primary + 78 reviewed - 8 repeated Query texts
    assert len(revisions) == 15
    revised = next(r for r in rgbdt if r["id"] == "rgbdt500_013_00000160::rgbdt500_013_00000160_q01")
    assert revised["solution"] == [0.514, 0.446, 0.62, 0.706]
    assert revised["label_source"] == "reviewed_pair"
    assert revised["negative_boxes"]
    assert all(prep.iou(row["solution"], bad) < 0.9
               for row in rgbdt + robo for bad in row["negative_boxes"])
    assert all(len(row["negative_boxes"]) == len(row["negative_kind"])
               for row in rgbdt + robo)
    assert all(len(row["images"]) == len(row["modalities"]) for row in rgbdt + robo)
    assert all(not Path(image).is_absolute() for row in rgbdt + robo for image in row["images"])
    for row in rgbdt + robo:
        assert len(row["messages"]) == 1 and row["messages"][0]["role"] == "user"
        visible = json.dumps(row["messages"], ensure_ascii=False)
        for secret in ("review_note", "evidence", "negative_boxes", "solution"):
            assert secret not in visible
    assert all(row["modalities"] == ["rgb", "depth"] and row["depth_policy"] == "visual"
               for row in robo)


def test_group_holdout_and_four_arm_schedule():
    missing = set()
    rgbdt, _ = prep.make_rgbdt_and_pairs(prep.DATA, missing)
    robo = prep.make_robo(prep.DATA, missing)
    external, excluded = prep.make_external_dev(prep.DATA, missing)
    assert len(external) == 94 and not missing
    rows = [r for r in rgbdt if r["group_id"] not in excluded] + robo
    train, holdout, counts = prep.holdout_split(rows, 2026)
    assert counts == {"rgbdt": 10, "robo": 200}
    assert {r["group_id"] for r in train}.isdisjoint({r["group_id"] for r in holdout})
    assert {r["group_id"] for r in train}.isdisjoint(excluded)
    a = prep.schedule(train, 80, 2026, local_pool=True)
    b = prep.schedule(train, 80, 2026, local_pool=True)
    assert a == b
    assert Counter(r["sampling_pool"] for r in a) == {
        "regular_rgbdt": 20, "regular_robo": 20, "wrong_object": 20, "extent": 20,
    }
    assert len({r["occurrence_id"] for r in a}) == 80
    assert len({r["id"] for r in a}) < 80  # repeated presentation is explicit
    assert all("extent" in r["negative_kind"] for r in a if r["sampling_pool"] == "extent")
    coverage = prep.coverage(train, a, local_pool=True)
    assert not coverage["all_source_groups_covered"]
    assert coverage["by_source"]["robo"]["unseen_queries"] > 0
    assert coverage["by_pool"]["extent"]["presentations"] == 20
    assert coverage["by_pool"]["extent"]["repeat_exposures"] >= 0


def test_formal_inputs_required():
    with pytest.raises(ValueError, match="city-media-root"):
        prep.build(output=prep.DATA / "quant_grounding_20261026" / "required_test",
                   data_root=prep.DATA, local_pool=False, presentations=20)
    rows = [prep.sample(sample_id="a", source="rgbdt", group_id="rgbdt:1",
                        query="the object", images=["a.png"], modalities=["rgb"],
                        depth_policy=None, bbox=[0.1, 0.1, 0.2, 0.2],
                        label_source="test")]
    with pytest.raises(ValueError, match="regular_city"):
        prep.schedule(rows, 20, 2026, local_pool=False)


def test_official_test_frames_excluded_from_city_train_and_dev():
    exclusions = prep.DATA / "quant_grounding_20261026/recovered_city_test_exclusions.json"
    test_frames = set(json.loads(exclusions.read_text(encoding="utf-8"))["excluded_frame_keys"])
    missing = set()
    city, dev, removed = prep.make_city(
        prep.DATA / "absent_city_media", prep.DATA,
        prep.DATA / "quant_grounding_20261026_formal_test",
        exclusions, missing,
    )
    assert len(city) == 3178
    assert len(dev) == 378
    assert len({row["group_id"] for row in dev}) == 74
    assert removed["official_test"] == 529
    assert removed["official_test_dev"] == 22
    assert all(row["group_id"].split(":", 1)[1] not in test_frames for row in city + dev)
    assert not missing  # media are checked only after final exclusions
    missing_final = prep.final_missing_media(city + dev, prep.DATA, prep.DATA / "absent_city_media")
    assert any({"absent_city_media", "depth"} <= set(Path(path).parts) for path in missing_final)


def test_portable_sources_point_to_data_root_media(tmp_path):
    paths = prep.source_paths(tmp_path / "data", tmp_path / "sources")
    assert paths.pairs == tmp_path / "sources/pairs.jsonl"
    assert paths.bundle == tmp_path / "data/grounding_aux_round1_20261025/bundle"
    assert paths.robo_media_dir == tmp_path / "data/external/RoboRefIt/subset/converted"


def test_group_exclusion_keeps_original_dev_groups_out_and_ignores_replaced_media(monkeypatch):
    excluded_file = prep.DATA / "quant_grounding_source/official_test_excluded_groups.json"
    external, original_dev_groups = prep.make_external_dev(prep.DATA, set())
    assert len(external) == 94
    assert "rgbdt:297" in original_dev_groups
    assert sum(row["group_id"] == "rgbdt:297" for row in external) == 6

    rgbdt, _ = prep.make_rgbdt_and_pairs(prep.DATA, set())
    final_images = {image for row in rgbdt for image in row["images"]}
    old_images = [prep.relative_image(Path(image), prep.DATA)
                  for row in prep.read_jsonl(prep.AUX / "inventory/rgbdt_train.jsonl")
                  for image in row["images"]]
    obsolete = next(prep.DATA / image for image in old_images if image not in final_images)
    original_is_file = Path.is_file
    monkeypatch.setattr(Path, "is_file", lambda path: False if path == obsolete else original_is_file(path))

    with tempfile.TemporaryDirectory(prefix="quant_group_test_", dir=prep.DATA) as directory:
        output = Path(directory)
        report = prep.build(output=output, data_root=prep.DATA, local_pool=True,
                            presentations=80, official_test_excluded_groups=excluded_file)
        train = prep.read_jsonl(output / "train_index.jsonl")
        holdout = prep.read_jsonl(output / "holdout.jsonl")
        dev = prep.read_jsonl(output / "external94_dev.jsonl")
        assert len(dev) == 88
        assert all(row["group_id"] != "rgbdt:297" for row in dev)
        assert {row["group_id"] for row in train + holdout}.isdisjoint(original_dev_groups)
        assert report["excluded_group_ids"] == ["rgbdt:297"]
        assert report["official_test_group_removed_rows"]["external94_dev"] == {"rgbdt": 6}
        assert report["dev_rows"]["external_source_reviewed"] == 94
        assert report["dev_rows"]["external_actual"] == 88
        assert report["missing_media_count"] == 0


def test_cpu_only_assertion(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: True)))
    with pytest.raises(RuntimeError, match="CPU-only"):
        prep.assert_cpu_only()
