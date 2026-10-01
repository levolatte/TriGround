from experiments.evidence_decision import prioritize_blind_jobs as prioritizer


def _job(sample_id, bucket="binding", query="a person"):
    return {"sample_id": sample_id, "selection_bucket": bucket, "query": query,
            "available_modalities": ["rgb", "ir", "depth_raw"],
            "images": {"rgb": "/remote/rgb.png", "ir": "/remote/ir.png",
                       "depth_raw": "/remote/depth.png"}}


def _task(query="a person"):
    return {"query": query, "depth_encoding": "city_mm",
            "available_modalities": ["rgb", "ir", "depth_raw"],
            "images": {"rgb": "/remote/rgb.png", "ir": "/remote/ir.png",
                       "depth_raw": "/remote/depth.png"},
            "ir_rgb_registration": "normalized_shared_frame"}


def test_camera_depth_proxy_excludes_left_right_and_reference_cases():
    clean = prioritizer.classify(
        _job("clean", query="Which person is farther from the camera?"),
        _task(query="Which person is farther from the camera?"),
        {"query_info": {"target_category": "person", "relation_type": "camera_far",
                         "reference_categories": []},
         "candidates": [{"role": "target"}, {"role": "target"}]},
    )
    mixed = prioritizer.classify(
        _job("mixed", query="Which person on the left is farther from the camera?"),
        _task(query="Which person on the left is farther from the camera?"),
        {"query_info": {"target_category": "person", "relation_type": "camera_far",
                         "reference_categories": []},
         "candidates": [{"role": "target"}, {"role": "target"}]},
    )
    reference = prioritizer.classify(
        _job("reference", query="Which person is closer to the camera?"),
        _task(query="Which person is closer to the camera?"),
        {"query_info": {"target_category": "person", "relation_type": "other",
                         "reference_categories": ["table"]},
         "candidates": [{"role": "target"}, {"role": "target"}]},
    )

    assert clean["priority_features"]["camera_distance_city_mm_clean"] is True
    assert mixed["priority_features"]["camera_distance_city_mm_clean"] is False
    assert mixed["priority_features"]["camera_distance_city_mm_mixed"] is True
    assert reference["priority_features"]["camera_distance_city_mm_clean"] is False


def test_missing_semantic_categories_are_not_claimed_as_category_gaps():
    result = prioritizer.classify(
        _job("gap", query="The second cup next to the jar"),
        _task(query="The second cup next to the jar"),
        {"query_info": {"target_category": "cup", "reference_categories": ["jar"],
                         "relation_type": "other", "scope": "group"},
         "candidates": [{"role": "target", "sources": [{"modality": "rgb"}]}]},
    )

    assert result["priority_features"]["reference_proposal_gap"] is True
    assert result["priority_features"]["semantic_category_match"] is None
    assert "reference_proposal_gap" in result["priority_reasons"]


def test_first_batch_uses_all_three_buckets():
    rows = []
    for bucket in prioritizer.BUCKETS:
        for index in range(3):
            rows.append({"sample_id": f"{bucket}-{index}", "selection_bucket": bucket,
                         "priority_score": index, "priority_reasons": [],
                         "priority_features": {}})

    first = prioritizer._first_batch(rows, batch_size=3)

    assert [row["selection_bucket"] for row in first] == list(prioritizer.BUCKETS)


def test_first_batch_keeps_all_camera_depth_jobs_when_they_fit_bucket_quotas():
    rows = []
    for index in range(3):
        features = {'camera_distance_city_mm_clean': index == 0,
                    'camera_distance_city_mm_mixed': index == 1,
                    'camera_distance_city_mm_any': index < 2}
        rows.append({"sample_id": f"binding-{index}", "selection_bucket": "binding",
                     "priority_score": 1, "priority_reasons": [], "priority_features": features})
    for bucket in ('initial_open_choice', 'finish_sufficiency_review'):
        for index in range(3):
            rows.append({"sample_id": f"{bucket}-{index}", "selection_bucket": bucket,
                         "priority_score": 1, "priority_reasons": [],
                         "priority_features": {"camera_distance_city_mm_any": False}})

    first = prioritizer._first_batch(rows, batch_size=6)

    assert {row["sample_id"] for row in first if row["selection_bucket"] == "binding"} >= {
        "binding-0", "binding-1"
    }
    assert {row["selection_bucket"] for row in first} == set(prioritizer.BUCKETS)
