from tools.prepare_instance_candidates import build_candidates


def test_candidates_use_complete_image_class_and_exclude_duplicates_or_overlap():
    source = {
        # Same sequence metadata, but a different complete image: no cross-image candidate.
        "other-image": {
            "visible": "../visible/other.png",
            "infrared": "../infrared/other.png",
            "depth": "../depth/other.png",
            "query": "other car",
            "bbox": [0.1, 0.1, 0.3, 0.3],
            "class_name": "car",
            "sequence_id": "same-sequence",
        },
        "positive": {
            "visible": "../visible/frame.png",
            "infrared": "../infrared/frame.png",
            "depth": "../depth/frame.png",
            "query": "the car",
            "bbox": [0.1, 0.1, 0.5, 0.5],
            "class_name": "car",
            "sequence_id": "same-sequence",
            "original_query": "old car",
            "review_decision": "query_corrected",
        },
        # Exact duplicate of the positive: it must be deduplicated and not become a negative.
        "duplicate": {
            "visible": "../visible/frame.png",
            "infrared": "../infrared/frame.png",
            "depth": "../depth/frame.png",
            "query": "same car wording",
            "bbox": [0.1, 0.1, 0.5, 0.5],
            "class_name": "car",
            "sequence_id": "same-sequence",
        },
        # Distinct but IoU >= 0.5 with the positive: not a negative.
        "overlap": {
            "visible": "../visible/frame.png",
            "infrared": "../infrared/frame.png",
            "depth": "../depth/frame.png",
            "query": "overlapping label",
            "bbox": [0.1, 0.1, 0.5, 0.45],
            "class_name": "car",
            "sequence_id": "same-sequence",
        },
        # Distinct and sufficiently separated: it is a known negative candidate.
        "negative": {
            "visible": "../visible/frame.png",
            "infrared": "../infrared/frame.png",
            "depth": "../depth/frame.png",
            "query": "the other car",
            "bbox": [0.6, 0.6, 0.8, 0.8],
            "class_name": "car",
            "sequence_id": "same-sequence",
        },
    }

    output, stats = build_candidates(source)

    assert output["other-image"]["negative_bboxes"] == []
    assert output["positive"]["positive_bbox"] == [0.1, 0.1, 0.5, 0.5]
    assert output["positive"]["negative_bboxes"] == [[0.6, 0.6, 0.8, 0.8]]
    assert output["duplicate"]["negative_bboxes"] == [[0.6, 0.6, 0.8, 0.8]]
    assert stats["duplicate_bbox_occurrences_removed"] == 1
    assert stats["overlap_bboxes_excluded"] == 3
