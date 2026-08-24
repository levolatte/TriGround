from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path

import pytest

from tools.review_grounding import Handler, ReviewServer, ReviewStore, build_dataset, safe_image_path


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_build_dataset_groups_scenes_and_attaches_predictions(tmp_path: Path):
    manifest = tmp_path / "val.json"
    write_json(
        manifest,
        {
            "one": {
                "visible": "visible/a.png",
                "infrared": "infrared/a.png",
                "depth": "depth/a.png",
                "query": "The leftmost person",
                "bbox": [0.1, 0.1, 0.2, 0.2],
                "class_name": "person",
                "scale_bin": "small",
                "weak_label": True,
            },
            "two": {
                "visible": "visible/a.png",
                "infrared": "infrared/a.png",
                "depth": "depth/a.png",
                "query": "The second person from the left",
                "bbox": [0.4, 0.1, 0.5, 0.2],
                "class_name": "person",
                "scale_bin": "large",
                "weak_label": True,
            },
        },
    )
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        json.dumps({"id": "one", "prediction": [0.1, 0.1, 0.2, 0.2]}) + "\n",
        encoding="utf-8",
    )

    dataset = build_dataset(manifest, predictions)

    assert dataset["scene_count"] == 1
    assert dataset["sample_count"] == 2
    assert dataset["scenes"][0]["samples"][0]["prediction_iou"] == pytest.approx(1.0)
    assert dataset["scenes"][0]["samples"][1]["query_type"] == "ordinal"
    assert dataset["scenes"][0]["samples"][1]["same_class_count"] == 2


def test_review_store_merges_and_persists_updates(tmp_path: Path):
    path = tmp_path / "reviews.json"
    store = ReviewStore(path, "manifest-hash")

    store.update(["sample"], {"decision": "ambiguous"})
    store.update(
        ["sample"],
        {
            "cause": "data_ambiguous",
            "note": "missing person",
            "corrected_query": "The person in the red coat",
        },
    )
    loaded = ReviewStore(path, "manifest-hash")

    assert loaded.data["reviews"]["sample"]["decision"] == "ambiguous"
    assert loaded.data["reviews"]["sample"]["cause"] == "data_ambiguous"
    assert loaded.data["reviews"]["sample"]["note"] == "missing person"
    assert loaded.data["reviews"]["sample"]["corrected_query"] == "The person in the red coat"
    with pytest.raises(ValueError, match="Unknown decision"):
        loaded.update(["sample"], {"decision": "invented"})


def test_safe_image_path_rejects_traversal(tmp_path: Path):
    root = tmp_path / "data"
    root.mkdir()
    image = root / "image.png"
    image.write_bytes(b"png")
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")

    assert safe_image_path(root, "image.png") == image.resolve()
    with pytest.raises(FileNotFoundError):
        safe_image_path(root, "../secret.txt")


def test_http_api_serves_dataset_image_and_saves_review(tmp_path: Path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    image = data_root / "image.png"
    image.write_bytes(b"fake-png")
    manifest = tmp_path / "val.json"
    write_json(
        manifest,
        {
            "sample": {
                "visible": "image.png",
                "infrared": "image.png",
                "depth": "image.png",
                "query": "The person near the center of the image",
                "bbox": [0.1, 0.1, 0.2, 0.2],
                "class_name": "person",
                "scale_bin": "large",
            }
        },
    )
    dataset = build_dataset(manifest, None)
    store = ReviewStore(tmp_path / "reviews.json", dataset["manifest_sha256"])
    html = tmp_path / "index.html"
    html.write_text("reviewer", encoding="utf-8")
    server = ReviewServer(
        ("127.0.0.1", 0),
        Handler,
        dataset=dataset,
        data_root=data_root,
        store=store,
        html_path=html,
    )
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        bootstrap = json.loads(urllib.request.urlopen(f"{base}/api/bootstrap").read())
        assert bootstrap["sample_count"] == 1
        assert urllib.request.urlopen(f"{base}/image?path=image.png").read() == b"fake-png"
        request = urllib.request.Request(
            f"{base}/api/review",
            data=json.dumps(
                {"sample_ids": ["sample"], "patch": {"decision": "valid"}}
            ).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        response = json.loads(urllib.request.urlopen(request).read())
        assert response["ok"] is True
        assert store.data["reviews"]["sample"]["decision"] == "valid"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
