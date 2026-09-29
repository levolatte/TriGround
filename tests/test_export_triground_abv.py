"""Small-file export checks; no cloud, image decode, or model is involved."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools.export_triground_abv import export_release


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return str(path)


def _release(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    images = {}
    for name, contents in (("rgb.jpg", b"rgb-original"),
                           ("ir.jpg", b"ir-original"),
                           ("raw_depth.png", b"\x89PNG\x00raw-16-bit-depth")):
        path = source / name
        path.write_bytes(contents)
        images[name] = path
    two = {"id": "two-real-images", "image": [str(images["rgb.jpg"]), str(images["ir.jpg"])],
           "conversations": [{"from": "human", "value": "Which object?"},
                             {"from": "gpt", "value": '{"bbox_2d":[1,2,3,4]}'}]}
    three = {"id": "three-real-images", "image": [str(path) for path in images.values()],
             "conversations": two["conversations"]}
    city = {"id": "old-city", "image": ["scene/rgb.jpg", "scene/ir.jpg", "scene/depth.png"],
            "conversations": two["conversations"]}
    ready = source / "ready"
    manifests = {arm: _json(ready / f"{arm}.json", rows)
                 for arm, rows in {"A": [two, city], "B": [three, city],
                                   "V": [three, city]}.items()}
    diagnostic = _json(ready / "diagnostics" / "normal.json", [two, city])
    gt = _json(ready / "diagnostics" / "gt.json", {
        "two-real-images": {"bbox": [1, 2, 3, 4],
                            "visible": str(images["rgb.jpg"]),
                            "infrared": str(images["ir.jpg"]), "depth": ""}
    })
    scene = _json(ready / "diagnostics" / "scene_map.json", {"two-real-images": "scene-1"})
    classes = _json(ready / "diagnostics" / "class_map.json", {"two-real-images": "ir"})
    release = {"status": "ready", "steps": 600, "seed": 2026,
               "manifests": manifests, "diagnostics": {"normal": diagnostic},
               "gt_manifest": gt, "scene_map": scene, "class_map": classes,
               "counts": {"approved_unique_train": 2},
               "path_roots": {"city_old_cloud": "/old/city/train/",
                              "new_review_assets_local": str(source)},
               "audit_files": {"accepted_reviews": "accepted_reviews.jsonl",
                               "scene_exposure": "scene_exposure.json",
                               "ancestor_overlap": "ancestor_overlap.json"}}
    (source / "accepted_reviews.jsonl").write_text(
        json.dumps({"source_image": str(images["rgb.jpg"])}) + "\n", encoding="utf-8"
    )
    _json(source / "scene_exposure.json", {"A": {"scene-a": 2}})
    _json(source / "ancestor_overlap.json", {"train_diagnostic_overlap": 0})
    release_path = Path(_json(source / "release.json", release))
    return release_path, images


def test_export_preserves_two_and_three_image_order_bytes_and_deduplicates(tmp_path):
    release_path, images = _release(tmp_path)
    deployment = tmp_path / "deployment"
    summary = export_release(release_path, "/cloud/run/deployment", deployment,
                             "/cloud/city/train")
    assert summary["unique_copied_images"] == 3
    assert summary["native_rows"] == {"A": 2, "B": 2, "V": 2, "diagnostics/normal": 2}
    exported = json.loads((deployment / "release.json").read_text(encoding="utf-8"))
    assert exported["status"] == "ready" and exported["counts"] == {"approved_unique_train": 2}
    assert exported["manifests"]["A"] == "/cloud/run/deployment/manifests/A.json"
    assert exported["gt_manifest"] == "/cloud/run/deployment/manifests/diagnostics/gt_manifest.json"
    exported_gt = json.loads((deployment / "manifests/diagnostics/gt_manifest.json").read_text())
    assert exported_gt["two-real-images"]["bbox"] == [1, 2, 3, 4]
    assert exported_gt["two-real-images"]["depth"] == ""
    A = json.loads((deployment / "manifests/A.json").read_text(encoding="utf-8"))
    B = json.loads((deployment / "manifests/B.json").read_text(encoding="utf-8"))
    V = json.loads((deployment / "manifests/V.json").read_text(encoding="utf-8"))
    diag = json.loads((deployment / "manifests/diagnostics/normal.json").read_text(encoding="utf-8"))
    assert B == V
    assert len(A[0]["image"]) == len(diag[0]["image"]) == 2
    assert len(B[0]["image"]) == 3
    assert A[0]["image"] == diag[0]["image"] == B[0]["image"][:2]
    assert exported_gt["two-real-images"]["visible"] == A[0]["image"][0]
    assert exported_gt["two-real-images"]["infrared"] == A[0]["image"][1]
    assert A[1]["image"] == B[1]["image"] == [
        "scene/rgb.jpg", "scene/ir.jpg", "scene/depth.png"
    ]
    assert A[0]["conversations"] == B[0]["conversations"]
    assert len(list((deployment / "assets").iterdir())) == 3
    for original, exported_path in zip(images.values(), B[0]["image"]):
        copied = deployment / "assets" / Path(exported_path).name
        assert copied.suffix == original.suffix
        assert copied.read_bytes() == original.read_bytes()
    assert "new_review_assets_local" not in exported["path_roots"]
    assert exported["accepted_reviews"] == "/cloud/run/deployment/evidence/accepted_reviews.jsonl"
    assert exported["scene_exposure"] == "/cloud/run/deployment/evidence/scene_exposure.json"
    assert exported["audit_files"] == {
        "accepted_reviews": "/cloud/run/deployment/evidence/accepted_reviews.jsonl",
        "scene_exposure": "/cloud/run/deployment/evidence/scene_exposure.json",
        "ancestor_overlap": "/cloud/run/deployment/evidence/ancestor_overlap.json",
    }
    assert (deployment / "evidence/accepted_reviews.jsonl").read_bytes() == (
        release_path.parent / "accepted_reviews.jsonl").read_bytes()


def test_export_maps_existing_cloud_city_absolute_without_copy(tmp_path):
    release_path, _ = _release(tmp_path)
    release = json.loads(release_path.read_text())
    rows = json.loads(Path(release["manifests"]["A"]).read_text())
    rows[1]["image"][0] = "/old/city/train/scene/rgb.jpg"
    _json(Path(release["manifests"]["A"]), rows)
    deployment = tmp_path / "deployment"
    export_release(release_path, "/cloud/run/deployment", deployment, "/cloud/city/train")
    A = json.loads((deployment / "manifests/A.json").read_text())
    assert A[1]["image"][0] == "/cloud/city/train/scene/rgb.jpg"


def test_export_refuses_unready_and_existing_deployment(tmp_path):
    release_path, _ = _release(tmp_path)
    release = json.loads(release_path.read_text())
    release["status"] = "draft_unapproved"
    _json(release_path, release)
    output = tmp_path / "deployment"
    with pytest.raises(ValueError, match="human-approved ready"):
        export_release(release_path, "/cloud/run/deployment", output)
    assert not output.exists()
    release["status"] = "ready"
    _json(release_path, release)
    output.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        export_release(release_path, "/cloud/run/deployment", output)


def test_export_cli_writes_no_local_paths_to_deployment(tmp_path):
    release_path, _ = _release(tmp_path)
    output = tmp_path / "deployment"
    result = subprocess.run(
        [sys.executable, "-m", "tools.export_triground_abv", "--release", str(release_path),
         "--cloud-root", "/cloud/run/deployment", "--output-dir", str(output)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    exported = (output / "release.json").read_text(encoding="utf-8")
    assert str(release_path.parent) not in exported
    assert "/cloud/run/deployment/assets/" in (output / "manifests/A.json").read_text(encoding="utf-8")
