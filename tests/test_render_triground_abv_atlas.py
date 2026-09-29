from __future__ import annotations

import json

from PIL import Image

from tools.render_triground_abv_atlas import build
from tools.report_triground_abv import build_report


BOX = [0.1, 0.1, 0.4, 0.4]


def _fixture(tmp_path, *, row_images: bool):
    originals = []
    for modality, color in (("rgb", "red"), ("ir", "gray"), ("depth", "blue")):
        image = tmp_path / f"actual_{modality}.png"
        Image.new("RGB", (64, 64), color).save(image)
        originals.append(str(image))
    gt = {
        f"sample-{index}": {
            "visible": f"/old-cloud/rgb{index}.png",
            "infrared": f"/old-cloud/ir{index}.png",
            "depth": f"/old-cloud/depth{index}.png",
            "bbox": BOX,
            "query": "Find <red> object" if row_images else None,
        }
        for index in (1, 2)
    }
    gt_path = tmp_path / "gt.json"
    gt_path.write_text(json.dumps(gt), encoding="utf-8")
    paths = {}
    for arm, hit in (("C0", "sample-1"), ("A", "sample-2")):
        path = tmp_path / f"{arm}.jsonl"
        rows = []
        for sample_id in gt:
            row = {
                "id": sample_id,
                "prediction": BOX if sample_id == hit else None,
                "raw_text": "{\"bbox\":[0.1,0.1,0.4,0.4]}" if sample_id == hit else "invalid",
            }
            if row_images:
                row["image"] = originals
            rows.append(row)
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        paths[arm] = path
    report = build_report(gt, paths, bootstrap_replicates=2)
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return gt_path, paths, report_path, originals


def test_all_rescues_and_harms_with_original_images_and_manual_reasons(tmp_path):
    gt, paths, report, originals = _fixture(tmp_path, row_images=True)
    output = tmp_path / "atlas"
    result = build(report, gt, [f"{name}={path}" for name, path in paths.items()], output)
    assert result["flip_cases"] == 2
    assert result["missing_image_entries"] == 0
    cases = json.loads((output / "cases.json").read_text(encoding="utf-8"))
    assert [case["id"] for case in cases] == ["sample-1", "sample-2"]
    assert all(case["human_reason"] == "unknown" for case in cases)
    assert all(case["images"]["visible"] == originals[0] for case in cases)
    assert {"C0_to_A：救回", "C0_to_A：伤害"} == {
        flip for case in cases for flip in case["flips"]
    }
    page = (output / "cases.html").read_text(encoding="utf-8")
    assert "Find &lt;red&gt; object" in page
    assert "同一对象但框不准" in page
    assert "不叠加 RGB 坐标框" in page
    assert len(list((output / "visuals").glob("*.jpg"))) == 8


def test_native_manifest_fallback_for_moved_cloud_paths_and_query(tmp_path):
    gt, paths, report, originals = _fixture(tmp_path, row_images=False)
    native = tmp_path / "native.json"
    native.write_text(json.dumps([
        {"id": f"sample-{index}", "image": originals, "conversations": [{
            "value": "Locate the object described by this query: red vehicle\nReturn only JSON"
        }]}
        for index in (1, 2)
    ]), encoding="utf-8")
    output = tmp_path / "atlas"
    result = build(report, gt, [f"{name}={path}" for name, path in paths.items()], output,
                   native_manifest=native)
    assert result["missing_image_entries"] == 0
    cases = json.loads((output / "cases.json").read_text(encoding="utf-8"))
    assert all(case["images"]["infrared"] == originals[1] for case in cases)
    assert all(case["query"] == "red vehicle" for case in cases)
