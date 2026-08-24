import json

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from tools.prepare_refcoco import convert_dataset, merge_manifests


def test_converts_refcoco_parquet_to_normalized_manifest(tmp_path):
    coco = tmp_path / "coco"
    snapshot = tmp_path / "snapshot"
    output = tmp_path / "output"
    coco.mkdir()
    snapshot.mkdir()
    Image.new("RGB", (100, 50)).save(coco / "sample.jpg")
    table = pa.Table.from_pylist([{
        "split": "train",
        "ref_id": 7,
        "ann_id": 8,
        "file_name": "sample_crop_2.jpg",
        "bbox": [10.0, 5.0, 60.0, 45.0],
        "raw_image_info": json.dumps({"width": 100, "height": 50, "file_name": "sample.jpg"}),
        "sentences": [{"raw": "the object", "sent_id": 9}],
    }])
    pq.write_table(table, snapshot / "train.parquet")

    counts = convert_dataset("refcoco", snapshot, coco, output)
    record = json.loads((output / "refcoco_train.jsonl").read_text().strip())
    assert counts == {"train": 1, "val": 0}
    assert record["query"] == "the object"
    assert record["bbox"] == [0.1, 0.1, 0.6, 0.9]

    for name in ("refcocoplus", "refcocog"):
        (output / f"{name}_train.jsonl").write_text("", encoding="utf-8")
        (output / f"{name}_val.jsonl").write_text("", encoding="utf-8")
    merged = merge_manifests(output)
    assert merged == {"train": 1, "val": 0}
    assert (output / "train.jsonl").is_file()
