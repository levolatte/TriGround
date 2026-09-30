"""Boundary tests for GT-free multimodal object-data preparation."""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from experiments.visual_agent import prepare_object_data as prep


def city_row(sample_id: str, group: str, query: str, box=None):
    box = box or [100, 100, 300, 300]
    prompt = ("<image>\n<image>\n<image>\nThese are aligned views of the same scene in this order: "
              "RGB, infrared, depth. Locate the object described by this query: "
              f"{query}\nReturn only JSON")
    return {
        "id": sample_id, "class_name": "must_not_become_query_or_category",
        "image": [f"visible/{group}.png", f"infrared/{group}.png",
                  f"depth_rgb/{group}.png"],
        "conversations": [{"from": "human", "value": prompt},
                          {"from": "gpt", "value": json.dumps({"bbox_2d": box})}],
    }


def rgbt_row(sample_id: str, group: str, query: str):
    return {"id": sample_id, "source": "rgbt_groundbench_flir", "aux_type": "ir",
            "rgb": f"../raw/{group}_rgb.png", "aux": f"../raw/{group}_ir.png",
            "query": query, "bbox": [0.1, 0.2, 0.6, 0.8], "scene_id": f"flir:{group}",
            "class_name": "forbidden_gt_class"}


def robo_row(sample_id: str, group: str, query: str):
    return {"id": sample_id, "source": "roborefit", "rgb": f"../final_dataset/train/image/{group}.png",
            "depth": f"../final_dataset/train/depth/{group}.png", "query": query,
            "bbox": [0.1, 0.2, 0.6, 0.8], "scene_id": f"roborefit:train:{group}",
            "original_image_id": group, "class_name": "forbidden_gt_class"}


class PrepareObjectDataTests(unittest.TestCase):
    def test_unique_query_selection_respects_group_cap_and_reserved_queries(self):
        records = [
            {"id": "a1", "image_group": "city:a", "query": "The red car."},
            {"id": "a2", "image_group": "city:a", "query": "The blue car"},
            {"id": "a3", "image_group": "city:a", "query": "The blue car "},
            {"id": "b1", "image_group": "city:b", "query": "The red car"},
            {"id": "b2", "image_group": "city:b", "query": "The white car"},
            {"id": "c1", "image_group": "city:c", "query": "The black car"},
        ]
        selected, stats = prep.select_unique_queries(
            records, 3, max_per_group=2, seed=2032, reserved_queries={"the red car"})
        queries = [prep.normalized_query(row["query"]) for row in selected]
        self.assertEqual(len(selected), 3)
        self.assertEqual(len(set(queries)), 3)
        self.assertNotIn("the red car", queries)
        counts = {group: sum(row["image_group"] == group for row in selected)
                  for group in {row["image_group"] for row in selected}}
        self.assertLessEqual(max(counts.values()), 2)
        self.assertEqual(stats["selected_queries"], 3)

    def test_source_rows_keep_only_real_modalities_and_gt_private(self):
        city = prep._city_record(city_row("c1", "scene1", "The red car"), "/cloud/city/train")
        rgbt = prep._rgbt_record(rgbt_row("r1", "frame1", "A silver sedan"), "/cloud/rgbt")
        robo = prep._robo_record(robo_row("b1", "0000001", "Pick the blue cup"),
                                 "F:/data/robo/converted", "F:/data/robo/converted")
        for row in (city, rgbt, robo):
            public = prep._public(row)
            encoded = json.dumps(public, ensure_ascii=False)
            self.assertNotIn("bbox", encoded)
            self.assertNotIn("class_name", encoded)
            self.assertNotIn("must_not_become_query_or_category", encoded)
            self.assertNotIn("forbidden_gt_class", encoded)
            self.assertEqual(len(row["_bbox"]), 4)
        self.assertEqual(set(rgbt["images"]), {"rgb", "ir"})
        self.assertEqual(rgbt["ir_rgb_registration"], "normalized_shared_frame")
        self.assertEqual(set(robo["images"]), {"rgb", "depth_raw", "depth_visual"})
        self.assertEqual(robo["depth_encoding"], "unknown")
        self.assertEqual(robo["ir_rgb_registration"], "not_available")
        self.assertEqual(prep._cloud_path("F:/AIC/data/robo/a.png", "F:/AIC", "/cloud/root"),
                         "/cloud/root/data/robo/a.png")
        self.assertEqual(prep._cloud_path("/already/cloud/a.png", "F:/AIC", "/cloud/root"),
                         "/already/cloud/a.png")

    def test_rgbdt_visualization_preserves_numeric_order_without_near_far_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "depth.png"
            visual = root / "depth_visual.png"
            Image.fromarray(np.array([[0, 100, 200, 400, 800]], dtype=np.uint16)).save(raw)
            prep._depth_visual(raw, visual)
            values = np.asarray(Image.open(visual).convert("RGB"))[:, :, 0]
            self.assertEqual(int(values[0, 0]), 0)
            self.assertLess(int(values[0, 1]), int(values[0, 2]))
            self.assertLess(int(values[0, 2]), int(values[0, 4]))

    def test_prepare_creates_isolated_splits_and_gt_free_stage_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            city_train = root / "city_train.json"
            city_train.write_text(json.dumps([
                city_row("city_keep", "keep", "The red car"),
                city_row("city_val", "val", "The yellow bus"),
                city_row("city_diag", "diag", "The blue bike"),
                city_row("city_000002_025_frame001", "shared", "The violet tree"),
            ]), encoding="utf-8")
            city412 = root / "city412.jsonl"
            city412.write_text(json.dumps({"id": "held-city", "images": {"rgb": "/x/visible/val.png"}}) + "\n",
                                encoding="utf-8")
            holdout = root / "holdout.jsonl"
            holdout.write_text(json.dumps({"id": "old-t", "images": {"rgb": "/x/visible/old.png"}}) + "\n",
                               encoding="utf-8")
            rgbt = root / "rgbt.jsonl"
            rgbt.write_text("\n".join(json.dumps(row) for row in [
                rgbt_row("rgbt_keep", "keep", "A silver sedan"),
                rgbt_row("rgbt_diag", "diag", "A white truck"),
            ]) + "\n", encoding="utf-8")
            robo = root / "robo.jsonl"
            robo.write_text("\n".join(json.dumps(row) for row in [
                robo_row("robo_keep", "keep", "Pick the blue cup"),
                robo_row("robo_diag", "diag", "Pick the green cup"),
            ]) + "\n", encoding="utf-8")
            diagnostics = root / "diagnostics.json"
            diagnostics.write_text(json.dumps([
                {"id": "d-city", "source": "city", "scene_id": "city:diag"},
                {"id": "d-rgbt", "source": "rgbt_groundbench_flir", "scene_id": "flir:diag"},
                {"id": "d-robo", "source": "roborefit", "scene_id": "roborefit:train:diag"},
            ]), encoding="utf-8")
            release = root / "release.json"
            release.write_text(json.dumps({"counts": {"approved_unique_train": 232,
                                                        "approved_diagnostic": 70}}), encoding="utf-8")

            archive_path = root / "train.zip"
            frames = [f"{index:08d}.png" for index in range(1, 11)]
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for sequence in ("001", "002", "003", "004"):
                    archive.writestr(f"{sequence}/groundtruth.txt", "".join(
                        f"{frame},0,0,2,2\n" for frame in frames))
                    for frame in frames:
                        for modality in ("color", "infrared"):
                            image_path = root / f"{sequence}_{modality}_{frame}"
                            Image.new("RGB", (4, 3), (10, 20, 30)).save(image_path, format="PNG")
                            archive.write(image_path, f"{sequence}/{modality}/{frame}")
                        depth_path = root / f"{sequence}_depth_{frame}"
                        Image.fromarray(np.ones((3, 4), dtype=np.uint16) * 1000).save(depth_path)
                        archive.write(depth_path, f"{sequence}/depth/{frame}")
            plan = root / "plan.json"
            plan.write_text(json.dumps({
                "pilot": [{"sequence": "004", "frame": frames[0]}],
                "external_review_sequences": ["002"],
                "remaining_train_sequences": ["001", "003"],
            }), encoding="utf-8")
            external = root / "external.jsonl"
            external.write_text(json.dumps({"sequence": "002"}) + "\n", encoding="utf-8")

            out = root / "out"
            args = SimpleNamespace(
                city_train=city_train, city412=city412, old_t_holdout=holdout,
                mainline_diagnostics=diagnostics, mainline_release=release,
                rgbt_manifest=rgbt, robo_manifest=robo,
                city_image_root="/cloud/city/train", rgbt_image_root="/cloud/rgbt",
                robo_image_root="F:/AIC/data/robo/converted", robo_depth_visual_root="F:/AIC/data/robo/converted",
                rgbdt_plan=plan, rgbdt_external_manifest=external, rgbdt_zip=archive_path,
                rgbdt_count=1, city_count=1, rgbt_count=1, robo_count=1,
                output_dir=out, seed=2032,
                local_root="F:/AIC", cloud_root="/root/autodl-tmp/rematch_20260922",
            )
            inventory = prep.prepare(args)
            public = prep.read_jsonl(out / "manifest.jsonl")
            private = prep.read_jsonl(out / "private_labels_offline.jsonl")
            annotation_jobs = prep.read_jsonl(out / "annotation_jobs.jsonl")
            self.assertEqual({row["id"] for row in public}, {"city_keep", "rgbt_keep", "robo_keep"})
            self.assertEqual(len(private), 4)
            self.assertEqual(len(annotation_jobs), 1)
            self.assertNotIn("query", annotation_jobs[0])
            self.assertNotIn("bbox", annotation_jobs[0])
            self.assertEqual(inventory["actual"]["globally_unique_explicit_queries"], 3)
            self.assertEqual(inventory["actual"]["rgbdt_annotation_jobs"], 1)
            self.assertEqual(inventory["exclusions"]["known_shared_video_groups"], 1)
            cloud_public = prep.read_jsonl(out / "cloud_manifest.jsonl")
            self.assertEqual({row["id"] for row in cloud_public}, {"city_keep", "rgbt_keep", "robo_keep"})
            robo_cloud = next(row for row in cloud_public if row["source"] == "roborefit")
            self.assertTrue(robo_cloud["images"]["rgb"].startswith("/root/autodl-tmp/rematch_20260922/"))
            self.assertTrue((out / "group_splits.jsonl").exists())
            for row in public:
                self.assertFalse({"bbox", "gt", "target_bbox", "class_name"} & set(row))
                self.assertIn("depth_encoding", row)
                self.assertIn("ir_rgb_registration", row)
            for stage in ("foundation_candidate", "observation", "next_action"):
                jobs = prep.read_jsonl(out / f"{stage}_jobs.jsonl")
                self.assertEqual(len(jobs), 3)
                self.assertTrue(all(job["status"] == "pending" for job in jobs))
            selected_sequences = {row["sequence"] for row in annotation_jobs}
            self.assertTrue(selected_sequences <= {"001", "003"})
            self.assertNotIn("002", selected_sequences)
            self.assertEqual(len(list((out / "images/rgbdt500").rglob("*.png"))), 4)


if __name__ == "__main__":
    unittest.main()
