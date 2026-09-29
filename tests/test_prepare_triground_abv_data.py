from __future__ import annotations

import ast
import csv
import json
import re
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from tools.prepare_qwen3vl_native_sft import _trimodal_prompt, native_prompt
from tools.prepare_triground_abv_data import (
    OUT,
    CITY_RAW_ZIP,
    _schedule,
    _select_release,
    _approved,
    accepted_candidates,
    DECISION_FIELDS,
    city_raw_labels,
    diagnostic_gt_record,
    read_jsonl,
    release,
    render_review,
)
from tools.report_rematch_experiment import load_manifest
from tools.run_triground_abv import validate_release


class TrigroundDataIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not (OUT / "data/candidates.jsonl").is_file() or not CITY_RAW_ZIP.is_file():
            raise unittest.SkipTest("local candidate atlas and author ZIP are required for integration checks")
        cls.candidates = read_jsonl(OUT / "data/candidates.jsonl")

    def test_city_float_gt_and_bad_class_scene_removed(self) -> None:
        original = city_raw_labels()
        city = [row for row in self.candidates if row["source"] == "city"]
        self.assertTrue(city)
        self.assertFalse(any("hehe_235_000002_044_00000109:nearfar" in row["task_id"] for row in city))
        self.assertFalse(any("000014_022_00000069:nearfar" in row["task_id"] for row in city))
        for row in city:
            self.assertEqual(row["bbox"], original[row["source_id"]]["bbox"])
            if row["category"] in {"depth_relation", "diag_depth"}:
                self.assertGreaterEqual(row["depth_core"]["core_valid_fraction"], .5)
        self.assertTrue(any(any(round(value * 1000) != value * 1000 for value in row["bbox"])
                            for row in city))

    def test_review_payload_preserves_candidates_and_all_translations(self) -> None:
        page = (OUT / "review/index.html").read_text(encoding="utf-8")
        embedded = re.search(r'<script id="review-data" type="application/json">(.*?)</script>',
                             page, re.S).group(1)
        payload = json.loads(embedded)
        self.assertEqual(payload["version"], "abv-review-v2")
        self.assertEqual(sum(len(bundle["tasks"]) for bundle in payload["bundles"]), len(self.candidates))
        self.assertEqual(payload["bundles"][0]["tasks"][0]["proposed_query"],
                         self.candidates[0]["proposed_query"])
        queries = {row[key].strip() for row in self.candidates
                   for key in ("proposed_query", "original_query") if row.get(key)}
        self.assertEqual(set(payload["translations"]), queries)
        self.assertTrue(all(re.search(r"[\u4e00-\u9fff]", value)
                            for value in payload["translations"].values()))
        originals = {row["task_id"]: row for row in self.candidates}
        for bundle in payload["bundles"]:
            for task in bundle["tasks"]:
                self.assertEqual({key: value for key, value in task.items() if key != "answer_image"},
                                 originals[task["task_id"]])
        # The template is shared; Query text is populated from the payload.
        self.assertNotIn(self.candidates[0]["original_query"], page.replace(embedded, ""))

    def test_unapproved_csv_cannot_release(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "data").mkdir()
            (output / "review").mkdir()
            (output / "data/candidates.jsonl").write_bytes((OUT / "data/candidates.jsonl").read_bytes())
            (output / "review/decisions_blank.csv").write_bytes(
                (OUT / "review/decisions_blank.csv").read_bytes())
            with self.assertRaisesRegex(ValueError, "minimums not met"):
                release(output / "review/decisions_blank.csv", steps=600, output=output)
            self.assertFalse(list((output / "data").glob("release_*_seed*/release.json")))

    def test_400_and_600_schedule_is_paired_and_capped(self) -> None:
        # Drafts are only used in memory to test the sampler; no ready files are written.
        mock_accepted = [{**row, "query": row["proposed_query"]} for row in self.candidates]
        # At 400 steps each relation quota rounds down to an even count
        # (340 Depth + 298 competition); the two spare positions go to City.
        for steps, old_count, new_count in ((400, 1922, 1278), (600, 2880, 1920)):
            selected, diagnostic, _ = _select_release(mock_accepted, steps)
            self.assertEqual(len(diagnostic), 96)
            with tempfile.TemporaryDirectory() as directory:
                arms = _schedule(selected, steps, Path(directory), OUT)
            self.assertEqual({name: len(rows) for name, rows in arms.items()},
                             {"A": steps*8, "B": steps*8, "V": steps*8})
            self.assertEqual(arms["B"], arms["V"])
            self.assertEqual(sum(row["task_category"] == "old_city" for row in arms["B"]), old_count)
            self.assertEqual(sum(row["task_category"] != "old_city" for row in arms["B"]), new_count)
            for position, (a, b) in enumerate(zip(arms["A"], arms["B"], strict=True)):
                if position % 80 < 48:
                    self.assertEqual(a, b)
            old_ids = Counter(row["origin_query_id"] for row in arms["A"])
            self.assertLessEqual(max(old_ids.values()), 2)
            new_ids = Counter(row["source_task_id"] for row in arms["B"]
                              if row["task_category"] != "old_city")
            self.assertLessEqual(max(new_ids.values()), 8)
            self.assertEqual(len(set(row["id"] for row in arms["B"])), steps*8)

class TrigroundDataUnitTest(unittest.TestCase):
    def test_review_refresh_preserves_human_csv_and_escapes_embedded_query(self) -> None:
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            review = output / "review"
            review.mkdir()
            original = output / "rgb.png"
            Image.new("RGB", (32, 24)).save(original)
            for name in ("decisions.csv", "decisions_blank.csv"):
                (review / name).write_text("human record,do not replace\n", encoding="utf-8")
            task = {"task_id": "toy", "bundle_id": "scene", "scene_id": "scene",
                    "category": "ir_complement", "split": "train", "source": "fixture",
                    "images": {"rgb": str(original)}, "bbox": [.1, .1, .8, .8],
                    "proposed_query": '</script><img src=x onerror="bad">'}
            render_review([task], output, refresh_decision_template=False)
            for name in ("decisions.csv", "decisions_blank.csv"):
                self.assertEqual((review / name).read_text(encoding="utf-8"), "human record,do not replace\n")
            page = (review / "index.html").read_text(encoding="utf-8")
            embedded = re.search(r'<script id="review-data" type="application/json">(.*?)</script>',
                                 page, re.S).group(1)
            self.assertNotIn("<", embedded)
            self.assertEqual(json.loads(embedded)["bundles"][0]["tasks"][0]["proposed_query"], task["proposed_query"])

    def test_modality_approval_needs_written_evidence(self) -> None:
        row = {"task_id": "toy:ir", "category": "ir_complement"}
        decision = {"decision": "approve", "reviewer": "fixture", "object_binding": "yes",
                    "full_semantics": "yes", "query_unique": "yes", "rgb_only_unique": "no",
                    "ir_unique": "yes", "aux_necessary": "yes", "note": ""}
        with self.assertRaisesRegex(ValueError, "written visual evidence"):
            _approved(row, decision)

    def test_quick_review_retains_without_name_or_fabricated_modality_evidence(self) -> None:
        candidates = [{"task_id": "toy:ir", "bundle_id": "toy", "category": "ir_complement",
                       "proposed_query": "A person at night."}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decisions.csv"
            with path.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=DECISION_FIELDS)
                writer.writeheader()
                writer.writerow({"task_id": "toy:ir", "decision": "approve", "review_mode": "quick"})
            accepted = accepted_candidates(candidates, path)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["review_status"], "human_quick_approved")
        self.assertEqual(accepted[0]["reviewer"], "")
        self.assertEqual(accepted[0]["modality_judgment"], "")
        self.assertEqual(accepted[0]["approval_evidence"]["rgb_only_unique"], "")
        self.assertEqual(accepted[0]["approval_evidence"]["aux_necessary"], "")
        for decision in ("", "reject"):
            self.assertFalse(_approved(candidates[0], {"decision": decision, "review_mode": "quick"}))

    def test_missing_prompt_preserves_normal_city_and_two_image_rgbt(self) -> None:
        query = "The nearer car."
        self.assertEqual(native_prompt(query, ("rgb", "infrared", "depth")),
                         _trimodal_prompt(query))
        ir_missing = native_prompt(query, ("rgb", "infrared"), "visual", ("infrared",))
        self.assertEqual(ir_missing.count("<image>"), 2)
        self.assertIn("all-black placeholders", ir_missing)
        self.assertNotIn("depth", ir_missing.casefold())
        with self.assertRaises(ValueError):
            native_prompt(query, ("rgb", "infrared"), "visual", ("rgb",))

    def test_diagnostic_gt_is_read_by_real_evaluator_and_report(self) -> None:
        row = {"task_id": "toy:depth", "scene_id": "toy:scene", "source_id": "toy:source",
               "category": "diag_depth", "target_object_id": "toy:object",
               "bbox": [0.1234567, 0.25, 0.75, 0.9],
               "images": {"rgb": "visible.png", "infrared": "infrared.png", "depth": "depth.png"},
               "proposed_query": "The nearest person."}
        sample_id = f"abv:diag:{row['task_id']}"
        gt = {sample_id: diagnostic_gt_record({**row, "query": row["proposed_query"]})}
        # The full evaluator imports GPU libraries unavailable in this CPU audit.
        # Execute its actual four target-manifest functions with a tiny CPU torch shim.
        source = Path(__file__).resolve().parents[1] / "tools/evaluate_pretrained_grounder.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        names = {"_validated_target", "_read_jsonl", "load_target_manifest", "apply_target_manifest"}
        definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                                  *definitions], type_ignores=[])
        module = ast.fix_missing_locations(module)
        namespace = {"json": json, "Path": Path, "Any": Any,
                     "torch": SimpleNamespace(tensor=np.asarray, isfinite=np.isfinite)}
        exec(compile(module, str(source), "exec"), namespace)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gt.json"
            path.write_text(json.dumps(gt), encoding="utf-8")
            report_manifest = load_manifest(path)
            self.assertEqual(report_manifest[sample_id]["bbox"], row["bbox"])
            evaluated = namespace["apply_target_manifest"]([{"id": sample_id, "bbox": [0, 0, 1, 1]}], path)
            self.assertEqual(evaluated[0]["bbox"], row["bbox"])
            self.assertEqual(evaluated[0]["target_source"], str(path.resolve()))

    def test_seed_2026_is_accepted_by_real_runner_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            manifest = folder / "tiny_mock.json"
            manifest.write_text(json.dumps([{"id": "mock"}] * 3200), encoding="utf-8")
            release_stub = {"status": "ready", "steps": 400, "seed": 2026,
                            "manifests": {arm: str(manifest) for arm in ("A", "B", "V")},
                            "diagnostics": {condition: str(manifest) for condition in
                                            ("normal", "ir_missing", "depth_missing", "both_missing")}}
            validate_release(release_stub)
            release_stub["seed"] = 20260927
            with self.assertRaisesRegex(ValueError, "planned seeds"):
                validate_release(release_stub)


if __name__ == "__main__":
    unittest.main()
