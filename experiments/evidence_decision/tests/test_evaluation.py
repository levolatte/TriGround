"""Checks for split leakage, fixed denominators, legal final boxes and replay inputs."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "code"))
from experiments.evidence_decision import evaluate as ev
from experiments.evidence_decision import prepare as prep
from experiments.evidence_decision import replay
from experiments.evidence_decision import bootstrap
from experiments.evidence_decision.vision_tools import CandidatePool


def write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


class PreparationTests(unittest.TestCase):
    def test_real_snapshot_exclusions_and_group_split(self):
        source = ROOT / "results/gu_diagnosis_20260927/source_snapshot"
        train = prep.read_json(source / "city_train.json")
        val = prep.read_json(source / "city_val.json")
        eligible, exclusion = prep.eligible_train(train)
        self.assertEqual((len(train), len(val)), (3707, 412))
        self.assertEqual((exclusion["shared_sequence_queries"], exclusion["shared_sequence_groups"]), (96, 11))
        self.assertEqual((exclusion["main_diagnostic_queries"], exclusion["main_diagnostic_groups"]), (8, 3))
        self.assertEqual((len(eligible), exclusion["kept_groups"]), (3603, 667))
        dev = prep.select_dev_groups(val)
        self.assertEqual(len({prep.image_group(r) for r in dev}), 16)
        self.assertEqual({r["id"] for r in dev},
                         {r["id"] for r in val if prep.image_group(r) in
                          {prep.image_group(d) for d in dev}})
        smoke = prep.select_distinct_starts(eligible, 32, 2026)
        t = prep.select_distinct_starts(
            [r for r in eligible if prep.image_group(r) not in {prep.image_group(s) for s in smoke}],
            250, 2027)
        self.assertEqual(len({prep.image_group(r) for r in smoke + t}), 282)
        debug = prep.select_debug32(eligible)
        self.assertEqual(len({prep.image_group(r) for r, _ in debug}), 32)
        self.assertEqual({intent: sum(label == intent for _, label in debug) for intent in
                          ("keep_part_group_to_check", "same_class_competition_to_check",
                           "spatial_relation_to_check", "rgb_ir_quality_to_check",
                           "depth_validity_to_check")},
                         {"keep_part_group_to_check": 4, "same_class_competition_to_check": 8,
                          "spatial_relation_to_check": 8, "rgb_ir_quality_to_check": 6,
                          "depth_validity_to_check": 6})
        row = prep.inference_row(val[0], "/data/city/train")
        self.assertEqual(set(row), {"id", "query", "images", "depth_encoding"})
        self.assertNotIn("bbox", json.dumps(row))


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.gt = {
            "a": {"visible": "x/one.png", "bbox": [.1, .1, .3, .3], "query": "first"},
            "b": {"visible": "x/one.png", "bbox": [.5, .5, .7, .7], "query": "second"},
            "c": {"visible": "x/two.png", "bbox": [.2, .2, .4, .4], "query": "third"},
        }
        self.gt_path = self.root / "gt.json"
        self.gt_path.write_text(json.dumps(self.gt), encoding="utf-8")
        self.baseline = [
            {"id": "a", "bbox": [.1, .1, .3, .3]},
            {"id": "b", "bbox": [.1, .1, .3, .3]},
            {"id": "c", "bbox": [.2, .2, .4, .4]},
        ]
        self.base_path = self.root / "base.jsonl"
        write_jsonl(self.base_path, self.baseline)

    def test_null_failure_and_candidate_coverage_keep_full_denominator(self):
        candidates = [{"id": 1, "role": "target", "bbox": [.5, .5, .7, .7]}]
        rows = [
            {"id": "a", "final_status": "FINISHED", "selected_id": "KEEP", "bbox": [.1, .1, .3, .3],
             "initial_bbox": [.1, .1, .3, .3], "initial_candidates": [], "final_candidates": [],
             "metrics": {"tool_calls": 1, "tool_status_counts": {"UNKNOWN": 1}}},
            {"id": "b", "final_status": "FINISHED", "selected_id": 1, "bbox": [.5, .5, .7, .7],
             "initial_bbox": [.1, .1, .3, .3], "initial_candidates": [], "final_candidates": candidates,
             "metrics": {"tool_calls": 2, "search_calls": 1}},
            {"id": "c", "final_status": "INVALID_FINAL", "selected_id": None, "bbox": None,
             "initial_bbox": [.2, .2, .4, .4], "initial_candidates": [], "final_candidates": [],
             "metrics": {"invalid_actions": 1}},
        ]
        path = self.root / "run.jsonl"
        write_jsonl(path, rows)
        loaded = ev.load_run(path, set(self.gt))
        score = ev.score_subset(loaded, self.gt, list(self.gt))
        self.assertEqual((score["n"], score["hits_0.5"], score["initial_coverage_hits_0.5"],
                          score["final_coverage_hits_0.5"]), (3, 2, 2, 3))
        self.assertEqual(score["covered_but_final_wrong"], 1)
        self.assertEqual(score["tool_status_counts"], {"UNKNOWN": 1})
        pair = ev.compare(ev.load_run(self.base_path, set(self.gt)), loaded, self.gt, list(self.gt), 200)
        self.assertEqual((pair["corrected"], pair["damaged"], pair["net_hits_0.5"]), (1, 1, 0))
        self.assertEqual(pair["image_groups"], 2)

    def test_keep_must_equal_initial_and_missing_ids_fail(self):
        wrong = [{"id": "a", "final_status": "FINISHED", "selected_id": "KEEP",
                  "bbox": [.11, .1, .3, .3], "initial_bbox": [.1, .1, .3, .3]}]
        path = self.root / "wrong.jsonl"
        write_jsonl(path, wrong)
        with self.assertRaisesRegex(ValueError, "KEEP changed"):
            ev.load_run(path, {"a"})
        with self.assertRaisesRegex(ValueError, "incomplete denominator"):
            ev.load_run(self.base_path, {"a", "b", "c", "d"})

    def test_predicted_bbox_finish_has_no_fake_id_and_is_scored_separately(self):
        rows = [
            {"id": "a", "final_status": "FINISHED", "finish_source": "predicted_bbox",
             "selected_id": None, "selected_public_id": None, "bbox": [.1, .1, .3, .3],
             "initial_bbox": [.5, .5, .7, .7], "initial_candidates": [], "final_candidates": [],
             "metrics": {"elapsed_seconds": 1.0}},
            {"id": "b", "final_status": "FINISHED", "finish_source": "candidate",
             "selected_id": "P2", "selected_public_id": "P2", "bbox": [.5, .5, .7, .7],
             "initial_bbox": [.5, .5, .7, .7], "initial_candidates": [
                 {"id": "P2", "role": "target", "finish_eligible": True, "coordinate_frame": "rgb",
                  "bbox": [.5, .5, .7, .7]}],
             "final_candidates": [
                 {"id": "P2", "role": "target", "finish_eligible": True, "coordinate_frame": "rgb",
                  "bbox": [.5, .5, .7, .7]}]},
            {"id": "c", "final_status": "INPUT_BUDGET_EXHAUSTED", "finish_source": None,
             "selected_id": None, "selected_public_id": None, "bbox": None,
             "initial_bbox": [.7, .7, .9, .9], "initial_candidates": [], "final_candidates": []},
        ]
        path = self.root / "bbox_finish.jsonl"
        write_jsonl(path, rows)
        loaded = ev.load_run(path, set(self.gt))
        score = ev.score_subset(loaded, self.gt, list(self.gt))
        self.assertEqual((score["n"], score["valid_final_bbox"], score["hits_0.5"]), (3, 2, 2))
        self.assertEqual(score["finish_source_counts"],
                         {"predicted_bbox": 1, "candidate": 1, "no_legal_final": 1})
        self.assertEqual(score["search_and_bbox_coverage"]["predicted_bbox_unique_rescues_without_final_candidate_coverage"], 1)
        self.assertEqual(score["initial_pool_strata"]["uncovered"]["n"], 2)

    def test_predicted_bbox_rejects_any_public_id_and_candidate_mode_requires_eligible_id(self):
        predicted_with_id = [{"id": "a", "final_status": "FINISHED", "finish_source": "predicted_bbox",
                              "selected_id": "P1", "selected_public_id": None,
                              "bbox": [.1, .1, .3, .3], "initial_bbox": [.1, .1, .3, .3]}]
        path = self.root / "bbox_fake_id.jsonl"
        write_jsonl(path, predicted_with_id)
        with self.assertRaisesRegex(ValueError, "predicted_bbox must not have a selected ID"):
            ev.load_run(path, {"a"})

        bad_candidate = [{"id": "a", "final_status": "FINISHED", "finish_source": "candidate",
                          "selected_id": "P1", "selected_public_id": "P1",
                          "bbox": [.1, .1, .3, .3], "initial_bbox": [.1, .1, .3, .3],
                          "final_candidates": [{"id": "P1", "role": "reference", "finish_eligible": False,
                                                "bbox": [.1, .1, .3, .3]}]}]
        write_jsonl(path, bad_candidate)
        with self.assertRaisesRegex(ValueError, "no eligible RGB box"):
            ev.load_run(path, {"a"})


class ReplayTests(unittest.TestCase):
    def test_union_preserves_numeric_baseline_alias_and_new_ids(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = root / "manifest.jsonl"
            fixed = root / "fixed.jsonl"
            dynamic = root / "dynamic.jsonl"
            out = root / "union.jsonl"
            write_jsonl(manifest, [{"id": "a"}])
            write_jsonl(fixed, [{"id": "a", "c_bbox": [.1, .1, .3, .3],
                                 "query_info": {"scope": "single"},
                                 "candidates": [
                                     {"id": 2, "role": "target", "bbox": [.1, .1, .3, .3],
                                      "is_baseline": True, "sources": [], "depth": {"status": "pending"}, "mask_path": None},
                                     {"id": 4, "role": "target", "bbox": [.5, .5, .7, .7],
                                      "is_baseline": False, "sources": [], "depth": {"status": "pending"}, "mask_path": None}]}])
            write_jsonl(dynamic, [{"id": "a", "final_candidates": [
                {"id": "KEEP", "role": "target", "bbox": [.1, .1, .3, .3], "sources": []},
                {"id": "4", "role": "target", "bbox": [.5, .5, .7, .7], "sources": []},
                {"id": "5", "role": "target", "bbox": [.7, .7, .9, .9], "sources": []}]}])
            args = type("Args", (), {"manifest": manifest, "fixed_cache": fixed,
                                     "dynamic_predictions": dynamic, "output_cache": out})()
            result = replay.union(args)
            self.assertEqual(result["new_candidate_entries"], 1)
            pool = CandidatePool(replay.read_jsonl(out)[0], (100, 100))
            self.assertEqual([c["id"] for c in pool.public_candidates()], ["KEEP", "4", "5"])

    def test_permutation_keeps_rgb_and_swaps_matching_depth_pair(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rows = [{"id": str(i), "query": "q", "images": {
                "rgb": f"r{i}", "ir": f"i{i}", "depth_raw": f"d{i}", "depth_visual": f"v{i}"},
                "depth_encoding": "city_mm"} for i in range(3)]
            manifest = root / "manifest.jsonl"
            write_jsonl(manifest, rows)
            args = type("Args", (), {"manifest": manifest, "candidate_cache": None,
                                     "output_dir": root / "out", "count": 3, "seed": 2026})()
            replay.permute(args)
            correct = replay.read_jsonl(args.output_dir / "correct.jsonl")
            ir = replay.read_jsonl(args.output_dir / "ir_swapped.jsonl")
            depth = replay.read_jsonl(args.output_dir / "depth_swapped.jsonl")
            for a, b, c in zip(correct, ir, depth):
                self.assertEqual(a["id"], b["id"])
                self.assertEqual(a["images"]["rgb"], b["images"]["rgb"])
                self.assertNotEqual(a["images"]["ir"], b["images"]["ir"])
                self.assertNotEqual(a["images"]["depth_raw"], c["images"]["depth_raw"])
                donor = c["images"]["depth_raw"][1:]
                self.assertEqual(c["images"]["depth_visual"], f"v{donor}")


class BootstrapTests(unittest.TestCase):
    def test_original_prompt_and_cpu_candidate_resume(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = root / "manifest.jsonl"
            baseline = root / "baseline.jsonl"
            query_info = root / "query_info.jsonl"
            prompt_path = root / "prompts.json"
            output = root / "candidates.jsonl"
            row = {"id": "sample", "query": "the red wheel", "images": {
                "rgb": "rgb.png", "ir": "ir.png", "depth_visual": "visual.png", "depth_raw": "raw.png"},
                "depth_encoding": "city_mm"}
            write_jsonl(manifest, [row])
            write_jsonl(baseline, [{"id": "sample", "prediction": [.1, .2, .3, .4]}])
            write_jsonl(query_info, [{"id": "sample", "target_category": "wheel",
                "reference_categories": [], "relation_type": "none", "scope": "part", "parsed": True}])
            p_args = type("Args", (), {"manifest": manifest, "output": prompt_path,
                "model": None, "adapter": None, "baseline_output": None, "query_output_dir": None})()
            self.assertEqual(bootstrap.prompts(p_args)["queries"], 1)
            from tools.prepare_qwen3vl_native_sft import _trimodal_prompt
            self.assertEqual(json.loads(prompt_path.read_text())["sample"], _trimodal_prompt(row["query"]))
            c_args = type("Args", (), {"manifest": manifest, "baseline": baseline,
                "query_info": query_info, "data_root": root, "model": "unused-dino",
                "output": output, "threads": 4, "resume": False, "max_run_seconds": None})()
            self.assertTrue(bootstrap.candidates(c_args)["complete"])
            candidate = bootstrap.read_jsonl(output)[0]
            self.assertEqual(candidate["c_bbox"], [.1, .2, .3, .4])
            self.assertEqual(len(candidate["candidates"]), 1)
            c_args.resume = True
            self.assertEqual(bootstrap.candidates(c_args)["completed"], 1)
            c_args.threads = 8
            with self.assertRaisesRegex(ValueError, "resume config differs"):
                bootstrap.candidates(c_args)

    def test_training_gt_is_separate_and_keeps_raw_float_bbox(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = root / "train.jsonl"
            raw = root / "qwen_generation_train_100.json"
            output = root / "scoring" / "train_gt.json"
            row = {"id": "sample", "query": "target", "images": {
                "rgb": "rgb.png", "ir": "ir.png", "depth_visual": "v.png", "depth_raw": "d.png"},
                "depth_encoding": "city_mm"}
            write_jsonl(manifest, [row])
            raw.write_text(json.dumps({"sample": {"query": "target", "bbox": [.123456, .2, .3, .4]}}))
            args = type("Args", (), {"manifest": manifest, "train_source": raw, "output": output})()
            bootstrap.training_gt(args)
            self.assertEqual(json.loads(output.read_text())["sample"]["bbox"][0], .123456)
            args.output = root / "train_gt.json"
            with self.assertRaisesRegex(ValueError, "scoring directory"):
                bootstrap.training_gt(args)


if __name__ == "__main__":
    unittest.main()
