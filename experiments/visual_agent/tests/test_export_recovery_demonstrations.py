"""Offline recovery export keeps actual observations and excludes bad probes."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from experiments.visual_agent.export_recovery_demonstrations import export


KEEP = [0.05, 0.05, 0.2, 0.2]
TARGET = [0.5, 0.5, 0.7, 0.7]


def candidate(candidate_id, box):
    return {"id": str(candidate_id), "role": "target", "bbox": box}


def event(step, action, status, before, after=None, *, images=None, data=None):
    return {"step": step, "action": action, "executed_action": action,
            "candidates_before": before, "candidates_after": after if after is not None else before,
            "observation": {"status": status, "text": f"real {status}",
                            "images": images or [], "data": data or {}}, "tool_seconds": 0.1}


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


class ExportRecoveryDemonstrationsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.train = self.root / "t_train200.jsonl"
        self.holdout = self.root / "t_holdout50.jsonl"
        self.debug = self.root / "train32.jsonl"
        self.gt = self.root / "gt.json"
        self.base = self.root / "base.jsonl"
        self.candidate_branches = self.root / "candidate_branches"
        self.protocol_feedback = self.root / "protocol_feedback"
        self.search_branches = self.root / "search_branches"
        for directory in (self.candidate_branches, self.protocol_feedback, self.search_branches):
            directory.mkdir()
        self.initial = [{"role": "user", "content": [{"type": "text", "text": "real initial state"}]}]
        self.a_pool = [candidate("KEEP", KEEP), candidate("7", TARGET),
                       candidate("8", [0.75, 0.5, 0.9, 0.7])]
        self.c_initial = [candidate("KEEP", KEEP)]
        self.c_found = [*self.c_initial, candidate("9", TARGET)]
        rows = [self.manifest(sample_id) for sample_id in ("A", "B", "C")]
        write_jsonl(self.train, rows)
        write_jsonl(self.holdout, [self.manifest("H")])
        write_jsonl(self.debug, [self.manifest("Z")])
        self.gt.write_text(json.dumps({"A": {"bbox": TARGET}, "B": {"bbox": KEEP},
                                       "C": {"bbox": TARGET}}), encoding="utf-8")
        base_rows = [
            self.base_row("A", {"action": "inspect_regions", "candidate_ids": ["KEEP", "7"],
                                "modalities": ["rgb", "ir"], "view": "cross",
                                "evidence_note": "template explanation"}),
            self.base_row("A", {"action": "finish", "candidate_id": "7"}, final=True),
            self.base_row("B", {"action": "finish", "candidate_id": "KEEP"}, final=True),
        ]
        write_jsonl(self.base, base_rows)
        inspect = event(0, {"action": "inspect_regions", "candidate_ids": ["KEEP", "7"],
                            "modalities": ["rgb", "ir"], "view": "cross",
                            "evidence_note": "real comparison"}, "OK", self.a_pool,
                        images=[{"path": "/real/a_ir.png", "modality": "ir", "candidate_ids": ["7"]}])
        self.branch = {"id": "A", "candidate_id": "7", "prefix_events": [], "event": inspect,
                       "candidates": self.a_pool, "parent_trace_path": str(self.root / "parent_trace.json")}
        (self.root / "parent_trace.json").write_text(json.dumps({"initial_messages": self.initial}),
                                                      encoding="utf-8")
        write_jsonl(self.candidate_branches / "traces/A/branches.jsonl", [self.branch])
        invalid = event(0, {"action": "inspect_regions", "candidate_ids": ["KEEP", "7", "8"],
                            "modalities": ["rgb", "ir"], "view": "cross"}, "ERROR", self.a_pool,
                        data={"kind": "protocol"})
        self.protocol = {"id": "A", "status": "ERROR", "intentional_invalid_action": True,
                         "candidate_ids": ["KEEP", "7", "8"], "pool_before": self.a_pool,
                         "pool_after": self.a_pool, "event": invalid, "prefix_events": [invalid]}
        write_jsonl(self.protocol_feedback / "traces/A/protocol_feedback.jsonl", [self.protocol])
        search = event(0, {"action": "search_candidates", "category": "car", "region": "right",
                           "modality": "rgb", "role": "target", "evidence_note": "probe"},
                       "OK", self.c_initial, self.c_found, data={"appended_ids": ["9"]})
        search["messages"] = self.initial
        inspect_c = event(1, {"action": "inspect_regions", "candidate_ids": ["KEEP", "9"],
                              "modalities": ["rgb", "ir"], "view": "cross"}, "OK", self.c_found,
                          images=[{"path": "/real/c_rgb.png", "modality": "rgb", "candidate_ids": ["9"]}])
        inspect_c["messages"] = self.initial
        self.search_branch = {
            "id": "C", "candidate_id": "9", "trajectory_origin": "scripted_search_branch",
            "branch_key": "rgb/right", "manifest_row": self.manifest("C"),
            "initial_messages": self.initial, "search_event": search, "inspect_event": inspect_c,
            "terminal_messages": self.initial, "candidates": self.c_found,
        }
        write_jsonl(self.search_branches / "traces/C/branches.jsonl", [self.search_branch])
        empty = event(0, {"action": "search_candidates", "category": "car", "region": "left",
                          "modality": "ir", "role": "target"}, "EMPTY", self.c_initial)
        write_jsonl(self.search_branches / "traces/C/search_events.jsonl", [empty])

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def manifest(sample_id):
        return {"id": sample_id, "query": f"query {sample_id}",
                "images": {"rgb": f"{sample_id}.png"}}

    @staticmethod
    def base_row(sample_id, action, *, final=False):
        return {"id": sample_id, "image_group": f"{sample_id}.png",
                "target_action": json.dumps(action, separators=(",", ":")),
                "original_action": action, "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "real base decision"}]}], "is_final": final,
                "origin": "scripted", "label_source": "base"}

    def args(self, name="out"):
        return SimpleNamespace(base=self.base, candidate_branches=self.candidate_branches,
                               protocol_feedback=self.protocol_feedback,
                               search_branches=self.search_branches, train_manifest=self.train,
                               holdout_manifest=self.holdout, debug_manifest=self.debug,
                               gt=self.gt, output_dir=self.root / name)

    def test_exports_only_real_recovery_actions_and_minimal_labels(self):
        inventory = export(self.args())
        rows = [json.loads(line) for line in (self.root / "out/train.jsonl").read_text().splitlines()]
        self.assertEqual(inventory["base_unique_starts"], 2)
        self.assertEqual(inventory["new_search_unique_starts"], 1)
        self.assertEqual(inventory["protocol_replay_starts"], 1)
        self.assertEqual(inventory["empty_replays"], 1)
        self.assertEqual((inventory["base_rows"], inventory["total_rows"]), (3, 11))
        self.assertEqual(inventory["unique_keep_fraction"], 1 / 3)
        self.assertEqual(inventory["label_format"], "minimal_action_json_without_optional_evidence_note")
        self.assertEqual(inventory["selected_search_branches_for_semantic_review"][0]["category"], "car")
        self.assertTrue(all("evidence_note" not in row["target_action"] for row in rows))
        self.assertIn("evidence_note", rows[0]["original_action"])
        self.assertEqual(sum(row["label_source"] == "offline_protocol_replay_actual_inspect" for row in rows), 1)
        self.assertEqual(sum(row["label_source"] == "offline_selected_actual_search" for row in rows), 2)
        self.assertFalse(any('"candidate_ids":["KEEP","7","8"]' in row["target_action"] for row in rows))
        replay = next(row for row in rows if (row.get("replay") or {}).get("kind") == "empty_then_successful_search")
        self.assertIn("EMPTY", json.dumps(replay["messages"]))

    def test_rejects_protocol_pool_mutation_without_fallback(self):
        bad = deepcopy(self.protocol)
        bad["pool_after"] = self.c_initial
        write_jsonl(self.protocol_feedback / "traces/A/protocol_feedback.jsonl", [bad])
        inventory = export(self.args("bad_out"))
        self.assertEqual(inventory["protocol_replay_starts"], 0)
        self.assertTrue(any(row["kind"] == "protocol" for row in inventory["rejected"]))

    def test_protocol_recovery_may_select_another_current_target(self):
        pool = [*self.a_pool, candidate("9", [0.25, 0.5, 0.4, 0.7])]
        branch = deepcopy(self.branch)
        branch["candidates"] = pool
        branch["event"]["candidates_before"] = pool
        branch["event"]["candidates_after"] = pool
        write_jsonl(self.candidate_branches / "traces/A/branches.jsonl", [branch])
        probe = deepcopy(self.protocol)
        probe["candidate_ids"] = ["KEEP", "8", "9"]
        probe["pool_before"] = pool
        probe["pool_after"] = pool
        probe["event"]["action"]["candidate_ids"] = probe["candidate_ids"]
        probe["event"]["executed_action"]["candidate_ids"] = probe["candidate_ids"]
        probe["event"]["candidates_before"] = pool
        probe["event"]["candidates_after"] = pool
        probe["prefix_events"] = [probe["event"]]
        write_jsonl(self.protocol_feedback / "traces/A/protocol_feedback.jsonl", [probe])
        inventory = export(self.args("different_target"))
        self.assertEqual(inventory["protocol_replay_starts"], 1)
        rows = [json.loads(line) for line in (self.root / "different_target/train.jsonl").read_text().splitlines()]
        self.assertTrue(any(row["label_source"] == "offline_gt_existing_target_protocol_replay" and
                            json.loads(row["target_action"])["candidate_id"] == "7" for row in rows))

    def test_rejects_holdout_branch_at_input_boundary(self):
        held = deepcopy(self.search_branch)
        held["id"] = "H"
        write_jsonl(self.search_branches / "traces/H/branches.jsonl", [held])
        with self.assertRaisesRegex(ValueError, "nontraining branch"):
            export(self.args("held_out"))


if __name__ == "__main__":
    unittest.main()
