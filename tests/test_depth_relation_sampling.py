"""CPU checks for mixed single and required-pair Depth tasks."""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from tools.prepare_triground_abv_data import (
    DECISION_FIELDS,
    _schedule,
    _select_release,
    accepted_candidates,
)


def task(category: str, number: int, *, bundle: str | None = None,
         scene: str | None = None, source: str = "fixture", version: str | None = None,
         pair_required: bool = False, family: str = "nearer") -> dict:
    task_id = f"{category}:{number}"
    row = {"task_id": task_id, "bundle_id": bundle or task_id,
           "scene_id": scene or f"scene:{number}", "category": category,
           "split": "diagnostic" if category.startswith("diag_") else "train",
           "source": source, "proposed_query": f"Find object {number}.",
           "query": f"Find object {number}.", "relation_type": family,
           "depth_policy": "millimeter", "bbox": [.1, .1, .7, .7],
           "images": {}, "review_status": "human_quick_approved"}
    if version:
        row.update(construction_version=version, pair_required=pair_required)
    return row


def training_base() -> list[dict]:
    rows = []
    for index in range(28):
        rows.extend(task("robo_competition", index*2 + side,
                         bundle=f"robo:{index}", source="roborefit") for side in range(2))
    rows.extend(task("ir_complement", index) for index in range(60))
    rows.extend(task("reliability", index, source="city") for index in range(30))
    rows.extend(task("reliability", index+30, source="rgbt_fixture") for index in range(15))
    rows.extend(task("reliability", index+45, source="roborefit") for index in range(15))
    return rows


class DepthRelationSamplingTest(unittest.TestCase):
    def test_approval_only_requires_explicit_v2_and_legacy_pairs(self) -> None:
        candidates = [task("depth_relation", 0, version="depth_relations_v2"),
                      task("depth_relation", 1, bundle="new-pair", version="depth_relations_v2",
                           pair_required=True),
                      task("depth_relation", 2, bundle="new-pair", version="depth_relations_v2",
                           pair_required=True),
                      task("depth_relation", 3, bundle="old-pair"),
                      task("depth_relation", 4, bundle="old-pair"),
                      task("robo_competition", 5, bundle="robo-pair"),
                      task("robo_competition", 6, bundle="robo-pair"),
                      task("diag_depth", 7, bundle="diag-pair"),
                      task("diag_depth", 8, bundle="diag-pair"),
                      task("diag_depth", 9, version="depth_relations_v2")]
        approved = {0, 1, 3, 5, 7, 9}
        with tempfile.TemporaryDirectory() as directory:
            decisions = Path(directory) / "decisions.csv"
            with decisions.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=DECISION_FIELDS)
                writer.writeheader()
                for index, row in enumerate(candidates):
                    writer.writerow({"task_id": row["task_id"],
                                     "decision": "approve" if index in approved else "reject",
                                     "review_mode": "quick"})
            accepted = accepted_candidates(candidates, decisions)
        self.assertEqual({row["task_id"] for row in accepted},
                         {candidates[0]["task_id"], candidates[9]["task_id"]})

    def test_release_uses_64_depth_tasks_and_never_cuts_diagnostic_pairs(self) -> None:
        depth = [task("depth_relation", index, scene=f"depth-scene:{index % 10}",
                      version="depth_relations_v2", family="nearer" if index < 40 else "middle")
                 for index in range(80)]
        diagnostic = [task("diag_depth", index, bundle=f"diag:{index//2}") for index in range(34)]
        selected, diag, _ = _select_release(training_base() + depth + diagnostic, 600)
        chosen = [row for row in selected if row["category"] == "depth_relation"]
        self.assertEqual(len(chosen), 64)
        scenes = Counter(row["scene_id"] for row in chosen)
        self.assertLessEqual(max(scenes.values()) - min(scenes.values()), 1)
        self.assertEqual(Counter(row["relation_type"] for row in chosen)["middle"], 32)
        self.assertEqual(len(diag), 32)
        diag_bundles = Counter(row["bundle_id"] for row in diag)
        self.assertEqual(set(diag_bundles.values()), {2})

        # A pair at the quota boundary is either fully included or left out.
        mixed = [task("depth_relation", index, version="depth_relations_v2",
                      scene=f"mixed:{index}") for index in range(62)]
        for pair_index in range(2):
            mixed.extend(task("depth_relation", 62 + pair_index*2 + side,
                              bundle=f"mixed-pair:{pair_index}", scene=f"pair-scene:{pair_index}",
                              version="depth_relations_v2", pair_required=True,
                              family="farther") for side in range(2))
        selected, _, _ = _select_release(training_base() + mixed, 600)
        chosen_ids = {row["task_id"] for row in selected if row["category"] == "depth_relation"}
        self.assertEqual(len(chosen_ids), 64)
        for pair_index in range(2):
            self.assertIn(len(chosen_ids & {f"depth_relation:{62 + pair_index*2 + side}"
                                                for side in range(2)}), {0, 2})

    def test_mixed_schedule_keeps_pair_counts_and_full_arm_budgets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.png"
            Image.new("RGB", (8, 8)).save(image)
            city = root / "city.json"
            city.write_text(json.dumps([
                {"id": f"old:{scene}:{query}", "image": [f"old_{scene}.png"]}
                for scene in range(500) for query in range(6)]), encoding="utf-8")
            output = root / "output"
            (output / "data").mkdir(parents=True)
            (output / "data/candidates.jsonl").write_text("", encoding="utf-8")
            (root / "assets").mkdir()
            depth = [task("depth_relation", index, version="depth_relations_v2") for index in range(48)]
            for pair_index in range(8):
                depth.extend(task("depth_relation", 48 + pair_index*2 + side,
                                  bundle=f"depth-pair:{pair_index}",
                                  version="depth_relations_v2", pair_required=True)
                             for side in range(2))
            selected, _, _ = _select_release(training_base() + depth, 600)
            for row in selected:
                row["images"] = {"rgb": str(image), "infrared": str(image), "depth": str(image)}
            with patch("tools.prepare_triground_abv_data.CITY", city):
                expected_by_steps = {
                    400: {"depth_relation": 340, "robo_competition": 298,
                          "ir_complement": 320, "reliability": 320},
                    600: {"depth_relation": 512, "robo_competition": 448,
                          "ir_complement": 480, "reliability": 480},
                }
                for steps in (400, 600):
                    expected = expected_by_steps[steps]
                    new_target = sum(expected.values())
                    old_target = steps*8-new_target
                    arms = _schedule(selected, steps, root / "assets", output)
                    self.assertEqual({arm: len(rows) for arm, rows in arms.items()},
                                     {"A": steps*8, "B": steps*8, "V": steps*8})
                    self.assertEqual(arms["B"], arms["V"])
                    self.assertEqual(sum(row["task_category"] == "old_city" for row in arms["B"]),
                                     old_target)
                    self.assertEqual(sum(row["task_category"] != "old_city" for row in arms["B"]),
                                     new_target)
                    self.assertEqual(Counter(row["task_category"] for row in arms["B"]
                                             if row["task_category"] != "old_city"),
                                     Counter(expected))
                    for position, (a, b) in enumerate(zip(arms["A"], arms["B"], strict=True)):
                        if position % 80 < 48:
                            self.assertEqual(a, b)
                    counts = Counter(row["source_task_id"] for row in arms["B"]
                                     if row["task_category"] != "old_city")
                    self.assertLessEqual(max(counts.values()), 8)
                    old_counts = Counter(row["origin_query_id"] for row in arms["A"])
                    self.assertLessEqual(max(old_counts.values()), 2)
                    for pair_index in range(8):
                        self.assertEqual(counts[f"depth_relation:{48 + pair_index*2}"],
                                         counts[f"depth_relation:{49 + pair_index*2}"])
                    for pair_index in range(28):
                        self.assertEqual(counts[f"robo_competition:{pair_index*2}"],
                                         counts[f"robo_competition:{pair_index*2+1}"])
                sparse = ([row for row in selected if row["category"] == "depth_relation"
                           and not row["pair_required"]]
                          + [row for row in selected if row["category"] == "robo_competition"][:42]
                          + [row for row in selected if row["category"] == "ir_complement"][:40]
                          + [row for row in selected if row["category"] == "reliability"][:45])
                sparse_arms = _schedule(sparse, 600, root / "assets", output)
                self.assertEqual(len(sparse_arms["B"]), 4800)
                self.assertEqual(sparse_arms["B"], sparse_arms["V"])
                self.assertEqual(sum(row["task_category"] != "old_city" for row in sparse_arms["B"]),
                                 1400)
                self.assertEqual(sum(row["task_category"] == "old_city" for row in sparse_arms["B"]),
                                 3400)

    def test_realistic_pool_uses_independent_scaled_quotas(self) -> None:
        depth = [task("depth_relation", index, version="depth_relations_v2",
                      pair_required=False) for index in range(48)]
        for pair_index in range(4):
            depth.extend(task("depth_relation", 48 + pair_index*2 + side,
                              bundle=f"depth-pair:{pair_index}",
                              version="depth_relations_v2", pair_required=True)
                         for side in range(2))
        selected, _, _ = _select_release(training_base() + depth, 600)
        self.assertEqual(Counter(row["category"] for row in selected), Counter({
            "depth_relation": 56, "robo_competition": 56,
            "ir_complement": 60, "reliability": 60,
        }))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.png"
            Image.new("RGB", (8, 8)).save(image)
            city = root / "city.json"
            city.write_text(json.dumps([
                {"id": f"old:{scene}:{query}", "image": [f"old_{scene}.png"]}
                for scene in range(500) for query in range(6)]), encoding="utf-8")
            output = root / "output"
            (output / "data").mkdir(parents=True)
            (output / "data/candidates.jsonl").write_text("", encoding="utf-8")
            assets = root / "assets"
            assets.mkdir()
            for row in selected:
                row["images"] = {"rgb": str(image), "infrared": str(image), "depth": str(image)}

            expected_by_steps = {
                400: {"old_city": 1964, "depth_relation": 298,
                      "robo_competition": 298, "ir_complement": 320, "reliability": 320},
                600: {"old_city": 2944, "depth_relation": 448,
                      "robo_competition": 448, "ir_complement": 480, "reliability": 480},
            }
            pair_rows: dict[str, list[str]] = defaultdict(list)
            for row in selected:
                if row["category"] == "robo_competition" or row.get("pair_required") is True:
                    pair_rows[row["bundle_id"]].append(row["task_id"])
            with patch("tools.prepare_triground_abv_data.CITY", city):
                for steps in (400, 600):
                    arms = _schedule(selected, steps, assets, output)
                    expected = expected_by_steps[steps]
                    self.assertEqual({arm: len(rows) for arm, rows in arms.items()},
                                     {"A": steps*8, "B": steps*8, "V": steps*8})
                    self.assertEqual(arms["B"], arms["V"])
                    self.assertEqual(Counter(row["task_category"] for row in arms["B"]),
                                     Counter(expected))
                    self.assertTrue(all(row["task_category"] == "old_city" for row in arms["A"]))
                    common_positions = [position for position in range(steps*8)
                                        if position % 80 < 48]
                    self.assertEqual(len(common_positions), steps*8*3//5)
                    self.assertTrue(all(arms["A"][position] == arms["B"][position]
                                        for position in common_positions))
                    for arm in ("A", "B"):
                        old_counts = Counter(row["origin_query_id"] for row in arms[arm]
                                             if row["task_category"] == "old_city")
                        self.assertLessEqual(max(old_counts.values()), 2)
                    special_counts = Counter(row["source_task_id"] for row in arms["B"]
                                             if row["task_category"] != "old_city")
                    self.assertLessEqual(max(special_counts.values()), 8)
                    for task_ids in pair_rows.values():
                        self.assertEqual(len(task_ids), 2)
                        self.assertEqual(special_counts[task_ids[0]], special_counts[task_ids[1]])


if __name__ == "__main__":
    unittest.main()
