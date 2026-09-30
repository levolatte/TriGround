import json
from pathlib import Path
import tempfile
import unittest

from experiments.evidence_decision.merge_reviewed_data import merge


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')


class MergeReviewedDataTests(unittest.TestCase):
    def test_replaces_samples_with_reviewed_steps_even_if_incomplete(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            seed_path, split_path, manifest_path = root / 'train.jsonl', root / 'split.jsonl', root / 'manifest.jsonl'
            collect_dir = root / 'collect' / 'binding'
            collect_dir.mkdir(parents=True)
            output_path = root / 'train_full.jsonl'

            seed = [
                {'origin': 'teacher', 'sample_id': 'complete', 'decision_index': 0,
                 'image_group': 'g_complete', 'source': 'city', 'target_action': 'OLD PREFIX'},
                {'origin': 'teacher', 'sample_id': 'complete', 'decision_index': 1,
                 'image_group': 'g_complete', 'source': 'city', 'target_action': 'OLD END'},
                {'origin': 'teacher', 'sample_id': 'incomplete', 'decision_index': 0,
                 'image_group': 'g_incomplete', 'source': 'city',
                 'target_action': '<tool_call>{"name":"finish","arguments":{"bbox":[0,0,1,1]}}</tool_call>'},
            ] + [
                {'origin': 'grounding_rehearsal', 'trajectory_source': 'grounding_rehearsal_not_agent_trajectory',
                 'sample_id': f'rehearsal-{i}', 'image_group': f'r{i}'} for i in range(750)
            ]
            dump_jsonl(seed_path, seed)
            dump_jsonl(split_path, [
                {'image_group': 'held-dev', 'split': 'dev40'},
                {'image_group': 'held-final', 'split': 'final80'},
            ])
            dump_jsonl(manifest_path, [
                {'id': 'complete', 'source': 'city', 'image_group': 'g_complete'},
                {'id': 'incomplete', 'source': 'roborefit', 'image_group': 'g_incomplete'},
                {'id': 'held', 'source': 'city', 'image_group': 'held-dev'},
            ])
            dump_jsonl(collect_dir / 'records.jsonl', [
                {'id': 'complete', 'complete': True},
                {'id': 'incomplete', 'complete': False},
                {'id': 'held', 'complete': True},
            ])
            dump_jsonl(collect_dir / 'decisions.jsonl', [
                {'sample_id': 'complete', 'decision_index': 0, 'step': 0,
                 'image_group': 'g_complete', 'target_action': 'Checked the right chair.\n'
                 '<tool_call>{"name":"inspect","arguments":{"ids":["P1"],"modalities":["rgb"]}}</tool_call>',
                 'messages': [{'role': 'tool', 'name': 'inspect', 'content': [
                     {'type': 'image', 'image': 'inspect.png', 'modality': 'rgb'}]}],
                 'final_iou_diagnostic': 0.9, 'final_iou_used_for_supervision': False,
                 'gt_bbox': [0, 0, 1, 1]},
                {'sample_id': 'incomplete', 'decision_index': 1, 'step': 1,
                 'image_group': 'g_incomplete', 'target_action': '<tool_call>{"name":"finish","arguments":{"bbox":[0,0,1,1]}}</tool_call>'},
                {'sample_id': 'held', 'decision_index': 0, 'step': 0,
                 'image_group': 'held-dev', 'target_action': '<tool_call>{"name":"finish","arguments":{"bbox":[0,0,1,1]}}</tool_call>'},
            ])

            summary = merge(seed_path, [collect_dir], split_path, manifest_path, output_path)
            rows = [json.loads(line) for line in output_path.read_text(encoding='utf-8').splitlines()]
            complete_rows = [row for row in rows if row.get('origin') == 'teacher' and row['sample_id'] == 'complete']
            self.assertEqual([row['decision_index'] for row in complete_rows], [0])
            self.assertEqual(complete_rows[0]['trajectory_source'], 'new_real_teacher_trajectory')
            self.assertNotIn('final_iou_diagnostic', complete_rows[0])
            self.assertNotIn('gt_bbox', complete_rows[0])
            self.assertNotIn('OLD PREFIX', json.dumps(complete_rows))
            incomplete_rows = [row for row in rows if row.get('origin') == 'teacher' and row['sample_id'] == 'incomplete']
            self.assertEqual([row['decision_index'] for row in incomplete_rows], [1])
            self.assertEqual(incomplete_rows[0]['trajectory_source'], 'new_real_teacher_trajectory')
            self.assertNotIn('KEEP PREFIX', json.dumps(incomplete_rows))
            self.assertEqual(sum(row.get('trajectory_source') == 'grounding_rehearsal_not_agent_trajectory'
                                 for row in rows), 750)
            self.assertEqual(sum(row.get('sample_id') == 'incomplete' for row in rows), 1)
            self.assertFalse(any(row.get('image_group') in {'held-dev', 'held-final'} for row in rows))
            self.assertEqual(summary['total_rows'], len(rows))
            self.assertEqual(summary['teacher_action_counts'], {'finish': 1, 'inspect': 1})
            self.assertEqual(summary['teacher_note_coverage'], 0.5)
            self.assertEqual(summary['teacher_actual_evidence_coverage']['visual_inspection']['decision_rows'], 1)
            self.assertEqual(summary['observed_tool_image_modalities'], {'rgb': 1})
            self.assertEqual(summary['merge']['samples_excluded_holdout'], 1)
            self.assertEqual(summary['merge']['incomplete_samples_seen'], 1)
            self.assertEqual(summary['merge']['new_teacher_samples'], 2)
            self.assertIn('not asserted', summary['target_4200_rows'])

    def test_duplicate_sample_across_buckets_fails_to_prevent_prefix_mixing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dirs = [root / 'a', root / 'b']
            for directory in dirs:
                directory.mkdir()
                dump_jsonl(directory / 'records.jsonl', [{'id': 'same', 'complete': True}])
                dump_jsonl(directory / 'decisions.jsonl', [
                    {'sample_id': 'same', 'decision_index': 0, 'image_group': 'g',
                     'target_action': '<tool_call>{"name":"inspect","arguments":{}}</tool_call>'}
                ])
            # Use the smallest seed/split/manifest setup; merge must fail before output.
            seed = root / 'seed.jsonl'; dump_jsonl(seed, [])
            split = root / 'split.jsonl'; dump_jsonl(split, [])
            manifest = root / 'manifest.jsonl'
            dump_jsonl(manifest, [{'id': 'same', 'source': 'city', 'image_group': 'g'}])
            with self.assertRaisesRegex(ValueError, 'appears in multiple collection directories'):
                merge(seed, dirs, split, manifest, root / 'out.jsonl')

    def test_sample_without_reviewed_decisions_keeps_seed_teacher_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            collect_dir = root / 'collect'
            collect_dir.mkdir()
            seed = root / 'seed.jsonl'
            dump_jsonl(seed, [{'origin': 'teacher', 'sample_id': 'unreviewed', 'decision_index': 0,
                               'image_group': 'group', 'target_action': 'seed action'}])
            split = root / 'split.jsonl'; dump_jsonl(split, [])
            manifest = root / 'manifest.jsonl'
            dump_jsonl(manifest, [{'id': 'unreviewed', 'source': 'city', 'image_group': 'group'}])
            dump_jsonl(collect_dir / 'records.jsonl', [{'id': 'unreviewed', 'complete': False}])
            dump_jsonl(collect_dir / 'decisions.jsonl', [])
            summary = merge(seed, [collect_dir], split, manifest, root / 'out.jsonl')
            rows = [json.loads(line) for line in (root / 'out.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['target_action'], 'seed action')
            self.assertEqual(summary['merge']['samples_without_decisions'], 1)


if __name__ == '__main__':
    unittest.main()
