"""Export an expanding, reviewed object-teacher training snapshot (CPU only)."""
from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'code'))

from experiments.evidence_decision.collect_object_teachers import collect  # noqa: E402
from experiments.evidence_decision.evaluate import iou  # noqa: E402
from experiments.evidence_decision.merge_reviewed_data import merge, read_jsonl  # noqa: E402


RESULTS_ROOT = PROJECT_ROOT / 'results/visual_agent/evidence_decision_20260930'
DATA_ROOT = RESULTS_ROOT / 'data'
TEACHER_ROOT = RESULTS_ROOT / 'teachers'
LABELS_PATH = DATA_ROOT / 'train_labels_offline.jsonl'
SEED_PATH = DATA_ROOT / 'train_full.jsonl'
SPLIT_PATH = DATA_ROOT / 'holdout120_group_split.jsonl'
MANIFEST_PATH = DATA_ROOT / 'train_manifest.jsonl'
OUTPUT_PATH = DATA_ROOT / 'train_collected_snapshot.jsonl'
SUMMARY_PATH = DATA_ROOT / 'train_collected_snapshot.summary.json'
SELECTION_PATH = DATA_ROOT / 'train_collected_snapshot.selection.json'
REPORT_PATH = DATA_ROOT / 'train_collected_snapshot.md'
TOOL_CALL_PATTERN = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)

# Rank recovery first, then the prior general/additional sources; fresh remains a
# separate raw episode/review input and is never re-imported from fresh_collected*.
SOURCE_SPECS = (
    ('collection_recovery', 'collection_recovery/episodes', 'collection_recovery/reviews'),
    ('collection_missing', 'collection_missing/episodes', 'collection_missing/reviews'),
    ('collection', 'collection/episodes', 'collection/reviews'),
    ('collection_additional', 'collection_additional/episodes', 'collection_additional/reviews'),
    ('fresh', 'fresh', 'fresh_reviews.jsonl'),
)
SOURCE_RANK = {name: rank for rank, (name, _, _) in enumerate(SOURCE_SPECS)}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines()
            if line.strip()]


def review_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.rglob('*.jsonl'))
    return []


def load_reviews(paths: list[Path]) -> tuple[dict[tuple[str, int], dict], int]:
    reviews = {}
    row_count = 0
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row['id']), int(row['step']))
            if key in reviews:
                raise ValueError(f'duplicate review {key} in {path}:{line_number}')
            reviews[key] = row
            row_count += 1
    return reviews, row_count


def final_iou(state: dict, label: dict) -> float | None:
    final = next((step for step in reversed(state['steps'])
                  if step.get('call', {}).get('name') == 'finish'
                  and step['observation']['status'] == 'OK'
                  and step['observation'].get('data', {}).get('bbox') is not None), None)
    if final is None:
        return None
    target = label.get('bbox', label.get('bbox_xyxy_normalized'))
    return iou(final['observation']['data']['bbox'], target)


def action_name(row: dict) -> str:
    match = TOOL_CALL_PATTERN.search(str(row.get('target_action', '')))
    return str(json.loads(match.group(1)).get('name', 'unknown')) if match else 'unknown'


def main() -> None:
    labels = {str(row['id']): row for row in read_jsonl(LABELS_PATH)}
    manifest = {str(row['id']): row for row in read_jsonl(MANIFEST_PATH)}
    held_groups = {str(row['image_group']) for row in read_jsonl(SPLIT_PATH)
                   if row.get('split') in {'dev40', 'final80'}}

    source_reviews: dict[str, dict[tuple[str, int], dict]] = {}
    source_reports = {}
    candidates = []
    for source, episode_rel, reviews_rel in SOURCE_SPECS:
        episode_root = TEACHER_ROOT / episode_rel
        if source == 'fresh' and (episode_root / 'episodes').is_dir():
            episode_root = episode_root / 'episodes'
        review_root = TEACHER_ROOT / reviews_rel
        paths = sorted(episode_root.rglob('episode.json')) if episode_root.is_dir() else []
        review_paths = review_files(review_root)
        reviews, review_count = load_reviews(review_paths)
        source_reviews[source] = reviews
        source_reports[source] = {
            'episode_root': str(episode_root), 'review_root': str(review_root),
            'episode_files': len(paths), 'review_files': [str(path) for path in review_paths],
            'review_rows': review_count,
        }
        for episode_path in paths:
            state = json.loads(episode_path.read_text(encoding='utf-8'))
            sample_id = str(state['sample_id'])
            row = state.get('row', {})
            group = str(manifest.get(sample_id, {}).get(
                'image_group', row.get('image_group', '')))
            if sample_id not in labels:
                raise KeyError(f'{sample_id} has no offline training label')
            if not group:
                raise ValueError(f'{sample_id} has no image_group')
            applicable_reviews = {key: review for key, review in reviews.items()
                                  if key[0] == sample_id}
            score = final_iou(state, labels[sample_id])
            qualified = 0
            partial_finish_reviews = 0
            for step in state['steps']:
                review = applicable_reviews.get((sample_id, int(step['index'])))
                is_finish = step['call']['name'] == 'finish'
                if is_finish and not state.get('complete') and review:
                    partial_finish_reviews += 1
                if (review and review.get('use_action') is True and step.get('protocol_valid')
                        and (not is_finish or (state.get('complete') and score is not None and score >= 0.5))):
                    qualified += 1
            candidates.append({
                'source': source, 'sample_id': sample_id, 'image_group': group,
                'episode_path': episode_path, 'schema_version': state.get('schema_version'),
                'complete': bool(state.get('complete')), 'raw_steps': len(state['steps']),
                'review_rows': len(applicable_reviews), 'qualified_reviewed_steps': qualified,
                'partial_finish_reviews_not_trainable': partial_finish_reviews,
            })

    by_sample: dict[str, list[dict]] = defaultdict(list)
    for candidate in candidates:
        by_sample[candidate['sample_id']].append(candidate)

    selected = []
    for sample_id, rows in by_sample.items():
        qualified_rows = [row for row in rows if row['qualified_reviewed_steps'] > 0]
        eligible_rows = qualified_rows or [row for row in rows if row['complete']]
        if not eligible_rows:
            continue
        eligible_rows.sort(key=lambda row: (
            not (row['source'] == 'collection_recovery' and row['qualified_reviewed_steps'] > 0),
            not row['complete'], -row['qualified_reviewed_steps'], SOURCE_RANK[row['source']],
            str(row['episode_path']),
        ))
        selected.append(eligible_rows[0])
    selected.sort(key=lambda row: (SOURCE_RANK[row['source']], row['sample_id']))
    winner_by_sample = {row['sample_id']: row for row in selected}

    for candidate in candidates:
        winner = winner_by_sample.get(candidate['sample_id'])
        if winner is None:
            candidate['selection_status'] = 'excluded'
            candidate['selection_reason'] = 'partial_trajectory_without_qualified_steps'
            continue
        candidate['selection_status'] = 'selected' if candidate is winner else 'excluded'
        if candidate is winner:
            if candidate['source'] == 'collection_recovery' and candidate['qualified_reviewed_steps']:
                reason = 'preferred_recovery_trajectory_with_qualified_steps'
            elif candidate['complete']:
                reason = 'selected_complete_original_trajectory'
            else:
                reason = 'selected_qualified_partial_original_trajectory'
            candidate['selection_reason'] = reason
        else:
            candidate['selection_reason'] = f'duplicate_sample_lower_rank_than_{winner["source"]}'
        if candidate is winner and candidate['image_group'] in held_groups:
            candidate['selection_status'] = 'excluded'
            candidate['selection_reason'] = 'frozen_dev_or_final_group'

    chosen = [row for row in selected if row['image_group'] not in held_groups]
    if len({row['sample_id'] for row in chosen}) != len(chosen):
        raise AssertionError('sample-level trajectory selection is not unique')
    if any(row['image_group'] in held_groups for row in chosen):
        raise AssertionError('held-out group reached collection export')

    selection = {
        'status': 'expanding_collection_snapshot_not_final',
        'source_inputs': source_reports,
        'source_priority': SOURCE_RANK,
        'candidate_trajectories': [
            {**row, 'episode_path': str(row['episode_path'])} for row in candidates
        ],
        'counts': {
            'episode_trajectories_seen': len(candidates),
            'unique_samples_seen': len(by_sample),
            'selected_trajectories': len(chosen),
            'selected_complete_trajectories': sum(row['complete'] for row in chosen),
            'selected_partial_trajectories': sum(not row['complete'] for row in chosen),
            'selected_trajectories_with_qualified_steps': sum(
                row['qualified_reviewed_steps'] > 0 for row in chosen),
            'partial_finish_reviews_not_trainable': sum(
                row['partial_finish_reviews_not_trainable'] for row in chosen),
            'excluded_trajectories': len(candidates) - len(chosen),
            'selected_by_source': dict(Counter(row['source'] for row in chosen)),
            'excluded_by_reason': dict(Counter(row['selection_reason'] for row in candidates
                                               if row['selection_status'] == 'excluded')),
        },
    }
    SELECTION_PATH.write_text(json.dumps(selection, ensure_ascii=False, indent=2) + '\n',
                              encoding='utf-8')

    collect_dirs = []
    collect_summaries = {}
    with tempfile.TemporaryDirectory(prefix='export_evidence_training_snapshot_') as temp_name:
        temp_root = Path(temp_name)
        for source in SOURCE_RANK:
            source_rows = [row for row in chosen if row['source'] == source]
            if not source_rows:
                continue
            input_root = temp_root / 'inputs' / source
            episodes_dir = input_root / 'episodes'
            episodes_dir.mkdir(parents=True)
            review_rows = []
            source_review_map = source_reviews[source]
            for candidate in source_rows:
                sample_id = candidate['sample_id']
                episode_dir = episodes_dir / sample_id
                episode_dir.mkdir()
                shutil.copyfile(candidate['episode_path'], episode_dir / 'episode.json')
                state = json.loads(candidate['episode_path'].read_text(encoding='utf-8'))
                partial_finish_indices = {
                    int(step['index']) for step in state['steps']
                    if step['call']['name'] == 'finish'
                } if not candidate['complete'] else set()
                review_rows.extend(
                    review for (review_id, step_index), review in source_review_map.items()
                    if review_id == sample_id and step_index not in partial_finish_indices
                )
            reviews_path = input_root / 'reviews.jsonl'
            reviews_path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n'
                                              for row in review_rows), encoding='utf-8')
            export_dir = temp_root / source
            export_summary = collect(episodes_dir, reviews_path, export_dir, LABELS_PATH,
                                     migrate_legacy_v1=True)
            collect_dirs.append(export_dir)
            collect_summaries[source] = export_summary

        summary = merge(SEED_PATH, collect_dirs, SPLIT_PATH, MANIFEST_PATH,
                        OUTPUT_PATH, SUMMARY_PATH)

    seed = read_jsonl(SEED_PATH)
    exported = read_jsonl(OUTPUT_PATH)
    seed_nonteacher = [row for row in seed if row.get('origin') != 'teacher']
    output_nonteacher = [row for row in exported if row.get('origin') != 'teacher']
    if seed_nonteacher != output_nonteacher:
        raise AssertionError('the frozen seed nonteacher rows, including the 750 rehearsal rows, changed')
    if summary['output_rows_in_holdout_groups'] != 0:
        raise AssertionError('held-out image_group leaked into the snapshot')

    new_teacher = [row for row in exported if row.get('trajectory_source') == 'new_real_teacher_trajectory']
    partial_sample_ids = {row['sample_id'] for row in chosen if not row['complete']}
    partial_teacher_rows = [row for row in new_teacher if row['sample_id'] in partial_sample_ids]
    complete_teacher_rows = [row for row in new_teacher if row['sample_id'] not in partial_sample_ids]
    partial_action_counts = Counter(action_name(row) for row in partial_teacher_rows)
    complete_action_counts = Counter(action_name(row) for row in complete_teacher_rows)
    if partial_action_counts.get('finish', 0):
        raise AssertionError('an incomplete trajectory supplied finish supervision')
    new_sample_ids = {str(row['sample_id']) for row in new_teacher}
    retained_seed_teacher_ids = {
        str(row.get('sample_id')) for row in exported if row.get('origin') == 'teacher'
        and row.get('trajectory_source') != 'new_real_teacher_trajectory'
    }
    if new_sample_ids & retained_seed_teacher_ids:
        raise AssertionError('seed and new teacher rows were mixed for the same sample')
    path_prefixes = Counter()
    image_parts = 0
    empty_image_paths = 0
    for row in new_teacher:
        row_images = 0
        for message in row.get('messages', []):
            for part in message.get('content', []) if isinstance(message.get('content'), list) else []:
                if part.get('type') != 'image':
                    continue
                image_parts += 1
                row_images += 1
                image_path = part.get('image') or part.get('path')
                if not isinstance(image_path, str) or not image_path:
                    empty_image_paths += 1
                    continue
                path_prefixes['/'.join(image_path.split('/')[:4]) if image_path.startswith('/')
                              else image_path.split('/')[0]] += 1
        if row_images == 0:
            raise AssertionError(f"new teacher row {row['id']} has no image part")
    if empty_image_paths:
        raise AssertionError(f'{empty_image_paths} image parts have no usable path')

    summary.update({
        'snapshot_status': 'expanding_collection_snapshot_not_final',
        'selection_path': str(SELECTION_PATH),
        'selection_counts': selection['counts'],
        'source_inputs': source_reports,
        'collector_summaries_by_source': collect_summaries,
        'seed_nonteacher_rows_retained_exactly': len(output_nonteacher),
        'seed_new_teacher_sample_overlap': 0,
        'new_teacher_rows_by_trajectory_completion': {
            'complete': len(complete_teacher_rows), 'partial': len(partial_teacher_rows)},
        'new_teacher_actions_by_trajectory_completion': {
            'complete': dict(sorted(complete_action_counts.items())),
            'partial': dict(sorted(partial_action_counts.items()))},
        'new_teacher_image_path_audit': {
            'rows_with_images': len(new_teacher), 'image_parts': image_parts,
            'empty_paths': empty_image_paths, 'path_prefix_counts': dict(path_prefixes),
            'note': 'Paths are preserved from the source traces; remote file existence is checked on the collection host.',
        },
    })
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    REPORT_PATH.write_text(
        '# 采集课程扩充快照\n\n'
        '这是当前可用原始采集轨迹的扩充中快照，不是冻结版课程。每个样本仅选一份原始会话轨迹，'
        '不拼接不同会话的前缀；完整轨迹优先，已执行且独立盲审认可的部分轨迹也会保留。'
        '有合格审核步骤的恢复轨迹优先，其余按完整状态、合格步骤数与来源顺序选择。'
        '开发集/最终集图组不会进入训练输出。\n\n'
        '导出仅监督审核认可且协议有效的原动作；不完整轨迹上的 finish 不作为监督，训练集 GT '
        '只在离线过滤已完成轨迹中的错误 finish 时使用。'
        '错误工具历史保留在原轨迹上下文中，合格的 search/inspect/depth 等证据动作仍可监督。'
        '输入为 raw `fresh` 及 `fresh_reviews.jsonl`；不会再次读入 `fresh_collected*` 旧导出。\n\n'
        f'- 训练行数：{summary["total_rows"]}（teacher {summary["teacher_rows"]}，'
        f'原seed非teacher行逐行保留 {summary["seed_nonteacher_rows_retained_exactly"]} 条）。\n'
        f'- 原始轨迹：{selection["counts"]["episode_trajectories_seen"]}，选中 '
        f'{selection["counts"]["selected_trajectories"]} 条；'
        f'选中完整轨迹 {selection["counts"]["selected_complete_trajectories"]} 条，'
        f'选中部分轨迹 {selection["counts"]["selected_partial_trajectories"]} 条。\n'
        f'- 新teacher动作计数：`{json.dumps(summary["teacher_action_counts_by_trajectory"], ensure_ascii=False)}`。\n'
        f'- 新teacher按轨迹状态计数：`{json.dumps(summary["new_teacher_actions_by_trajectory_completion"], ensure_ascii=False)}`。\n'
        f'- 冻结holdout输出行：{summary["output_rows_in_holdout_groups"]}。\n'
        f'- 逐轨迹选择记录：`{SELECTION_PATH.name}`。\n'
        f'- 完整统计：`{SUMMARY_PATH.name}`。\n',
        encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
