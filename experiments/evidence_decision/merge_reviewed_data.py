"""Merge explicitly reviewed teacher trajectories into the frozen seed."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re


WORKSPACE = Path(__file__).resolve().parents[3]
DEFAULT_DATA = WORKSPACE / 'results/visual_agent/evidence_decision_20260930/data'
_TOOL_CALL = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines()
            if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n'
                            for row in rows), encoding='utf-8')


def _held_groups(path: Path) -> set[str]:
    rows = read_jsonl(path)
    return {str(row['image_group']) for row in rows
            if row.get('split') in {'dev40', 'final80'}}


def _teacher_rows(collect_dirs: list[Path], held_groups: set[str],
                  sources: dict[str, str]) -> tuple[list[dict], set[str], dict]:
    selected: dict[str, dict[tuple[str, int], dict]] = {}
    groups_by_sample: dict[str, str] = {}
    report = {'complete_samples_seen': 0, 'incomplete_samples_seen': 0,
              'samples_excluded_holdout': 0, 'samples_without_decisions': 0,
              'duplicate_decisions': 0}
    for directory in collect_dirs:
        records_path, decisions_path = directory / 'records.jsonl', directory / 'decisions.jsonl'
        records = read_jsonl(records_path)
        decisions = read_jsonl(decisions_path)
        by_sample: dict[str, dict] = {}
        for record in records:
            sample_id = str(record['id'])
            if sample_id in by_sample:
                raise ValueError(f'{records_path}: duplicate sample record {sample_id}')
            by_sample[sample_id] = record
        decision_rows: dict[str, list[dict]] = defaultdict(list)
        for decision in decisions:
            decision_rows[str(decision['sample_id'])].append(decision)
        for sample_id, record in by_sample.items():
            if record.get('complete'):
                report['complete_samples_seen'] += 1
            else:
                report['incomplete_samples_seen'] += 1
            sample_decisions = decision_rows.get(sample_id, [])
            if not sample_decisions:
                report['samples_without_decisions'] += 1
                continue
            # Use the manifest's stable grouping when available; the trajectory row is the fallback.
            image_group = str(sample_decisions[0].get('image_group') or record.get('image_group') or '')
            image_group = sources.get(f'__group__:{sample_id}', image_group)
            if not image_group:
                raise ValueError(f'teacher sample {sample_id} has no image_group; cannot audit holdout')
            if image_group in held_groups:
                report['samples_excluded_holdout'] += 1
                continue
            if sample_id in selected:
                raise ValueError(f'sample {sample_id} appears in multiple collection directories; '
                                 'refusing to mix trajectories')
            groups_by_sample[sample_id] = image_group
            sample_map: dict[tuple[str, int], dict] = {}
            for original in sample_decisions:
                decision_index = int(original.get('decision_index', original.get('step')))
                key = (sample_id, decision_index)
                if key in sample_map:
                    if sample_map[key] != original:
                        raise ValueError(f'conflicting duplicate teacher decision {key} in {directory}')
                    report['duplicate_decisions'] += 1
                    continue
                # Collector rows are already action-only; explicitly drop offline scoring fields.
                row = {key: value for key, value in original.items()
                       if key not in {'final_iou_diagnostic', 'final_iou_used_for_supervision',
                                       'bbox', 'gt_bbox', 'ground_truth', 'iou'}}
                row['origin'] = 'teacher'
                row['source'] = sources.get(sample_id, row.get('source', 'unknown'))
                row['trajectory_source'] = 'new_real_teacher_trajectory'
                row['teacher_collection'] = directory.name
                sample_map[key] = row
            if not sample_map:
                report['samples_without_decisions'] += 1
                continue
            selected[sample_id] = sample_map
    rows = [row for sample_map in selected.values() for row in sample_map.values()]
    return rows, set(selected), {**report, 'new_teacher_samples': len(selected),
                                 'new_teacher_rows': len(rows), 'groups_by_sample': groups_by_sample}


def _action_name(row: dict) -> str:
    match = _TOOL_CALL.search(str(row.get('target_action', '')))
    if match:
        return str(json.loads(match.group(1)).get('name', 'unknown'))
    return 'unknown'


def _evidence_for_row(row: dict) -> tuple[set[str], int, set[str]]:
    categories: set[str] = set()
    modalities: set[str] = set()
    observations = 0
    for message in row.get('messages', []):
        if message.get('role') != 'tool':
            continue
        observations += 1
        name = str(message.get('name', '')).lower()
        if name == 'inspect':
            categories.add('visual_inspection')
        elif 'depth' in name:
            categories.add('depth_measurement')
        elif 'search' in name:
            categories.add('candidate_search')
        elif name:
            categories.add(f'tool:{name}')
        content = message.get('content', [])
        if isinstance(content, list):
            for part in content:
                if part.get('type') == 'image' and part.get('modality'):
                    modalities.add(str(part['modality']))
        for part in message.get('content', []) if isinstance(message.get('content'), list) else []:
            if part.get('type') == 'text':
                text = str(part.get('text', ''))
                # Tool payloads are JSON text in the exported trajectory.
                if '"modality":"depth' in text or '"modality": "depth' in text:
                    categories.add('depth_measurement')
                if '"searched' in text or '"candidates_added"' in text:
                    categories.add('candidate_search')
    return categories, observations, modalities


def summarize(rows: list[dict], held_groups: set[str], merge_report: dict) -> dict:
    teacher = [row for row in rows if row.get('origin') == 'teacher']
    actions = Counter(_action_name(row) for row in teacher)
    actions_by_trajectory = defaultdict(Counter)
    for row in teacher:
        actions_by_trajectory[str(row.get('trajectory_source', 'frozen_seed_teacher'))][
            _action_name(row)] += 1
    notes = sum(bool(str(row.get('target_action', '')).split('<tool_call>', 1)[0].strip())
                for row in teacher)
    action_evidence: dict[str, Counter] = defaultdict(Counter)
    modalities = Counter()
    observed_samples: dict[str, set[str]] = defaultdict(set)
    tool_observation_rows = 0
    for row in teacher:
        categories, observation_count, observed_modalities = _evidence_for_row(row)
        tool_observation_rows += observation_count
        for category in categories:
            action_evidence[category]['decision_rows_with_evidence'] += 1
            observed_samples[category].add(str(row['sample_id']))
        for modality in observed_modalities:
            modalities[modality] += 1
    by_source = Counter(str(row.get('source', 'unknown')) for row in rows)
    teacher_by_source = Counter(str(row.get('source', 'unknown')) for row in teacher)
    groups = {str(row.get('image_group', '')) for row in rows if row.get('image_group')}
    samples = {str(row.get('sample_id', row.get('id', ''))) for row in rows}
    groups_by_source: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row.get('image_group'):
            groups_by_source[str(row.get('source', 'unknown'))].add(str(row['image_group']))
    held_rows = [row for row in rows if str(row.get('image_group', '')) in held_groups]
    return {
        'total_rows': len(rows), 'teacher_rows': len(teacher),
        'teacher_action_counts': dict(sorted(actions.items())),
        'teacher_action_counts_by_trajectory': {
            source: dict(sorted(counts.items()))
            for source, counts in sorted(actions_by_trajectory.items())},
        'rows_with_supported_note': notes,
        'teacher_note_coverage': notes / len(teacher) if teacher else 0.0,
        'teacher_rows_by_dataset_source': dict(sorted(teacher_by_source.items())),
        'all_rows_by_dataset_source': dict(sorted(by_source.items())),
        'unique_image_groups': len(groups), 'unique_samples': len(samples),
        'groups_by_source': {source: len(source_groups)
                             for source, source_groups in sorted(groups_by_source.items())},
        'teacher_actual_evidence_coverage': {
            category: {'decision_rows': count['decision_rows_with_evidence'],
                       'samples': len(observed_samples[category])}
            for category, count in sorted(action_evidence.items())},
        'teacher_tool_observation_messages': tool_observation_rows,
        'observed_tool_image_modalities': dict(sorted(modalities.items())),
        'frozen_holdout_groups': len(held_groups), 'output_rows_in_holdout_groups': len(held_rows),
        'merge': {key: value for key, value in merge_report.items() if key != 'groups_by_sample'},
        'target_4200_rows': 'not asserted; this summary reports only rows actually merged',
    }


def merge(seed_path: Path, collect_dirs: list[Path], split_path: Path, manifest_path: Path,
          output_path: Path, summary_path: Path | None = None) -> dict:
    held_groups = _held_groups(split_path)
    manifest = read_jsonl(manifest_path)
    source_by_id = {str(row['id']): str(row.get('source', 'unknown')) for row in manifest}
    source_by_id.update({f'__group__:{row["id"]}': str(row.get('image_group', '')) for row in manifest})
    new_rows, replaced_samples, report = _teacher_rows(collect_dirs, held_groups, source_by_id)
    seed = read_jsonl(seed_path)
    kept = []
    removed_seed_teacher = 0
    for row in seed:
        group = str(row.get('image_group', ''))
        if group in held_groups:
            continue
        if row.get('origin') == 'teacher' and str(row.get('sample_id', '')) in replaced_samples:
            removed_seed_teacher += 1
            continue
        if row.get('origin') == 'teacher' and not row.get('source'):
            enriched = dict(row)
            enriched['source'] = source_by_id.get(str(row.get('sample_id', '')), 'unknown')
            kept.append(enriched)
        else:
            kept.append(row)
    # Stable order: frozen seed rows first, then new trajectories by collection order and step.
    merged = kept + new_rows
    seen: set[tuple[str, str]] = set()
    deduped = []
    for row in merged:
        if row.get('origin') == 'teacher':
            key = (str(row['sample_id']), str(row.get('decision_index', row.get('step'))))
            if key in seen:
                raise ValueError(f'duplicate sample+decision remained after whole-sample replacement: {key}')
            seen.add(key)
        deduped.append(row)
    write_jsonl(output_path, deduped)
    report['replaced_seed_teacher_rows'] = removed_seed_teacher
    report['seed_rows'] = len(seed)
    report['seed_teacher_rows'] = sum(row.get('origin') == 'teacher' for row in seed)
    report['seed_nonteacher_rows_retained'] = sum(row.get('origin') != 'teacher' for row in deduped)
    summary = summarize(deduped, held_groups, report)
    summary_path = summary_path or output_path.with_suffix('.summary.json')
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=Path, default=DEFAULT_DATA / 'train.jsonl')
    parser.add_argument('--split', type=Path, default=DEFAULT_DATA / 'holdout120_group_split.jsonl')
    parser.add_argument('--manifest', type=Path, default=DEFAULT_DATA / 'train_manifest.jsonl')
    parser.add_argument('--collect-dir', type=Path, action='append', required=True,
                        help='A collect_object_teachers output directory; repeat per bucket')
    parser.add_argument('--output', type=Path, default=DEFAULT_DATA / 'train_full.jsonl')
    parser.add_argument('--summary', type=Path)
    args = parser.parse_args()
    summary = merge(args.seed, args.collect_dir, args.split, args.manifest,
                    args.output, args.summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
