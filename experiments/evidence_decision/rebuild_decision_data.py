"""Export reviewed open-tool decisions, frozen group splits, and blind jobs.

Only source manifests, cached candidates, source annotations, reviewed teacher
episodes, and observed student prefixes are consumed. Ground truth is kept in
offline label/score files and is never included in a blind teacher job.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re

from . import collect_object_teachers, prepare_grounding_rehearsal


WORKSPACE = Path(__file__).resolve().parents[3]
OLD_CAP = WORKSPACE / 'results/visual_agent/capability_rebuild_20260929'
DEFAULT_OUT = WORKSPACE / 'results/visual_agent/evidence_decision_20260930/data'
REMOTE_ROOT = '/root/autodl-tmp/rematch_20260922'
REMOTE_B = REMOTE_ROOT + '/results/visual_agent/evidence_decision_20260930'


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines()
            if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n'
                            for row in rows), encoding='utf-8')


def _remote_path(value: str) -> str:
    original = str(value)
    normalized = original.replace('\\', '/')
    local_roots = (
        ('F:/AIC/results/visual_agent/capability_rebuild_20260929',
         REMOTE_ROOT + '/results/visual_agent/capability_rebuild_20260929'),
        ('F:/AIC/results/visual_agent/decision_rebuild_20260929',
         REMOTE_ROOT + '/results/visual_agent/decision_rebuild_20260929'),
        ('F:/AIC', REMOTE_ROOT),
    )
    for local, remote in local_roots:
        if normalized.casefold().startswith((local + '/').casefold()):
            return remote + normalized[len(local):]
    return normalized


def _map_paths(value):
    if isinstance(value, list):
        return [_map_paths(item) for item in value]
    if isinstance(value, dict):
        return {key: (_remote_path(item) if key in {'image', 'path'} and isinstance(item, str)
                      else _map_paths(item)) for key, item in value.items()}
    return value


_ORDINAL_RE = re.compile(r'\b(first|second|third|fourth|fifth|sixth|last|next|another|one of)\b', re.I)
_SPATIAL_RE = re.compile(r'\b(left|right|above|below|behind|front|in front|between|beside|next to|under|center|middle|among|through|across)\b', re.I)
_DISTANCE_RE = re.compile(r'\b(near|nearest|closest|far|farthest|furthest|distance|camera|foreground|background|closer|farther)\b', re.I)


def capability_slices(row: dict, candidate_row: dict | None = None) -> list[str]:
    """Coarse text/metadata proxies; tags do not assert visual evidence is present."""
    query = str(row.get('query', ''))
    query_info = (candidate_row or {}).get('query_info') or {}
    tags = ['text_proxy:all']
    if _ORDINAL_RE.search(query):
        tags.append('text_proxy:ordinal')
    if _SPATIAL_RE.search(query):
        tags.append('text_proxy:spatial_relation')
    if query_info.get('reference_categories') or query_info.get('relation_type') not in (None, '', 'none'):
        tags.append('text_proxy:reference_binding')
    if _DISTANCE_RE.search(query):
        tags.append('text_proxy:distance_language')
    available = set(row.get('available_modalities') or [])
    if 'ir' in available:
        tags.append('metadata:infrared_available')
    if 'depth_visual' in available:
        tags.append('metadata:depth_visual_available')
    if row.get('depth_encoding') == 'city_mm':
        tags.append('metadata:city_mm_depth')
    return tags


def _slice_manifest(rows: list[dict], candidate_by_id: dict[str, dict], split: str) -> list[dict]:
    result = []
    for raw in rows:
        row = _map_paths(raw)
        row['split'] = split
        row['capability_slices'] = capability_slices(raw, candidate_by_id.get(str(raw['id'])))
        result.append(row)
    return result


def _split_groups(groups: list[str], seed: int, dev_count: int = 40) -> tuple[list[str], list[str]]:
    ordered = sorted(set(map(str, groups)))
    if len(ordered) != 120 or not 0 < dev_count < len(ordered):
        raise ValueError(f'expected 120 unique holdout groups; found {len(ordered)}')
    random.Random(seed).shuffle(ordered)
    return ordered[:dev_count], ordered[dev_count:]


def _gt_dict(manifest_rows: list[dict], labels_by_id: dict[str, dict]) -> dict:
    result = {}
    for row in manifest_rows:
        sample_id = str(row['id'])
        label = labels_by_id[sample_id]
        result[sample_id] = {
            'bbox': label.get('bbox', label.get('bbox_xyxy_normalized')),
            'visible': row['images']['rgb'],
            'query': row['query'],
            'image_group': row['image_group'],
        }
    return result


def _cached_c_rows(manifest_rows: list[dict], candidates_by_id: dict[str, dict]) -> list[dict]:
    result = []
    for row in manifest_rows:
        sample_id = str(row['id'])
        candidate = candidates_by_id[sample_id]
        result.append({
            'id': sample_id,
            'image_group': row['image_group'],
            'visible': row['images']['rgb'],
            'query': row['query'],
            'bbox': candidate['c_bbox'],
            'prediction_source': 'cached candidate_row.c_bbox',
            'candidate_cache_source': 'capability_rebuild_20260929/data candidate cache',
            'fresh_inference': False,
            'model_checkpoint_provenance': 'unknown; not recorded in source candidate cache',
            'reused_from_existing_cache': candidate.get('reused_from_existing_cache'),
            'source_modalities': sorted({str(source.get('modality'))
                                         for item in candidate.get('candidates', [])
                                         if item.get('is_baseline')
                                         for source in item.get('sources', [])}),
        })
    return result


def _balanced_sample(rows: list[dict], count: int, rng: random.Random,
                     selected_ids: set[str]) -> list[dict]:
    pool = [row for row in rows if str(row['id']) not in selected_ids]
    rng.shuffle(pool)
    chosen = pool[:count]
    selected_ids.update(str(row['id']) for row in chosen)
    return chosen


def _teacher_job(bucket: str, index: int, task: dict, candidate: dict) -> dict:
    sample_id = str(task['id'])
    episode_dir = f'{REMOTE_B}/teachers/blind/{bucket}/{sample_id}'
    return {
        'job_id': f'{bucket}_{index:04d}',
        'sample_id': sample_id,
        'image_group': task['image_group'],
        'source': task.get('source'),
        'query': task['query'],
        'images': task.get('images', {}),
        'available_modalities': task.get('available_modalities', []),
        'capability_slices': capability_slices(task, candidate),
        'selection_bucket': bucket,
        'candidate_count': len(candidate.get('candidates', [])),
        'manifest_path': f'{REMOTE_B}/data/train_manifest.jsonl',
        'candidate_cache_path': f'{REMOTE_B}/data/train_candidates.jsonl',
        'episode_dir': episode_dir,
        'teacher_mode': 'blind_open_tool_episode',
        'label_status': 'pending_teacher_decision',
        'teacher_policy': 'Use only the visible query, images, candidates, and actual tool observations. The bucket is a sampling intent, not an answer. Hidden GT and offline labels are unavailable.',
        'continue_after_real_observation': True,
    }


def _blind_jobs(train_rows: list[dict], candidate_rows: list[dict], seed: int) -> tuple[list[dict], dict]:
    manifest_by_id = {str(row['id']): row for row in train_rows}
    candidates = {str(row['id']): row for row in candidate_rows}
    binding, general, remainder = [], [], []
    for sample_id, candidate in candidates.items():
        task = manifest_by_id[sample_id]
        query_info = candidate.get('query_info') or {}
        has_binding = bool(query_info.get('reference_categories')) or query_info.get('relation_type') not in (None, '', 'none')
        if has_binding and len(candidate.get('candidates', [])) >= 2:
            binding.append(task)
        elif len(candidate.get('candidates', [])) >= 2:
            general.append(task)
        else:
            remainder.append(task)

    rng = random.Random(seed)
    selected_ids: set[str] = set()
    quotas = [('binding', 1000, binding), ('initial_open_choice', 700, general)]
    jobs = []
    counts = {}
    for bucket, quota, pool in quotas:
        chosen = _balanced_sample(pool, quota, rng, selected_ids)
        counts[bucket] = {'target': quota, 'available_before_allocation': len(pool), 'pending': len(chosen),
                          'shortfall': max(0, quota - len(chosen))}
        jobs.extend(_teacher_job(bucket, index, row, candidates[str(row['id'])])
                    for index, row in enumerate(chosen, 1))

    remaining = [row for row in train_rows if str(row['id']) in candidates and
                 str(row['id']) not in selected_ids]
    chosen = _balanced_sample(remaining, 600, rng, selected_ids)
    counts['finish_sufficiency_review'] = {
        'target': 600,
        'available_before_allocation': len(remaining),
        'pending': len(chosen),
        'shortfall': max(0, 600 - len(chosen)),
        'sampling_basis': 'random remaining cached-candidate states; no GT, IoU, or label was used',
    }
    jobs.extend(_teacher_job('finish_sufficiency_review', index, row, candidates[str(row['id'])])
                for index, row in enumerate(chosen, 1))
    return jobs, {
        'seed': seed,
        'target_decisions': {'binding': 1000, 'initial_open_choice': 700,
                             'observation_continue': 700, 'finish_sufficiency_review': 600},
        'pending_blind_initial_jobs': counts,
        'pending_blind_initial_job_total': len(jobs),
        'observation_continue': {
            'target': 700,
            'pending_decisions': 0,
            'condition': 'Only count a next-decision example after a real non-finish tool call and its actual observation are recorded; collect these from open episodes above.',
        },
        'note': 'Buckets describe sampling strata only; they do not force the teacher target action.'
    }


def _recovery_jobs(failure_dir: Path, student_manifest: list[dict]) -> tuple[list[dict], dict]:
    by_id = {str(row['id']): row for row in student_manifest}
    jobs = []
    prefix_counts = Counter()
    for path in sorted(failure_dir.glob('*.json')):
        if path.name == 'inventory.json':
            continue
        failure = json.loads(path.read_text(encoding='utf-8-sig'))
        sample_id = str(failure['id'])
        task = by_id[sample_id]
        events = failure.get('events', [])
        prefix_events = max(0, len(events) - 1)
        prefix_counts[prefix_events] += 1
        jobs.append({
            'job_id': f'recovery_{len(jobs) + 1:03d}',
            'sample_id': sample_id,
            'image_group': task['image_group'],
            'source': task.get('source'),
            'query': task['query'],
            'manifest_path': f'{REMOTE_B}/data/student200_manifest.jsonl',
            'candidate_cache_path': f'{REMOTE_B}/data/student200_candidates.jsonl',
            'trace_path': f"{REMOTE_ROOT}/results/visual_agent/capability_rebuild_20260929/stage1_student200/traces/{sample_id}/trace.json",
            'prefix_events': prefix_events,
            'migrate_legacy_prefix': True,
            'teacher_mode': 'blind_recovery_after_real_prefix',
            'label_status': 'pending_teacher_decision',
            'teacher_policy': 'Continue from this real student history using only visible messages and recorded tool outputs. The omitted terminal event is not part of the prefix. Hidden GT and offline metrics are unavailable.',
        })
    return jobs, {
        'target': 450,
        'available_prefix_jobs': len(jobs),
        'shortfall': max(0, 450 - len(jobs)),
        'effective_prefix_event_counts': {str(key): value for key, value in sorted(prefix_counts.items())},
        'policy': 'Omit the final student event from every recovery prefix so the terminal selected ID/bbox is not shown as an answer.'
    }


def _write_gt(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def _training_view(row: dict) -> dict:
    # Keep labels and IoU diagnostics out of the uploaded model-training rows.
    fields = ('id', 'sample_id', 'image_group', 'decision_index', 'messages', 'tools',
              'target_action', 'origin', 'label_source', 'supervision_status', 'source',
              'trajectory_source', 'bbox_frame', 'bbox_target_kind', 'candidate_snapshot_available')
    return {key: row[key] for key in fields if key in row}


def export(output_dir: Path = DEFAULT_OUT, *, old_cap: Path = OLD_CAP,
           seed: int = 2032, dev_groups: int = 40, overwrite: bool = False) -> dict:
    old_data = old_cap / 'data'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f'{output_dir} is not empty; pass --overwrite to replace the generated data files')

    raw_train = read_jsonl(old_data / 'train_manifest.jsonl')
    raw_train_candidates = read_jsonl(old_data / 'train_candidates.jsonl')
    raw_train_labels = read_jsonl(old_data / 'train_labels.jsonl')
    raw_hold = read_jsonl(old_data / 'holdout_manifest.jsonl')
    raw_hold_candidates = read_jsonl(old_data / 'holdout_candidates.jsonl')
    raw_hold_labels = read_jsonl(old_data / 'holdout_labels.jsonl')
    raw_eval = read_jsonl(old_data / 'holdout_eval120_manifest.jsonl')
    raw_eval_candidates = read_jsonl(old_data / 'holdout_eval120_candidates.jsonl')
    raw_eval_labels = read_jsonl(old_data / 'holdout_eval120_labels.jsonl')
    raw_rehearsal_selection = read_jsonl(old_data / 'grounding_rehearsal750.jsonl')
    raw_student = read_jsonl(old_data / 'student200_manifest.jsonl')
    raw_student_candidates = read_jsonl(old_data / 'student200_candidates.jsonl')
    raw_stage1 = read_jsonl(old_cap / 'analysis_20260930' / 'stage1_train.audit.jsonl')

    candidate_train_by_id = {str(row['id']): _map_paths(row) for row in raw_train_candidates}
    candidate_hold_by_id = {str(row['id']): _map_paths(row) for row in raw_hold_candidates}
    candidate_eval_by_id = {str(row['id']): _map_paths(row) for row in raw_eval_candidates}
    labels_train_by_id = {str(row['id']): row for row in raw_train_labels}
    labels_hold_by_id = {str(row['id']): row for row in raw_hold_labels}
    labels_eval_by_id = {str(row['id']): row for row in raw_eval_labels}

    all_groups = sorted({str(row['image_group']) for row in raw_hold})
    dev_group_ids, final_group_ids = _split_groups(all_groups, seed, dev_groups)
    dev_group_set, final_group_set = set(dev_group_ids), set(final_group_ids)
    held_set = dev_group_set | final_group_set
    eval_by_group = {str(row['image_group']): row for row in raw_eval}
    if set(eval_by_group) != held_set:
        raise ValueError('the one-query evaluation manifest does not cover the frozen 120 holdout groups exactly')

    # The 163-row holdout is the full group pool; the 120-row version is the
    # one-query-per-group fixed scoring denominator.
    raw_dev_pool = [row for row in raw_hold if str(row['image_group']) in dev_group_set]
    raw_final_pool = [row for row in raw_hold if str(row['image_group']) in final_group_set]
    raw_dev_eval = [eval_by_group[group] for group in dev_group_ids]
    raw_final_eval = [eval_by_group[group] for group in final_group_ids]

    def candidates_for(rows, by_id):
        return [by_id[str(row['id'])] for row in rows if str(row['id']) in by_id]

    def labels_for(rows, by_id):
        return [by_id[str(row['id'])] for row in rows if str(row['id']) in by_id]

    train_rows = _slice_manifest(raw_train, candidate_train_by_id, 'train')
    train_candidates = [_map_paths(row) for row in raw_train_candidates]
    train_labels = [_map_paths(row) for row in raw_train_labels]
    dev_rows = _slice_manifest(raw_dev_eval, candidate_eval_by_id, 'dev40')
    final_rows = _slice_manifest(raw_final_eval, candidate_eval_by_id, 'final80')
    hold_rows = _slice_manifest(raw_eval, candidate_eval_by_id, 'holdout120')
    dev_pool_rows = _slice_manifest(raw_dev_pool, candidate_hold_by_id, 'dev40')
    final_pool_rows = _slice_manifest(raw_final_pool, candidate_hold_by_id, 'final80')
    hold_pool_rows = _slice_manifest(raw_hold, candidate_hold_by_id, 'holdout120')

    train_groups = {str(row['image_group']) for row in train_rows}
    stage1_groups = {str(row['image_group']) for row in raw_stage1}
    rehearsal_groups = {str(row['image_group']) for row in raw_rehearsal_selection}
    student_group_by_id = {str(row['id']): str(row['image_group']) for row in raw_student}
    student_groups = set(student_group_by_id.values())
    split_reference_groups = {
        'train_manifest': train_groups,
        'actual_stage1_train_2250': stage1_groups,
        'grounding_rehearsal750': rehearsal_groups,
        'student200': student_groups,
    }
    overlap = {name: sorted(held_set & groups) for name, groups in split_reference_groups.items()}
    if any(overlap.values()) or dev_group_set & final_group_set:
        raise ValueError(f'frozen holdout leakage: overlaps={{{", ".join(f"{k}:{len(v)}" for k,v in overlap.items())}}}')

    # Export existing explicit teacher reviews using the new current open schema.
    teacher_collections = []
    teacher_summaries = {}
    for batch in ('batch01', 'batch02'):
        episode_dir = old_cap / 'teachers' / batch
        review_path = old_cap / 'teachers' / f'{batch}_reviews.jsonl'
        collect_dir = output_dir / 'teacher_collections' / batch
        summary = collect_object_teachers.collect(
            episode_dir, review_path, collect_dir,
            old_data / 'train_labels.jsonl', migrate_legacy_v1=True,
        )
        teacher_summaries[batch] = summary
        teacher_collections.extend(_map_paths(row) for row in read_jsonl(collect_dir / 'decisions.jsonl'))

    train_manifest_path = output_dir / 'train_manifest.jsonl'
    labels_path = output_dir / 'train_labels_offline.jsonl'
    candidates_path = output_dir / 'train_candidates.jsonl'
    write_jsonl(train_manifest_path, train_rows)
    write_jsonl(labels_path, train_labels)
    write_jsonl(candidates_path, train_candidates)

    rehearsal_path = output_dir / 'grounding_rehearsal750.jsonl'
    rehearsal_summary = prepare_grounding_rehearsal.export(
        train_manifest_path, labels_path, rehearsal_path, 750, seed,
        candidates=candidates_path, selection_manifest=old_data / 'grounding_rehearsal750.jsonl',
    )
    rehearsal_rows = read_jsonl(rehearsal_path)
    train_jsonl_rows = [_training_view(row) for row in teacher_collections]
    train_jsonl_rows.extend(_training_view(row) for row in rehearsal_rows)
    write_jsonl(output_dir / 'train.jsonl', train_jsonl_rows)

    student_rows = [_map_paths(row) for row in raw_student]
    student_candidates = [_map_paths(row) for row in raw_student_candidates]
    write_jsonl(output_dir / 'student200_manifest.jsonl', student_rows)
    write_jsonl(output_dir / 'student200_candidates.jsonl', student_candidates)

    dev_candidates = candidates_for(raw_dev_eval, candidate_eval_by_id)
    final_candidates = candidates_for(raw_final_eval, candidate_eval_by_id)
    hold_candidates = candidates_for(raw_eval, candidate_eval_by_id)
    dev_labels = labels_for(raw_dev_eval, labels_eval_by_id)
    final_labels = labels_for(raw_final_eval, labels_eval_by_id)
    hold_labels = labels_for(raw_eval, labels_eval_by_id)

    output_sets = {
        'dev40': (dev_rows, dev_candidates, dev_labels),
        'final80': (final_rows, final_candidates, final_labels),
        'holdout120': (hold_rows, hold_candidates, hold_labels),
    }
    for prefix, (manifest_rows, candidate_rows, label_rows) in output_sets.items():
        write_jsonl(output_dir / f'{prefix}_manifest.jsonl', manifest_rows)
        write_jsonl(output_dir / f'{prefix}_candidates.jsonl', candidate_rows)
        write_jsonl(output_dir / f'{prefix}_labels_offline.jsonl', label_rows)
        _write_gt(output_dir / f'{prefix}_gt.json', _gt_dict(manifest_rows,
                   {str(row['id']): row for row in label_rows}))
        write_jsonl(output_dir / f'{prefix}_cached_c_baseline.jsonl',
                    _cached_c_rows(manifest_rows, candidate_eval_by_id))

    for prefix, rows, by_id in (
        ('dev40', raw_dev_pool, candidate_hold_by_id),
        ('final80', raw_final_pool, candidate_hold_by_id),
        ('holdout120', raw_hold, candidate_hold_by_id),
    ):
        pool_rows = _slice_manifest(rows, by_id, prefix)
        write_jsonl(output_dir / f'{prefix}_pool_manifest.jsonl', pool_rows)
        write_jsonl(output_dir / f'{prefix}_pool_candidates.jsonl', candidates_for(rows, by_id))
        write_jsonl(output_dir / f'{prefix}_pool_labels_offline.jsonl', labels_for(rows, labels_hold_by_id))

    blind_jobs, blind_quota = _blind_jobs(train_rows, train_candidates, seed)
    write_jsonl(output_dir / 'blind_teacher_jobs.jsonl', blind_jobs)
    for bucket in ('binding', 'initial_open_choice', 'finish_sufficiency_review'):
        write_jsonl(output_dir / f'blind_teacher_jobs_{bucket}.jsonl',
                    [row for row in blind_jobs if row['selection_bucket'] == bucket])

    recovery_dir = old_cap / 'student200_raw' / 'student_failures'
    recovery_jobs, recovery_quota = _recovery_jobs(recovery_dir, raw_student)
    write_jsonl(output_dir / 'recovery_jobs55.jsonl', recovery_jobs)

    split_rows = ([{'image_group': group, 'split': 'dev40', 'seed': seed}
                   for group in dev_group_ids] +
                  [{'image_group': group, 'split': 'final80', 'seed': seed}
                   for group in final_group_ids])
    write_jsonl(output_dir / 'holdout120_group_split.jsonl', split_rows)

    teacher_accepted = len(teacher_collections)
    bbox_rehearsal_actual = len(rehearsal_rows)
    actual_action_counts = Counter(json.loads(re.search(
        r'<tool_call>\s*(.*?)\s*</tool_call>', row['target_action'], re.S).group(1))['name']
        for row in teacher_collections)
    source_counts = Counter(str(row.get('source', 'unknown')) for row in train_rows)
    split_audit = {
        'seed': seed,
        'frozen_groups': {'dev40': dev_group_ids, 'final80': final_group_ids},
        'group_counts': {'dev40': len(dev_group_set), 'final80': len(final_group_set),
                         'holdout120': len(held_set)},
        'row_counts': {'old_holdout_full_pool': len(raw_hold),
                       'holdout_eval_one_per_group': len(raw_eval),
                       'dev40_eval': len(dev_rows), 'final80_eval': len(final_rows),
                       'dev40_pool': len(dev_pool_rows), 'final80_pool': len(final_pool_rows)},
        'overlap_with': {name: {'groups': values, 'count': len(values)}
                         for name, values in overlap.items()},
        'dev_final_group_intersection': [],
        'source_provenance': {
            'train_labels': str(old_data / 'train_labels.jsonl'),
            'holdout_labels': str(old_data / 'holdout_labels.jsonl'),
            'cached_c_baseline_model_checkpoint': 'unknown; only existing c_bbox cache is evidenced',
            'cached_c_baseline_rerun': False,
        },
        'capability_slices': 'Nonempty text/query and modality-metadata proxies; not visual truth labels.',
    }
    (output_dir / 'split_audit.json').write_text(json.dumps(split_audit, ensure_ascii=False, indent=2) + '\n',
                                                 encoding='utf-8')

    quota_summary = {
        'target': {'binding': 1000, 'initial_open_choice': 700,
                   'observation_continue': 700, 'finish_sufficiency_review': 600,
                   'recovery': 450, 'bbox_rehearsal': 750},
        'existing_reviewed_teacher_decisions_accepted': teacher_accepted,
        'existing_reviewed_teacher_actions': dict(actual_action_counts),
        'existing_review_batches': teacher_summaries,
        'pending_blind_initial_jobs': blind_quota,
        'recovery_jobs': recovery_quota,
        'bbox_rehearsal': rehearsal_summary,
        'accepted_training_rows_now': len(train_jsonl_rows),
        'accepted_training_rows_by_origin': {'reviewed_teacher_actions': teacher_accepted,
                                             'source_bbox_rehearsal': bbox_rehearsal_actual},
        'teacher_finish_policy': 'Teacher actions are preserved as observed; offline IoU>=0.5 only gates positive terminal-finish supervision. No wrong teacher action is rewritten from GT.',
    }
    (output_dir / 'data_summary.json').write_text(json.dumps({
        'train_manifest_rows': len(train_rows),
        'train_candidate_rows': len(train_candidates),
        'train_label_rows_offline': len(train_labels),
        'train_source_counts': dict(source_counts),
        'teacher_action_counts': dict(actual_action_counts),
        'accepted_teacher_rows': teacher_accepted,
        'bbox_rehearsal_rows': bbox_rehearsal_actual,
        'train_rows': len(train_jsonl_rows),
        'holdout_group_counts': split_audit['group_counts'],
        'blind_job_counts': dict(Counter(row['selection_bucket'] for row in blind_jobs)),
        'recovery_job_count': len(recovery_jobs),
        'cached_c_baseline_model_checkpoint_provenance': 'unknown',
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output_dir / 'teacher_job_quota.json').write_text(json.dumps(quota_summary, ensure_ascii=False, indent=2) + '\n',
                                                       encoding='utf-8')
    return quota_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-cap', type=Path, default=OLD_CAP)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--seed', type=int, default=2032)
    parser.add_argument('--dev-groups', type=int, default=40)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    print(json.dumps(export(args.output_dir, old_cap=args.old_cap, seed=args.seed,
                            dev_groups=args.dev_groups, overwrite=args.overwrite),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
