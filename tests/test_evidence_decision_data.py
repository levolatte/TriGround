import json

import pytest


@pytest.fixture
def data_builder():
    from experiments.evidence_decision import rebuild_decision_data
    return rebuild_decision_data


def test_capability_slices_are_nonempty_metadata_proxies(data_builder):
    row = {
        'query': 'The second chair to the left of the table, closest to camera',
        'available_modalities': ['rgb', 'ir', 'depth_visual'],
        'depth_encoding': 'city_mm',
    }
    candidate = {'query_info': {'reference_categories': ['table'], 'relation_type': 'left'}}

    slices = data_builder.capability_slices(row, candidate)

    assert 'text_proxy:all' in slices
    assert 'text_proxy:ordinal' in slices
    assert 'text_proxy:spatial_relation' in slices
    assert 'text_proxy:reference_binding' in slices
    assert 'text_proxy:distance_language' in slices
    assert 'metadata:infrared_available' in slices
    assert 'metadata:depth_visual_available' in slices
    assert 'metadata:city_mm_depth' in slices


def test_frozen_group_split_is_stable_disjoint_and_exact(data_builder):
    groups = [f'group-{index:03d}' for index in range(120)]

    dev_a, final_a = data_builder._split_groups(groups, seed=2032)
    dev_b, final_b = data_builder._split_groups(reversed(groups), seed=2032)

    assert (dev_a, final_a) == (dev_b, final_b)
    assert len(dev_a) == 40 and len(final_a) == 80
    assert not set(dev_a) & set(final_a)
    with pytest.raises(ValueError, match='expected 120 unique holdout groups'):
        data_builder._split_groups(groups[:-1], seed=2032)


def test_training_view_drops_offline_scores_and_ground_truth(data_builder):
    row = {
        'id': 'sample:teacher_step:0',
        'messages': [],
        'tools': [],
        'target_action': '<tool_call>{}</tool_call>',
        'final_iou_diagnostic': 0.92,
        'terminal_bbox': [0.1, 0.2, 0.3, 0.4],
        'bbox': [0.1, 0.2, 0.3, 0.4],
        'query': 'red chair',
    }

    clean = data_builder._training_view(row)

    assert clean['target_action'] == row['target_action']
    assert not {'final_iou_diagnostic', 'terminal_bbox', 'bbox', 'query'} & set(clean)


def test_blind_job_export_contains_no_offline_labels(data_builder):
    tasks, candidates = [], []
    for index in range(8):
        sample_id = f'sample-{index}'
        tasks.append({
            'id': sample_id,
            'image_group': f'group-{index}',
            'query': f'find object {index}',
            'images': {'rgb': '/remote/rgb.png'},
            'available_modalities': ['rgb'],
            'bbox': [0.1, 0.1, 0.2, 0.2],
            'label': 'hidden',
        })
        candidates.append({
            'id': sample_id,
            'query_info': {},
            'candidates': [{'id': 'P1'}, {'id': 'P2'}],
            'c_bbox': [0.1, 0.1, 0.2, 0.2],
        })

    jobs, summary = data_builder._blind_jobs(tasks, candidates, seed=2032)

    assert len(jobs) == 8
    assert summary['pending_blind_initial_jobs']['initial_open_choice']['shortfall'] == 692
    assert summary['observation_continue']['pending_decisions'] == 0
    for job in jobs:
        assert not {'bbox', 'label', 'labels', 'gt', 'final_iou'} & set(job)
        assert job['teacher_mode'] == 'blind_open_tool_episode'
        assert job['continue_after_real_observation'] is True
        assert 'Hidden GT' in job['teacher_policy']


def test_recovery_export_omits_terminal_event_payload(tmp_path, data_builder):
    failure_dir = tmp_path / 'failures'
    failure_dir.mkdir()
    (failure_dir / 'sample.json').write_text(json.dumps({
        'id': 'sample',
        'events': [
            {'type': 'tool_call', 'call': {'name': 'inspect'}},
            {'type': 'observation', 'text': 'actual first observation'},
            {'type': 'tool_call', 'call': {'name': 'finish', 'arguments': {'bbox': [0, 0, 1, 1]}}},
        ],
    }), encoding='utf-8')
    student = [{'id': 'sample', 'image_group': 'group-1', 'query': 'chair'}]

    jobs, summary = data_builder._recovery_jobs(failure_dir, student)

    assert len(jobs) == 1
    assert jobs[0]['prefix_events'] == 2
    assert jobs[0]['migrate_legacy_prefix'] is True
    assert 'events' not in jobs[0]
    assert 'arguments' not in jobs[0]
    assert summary['effective_prefix_event_counts'] == {'2': 1}


def test_legacy_episode_migration_uses_current_open_tools():
    from experiments.evidence_decision.collect_object_teachers import _rerender_legacy_steps
    from experiments.evidence_decision.object_controller import TOOL_SCHEMAS

    call = {'name': 'finish', 'arguments': {'bbox': [0.1, 0.2, 0.3, 0.4]}}
    state = {
        'row': {'query': 'the red chair', 'images': {}},
        'initial_candidates': [],
        'initial_messages': [],
        'schema_version': 'object-teacher-episode-v1',
        'steps': [{
            'index': 0,
            'call': call,
            'note': 'The red chair is distinct.',
            'observation': {'status': 'OK', 'text': 'bbox accepted',
                            'data': {'bbox': call['arguments']['bbox']}, 'images': []},
            'protocol_valid': True,
            'recoverable': False,
            'candidates_after': [],
        }],
    }

    migrated = _rerender_legacy_steps(state)[0]

    assert migrated['call'] == call
    assert migrated['target_action'].endswith(json.dumps(call, separators=(',', ':')) + '</tool_call>')
    assert migrated['tools'] == TOOL_SCHEMAS
    assert migrated['source_prompt_version'] == 'legacy-unversioned-v1'


def test_legacy_budget_finish_migration_keeps_tools_as_one_item_list():
    from experiments.evidence_decision.collect_object_teachers import _rerender_legacy_steps
    from experiments.evidence_decision.object_controller import FINISH_SCHEMA

    steps = []
    for index in range(6):
        call = {'name': 'inspect', 'arguments': {'ids': ['P1'], 'modalities': ['rgb'], 'region': 'full'}}
        steps.append({
            'index': index,
            'call': call,
            'note': 'Compare the current candidates.',
            'observation': {'status': 'OK', 'text': 'inspection recorded',
                            'data': {'regions': []}, 'images': []},
            'protocol_valid': True,
            'recoverable': False,
            'candidates_after': [],
        })
    final_call = {'name': 'finish', 'arguments': {'bbox': [0.1, 0.2, 0.3, 0.4]}}
    steps.append({
        'index': 6,
        'call': final_call,
        'note': 'The supported box is ready.',
        'observation': {'status': 'OK', 'text': 'bbox accepted',
                        'data': {'bbox': final_call['arguments']['bbox']}, 'images': []},
        'protocol_valid': True,
        'recoverable': False,
        'candidates_after': [],
    })
    state = {
        'row': {'query': 'the red chair', 'images': {}},
        'initial_candidates': [],
        'initial_messages': [],
        'schema_version': 'object-teacher-episode-v1',
        'steps': steps,
    }

    migrated = _rerender_legacy_steps(state)

    tools = migrated[6]['tools']
    assert isinstance(tools, list)
    assert len(tools) == 1
    assert tools[0] == FINISH_SCHEMA
    assert migrated[6]['call'] == final_call
