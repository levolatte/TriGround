from tools.report_aux_selection import candidate_summary
from tools.run_aux_selection_queue import check_complete
import json
import pytest


def test_detector_coverage_uses_detector_box_not_merged_c_box():
    target = [.1, .1, .2, .2]
    wrong = [.7, .7, .8, .8]
    evidence = {'a': {'query_info': {'scope': 'single'}, 'candidates': [
        {'role': 'target', 'bbox': target, 'sources': [{'modality': 'c', 'bbox': target}, {'modality': 'ir', 'bbox': wrong}]}
    ]}}
    runs = {name: {'rows': {'a': {'primary': target}}} for name in ['C', 'selection']}
    result = candidate_summary(evidence, {'a': {'bbox': target}}, runs)
    assert result['target_pool_covered'] == 1
    assert result['ir_detector_covered'] == 0


def test_queue_partial_result_cannot_mark_complete(tmp_path):
    manifest, results = tmp_path/'m.jsonl', tmp_path/'p.jsonl'
    manifest.write_text('\n'.join(json.dumps({'id': x}) for x in ['a', 'b']))
    results.write_text(json.dumps({'id': 'a', 'parsed': True}))
    with pytest.raises(AssertionError):
        check_complete(manifest, results)
