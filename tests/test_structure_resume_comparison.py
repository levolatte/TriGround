import pytest
import torch

from tools.compare_structure_resume import comparable_config, equal


def test_peft_target_module_order_is_semantically_unordered():
    left = {'target_modules': ['q_proj', 'v_proj'], 'r': 8}
    right = {'target_modules': ['v_proj', 'q_proj'], 'r': 8}
    equal(comparable_config(left, 'adapter_config.json'), comparable_config(right, 'adapter_config.json'))
    assert left['target_modules'] == ['q_proj', 'v_proj']
    with pytest.raises(AssertionError):
        equal(comparable_config(left, 'adapter_config.json'), comparable_config({**right, 'r': 16}, 'adapter_config.json'))
    with pytest.raises(AssertionError):
        equal(comparable_config(left, 'adapter_config.json'), comparable_config({**right, 'target_modules': ['k_proj']}, 'adapter_config.json'))
    with pytest.raises(AssertionError):
        equal(comparable_config({'target_modules': '.*q_proj'}, 'adapter_config.json'),
              comparable_config({'target_modules': '.*v_proj'}, 'adapter_config.json'))


def test_resume_sample_order_and_tensors_remain_exact():
    with pytest.raises(AssertionError):
        equal({'consumed': ['a', 'b']}, {'consumed': ['b', 'a']})
    with pytest.raises(AssertionError):
        equal(torch.tensor([1.]), torch.tensor([1.00001]))
