"""`BboxCompleteCriteria` 的边界测试。

这是"提前停止"的唯一安全阀：一旦判据比解析器更宽松，就会在答案完整之前截断生成，
产出静默的错误结果。因此每条边界都要有反例。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.evaluate_pretrained_grounder import BBOX_KEYED_PATTERN, parse_generated_bbox  # noqa: E402
from tools.predict_native_submission import BboxCompleteCriteria  # noqa: E402


class FakeTokenizer:
    """把"已生成的 token"直接当作字符串返回，从而精确控制判据看到的文本。"""

    def __init__(self, texts: list[str]) -> None:
        self.texts = texts

    def batch_decode(self, _ids, **_kwargs) -> list[str]:
        return list(self.texts)


def run_criteria(texts: list[str], prompt_length: int = 5) -> list[bool]:
    criteria = BboxCompleteCriteria(FakeTokenizer(texts), prompt_length)
    input_ids = torch.zeros((len(texts), prompt_length + 3), dtype=torch.long)
    return criteria(input_ids, None).tolist()


def test_stops_on_complete_keyed_bbox() -> None:
    text = '{"bbox_2d":[476,590,565,920]}'
    assert run_criteria([text]) == [True]


def test_stops_on_complete_bbox_inside_verbose_zeroshot_output() -> None:
    """零样本 27B 的真实形态：bbox 之后还跟着重复查询文本的 label。"""
    text = '```json\n[{"bbox_2d": [476, 590, 565, 920], "label": "The person in the plaid jacket"}]\n```'
    assert run_criteria([text]) == [True]


def test_does_not_stop_on_partial_bbox() -> None:
    """少了第四个坐标时绝不能停——那会截断出无法解析的输出。"""
    for text in ('{"bbox_2d":[476,590,565', '{"bbox_2d":[476,590,565,',
                 '{"bbox_2d":[476,590,565,920', '{"bbox_2d":['):
        assert run_criteria([text]) == [False], text


def test_does_not_stop_on_stray_bracket_before_the_bbox() -> None:
    """刻意不用 stop_strings=']'：更早出现的 ']' 会造成误截断，本判据必须免疫。"""
    assert run_criteria(['Here is a list [1, 2, 3] and then {"bbox_2d":[1,2,3']) == [False]


def test_does_not_stop_without_the_key() -> None:
    """裸数组即使能解析也不停：解析器取最后一个匹配，裸数组上提前收工有取错风险。"""
    assert parse_generated_bbox('[476, 590, 565, 920]') is not None
    assert run_criteria(['[476, 590, 565, 920]']) == [False]


def test_does_not_stop_when_the_keyed_box_is_out_of_range() -> None:
    """越界坐标能匹配正则、也会被解析器拒绝——两个条件缺一不可，因此必须继续生成。"""
    assert BBOX_KEYED_PATTERN.search('{"bbox_2d":[-1.5, 2.25, 300, 4.0]}') is not None
    assert parse_generated_bbox('{"bbox_2d":[-1.5, 2.25, 300, 4.0]}') is None
    assert run_criteria(['{"bbox_2d":[-1.5, 2.25, 300, 4.0]}']) == [False]


def test_per_sequence_semantics_in_a_batch() -> None:
    """批次内有的完成、有的没完成，返回值必须逐序列区分，不能整批一起停。"""
    texts = ['{"bbox_2d":[1,2,3,4]}', '{"bbox_2d":[1,2,3', '{"bbox_2d":[5,6,7,8]}']
    assert run_criteria(texts) == [True, False, True]


def test_negative_and_decimal_coordinates_do_not_stop() -> None:
    """越界坐标能匹配正则，但解析器会拒绝——因此**不允许**停止。

    这是本判据最关键的一条：若按"正则命中即停"，模型先输出非法框、再输出合法框时会被
    提前截断，把本可救回的题变成兜底框。判据必须是"已经拿到合法答案"。
    """
    assert run_criteria(['{"bbox_2d":[-1.5, 2.25, 300, 4.0]}']) == [False]
    assert parse_generated_bbox('{"bbox_2d":[-1.5, 2.25, 300, 4.0]}') is None


def test_stops_once_an_invalid_box_is_followed_by_a_valid_one() -> None:
    """先非法后合法：解析器取最后一个匹配，此时判据应当放行。"""
    text = '{"bbox_2d":[-1.5, 2.25, 300, 4.0]}\n{"bbox_2d":[10, 20, 30, 40]}'
    assert run_criteria([text]) == [True]
    assert parse_generated_bbox(text) == [0.01, 0.02, 0.03, 0.04]


def test_stopping_never_loses_the_parseable_answer() -> None:
    """核心不变量：判据为真 ⟺ 解析器能拿到一个合法框。"""
    positives = [
        '{"bbox_2d":[476,590,565,920]}',
        '```json\n[{"bbox_2d": [476, 590, 565, 920], "label": "x"}]\n```',
        '<think>\n\n</think>\n\n{"bbox_2d":[565,7,595,66]}',
    ]
    for text in positives:
        assert run_criteria([text]) == [True], text
        assert parse_generated_bbox(text) is not None, text

    negatives = [
        '{"bbox_2d":[476,590,565',          # 未写完
        '{"bbox_2d":[-1.5, 2.25, 300, 4.0]}',  # 越界，解析器拒绝
        '{"bbox_2d":[500, 500, 400, 600]}',    # 反向坐标
        'Here is a list [1, 2, 3]',            # 更早的 ']' 不应误触发
        '[476, 590, 565, 920]',                # 无键裸数组：解析器有兜底但判据不放行
    ]
    for text in negatives:
        assert run_criteria([text]) == [False], text
