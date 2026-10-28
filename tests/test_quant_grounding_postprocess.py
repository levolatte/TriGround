from tools.quant_grounding_postprocess import normalize_bbox_completion


MALFORMED = '{"bbox_2d":[593,362),(670,416]}'
REPAIRED = '{"bbox_2d":[593,362,670,416]}'


def test_repairs_observed_crossed_coordinate_pair_punctuation():
    assert normalize_bbox_completion(
        MALFORMED, format_policy="punctuation"
    ) == REPAIRED
    assert normalize_bbox_completion(MALFORMED) == MALFORMED


def test_recognizes_only_the_two_explicit_coordinate_pair_spellings():
    assert normalize_bbox_completion(
        '{"bbox_2d":[(593,362),(670,416)]}', format_policy="punctuation"
    ) == REPAIRED
    assert normalize_bbox_completion(
        '{"bbox_2d":[[593,362],[670,416]]}', format_policy="punctuation"
    ) == REPAIRED


def test_valid_json_is_returned_byte_for_byte_unchanged():
    valid = ' { "bbox_2d" : [593, 362, 670, 416] } <|im_end|> '
    assert normalize_bbox_completion(valid, format_policy="punctuation") == valid


def test_thinking_text_and_completion_wrappers_are_preserved():
    reasoning = "<think>inspect {bbox_2d}; keep these exact bytes\n"
    completion = reasoning + "</think>\n" + MALFORMED + "\n<|im_end|>"
    normalized = normalize_bbox_completion(
        completion, answer_mode="thinking", format_policy="punctuation"
    )
    assert normalized == reasoning + "</think>\n" + REPAIRED + "\n<|im_end|>"
    assert normalized.startswith(reasoning)


def test_thinking_requires_one_valid_boundary_and_does_not_repair_reasoning():
    invalid = [
        "<think>" + MALFORMED,  # missing final-answer boundary
        "first</think>" + MALFORMED + "</think>",
        "<think>first</think>second</think>" + MALFORMED,
        "<think>first</think>" + MALFORMED + " <think>extra",
        "<think>first</think> " + MALFORMED + " trailing text",
    ]
    for completion in invalid:
        assert normalize_bbox_completion(
            completion, answer_mode="thinking", format_policy="punctuation"
        ) == completion


def test_fullmatch_rejects_extra_schema_objects_and_text():
    invalid = [
        '{"bbox_2d":[593,362),(670,416],"label":"person"}',
        MALFORMED + '{"bbox_2d":[593,362,670,416]}',
        "prefix " + MALFORMED,
        MALFORMED + " suffix",
    ]
    for completion in invalid:
        assert normalize_bbox_completion(completion, format_policy="punctuation") == completion


def test_requires_exactly_four_json_numbers_and_a_valid_bbox():
    invalid = [
        '{"bbox_2d":[(593,362),(670)]}',
        '{"bbox_2d":[(593,362),(1001,416)]}',
        '{"bbox_2d":[(670,416),(593,362)]}',
        '{"bbox_2d":[("593",362),(670,416)]}',
    ]
    for completion in invalid:
        assert normalize_bbox_completion(completion, format_policy="punctuation") == completion


def test_short_mode_does_not_repair_thinking_wrappers():
    completion = "<think>reason</think>" + MALFORMED
    assert normalize_bbox_completion(completion, format_policy="punctuation") == completion


def test_unknown_policy_fails_fast():
    try:
        normalize_bbox_completion(MALFORMED, format_policy="repair-everything")
    except ValueError as exc:
        assert "unknown format policy" in str(exc)
    else:
        raise AssertionError("unknown format policy did not fail")


def test_single_box_normalizes_a_real_fenced_singleton_label_answer():
    reasoning = "Inspect the requested object; these bytes stay unchanged.\n"
    final = """```json
[
\t{"bbox_2d": [142, 479, 270, 758], "label": "facial cleanser in a silvery and red conical bottled on the left"}
]
```"""
    completion = reasoning + "</think>\n\n" + final + "<|im_end|>"
    expected = reasoning + "</think>\n\n{\"bbox_2d\":[142,479,270,758]}<|im_end|>"
    assert normalize_bbox_completion(
        completion, answer_mode="thinking", format_policy="single_box"
    ) == expected


def test_single_box_preserves_bare_json_and_only_normalizes_final_answer():
    reasoning = '<think>reason contains {"bbox_2d":[0,0,1,1]} literally</think>\n'
    completion = reasoning + '[{"bbox_2d":[0,0,1000,1000]}]'
    assert normalize_bbox_completion(
        completion, answer_mode="thinking", format_policy="single_box"
    ) == reasoning + '{"bbox_2d":[0,0,1000,1000]}'

    no_boundary = '<think>{"bbox_2d":[0,0,1000,1000]}'
    assert normalize_bbox_completion(
        no_boundary, answer_mode="thinking", format_policy="single_box"
    ) == no_boundary


def test_single_box_rejects_ambiguous_or_invalid_final_payloads():
    invalid = [
        '[{"bbox_2d":[0,0,500,1000]},{"bbox_2d":[500,0,1000,1000]}]',
        '{"bbox_2d":[0,0,1000,1000],"extra":true}',
        '{"bbox_2d":[0,0,1000,1000],"bbox_2d":[0,0,500,1000]}',
        '{"bbox_2d":[0,0,1001,1000]}',
        '{"bbox_2d":[0,0,1000,1000],"label":7}',
        '```json\n{"bbox_2d":[0,0,1000,1000]}\n``` trailing',
    ]
    for final in invalid:
        completion = f"<think>reason</think>{final}"
        assert normalize_bbox_completion(
            completion, answer_mode="thinking", format_policy="single_box"
        ) == completion


def test_single_box_requires_one_boundary_and_legacy_punctuation_stays_separate():
    wrapped = '<think>reason</think>```json\n{"bbox_2d":[593,362),(670,416]}\n```'
    assert normalize_bbox_completion(
        wrapped, answer_mode="thinking", format_policy="single_box"
    ) == wrapped
    assert normalize_bbox_completion(
        MALFORMED, format_policy="punctuation"
    ) == REPAIRED
    invalid_boundary = '<think>first</think>again</think>```json\n{"bbox_2d":[0,0,1000,1000]}\n```'
    assert normalize_bbox_completion(
        invalid_boundary, answer_mode="thinking", format_policy="single_box"
    ) == invalid_boundary
