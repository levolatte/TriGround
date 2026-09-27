from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tools.evaluate_pretrained_grounder import (
    apply_target_manifest,
    build_prompt,
    build_user_content,
    build_run_config,
    ensure_jsonl_append_separator,
    load_records,
    main,
    normalize_record,
    parse_generated_bbox,
    resolve_image_path,
    score_prediction,
    summarize_rows,
    validate_resume_rows,
    verify_run_config,
)


def test_parse_generated_bbox_uses_final_json_box():
    text = (
        'A rough proposal is {"bbox_2d":[20,30,40,50]}. '
        'After checking the image, {"bbox_2d":[100,200,900,800]}'
    )
    assert parse_generated_bbox(text) == [0.1, 0.2, 0.9, 0.8]


def test_parse_generated_bbox_rejects_missing_and_invalid_boxes():
    assert parse_generated_bbox("No bounding box was found.") is None
    assert parse_generated_bbox('{"bbox_2d":[900,200,100,800]}') is None
    assert parse_generated_bbox('{"bbox_2d":[100,200,1001,800]}') is None
    assert (
        parse_generated_bbox(
            '{"bbox_2d":[100,200,900,800]} then {"bbox_2d":[900,200,100,800]}'
        )
        is None
    )


def test_standard_qwen_record_separates_query_from_assistant_ground_truth():
    record = {
        "images": ["external_data/city_detection_prepared/train/visible/a.png"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": "Locate the red bicycle, output its bbox coordinates using JSON format",
                    },
                ],
            },
            {"role": "assistant", "content": '{"bbox_2d":[100,200,900,800]}'},
        ],
    }
    normalized = normalize_record("sample-1", record)
    assert normalized["query"] == "the red bicycle"
    assert normalized["bbox"] == [0.1, 0.2, 0.9, 0.8]
    assert "bbox_2d" not in build_prompt(normalized["query"], "egm")


def test_native_prompt_keeps_json_and_query_braces_literal():
    prompt = build_prompt("the sign marked {A}", "native")
    assert 'referring expression: "the sign marked {A}"' in prompt
    assert '{"bbox_2d":[x1,y1,x2,y2]}' in prompt


def test_loads_new_dictionary_manifest_and_resolves_project_prefixed_path(tmp_path):
    manifest = tmp_path / "val.json"
    manifest.write_text(
        json.dumps(
            {
                "city_001": {
                    "visible": "external_data/city_detection_prepared/train/visible/a.png",
                    "query": "the sign",
                    "bbox": [0.1, 0.2, 0.8, 0.9],
                }
            }
        ),
        encoding="utf-8",
    )
    records = load_records(manifest)
    data_root = tmp_path / "project"
    resolved = resolve_image_path(records[0]["images"][0], data_root, manifest)
    assert records[0]["id"] == "city_001"
    assert resolved == (
        data_root / "external_data/city_detection_prepared/train/visible/a.png"
    ).resolve()


def test_native_sft_manifest_keeps_prompt_gt_isolated_and_image_order(tmp_path):
    prompt = (
        "<image>\n<image>\n<image>\n"
        "These are RGB, infrared, and depth. Locate the object marked {A}.\n"
        'Return only {"bbox_2d":[x1,y1,x2,y2]}'
    )
    record = {
        "id": "tri-1",
        "image": ["visible/a.png", "infrared/a.png", "depth_rgb/a.png"],
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": '{"bbox_2d":[100,200,900,800]}'},
        ],
    }
    manifest = tmp_path / "trimodal_val.json"
    manifest.write_text(json.dumps([record]), encoding="utf-8")

    normalized = load_records(manifest)[0]
    assert normalized["prompt"] == prompt
    assert normalized["images"] == record["image"]
    assert normalized["bbox"] == [0.1, 0.2, 0.9, 0.8]
    assert "[100,200,900,800]" not in normalized["prompt"]

    content = build_user_content(
        normalized["prompt"],
        normalized["images"],
        prompt_has_image_placeholders=True,
    )
    assert [item["image"] for item in content if item["type"] == "image"] == record["image"]
    assert "".join(item["text"] for item in content if item["type"] == "text") == (
        prompt.replace("<image>", "")
    )

    data_root = tmp_path / "city" / "train"
    assert resolve_image_path(
        normalized["images"][0],
        data_root,
        manifest,
        from_data_root=normalized["images_from_data_root"],
    ) == (data_root / "visible/a.png").resolve()


def test_parse_failure_is_retained_and_scored_as_wrong():
    target = [0.1, 0.2, 0.9, 0.8]
    iou, accurate = score_prediction(None, target)
    assert (iou, accurate) == (0.0, False)
    rows = [
        {
            "prediction": target,
            "target": target,
            "parsed": True,
            "latency_seconds": 1.0,
            "generated_tokens": 10,
            "generation_cap_hit": False,
        },
        {
            "prediction": None,
            "target": target,
            "parsed": False,
            "latency_seconds": 2.0,
            "generated_tokens": 20,
            "generation_cap_hit": True,
        },
    ]
    summary = summarize_rows(rows)
    assert summary["samples"] == 2
    assert summary["parse_failures"] == 1
    assert summary["parse_rate"] == 0.5
    assert summary["acc_0.5"] == 0.5
    assert summary["hits"] == 1


def test_truncated_jsonl_record_is_an_explicit_error(tmp_path):
    manifest = tmp_path / "broken.jsonl"
    manifest.write_text(
        '{"id":"ok","image":"a.png","query":"the sign","bbox":[0.1,0.2,0.8,0.9]}\n'
        '{"id":"truncated"',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="trailing/incomplete"):
        load_records(manifest)


def test_target_manifest_overrides_sft_answer_and_requires_selected_ids(tmp_path):
    sft = tmp_path / "sft.json"
    sft.write_text(
        json.dumps(
            [
                {
                    "id": "a",
                    "image": ["visible/a.png"],
                    "conversations": [
                        {"from": "human", "value": "<image> locate sign"},
                        {"from": "gpt", "value": '{"bbox_2d":[100,200,900,800]}'},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    target = tmp_path / "raw.json"
    target.write_text(
        json.dumps(
            {
                "a": {"bbox": [0.25, 0.3, 0.7, 0.8]},
                "extra": {"bbox": [0.1, 0.1, 0.2, 0.2]},
            }
        ),
        encoding="utf-8",
    )
    records = load_records(sft)
    apply_target_manifest(records, target)
    assert records[0]["bbox"] == [0.25, 0.3, 0.7, 0.8]
    assert records[0]["target_source"] == str(target.resolve())

    missing = tmp_path / "missing.json"
    missing.write_text(json.dumps({"other": {"bbox": [0.1, 0.1, 0.2, 0.2]}}), encoding="utf-8")
    with pytest.raises(ValueError, match="does not completely cover"):
        apply_target_manifest(load_records(sft), missing)


def test_resume_rows_are_checked_by_id_and_target():
    records = [
        {"id": "a", "bbox": [0.1, 0.2, 0.8, 0.9]},
        {"id": "b", "bbox": [0.2, 0.3, 0.7, 0.8]},
    ]
    rows = [
        {"id": "b", "target": records[1]["bbox"], "prediction": None},
    ]
    existing = validate_resume_rows(rows, records)
    assert list(existing) == ["b"]
    with pytest.raises(ValueError, match="target mismatch"):
        validate_resume_rows(
            [{"id": "b", "target": [0.1, 0.3, 0.7, 0.8]}],
            records,
        )


def test_run_config_captures_precision_and_resume_mismatch():
    config = build_run_config(
        model="model",
        adapter=None,
        manifest=Path("manifest.json"),
        target_manifest=None,
        data_root=Path("data"),
        min_pixels=200704,
        max_pixels=602112,
        max_new_tokens=96,
        prompt_style="native",
    )
    assert config["min_pixels"] == 200704
    assert config["max_new_tokens"] == 96
    assert config["prompt_style"] == "native"
    assert config["tf32"] is False
    assert config["matmul_precision"] == "highest"
    verify_run_config(config, dict(config))
    changed = dict(config)
    changed["max_pixels"] += 1
    with pytest.raises(ValueError, match="run_config mismatch"):
        verify_run_config(config, changed)


def test_complete_resume_reuses_rows_and_writes_summary_without_cuda(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "a": {
                    "visible": "a.png",
                    "query": "the sign",
                    "bbox": [0.1, 0.2, 0.8, 0.9],
                }
            }
        ),
        encoding="utf-8",
    )
    target_manifest = tmp_path / "targets.json"
    target_manifest.write_text(
        json.dumps({"a": {"bbox": [0.2, 0.25, 0.75, 0.85]}}),
        encoding="utf-8",
    )
    output = tmp_path / "run"
    output.mkdir()
    config = build_run_config(
        model="model",
        adapter=None,
        manifest=manifest.resolve(),
        target_manifest=target_manifest.resolve(),
        data_root=tmp_path.resolve(),
        min_pixels=200704,
        max_pixels=602112,
        max_new_tokens=96,
        prompt_style="egm",
    )
    (output / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    row = {
        "id": "a",
        "target": [0.2, 0.25, 0.75, 0.85],
        "prediction": [0.2, 0.25, 0.75, 0.85],
        "parsed": True,
        "iou": 1.0,
        "acc_0.5": True,
        "latency_seconds": 0.1,
        "generated_tokens": 4,
        "generation_cap_hit": False,
    }
    (output / "predictions.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_pretrained_grounder.py",
            "--model",
            "model",
            "--manifest",
            str(manifest),
            "--target-manifest",
            str(target_manifest),
            "--data-root",
            str(tmp_path),
            "--output-dir",
            str(output),
            "--max-new-tokens",
            "96",
            "--resume",
        ],
    )
    main()
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["hits"] == 1
    assert summary["target_source"] == str(target_manifest.resolve())


def test_resume_append_separates_complete_jsonl_without_final_newline(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text('{"id":"a"}', encoding="utf-8")
    with path.open("a", encoding="utf-8") as handle:
        ensure_jsonl_append_separator(path, handle)
        handle.write('{"id":"b"}\n')
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2
