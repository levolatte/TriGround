import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from tools.generate_grounding_candidates import (
    build_request, candidate_from_output, load_cache, load_images, load_queries, prompt_for,
)


def make_images(root: Path):
    Image.new("RGB", (80, 40), (255, 0, 0)).save(root / "rgb.png")
    Image.new("RGB", (80, 40), (0, 255, 0)).save(root / "ir.png")
    Image.fromarray(np.full((40, 80), 1000, dtype=np.uint16)).save(root / "depth.png")


def test_official_inputs_ignore_labels_and_keep_modality_order(tmp_path):
    make_images(tmp_path)
    path = tmp_path / "queries.json"
    path.write_text(json.dumps({"q1": {"query": "left person", "visible": "rgb.png",
        "infrared": "ir.png", "depth": "depth.png", "bbox": [0.9, 0.9, 1, 1]}}))
    rows = load_queries(path, tmp_path, "trimodal")
    assert rows[0]["modalities"] == ["rgb", "infrared", "depth"]
    assert "bbox" not in rows[0]
    assert "0.9" not in prompt_for(rows[0], "qwen")
    images = load_images(rows[0], 1)
    assert images[0].getpixel((0, 0)) == (255, 0, 0)
    assert images[1].getpixel((0, 0)) == (0, 255, 0)
    assert images[2].getpixel((0, 0))[0] > 0
    with pytest.raises(ValueError, match="max-pixels"):
        load_images(rows[0], 2, 3200)


def test_sft_input_for_egm_removes_assistant_and_aux_images(tmp_path):
    make_images(tmp_path)
    path = tmp_path / "val.jsonl"
    path.write_text(json.dumps({"id": "r1", "images": ["rgb.png", "depth.png"],
        "modalities": ["rgb", "depth"], "query": "cup, near the blue plate",
        "messages": [{"role": "user", "content": "<image><image>unused"},
                     {"role": "assistant", "content": "SECRET_GROUND_TRUTH"}]}) + "\n")
    row = load_queries(path, tmp_path, "rgb")[0]
    calls = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls.append((messages, kwargs))
            return "template"

    request = build_request(row, Tokenizer(), "egm", 1)
    assert len(request["multi_modal_data"]["image"]) == 1
    assert calls[0][1]["enable_thinking"] is True
    assert len(calls[0][0]) == 1
    assert "cup, near the blue plate" in calls[0][0][0]["content"][-1]["text"]
    assert "SECRET_GROUND_TRUTH" not in str(calls)


def test_final_box_after_thinking_and_unfinished_reasoning():
    text = '<think>{"bbox_2d":[0,0,100,100]}</think>{"bbox_2d":[200,300,600,800]}'
    output = SimpleNamespace(index=0, text=text, token_ids=[1, 2], finish_reason="stop", cumulative_logprob=-4.0)
    assert candidate_from_output(output, "egm")["bbox"] == [0.2, 0.3, 0.6, 0.8]
    output.text = '<think>{"bbox_2d":[0,0,100,100]} still reasoning'
    output.finish_reason = "length"
    assert candidate_from_output(output, "egm")["bbox"] is None
    output.text = '<think>{"bbox_2d":[0,0,100,100]}</think>No final answer'
    output.finish_reason = "stop"
    assert candidate_from_output(output, "egm")["bbox"] is None


def test_resume_detects_changed_query_and_prompt(tmp_path):
    row = {"id": "1", "query": "cup", "image_paths": {"rgb": "rgb.png"},
           "modalities": ["rgb"], "depth_policy": "visual"}
    cache = tmp_path / "candidates.jsonl"
    cached = {**row, "prompt": prompt_for(row, "qwen"), "candidates": []}
    cache.write_text(json.dumps(cached) + "\n")
    assert load_cache(cache, [row], "qwen")["1"]["query"] == "cup"
    with pytest.raises(ValueError, match="query changed"):
        load_cache(cache, [{**row, "query": "bowl"}], "qwen")


def test_rgbt_converted_manifest_keeps_real_ir_input(tmp_path):
    make_images(tmp_path)
    path = tmp_path / "rgbt.jsonl"
    path.write_text(json.dumps({"id": "rgbt_1", "rgb": "rgb.png", "aux": "ir.png",
                               "aux_type": "ir", "query": "the distant person"}) + "\n")
    row = load_queries(path, tmp_path, "all")[0]
    assert row["modalities"] == ["rgb", "infrared"]
    assert list(row["image_paths"]) == ["rgb", "infrared"]


def test_raw_unknown_depth_is_not_silently_assumed_millimeters(tmp_path):
    make_images(tmp_path)
    row = {"id": "robo_1", "query": "the nearer cup", "modalities": ["rgb", "depth"],
           "depth_policy": "visual", "image_paths": {"rgb": str(tmp_path / "rgb.png"),
                                                       "depth": str(tmp_path / "depth.png")}}
    with pytest.raises(ValueError, match="depth"):
        load_images(row, 1)
