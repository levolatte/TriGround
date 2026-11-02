import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from tools import predict_vllm_quant_submission as pv


def test_request_preserves_native_prompt_images_and_disables_thinking(tmp_path):
    Image.new("RGB", (4, 3), "red").save(tmp_path / "rgb.png")
    captured = []

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            captured.append((messages, kwargs))
            return "native prompt"

    requests = pv.build_requests([{"id": "a", "images": ["rgb.png"],
                                   "messages": [{"role": "user", "content": "before<image>after"}]}],
                                 tmp_path, Processor())
    assert requests[0]["prompt"] == "native prompt"
    assert [part["type"] for part in captured[0][0][0]["content"]] == ["text", "image", "text"]
    assert captured[0][1] == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert requests[0]["multi_modal_data"]["image"][0].getpixel((0, 0)) == (255, 0, 0)


def test_chunked_resume_appends_full_batches_and_preserves_completed_bytes(tmp_path, monkeypatch):
    args = SimpleNamespace(output_dir=tmp_path, data_root=tmp_path, resume=True, chunk_size=2)
    saved = {"id": "a", "samples": [{"text": "old", "finish_reason": "stop"}]}
    raw = tmp_path / "predictions_raw.jsonl"
    original = (json.dumps(saved) + "\n").encode()
    raw.write_bytes(original)
    dataset = [{"id": key} for key in "abcd"]
    monkeypatch.setattr(pv, "build_requests",
                        lambda rows, root, processor, upscale=1.0: [row["id"] for row in rows])
    calls = []

    class LLM:
        def generate(self, requests, sampling, **kwargs):
            calls.append(requests)
            return [SimpleNamespace(prompt_token_ids=[1], outputs=[SimpleNamespace(
                text=key, token_ids=[2], finish_reason="stop")]) for key in requests]

    result = pv.run_chunks(args, dataset, [saved], LLM(), None, None, None)
    assert calls == [["b", "c"], ["d"]]
    assert raw.read_bytes().startswith(original)
    assert [row["id"] for row in pv.read_jsonl(raw)] == list("abcd")
    assert (result["completed_queries"], result["resumed_queries"], result["new_queries"]) == (4, 1, 3)


def test_nf4_options_keep_only_mandatory_quantization_invariants():
    config = dict(model="base", max_length=8192, seed=2026, min_pixels=200704, max_pixels=1204224)
    options = pv.build_llm_options(config)
    assert options["max_num_seqs"] == 8
    assert options["quantization"] == options["load_format"] == "bitsandbytes"
    assert options["hf_overrides"]["quantization_config"]["llm_int8_skip_modules"] == ["visual", "lm_head", "in_proj_ba"]
    # 性能开关可以是任意已验证组合，不能当成不变量。
    tuned = pv.build_llm_options({**config, "enforce_eager": False,
                                  "enable_prefix_caching": True,
                                  "max_num_batched_tokens": 32768})
    assert tuned["enforce_eager"] is False
    assert tuned["enable_prefix_caching"] is True
    assert tuned["max_num_batched_tokens"] == 32768
    assert tuned["quantization"] == "bitsandbytes"


def test_upscale_requires_matching_pixel_budget():
    parser = pv.build_parser()
    base = ["--dataset", "d", "--data-root", "r", "--model", "m", "--adapter", "a", "--output-dir", "o"]
    with pytest.raises(SystemExit):
        parser.parse_args(base)  # 分辨率三参数已无默认值，漏传即拒绝启动
    args = parser.parse_args(base + ["--max-pixels", "1204224", "--max-length", "8192", "--upscale", "2.0"])
    with pytest.raises(ValueError, match="upscale"):
        pv.main([*base, "--max-pixels", "1204224", "--max-length", "8192", "--upscale", "2.0"])
    assert args.upscale == 2.0


BASE_SUBMISSION = ["--dataset", "d", "--data-root", "r", "--model", "m", "--adapter", "a",
                   "--output-dir", "o", "--max-pixels", "8294400", "--max-length", "32768",
                   "--upscale", "2.0"]


def test_multi_sample_requires_nonzero_temperature():
    """温度 0 采 k 个样本只会得到 k 份相同结果，聚合等于没做——必须拒绝而不是静默浪费机器。"""
    with pytest.raises(ValueError, match="temperature > 0"):
        pv.main(BASE_SUBMISSION + ["--samples", "5", "--temperature", "0"])


def test_multi_sample_without_temperature_uses_legacy_default():
    """不传温度时沿用旧默认：k>1 用 1。这是评测工具的口径，两者必须一致。"""
    args = pv.build_parser().parse_args(BASE_SUBMISSION + ["--samples", "5"])
    assert args.samples == 5
    assert args.temperature is None  # 生效值在 main 内推导，None 表示「按 k 取默认」


def test_single_sample_rejects_nonzero_temperature():
    """单样本 + 非零温度不可复现，不能进正式提交。"""
    with pytest.raises(ValueError, match="not reproducible"):
        pv.main(BASE_SUBMISSION + ["--temperature", "0.5"])


def test_multi_sample_rejects_resume():
    """样本数无法从半截写的 raw 文件恢复，所以多采样禁止续跑。"""
    with pytest.raises(ValueError, match="resume only supports"):
        pv.main(BASE_SUBMISSION + ["--samples", "5", "--temperature", "0.5", "--resume"])


def test_out_of_range_temperature_is_rejected():
    with pytest.raises(ValueError, match="within"):
        pv.main(BASE_SUBMISSION + ["--samples", "5", "--temperature", "3"])


def test_run_chunks_writes_all_samples_and_keeps_single_sample_shape(tmp_path, monkeypatch):
    """k 个样本必须全部写出；k=1 时每题的 samples 长度仍为 1，打包路径不受影响。"""
    from types import SimpleNamespace

    monkeypatch.setattr(pv, "build_requests", lambda rows, root, processor, upscale=1.0: list(rows))

    def make_llm(samples):
        class LLM:
            def generate(self, requests, sampling, **kwargs):
                return [SimpleNamespace(prompt_token_ids=[1], outputs=[
                    SimpleNamespace(text=f"{sample_id}#{i}", token_ids=[2], finish_reason="stop")
                    for i in range(samples)]) for sample_id in requests]
        return LLM()

    def run(samples, tag):
        output = tmp_path / tag
        output.mkdir()
        args = SimpleNamespace(output_dir=output, data_root=tmp_path, resume=False,
                               chunk_size=2, samples=samples)
        result = pv.run_chunks(args, [{"id": key} for key in "ab"], [], make_llm(samples), None, None, None)
        rows = [json.loads(line) for line in (output / "predictions_raw.jsonl").read_text().splitlines()]
        return result, rows

    result_one, rows_one = run(1, "single")
    assert [len(row["samples"]) for row in rows_one] == [1, 1]
    assert result_one["samples_per_query"] == 1

    result_five, rows_five = run(5, "multi")
    assert [len(row["samples"]) for row in rows_five] == [5, 5]
    assert result_five["samples_per_query"] == 5
    # monkeypatch 让 build_requests 直接返回行对象，所以假 LLM 收到的 sample_id 是 dict。
    assert rows_five[0]["samples"][3]["text"].endswith("#3")


def test_run_chunks_rejects_wrong_sample_count(tmp_path, monkeypatch):
    """vLLM 返回的样本数与请求不符必须立刻报错，不能写出半截数据。"""
    from types import SimpleNamespace

    monkeypatch.setattr(pv, "build_requests", lambda rows, root, processor, upscale=1.0: list(rows))

    class ShortLLM:
        def generate(self, requests, sampling, **kwargs):
            return [SimpleNamespace(prompt_token_ids=[1], outputs=[
                SimpleNamespace(text="only", token_ids=[2], finish_reason="stop")]) for _ in requests]

    args = SimpleNamespace(output_dir=tmp_path, data_root=tmp_path, resume=False,
                           chunk_size=2, samples=3)
    with pytest.raises(RuntimeError, match="expected 3 completions"):
        pv.run_chunks(args, [{"id": "a"}], [], ShortLLM(), None, None, None)

