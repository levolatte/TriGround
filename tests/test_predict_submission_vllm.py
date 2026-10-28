"""`predict_submission_vllm` 两阶段流程的回归测试。

存在的直接原因：阶段2 重跑曾把 `build_requests` 的返回值（item 字典）当成 items 再传回去，
抛出 "too many values to unpack (expected 3)"，进程崩溃，**86 分钟推理结果全部丢失**。
这类错误必须在花 GPU 时间之前就被测试挡住。

测试不碰 vLLM：用一个假 llm 与假 tokenizer，只验证**数据流向与形状**。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import predict_submission_vllm as module  # noqa: E402
from tools.predict_submission_vllm import accept, build_requests  # noqa: E402


class FakeTokenizer:
    def apply_chat_template(self, messages, **_kwargs):
        # 必须把查询文本带进渲染结果，否则批量生成时无法按查询区分行为，
        # 阶段2 的重跑支路就测不到。
        parts = []
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            else:
                parts.extend(str(item.get("text", "")) for item in content
                             if isinstance(item, dict))
        return "<fake>" + " ".join(parts) + "<think>\n\n</think>\n"


def make_dataset(root: Path, count: int) -> tuple[Path, Path]:
    """造 count 条官方格式查询，三模态各一张小图。"""
    images = root / "Images"
    for modality in ("visible", "infrared", "depth"):
        (images / modality).mkdir(parents=True, exist_ok=True)
        for index in range(count):
            Image.new("RGB", (64, 48), (index * 7 % 255, 90, 40)).save(
                images / modality / f"{index:06d}.png")
    queries = {
        f"q{index:04d}": {
            "visible": f"Images/visible/{index:06d}.png",
            "infrared": f"Images/infrared/{index:06d}.png",
            "depth": f"Images/depth/{index:06d}.png",
            "query": f"object number {index}",
        }
        for index in range(count)
    }
    query_path = root / "queries.json"
    query_path.write_text(json.dumps(queries), encoding="utf-8")
    return query_path, root


def test_build_requests_accepts_triples(tmp_path: Path) -> None:
    query_path, root = make_dataset(tmp_path, 3)
    source = json.loads(query_path.read_text(encoding="utf-8"))
    items = [(index, sample_id, record)
             for index, (sample_id, record) in enumerate(source.items())]
    requests = build_requests(FakeTokenizer(), items, root, query_path, "trimodal")
    assert len(requests) == 3
    assert [item["id"] for item in requests] == list(source)
    assert all(len(item["request"]["multi_modal_data"]["image"]) == 3 for item in requests)


def test_sensor_linear_depth_prompt_uses_unknown_units_without_label_leak(tmp_path: Path) -> None:
    query_path, root = make_dataset(tmp_path, 1)
    source = json.loads(query_path.read_text(encoding="utf-8"))
    record = source["q0000"]
    record["bbox"] = [0.11, 0.22, 0.33, 0.44]
    record["evidence"] = "EVIDENCE_SECRET"
    source["q0000"] = record
    query_path.write_text(json.dumps(source), encoding="utf-8")
    items = [(0, "q0000", record)]

    current_prompt = build_requests(FakeTokenizer(), items, root, query_path, "trimodal")[0]["prompt"]
    sensor_prompt = build_requests(
        FakeTokenizer(), items, root, query_path, "trimodal",
        depth_policy="sensor_linear_20000",
    )[0]["prompt"]

    assert "brighter valid pixels are nearer" in current_prompt
    assert "physical distance unit is not established" not in current_prompt
    assert "fixed inverse mapping" in sensor_prompt
    assert "physical distance unit is not established" in sensor_prompt
    assert "object number 0" in sensor_prompt
    assert "EVIDENCE_SECRET" not in sensor_prompt
    assert str(record["bbox"]) not in sensor_prompt


def test_build_requests_rejects_item_dicts(tmp_path: Path) -> None:
    """把 build_requests 的返回值再传回去，必须给出**明确**报错而不是 unpack 崩溃。"""
    query_path, root = make_dataset(tmp_path, 2)
    source = json.loads(query_path.read_text(encoding="utf-8"))
    items = [(index, sample_id, record)
             for index, (sample_id, record) in enumerate(source.items())]
    requests = build_requests(FakeTokenizer(), items, root, query_path, "trimodal")
    with pytest.raises(TypeError, match="三元组"):
        build_requests(FakeTokenizer(), requests, root, query_path, "trimodal")


def test_retry_round_trips_through_build_requests(tmp_path: Path) -> None:
    """模拟阶段2：重跑项存的是原始元组，因此能再次交给 build_requests。"""
    query_path, root = make_dataset(tmp_path, 4)
    source = json.loads(query_path.read_text(encoding="utf-8"))
    items = [(index, sample_id, record)
             for index, (sample_id, record) in enumerate(source.items())]

    # 第一轮：前两条解析失败 → 记入 retry（存元组，不是 item 字典）
    retry = [entry for entry in items[:2]]
    assert all(len(entry) == 3 for entry in retry)
    requests = build_requests(FakeTokenizer(), retry, root, query_path, "trimodal")
    assert [item["id"] for item in requests] == [entry[1] for entry in retry]


@pytest.mark.parametrize(
    ("text", "ok"),
    [
        ('{"bbox_2d":[10,20,30,40]}', True),
        ('[{"bbox_2d": [476, 590, 565, 920], "label": "x"}]', True),
        ('{"bbox_2d":[10,20,30', False),          # 未写完
        ('{"bbox_2d":[-5, 20, 30, 40]}', False),  # 越界，解析器拒绝
        ('thinking about the image...', False),   # 思维链
    ],
)
def test_accept_matches_parser(text: str, ok: bool) -> None:
    assert (accept(text) is not None) is ok


def test_module_exposes_lora_and_chunking_flags() -> None:
    """命令行必须暴露这几个参数，否则脚本化的提交命令会静默用错配置。"""
    source = Path(module.__file__).read_text(encoding="utf-8")
    for flag in ("--adapter", "--max-lora-rank", "--chunk-size", "--mamba-cache-mode",
                 "--max-num-seqs"):
        assert flag in source, flag


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeLoRARequest:
    def __init__(self, name, lid, path):
        self.name, self.lora_id, self.lora_int_id, self.path = name, lid, lid, path


class FakeOutput:
    def __init__(self, text: str, token_ids, prompt_ids, finish_reason="stop"):
        self.outputs = [type("O", (), {"text": text, "token_ids": token_ids,
                                       "finish_reason": finish_reason})()]
        self.prompt_token_ids = prompt_ids


class FakeLLM:
    """按 prompt 里是否含特定标记决定输出，用来同时覆盖阶段1 成功与失败两条支路。"""

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.instances.append(self)

    def get_tokenizer(self):
        return FakeTokenizer()

    def generate(self, requests, sampling_params, **_kwargs):
        self.calls.append({"request_count": len(requests),
                           "sampling_params": sampling_params.kwargs})
        outputs = []
        for request in requests:
            text = request["prompt"]
            # 让大部分成功；含 "object number 1" 的故意给不可解析输出
            if "object number 1" in text:
                outputs.append(FakeOutput("thinking about it", [1, 2], [0]))
            else:
                outputs.append(FakeOutput('{"bbox_2d":[10,20,30,40]}', [1, 2, 3], [0]))
        return outputs


def install_fake_vllm(monkeypatch) -> None:
    import types

    fake_llm_module = types.ModuleType("vllm")
    fake_llm_module.LLM = FakeLLM
    fake_llm_module.SamplingParams = FakeSamplingParams
    fake_lora_module = types.ModuleType("vllm.lora")
    fake_lora_request = types.ModuleType("vllm.lora.request")
    fake_lora_request.LoRARequest = FakeLoRARequest
    monkeypatch.setitem(sys.modules, "vllm", fake_llm_module)
    monkeypatch.setitem(sys.modules, "vllm.lora", fake_lora_module)
    monkeypatch.setitem(sys.modules, "vllm.lora.request", fake_lora_request)


def test_main_end_to_end_writes_ordered_rows(tmp_path: Path, monkeypatch) -> None:
    """整条 main() 跑通：两条支路（阶段1 成功 + 阶段2 重跑）都要走到，且顺序正确。

    这条是被连续三次真实故障逼出来的——argparse 选项重复、retry 里存了错类型、
    adapter_path 未赋值，全都是"改完没跑就上云端"，每次都白丢几十分钟。
    """
    query_path, root = make_dataset(tmp_path, 6)
    out_dir = tmp_path / "out"
    install_fake_vllm(monkeypatch)
    monkeypatch.setattr(sys, "argv", [
        "predict_submission_vllm", "--model", "fake", "--modality", "trimodal",
        "--queries", str(query_path), "--data-root", str(root),
        "--expected-query-count", "6", "--chunk-size", "2", "--max-num-seqs", "4",
        "--output-dir", str(out_dir)])

    module.main()

    rows = [json.loads(line) for line in
            (out_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 6
    expected_ids = list(json.loads(query_path.read_text(encoding="utf-8")))
    assert [row["id"] for row in rows] == expected_ids, "追加写入必须保持官方 QueryID 顺序"
    # "object number 1" 在阶段1 失败，应落到阶段2
    stages = {row["id"]: row["stage"] for row in rows}
    assert stages["q0001"] == 2
    assert stages["q0000"] == 1
    assert (out_dir / "predictions.json").exists(), "完整时应自动打包"
    assert (out_dir / "submission.zip").exists()


def test_main_single_pass_generates_once_and_records_legal_fallback(
        tmp_path: Path, monkeypatch) -> None:
    query_path, root = make_dataset(tmp_path, 3)
    out_dir = tmp_path / "single-pass"
    FakeLLM.instances.clear()
    install_fake_vllm(monkeypatch)
    original_generate = FakeLLM.generate

    def generate_with_bare_box(self, requests, sampling_params, **kwargs):
        outputs = original_generate(self, requests, sampling_params, **kwargs)
        outputs[2] = FakeOutput('[100,200,300,400]', [1, 2, 3], [0])
        return outputs

    monkeypatch.setattr(FakeLLM, 'generate', generate_with_bare_box)
    monkeypatch.setattr(sys, "argv", [
        "predict_submission_vllm", "--model", "fake", "--modality", "trimodal",
        "--queries", str(query_path), "--data-root", str(root),
        "--expected-query-count", "3", "--chunk-size", "3", "--max-num-seqs", "4",
        "--output-dir", str(out_dir), "--single-pass"])

    module.main()

    rows = [json.loads(line) for line in
            (out_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    llm = FakeLLM.instances[-1]
    assert len(llm.calls) == 1
    assert llm.calls[0]["request_count"] == 3
    assert llm.calls[0]["sampling_params"] == {
        "temperature": 0.0, "max_tokens": module.MAX_NEW_TOKENS,
        "skip_special_tokens": True,
    }
    assert [row["stage"] for row in rows] == ["single_pass"] * 3
    failed = next(row for row in rows if row["id"] == "q0001")
    assert failed["parsed"] is False and failed["fallback"] is True
    assert failed["bbox"] == module.FALLBACK_BOX
    bare = next(row for row in rows if row["id"] == "q0002")
    assert bare["parsed"] is True and bare["fallback"] is False
    assert bare["bbox"] == [0.1, 0.2, 0.3, 0.4]
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["generation_mode"] == "single_pass"
    assert summary["single_pass_fallbacks"] == 1
    assert "stage2_retries" not in summary
    monkeypatch.setattr(sys, "argv", sys.argv + ["--package-only"])
    module.main()
    assert len(llm.calls) == 1


def test_main_package_only_round_trip(tmp_path: Path, monkeypatch) -> None:
    """--package-only 不加载模型，但同样要能读到 config（曾因 adapter_path 未赋值而崩）。"""
    query_path, root = make_dataset(tmp_path, 3)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "predict_submission_vllm", "--model", "fake", "--modality", "trimodal",
        "--queries", str(query_path), "--data-root", str(root),
        "--expected-query-count", "3", "--output-dir", str(out_dir)])
    source = json.loads(query_path.read_text(encoding="utf-8"))
    module.build_run_config("fake", None, query_path, root, "trimodal",
                            200704, 602112, 8192, True)
    # 手工造一份合法产物，模拟"跑完后再打包"
    out_dir.mkdir(parents=True, exist_ok=True)
    config = module.build_run_config("fake", None, query_path, root, "trimodal",
                                     200704, 602112, 8192, True)
    config.update({"engine": "vllm", "stage1_max_tokens": 64, "max_new_tokens": 128,
                   "max_num_seqs": 256, "mamba_cache_mode": "none",
                   "enable_prefix_caching": True, "lora_adapter": None,
                   "max_lora_rank": 32, "chunk_size": 256, "upscale": 1.0})
    (out_dir / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    from tools.predict_native_submission import PROMPTS, image_paths
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for index, (sample_id, record) in enumerate(source.items()):
            handle.write(json.dumps({
                "index": index, "id": sample_id, "query": record["query"],
                "image_paths": image_paths(record, root, query_path),
                "prompt": PROMPTS["trimodal"](record["query"]),
                "bbox": [0.1, 0.2, 0.3, 0.4],
                "parsed": True, "fallback": False, "generated_tokens": 1,
                "input_tokens": 1, "image_grid_thw": None, "generation_cap_hit": False,
                "latency_seconds": 0.0, "raw_text": '{"bbox_2d":[100,200,300,400]}',
                "stage": 1}) + "\n")
    monkeypatch.setattr(sys, "argv", [
        "predict_submission_vllm", "--model", "fake", "--modality", "trimodal",
        "--queries", str(query_path), "--data-root", str(root),
        "--expected-query-count", "3", "--output-dir", str(out_dir), "--package-only"])
    module.main()
    assert (out_dir / "submission.zip").exists()


def test_depth_policy_package_only_round_trip(tmp_path: Path, monkeypatch) -> None:
    """自定义深度提示词既用于新生成，也用于已有行的 package-only 校验。"""
    query_path, root = make_dataset(tmp_path, 2)
    out_dir = tmp_path / "sensor-linear"
    install_fake_vllm(monkeypatch)
    common = [
        "predict_submission_vllm", "--model", "fake", "--modality", "trimodal",
        "--depth-policy", "sensor_linear_20000", "--queries", str(query_path),
        "--data-root", str(root), "--expected-query-count", "2", "--chunk-size", "2",
        "--max-num-seqs", "4", "--output-dir", str(out_dir), "--single-pass",
    ]
    monkeypatch.setattr(sys, "argv", common)
    module.main()

    rows_path = out_dir / "predictions.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
    source = json.loads(query_path.read_text(encoding="utf-8"))
    assert rows[0]["prompt"] == module.prompt_for_query(
        source[rows[0]["id"]]["query"], "trimodal", "sensor_linear_20000")
    assert "physical distance unit is not established" in rows[0]["prompt"]
    config = json.loads((out_dir / "run_config.json").read_text(encoding="utf-8"))
    assert config["prompt_template"] == module.prompt_for_query(
        "{query}", "trimodal", "sensor_linear_20000")

    monkeypatch.setattr(sys, "argv", common + ["--package-only"])
    module.main()
    assert (out_dir / "submission.zip").exists()
    """真正构造一次 parser：重复定义选项会在 argparse 里抛 ArgumentError。

    这条是被真实故障逼出来的——加 `--adapter` 时没注意文件里已有一个，直到跑起来才
    报 "conflicting option string"，而那时已经准备花 86 分钟跑全量提交。
    只检查源码里"字符串存在"是不够的，必须让 argparse 真的注册一遍所有选项。
    """
    monkeypatch.setattr(sys, "argv", ["predict_submission_vllm", "--model", "m",
                                      "--modality", "trimodal", "--queries", "q.json",
                                      "--data-root", "d", "--output-dir", "out"])
    args = module.build_parser().parse_args()
    assert args.max_lora_rank == 32
    assert args.chunk_size == 256
    assert args.adapter is None


# ---- 超原生放大（Arm 2 之后的新实验）-----------------------------------------

def test_upscale_factor_one_returns_same_objects() -> None:
    """1.0 必须**原样返回同一批对象**——现役提交路径要逐像素不变。

    若这里悄悄复制或重采样，基线 314 就不可复现了，而实验会以"基线变了"这种
    最难排查的方式失败。
    """
    images = [Image.new("RGB", (64, 48), (10, 20, 30))]
    assert module.upscale_images(images, 1.0) is images


def test_upscale_doubles_size_and_keeps_order() -> None:
    images = [Image.new("RGB", (100, 50)), Image.new("RGB", (30, 70))]
    scaled = module.upscale_images(images, 2.0)
    assert [im.size for im in scaled] == [(200, 100), (60, 140)]
    assert len(scaled) == len(images)


def test_upscale_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="必须为正数"):
        module.upscale_images([Image.new("RGB", (10, 10))], 0.0)


def test_upscale_without_raising_max_pixels_is_rejected() -> None:
    """这是本实验最容易踩的陷阱，必须有回归测试锁住。

    处理器在图像超出 `max_pixels` 时会**下采样**。只放大而不提高 max_pixels，
    放大出来的图会被原样缩回去——模型收到的还是原尺寸，而日志、耗时、显存全正常。
    """
    with pytest.raises(ValueError, match="max-pixels"):
        module.check_upscale_consistency(1.5, 2073600)


def test_upscale_with_raised_max_pixels_passes() -> None:
    # 1.5² × 2073600 = 4665600
    module.check_upscale_consistency(1.5, 4665600)
    # 1.0 时不做检查（现役行为，max_pixels 保持原生即可）
    module.check_upscale_consistency(1.0, 2073600)


def test_upscale_tolerance_allows_processor_alignment() -> None:
    """处理器把尺寸对齐到 32 的倍数，实际像素不会精确等于 NATIVE × k²，需要容差。"""
    # 1.25² × 2073600 = 3240000；给略低一点的值，仍在 5% 容差内
    module.check_upscale_consistency(1.25, int(3240000 * 0.97))


def test_parser_accepts_upscale_and_defaults_to_one() -> None:
    """让 argparse 真的注册一遍 —— 重复或冲突的选项只有这时才会暴露。"""
    parser = module.build_parser()
    args = parser.parse_args(["--model", "m", "--modality", "trimodal",
                              "--queries", "q.json", "--data-root", "d",
                              "--output-dir", "out"])
    assert args.upscale == 1.0
    assert args.depth_policy is None
    args = parser.parse_args(["--model", "m", "--modality", "trimodal",
                              "--queries", "q.json", "--data-root", "d",
                              "--output-dir", "out", "--upscale", "1.5"])
    assert args.upscale == 1.5
    args = parser.parse_args(["--model", "m", "--modality", "trimodal",
                              "--queries", "q.json", "--data-root", "d",
                              "--output-dir", "out", "--depth-policy",
                              "sensor_linear_20000"])
    assert args.depth_policy == "sensor_linear_20000"
