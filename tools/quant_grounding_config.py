"""Qwen3.6-27B 四臂实验的纯 Python 配置与数据准备，不依赖 ms-swift。"""

from __future__ import annotations

import json
from pathlib import Path

from tools.prepare_qwen3vl_native_sft import bbox_to_qwen1000


ARMS = {"A", "B", "C", "D"}
PRECISIONS = {"nf4", "bf16"}
REWARDS = {
    "A": "quant_bbox_contrast",
    "B": "quant_bbox_acc05",
    "D": "quant_bbox_contrast_thinking",
}


def configure_unrestricted_hf_sampling(engine, request_config) -> dict:
    """Use explicit HF top_k=0 so Swift and Transformers do not refill model defaults."""
    engine.generation_config.top_k = 0
    effective = engine._prepare_generation_config(request_config)
    if effective.do_sample and (effective.temperature != 1 or effective.top_p != 1 or effective.top_k != 0):
        raise RuntimeError("HF rollout sampling differs from temperature=1/top_p=1/no top-k")
    return {"temperature": effective.temperature, "top_p": effective.top_p,
            "top_k": effective.top_k, "do_sample": effective.do_sample}


def load_config(path: Path, **overrides: str | None) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    config = {**config, **{key: value for key, value in overrides.items() if value is not None}}
    config["arm"] = config["arm"].upper()
    config["precision"] = config["precision"].lower()
    if config["arm"] not in ARMS or config["precision"] not in PRECISIONS:
        raise ValueError("arm 必须是 A/B/C/D，precision 必须是 nf4/bf16")
    if config["arm"] == "D" and not config.get("bounded_thinking", False) and config.get("max_completion_length", 384) != 384:
        raise ValueError("D 臂必须使用 384 token 的原生思考预算")
    if config.get("bounded_thinking", False) and config["arm"] != "D":
        raise ValueError("bounded_thinking 仅用于独立 D 修复实验")
    if config.get("bounded_thinking", False) and config.get("format_policy") != "single_box":
        raise ValueError("D 修复训练须使用 single_box 格式口径")
    if config.get("bounded_thinking", False):
        config["thought_limit"] = int(config.get("thought_limit", 256))
        config["answer_limit"] = int(config.get("answer_limit", 64))
        if min(config["thought_limit"], config["answer_limit"]) <= 0:
            raise ValueError("思考和答案词元预算必须为正")
        config["max_completion_length"] = config["thought_limit"] + config["answer_limit"] + 2
        config.setdefault("rollout_importance_sampling_mode", "token_truncate")
        config.setdefault("rollout_importance_sampling_threshold", 2.0)
        config.setdefault("sampling_replay_tokens", 64)
        config.setdefault("sampling_replay_atol", 0.05)
        if (config["rollout_importance_sampling_mode"] != "token_truncate"
                or config["rollout_importance_sampling_threshold"] != 2.0):
            raise ValueError("D 修复使用 Swift token_truncate 2.0 采样校正")
    if config["arm"] in {"A", "B"} and config.get("max_completion_length", 64) != 64:
        raise ValueError("A/B 臂必须使用 64 token 的短答案预算")
    if int(config.get("hf_max_batch_size", 1)) not in {1, 2}:
        raise ValueError("HF 生成并发上限只允许 1，或实测后的 2")
    config["train_micro_batch_size"] = int(config.get("train_micro_batch_size", 1))
    config["gradient_checkpointing_every_n_layers"] = int(
        config.get("gradient_checkpointing_every_n_layers", 1))
    config["num_generations"] = int(config.get("num_generations", 8))
    config["geometry_reward"] = config.get("geometry_reward", "capped_iou")
    config["shared_prefix"] = config.get("shared_prefix", False)
    if config["num_generations"] not in {2, 8}:
        raise ValueError("num_generations 只允许独立 G2 对照或原 G8")
    if config["geometry_reward"] not in {"capped_iou", "continuous_iou"}:
        raise ValueError("geometry_reward 必须为 capped_iou 或 continuous_iou")
    if config["geometry_reward"] == "continuous_iou" and config["arm"] != "B":
        raise ValueError("连续 IoU 辅助项仅用于独立 B 臂对照")
    if not isinstance(config["shared_prefix"], bool):
        raise ValueError("shared_prefix 必须是布尔值")
    if config["shared_prefix"]:
        if config["arm"] not in {"A", "B"} or not config.get("runtime_optimization", False):
            raise ValueError("shared_prefix 需要 A/B 与 runtime_optimization=true")
        if config["train_micro_batch_size"] != config["num_generations"]:
            raise ValueError("shared_prefix 每个微批必须包含同题完整 G 个回答")
    elif config["train_micro_batch_size"] not in {1, 2}:
        raise ValueError("普通路径 train_micro_batch_size 只允许 1 或 2")
    if config["gradient_checkpointing_every_n_layers"] not in {1, 2}:
        raise ValueError("gradient_checkpointing_every_n_layers 只允许 1 或 2")
    if config.get("matched_query_exposures") is not None:
        config["matched_query_exposures"] = int(config["matched_query_exposures"])
        if config["matched_query_exposures"] <= 0:
            raise ValueError("matched_query_exposures 必须为正")
        if config["arm"] != "C" and config["matched_query_exposures"] % 4:
            raise ValueError("RL 臂 matched_query_exposures 必须按每步 4 题取整")
        if config["arm"] == "C" and config["matched_query_exposures"] % 8:
            raise ValueError("C 臂 matched_query_exposures 必须按每步 8 条取整")
    if config.get("max_gpu_hours") is not None and float(config["max_gpu_hours"]) <= 0:
        raise ValueError("max_gpu_hours 必须为正")
    return config


def _absolute_images(row: dict, data_root: Path) -> list[str]:
    return [str((data_root / image).resolve()) for image in row["images"]]


def prepare_dataset(source: Path, destination: Path, data_root: Path, arm: str,
                    *, probe: bool = False, bounded_thinking: bool = False) -> int:
    """相对媒体路径在模型边界解析；C 只把 GT 放进监督回答。"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with source.open("r", encoding="utf-8") as inp, destination.open("w", encoding="utf-8") as out:
        for line in inp:
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row["messages"]
            if bounded_thinking:
                if arm != "D":
                    raise ValueError("受控思考数据只用于 D 修复实验")
                from tools.quant_grounding_thinking import apply_bounded_thinking_prompt
                messages = apply_bounded_thinking_prompt(messages)
            if any(message["role"] == "assistant" for message in messages):
                raise ValueError(f"{row['id']}: 训练输入含 assistant 回答")
            images = _absolute_images(row, data_root)
            if probe and len(images) != 3:
                continue
            if sum(message["content"].count("<image>") for message in messages) != len(images):
                raise ValueError(f"{row['id']}: image 标记与媒体数量不一致")
            missing = [image for image in images if not Path(image).is_file()]
            if missing:
                raise FileNotFoundError(f"{row['id']}: 找不到媒体 {missing}")
            if arm == "C":
                values = bbox_to_qwen1000(row["solution"])
                response = json.dumps({"bbox_2d": values}, ensure_ascii=False, separators=(",", ":"))
                written = {"messages": [*messages, {"role": "assistant", "content": response}], "images": images}
            else:
                written = {**row, "messages": messages, "images": images}
                for key in ("negative_boxes", "negative_kind"):
                    if key in written:
                        written[f"grounding_{key}"] = written.pop(key)
            out.write(json.dumps(written, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
            if probe and count == 8:
                break
    if not count:
        raise ValueError(f"空训练数据：{source}")
    if probe and count < 8:
        raise ValueError(f"真实三图 probe 需要至少 8 条，当前 {count}")
    return count


def swift_arguments(config: dict, dataset: Path, *, max_steps: int | None = None,
                    save_steps: int | None = None, resume: str | None = None) -> list[str]:
    """返回 ms-swift 4.5.3 原生参数，训练逻辑交给 SwiftSft/SwiftRLHF。"""
    arm = config["arm"]
    generations = int(config.get("num_generations", 8))
    steps = int(max_steps if max_steps is not None else config.get("max_steps", 1_000_000))
    args = [
        "--model", str(config["model"]),
        "--model_type", "qwen3_5",
        "--template", "qwen3_5",
        "--enable_thinking", "true" if arm == "D" else "false",
        "--dataset", str(dataset),
        "--output_dir", str(config["output_dir"]),
        "--add_version", "false",
        "--tuner_type", "lora",
        "--split_dataset_ratio", "0",
        "--dataset_shuffle", "false",
        "--train_dataloader_shuffle", "false",
        "--eval_strategy", "no",
        "--torch_dtype", "bfloat16",
        "--attn_impl", "sdpa",
        "--max_pixels", str(config.get("max_pixels", 1204224)),
        "--max_length", str(config.get("max_length", 8192)),
        "--freeze_vit", "true",
        "--freeze_aligner", "true",
        "--lora_rank", "32",
        "--lora_alpha", "64",
        "--lora_dropout", "0",
        "--target_modules", "all-linear",
        "--gradient_checkpointing", "true",
        "--per_device_train_batch_size", str(config.get("train_micro_batch_size", 1)),
        "--gradient_accumulation_steps", str((4 * generations if arm != "C" else 8) //
                                             int(config.get("train_micro_batch_size", 1))),
        "--learning_rate", str(config.get("learning_rate", 2e-6)),
        "--lr_scheduler_type", "constant",
        "--warmup_ratio", "0",
        "--weight_decay", "0",
        "--max_grad_norm", "1",
        "--max_steps", str(steps),
        "--save_strategy", "steps",
        "--save_steps", str(save_steps if save_steps is not None else config.get("save_steps", 100)),
        "--save_total_limit", str(max(int(config.get("save_total_limit", 12)),
                                      4 if config.get("matched_query_exposures") else 0)),
        "--logging_steps", "1",
        "--dataloader_num_workers", str(config.get("dataloader_num_workers", 4)),
        "--dataloader_persistent_workers", "true",
        "--dataset_num_proc", str(config.get("dataset_num_proc", 4)),
        "--seed", str(config.get("seed", 2026)),
        "--report_to", "none",
        "--ignore_args_error", "false",
    ]
    if config["precision"] == "nf4":
        args += ["--quant_method", "bnb", "--quant_bits", "4",
                 "--bnb_4bit_quant_type", "nf4", "--bnb_4bit_use_double_quant", "true",
                 "--bnb_4bit_compute_dtype", "bfloat16"]
    if arm != "C":
        plugin = config.get("reward_plugin", str(Path(__file__).with_name("swift_quant_grounding_reward.py")))
        args += [
            "--rlhf_type", "grpo",
            "--external_plugins", str(plugin),
            "--reward_funcs", ("quant_bbox_contrast_thinking_repaired"
                               if config.get("bounded_thinking", False) else
                               "quant_bbox_acc05_continuous_iou"
                               if config.get("geometry_reward") == "continuous_iou" else REWARDS[arm]),
            "--loss_type", "dapo",
            "--importance_sampling_level", "token",
            "--epsilon", "0.2", "--epsilon_high", "0.28",
            "--scale_rewards", "none", "--beta", "0.01",
            "--use_vllm", "false",
            "--num_generations", str(generations), "--generation_batch_size", str(generations),
            "--num_iterations", "1", "--dynamic_sample", "false",
            "--temperature", "1", "--top_p", "1", "--top_k", "-1",
            "--max_completion_length", str(config.get("max_completion_length", 384) if arm == "D" else 64),
            "--truncation_strategy", "delete",
            "--log_completions", "true",
        ]
        if config.get("bounded_thinking", False):
            args += ["--rollout_importance_sampling_mode", "token_truncate",
                     "--rollout_importance_sampling_threshold", "2.0",
                     "--log_rollout_offpolicy_metrics", "true"]
    if config.get("adapters"):
        args += ["--adapters", str(config["adapters"])]
        if arm != "C":
            args += ["--ref_adapters", str(config["adapters"])]
    checkpoint = resume or config.get("resume_from_checkpoint")
    if checkpoint:
        args += ["--resume_from_checkpoint", str(checkpoint)]
    return args
