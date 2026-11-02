"""四题 G2 纠错循环及其同题序 SFT 对照；不使用 Trainer。"""
from __future__ import annotations
import argparse
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.quant_correction import (QuestionSampler, ce_question_loss, classify,
    decide_route, mixed_response_loss, observed_size_ratio, random_state_from_json,
    random_state_to_json, serialize_gt)
from tools.quant_grounding_config import load_config, read_train_index

SCHEMA = "acc-correction-v1"

def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def append_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def rng_state() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state

def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])

def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def process_question(runtime, row, mode, retention_rng, config, jitter_rng=None):
    """参数在整个四题批次内不变；每题计算图在返回前释放。

    `jitter_rng` 为 None 时监督目标就是真值框（旧行为）。传入后改用容忍带抖动，
    见 `quant_correction.jitter_solution` 的理由说明。
    """
    record = {"id": row["id"]}
    adaptive = bool(config.get("jitter_adaptive", False))
    jitter_args = dict(
        min_iou=config.get("jitter_min_iou", 0.65),
        scale_low=config.get("jitter_scale_low", 0.85),
        scale_high=config.get("jitter_scale_high", 1.20),
        small_area=config.get("jitter_small_area"),
    )
    with runtime.question(row) as question:
        if mode == "sft":
            route = "gt"
            # sft 模式没有模型自己的框可供判断方向，所以这里只做非自适应的容忍带抖动，
            # 与旧行为一致。方向自适应只在纠错循环（下面 route=="gt"）里生效。
            target = serialize_gt(row, jitter_rng, adaptive=False, **jitter_args)
            loss = ce_question_loss(question.supervised_logps(target))
        else:
            completions = question.generate()
            result = classify(row, completions)
            decision = decide_route(result, retention_rng, config["keep_probability"])
            route = decision.route
            record.update(classification=result["route"], hits=result["hits"],
                          completions=[asdict(completion) for completion in completions],
                          selected_index=decision.selected_index)
            if route == "mixed":
                # Reference first; policy graphs never survive an adapter switch.
                refs = [question.logps(c.token_ids, reference=True) for c in completions]
                values = []
                for completion, ref, hit in zip(completions, refs, result["hits"]):
                    loss = mixed_response_loss(question.logps(completion.token_ids), ref, hit, config["beta"])
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"{row['id']}: 非有限回答损失")
                    # 两答平均与固定四题分母；微批1立即释放长提示计算图。
                    (loss / 8).backward()
                    values.append(float(loss.detach()))
                    del loss
                record.update(route=route, loss=sum(values) / 2)
                return record
            elif route == "gt":
                # 方向自适应：用模型这两个失败框相对真值的尺寸比，决定往哪边纠。
                # 上一版对所有"两答全错"的样本无条件放大 1.133 倍，救回 4 道但误伤 10 道
                # （311→305），因为把本来对的框也放大了。现在只在偏小/偏大时纠、且幅度温和。
                direction = (observed_size_ratio(result["predicted_boxes"], row["solution"])
                             if adaptive else None)
                target = serialize_gt(row, jitter_rng, direction=direction, adaptive=adaptive,
                                      **jitter_args)
                record.update(jitter_direction=direction)
                loss = ce_question_loss(question.supervised_logps(target))
            elif route == "retain":
                loss = ce_question_loss(question.logps(completions[decision.selected_index].token_ids))
            else:
                record.update(route=route, loss=0.0)
                return record
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{row['id']}: 非有限逐题损失")
        (loss / 4).backward()
        record.update(route=route, loss=float(loss.detach()))
    return record

def save_checkpoint(runtime, optimizer, output, state, sampler, retention_rng, config, jitter_rng=None):
    checkpoint = output / f"checkpoint-{state['presentations']}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    runtime.save_adapter(checkpoint)
    torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
    torch.save(rng_state(), checkpoint / "rng.pt")
    saved = state | {"schema": SCHEMA, "sampler": sampler.state_dict() if sampler else None,
        "retention_rng": random_state_to_json(retention_rng.getstate()),
        "jitter_rng": random_state_to_json(jitter_rng.getstate()) if jitter_rng else None,
        "config": config}
    # Only this last file marks a complete native checkpoint.
    write_json(checkpoint / "loop_state.json", saved)
    return checkpoint

def run(config: dict, mode: str, output: Path, resume: Path | None = None, runtime_factory=None):
    if mode not in {"correction", "sft"}:
        raise ValueError("mode 必须为 correction 或 sft")
    rows = read_train_index(Path(config["train_index"]))
    by_id = {row["id"]: row for row in rows}
    output.mkdir(parents=True, exist_ok=True)
    saved = json.loads((resume / "loop_state.json").read_text(encoding="utf-8")) if resume else None
    state = {"mode": mode, "presentations": 0, "updates": 0, "batches": 0}
    if saved:
        if saved["schema"] != SCHEMA or saved["mode"] != mode:
            raise ValueError("只支持同模式的新流程原生检查点")
        # 原生生成提示复用仅改变重复计算；GPU门禁通过后可在完整保存点启用。
        # 容忍带抖动三参数是**有意的方法变更**：它只改监督目标、不改奖励判定，
        # 且实际生效值会写进 output/run_config.json、逐题轨迹写进 order.jsonl，
        # 所以允许在完整保存点上切换，否则无法在已有轨迹上做单变量对照。
        mutable = {"stop_presentations", "checkpoint_every", "sequence_path", "generation_prefill_reuse",
                   "jitter_min_iou", "jitter_scale_low", "jitter_scale_high"}
        if {k: v for k, v in saved["config"].items() if k not in mutable} != {
                k: v for k, v in config.items() if k not in mutable}:
            raise ValueError("恢复配置改变了模型、题目或训练配方")
        state.update({key: saved[key] for key in state})
    elif (output / "order.jsonl").exists():
        raise ValueError("已有题序；请从新流程完整检查点恢复")
    if config["stop_presentations"] < state["presentations"]:
        raise ValueError("停止呈现量早于恢复位置")
    sequence = read_jsonl(Path(config["sequence_path"])) if mode == "sft" else None
    if sequence is not None:
        if len(sequence) < config["stop_presentations"]:
            raise ValueError("纠错题序尚未覆盖 SFT 本段预算")
        if any(item["presentation"] != i + 1 or item["id"] not in by_id
               for i, item in enumerate(sequence)):
            raise ValueError("纠错题序含非连续呈现编号或非训练题")
    journal = output / "order.jsonl"
    if saved:
        if state["presentations"] == 0:
            journal.touch(exist_ok=True)
            (output / "batches.jsonl").touch(exist_ok=True)
        committed = read_jsonl(journal)[:state["presentations"]]
        if len(committed) != state["presentations"]:
            raise ValueError("检查点对应的题序记录缺失")
        if sequence is not None and [r["id"] for r in committed] != [r["id"] for r in sequence[:len(committed)]]:
            raise ValueError("SFT 恢复前缀与纠错实际题序不一致")
        journal.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in committed), encoding="utf-8")
        metrics = output / "batches.jsonl"
        committed_metrics = read_jsonl(metrics)[:state["batches"]]
        metrics.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in committed_metrics), encoding="utf-8")
    seed_all(config["seed"])
    if runtime_factory is None:
        from tools.quant_correction_runtime import CorrectionRuntime
        runtime_factory = CorrectionRuntime
    runtime = runtime_factory(config, mode, adapter_path=str(resume) if resume else None)
    parameters = list(runtime.trainable_parameters())
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=0)
    sampler = QuestionSampler(rows, config["seed"]) if mode == "correction" else None
    retention_rng = random.Random(config["seed"] + 1)
    # 容忍带抖动用的独立随机源；配置里没用 jitter_min_iou 时保持 None，行为与旧版完全一致。
    jitter_rng = random.Random(config["seed"] + 2) if config.get("jitter_min_iou") else None
    if saved:
        optimizer.load_state_dict(torch.load(resume / "optimizer.pt", map_location="cpu", weights_only=False))
        if sampler:
            sampler.load_state_dict(saved["sampler"])
        retention_rng.setstate(random_state_from_json(saved["retention_rng"]))
        if jitter_rng is not None and saved.get("jitter_rng"):
            jitter_rng.setstate(random_state_from_json(saved["jitter_rng"]))
        # 从没开抖动的检查点续训并开启抖动，属于有意的方法变更（例如在 correction-1600 上
        # 改用容忍带监督）。此时抖动流从种子重新开始，不是位级复现，必须显式记录下来。
        restore_rng(torch.load(resume / "rng.pt", map_location="cpu", weights_only=False))
    write_json(output / "run_config.json", config | {"mode": mode, "schema": SCHEMA})
    checkpoint = resume
    if checkpoint is None:
        checkpoint = save_checkpoint(runtime, optimizer, output, state, sampler, retention_rng, config, jitter_rng)
    while state["presentations"] < config["stop_presentations"]:
        start_index = state["presentations"]
        batch = sampler.next_batch() if sampler else [by_id[r["id"]] for r in sequence[start_index:start_index + 4]]
        optimizer.zero_grad(set_to_none=True)
        synchronize()
        started = time.perf_counter()
        records = []
        for offset, row in enumerate(batch):
            record = process_question(runtime, row, mode, retention_rng, config, jitter_rng)
            record["presentation"] = start_index + offset + 1
            records.append(record)
            if sampler:
                sampler.observe(row["id"], record["classification"])
        active = sum(record["route"] != "skip" for record in records)
        grad_norm = None
        if active:
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            grad_norm = float(norm)
            optimizer.step()
            state["updates"] += 1
        state["presentations"] += 4
        state["batches"] += 1
        synchronize()
        metric = state | {"seconds": time.perf_counter() - started, "active_questions": active,
            "grad_norm": grad_norm, "routes": [r["route"] for r in records],
            "loss": sum(r["loss"] for r in records) / 4}
        append_jsonl(journal, records)
        append_jsonl(output / "batches.jsonl", [metric])
        print(json.dumps(metric, ensure_ascii=False), flush=True)
        if state["presentations"] % config["checkpoint_every"] == 0 or state["presentations"] == config["stop_presentations"]:
            checkpoint = save_checkpoint(runtime, optimizer, output, state, sampler, retention_rng, config, jitter_rng)
    result = state | {"checkpoint": str(checkpoint)}
    write_json(output / "completed.json", result)
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("correction", "sft"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    run(load_config(args.config), args.mode, args.output_dir, args.resume)

if __name__ == "__main__":
    main()
