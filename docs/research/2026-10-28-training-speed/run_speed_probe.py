"""Run structural-speed probes at an existing training checkpoint boundary.

Only the A/B queue's waiting parent is suspended; its current training child
finishes normally. The parent is resumed in finally, including probe failures.
All probe code, checkpoints and outputs are isolated from the A/B continuation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--code", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--parent-pid", required=True, type=int)
    parser.add_argument("--child-pid", type=int)
    parser.add_argument("--reuse-successful-preparation", type=Path)
    parser.add_argument("--boundary-checkpoint", type=Path)
    parser.add_argument("--forward-diagnostics", action="store_true")
    parser.add_argument("--operator-diagnostics", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    state = {"parent_pid": args.parent_pid, "child_pid": args.child_pid, "jobs": [],
             "code": str(args.code), "original_queue_resumed": False}

    def write(**changes):
        state.update(changes, updated_at=datetime.now(timezone.utc).isoformat())
        (args.output / "state.json").write_text(json.dumps(state, indent=2) + "\n")
        print(json.dumps(changes), flush=True)

    def job(name, command, env=None):
        write(status="running", current_job=name)
        record = {"name": name, "command": command, "started_at": datetime.now(timezone.utc).isoformat()}
        state["jobs"].append(record)
        write()
        started = time.monotonic()
        with (args.output / f"{name}.log").open("w") as stream:
            result = subprocess.run(command, cwd=args.code, env=env, stdout=stream, stderr=subprocess.STDOUT)
        record.update(exit_code=result.returncode, seconds=time.monotonic() - started)
        write()
        if result.returncode:
            raise RuntimeError(f"{name} exited {result.returncode}; see {name}.log")

    def cmdline(pid):
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()

    parent_command = cmdline(args.parent_pid)
    if "run_ab_to550.py" not in parent_command or "--train-segment" in parent_command:
        raise RuntimeError("The selected parent is not the original A/B queue")
    paused = False
    try:
        os.kill(args.parent_pid, signal.SIGSTOP)
        paused = True
        queue = read(args.root / "runs/ab550_20261028/state.json")
        current = queue["jobs"][-1]
        if queue["status"] != "running" or current["status"] != "running":
            raise RuntimeError("The original queue has no active job to finish")
        child_pid = args.child_pid or current["pid"]
        if current["pid"] != child_pid:
            raise RuntimeError("Requested child differs from the original queue's current job")
        child_stat = Path(f"/proc/{child_pid}/stat").read_text().split(")", 1)[1].split()
        if int(child_stat[1]) != args.parent_pid:
            raise RuntimeError("The selected job is not a child of the original queue")
        child_command = " ".join(current["command"]) if child_stat[0] == "Z" else cmdline(child_pid)
        if "run_ab_to550.py" not in child_command and "eval_quant_grounding.py" not in child_command:
            raise RuntimeError("The current child is neither A/B training nor evaluation")
        write(status="waiting_for_boundary", current_job=current["label"], child_pid=child_pid,
              parent_command=parent_command, child_command=child_command)
        while Path(f"/proc/{child_pid}/stat").exists():
            if Path(f"/proc/{child_pid}/stat").read_text().split(")", 1)[1].split()[0] == "Z":
                break
            time.sleep(30)
        checkpoint = args.boundary_checkpoint or args.root / "runs/ab550_20261028/A/train/checkpoint-100"
        expected_step = int(checkpoint.name.removeprefix("checkpoint-"))
        required = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_model.safetensors")
        if any(not (checkpoint / name).is_file() for name in required) or read(checkpoint / "trainer_state.json")["global_step"] != expected_step:
            raise RuntimeError(f"Current child ended without complete {checkpoint}; original queue will resume")
        write(boundary_checkpoint=str(checkpoint), boundary_step=expected_step)
        python = str(args.root / "train_env/bin/python")
        warmup = args.root / "runs/shared_format_warmup/training/checkpoint-50"
        model = "/root/autodl-tmp/rematch_models/Qwen3.6-27B"
        data_root = Path("/root/autodl-tmp/rematch_20260922/data")
        formal = data_root / "quant_grounding_20261026_formal/schedule.jsonl"
        config = args.root / "runs/initial_seed2026/A_seed2026/arm_config.json"
        benchmark = str(args.code / "tools/benchmark_shared_prefix.py")
        if args.reuse_successful_preparation:
            prior = read(args.reuse_successful_preparation / "state.json")
            for name in ("cuda_small_fla", "prepare_fixed"):
                completed = [record for record in prior["jobs"] if record["name"] == name]
                if not completed or completed[-1].get("exit_code") != 0:
                    raise RuntimeError(f"Cannot reuse incomplete prior {name}")
            bundle = args.reuse_successful_preparation / "fixed_bundle"
            read(bundle / "manifest.json")
            write(reused_preparation=str(args.reuse_successful_preparation))
        else:
            bundle = args.output / "fixed_bundle"
            environment = dict(os.environ, AIC_RUN_CUDA_SHARED_PREFIX="1")
            job("cuda_small_fla", [python, "-m", "pytest", "-s", "-q", "tests/test_qwen35_shared_prefix.py",
                                   "-k", "cuda_fla_long"], environment)
            job("prepare_fixed", [python, benchmark, "prepare", "--formal", str(formal),
                                  "--data-root", str(data_root), "--output", str(bundle)])
        if args.operator_diagnostics:
            output = args.output / "operator_diagnostic"
            job("operator_diagnostic", [python, benchmark, "diagnose-operators", "--bundle", str(bundle),
                                        "--config", str(config), "--adapter", str(warmup),
                                        "--model", model, "--precision", "nf4", "--output", str(output)])
            read(output / "operator_diagnostic.json")
            write(status="completed", current_job=None, diagnostic_output=str(output),
                  note="Layer-zero operator diagnosis only; inspect before changing shared prefix")
            return
        if args.forward_diagnostics:
            output = args.output / "forward_diagnostic"
            command = [python, benchmark, "diagnose", "--bundle", str(bundle),
                       "--config", str(config), "--adapter", str(warmup),
                       "--model", model, "--precision", "nf4"]
            job("forward_diagnostic", command + ["--output", str(output)])
            diagnosis = read(output / "diagnostic.json")["forward_diagnostics"]
            first = diagnosis["probes"]["shared_1"]["trace_comparison"]["first_above_native_repeat_noise_event"]
            detail_output = None
            if first and first.startswith("layers."):
                layer = int(first.split(".")[1])
                detail_output = args.output / f"forward_layer_{layer}"
                job(f"forward_layer_{layer}", command + ["--trace-layer", str(layer),
                                                        "--output", str(detail_output)])
                read(detail_output / "diagnostic.json")
            write(status="completed", current_job=None, diagnostic_output=str(output),
                  detail_output=str(detail_output) if detail_output else None,
                  note="Forward diagnosis only; inspect first divergence before changing or adopting shared prefix")
            return
        for groups, repeats in ((1, 1), (4, 2)):
            output = args.output / f"fixed_g{groups}"
            job(f"fixed_g{groups}", [python, benchmark, "run", "--bundle", str(bundle), "--config", str(config),
                                    "--adapter", str(warmup), "--model", model, "--precision", "nf4",
                                    "--groups", str(groups), "--repeats", str(repeats), "--output", str(output)])
            comparison = read(output / "result.json")
            # This only prevents clearly divergent prototypes entering the online probe.
            # Adoption still requires reading the complete paired evidence.
            for row in comparison["runs"]:
                if row["variant"] != "shared":
                    continue
                gradient = row["gradient_comparison"]
                logps = row["logp_comparison"]["current"]
                if (not gradient["finite"] or gradient["relative_l2"] is None
                        or gradient["relative_l2"] > 0.05 or logps["max_abs"] > 0.1):
                    raise RuntimeError(f"fixed_g{groups}: numerical differences require investigation before online training")
        source = read(config)
        source.pop("matched_query_exposures", None)
        source.pop("resume_from_checkpoint", None)
        online_outputs = {}
        for variant in ("baseline", "shared"):
            output = args.output / f"online_{variant}"
            online = dict(source, model=model, dataset=str(formal), data_root=str(data_root),
                          adapters=str(warmup), output_dir=str(output), num_generations=8,
                          shared_prefix=variant == "shared", train_micro_batch_size=8 if variant == "shared" else 1,
                          runtime_optimization=variant == "shared", geometry_reward="capped_iou",
                          max_steps=2, max_gpu_hours=None, save_hours=[], save_steps=2, save_total_limit=1,
                          gradient_checkpointing_every_n_layers=1, deterministic_training=False)
            config_path = args.output / f"online_{variant}.json"
            config_path.write_text(json.dumps(online, indent=2) + "\n")
            job(f"online_{variant}", [python, str(args.code / "tools/train_quant_grounding.py"),
                                       "--config", str(config_path)])
            if read(output / "checkpoint-2/trainer_state.json")["global_step"] != 2:
                raise RuntimeError(f"{variant} did not save the complete two-update probe")
            online_outputs[variant] = str(output)
        write(status="completed", current_job=None, online_outputs=online_outputs,
              note="Read numerical and online timing evidence before enabling shared prefix in A/B; no algorithm change applied")
    except BaseException as error:
        write(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if paused:
            os.kill(args.parent_pid, signal.SIGCONT)
            write(original_queue_resumed=True)


if __name__ == "__main__":
    main()
