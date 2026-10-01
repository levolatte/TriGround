"""Small serial runner for the frozen visual-agent experiment phases.

One shared budget ledger can span Z debug, development and full phases.  A
different ledger is required for the independent T budget.  This module never
reads ground truth or selects a winning mechanism.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .profiles import PROFILES


COHORTS = {
    "debug": "train32",
    "dev-fast": "dev16groups",
    "dev-capacity": "dev16groups",
    "full": "city412",
    "t-generate": "t250",
}
Z_CONTROLLERS = ("native", "c-lora")


def _csv(value):
    return tuple(piece.strip() for piece in (value or "").split(",") if piece.strip())


def task_specs(args):
    """Return ordered (controller, mechanism, profile) tuples."""
    default_controllers = {
        "debug": Z_CONTROLLERS, "dev-fast": Z_CONTROLLERS,
        "dev-capacity": (), "full": (), "t-generate": ("native",),
    }
    controllers = _csv(args.controllers) if args.controllers is not None else default_controllers[args.phase]
    if args.phase in {"debug", "dev-fast"}:
        if controllers != Z_CONTROLLERS:
            raise ValueError(f"{args.phase} requires --controllers native,c-lora")
        if args.mechanisms:
            raise ValueError(f"{args.phase} has a frozen mechanism list")
        mechanisms = ("C", "D") if args.phase == "debug" else ("A", "B", "S", "C", "D")
        return [(controller, mechanism, "fast") for controller in controllers for mechanism in mechanisms]
    if args.phase == "dev-capacity":
        if len(controllers) != 1 or controllers[0] not in Z_CONTROLLERS:
            raise ValueError("dev-capacity needs one native or c-lora controller")
        if args.mechanisms:
            raise ValueError("dev-capacity has the frozen B,S,C,D list")
        if args.profile not in {"capacity24", "capacity48"}:
            raise ValueError("dev-capacity needs --profile capacity24 or capacity48")
        return [(controllers[0], mechanism, args.profile) for mechanism in ("B", "S", "C", "D")]
    if args.phase == "full":
        if controllers:
            raise ValueError("full uses explicit --mechanisms controller:mechanism:profile entries")
        specs = []
        for raw in _csv(args.mechanisms):
            parts = raw.split(":")
            if len(parts) != 3:
                raise ValueError(f"full entry needs controller:mechanism:profile: {raw}")
            controller, mechanism, profile = parts
            if controller not in {"native", "c-lora", "t-lora"} or mechanism not in {"A", "B", "S", "C", "D"} or profile not in PROFILES:
                raise ValueError(f"invalid full entry: {raw}")
            if controller == "t-lora" and profile != "sft":
                raise ValueError("t-lora full entry must use sft profile")
            if controller == "c-lora" and profile == "sft":
                raise ValueError("sft full entry supports native or t-lora only")
            specs.append((controller, mechanism, profile))
        if not specs or len(set(specs)) != len(specs):
            raise ValueError("full needs a nonempty, duplicate-free explicit list")
        return specs
    if args.phase == "t-generate":
        if controllers != ("native",):
            raise ValueError("t-generate uses --controllers native")
        mechanisms = _csv(args.mechanisms)
        if len(mechanisms) != 1 or mechanisms[0] not in {"C", "D"}:
            raise ValueError("t-generate needs exactly one chosen mechanism C or D")
        return [("native", mechanisms[0], "sft")]
    raise ValueError(f"unknown phase: {args.phase}")


def _manifest_path(args):
    path = Path(args.manifests)
    cohort = COHORTS[args.phase]
    if path.suffix.lower() == ".jsonl":
        if path.stem != cohort:
            raise ValueError(f"{args.phase} requires the frozen {cohort}.jsonl manifest")
        return path
    return path / f"{cohort}.jsonl"


def _run_name(spec, limit):
    name = "_".join(spec)
    return f"{name}_limit{limit}" if limit is not None else name


def _adapter(args, controller):
    if controller == "c-lora":
        if not args.c_adapter:
            raise ValueError("c-lora needs --c-adapter")
        return args.c_adapter
    if controller == "t-lora":
        if not args.t_adapter:
            raise ValueError("t-lora needs --t-adapter")
        return args.t_adapter
    return None


def task_command(args, spec, output_dir, remaining_seconds):
    controller, mechanism, profile = spec
    command = [str(args.python), "-m", "experiments.evidence_decision.run",
               "--mechanism", mechanism, "--controller", controller,
               "--profile", profile, "--manifest", str(_manifest_path(args).resolve()),
               "--candidate-cache", str(Path(args.candidate_cache).resolve()),
               "--output-dir", str(output_dir), "--model", str(args.model),
               "--device", str(args.device), "--max-run-seconds", str(remaining_seconds)]
    adapter = _adapter(args, controller)
    if adapter:
        command += ["--adapter", str(adapter)]
    if profile == "sft":
        command += ["--memory", "latest"]
    if args.evidence_cache:
        command += ["--evidence-cache", str(Path(args.evidence_cache).resolve())]
    if args.dino_model:
        command += ["--dino-model", str(args.dino_model)]
    if args.sam_model:
        command += ["--sam-model", str(args.sam_model)]
    if args.limit is not None:
        command += ["--limit", str(args.limit)]
    for mapping in args.path_map:
        command += ["--path-map", mapping]
    return command


def _jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def _append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()


def _write_json(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _session_dir(args, specs):
    if args.phase == "debug":
        label = "both-CD-fast"
    elif args.phase == "dev-fast":
        label = "both-ABSCD-fast"
    elif args.phase == "dev-capacity":
        label = f"{specs[0][0]}-BSCD-{specs[0][2]}"
    elif args.phase == "t-generate":
        label = f"native-{specs[0][1]}-sft"
    else:
        label = "__".join(f"{c[0]}{m}{p.replace('capacity', 'cap')}" for c, m, p in specs)
    if args.limit is not None:
        label += f"-limit{args.limit}"
    return Path(args.run_root) / "batches" / args.phase / label


def _frozen_config(args, specs, ledger_path):
    return {
        "phase": args.phase, "cohort": COHORTS[args.phase],
        "specs": [list(spec) for spec in specs],
        "code_root": str(Path(args.code_root).resolve()),
        "manifest": str(_manifest_path(args).resolve()),
        "candidate_cache": str(Path(args.candidate_cache).resolve()),
        "evidence_cache": str(Path(args.evidence_cache).resolve()) if args.evidence_cache else None,
        "model": str(args.model), "c_adapter": args.c_adapter, "t_adapter": args.t_adapter,
        "dino_model": args.dino_model, "sam_model": args.sam_model,
        "python": str(args.python), "device": args.device, "path_map": list(args.path_map),
        "limit": args.limit, "budget_hours": args.budget_hours,
        "budget_ledger": str(ledger_path.resolve()),
    }


def execute(args):
    specs = task_specs(args)
    if args.limit is not None and (args.phase != "debug" or args.limit <= 0):
        raise ValueError("--limit is only supported for debug and must be positive")
    if args.budget_hours <= 0:
        raise ValueError("--budget-hours must be positive")
    run_root = Path(args.run_root).resolve()
    ledger_path = (run_root / args.budget_ledger).resolve() if args.budget_ledger and not args.budget_ledger.is_absolute() else (
        args.budget_ledger.resolve() if args.budget_ledger else run_root / ("t_budget.jsonl" if args.phase == "t-generate" else "z_budget.jsonl"))
    session_dir = _session_dir(args, specs)
    config = _frozen_config(args, specs, ledger_path)
    tasks = [(spec, run_root / "runs" / COHORTS[args.phase] / args.phase / _run_name(spec, args.limit)) for spec in specs]
    ledger = _jsonl(ledger_path)
    if any(entry.get("budget_hours") != args.budget_hours for entry in ledger):
        raise ValueError("shared budget ledger uses a different budget-hours value")
    consumed = sum(float(entry["elapsed_seconds"]) for entry in ledger)
    if args.dry_run:
        preview = [{"task": _run_name(spec, args.limit), "output_dir": str(directory),
                    "argv": task_command(args, spec, directory, max(0.0, args.budget_hours * 3600 - consumed))} for spec, directory in tasks]
        return {"status": "dry_run", "budget_ledger": str(ledger_path),
                "budget_remaining_hours": max(0.0, args.budget_hours - consumed / 3600),
                "config": config, "tasks": preview}

    config_path = session_dir / "batch_config.json"
    if config_path.exists():
        if not args.resume:
            raise FileExistsError(f"batch exists: {session_dir}; pass --resume")
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("resume batch config differs from frozen config")
    elif args.resume:
        raise FileNotFoundError(f"cannot resume without {config_path}")
    else:
        _write_json(config_path, config)
        commands = session_dir / "commands.jsonl"
        for spec, directory in tasks:
            _append_jsonl(commands, {"task": _run_name(spec, args.limit),
                                     "argv_template": task_command(args, spec, directory, "<remaining_budget_seconds>")})

    for spec, directory in tasks:
        task_name = _run_name(spec, args.limit)
        execution_path = directory / "execution.json"
        if execution_path.exists() and json.loads(execution_path.read_text(encoding="utf-8")).get("complete") is True:
            continue
        remaining = args.budget_hours * 3600 - consumed
        if remaining <= 0:
            return {"status": "budget_interrupted", "task": task_name,
                    "used_hours": consumed / 3600, "budget_hours": args.budget_hours}
        command = task_command(args, spec, directory, remaining)
        if (directory / "run_config.json").exists():
            command.append("--resume")
        attempts = sum(entry.get("output_dir") == str(directory) for entry in ledger) + 1
        log_dir = session_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / f"{task_name}_attempt{attempts:02d}.out.log"
        stderr_path = log_dir / f"{task_name}_attempt{attempts:02d}.err.log"
        wall_started = datetime.now(timezone.utc).isoformat()
        start = time.perf_counter()
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            process = subprocess.run(command, cwd=Path(args.code_root).resolve(), stdout=stdout, stderr=stderr, check=False)
        elapsed = time.perf_counter() - start
        consumed += elapsed
        execution = json.loads(execution_path.read_text(encoding="utf-8")) if execution_path.exists() else None
        complete = bool(execution and execution.get("complete") is True)
        outcome = "complete" if process.returncode == 0 and complete else (
            "budget_interrupted" if process.returncode == 0 and execution is not None else "failed")
        entry = {"phase": args.phase, "task": task_name, "output_dir": str(directory),
                 "started_utc": wall_started, "elapsed_seconds": elapsed,
                 "budget_hours": args.budget_hours, "argv": command,
                 "stdout": str(stdout_path), "stderr": str(stderr_path),
                 "exit_code": process.returncode, "execution_complete": complete,
                 "outcome": outcome}
        _append_jsonl(ledger_path, entry)
        ledger.append(entry)
        if outcome != "complete":
            return {"status": outcome, "task": task_name, "exit_code": process.returncode,
                    "used_hours": consumed / 3600, "budget_hours": args.budget_hours,
                    "stdout": str(stdout_path), "stderr": str(stderr_path)}
    return {"status": "complete", "tasks": len(tasks), "used_hours": consumed / 3600,
            "budget_hours": args.budget_hours, "budget_ledger": str(ledger_path)}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", required=True, choices=COHORTS)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--code-root", type=Path, required=True)
    p.add_argument("--manifests", type=Path, required=True)
    p.add_argument("--candidate-cache", type=Path, required=True)
    p.add_argument("--evidence-cache", type=Path)
    p.add_argument("--model", required=True)
    p.add_argument("--c-adapter")
    p.add_argument("--t-adapter")
    p.add_argument("--dino-model")
    p.add_argument("--sam-model")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--budget-hours", type=float, required=True)
    p.add_argument("--budget-ledger", type=Path)
    p.add_argument("--profile", default="capacity24", choices=PROFILES)
    p.add_argument("--controllers")
    p.add_argument("--mechanisms", help="full: comma-separated controller:mechanism:profile; T: C or D")
    p.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW")
    p.add_argument("--limit", type=int)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv=None):
    result = execute(parser().parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 1 if result["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
