"""Cloud-only overnight continuation; at most two retries per stage."""
import json, os, re, shutil, subprocess, time
from pathlib import Path

ROOT = Path("/root/autodl-tmp/rematch_20260922")
CODE = ROOT / "code"
OUT = ROOT / "results/first_batch"
LOGS = ROOT / "logs"
PY = "/root/miniconda3/bin/python"
MODEL = "/root/rematch_models/Qwen3-VL-8B-Instruct"
DATA = ROOT / "data/city/train"
GT = DATA / "target_v2/qwen_generation_val.json"
SFT = DATA / "target_v2/qwen3vl_native_sft"
FATAL = re.compile(r"out of memory|OutOfMemoryError|non.?finite|NaN|gradient.*(missing|disconnected)|Less than 3 GiB|No space left|AssertionError|ModuleNotFoundError|ImportError|SyntaxError|JSONDecodeError|resume.*mismatch", re.I)

def log(message):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), message, flush=True)

def alive(pid):
    p = Path(f"/proc/{pid}/stat")
    return p.exists() and p.read_text().split()[2] != "Z"

def space():
    return all(shutil.disk_usage(p).free >= 3 * 1024**3 for p in (ROOT, Path(MODEL)))

def stage(name, command, env=None, retries=2, blocked_logs=()):
    marker = OUT / f"{name}.complete"
    if marker.exists():
        return True
    for attempt in range(retries + 1):
        if not space():
            log(f"STOP {name}: free disk below 3 GiB")
            return False
        path = LOGS / f"overnight_{name}_attempt{attempt}.log"
        if path.exists():
            raise RuntimeError(f"Refuse to overwrite attempt evidence: {path}")
        log(f"START {name} attempt={attempt}")
        with path.open("w") as stream:
            result = subprocess.run(command, cwd=CODE, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode == 0:
            marker.touch()
            log(f"COMPLETE {name}")
            return True
        texts = [path.read_text(errors="replace")[-20000:]]
        for p in blocked_logs:
            if p.exists():
                text = p.read_text(errors="replace")
                texts.append(text[-20000:])
                shutil.copyfile(p, LOGS / f"overnight_{name}_attempt{attempt}_{p.name}")
        log(f"FAILED {name} rc={result.returncode}")
        if FATAL.search("\n".join(texts)) or attempt == retries:
            log(f"STOP {name}: requires review or retry limit reached")
            return False
        time.sleep(60 * (attempt + 1))
    return False

def evaluate(name, variant, adapter):
    return stage(name, [PY, "tools/evaluate_pretrained_grounder.py",
        "--model", MODEL, "--adapter", str(adapter), "--manifest", str(SFT/f"{variant}_val.json"),
        "--target-manifest", str(GT), "--data-root", str(DATA), "--output-dir", str(OUT/name),
        "--prompt-style", "native", "--max-pixels", "602112", "--min-pixels", "200704",
        "--max-new-tokens", "128", "--resume"])

def report():
    names = ["r0", "r2_epoch1", "r2_epoch2", "m2_epoch1", "m2_epoch2", "l0",
             "r2_seed2027_epoch1", "r2_seed2027_epoch2", "m2_seed2027_epoch1", "m2_seed2027_epoch2"]
    command = [PY, "tools/report_rematch_experiment.py", "--manifest", str(GT)]
    for name in names:
        if (OUT/name/"summary.json").exists():
            command += ["--run", f"{name}={OUT/name/'predictions.jsonl'}"]
    command += ["--output-dir", str(OUT/"overnight_report")]
    subprocess.run(command, cwd=CODE, check=True)

def main():
    pid = int((LOGS/"first_batch_runner.pid").read_text())
    log(f"WAIT existing main pid={pid}")
    while alive(pid):
        time.sleep(30)
    required = ["r2_train", "r2_epoch1", "r2_epoch2", "m2_train", "m2_epoch1", "m2_epoch2"]
    if not all((OUT/f"{n}.complete").exists() for n in required):
        # Completed EGM markers are mandatory: never reload retired weights.
        if not all((OUT/f"{n}.complete").exists() for n in ["e0_smoke", "e0"]):
            raise RuntimeError("Retired EGM completion markers missing; refusing old runner")
        logs = [LOGS/f"{n}.log" for n in required if not (OUT/f"{n}.complete").exists()]
        if any(FATAL.search(p.read_text(errors="replace")[-20000:]) for p in logs if p.exists()):
            log("MAIN stopped on deterministic error; no automatic training retry")
        else:
            stage("main_recovery", ["bash", "scripts/run_rematch_first_batch.sh"], retries=1, blocked_logs=logs)
    # All commands are serial. Existing Locate waiter is explicitly retired before this starts.
    while subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
        time.sleep(30)
    if (ROOT/"models/LocateAnything-3B/.download_complete").exists():
        stage("locate_sequence", ["bash", "scripts/run_locate_after_first_batch.sh"],
              blocked_logs=[LOGS/"l0_smoke.log", LOGS/"l0.log"])
    else:
        log("STOP Locate: weights incomplete, do not change mirror")
    report()
    summaries = [OUT/n/"summary.json" for n in ["r2_epoch2", "m2_epoch2"]]
    if all(p.exists() for p in summaries):
        r, m = [json.loads(p.read_text()) for p in summaries]
        delta = m["hits"] - r["hits"]
        (OUT/"overnight_decision.json").write_text(json.dumps({
            "r2_hits": r["hits"], "m2_hits": m["hits"], "delta": delta,
            "seed2027_pair": delta >= 4,
            "note": "Preapproved second seed gate; not a significance claim."}, indent=2))
        if delta >= 4:
            log(f"SEED2027_PAIR triggered net={delta}")
            env = os.environ.copy()
            env.update(DATA_ROOT=str(DATA), QWEN_FINETUNE_DIR=str(ROOT/"third_party/Qwen3-VL/qwen-vl-finetune"),
                       MODEL_PATH=MODEL, SEED="2027", EPOCHS="2", LEARNING_RATE="1e-5", MAX_STEPS="-1",
                       PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", TOKENIZERS_PARALLELISM="false")
            for variant, name in [("rgb", "r2_seed2027"), ("trimodal", "m2_seed2027")]:
                env.update(DATASET_VARIANT=variant, OUTPUT_DIR=str(OUT/name))
                if not stage(name+"_train", ["bash", "scripts/run_qwen3vl_native_lora.sh"], env=env):
                    break
                if not all(evaluate(f"{name}_epoch{epoch}", variant, OUT/name/f"checkpoint-{step}")
                           for epoch, step in [(1, 464), (2, 928)]):
                    break
            report()
        else:
            log(f"No seed2027 pair: net={delta}; await result-based next route")
    (OUT/"overnight_finished.json").write_text(json.dumps({"finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}))
    log("OVERNIGHT FINISHED; inspect report and failed attempt logs")

if __name__ == "__main__":
    main()
