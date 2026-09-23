#!/usr/bin/env bash
#SBATCH --job-name=triground-8b-rdt-city
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
cd "${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
source scripts/slurm_env.sh

run_dir="${RUN_ROOT:-runs/qwen3_vl_8b_rdt_qwen_city-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}}"
[[ "$run_dir" = /* ]] || run_dir="$PWD/$run_dir"
[[ ! -e "$run_dir" ]] || { echo "Run directory already exists: $run_dir" >&2; exit 1; }
mkdir -p "$run_dir"

"$python_bin" - "$run_dir" <<'PY'
import os
from pathlib import Path
import sys
import yaml
from mm_grounding.config import load_config

run_dir = Path(sys.argv[1])
raw = yaml.safe_load(Path("configs/qwen3_vl_8b_rdt_qwen_city.yaml").read_text(encoding="utf-8"))
raw["output_dir"] = str(run_dir)
raw["model"]["backbone"] = os.environ["BACKBONE"]
if os.environ.get("CITY_ROOT"):
    root = Path(os.environ["CITY_ROOT"]).resolve() / "train/target_v2"
    raw["data"]["train_manifest"] = str(root / "qwen_generation_train_100.json")
    raw["data"]["val_manifest"] = str(root / "qwen_generation_val.json")
for field in ("train_manifest", "val_manifest"):
    path = Path(raw["data"][field]).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{field}: {path}")
    raw["data"][field] = str(path)
config_path = run_dir / "config.yaml"
config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
load_config(config_path)
PY

config="$run_dir/config.yaml"
train_manifest="$("$python_bin" -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["data"]["train_manifest"])' "$config")"
val_manifest="$("$python_bin" -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["data"]["val_manifest"])' "$config")"
"$python_bin" tools/audit_manifest_overlap.py "$train_manifest" "$val_manifest" \
    --fail-on-overlap --output "$run_dir/audit_train_val.json"
manual_val="${MANUAL_VAL_MANIFEST:-$(dirname "$val_manifest")/manual_split/val.json}"
if [[ -f "$manual_val" ]]; then
    "$python_bin" tools/audit_manifest_overlap.py "$train_manifest" "$manual_val" \
        --fail-on-overlap --output "$run_dir/audit_train_manual_val.json"
fi
test_manifest="$(dirname "$val_manifest")/test_combined_284_scene_safe.json"
if [[ -f "$test_manifest" ]]; then
    "$python_bin" tools/audit_manifest_overlap.py "$train_manifest" "$test_manifest" \
        --additional-train "$val_manifest" --fail-on-overlap \
        --output "$run_dir/audit_generated_heldout_test.json"
fi

"$python_bin" tools/preflight.py --config "$config" --device cuda \
    --max-samples "${SMOKE_SCAN_SAMPLES:-64}" --optimizer-steps 2 \
    --expected-val-samples 412 \
    |& tee "$run_dir/preflight.log"
"$python_bin" train.py --config "$config" --device cuda |& tee "$run_dir/train.log"
"$python_bin" evaluate.py --config "$config" --device cuda \
    --checkpoint "$run_dir/best_phase_a.pt" \
    --output "$run_dir/validation.json" --rows-output "$run_dir/validation.jsonl" \
    |& tee "$run_dir/evaluate.log"
if [[ -f "$manual_val" ]]; then
    "$python_bin" evaluate.py --config "$config" --device cuda \
        --checkpoint "$run_dir/best_phase_a.pt" --manifest "$manual_val" \
        --output "$run_dir/manual_validation.json" \
        --rows-output "$run_dir/manual_validation.jsonl" \
        |& tee "$run_dir/evaluate_manual.log"
fi
printf '8B RDT training and validation completed: %s\n' "$run_dir"
