#!/usr/bin/env bash
#SBATCH --job-name=triground-8b-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
cd "${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
source scripts/slurm_env.sh
run_dir="${RUN_ROOT:-runs/smoke-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}}"
[[ "$run_dir" = /* ]] || run_dir="$PWD/$run_dir"
"$python_bin" tools/prepare_slurm_run.py --config-dir "${CONFIG_DIR:-configs}" --run-dir "$run_dir" --smoke
mkdir -p "$run_dir/logs"
"$python_bin" -m pytest -q |& tee "$run_dir/logs/pytest.log"

# Independent IR, Depth, and fresh Joint Fusion: no trained checkpoint prerequisite.
for stage in stage1a_ir stage1b_depth stage2_joint; do
    extra=()
    if [[ "$stage" == stage2_joint ]]; then extra=(--expected-val-samples 119); fi
    "$python_bin" tools/preflight.py \
        --config "$run_dir/configs/qwen3_vl_8b_${stage}.yaml" \
        --device cuda --max-samples "${SMOKE_SCAN_SAMPLES:-64}" --optimizer-steps 2 \
        "${extra[@]}" \
        |& tee "$run_dir/logs/preflight_${stage}.log"
done

# Exercise generation, bbox parsing and all four modes on two samples.
# Random/zero-initialized fusion here is only a pipeline check, not an accuracy result.
"$python_bin" evaluate.py \
    --config "$run_dir/configs/qwen3_vl_8b_stage2_joint.yaml" \
    --device cuda --subset-size 2 \
    --output "$run_dir/smoke_generation.json" \
    --rows-output "$run_dir/smoke_generation.jsonl" \
    |& tee "$run_dir/logs/smoke_generation.log"
"$python_bin" - "$run_dir/smoke_generation.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)
for mode in ("rgb_baseline", "rgb_ir", "rgb_depth", "rgb_ir_depth"):
    if report[mode]["parse_rate"] <= 0:
        raise RuntimeError(f"No generated bbox parsed successfully in {mode}")
PY
printf 'Smoke checks completed: %s\n' "$run_dir"
