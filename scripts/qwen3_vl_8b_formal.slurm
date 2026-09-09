#!/usr/bin/env bash
#SBATCH --job-name=triground-8b-formal
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
run_dir="${RUN_ROOT:-$PWD/runs/slurm-formal-${SLURM_JOB_ID:?submit with sbatch}}"
[[ "$run_dir" = /* ]] || run_dir="$PWD/$run_dir"
"$python_bin" tools/prepare_slurm_run.py --config-dir "${CONFIG_DIR:-configs}" --run-dir "$run_dir"
mkdir -p "$run_dir/logs"

# Sequential single-GPU execution. Each stage loads only this run's predecessors.
for stage in stage1a_ir stage1b_depth stage2_joint stage2_weak stage2_clean; do
    config="$run_dir/configs/qwen3_vl_8b_${stage}.yaml"
    extra=()
    if [[ "$stage" == stage2_* ]]; then extra=(--expected-val-samples 119); fi
    "$python_bin" tools/preflight.py --config "$config" \
        --device cuda --max-samples 4 --optimizer-steps 2 \
        "${extra[@]}" \
        |& tee "$run_dir/logs/preflight_${stage}.log"
    "$python_bin" train.py --config "$config" --device cuda \
        |& tee "$run_dir/logs/train_${stage}.log"
    test -s "$run_dir/$stage/best_phase_a.pt"
done

config="$run_dir/configs/qwen3_vl_8b_stage2_clean.yaml"
checkpoint="$run_dir/stage2_clean/best_phase_a.pt"
eval_manifest="$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1]))["eval_manifest"])' "$run_dir/plan.json")"

# No subset limit: evaluate the full manifest. The default is the 119-query validation
# set. EVAL_MANIFEST can name a separately labelled holdout; never official queries.
"$python_bin" evaluate.py --config "$config" --device cuda \
    --manifest "$eval_manifest" --rgb-only \
    --output "$run_dir/native_rgb.json" --rows-output "$run_dir/native_rgb.jsonl" \
    |& tee "$run_dir/logs/native_rgb.log"
"$python_bin" evaluate.py --config "$config" --checkpoint "$checkpoint" --device cuda \
    --manifest "$eval_manifest" \
    --output "$run_dir/final_evaluation.json" --rows-output "$run_dir/final_evaluation.jsonl" \
    |& tee "$run_dir/logs/final_evaluation.log"
printf 'Five-stage training and full evaluation completed: %s\n' "$run_dir"
