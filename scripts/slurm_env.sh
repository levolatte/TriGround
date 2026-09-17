#!/usr/bin/env bash
# Sourced after cd to the repository. Activate the prepared environment before sbatch.
set -euo pipefail
python_bin="${PYTHON:-python}"
export BACKBONE="${BACKBONE:-../models/Qwen3-VL-8B-Instruct}"
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
# Download the backbone on a login/download node beforehand.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

command -v "$python_bin" >/dev/null || {
  echo "Python executable is unavailable: $python_bin" >&2
  exit 1
}
if [[ "$BACKBONE" = /* || "$BACKBONE" = .* ]]; then
  [[ -s "$BACKBONE/config.json" ]] || {
    echo "Local backbone is incomplete: $BACKBONE/config.json" >&2
    exit 1
  }
fi

printf 'Job=%s Host=%s Repository=%s Python=%s Backbone=%s\n' \
  "${SLURM_JOB_ID:-local}" "$(hostname)" "$PWD" "$python_bin" "$BACKBONE"
if git_commit="$(git rev-parse HEAD 2>/dev/null)"; then
  printf 'GitCommit=%s\n' "$git_commit"
else
  printf 'GitCommit=unavailable (source archive without .git)\n'
fi
nvidia-smi
"$python_bin" - <<'PY'
import os, torch, transformers
if not torch.cuda.is_available():
    raise RuntimeError("This job requires an allocated CUDA GPU")
if not torch.cuda.is_bf16_supported():
    raise RuntimeError("The 8B configs require BF16 support")
total_mib = torch.cuda.get_device_properties(0).total_memory // 2**20
minimum_mib = int(os.environ.get("TRIGROUND_MIN_GPU_MEMORY_MIB", "23000"))
if total_mib < minimum_mib:
    raise RuntimeError(f"Qwen3-VL-8B requires at least {minimum_mib} MiB, found {total_mib} MiB")
print({"torch": torch.__version__, "transformers": transformers.__version__,
       "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
       "gpu_memory_mib": total_mib})
PY
