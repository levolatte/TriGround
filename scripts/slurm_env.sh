#!/usr/bin/env bash
# Sourced after cd to the repository. Activate the prepared environment before sbatch.
set -euo pipefail
python_bin="${PYTHON:-python}"
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
# Download the backbone on a login/download node beforehand.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

printf 'Job=%s Host=%s Repository=%s Python=%s\n' "${SLURM_JOB_ID:-local}" "$(hostname)" "$PWD" "$python_bin"
git rev-parse HEAD
nvidia-smi
"$python_bin" - <<'PY'
import torch, transformers
if not torch.cuda.is_available():
    raise RuntimeError("This job requires an allocated CUDA GPU")
if not torch.cuda.is_bf16_supported():
    raise RuntimeError("The 8B configs require BF16 support")
print({"torch": torch.__version__, "transformers": transformers.__version__,
       "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda})
PY
