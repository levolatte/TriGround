#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

"${PYTHON_BIN}" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade "pip==25.3" "setuptools==80.9.0" wheel
python -m pip install \
  "torch==2.8.0" "torchvision==0.23.0" \
  --index-url "${TORCH_INDEX_URL}"
python -m pip install -r requirements-experiment.txt
python -m pip install --no-deps -e .

python -c "import torch; print({'torch': torch.__version__, 'cuda_runtime': torch.version.cuda, 'cuda_available': torch.cuda.is_available()})"
python -m pip check
python -m compileall -q src tools train.py evaluate.py

echo "Environment installed. Run the real-model check with:"
echo "python tools/preflight.py --config configs/multimodal.yaml --device cuda --backward"
