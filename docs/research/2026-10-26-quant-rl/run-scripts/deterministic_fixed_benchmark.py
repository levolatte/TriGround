"""Run the existing fixed-answer experiment with explicit PyTorch determinism."""
import json
import os
from pathlib import Path
import sys

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

ROOT = Path('/root/autodl-tmp/quant_rl_20261026')
sys.path.insert(0, str(ROOT / 'code'))
from tools.benchmark_quant_grounding import main

settings = {'torch_deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cublas_workspace_config': os.environ['CUBLAS_WORKSPACE_CONFIG']}
print(json.dumps({'deterministic_diagnostic': settings}), flush=True)
main()
variant = sys.argv[sys.argv.index('--variant') + 1]
output = Path(sys.argv[sys.argv.index('--output') + 1]) / variant
(output / 'determinism.json').write_text(json.dumps(settings, indent=2))
