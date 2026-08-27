# TriGround: RGB–IR–Depth Referring Grounding

TriGround extends Qwen3-VL-2B-Instruct with lightweight RGB/infrared/depth
adaptation and fusion for referring-expression grounding. The language model
and visual backbone remain frozen in the released configurations; only the
project-specific adaptor and fusion parameters are trained.

The model receives aligned visible, infrared, and depth images plus a text
query, and generates a normalized visible-image bounding box.

```text
RGB ──────────────── primary visual path ───────────────┐
IR ── adaptor ──┐                                      ├─ Qwen bbox generation
Depth ─ adaptor ─┴─ query-aware residual fusion ────────┘
```

## Released model families

| Model | Fusion | combined284 mIoU | Acc@0.5 | Acc@0.7 |
| --- | --- | ---: | ---: | ---: |
| `TriGround-RDT-WS-v1-manual-ft1` | RDT-deep | **0.5952** | **69.01%** | 54.93% |
| `TriGround-Parallel-A-v1` | Parallel joint fusion | 0.5882 | 67.96% | **55.28%** |
| Native RGB baseline | None | 0.5848 | 67.25% | 53.87% |

The RDT checkpoint is the recommended general model. The parallel checkpoint
is retained as a meaningful architectural comparison and has slightly higher
strict Acc@0.7 on this split.

See [MODEL_REGISTRY.md](MODEL_REGISTRY.md) and
[release_models/MODEL_CARD.md](release_models/MODEL_CARD.md) for provenance,
training details, limitations, and checkpoint hashes.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On Windows, activate the environment with `.venv\Scripts\activate`.

The released checkpoints contain only TriGround parameters. Download
Qwen3-VL-2B-Instruct separately through Transformers or point `model.backbone`
in a config to an existing local snapshot.

## Data format

Manifests are JSON objects keyed by query ID. Each record contains aligned
modalities, a query, and a normalized `xyxy` box:

```json
{
  "query-id": {
    "visible": "Images/visible/000001.png",
    "infrared": "Images/infrared/000001.png",
    "depth": "Images/depth/000001.png",
    "query": "the pedestrian beside the vehicle",
    "bbox": [0.12, 0.21, 0.38, 0.84]
  }
}
```

All paths are resolved relative to the manifest unless a tool exposes an
explicit data-root argument. Training boxes must satisfy
`0 <= x1 < x2 <= 1` and `0 <= y1 < y2 <= 1`.

## Training

The public configs use the Hugging Face model ID and relative data paths. Edit
the manifest paths for your local dataset layout.

### RDT-deep route

The historical weak-supervision run used two initial epochs followed by the
recorded low-learning-rate extension to epoch 5:

```bash
python tools/preflight.py --config configs/multimodal_rdt_deep_reviewed.yaml
python train.py --config configs/multimodal_rdt_deep_reviewed.yaml
python train.py --config configs/multimodal_rdt_deep_reviewed_extend_e5.yaml
```

Target-domain manual fine-tuning starts from the historical RDT checkpoint:

```bash
python train.py --config configs/triground_rdt_ws_v1_manual_ft1.yaml
```

### Parallel-fusion route

This route trains independent IR and depth paths, calibrates them jointly, and
then trains query-conditioned reliability fusion:

```bash
python train.py --config configs/stage1a_ir.yaml
python train.py --config configs/stage1b_depth.yaml
python train.py --config configs/stage2_joint_calibration.yaml
python train.py --config configs/stage2_weak1024_raw.yaml
python train.py --config configs/stage2_clean_after_weak1024.yaml
python train.py --config configs/stage2_joint_fusion_v2.yaml
```

## Evaluation

Use the config that matches the checkpoint architecture:

```bash
python evaluate.py \
  --config configs/triground_rdt_ws_v1_manual_ft1.yaml \
  --checkpoint triground-rdt-ws-v1-manual-ft1.pt \
  --manifest path/to/validation.json \
  --output evaluation.json
```

For the alternative model, use `configs/stage2_joint_fusion_v2.yaml` with
`triground-parallel-a-v1.pt`.

## Competition-style prediction

The submission tool preserves every source field, replaces only `bbox`, checks
query order and normalized box validity, supports resumable JSONL progress, and
creates a ZIP containing the final JSON file.

```bash
python tools/predict_competition_submission.py \
  --config configs/triground_rdt_ws_v1_manual_ft1.yaml \
  --checkpoint triground-rdt-ws-v1-manual-ft1.pt \
  --queries path/to/queries.json \
  --data-root path/to/dataset \
  --progress outputs/progress.jsonl \
  --output-json outputs/predictions.json \
  --output-zip outputs/predictions.zip
```

## Checkpoint format

Release checkpoints contain:

- project-specific model tensors;
- the architecture/training config;
- epoch, step, and selection score metadata;
- a stable artifact name and tag.

They exclude the Qwen backbone, optimizer, scheduler, scaler, datasets, and
competition inputs. Load them with `mm_grounding.checkpoint.load_model_checkpoint`.

## Reproducibility and limitations

- The code test suite passes on CPU; actual training requires a CUDA GPU.
- Sensor alignment and depth units must match the manifest/config assumptions.
- The `new154` portion of `combined284` overlaps the original weak-supervision
  training set. Absolute `combined284` scores are therefore not clean
  generalization estimates. Use independent scene-safe validation for claims.
- Released weights are adapters/fusion parameters and require the matching base
  model and configuration.

## Tests

```bash
pytest -q
ruff check .
```

## License

No redistribution license has been selected yet. Copyright remains with the
project owner. Add an explicit code/model license before accepting third-party
reuse or contributions. Qwen and all datasets retain their own licenses.
