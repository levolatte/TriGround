# Qwen3-VL-8B shared auxiliary embedding: launch guide

Branch: `qwen3-vl-8b-rdt-shared-aux`.
The sole training launcher is `scripts/qwen3_vl_8b_rdt_qwen_city.slurm`, using
`configs/qwen3_vl_8b_rdt_qwen_city.yaml`. The previous launchers remain on the
`qwen3-vl-8b` branch. The shared `train.py`, evaluation and environment tools
remain available here.

## Prepare the experiment machine

Run from the repository root in a CUDA environment with project dependencies
and BF16 support. Prepare the complete 8B weights and aligned RGB/IR/depth data.

```bash
export PYTHON=python
export BACKBONE=../models/Qwen3-VL-8B-Instruct
export CITY_ROOT=../city_detection_prepared
```

Manifests under `$CITY_ROOT/train/target_v2/`:
`qwen_generation_train_100.json` (3707 records) and
`qwen_generation_val.json` (412 records). Image paths must resolve on this
machine. Model loading is offline; a previously cached Hugging Face model ID
can also be used for BACKBONE.

## Preflight only (no full training)

```bash
PREFLIGHT_ONLY=1 bash scripts/qwen3_vl_8b_rdt_qwen_city.slurm
# With Slurm:
PREFLIGHT_ONLY=1 sbatch scripts/qwen3_vl_8b_rdt_qwen_city.slurm
```

This checks data overlap, selects the largest visual-token sample among the
first 64 training records, and runs two AdamW steps. It records memory, timing,
gradients and actual shared-embedding weight updates, then exits. Scanning 64
records does not guarantee the full dataset's largest sample is covered; set
`SMOKE_SCAN_SAMPLES=3707` to scan every training record.

## Full training

```bash
PREFLIGHT_ONLY=0 bash scripts/qwen3_vl_8b_rdt_qwen_city.slurm
# With Slurm:
PREFLIGHT_ONLY=0 sbatch scripts/qwen3_vl_8b_rdt_qwen_city.slurm
```

Order: overlap audit -> two-step GPU preflight -> two epochs of Phase A ->
best-checkpoint validation. An available manual validation manifest receives
an additional evaluation. The independent test manifest is checked only for
overlap. Any failure stops the pipeline, including failures piped through tee.

Defaults: one GPU, eight CPUs, 64 GB host RAM and a 72-hour job limit. The time
limit is not a runtime estimate. Environment checks require roughly 23 GB GPU
memory, but actual fit must be confirmed by the new preflight. Supply cluster
partition/account settings through sbatch options.

## Paths and results

- `REPO_DIR`: repository root; defaults to Slurm submission directory/current directory.
- `RUN_ROOT`: new, nonexistent output directory; defaults to a job ID/timestamp path.
- `MANUAL_VAL_MANIFEST`: optional manual validation manifest override.
- `SMOKE_SCAN_SAMPLES`: preflight scan size, default 64.

Each run saves `config.yaml`, `audit_*.json` and `preflight.log`. Full training
also saves training logs, checkpoints and validation JSON/JSONL. Separate
preflight and training jobs use separate run directories. Preflight's trial
weights are not carried into training. The launcher does not resume old runs.

CPU checks validate code and launcher control flow. Full 8B CUDA memory,
throughput and model quality remain to be tested on the experiment machine.
The older independent-encoder GPU results do not validate this branch.
