# Qwen3-VL-8B TriGround upgrade

The recommended path keeps the Qwen language and vision backbones frozen and
does not inject or train Vision LoRA. It uses the native Qwen3-VL-8B vision
configuration and reads all backbone dimensions at runtime.

## Recommended E configuration

- Backbone: `Qwen/Qwen3-VL-8B-Instruct`, BF16.
- IR/depth adaptor bottleneck: 256.
- Fusion width: 512 with 8 attention heads.
- Fusion mixer width: 2048.
- Two independent query encoders: width 128, one layer, 4 heads.
- Fusion layers: `[8, 16, 24, 26]`. The first three align with the native
  DeepStack extraction layers; the last is the final vision block.
- Vision LoRA: disabled in every stage (`phase_b_epochs: 0`).

The parallel vision forward applies fusion before collecting a DeepStack
feature. Consequently, the features extracted after blocks 8, 16, and 24 are
multimodal, while preserving Qwen's native DeepStack mergers and LLM injection
path.

## Run order

```text
configs/qwen3_vl_8b_stage1a_ir.yaml
configs/qwen3_vl_8b_stage1b_depth.yaml
configs/qwen3_vl_8b_stage2_joint.yaml  # calibration on reviewed manual data
configs/qwen3_vl_8b_stage2_weak.yaml   # scene-safe weak adaptation
configs/qwen3_vl_8b_stage2_clean.yaml  # reviewed clean recovery; final candidate
```

Stage 2 merges the two sparse Stage 1 checkpoints with collision detection.
An overlapping key is accepted only when both tensors are identical; a
different value, unknown key, or shape mismatch is an error.

## Target-domain data policy

The manually reviewed, combined, test-safe source contains 1,042 samples. Its
group-safe split uses 923 samples from 249 sequences for training and 119
samples from 24 sequences for validation. The two manifests are disjoint and
their union is the complete 1,042-sample source; no eligible reviewed sample is
discarded. Records inherited from the first review round can retain the legacy
`weak_label: true` field, so supervision provenance is determined from
`label_source: manual_review_v1|manual_review_v2` and `review_decision`.

Weak supervision is not mixed into the reviewed loader. The intermediate weak
stage uses `train_weak_scene_safe.json`: 992 samples from 992 distinct scenes,
after removing sequences held out by the 284-sample independent test set. The
final clean-recovery stage returns to the 923 reviewed training samples and
selects its checkpoint only on the 119 reviewed validation samples. The
independent 284-sample test set remains untouched by training and selection.

## RGB controls

Use the native 8B checkpoint as the untrained RGB baseline. A separately
trained RGB-only control is intentionally omitted because the selected policy
freezes the complete Qwen backbone and introduces no RGB-only trainable module.
For the trained TriGround checkpoint, report both multimodal inference and its
`rgb_only` path; this measures the gain from enabled auxiliary fusion without
changing the backbone or training a separate RGB model.

## Initialization and gradients

Adaptor up projections and fusion restore projections are zero initialized.
The first backward pass is expected to update the restore/up projection while
upstream fusion layers may have zero gradients. After the first optimizer
update, gradients must reach upstream mixer, attention, query, and adaptor
parameters. Frozen Qwen parameters must never receive parameter gradients.

Training prints a JSON parameter report with total and trainable counts for the
model, backbone, fusion, modality adaptors, query encoders, and active fusion
stages. Checkpoints record their format version, actual saved parameter names,
PyTorch version, Transformers version, full configuration, seed, and optimizer
state.

Before a full run on an L40, execute a complete optimizer step using a sample
at the configured maximum visual-token budget and record peak allocated and
reserved GPU memory, token count, and step time.

After the model is available in the local Hugging Face cache, run:

```bash
scripts/run_qwen3_vl_8b_preflight.sh
```

The preflight scans 64 training samples, selects the one with the largest
processed visual-token count, and runs two AdamW steps. The second step verifies
that gradients pass beyond the zero-initialized restore projections.
