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
configs/qwen3_vl_8b_stage2_joint.yaml
```

Stage 2 merges the two sparse Stage 1 checkpoints with collision detection.
An overlapping key is accepted only when both tensors are identical; a
different value, unknown key, or shape mismatch is an error.

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
