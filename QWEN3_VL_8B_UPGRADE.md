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

## Early IR/Depth complementary fusion on the new City data

`configs/qwen3_vl_8b_rdt_qwen_city.yaml` is a separate 8B experiment using the
`rdt_deep` implementation with shared auxiliary patch embedding, an
RDTTrack-inspired Qwen adaptation.
Its IR and depth branches attenuate their overlapping feature components
before a recurrent prompt is injected at every
Qwen vision block. Qwen's language and vision weights stay frozen. The run
starts from the 8B backbone; 2B fusion checkpoints are incompatible.

`rdt_shared_aux_patch_embed: true` creates one trainable copy of Qwen's native
RGB patch embedding, initialized from its pretrained weights. IR and depth
both call that same auxiliary module and shared normalization, followed by
separate modality projections. The original RGB patch embedding stays frozen.
The option defaults to false to preserve historical independent-encoder
configurations and checkpoint layouts. Independent-encoder checkpoints cannot
be used as a direct resume of the shared-embedding experiment.

The training manifest contains 3,707 Qwen-generated queries and the validation
manifest contains 412 queries. Both are under
`../city_detection_prepared/train/target_v2/` by default. These labels passed
the local rules and image checks described in the dataset README, but are not
equivalent to manually reviewed labels. The training manifest has no ID, image,
or sequence overlap with the independent 284-sample test manifest.

After placing the 8B backbone and the three aligned City image directories on
the GPU machine, run the dedicated preflight, training, and validation job:

```bash
export CITY_ROOT=../city_detection_prepared
sbatch scripts/qwen3_vl_8b_rdt_qwen_city.slurm
```

On a GPU machine without Slurm, run the same script directly with an existing
Python environment, for example:

```bash
PYTHON=/root/miniconda3/bin/python bash scripts/qwen3_vl_8b_rdt_qwen_city.slurm
```

`BACKBONE`, `CITY_ROOT`, `RUN_ROOT`, `PYTHON`, and `MANUAL_VAL_MANIFEST` can
override the default locations. Each job writes its resolved config, overlap
audit, preflight log, checkpoints, and validation results into one new run
directory. The preflight
must succeed before training starts. Validation uses the selected Phase A
checkpoint. When the original manually reviewed validation manifest is
available, the script also audits its separation from training and evaluates
it without using it for checkpoint selection. When the independent 284-sample
test manifest is available, the script audits overlap but does not evaluate it.

### Paper correspondence audit (2026-09-23)

Sources: [paper, Sections 4.1–4.3](https://arxiv.org/html/2509.24741v1#S4),
[official implementation, commit 794de41](https://github.com/xuefeng-zhu5/RDTTrack/blob/794de41ff3f52ed100e4f449cf22ceb4932e0d36/lib/models/rdtt/vit_ce_prompt.py).
The current model is **an adaptation, not an exact RDTTrack reproduction**.

| Component | Correspondence and differences |
| --- | --- |
| Fusion arithmetic | Local `SafePostEmbedFusion._remove_overlap` uses channel-summed dot products and squared norms. The reference `DepthIR_ort_block` uses elementwise products and an unsquared channel norm. These are different operators. |
| Auxiliary embedding | The new 8B City config shares one auxiliary patch embed before modality-specific projections, matching the reference's sharing pattern. Historical `rdt_deep` configs retain independent encoders; `legacy_patch` already used a shared auxiliary MLP encoder. |
| Recurrent prompts | Both carry the preceding prompt into the next layer. Local projections add GELU, use separate normalization parameters, and expand prompt width to 256; reference prompt width is 8. |
| Fovea | Local per-image `softmax(scale * features)` follows reference code. The paper's displayed Eq. (5) places lambda differently. |
| Initialization/injection | Local restore weights start at zero and injection has a learned scale initialized to 0.001, plus modality dropout. These are project choices. |
| Task/backbone | Frozen OSTrack, template/search tracking and focal/L1/GIoU losses become frozen Qwen, full-image language grounding and bbox token cross-entropy. Qwen retains its positional encoding and DeepStack. |

For vectors `d,t`, local subtraction is
`d - sigmoid(alpha_logit) * sum(d*t) / max(sum(t*t), eps) * t`.
The reference instead subtracts
`sigmoid(alpha) * (d*t) / (norm(t)+eps) * t`.
Eq. (2) describes an inner product with an unsquared denominator, so paper,
reference code, and this adaptation must be distinguished. Local subtraction
is only fully orthogonal to the original other vector when its coefficient
is one (away from the epsilon clamp); there is no separate orthogonality loss.

After the shared-embedding correction, 44 adapter/model/config CPU tests passed,
including two training steps, shared-module calls for both modalities, frozen
RGB weights, and BF16 auxiliary embedding with FP32 fusion projections. A
non-collinear vector example in the earlier audit confirmed the overlap
operators differ; that arithmetic is unchanged by this correction.

Earlier GPU execution, memory and timing results concern the independent-encoder
version. The shared-embedding version needs renewed GPU preflight on the
experiment machine before full training. Earlier runtime estimates are
provisional and must be updated from that preflight.

### Suitability audit after the shared-embedding correction (2026-09-23)

**Decision:** the architecture is a plausible Qwen grounding adaptation of the
paper's complementary-fusion idea. Execution and gradient tests do not establish
an accuracy gain, nor the same gain measured on the paper's tracking task.

Required implementation fixes made in this audit:

- The copied trainable auxiliary patch embedding now keeps FP32 parameters.
  Copying the BF16 backbone without conversion left AdamW updating BF16 weights
  directly. At the configured learning rate of `4e-5`, a scalar diagnostic with
  unit gradients produced zero updates for BF16 weights near 0.02 and 0.1, while
  FP32 versions updated. This demonstrates a rounding risk, not the fraction of
  actual model weights affected. Autocast can still compute the embedding in BF16.
- Preflight now uses the training AMP dtype and gradient scaler policy. Previously
  it omitted autocast, so it did not validate the same numerical path as training.
  On the second shared-embedding optimizer step it checks gradient reachability,
  FP32 parameter storage, and actual parameter changes, and records update fraction.

Design judgments and remaining experimental questions:

| Area | Judgment | Action |
| --- | --- | --- |
| Qwen integration | Patch order/grid checks, native RGB positions/rotary embeddings, all vision blocks, DeepStack and final merger are retained. Auxiliary tokens enter at aligned RGB positions. | Keep. No separate auxiliary positional embedding is required merely to make this pointwise injection spatially aligned. |
| Shared auxiliary embedding | Sharing constrains parameter count; separate projections permit modality-specific features. Depth is encoded as log-distance/validity/zero channels, while IR uses image intensities, so their input statistics differ. | Keep the shared trainable embedding and normalization; test each modality's usefulness. Sharing alone does not prove feature alignment. |
| Overlap subtraction | For fixed other-modality features and coefficient in [0,1], the local vector projection shrinks the parallel component without increasing feature norm. At initialization the retained parallel coefficient is about 0.378. Independently learned projections do not ensure semantic redundancy is removed. | Keep as a defensible operator. Reference arithmetic is an optional controlled ablation, not a necessary Qwen compatibility fix. |
| Recurrent prompts | Zero restore weights preserve RGB exactly at initialization, but the initial auxiliary signal is blocked and later prompts initially receive zero preceding prompts. Two-step tests verify gradients become available; they do not prove useful information survives all 27 layers. | Keep for the initial pilot; inspect auxiliary sensitivity and per-layer injection magnitude before increasing scale or changing initialization. |
| Injection scale | `0.001` plus zero restore can yield very small early changes, some rounded away in BF16 additions. | Compare actual forward changes as well as gradients. If negligible after a short pilot, test scale 0.01 as a separate experiment; do not assume improvement. |
| Capacity/data | The 27 prompt blocks at width 256 alone contain 24,057,270 parameters, while training has 3,707 query records. Records may share scenes/images and are not independent examples. | Watch the train/validation gap; consider width 64 or 128 only if overfitting appears. Scaling the language model to 8B does not itself require width 256. |
| Task supervision | Qwen bbox-token loss preserves the current language-grounding output interface; it differs from tracking classification and geometric losses. | Keep the current objective for the baseline. Add geometric supervision only as a separately measured change. |

Validation: **46 CPU adapter/model/config/preflight tests passed**. The added
native Qwen vision test uses a small randomly initialized three-layer vision
model, BF16 autocast, gradient checkpointing and three DeepStack outputs. It
checks initial RGB equivalence, frozen vision weights, and nonzero auxiliary
gradients and weight changes after two updates. It is not an 8B checkpoint or
CUDA test, and cannot establish full-model memory use or accuracy.

On the experiment machine, first repeat the two-step GPU preflight, then use a
50-100 optimizer-step pilot before committing to the full run. Compare the
frozen RGB baseline, trained full fusion, and the same trained checkpoint with
IR or depth masked (using the training modality-dropout convention). Include
an RGB-only trainable-prompt control to separate auxiliary information gains
from added trainable capacity; use a matched no-overlap-subtraction ablation to
attribute gains specifically to the projection operator. Select changes on
validation data and preserve the held-out test set for the final evaluation.
Compare IoU/Acc@0.5 and parsing validity on the same queries, particularly the
manually reviewed validation set; generated labels alone are insufficient to
claim equivalent benefits to the paper. No full training was run in this audit.
