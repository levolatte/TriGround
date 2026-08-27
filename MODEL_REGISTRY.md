# Model Registry

## `RECOMMENDED-TRIGROUND-RDT-WS-V1-MANUAL-FT1`

| Field | Value |
| --- | --- |
| Name | `TriGround-RDT-WS-v1-manual-ft1` |
| Release asset | `triground-rdt-ws-v1-manual-ft1.pt` |
| Architecture | `rdt_deep` multi-layer visual fusion |
| Initialization | `LANDMARK-TRIGROUND-RDT-WS-V1` |
| Fine-tuning | 923 manually reviewed target-domain samples, one epoch |
| Selection | Independent 119-sample validation split |
| Backbone | Frozen Qwen3-VL-2B-Instruct; Vision LoRA disabled |
| Config | `configs/triground_rdt_ws_v1_manual_ft1.yaml` |

On `combined284`: mIoU 0.5952, Acc@0.5 69.01%, Acc@0.7 54.93%.

## `LANDMARK-TRIGROUND-RDT-WS-V1`

| Field | Value |
| --- | --- |
| Name | `TriGround-RDT-WS-v1` |
| Release asset | `triground-rdt-ws-v1.pt` |
| Architecture | `rdt_deep` multi-layer visual fusion |
| Historical checkpoint | `runs/multimodal_rdt_deep_reviewed/best_phase_a.pt` |
| Best point | Phase A, epoch 5 |
| Training | Original 7,200-sample weak-supervision set |
| Backbone | Frozen Qwen3-VL-2B-Instruct; Vision LoRA disabled |
| Config | `configs/multimodal_rdt_deep_reviewed_extend_e5.yaml` |

On the complete 1,152-sample historical reviewed validation set, this was the
first three-modal model to exceed native RGB Acc@0.5: 62.41% (719/1152) versus
62.07% (715/1152). Its mIoU and Acc@0.7 remained below RGB, so it must not be
described as leading on every metric.

This checkpoint is not the early `legacy_patch` model. Reproduction and
fine-tuning must use `fusion_type: rdt_deep`.

## `BASELINE-TRIGROUND-PARALLEL-A-V1`

| Field | Value |
| --- | --- |
| Name | `TriGround-Parallel-A-v1` |
| Release asset | `triground-parallel-a-v1.pt` |
| Architecture | `parallel_backbone` with query-conditioned joint fusion |
| Historical checkpoint | `runs/stage2_joint_fusion_v2/last_phase_a.pt` |
| Config | `configs/stage2_joint_fusion_v2.yaml` |

On `combined284`: mIoU 0.5882, Acc@0.5 67.96%, Acc@0.7 55.28%. It is retained
because it represents the alternative fusion route and has slightly higher
strict Acc@0.7 than the recommended RDT checkpoint.

## Evaluation caveat

The `new154` portion of `combined284` overlaps the original weak-supervision
training data used by `TriGround-RDT-WS-v1`. Absolute `combined284` results are
not clean generalization estimates. Use independent scene-safe validation for
scientific claims. Comparing checkpoints on the same fixed split remains useful
as a controlled engineering diagnostic, but does not remove the leakage.

