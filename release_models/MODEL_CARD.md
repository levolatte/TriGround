# TriGround representative checkpoints

These files contain only the project-specific trainable parameters. They do not
contain the Qwen3-VL-2B-Instruct backbone. Load them with this repository's
`load_model_checkpoint` function and the matching configuration.

## Recommended model

### `triground-rdt-ws-v1-manual-ft1.pt`

- Tag: `RECOMMENDED-TRIGROUND-RDT-WS-V1-MANUAL-FT1`
- Architecture: `rdt_deep`
- Initialization: `TriGround-RDT-WS-v1`
- Fine-tuning: 923 manually reviewed target-domain samples, one epoch
- Backbone: frozen
- Vision LoRA: disabled
- Config: `configs/triground_rdt_ws_v1_manual_ft1.yaml`

On `combined284`, this checkpoint obtained mIoU 0.5952, Acc@0.5 69.01%, and
Acc@0.7 54.93%. The RGB baseline obtained 0.5848, 67.25%, and 53.87%,
respectively.

## Alternative fusion baseline

### `triground-parallel-a-v1.pt`

- Tag: `BASELINE-TRIGROUND-PARALLEL-A-V1`
- Architecture: `parallel_backbone`
- Config: `configs/stage2_joint_fusion_v2.yaml`

On `combined284`, this checkpoint obtained mIoU 0.5882, Acc@0.5 67.96%, and
Acc@0.7 55.28%. It remains useful because its strict Acc@0.7 was one prediction
higher than the recommended RDT model.

## Evaluation caveat

The `new154` portion of `combined284` overlaps the original weak-supervision
training data used by `TriGround-RDT-WS-v1`. Therefore, absolute `combined284`
results are not clean generalization estimates. Keep the old scene-safe subset
and independent validation results when reporting scientific conclusions.

## License and base model

These artifacts contain adapter/fusion parameters only. Users must obtain the
Qwen3-VL-2B-Instruct backbone separately and comply with its license and the
licenses of the training datasets. No competition images or query manifests are
included in these artifacts.
