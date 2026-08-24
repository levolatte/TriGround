# RGB + IR + Depth Grounding

## Fast manual review

Double-click `start_reviewer.cmd`, or run:

```bash
python tools/review_grounding.py \
  --manifest ../city_detection_prepared/train/grounding_final_val.json \
  --data-root ../city_detection_prepared/train \
  --predictions ../qwen_rgb_error_analysis/results.jsonl \
  --reviews ../qwen_rgb_error_analysis/grounding_final_val_reviews.json \
  --open
```

The local reviewer groups queries into scenes, starts with predictions hidden,
supports RGB/infrared/depth switching and keyboard decisions, and saves every
decision atomically without changing the source manifest.

Default training data uses the final quality-controlled manifests:

- `../city_detection_prepared/train/grounding_final_train.json`
- `../city_detection_prepared/train/grounding_final_val.json`
- audit report: `../city_detection_prepared/train/grounding_final_report.json`

Regenerate them without overwriting the original labels or earlier manifests:

```bash
python tools/optimize_city_grounding.py \
  --data-root ../city_detection_prepared/train
```

The final conversion groups source sequences and RGB near-duplicates before
splitting, caps a scene at 12 queries and a class at 4 queries per scene, limits
geometry-only ordinal descriptions to the first three positions, and retains
small targets with explicit `scale_bin` and `processed_short_side` metadata.

本项目只实现一个三模态目标定位模型：

```text
Depth ─┐
       ├─ shared auxiliary encoder ─ bidirectional redundancy suppression ─┐
IR ────┘                                                                  ├─ D-TIR feature
RGB ──────────────────────────────── RGB-guided fovea prompt ──────────────┘
                                      ↓
                       gated residual fused visual patches
                                      ↓
                   Qwen3-VL native autoregressive grounding
                                      ↓
                    {"bbox_2d":[x1,y1,x2,y2]}
```

模型不包含 `<LOC>`、独立 bbox decoder、RGB-only 训练模型、单独 IR/Depth Adapter，
也不保留 legacy 模型。RGB-only 只作为评估时的原生 Qwen 基线。

## 训练

- Phase A：冻结整个 Qwen3-VL，只训练 RDT 风格三模态融合模块。
- Phase B：从最佳 Phase A checkpoint 继续，训练融合模块和 Qwen Vision 最后若干层的 LoRA。
- LLM 始终冻结。
- 监督目标是 Qwen 原生 `bbox_2d` JSON 文本，坐标范围为 `0–1000`。

```bash
python tools/preflight.py --config configs/multimodal.yaml --offline
python tools/preflight.py --config configs/multimodal.yaml --device cuda --backward
python train.py --config configs/multimodal.yaml
python evaluate.py --config configs/multimodal.yaml --checkpoint runs/multimodal/best.pt
```

数据清单必须包含对齐的 `visible`/`infrared`/`depth`、英文 `query`，以及归一化
`[x1,y1,x2,y2]` bbox。默认配置使用 `../city_detection_prepared`。
