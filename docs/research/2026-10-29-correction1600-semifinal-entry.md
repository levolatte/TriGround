# 纠错1600半决赛推理入口

新增 `tools/predict_quant_submission.py`，用于将官方无标签查询接入纠错模型的量化推理流程。

- 直接复用 `eval_quant_grounding.generate_rows`：HF（Hugging Face 推理引擎）、NF4（4位正态量化）、短框模式、贪心解码、64词元上限；默认图像像素范围200704～1204224、上下文8192词元。
- 三图顺序为RGB、红外、深度；提示来自 `prepare_qwen3vl_native_sft.native_prompt`。16位单通道毫米深度复用训练阶段的逆灰度映射，8位RGB现成深度可视化保留原图像素；不改动官方原图。清单按实际编码记录 `millimeter` 或 `visual`，配置记录混合编码策略，三图提示保持既有毫米深度口径。
- 默认使用纠错评价的 `punctuation` 标点规范化；只修复已定义的四坐标标点错误，不改坐标、不选择多个框、不提供替代框。
- 复用 `export_quant_submission.make_submission`，保留官方查询顺序及所有非 `bbox` 字段；默认要求5691条查询完整覆盖，拒绝缺失、重复、非法或截断答案。
- 全量推理保存原始生成 `predictions_raw.jsonl`，并生成 `predictions.json` 和根目录仅含该JSON的 `submission.zip`。
- `--prepare-only --limit N` 只准备前N条查询的图像与无标签清单；`--limit N` 推理不生成完整提交包。`--predictions PATH` 支持仅用CPU将保存的原始结果打包。

## 云端运行

```bash
python tools/predict_quant_submission.py \
  --queries DATA/queries/queries.json --data-root DATA \
  --model /root/autodl-tmp/rematch_models/Qwen3.6-27B \
  --adapter /root/autodl-tmp/quant_rl_20261026/runs/acc_correction_20261029/correction/train/checkpoint-1600 \
  --output-dir OUT --format-policy punctuation
```

入口依赖：`eval_quant_grounding.py`、`export_quant_submission.py`、`prepare_qwen3vl_native_sft.py`、`quant_grounding_reward.py`、`quant_grounding_postprocess.py`、`quant_grounding_config.py`、`verify_acc05_dev.py`。生成所需第三方依赖与现有纠错评价一致；准备阶段使用NumPy和Pillow。

推理输出目录需全新，避免覆盖已有原始输出。准备后的清单记录本机绝对路径，跨机器时应在目标机器重新准备。
