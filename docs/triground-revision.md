# 冻结定位器与独立修正分支

这是 2026-09-29 结构修复候选，CPU 已验证，尚无新 GPU 成绩。根因与实测证据见 `F:/AIC/docs/research/2026-09-29-tm-execution/ROOT_CAUSE.md`；数据与 CPU 交付见 `F:/AIC/docs/research/2026-09-29-structural-repair/README.md`。

## 路径

第一次用冻结的原定位适配器输出 RGB 框；第二次用单独的修正适配器，输入同样的原图、完整 Query 和实际预测框，返回 `{"action":"keep"}` 或 `{"action":"replace","bbox_2d":[...]}`。KEEP 原样复制第一步输出。REPLACE 必须给 RGB 0–1000 坐标。输出非法计失败，不用静默 KEEP 掩盖。

修正数据导出（输出必须是新目录）：

```powershell
python -m tools.prepare_grounding_revision --release-dir F:/AIC/results/triground_tm_20260929/data/release_600_seed2028 --baseline-dir F:/AIC/results/triground_tm_20260929/stage1/evaluation/A/fit232 --output-dir <新的输出目录>
```

当前真实导出为 `F:/AIC/results/triground_revision_20260929/data/verified_normal/train.json`。它是 215 个唯一样本，不是 600 步排程。仍用现有 `scripts/run_qwen3vl_native_lora.sh`，`ANNOTATION_PATH` 指向新清单，`LORA_SCOPE=language`、`LOSS_REDUCTION=sample_mean`；仅训练独立输出目录的修正适配器，不训练/覆盖作为第一步依赖的历史 A。步数、重复和初始化实验尚未冻结，不能直接照搬旧 run-all。

训练完成后的评估入口（**下面是用法，不是已执行命令**）：

```bash
python -m tools.evaluate_pretrained_grounder \
  --model /path/to/Qwen3-VL-8B-Instruct \
  --adapter /path/to/retained_A \
  --aux-adapter /path/to/trained_revision_adapter \
  --inference-mode ground_then_verify \
  --manifest /path/to/original_localization_manifest.json \
  --target-manifest /path/to/raw_float_gt.json \
  --data-root /path/to/data --output-dir /path/to/new_result \
  --max-new-tokens 128 --min-pixels 200704 --max-pixels 602112
```

传入的评估清单是普通 RGB 定位题，不是含 KEEP 答案的训练清单。GT 清单只用于评分，不能拼入提示。两个 LoRA 共用冻结基座，推理均 BF16。评估配置记录两适配器路径，恢复时不允许混用模式或辅助权重。

逐题输出保留第一步文本/框/网格、第二步原文、最终框和两次耗时。汇总 `revision_retention` 同时报原命中、最终命中、救回、损害、正确框被替换和错误框被保留。永远 KEEP 是必须比较的零增益对照，不能把“保住 A”当成性能提升。

## 辅助数据与评分

`prepare_depth_reading_pairs` 导出去重的正反两序裁图读取题；它使用人工 ROI，不能用于正式定位输入。`evaluate_aux_reading` 额外报告 `depth_relation.paired_order_check`，必须两序都答对才算该对通过。

`prepare_rgb_ir_controls` 导出人工目视核准的 RGB 足够/真实 IR 对照。每条正常与 IR 置空保持完整 Query、RGB 和原 GT 一致，评估需显式给它导出的 `gt.json`，避免使用训练框整数舍入后的 GT。新包 16 条只作描述性诊断；原 IR 自动升级驱动仍要求 32 条真正含 IR 的对照，未达到不能启动后续完整评估。

所有新目录与历史快照分开，原数据不覆盖。GPU 接续处于暂停状态。
