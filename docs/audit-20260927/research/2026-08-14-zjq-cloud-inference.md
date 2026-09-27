# zjq 云电脑模型首版官方测试集推理记录

日期：2026-08-14

## 数据与环境

- 本地官方测试集：6001 个文件，约 13.1 GB。
- 测试 Query：9555 条；visible、infrared、depth 各 2000 张。
- 云电脑：NVIDIA GeForce RTX 4090 D，24 GB 显存。
- 运行环境：Conda base、PyTorch 2.8.0+cu128、Transformers 5.14.1、PEFT 0.20.0、bitsandbytes 0.50.0。
- 使用 checkpoint：`runs/stage2_city_multimodal`，基座为远端 `models/Qwen3-VL-2B-Instruct`。

## 推理过程

使用队友仓库 `code/predict.py` 读取三模态和 Query，按四个不重叠分片并行推理。分片 0、2、3 首次或重试完成；分片 1 曾因模型输出 `[{161,659,176,690}]` 这类非标准括号格式退出，随后用临时解析器重跑成功。临时解析器只在严格解析失败时提取一组四坐标并调用原有 `sanitize_bbox`，没有修改仓库源文件。

## 结果与校验

- 合并结果：9555 条。
- 本地 JSON：[zjq_stage2_official_test_predictions.json](../../results/zjq_stage2_official_test_predictions.json)
- 提交 ZIP：[zjq_stage2_official_test_submission.zip](../../results/zjq_stage2_official_test_submission.zip)
- 原始模型输出：[zjq_stage2_official_test_raw_outputs.json](../../results/zjq_stage2_official_test_raw_outputs.json)
- 远端 `tools/check_submission.py` 校验通过：QueryID 完整匹配、非 `bbox` 字段保持不变、所有 bbox 有限且归一化。
- 本地复核通过：9555 条预测、9555 条 raw 输出、0 个非法 bbox、0 条字段变更；ZIP 仅含 `predictions_final.json` 且内容一致。

## 限制

官方测试集没有标签，因此本次只能验证提交文件格式，不能计算 ACC@0.5 或判断实际比赛名次。
