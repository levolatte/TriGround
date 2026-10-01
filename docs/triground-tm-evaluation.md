# T/M 读取探针与 IR 两次推理评估接口

状态：2026-09-29 本地实现与 CPU 逻辑验证完成；尚未在真实 GPU 模型上运行。比赛正式输出仍按 RGB 图归一化框评分。此点已与 `contest/全球校园人工智能算法精英大赛 多模态赛道.html` 的标签和输出要求核对。

## 输入约定

正式 bbox manifest 可用现有 SFT `image` / `conversations` 格式，新增 `modalities` 与 `image` 同序（`rgb`、`ir`、`depth`）、原始完整 `query`。黑图占位另写 `missing_modalities_actual`（如 `["ir"]`）。两次模式遇到没有显式模态顺序的多图记录会直接报错，避免猜测第二张图为 IR。正式评分建议始终指定同一个原始浮点 GT manifest；SFT 的千分位答案只作为没有独立 GT 时的既有兼容路径。

读取 manifest 可混合 `task_type=ir_bbox` 与 `depth_relation`。每条保留 `image`、`modalities`、`conversations` 的一条 human 提示词以及独立 `expected_answer`。IR 读取答案为 `{"bbox_2d":[x1,y1,x2,y2]}` 或 `{"bbox_2d":null}`，坐标系 `qwen_0_1000`；另以 `ir_gt_bbox` 保留审核原始 0–1 浮点框作评分 GT，unknown 时为 null。历史记录缺此字段才用千分位答案评分。Depth 读取答案为 `A_nearer`、`B_nearer`、`unknown`，坐标系 `categorical`。两个 Depth 裁片可使用 `modalities=["depth","depth"]`。`expected_answer`、`ir_gt_bbox` 与 assistant 内容从不进入模型输入。

## 调用

```powershell
python -m tools.evaluate_pretrained_grounder --model MODEL --adapter ADAPTER --manifest T_OR_M_JSON --target-manifest RAW_GT_JSON --data-root DATA_ROOT --output-dir OUTPUT --inference-mode direct
python -m tools.evaluate_pretrained_grounder --model MODEL --adapter ADAPTER --manifest T_OR_M_JSON --target-manifest RAW_GT_JSON --data-root DATA_ROOT --output-dir OUTPUT_IR --inference-mode ir_then_ground
python -m tools.evaluate_aux_reading --model MODEL --adapter ADAPTER --manifest MIXED_READING_JSON --data-root DATA_ROOT --output-dir OUTPUT_READ
```

不用 Adapter 时省略 `--adapter`。正式 bbox 默认 `direct`，保留原单次调用与原 `--max-new-tokens` 默认值。`ir_then_ground` 对真实 IR 先只给 IR 图和完整 Query，最多 64 tokens；首答有效框才将 IR 千分位框及其坐标系作为可能错误的建议加入第二次原始全图提示，最终最多 128 tokens。首答 `null` 或非法时第二次提示词保持原样；无 IR 或 IR 黑图占位则仅做原单次 direct 调用。没有候选框、裁剪、GT 注入或生成重试。

正式输出为 `predictions.jsonl`、`summary.json`、`run_config.json`。两次模式逐行保存 `first_raw_text`、`first_box_ir`（IR 千分位）、`first_status`、`calls`、`total_latency_seconds`、`final_prompt` 及最终 `raw_text`；汇总单列首答解析失败和调用数。最终框继续按原始浮点 GT 计算 IoU、ACC@0.5，未解析框保留在全样本分母。读取输出为 `predictions.jsonl` 和 `summary.json`，IR 分别报告已知目标的全分母 IoU/ACC、有效框内 IoU/ACC 和 unknown 识别；Depth 按三种真实答案分表及混淆计数。

## 验证与交接

`tests/test_tm_evaluation.py` 覆盖真实双图无 IR、IR 黑图占位、IR null/非法输出、IR 与 RGB 坐标隔离、监督答案不进提示词、direct 单调用、缺失多图模态顺序报错、混合探针与双 Depth 裁片、空读取集、未解析样本全分母及统一图像转 RGB。使用 `code/.venv/Scripts/python.exe -m pytest tests/test_evaluate_pretrained_grounder.py tests/test_tm_evaluation.py -q`，共 24 项通过；语法编译及 `git diff --check` 亦通过。实际预览数据 `preview_400_seed2028` 的正式诊断四臂均 87 条：normal/Depth 缺失臂各 55 条有真实 IR，IR 缺失/双缺失臂均 0 条启动 IR 首轮；Depth 读取 46 条可读，当前 IR 读取探针为空集，已允许明确输出 0 样本汇总。没有使用 GPU，也没有改训练或数据代码。
