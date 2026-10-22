# 2026-10-22 当前 Qwen3.6-27B 训练与提交链路接手审计

## 审计边界与结论

本次只读检查 `code/` 中 10 月 2 日以来的 Qwen3.6-27B 训练、评估、推理、打包链路，并核对本地 `results/`、`deliverables/作品材料/` 与文档。没有运行 GPU、重新训练、联网下载或改动代码。比赛指标和提交格式以 `contest/` 为准：ACC@0.5 按 IoU≥0.5 的查询比例计，预测框采用归一化 xyxy，仅允许修改官方查询记录的 `bbox` 字段。

阅读覆盖：`code/scripts/` 实际有 **72 个文件**（不含目录），均已阅读全文，包括 8B 历史训练器 `run_qwen3vl_native_lora.sh` 的 837 行、G/U 历史队列和 10 月新脚本。分配的 14 个 `tools` 文件均已阅读全文：`predict_submission_vllm.py`、`native_model_loading.py`、`native_lora_training.py`、`predict_native_submission.py`、`eval_qwen_native_grounding.py`、`eval_qwen_rgb_grounding.py`、`convert_manifest_to_swift.py`、`probe_native_resolution.py`、`enrich_city_queries.py`、`merge_by_predicted_size.py`、`audit_test_overlap.py`、`audit_manifest_overlap.py`、`filter_sequence_overlap.py`、`verify_vllm_token_parity.py`。另全文阅读 `check_submission_format.py` 和 `verify_acc05_dev.py`；`build_capability_slices.py` 阅读了主要构建、转换和混合逻辑，但未逐行审完。其余 `tools/` 文件仅按引用索引，不在本审计的逐文件覆盖声明中。

当前最高的**已记录官方成绩**是 Arm 2 的 checkpoint-500、三模态、2.0 倍图像放大：0.8012；Arm 4 checkpoint-600 同倍率记录为 0.7965。两者是不同适配器，不可把 Arm 4/5 的数据修复或视觉塔调整归到 0.8012。官方成绩来自仓库 `HANDOFF.md` 的记录，本地没有官方真值和平台回执，不能独立复算官方分数。最终交付的 `deliverables/作品材料/赛题结果文档/submission.zip` 内 `predictions.json` 与 `results/submission_k20_20261004/submission.zip` **逐字节相同**，可确定交付包是已记录为 Arm 2 ck500 k2.0 的预测包，不是 Arm 4/5。交付适配器本身没有 `trainer_state.json`，safetensors 元数据也只有 `format=pt`；其文件头与 `args.json` 能证实语言 LoRA 结构、Arm 2 运行目录和训练参数，**不能仅凭本地权重严格证明其来自第 500 步**。第 500 步身份依赖推理 `run_config.json`、文档和打包血缘记录。

## 运行血缘

| 环节 | 实际证据与限制 |
|---|---|
| 基座和适配器 | `results/submission_k20_20261004/run_config.json` 指向 Qwen3.6-27B 与 `/root/autodl-tmp/runs/q36_arm2_capability/archived/checkpoint-500`。本地 `deliverables/作品材料/模型文件/adapter_model.safetensors` 为 933,974,032 字节；其头部含 992 个张量，全部名称属于 `language_model`，无视觉/merger 张量。`adapter_config.json` 是 rank 32、alpha 64、dropout 0.05、语言层正则。故**权重在本地**，并非只留在云端；基座本身未随作品材料打包。|
| 实际训练参数 | 同目录 `args.json` 的 `output_dir` 为 `q36_arm2_capability/v0-20261003-181254`，数据路径 `/root/sprint_qwen36/slices/mixture.jsonl`，`max_steps=600`、批量 1、梯度累积 8、学习率 5e-6、`max_pixels=2073600`、`freeze_vit=true`、`freeze_aligner=true`。最终选择第 500 步。训练源清单和图像不在作品材料中；本地参数文件只能证明本次运行引用路径及参数，不能单独复训。|
| 推理 | `tools/predict_submission_vllm.py` 用 `--upscale 2.0` 与 `max_pixels=8294400` 联动；配置为上下文 32768、并发 32、分块 256。图像以 LANCZOS 放大；64 token 首轮解析失败才重试至 128 token。`run_config.json` 保留相同配置。|
| 产物 | `summary.json` 记录 5690/5690 完成、0 解析失败、0 兜底、0 二轮重试。其 `total_generation_seconds=0.0` 是从已保存结果重新打包的统计，**不代表完整推理耗时**。|

`scripts/train_qwen36_27b_swift.sh` 顶部称与 8B 基线 A “唯一差别是基座和像素预算”，这不够准确：A 原为 144 模块约 30.67M 可训练参数，本次 27B 适配器是语言模型 992 个 LoRA 张量约 233M；冻结视觉塔相同，不等于可训练范围和容量相同。与 A 的成绩差不能单独归因于基座。

## 提交文件的独立检查

使用本地官方 `data/复赛数据集-基于大模型的多模态视觉理解与推理/queries/queries.json` 和 `tools/check_submission_format.py` 检查 Arm 2、Arm 4、Arm 5 三个 k2.0 结果：各 5690 条、非 bbox 字段改动 0、非法框 0、ZIP 仅有根目录 `predictions.json`，均输出 `SUBMISSION_FORMAT_OK`。另独立读取 ZIP 并与外部 JSON 比较，三者均逐字节相同。`check_submission_format.py` 自身只检验 ZIP 成员名，不比较成员内容；这次另行补查后没有实际不一致。

`results/submission_k20_20261004/run_config.json`、`summary.json`、`predictions.json` 和 `submission.zip` 均在本地，故当前包无需云端才可做格式复核。没有官方标注，因此无法检验 0.8012 对应的逐题命中或将任何本地 412 题读数当作官方结论。

## 评估与解释边界

`tools/verify_acc05_dev.py` 直接由预测框和独立真值重算 IoU/ACC，识别 `prediction` 与 `bbox` 两种字段，按图像组做配对 bootstrap（重采样）。这比依赖日志里已有命中字段可信。其 `binomial_floor` 是**单臂**标准误；不能把两个单臂标准误平方相加当作同题配对差值的误差。代码注释把单臂标准误称为“同分母配对差值粗略分辨率”，`deliverables/.../代码说明.md` 进一步以约 12.4 题判两配置差别，均应视为启发式而非成对实验的检验；正式比较应使用逐题翻转表、图组配对 bootstrap，并报告区间。City412 已被多轮选模、选检查点、选倍率反复使用，且与官方 5690 题分布不同，不是独立留出集。

历史终盘段落把 Arm 5 k2.0 的官方成绩写作按其他模型放大收益推算的约 0.796；**本地不存在该配置的官方实测记录**。`HANDOFF.md` 中“官方实际约 +0.014”也属推算，不应写为官方实测。Arm 4 k2.0 对 Arm 2 k2.0 的方向（0.7965 < 0.8012）由两条已记分数支持，但两组训练差异较多，不能归因于深度、数据或视觉塔某一个因素。

## 数据问题及与成绩的关系

`tools/build_capability_slices.py` 的训练配比针对 City412 中小目标、歧义实例等失败类型，使用 City、RGBT、RoboRefIt、RGBDT500 等来源，预混合以避免训练数据集耗尽策略改变呈现次数。两视图和三视图提示词分别与 `<image>` 数量一致。RGBT 的原标注为绝对 xywh，经转换为归一化 xyxy；RoboRefIt 原始 uint16 深度需改用可视化版本；官方同格式的 RGBDT500_prepared 深度也需按推理路径渲染。后续 Arm 3/4/5 修复了深度及分布问题，并对相同测试帧做剔除。

`HANDOFF.md` 的历史终盘段落记录 Arm 2 含 508 条训练行，其引用图像与测试图重叠；后续审计描述为114个City帧，另列出219张官方图、135图组、767题可能受影响的统计。不同单位不能直接互换，本次没有原始审计映射来独立确认全部对应关系。Arm 4/5 按City帧排除了重叠。已有记录不提供各受影响题的逐题真值与因果贡献，不能从两臂的 0.0047 分差推断重叠带来的具体收益。队内记录有向官方咨询同源外部数据的口头结论，本地没有书面裁定；应在技术材料中如实说明，不能把最高分当成严格无重叠的泛化结果。`docs/复赛作品文档_技术素材.md` 第 43、79–96 行将“6928 条训练数据、深度修复、去交集”等后续改动与最终 0.8012 并列，容易造成它们共同产生 0.8012 的错觉；最终包的 Arm 2 适配器并未包含这些后续修复。

## 可执行性与代码缺口

完整仓库中 `tools/predict_submission_vllm.py`、`predict_native_submission.py`、`evaluate_pretrained_grounder.py`、`native_model_loading.py` 构成推理链路；项目 `.venv` 的 CPU 依赖仅为 PyTorch 2.8/Transformers 4.57，**无法代替文档列明的 Qwen3.6 GPU 环境**做端到端推理。运行五组针对性测试：42 通过、2 跳过（`test_predict_submission_vllm.py`、`test_predict_native_submission.py`、`test_convert_manifest_to_swift.py`、`test_merge_by_predicted_size.py`、`test_enrich_city_queries.py`）。这些只覆盖解析、格式、配置和数据转换，不证明适配器在目标 vLLM/Transformers 版本上加载成功。

作品材料 `代码与数据` 内 8 个 `tools`、2 个 `scripts` 与当前仓库同名文件逐字节一致，但**不是独立完整代码包**：没有 `src/mm_grounding`。`tools/evaluate_pretrained_grounder.py:19–21` 无条件导入 `mm_grounding.boxes/engine/metrics`，而 `predict_submission_vllm.py` 在参数解析之前导入前者。从作品材料目录清空 `PYTHONPATH` 运行 `python tools/predict_submission_vllm.py --help` 即 `ModuleNotFoundError: No module named 'mm_grounding'`。由于 `mm_grounding/__init__.py` 还会递归导入 config/model，仅复制三个直接引用文件也不完整。完整仓库含该包；审计结论只针对**作品材料单独运行**。作品材料还未附 `/root/sprint_qwen36/slices/mixture.jsonl`、训练图像与基座，`代码与数据/代码说明.md` 的训练命令不能直接从该目录重现；需要明确外部依赖。

其他非当前最优路径的问题：`tools/predict_native_submission.py` 的零样本 `--package-only` 读取保存的 `adapter: null` 后执行 `Path(None)`，会失败；`scripts/gate_arm5.sh` 的无适配器重打包分支还会与原配置冲突。`scripts/verify_official_test.py` 输出 `OFFICIAL_TEST_BROKEN` 时仍以 0 退出，不能作为可靠 shell 阻断闸门。`scripts/push_checkpoints_to_inference.sh` 只靠目标适配器文件存在跳过复制，不能识别中断的半成品。`tools/verify_vllm_token_parity.py` 若双方公共 ID 为空也会输出 `PARITY_OK`。这些不影响已完成的 Arm 2 包格式检查，后续若重跑该路径应先修复对应入口。

## 建议的接续顺序

1. 先校正技术方案中最终成绩的血缘：Arm 2 ck500 + k2.0；将深度修复、去交集、视觉塔方案归属 Arm 3/4/5，并明确 Arm 5 k2.0 官方未测。
2. 若作品材料要求独立运行，补入完整 `src/mm_grounding` 及运行依赖说明，再以隔离目录做 `--help` 和提交格式预检；本次接手审阅未改包。
3. 若还要开展新实验，以当前 Arm 2 k2.0 为冻结基线，按同一机器/同一 412 题逐题配对比较，并另找未被反复使用的测试或获得官方反馈判断迁移。对后续每个改动记录单一假设、输入清单和独立产物，避免把多个同时改动归为单因。
