# 核心融合源码接手审计（2026-09-27）

本次对 source-inventory.json 中 owner=core 的 **47 个文件、6550 行**逐文件完整阅读；另完整阅读审计包内两个历史几何脚本，共 260 行。逐文件的已读范围、作用与状态见 [core-coverage.json](core-coverage.json)。这是静态源码审计与现有 CPU 测试，不代表已加载模型权重、核对云端环境或重跑 GPU 成绩。比赛口径以 contest/基于大模型的多模态视觉理解与推理.pdf 第 2–4、6 页为准：目标是 RGB 图上的归一化 xyxy 框，ACC@0.5 按每条 Query 的 IoU≥0.5 计；提交时只能补 bbox 字段。

## 两条路线的界线

本清单主体是**历史自定义融合路线**。train.py:46–64 加载本包 YAML、Qwen processor 与 MultiModalGrounder；src/mm_grounding/model.py:335–409 在 Qwen3-VL 上选择原始补丁融合、Patch Embed 后融合、逐层 RDT 提示或 RGB/IR/Depth 并行视觉主干。各模态在 src/mm_grounding/adapters.py:868–975 的可训练适配器与阶段融合模块汇合，Qwen 主干在阶段 A 冻结（model.py:434–473）。它不是当前 C1500/M2/G/U/B/G*/U* 使用的原生 LoRA 实现。

当前原生路线的入口是 code/scripts/run_qwen3vl_native_lora.sh:4–16,55–67,93–136,673–675：根据 rgb/trimodal 选择另行准备的 SFT 标注，调用外部 Qwen finetune 的 train_qwen_module.train，启用 PEFT LoRA；这条入口不读取本清单的 22 份融合 YAML，也不调用 src/mm_grounding 的 train/evaluate。脚本自身还增加了样本消费追踪、断点与预检钩子（同脚本:473–543,567–669）；其完整执行审计属于另一组范围。不能用本包源码推断现行 LoRA 的内部注意力、损失归因或最新成绩。code/pyproject.toml:16 和 requirements-experiment.txt:2 的 Transformers 4.x 约束是旧包依赖元数据；审计包 README.md:64–69 记录的本轮云端 Transformers 5.14.1 也不能由本地旧配置重建。

## 历史链路：数据 → 提示 → 模型 → 损失 → 指标 → 检查点

| 环节 | 源码确认的行为与口径 |
| --- | --- |
| 数据 | src/mm_grounding/data.py:43–52 接受按 ID 索引的 JSON 或 JSONL；:81–101 检查 bbox 为 [0,1] 的非空 xyxy，保留原浮点值用于评分。IR 经灰度读取、双线性缩放到 RGB 尺寸并转三通道（:103–109）。Depth 原值按配置 scale/clip 转对数可视化，第二通道是有效掩码，再最近邻缩放（:17–29,110–118）。|
| 提示和监督 | data.py:132–137 要求仅输出 bbox_2d JSON 的 0–1000 整数坐标；:166–195 仅把 RGB 图放入对话提示，将原归一化框四舍五入乘 1000 后放在 assistant 答案，提示 token 不计损失。IR/Depth 经 image_processor 单独成为张量，并核对图像网格（:212–224）；它们不是第二、第三张对话图。|
| 模型 | model.py:136–229 的 parallel_backbone 路线让三图通过共享的冻结视觉块，并在指定层把辅助残差加入 RGB；DeepStack 输出取融合后的 RGB。查询支路取 Qwen 输入词向量并 detach，再过本包 QueryTokenEncoder（model.py:540–573；adapters.py:364–427）。其他三种融合结构仍可构建，但本清单 22 份 YAML 实际只选 parallel_backbone 或 rdt_deep。|
| 损失 | model.py:691–700 把 Qwen 原生 output.loss 直接当 token_loss。只有启用辅助框头时才加 Smooth L1 与 GIoU（:703–730）；coordinate_token_loss 在 :713–725 单独计算和报告，却没有额外加入 total，坐标 token 已由原生 CE 监督。config.py:174–180 限制辅助框头只用于 safe_post_embed。本清单 22 份配置逐个核对后，auxiliary_bbox_enabled 均为 false（含省略后由 config.py:36 默认关闭），vision_lora_enabled 也均为 false（config.py:39）。所以这些配置没有优化辅助框 GIoU 损失，也没有训练本包的视觉 LoRA；不能因仓库有几何损失或 LoRA 代码就描述为实际使用。|
| 训练与选择 | engine.py:290–324 仅把 requires_grad 参数放入 AdamW；:417–718 做累积、评估、探针和保存。模型主选择按 ACC@0.5、mIoU、ACC@0.7、解析率字典序（:19–43,668–706）；训练中的 subset 和结束时 full 分开记录（:656–714）。train.py:16–38,78–104 的 subset 是按 class_name/scale_bin 分层的确定性抽样。|
| 评分 | engine.py:174–184 从生成文本解析 bbox_2d 并除以 1000；:187–287 用原浮点 GT 计 IoU，未解析回答记零框并单独记 parse_rate。metrics.py:8–16 的 ACC@0.5 判据是 IoU≥0.5，与官方阈值一致。evaluate.py:171–225 对历史并行结构可分别跑三图、RGB+IR、RGB+Depth 和 RGB 基线；非并行 joint 只跑三图和 RGB。这里的 rgb_baseline 是同一加载模型关闭融合后的结果，不能无条件称为全新预训练模型。|
| 检查点 | checkpoint.py:25–88 保存需训练参数；joint 阶段也保存 fusion 下虽已冻结的已初始化参数（:40–45）。紧凑 best 不含优化器、调度器和 global_step，完整 last 可恢复（:71–81,199–216）；多来源初始化逐键检查冲突（:163–196）。运行元数据记录版本、配置、评分和选择顺序（:49–69）。|

## 确认的问题与证据强度

1. **旧 legacy_patch 的所谓正交去重公式不是真正向量投影。** adapters.py:48–52 用逐元素乘积再乘对方向量；正交投影应先对通道求点积，再除对方向量的平方范数。本包后来的 SafePostEmbedFusion 已写出这种点积公式（adapters.py:117–128）。这是可由代数直接确认的历史实现问题，但清单内配置没有选择 legacy_patch，也没有证据把它归因为当前分数变化。

2. **默认查询融合支路不保留词序。** config.py:34 默认 query_position_encoding=none；adapters.py:380–427 对词向量仅做逐 token 投影和没有位置编码的 Transformer Encoder。tests/test_adapters.py:14–34 验证该支路对 Query 重排等变，sinusoidal 版本能破坏这种等变。本清单除 positional 对照配置外均沿用 none（configs/stage2_joint_fusion_v3_control.yaml:14 与 positional.yaml:14）；这可能限制历史融合支路对序数或关系语序的表达，但 Qwen 正常语言路径仍读取有序提示，**不能推断整个模型对语序无感**，也未做受控成绩归因。

3. **外部深度单位假设未获本次代码证据支持。** data.py:17–29 把原像素除以 depth_scale，所有列出的深度配置用 1000.0 与 20.0，例如 configs/qwen3_vl_8b_stage1b_depth.yaml:25–32。官方 PDF 第 3 页明确的是赛事 City 深度毫米、小值近；审计包 DATASETS.md:30–38,48–54 明确 RoboRefIt/RGBDT500 的物理单位未充分确认。因此可以说旧 Stage1B 采用了米制解释的参数，不能说它教会了可靠的米制近远。实际旧 RoboRefIt 图像值如何映射，需要原图和该阶段配置/权重实验复核。该历史配置的验证路径还是 RoboRefIt testA（同 YAML:27），不要与当前另行抽取、未用 testA/testB 的 2000 训练池混为一谈。

4. **稀疏检查点加载可能接受缺失的待训练参数。** checkpoint.py:154–160 在 trainable_only 为真时忽略全部 missing keys，只拒绝 unexpected；初始化合并也有意容许尚未训练模块缺失（:163–196）。这是多阶段装配所需，但若把旧检查点和不匹配的融合配置组合起来，部分参数可保持随机初值而加载不报错。当前没有证据表明已报告结果发生此错误；应以实际 checkpoint 的 saved_parameter_names、目标配置和已加载参数核对，不根据成绩推断。

5. **训练内与独立评估的数值口径存在代码差异，原因未验证。** train.py:48–49 在 CUDA 时设 FP32 matmul precision=high，evaluate.py:143–151 没有同一设置；两者调用同一 engine.evaluate（train 路线见 engine.py:660、独立路线见 evaluate.py:189–205）。这可解释为什么需要单独复核旧 8B 分数差异，但尚无 GPU 同检查点受控重评，不能断言差异就是该设置造成。还须核对具体 checkpoint、输入清单及运行环境。

6. **审计包中的两份几何脚本是历史证据，不具备包内即跑的路径条件。** root_cause/analyze_full412_geometry.py:7–17 写死原机器 F:/AIC/results 路径；root_cause/geometry_audit.py:11–15 以当前复制位置上溯三层得到 docs/audit-20260927，再拼 results/gu_diagnosis_20260927，和包内 evidence/gu_diagnosis_20260927 不同。两脚本的公式和身份联结已完整阅读，未在接手中运行；它们保存的数值结论应以包内结果文件、来源记录和现行报告交叉核对，不能宣称克隆审计分支即可重算。

7. **early_stopping_min_delta 配置目前只被校验，没有参与早停判据。** config.py:70 声明该字段，:197 检查非负；engine.py:668–680 直接按字典序比较候选和最佳指标，:708–716 仅按 stale_evals 与 early_stopping_patience 停止，没有读取 min_delta。因而例如 configs/stage2_joint_fusion_v2_warm_projection.yaml:41 或 configs/stage2_joint_fusion_v3_control.yaml:38 把它设成 0，不会改变这段训练选择/早停行为。tests/test_config.py:126–132 只断言配置值，没有行为测试。此结论是源码级确定的历史配置空效，不推断它改变了某次既有分数。

## 验证与仍待核实

在 F:/AIC/code 使用既有环境执行：F:/AIC/code/.venv/Scripts/python.exe -m pytest -q tests/test_adapters.py tests/test_auxiliary_training.py tests/test_compact_checkpoint.py tests/test_config.py tests/test_data.py tests/test_lora.py tests/test_model.py tests/test_resume_learning_rate.py tests/test_selection.py tests/test_sparse_checkpoint.py tests/test_training_output_lifetime.py。结果 **66 passed in 8.20s，退出码 0**。系统默认 Python 无 pytest；未因此安装依赖或改变生产源码。另用 YAML 解析遍历 22 份配置，得到融合类型仅 parallel_backbone/rdt_deep、辅助框头启用 0、视觉 LoRA 启用 0。

> .................................................................. [100%]
> 66 passed in 8.20s

测试主要使用小型假主干和合成框，证明接口、梯度连接、保存/恢复及局部数值行为，不能替代真实 Qwen 权重、云端版本、原图配准、外部深度语义或 412/官方样本复验。本次没有启动 GPU、训练、推理、下载、提交或修改 GT；没有根据分数臆测模型内部机制。当前原生 LoRA 训练与评分的完整实现和真实消费记录由并行审计范围单独核查，本报告仅界定两条代码路线及历史核心模块的可证事实。
