# G/U 诊断训练与评估契约独立复审（2026-09-27）

## 范围与结论

本复审只读取本地 `results/gu_diagnosis_20260927/cloud_snapshot/`、上一轮 `results/gu_pilot_20260926/final_cloud_snapshot/`、构造/启动/评分代码及已有审计；没有连接云端、启动 GPU、修改生产代码或重跑模型。逐行复算脚本和结果为 [`root_cause/gu_execution_reaudit.py`](root_cause/gu_execution_reaudit.py) 与 [`root_cause/gu_execution_reaudit.json`](root_cause/gu_execution_reaudit.json)。

**未发现能够把本轮 B/G*/U* 差距解释为训练漏样、100步重启学习率、图像列对调或三臂评估条件不一致的执行错误。** 发现一个确定的残余对照混杂：B 的98个新 City Query 对应的98个图像组，**98/98已在共同的旧1200 City训练位置出现**。B 比较的是已见图像上的新 Query；G*/U* 新400位置则引入 RGBDT 图像。B 按批准的定义正确构造，这不是代码 bug，但 B 与 G*/U* 的成绩差不能单独归因为“新增数据源/监督质量”。

## 清单与实际消费：已排除

`code/tools/prepare_gu_diagnosis.py:445-460` 将原排程中的 City 行直接复制，并按旧400位置替换新行。最终云端清单对比确认：B/G*/U*各1600行，三个清单的旧1200 City行均逐字等于旧 G 和旧 U 清单相同位置；旧 G/U 这1200行自身也完全相同。新400位置的 G*/U* 图像路径、ID和顺序400/400相同；217条 RGB 行全行相同。剩余183条（IR 128、Depth 55）提示正文至输出指令前完全相同，完整提示只把输出图坐标改成对应模态，答案框随之改变；深度编码说明两臂均保留。构造源见 `code/tools/prepare_gu_diagnosis.py:79-90,247-256,391-443`。旧 G/U 中183条辅助提示省略深度说明的已知混杂，在这轮 G*/U* 配对中已消除。

B 的400个位置映射98个 City Query，重复次数沿原98个 RGBDT Query 的频次，总和400。构造器排除了旧 City **任务ID和同文本 Query**（`code/tools/prepare_gu_diagnosis.py:320-365`），但没有排除旧图像组。最终 `manifests/b_query_mapping.jsonl` 对旧1200元数据 `group` 逐行连接显示：98组均已见，69组先前出现2次、29组出现1次。例：`city_000005_032_00000205_001` 所在图像组在旧1200出现2次，然后被映射到新 Query `rgbdt500_005_00000198::q01` 对应的8个新增位置。B确有98个未在旧1200暴露的 Query，但没有98个全新 City 图像组。

三臂 `consumed_samples.jsonl` 各1600条，按行恰等于相应训练清单 ID。启动器以 `SequentialSampler` 和零工作线程取样，在成功 `training_step` 后才写ID（`code/scripts/run_qwen3vl_native_lora.sh:490-510`）；恢复时按checkpoint的800条前缀核对并截去可能的未提交尾部（同文件 `:627-669`）。因此云端落盘证据支持“各200更新×8微批、无已观察到的二次打乱/漏样”，并不把单纯 `__getitem__` 调用次数误当实际训练消费。

## 初始化、学习率与可训练范围：已排除

`code/scripts/run_gu_diagnosis.sh:102-127,193-201` 对三臂第一段均从 M2 `checkpoint-928` 初始化 LoRA（低秩适配），第二段从各自 `checkpoint-100` 完整恢复；两段均传总 `MAX_STEPS=200`。三臂 `native_train_config.json` 除清单和输出目录外的核心设置一致：seed2026、1 epoch、batch1×梯度累积8、最大长度4096、像素200704–602112、AdamW fused、线性学习率5e-6、无warmup、BF16/SDPA、TF32关闭。配置文件记录第一段的初始化设置，不可把其中 `stop_after_step=100` 当成第二段重启到100步；第二段的实际恢复路径见 `logs/*_train200_*` 第304行。第一段日志第6行均写载入并逐张量验证M2的288个LoRA张量；第二段有显式checkpoint恢复记录。

三臂的 `checkpoint-100/trainer_state.json` 都为 `global_step=100/max_steps=200/epoch=0.5`；`checkpoint-200` 都为 `200/200/1.0`。日志步号由1、10…100连续至110…200，100步学习率2.525e-6，200步2.5e-8，没有100处重新回到5e-6的迹象。`run_qwen3vl_native_lora.sh:99-135,399-469` 将可训练参数限定为**语言模型注意力 q/k/v/o 的 LoRA**，视觉编码器和多模态投影器冻结；第1步日志记录288/288可训练张量有梯度，损失和梯度有限。此为确切训练能力边界，并非本轮误冻结的证据。云端文本快照未含可重算的权重和优化器二进制，不能由文本独立证明每个权重更新值。

监督采用上游 Qwen 数据处理器：`tmp/source-audits/Qwen3-VL-current-20260922/qwen-vl-finetune/qwenvl/data/data_processor.py:140-199` 按 `<image>` 顺序填图，`:202-240` 将用户和视觉 token 设为 `-100`，只监督 assistant 答案及结束符。新轮8个代表样本的真实预检日志 `logs/preflight_resume2_20260927T021440_attempt1.log:301-317` 显示三图网格、完整 `bbox_2d` 监督文本、1857–1884输入token（小于4096）；正式三臂各只审计首个训练样本，不能把8个代表样本当作1600条逐token全量核对。预检足以排除普遍的答案全遮罩和模板全部截断，但不能排除某个未审计样本的局部问题。

## 训练提示、Depth 与验证：共同条件及边界

**不能说三臂的1600条训练提示全部统一。** 共同旧1200 City行保留 `These are aligned views...` 和“有效深度越亮越近”的 City 原提示；G*/U* 新400行使用 `These are views of the same scene...` 与“原始传感器正值越小越亮，物理距离单位未确认”的统一新提示。云清单核对恰为1200/400。B的新400使用旧 City 模板。G*/U*之间的400行共用新模板，仅183条必要输出模态指令和框变化；B与G*/U*则同时改变图像域、Query、目标框和提示表述。因此 G*/U*配对能检验这套固定训练包装下的辅助坐标监督替换，B对照不能单独隔离某一种数据因素。

City旧1200训练和完整412验证都引用 `target_v2/qwen3vl_native_sft/depth_rgb/` 的已渲染三通道Depth；统一的 City 处理公式位于 `code/tools/prepare_qwen3vl_native_sft.py:37-57`。三臂412评估的 `run_config.json` 都采用同一个 `trimodal_val.json`、原浮点GT `qwen_generation_val.json`、像素界、模型、生成上限和精度策略；落盘预测逐行 `id/image/prompt/target/target_source_kind` 412/412相同。评估代码 `code/tools/evaluate_pretrained_grounder.py:644-748` 按同顺序打开图像、用同像素界处理、从输出解析框；三臂摘要均为412/412解析且无128 token上限命中。由此未见 City Depth 训练与验证发生路径/渲染政策错接或三臂评估混版。新400 RGBDT 的Depth按其自身固定逆映射显示，数值物理含义未确认；它与 City 的毫米深度并非同一个已验证的物理语义，属数据/任务迁移条件，不能被这份执行审计证明为致因。

还有一个共同的细节：上游训练处理器拆分 `<image>` 后会对文字片段 `.strip()`（`data_processor.py:165-182`），评估器 `evaluate_pretrained_grounder.py:428-437` 保留片段首尾空白。故训练和推理的精确文本token可能有空白差异；所有三臂共用这条路径，当前没有证据显示它解释臂间差距，也没有在本次CPU审计中重编码全部样本。

## 结论使用限制与未测机制

新 B/G*/U* 均为200次更新；历史 C 累计1500次更新且训练分布不同。C的412成绩优于本轮各臂，不能据此断言新增监督造成负迁移。即使对共同预算的 B/G*/U*，B 图像全为训练已见组，使其不能成为“仅替换数据源、其他不变”的因果对照。G*/U*是相对严格的一对，但U*把183/1600次最终RGB框监督改为IR/Depth输出；从评估差值不能再拆分为“跨模态能力收益”与“直接RGB监督机会成本”。

本次未测模型内部是否真正利用IR/Depth、LoRA是否足以适配两路视觉表示、不同监督的梯度是否冲突，也未在所有1600条上逐token验证loss mask。上述均是机制假设，不写成已发现的执行根因。完整412成绩与逐题阈值归因应以主报告及 `cloud_snapshot/report_city412/` 为准；本复审只裁定训练/评估契约和残余混杂。
