# 执行链只读审计（2026-09-27）

范围：`source-inventory.json` 中 `owner=execution` 的 94 个文件，共 13,543 行；逐文件完整阅读范围见 `execution-coverage.json`。另核查了相关训练配置、完整消费轨迹、训练清单、City 原始 GT、七份完整预测和云端快照。此次没有修改生产代码，没有启动 GPU 推理、训练、下载、自动化或 Git 操作。结论只覆盖已有代码和保留下来的证据，不能把静态阅读当成重新执行云端实验。

## 可直接用于路线判断的结论

1. 当前的 B、G★、U★ 和历史 G/U 是 **原生 Qwen3-VL-8B + PEFT LoRA** 分支，均从 M2 `checkpoint-928` 适配器初始化，用 200 次优化器更新、学习率 `5e-6` 的独立新阶段。训练使用同一原生启动器和推理/评分器。旧自定义融合（`mm_grounding` 中的三模态适配器、专家与阶段式训练）是历史分支，不能把它的模块或成绩归因于这次原生 LoRA 实验。[训练入口](../../../code/scripts/run_qwen3vl_native_lora.sh:1)、[新诊断编排](../../../code/scripts/run_gu_diagnosis.sh:102)、[历史 G/U 编排](../../../code/scripts/run_gu_pilot.sh:102)。
2. 独立 CPU 复算原始 City 412 条 GT 后，C 为 **292/412**，M2 为 **285/412**；新 B/G★/U★ 为 **287/284/286**。U★ 比 G★ 多 2 命中，但比 C 少 6，且 G★→U★ 有 5 条由错变对、3 条由对变错。数据不支持把 U★ 视为已经超过现有 C 的新主模型。明细及逐 ID 翻转见 `execution-recheck-results.json`。
3. B 与 G★ 的 400 个新位置同时换了图像、Query 和目标框；G★ 与 U★ 的 183 个辅助任务位置保持图像不变，但同时换了提示词和目标框。因而 B→G★ 不能单独识别“外部数据”或“辅助模态”的效应；G★→U★ 也不能单独识别目标模态监督的效应。此处是实验解释边界，不是训练程序报错。
4. 保存的完整预测均为 412/412 个唯一、同序 ID，七臂解析成功 412/412；原始 GT 与每行保存目标一致，保存的 `hit` 与重新计算的 IoU≥0.5 一致。评分工具默认拒绝缺失 ID，显式部分运行只可作观察进度。[评测完整性](../../../code/tools/report_rematch_experiment.py:147)、[重新评分](../../../code/tools/report_rematch_experiment.py:212)。

## 原生训练、恢复、推理及评分路径

`run_gu_diagnosis.sh` 固定 `M2=.../first_batch/m2/checkpoint-928`，设置 `PRESERVE_MANIFEST_ORDER=1`、`SEED=2026`，以 `train_segment` 为 B/G★/U★ 依次运行 100 与 200 步；第二段显式传 `RESUME_FROM_CHECKPOINT`，**总 horizon 仍是 200 步**，不是重新开始一个 200 步阶段。[初始化](../../../code/scripts/run_gu_diagnosis.sh:12)、[顺序](../../../code/scripts/run_gu_diagnosis.sh:37)、[训练参数传入](../../../code/scripts/run_gu_diagnosis.sh:102)、[100→200 与全 412 评估](../../../code/scripts/run_gu_diagnosis.sh:193)。历史 G/U 用同一训练入口、同样总 200 步，但其训练数据清单与本轮 B/G★/U★ 不是同一实验。[历史序列](../../../code/scripts/run_gu_pilot.sh:150)。

原生启动器的实参为 AdamW fused（β₁=0.9、β₂=0.999、ε=1e-8、权重衰减 0），线性学习率调度、零 warmup、梯度裁剪 1、batch 1、梯度累积 8、bf16、TF32 关闭、SDPA、`max_pixels=602112`、`min_pixels=200704`、最大长度 4096。[训练实参](../../../code/scripts/run_qwen3vl_native_lora.sh:98)、[记录配置与续跑比对](../../../code/scripts/run_qwen3vl_native_lora.sh:225)。新阶段从 M2 的 PEFT 适配器加载 **可训练权重**，但重新创建优化器与调度器；第 100 步的恢复要求 `trainer_state.json`、`optimizer.pt`、`scheduler.pt` 和 RNG 状态，并把显式检查点传给 `Trainer.train`。[适配器初始化](../../../code/scripts/run_qwen3vl_native_lora.sh:400)、[恢复要求](../../../code/scripts/run_qwen3vl_native_lora.sh:516)。`SequentialSampler` + 0 个数据加载进程固定清单顺序，消费轨迹在每次 `training_step` 后追加；恢复时会截去检查点之后尚未提交的轨迹。[顺序和轨迹](../../../code/scripts/run_qwen3vl_native_lora.sh:490)。

快照中 B/G★/U★ 各有 1,600 条 `consumed_samples.jsonl`，与各自 1,600 条训练清单 **ID 顺序逐行相同**；历史 G/U 的 1,600 条也各自与对应清单一致。三条新训练的 `trainer_state.json` 均为 `global_step=max_steps=200`；LR 日志在步 100/110/200 分别为约 `2.525e-6`、`2.275e-6`、`2.5e-8`，符合一次 200 步线性衰减，且没有在第 101 步回升至 `5e-6`。`native_train_config.json` 保留的是第 100 步阶段创建的 `stop_after_step=100`，而最终 `trainer_state.json` 是 200 步；这个表面差异来自同一输出目录续跑，不能仅看前者误判只训练了 100 步。**证据限制**：本地快照未包含 `optimizer.pt`、`scheduler.pt`、RNG 二进制或完整适配器张量；恢复代码要求这些文件，保留下来的状态/日志与轨迹支持实际延续，但本次无法直接比较云端优化器张量或重放逐步参数更新。

City 评估由编排器使用原生 prompt、原始 `qwen_generation_val.json` 作为 `--target-manifest`、最大生成 128 token、`--resume`；评估器从输入 manifest 取用户提示、从目标 manifest 覆盖评分框。[调用](../../../code/scripts/run_gu_diagnosis.sh:129)、[覆盖原始 GT](../../../code/tools/evaluate_pretrained_grounder.py:274)。生成 `messages` 仅由 prompt 和 RGB/IR/Depth 图像构造；生成后才解析 JSON 框并取 GT 算 IoU。[生成输入](../../../code/tools/evaluate_pretrained_grounder.py:680)、[解析与评分](../../../code/tools/evaluate_pretrained_grounder.py:716)。`parse_generated_bbox` 使用输出文本中的键名框，最终 `mm_grounding.engine.parse_bbox` 做框合法性；无有效框时 `prediction=None`、IoU=0，仍在评估分母内。[解析入口](../../../code/tools/evaluate_pretrained_grounder.py:33)、[零分处理](../../../code/tools/evaluate_pretrained_grounder.py:440)。

比赛侧预测器是另一条不带 GT 的推理入口：它要求输入 Query 记录无 `bbox`、默认检查 5,690 条，加载原生基础模型 + PEFT 适配器，按输入顺序生成，无法解析时写全图框作为合法回退，仅在全量完成后打包。[输入边界](../../../code/tools/predict_native_submission.py:57)、[加载与生成](../../../code/tools/predict_native_submission.py:294)、[打包](../../../code/tools/predict_native_submission.py:177)。这里只审计静态代码和本地 CPU 测试；没有对官方无标签集实际运行，不能给出官方分数。`predict_competition_submission.py` 是历史自定义融合推理入口，不能用它描述当前原生分支。

## 独立复算（412 条同一批 City 验证样本）

`F:/Downloads/qwen_generation_train_val_manifests.zip` 内的原始 `qwen_generation_val.json` 与保留的 `source_snapshot/city_gt.json` 逐项相同，均为 412 条。脚本直接从预测 `bbox` 与原始 GT 用 Python 浮点运算 IoU，不读预测文件保存的命中值作为结果；只把保存的命中/IoU用于交叉核验。由于原评估器采用 torch float32，保存 IoU 与 Python float64 的数值比较容差为 `1e-5`，所有阈值命中完全相同。

| 模型 | ACC@0.5 | ACC@0.7 | mIoU | 解析 |
|---|---:|---:|---:|---:|
| M2 | 285/412 = 0.6917 | 214/412 = 0.5194 | 0.5983 | 412/412 |
| C（历史 City1500） | 292/412 = 0.7087 | 224/412 = 0.5437 | 0.6136 | 412/412 |
| 旧 G200 | 288/412 = 0.6990 | 203/412 = 0.4927 | 0.5944 | 412/412 |
| 旧 U200 | 283/412 = 0.6869 | 202/412 = 0.4903 | 0.5962 | 412/412 |
| B200 | 287/412 = 0.6966 | 208/412 = 0.5049 | 0.5978 | 412/412 |
| G★200 | 284/412 = 0.6893 | 203/412 = 0.4927 | 0.5914 | 412/412 |
| U★200 | 286/412 = 0.6942 | 209/412 = 0.5073 | 0.5995 | 412/412 |

成对翻转（左→右，错变对 / 对变错）：B→G★ 为 2/5；B→U★ 为 4/5；G★→U★ 为 5/3；C→U★ 为 3/9；旧 G→G★ 为 1/5；旧 U→U★ 为 6/3。具体样本 ID 在 `execution-recheck-results.json`，例如 G★→U★ 的错变对含 `city_000081_004`、`city_shuming_1062_00000100_002`，对变错含 `city_001685_008`。没有把同一 412 条重复试验中的 2 个净增解释为显著提升。

## 清单配对到底控制了什么

新三臂训练清单都含 1,200 个相同 City 旧位置 + 400 个新位置，三份 `gu200_...` 呈现 ID 和消费顺序相同；每步 8 个样本，200 步恰好完整消费 1,600 条。B→G★ 有 **400 行图像、400 行 prompt、400 行答案同时不同**，只有前 1,200 行完全相同。G★→U★ 有 1,417 行完全相同、183 行图像不变但 prompt 和答案同时不同（其余 217 个新位置也相同）。旧 G→旧 U 同样有 183 行 prompt 和答案差异、图像无差异。源记录并非每臂同名同题的完全一因素随机实验。清单统计来源：`cloud_snapshot/manifests/summary.json`，逐行复核见 `verify_execution.py`。

具体可人工查验：在 `cloud_snapshot/manifests/train_{b,gstar,ustar}.json` 的第二行，`gu200_000002`：B 为 City 的白色 A 形牌，答案 `[474,503,568,725]`；G★ 为 RGBDT 动物，RGB 答案 `[534,447,902,918]`；U★ 保持 RGBDT 图像，但请求红外坐标，答案 `[526,443,911,928]`。在第 1,201 行 `gu200_001201`，B 为 City 灯杆 `[228,229,233,244]`，G★/U★ 为 RGBDT 猴子，分别给 RGB `[503,508,534,578]` 与 IR `[497,499,542,579]`。这些 ID 是训练呈现序号，**不是 City 验证集 ID**；不能据其同名误判样本相同或泄漏。

## GT、恢复与分母的边界

- 已有证据表明本轮 412 评估把 GT 留在评分侧，没有在 `messages` 中直接送给模型；本地七份预测的每个 `target` 与原始 GT 相同。该结论不替代全仓数据划分/原图近重复审计，也不证明模型绝无训练集相似图像。训练监督数据自然含其自身目标框。
- 评估恢复校验 ID 唯一、属于 manifest、保存 target 等于当前 target；配置值也要一致。[恢复记录校验](../../../code/tools/evaluate_pretrained_grounder.py:334)、[配置校验](../../../code/tools/evaluate_pretrained_grounder.py:382)。`summarize_rows` 会从预测框与 GT 重算 `grounding_metrics`，但单独的 `hits` 字段优先读取已有行保存的 `hit`/`acc_0.5`/`iou`，没有重新计算；若**人为篡改**保存行，摘要的 `hits` 与 `acc_0.5` 可不一致。[摘要](../../../code/tools/evaluate_pretrained_grounder.py:451)。本轮七份文件逐行复算后没有这种不一致；正式比较应继续以 `report_rematch_experiment.py` 的原始 GT 复算为准。
- `report_rematch_experiment.py` 默认 `allow_partial=False`，缺任一 ID 即拒绝；如果显式允许 partial，`score_run` 分母变为已保存的行数，不能把它当完整 412 成绩。[完整性门槛](../../../code/tools/report_rematch_experiment.py:147)、[部分分母](../../../code/tools/report_rematch_experiment.py:191)。模态干预报告把 partial 只列作 `observed_metrics`，排除配对主结果。[干预报告](../../../code/tools/report_modality_interventions.py:40)。
- 报告中的 `candidate_union` 是拿 GT 离线量候选召回上限的分析，不是当前提交推理器的选择策略或真实模型分数。[候选覆盖](../../../code/tools/report_rematch_experiment.py:380)。

## 运行验证与尚未覆盖的边界

复算命令：

```powershell
& 'F:/AIC/code/.venv/Scripts/python.exe' 'F:/AIC/docs/research/2026-09-27-takeover-audit/verify_execution.py'
```

输出：`execution-recheck-results.json`、`execution-recheck-console.log`。复算脚本只依赖 Python 标准库；若原始 ZIP 不在 `F:/Downloads`，仍能使用审计包内 GT 快照，JSON 的 `original_archive_matches_snapshot` 会是 `null`。此次原始 ZIP 在本机，结果为 `true`。

既有 CPU 测试原样执行（命令输出原文保存为 `execution-pytest.log`）：

```powershell
& 'F:/AIC/code/.venv/Scripts/python.exe' -m pytest -q tests/test_native_lora_runner.py tests/test_predict_native_submission.py tests/test_evaluate_pretrained_grounder.py tests/test_report_rematch_experiment.py tests/test_gu_pilot_gate.py tests/test_slurm_scripts.py
```

结果：**42 passed、1 failed、1 warning**，退出码 1。失败为 `tests/test_slurm_scripts.py::test_slurm_environment_allows_source_archive_without_git`：测试在 Windows 上用 `subprocess.run(..., text=True, encoding="utf-8")` 读取 Git Bash 输出，后台 reader 解码时报 `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xce in position 15`，随后 `result.stdout=None`，`tests/test_slurm_scripts.py:123` 的包含判断抛 `TypeError`。这是当前 Windows 测试环境的编码/测试夹具边界；本次没有改实现或测试，也**不把此测试记为通过**。系统 Python 上 `python -m pytest` 因未安装 pytest 无法运行，随后使用项目 `.venv` 完成上述验证。

未运行 GPU 实验、未加载模型权重，不能验证真实数值梯度、云端完整优化器状态、原生模型视觉性能或官方无标签提交表现。历史融合/专家工具与审核界面已纳入逐文件阅读，但没有运行其训练、旧队列、Web 审核服务或其全部测试；本报告的实证结论集中于原生 LoRA 七臂与现存预测证据。
