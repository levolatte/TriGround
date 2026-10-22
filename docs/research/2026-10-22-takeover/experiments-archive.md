# experiments 与 2026-09-27 审计源码接手审阅

日期：2026-10-22。范围为 `code/experiments/evidence_decision`、`code/experiments/visual_agent`、`code/docs/audit-20260927`，以及 `code/docs/audit-20260929/recompute.py`；本报告依据当前仓库源码、已存结果文档及历史审计，不代表新运行的模型实验。赛题要求仍以 `contest/` 官方材料为准。

## 阅读范围与去重

- `evidence_decision`：目录 255 个文件，其中 Python 96 个、约 1.08 MB；`visual_agent`：198 个文件，其中 Python 77 个、约 0.84 MB。对两目录全部 Python 文件做路径、字节和模块级 AST 清点；逐篇阅读 `evidence_decision` 的 59 个非测试 Python，并用共享文件的同一实现与 18 份差异对照覆盖 `visual_agent` 的 49 个非测试 Python。控制器、图像工具、教师状态、训练、评估及运行脚本均读了完整实现。前者测试文件 37 个/163 个 `test_` 方法，后者 28 个/120 个；测试命名和覆盖对象均清点，未逐行精读每个测试体。
- 两目录共有同名 Python 文件 77 个，其中 30 个字节相同；另 47 个字节不同，将包路径差异归一后 29 个内容相同、18 个存在实质差异。`evidence_decision` 独有 19 个 Python 文件，包括 `object_*` 新流程及新增测试。因而不能把两树的 173 个 Python 文件当成 173 份独立实现。
- 历史审计目录 426 个文件，包含 32 个 Python 源码、约 0.51 MB；没有独立 `.js`、`.sh`、`.ps1` 或 `.ipynb`。`annotation_builders/make_natural100_acceptance.py` 含内嵌 JavaScript，已阅读其 CSV 导出逻辑。32 个 Python 文件按字节比较后是 26 种内容：`code_snapshot` 中 4 个与 `code_final` 相同，`code_patches` 中 2 个与 `code_final` 相同。对 25 种非测试实现逐篇阅读全文或读取与最终版的完整差异，包含 12 个标注构建脚本、几何分析、候选/深度、选择/细化、报表、队列和输出审计；其余 1 种为测试，清点其测试函数与覆盖对象。标注脚本的人工描述已读源码，但没有按原图逐条确认描述真实性。
- 另逐行读了 `code/docs/audit-20260929/recompute.py`：它调用 `tools.report_triground_abv.build_report/write_outputs`，重算 C0/A/B/V 的 city412 与缺模态诊断；检查 A/B/V 每臂消费 4800 条、消费顺序等于 manifest、恢复状态为 `pass`。这是审计复算入口，不是模型训练/推理入口；本轮没有执行。与前述 32 个合计，历史 docs 代码为 33 个 Python 文件。
- 结果与决策文档重点阅读：两实验目录的 `RESULTS.md`、`HANDOFF.md`、`AUDIT_ROOT_CAUSES.md`，`evidence_decision/POSTMORTEM_20261001.md`、`TRAINING_COVERAGE_AUDIT_20260930.md`、`SFT_INTEGRATION_REVIEW.md`，`visual_agent/SINGLE_OBSERVATION_PLAN_20261001.md`；历史 `aux_selection_refine_20260927/CONCLUSIONS.md`、`research/2026-09-27-gu-deep-root-cause.md`、`research/2026-09-27-aux-selection-review.md` 和 `DATASETS.md`。历史数据清单、逐题预测及所有审计 JSONL 未全量重算。未启动测试、GPU 实验或联网。
- **未完整阅读全文的非测试 Python 路径：无。** 两个实验目录和上述历史审计范围的重复文件，以字节一致性或完整差异核对其语义；没有把 65 个实验测试文件及历史 1 个测试文件逐行精读。此阅读结论仅指本任务明确划定的目录，不扩展为整个仓库所有源码。

## 代码结构与阶段边界

1. `visual_agent` 的旧路线以 C 直接定位和候选池为起点。`vision_tools.py` 生成 RGB/IR 检测与掩码、深度统计及候选图册；`controller.py` 维护消息、工具预算、动作解析和终局；`object_controller.py`/`object_tools.py` 另有对象级 `inspect`、`search`、`depth`、`finish` 接口。`export_trajectories.py`、`export_recovery_demonstrations.py`、`train.py` 负责行为示范与 LoRA 监督，`evaluate.py` 负责完整分母评分。旧 `controller.py` 的动作是自定义 JSON，主循环依赖模型选择工具和候选 ID；查到合法输出或预算结束并不等于正确定位。
2. `evidence_decision` 是新接口的独立拷贝，不是旧实验已经验证的改名版。新增 `prepare_object_assets.py`、`prepare_grounding_rehearsal.py`、`collect_object_teachers.py`、`merge_reviewed_data.py`、`align_baseline_candidates.py`、`prepare_evaluation_subset.py`、`run_gpu_phase.py` 等，形成固定 C 候选、真实取证、人工盲审、bbox 复习、训练、配对评估链。`object_controller.py:51-56,118-145` 改用原生 `<tool_call>` 语法并明确候选只是框假设；`:160-179,205-218` 加入来源/深度元数据；`:403-450` 保留近近历史与预算信息；`:453-685` 可处理工具错误、恢复和直接 bbox 终局。`object_tools.py:216-240,546-749,902-` 提供公开候选 ID、联合查看、深度状态、搜索缓存和 bbox `finish`。`object_teacher.py:473-583` 将教师恢复动作对齐真实状态。新接口确实修了若干旧协议问题，但不能由接口完善推出成绩改善。
3. 2026-09-27 历史 `aux_selection_refine` 是另一条冻结 C 外挂路线：Query 类别解析、RGB/IR GroundingDINO、SAM2 主体掩码与 City 原始深度形成候选，Qwen 先从候选中选择，再在预测裁剪中输出局部框并映射回全图。`code_snapshot` 保存早期接口；`code_patches` 修空检测标签和数字候选 ID；`code_final` 是实际 412 题运行的 v1；`code_corrections_not_evaluated/predict_aux_selection.py:32` 是后续居中截边修正版。不能把修正版源码对应到实测 221/412。

## 已有结果：只按实际可比口径使用

| 条件 | ACC@0.5 | 同条件证据与限制 |
|---|---:|---|
| 冻结 C1500，城市开发 412 | 292/412，mIoU .613628 | 各路线共同参照；反复使用的开发集，非官方测试。 |
| 历史冻结 C + 候选选择 | 273/412，mIoU .577475 | 对 C 纠正 10、破坏 29。 |
| 历史候选选择 + 局部细化 v1 | 221/412，mIoU .471095 | 对选择再纠正 1、破坏 53；修正版未 GPU 评价。 |
| 旧 `visual_agent` 重构未训练控制器 | 212/412，mIoU .460494 | 对 C 纠正 17、破坏 97，候选覆盖未增加。 |
| 旧对象工具新接口，同 C 权重 | 225/412，mIoU .485008 | 旧阶段接口，非新 `evidence_decision` 配对结果。 |
| 新 `evidence_decision` 同 C 权重，未训练 | 217/412，mIoU .468091 | 同 412 题 C 候选；394 合法终局，纠正 11、破坏 86，净 -75。 |

来源：历史 [`CONCLUSIONS.md`](../../../docs/audit-20260927/evidence/aux_selection_refine_20260927/CONCLUSIONS.md) 第 13-23 行；[`visual_agent/RESULTS.md`](../../../experiments/visual_agent/RESULTS.md) 第 38-61 行；[`evidence_decision/RESULTS.md`](../../../experiments/evidence_decision/RESULTS.md) 第 76-85 行。旧学生 200 题为 145/200、直接 C 同 200 题为 152/200，但其中 106 题/129 图组与训练源重合，且缺少同接口训练前对照；它只能诊断训练源行为，不能声称正式 412 分数或独立泛化。新路线开发 40 是 C 29/40、新接口未训练 24/40；最终 80 未评分、正式新 SFT 未启动。最新 1064 行仅为准备中课程快照：314 条审核教师 + 750 条 bbox 复习，113 个教师终局全部选 ID，没有合格自主 bbox 终局。见新 `RESULTS.md` 第 65、76-86 行及 `POSTMORTEM_20261001.md`。

新 412 对照还有更具体的行为：初始候选覆盖 339/412；搜索仅增 1 题覆盖；初始未覆盖 73 题全部失败；自主 bbox 输出 2 次且都错。`diagnostic_city412_pretrain` 记录了大量重复/缓存查看；因此核心障碍既包括候选外目标，也包括候选内选择和信息使用。`8192` 输入上限、同一 C 候选池的该对照不能简单归咎于旧阶段 `4096` 超限；但这仍只是未训练行为，不能判断新 SFT 的效果。

## 失败机制与证据强度

**历史候选选择/细化。** 候选池离线 GT 覆盖 339/412，比 C 多覆盖 47 道 C 错题，但 66 道已有正确候选仍被选错。RGB 检测原框覆盖 238、IR 原框 96；IR 相对该 RGB 检测在 C 错题上没有独有覆盖。此为该构造/阈值下的覆盖结论，不能推论红外模态无价值。进入细化 270 道，166 KEEP、104 数字框；数字坐标反算最大算术误差为 0，却有 53 道把选择正确改错。53 道中的 25 道，原始数字若事后按**全局**而非指定的裁剪图坐标解释，可以命中；这支持模型常混淆参考图坐标，不代表可部署地忽略映射。原 v1 边缘裁剪策略另有 41 道边界差异、1 道一像素差，波及 11 道细化退化，但其余 42 道并未受影响；修正版没有 GPU 成绩。16 条预检只验证格式未验证定位身份，事后 5 次数字细化都未命中，说明放行门槛没有测到关键行为。见历史 `CONCLUSIONS.md` 第 37-85 行及 `code_final/predict_aux_selection.py` 与未测修正版的差异。

**旧多步 Agent。** 旧根因审计记录：候选 ID 被当作对象身份，RGB/IR 同位置框被合并后互相替代；0.9 IoU 去重和带框概览降低区分度；模型频繁选择 RGB 查看，搜索/深度目标动作缺少充分训练；深度 `UNKNOWN` 后仍会作超证据断言，且有空间关系/相机近远混淆。旧 2250 行/282 步 SFT 主要是训练源行为模仿，搜索与深度动作的真实学习信号极弱；T1/T2/T4 留出 50 对 C42 分别 40/37/41，开发 78 对 C60 分别 57/54/60，未显示稳定净收益。部分旧 `AUDIT_ROOT_CAUSES.md` 所指协议缺陷已在新 `object_controller`/`object_tools` 修复；不应将旧源码缺陷无条件当成当前代码缺陷。

**新证据决策。** 新实现对候选语义、历史保留、直接框终局和错误恢复作了针对性修改，却仍在同权重训练前 412 对照出现 11 改对/86 改错。不能把退化硬归因于单一模态、LoRA 或教师质量；正式 LoRA 尚未训练。`POSTMORTEM_20261001.md` 明确记录用户据此叫停、没有评分最终 80。当前可成立的判断是多步决策的净损失大、课程数据的关键 bbox 能力未补齐，继续扩大采集前应先定位具体退化路径。

**历史 G/U 三模态数据审计。** 完整 412 上 M2 285、C 292、等预算 B/G*/U* 287/284/286。每臂 1600 次是呈现次数，不是 1600 条独立新 Query；新增约 98 条 RGBDT Query/图组反复呈现。U* 在固定预算内用 128 次 IR 框、55 次 Depth 框替换了 RGB 框呈现；新增 Depth 没有经核准的近远层级监督。83 个辅助标签中，已知正确 RGB GT 框直接复制坐标就有 81 个可过 IoU .5；这是离线标签几何捷径，不是 RGB-only 模型得分。新增数据对小目标、竞争实例、参照关系的信息密度低，且占用既有 RGB 预算；更可信的根因是训练信息与最终错误类型不匹配，而非已证明某一模态无用。见 `research/2026-09-27-gu-deep-root-cause.md` 第 27-61、129 行及 `DATASETS.md` 第 30-32 行。

## 当前源码中值得优先核对的点

1. **bbox 复习样本可能丢失所有图像，阻止下一次训练。** `prepare_grounding_rehearsal.py:67-76` 用 `build_initial_messages(task,..., images=[])` 导出；`object_controller.py:247-253` 对路径不存在的 RGB/IR/Depth 直接 `continue`。若在本机导出云端绝对路径的 manifest，就会生成纯文本复习样本；`train.py:67-107` 只检查前缀/完整提示的图像网格相同，不要求有图，因此 CPU 预检也可通过。仓库当前未找到 1064/865 行的原始 `train_full.jsonl`，**尚未直接证实那 750 行已经无图**。先对原始 JSONL 数图像块、核对导出机路径，再决定修复；修法应在导出边界要求存在至少 RGB 帧并 fail fast，不要静默补空图或降级为无图训练。该风险不解释已经跑完的训练前 217/412。
2. **旧探针入口与原生工具监督格式脱节。** `probe_decisions.py:34-79` 的 `probe_rows` 仍执行 `json.loads(row['target_action'])` 并期待 `target['action']`，而新复习导出 `prepare_grounding_rehearsal.py:96-105` 的目标是 `<tool_call>{"name":"finish",...}</tool_call>`。拿新 SFT 行调用旧 `--decisions` 路径会解析失败；同文件的 `--paired-decisions` 原生路径是另一实现。需使用正确入口或统一解析器，别把此错误当成模型行为。
3. **覆盖与成绩必须分别看。** `object_tools` 的候选、深度和缓存机制具有诊断价值，`evaluate.py` 的完整分母、合法框和配对纠错统计可复用；但候选覆盖 339/412 不是系统 339/412。下次任何接口改动先以冻结 C 同图组、同候选、同预算小样本观察“纠正/破坏”和目标身份，再决定是否训练。
4. **历史修正版不可替代负结果。** `code_corrections_not_evaluated/predict_aux_selection.py:32` 明示 `centered-clip-v2-not-gpu-evaluated`；实际 221/412 只能引用 `code_final` 的 v1。保留历史脚本、逐题 ID、负结果作为诊断资产，不恢复已停的多步 GPU 路线。

可保留的价值是：冻结 C 锚点和完整分母评分、来源清楚的 RGB/IR/Depth 候选与主体深度统计、成对行为审计、训练/留出图组隔离，以及历史错误样本和图册。它们适合作为单步推理或明确受控的小实验的输入与检查工具；目前没有证据支持把旧多步控制器或未训练新接口作为主提交路线。
