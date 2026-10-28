# 队友 RGBDT500_prepared 批次来源与文字风格

## 实际恢复范围

本地原始 `grounding_final_train.json` 未找到；已从 Arm5 训练产物恢复 **919 个唯一训练记录**，不是历史记载的全部 992 条。

| 本地来源 | source=rgbdt500_prepared 行数 | 唯一 ID |
|---|---:|---:|
| `F:/AIC/results/arm5_artifacts_20261005/slices_arm5_mixture.jsonl` | 919 | 919 |
| `F:/AIC/results/arm5_artifacts_20261005/slice_trimodal.jsonl` | 878 | 878 |
| 两文件合并 | 1797 次呈现 | 919 |

第二份文件的 878 条完全在第一份中。纯来源恢复文件为 `F:/AIC/.work/rgbdt_annotation_comparison/teammate_recovered.jsonl`，以 `rgbdt500p:train_<序列>_<帧>` 为去重键；每条附 `recovered_from` 原文件绝对路径与行号。Query、类别、序列、原图路径、weak_label 等均从真实训练行恢复。框来自训练回答的 0–1000 整数坐标，只有 0.001 精度；原始归一化框的小数精度未恢复。

919 条包括 **878 trimodal（三模态切片）与 41 small（极小目标切片）**。`slices_arm5_summary.json:13` 直接记载 mixture 使用队友来源 919 条；构造器 `code/tools/build_capability_slices.py:393` 将极小框放 small 池，其余放 trimodal；`:476` 的预混合超过配额时无放回抽样。历史 992 减去 878 得到 114 个潜在 small，mixture 选中 41 后缺 73，**符合小目标池抽样缺失的解释，但未拿到原清单前不能确定缺失的全部原因**。构造器另有空 Query/超过 300 字符过滤，不能排除其影响。

可继续取回的原路径为 `/root/autodl-tmp/rematch_20260922/data/external/RGBDT500_prepared/RGBDT500_prepared/train/grounding_final_train.json`；原 ZIP 曾为 `/root/RGBDT500_prepared.zip`。完整切片还可能是 `/root/sprint_qwen36/slices_arm5/slice_small.jsonl`，其中可过滤该 source。当前没有连接远端。

## 来源证据与身份边界

`F:/AIC/code/tools/build_capability_slices.py:335` 将此来源称为“队友新传”，记载 992 条、121 类；`:350` 记载“Qwen API 生成 + 同模型自审”。这是已有代码中的来源记述，**没有找到该批生成提示词、模型请求日志、原 dataset_report.json 或逐条真人复核日志**，因此不能把注释视为独立核验过的完整生产流程。

恢复的 919 条全保留 `weak_label=true`，有 118 个 class_name、355 个 sequence_id，每个帧 ID 只有 1 条 Query。序列中 247 个各 3 条、70 个各 2 条、38 个各 1 条；这些是连续视频/序列样本而非 919 个独立场景。没有直接证据区分此批由 sql 或 zjq 中哪位具体制作，不指定姓名。`weak_label=true` 只是字段本身，不能单独证明后续从未人审。

`data/quant_grounding_source/rgbdt_train.jsonl` 是另一来源：98 条，首条 label_source 为 `released181_rgb`，不能与这批 992 混称。历史 `target_v2.zip` 的 992 条则是 City 弱监督，927 个 sequence，与 RGBDT500_prepared 同数不代表同一批。

## 919 条恢复子集的风格

| 指标 | 结果 |
|---|---:|
| 以 The 开头 | 918/919 |
| 问号 | 0 |
| 字符数中位/P90/范围 | 53 / 66 / 27–88 |
| 单词数中位/P90/范围 | 10 / 13 / 6–18 |
| ≥17 词 | 1 |
| 不同 Query 文本 | 884 |
| 重复文本涉及记录 | 60 |
| 颜色/明暗词 | 632（68.8%） |
| 左右词 | 223（24.3%） |
| 序数/极值词 | 17（1.85%） |
| 关系/前景背景词 | 350（38.1%） |
| 明确红外/热词 | 0 |
| 距离/近远词 | 9（0.98%） |

词面规则与计数见 `teammate_style_statistics.json`；类别可重叠。“closest to the left bank”也能靠 RGB 关系完成，9 条命中不能解释为 9 条真实深度监督。重复文本多发生在同序列不同帧，不能直接判为错标。

主导形式是“定冠词 + 颜色/类别 + 动作/位置/参照物”的短名词描述，文字通顺、对象类别丰富，明显不是纯粹把类别名照抄成 Query。现有证据支持它适合作为 RGB 外观和位置描述补充，但其长指令、序数及显式跨模态覆盖较薄；不能据此断言框不准或整批语义错误。

可直接引用的原始例子（均为 `slices_arm5_mixture.jsonl` 行号）：

- **5594，train_001_00000020**：`The white umbrella situated further back near the building entrance.` —— 颜色、相对纵深、参照物组合。
- **4914，train_112_00000001**：`The second teal bicycle from the left with a black basket.` —— 少量明确序数例子。
- **1，train_245_00000170**：`The large wooden sign board located on the right side.` —— 典型外观+绝对位置短句。
- **4225，train_328_00000042**：`The white duck swimming in front of the blue boat, to the left of the other white ducks.` —— 本批最长句之一，确有同类竞争对象指代。
- **5385，train_390_00000001**：`The lemur furthest away near the top edge of the grass.` —— 近远极值与画面位置混合，未证实需要物理深度。

没有逐条重看对应图像，因此本次不报告 Query↔框错误率，不把模态词少归因为成绩下降，不以任何 Arm 结果解释数据质量因果。

## 给主任务的建议

新增标注应保留这批的简洁英语与准确外观线索，同时补足它缺少的对象单位、同类候选范围、序数计数方向、清晰参照物和有实际 IR/Depth 证据的消歧。每个框分别记录目标是否包含人/载具/手持物等范围约定；同图多目标先统一候选集合，再生成 Query，避免只把既有跟踪框改写成多条近义句。精标与本批按序列/场景隔离做等训练量对照，才能判断精标收益与粗标噪声。

交接缺口：缺原始 992 清单中的 73 条、原框精度、生成/复核流程直接证据、实际图像语义复核、具体制作队员身份。
