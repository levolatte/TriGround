# 2026-09-27 接手审计：数据链与训练呈现

本稿只审计已保存的数据、人工决定、生成代码和实际消费轨迹，不修改标签或训练实现。独立复算程序为 `recompute_data_coverage.py`，运行后写 `data-coverage.json`；程序只用 Python 标准库，读取 `code/docs/audit-20260927` 审计快照。逐文件阅读覆盖表为 `data-source-coverage.json`：当前数据责任源码 51/51 文件、8372/8372 行，历史注释构建器 12/12 文件、742/742 行；源码阅读不等于所有执行路径均已运行。复算成功于 2026-09-27。本稿涉及比赛任务定义，已对照 `contest/基于大模型的多模态视觉理解与推理.pdf` 第 2–4、6–8 页：最终评估目标是 RGB 坐标系归一化框，ACC@0.5 是主指标；官方材料称官方原始 Depth 为 16 位毫米值，但不能据此推定外部 RGBDT/Robo 数据的单位。

## 数据血缘及人工门槛

RGBDT500 获取程序从作者 `Train.zip` 的 400 序列中选择有作者框的帧，保存 RGB、IR、原始 16 位 Depth，同时将 Depth 按固定反向数值映射成展示图；它明确把原始单位标为 `unverified`，且源数据没有自然 Query（`code/tools/acquire_rgbdt500.py:125-228,245-309`）。后续 `trial200` 对象证据的盲审只产生临时 `blind_passed`，并未自动授予人工批准（`code/tools/prepare_multimodal_evidence.py` 的 `merge_review`、`export_rows`；`code/tools/prepare_same_day_pilot.py:34-100`）。其中 blind reviewer 的任务不提供目标框，目标框存在私有索引；机器通过条件是预测框 IoU≥0.5 且无歧义、证据确认。这里的盲审仍不等于独立图像真值。

两份实际 CSV 各 100 行：自然 Query 决定为 `correct_unique=98, uncertain=1, wrong=1`，对应关系决定为 `accept=97, uncertain=1, reject=2`。从 `cloud_snapshot/source/pending_candidates.jsonl` 的 252 个候选任务逐一与 CSV 及放行记录联结，181 条放行任务全部满足自然 Query `correct_unique`，辅助任务还满足对应关系 `accept`。未放行 71 条中，按实际导出优先级为 5 条被拒、66 条未具备放行决定；一条自然决定缺失但对应关系为 `uncertain` 的任务仍归待审。生成和导出门槛见 `code/tools/prepare_same_day_pilot.py:121-170`。独立脚本还逐条检查 `source/reviewed_evidence.jsonl` 中 Query→`target_object_id`→同对象分模态框，与 `pending_candidates` 及 181 任务的文本、模态、RGB/IR 路径一致。放行不是 181 个独立场景：按 `source_id + Query` 为 98 条 Query，按 `group_id` 为 98 个图组，输出模态为 RGB 98、IR 58、Depth 25。

放行任务从原始候选浮点框量化为 0–1000 整数答案。`code/tools/prepare_qwen3vl_native_sft.py:22-34` 用十进制 half-up（五入）量化；`code/tools/prepare_next_stage_data.py` 的 `_native_sample` 对 RGBDT 用真实三图、Query 和 `sensor_linear_20000` 提示，辅助目标另指定 IR/Depth 坐标。`code/tools/prepare_same_day_pilot.py:147-169` 写入任务和元数据，后者记录 `review_status=human_accepted`。物理距离单位在当前提示里明确写“未确立”，而 `code/tools/prepare_next_stage_data.py` 的 `_nearfar_task` 仅允许 `millimeter`/`meter`，排除了此类 RGBDT 数据的数值近远题。放行对象对应框只教同一 Query 在不同图像的框位置；它没有产生经过人工核准的“辅助模态决定最终 RGB 目标”的关系标签。

City 当前来源快照是 `source_snapshot/city_train.json` 3707 条、681 个 RGB 图组；`city_val.json` 与 `city_gt.json` 是 412 条、78 个图组。训练与验证清单的精确 Query ID、RGB 路径交集均为 0，但**这不是场景独立证明**。`cloud_snapshot/report_city412_diagnostic/summary.json` 保存的已知地点重用名单为 47 条、7 个验证图组；脚本验证名单与 412 条验证 ID 完整互斥划分为 47/365。名单依据是此前七组地点目视判定，本文没有重新独立目视认证；余下 365 条仅是未列入这 47 条。当前开发集 412 的 Depth 路径后缀均为 `.png`；主线程另对官方复赛测试 `queries.json` 核实 5690 条中 Depth 路径为 5515 个 `.png` 和 175 个 `.jpg`，其无 GT。文件后缀只说明路径，不足以证明解码后的位深或物理单位，尤其不能将 `.jpg` 数值直接按毫米解释。

## B/G*/U* 的真实训练呈现

旧 G/U 计划构造是每 8 条中 6 条 City、2 条新来源；每 200 步有 1200 个旧 City 呈现和 400 个新来源呈现，新来源任务分模态 217 RGB、128 IR、55 Depth（`code/tools/prepare_gu_same_day_pilot.py:168-180,216-255`）。`code/tools/prepare_gu_diagnosis.py:260-339` 读取旧序列及 181 放行任务，保留 98 条源 Query 的重复频次；`:341-383` 为 B 映射 98 个新的 City 训练 Query；`:392-460` 在旧 400 个位置替换三臂样本。B 是新 City Query；G* 是同一 RGBDT Query 的 RGB 框；U* 是该源任务指定的 RGB/IR/Depth 框。G*/U* 新位置的人类提示主体相同，目标图坐标文字和答案按模态变化（`code/tools/prepare_gu_diagnosis.py:89-99,247-256,401-443`）。共同旧 City 位置逐行照拷，不应称三臂共有 4800 道新题。

我读取了三份实际 `train_{b,gstar,ustar}.json`、逐行元数据，以及三份 `cloud_snapshot/{b,gstar,ustar}/consumed_samples.jsonl`。三臂各 1600 个唯一 presentation ID（呈现编号），且各自消费 ID **逐位置**等于训练清单和元数据 ID；共同的 1200 个旧 City 位置三臂完整样本对象一致。400 个新位置中，G*/U* 217 行（RGB 监督）完整一致；183 行（128 IR + 55 Depth）图像和 Query 仍一致，但目标坐标提示及答案各不相同。G*/U* 的新来源仅 98 条源 Query，重复呈现 400 次。B 的 400 个位置映射到 98 条新 City Query、98 个图组，而这 **98 个图组全部已在共同旧 1200 位置出现**；故 B 与 G*/U* 的比较含图像熟悉度及来源差异，并非纯标签信息对照。消费文件只记 ID，不逐字记录读入时的图像张量、token 或梯度；这部分执行语义需与训练入口及云端模型证据联审。

三条可从实际记录定位的字段示例：

| 呈现 ID | 源与人工决定 | 三臂在该位置的监督含义 |
|---|---|---|
| `gu200_000005` | G*/U* 指向 `rgbdt500_056_00000227::q01::rgb`，`source_id` 是 `...::q01`，`group=056`；自然决定 `correct_unique` 在 natural CSV 第 97 行；放行元数据第 29 行。候选第 40 行的 `object_id=target`、RGB 浮点框映射到鸟目标。 | G*/U* 都要求 RGB 坐标 `[391,532,497,715]`，对应 217 条全行同之一；B 同位换为 City 的 `city_hehe_22_00000004_001`，框 `[332,437,353,518]`。位置 ID 表示呈现，不表示源 Query ID。详见 `train_*_metadata.jsonl:5`。 |
| `gu200_000002` | G*/U* 源任务 `rgbdt500_027_00000301::q01::infrared`，自然决定 `correct_unique` 在 natural CSV 第 32 行，对应决定 `accept` 在 correspondence CSV 第 28 行；候选第 24 行与放行元数据第 18 行均标明 `target_modality=infrared`。 | 两臂读同图同 Query；G* 要 RGB 框 `[534,447,902,918]`，U* 要 IR 框 `[526,443,911,928]`。B 同位为 `city_000014_025_00000184_001`，Query 是白色 A 字立牌，框 `[474,503,568,725]`。`train_*_metadata.jsonl:2`、`consumed_samples.jsonl:2` 可串起身份。 |
| `gu200_000091` | G*/U* 源任务 `rgbdt500_310_00000337::q01::depth`，自然决定 `correct_unique` 在 natural CSV 第 62 行，Depth 对应决定 `accept` 在 correspondence CSV 第 94 行；候选第 186 行和放行元数据第 138 行指向同一 `target` 伞篷。 | G* 要 RGB 框 `[377,399,691,495]`，U* 要 Depth 图坐标框 `[369,389,707,516]`；B 换为 City 大象牌框 `[56,766,210,1000]`。`train_*_metadata.jsonl:91`、`consumed_samples.jsonl:91` 核对呈现。Depth 框是图像对应监督，不提供米制距离真值。 |

以上三条的详细字段和值也保存在 `data-coverage.json` 的 `examples`。JSON 内的 `candidate` 浮点框仅用于标注与训练答案；训练提示没有携带这些数值。`fit_legacy181_gt.json`、`fit_canonical_aux83_gt.json` 和 `city96_gt.json` 是评估旁路 GT 文件；本次只从数据清单与生成代码确认其未写进三臂训练提示，尚未独立审计所有推理入口对旁路文件的读取边界。源任务 `rgbdt500_056_00000227` 的 IR 框上边界相对 RGB 差异较大（候选第 40–41 行），说明分模态框不是总由 RGB 数字直接照抄；几何位置接近也仍可能让模型只靠 RGB 推断，不能据此宣称真正利用了 IR。

## 历史生成器、风险与证据边界

`annotation_builders/` 12 个历史 Python 已逐文件阅读。`build_rgbt_batch001.py` 的 `META` 字典硬编码 26 个 RGBT 对象的 IR 框与图像文字判断，明确 Depth 为 `None`，并仅写 `provisional`；10 个 `make_city_batch002/004_part*.py` 以 `MANUAL` 固定手写对象描述、IR/Depth 框，默认 `provisional`，部分因辅助证据不足保持空 Query。`make_natural100_acceptance.py` 从 121 个 RGB 机器通过候选中固定种子抽 100 项，并导出**初始空白**人工决定表；不会自己写出接受决定。这些是历史一次性材料构建器，路径硬编码到旧 `F:/AIC/results/...`，不能当当前会自动运行的标注器，也不能把里面的英文图像描述视为新生成的 GT。它们不是本次 181 放行的授权来源，授权以实际人工 CSV 和导出记录为准。

已证实的主要口径风险：把 181 个分模态输出当 181 个独立 Query、把 400 重复呈现当 400 条新 Query、把 B 当图像完全陌生的 City 对照、把 47 名单之外 365 题称为严格独立验证、把几何对应正确当作 IR/Depth 必需、把外部 16 位 Depth 数值写成毫米。这些都会高估当前新监督的解释力。原生 City 提示及 B/G*/U* 新提示并不逐字同模板：旧 City 被原样保留，新 400 使用统一模板，B 也是 City 原 Query/框重包装；需将提示包装作为比较限制。正式评测框需 RGB 坐标；U* 的 IR/Depth 输出是辅助训练目标，不能直接等同最终比赛输出。

未验证：审计快照缺原始图像和云端权重，本文没有新一轮全量视觉复核、深度物理标定、像素级配准检查、模型读模态因果检验或官方私有测试成绩。尽管本地有部分 RGBDT 源图路径，逐条对象绑定的本次独立核验是**结构及数值血缘核验**，不等于重新认定人工框视觉正确。47 地点名单仅做 ID/图组复算；尚不能证明其他 365 没有场景复用。数据生成代码之外的训练执行、评价阈值和模型内部行为交给其他审计范围联审。系统默认 `python -m pytest` 因缺少 `pytest` 模块未能运行上述三个现有 CPU 测试文件；本数据审阅范围未改用项目虚拟环境运行这三项，不安装依赖。其他模块已用项目现有 `.venv` 运行各自测试，汇总见 `validation.md`。本数据链标准库复算脚本已执行通过。
