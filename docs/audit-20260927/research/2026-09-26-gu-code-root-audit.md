# G/U 首轮未提分：训练与评估代码独立审计（2026-09-26）

## 结论及边界

依据 `results/gu_pilot_20260926/final_cloud_snapshot/` 的真实 Linux 落盘清单、消费轨迹、训练状态、日志及预测，**未发现能直接解释 G200=288/412、U200=283/412 的已证实训练或评分实现错误**。两组确实从 M2 LoRA 初始化，以相同顺序消费 1600 条、各完成 200 次更新；412 条验证输入一致，解析全部成功，生成均未触及 128 token 上限。G 相对 M2 净增 3 题，U 相对 M2 净减 2 题；C 为历史不同训练预算的 292/412，不能把 G/C 的差值归因于 G/U 策略。

**已证实的实验设计混杂因素**：G/U 之间的 183 条辅助模态呈现不仅改变了答案坐标系，还改变了提示措辞；U 的辅助提示省略了 G 提示中的深度图数值解释。因此本次是两种完整训练包装策略的成对对照，无法从 5 题差距单独识别“监督坐标系”的因果效应。它并非相对既定方案的执行 bug，也没有证据证明去掉深度说明造成了降分。

本审计只做 CPU/只读核验；没有加载模型权重、做新推理或训练，不能声称观察到模型注意力、视觉特征使用、梯度方向或辅助模态表征质量。本文不以早期 Windows dryrun 清单代替最终云端清单。

## 实际数据与唯一差异

脚本 `code/tools/prepare_gu_same_day_pilot.py:183-191` 对非 RGB 新任务仅把 G 的 `conversations` 换成同图同 Query 的 RGB 兄弟任务；U 保留原对话。该脚本 `:216-255` 每个 8 条梯度累积块安排 6 City + 2 新任务。云端 `manifests/seed2026/{g_train,u_train}.json` 和 `metadata.jsonl` 独立逐行核验：1600/1600 ID、图像路径、排列全相同，提示与答案均恰好 183 行不同，即 IR 128、Depth 55；另 217 条新 RGB 和 1200 条 City 训练对话完全相同。三图占位符每行均为 3。云端 `summary.json` 记录 98 个新 Query 图组、181 个输出任务，新任务呈现共 400；这是一个小范围、多次重复的补充监督，不是 400 个独立新 Query。

已确认 U 的 183 条目标提示明确写 `infrared image` 或 `depth image`，答案均为 `bbox_2d` 的 0–1000 整数；G 对应行明确写 RGB 坐标。`code/tools/prepare_next_stage_data.py:209-218,250-260` 与 `code/tools/prepare_same_day_pilot.py:147-166` 给出原始转换链，`code/tools/prepare_qwen3vl_native_sft.py:22-34` 将 0–1 xyxy 标签乘 1000 后四舍五入。云端逐行核验 183 组 G/U 答案的四坐标均不同；IR/Depth 各呈现的最大单坐标差中位数为 12/15（千分坐标单位）。这符合不同图像坐标系的设计，不能据数值差认定标错。

图像路径也逐列核验：最终两组 manifest 的第 1 列为 City `visible` 1200 / 新 RGBDT `color` 400，第 2 列均 `infrared`，第 3 列均已可视化的 `depth_rgb`。`code/tools/prepare_next_stage_data.py:173-206` 把新 RGBDT 原始深度的正整数传给固定反向灰度映射，零值映为无效黑色；`code/tools/prepare_qwen3vl_native_sft.py:37-57` 的函数实现与 G 的深度文字说明相符。云端同步回执曾逐路径 PIL 校验，当前本地文本快照本身不含全部原图，故此处只确认路径/转换代码与先前回执，不声称重新逐像素审查 294 张图。

提示混杂因素可精确复现：G 的全部 183 条辅助位置采用 RGB 兄弟提示，包含 `This depth visualization uses a fixed inverse mapping ... physical distance unit is not established`；U 对应 183 条采用 `Views are in this order: rgb, infrared, depth ... Return its box in the infrared/depth image`，均没有深度数值说明。217 条新 RGB 的两组提示都包含该说明。来源为 `code/tools/prepare_next_stage_data.py:216-218,250-260`、`code/tools/prepare_qwen3vl_native_sft.py:94-123` 和最终清单逐行计数。City 原提示使用 `These are aligned views...`（同文件 `:83-90`），新 RGBDT 数据使用 `These are views of the same scene...`；新域提示与 City 不完全相同，是两组共同承受的域/模板变化。IR/Depth 辅助提示未宣称三图像素对齐，故未看到“把非对齐数据当同坐标”的明确错误。

## 训练执行、损失与恢复

`code/scripts/run_gu_pilot.sh:80-103,164-171` 首段将 M2 checkpoint-928 作为 **adapter 初始化**，第二段从各自 checkpoint-100 **完整恢复**；传入总步数始终 200，LR 5e-6。实际 `g_2026/native_train_config.json` 与 `u_2026/native_train_config.json` 除训练清单和输出目录外，模型、初始化、种子、像素、LoRA、优化器、累积、顺序等一致。`code/scripts/run_qwen3vl_native_lora.sh:93-137,399-449` 固定语言模型 q/k/v/o 的 rank32 LoRA、BF16、SDPA、batch1×accum8、线性 200 步；视觉编码器和多模态投影器没有训练参数。代码核验 PEFT 载入的 288 个 LoRA 张量与 M2 文件逐张量相同，日志有 `initialized and verified 288 ... fresh optimizer and scheduler`。预检的优化器/scheduler 1→2 恢复已在 `preflight_receipt.json` 和 `logs/preflight_*` 留证；正式两组 checkpoint-200 的 `trainer_state.json` 均是 global_step=200、末尾 LR=2.5e-8。不能仅从文本快照重算正式 checkpoint 的权重/优化器二进制内容，但没有见到重启 LR 或丢失更新的迹象。

顺序不是凭 dataset `__getitem__` 推断。启动器 `code/scripts/run_qwen3vl_native_lora.sh:490-510,627-669` 启用 SequentialSampler、单工作线程，并在成功 `training_step` 后记录 ID；100 步恢复时按 checkpoint 裁定前缀。最终 `g_2026/consumed_samples.jsonl` 和 `u_2026/consumed_samples.jsonl` 各 1600 行，均逐条等于相应最终 manifest 顺序，`run_gu_pilot.sh:151-161` 的门槛检查也通过。由此排除二次打乱/漏样作为本轮主要原因。

监督使用上游 Qwen 数据处理器：`tmp/source-audits/Qwen3-VL-current-20260922/qwen-vl-finetune/qwenvl/data/data_processor.py:140-199` 按 `<image>` 顺序从路径列表填入图像，`:202-240` 用 `-100` 屏蔽用户提示及视觉 token，只对 assistant 回答及结束符开 loss；不是专用 IoU 损失。启动器 `code/scripts/run_qwen3vl_native_lora.sh:587-625` 在预检 8 个代表样本、正式运行首样本解码受监督 token 并检查完整答案。云端 `logs/preflight_resume2_20260926T183756_attempt1.log:299-314` 的 8 个样本覆盖 City、新 RGB、IR、Depth，各有 3 个图像 grid、完整 `bbox_2d` 答案、输入约 1831–1883 token，低于 4096。其余 1592 条未逐一留下 token 解码审计，不能把代表样本检查写成全样本严格证明；清单提示最长 589 字符、答案最长 31 字符，未见异常长文本。

两组 checkpoint-200 日志每 10 步 loss 与梯度范数均有限。G 平均已记 loss 前/后 100 步约 0.5090/0.5013，U 约 0.5400/0.5298；U 较高并不能跨不同答案坐标系直接解释性能。首更新 288/288 可训练张量有梯度，U 第 10 步记录的梯度范数 6.46875，高于 G 同步记录，但按 `max_grad_norm=1.0` 裁剪且随后正常，不构成已证实数值故障；日志不提供逐层梯度或模态使用率。

## 评估一致性与历史 Query

`code/scripts/run_gu_pilot.sh:106-126,164-180` 对 G/U 的 100/200 检查点使用相同 `trimodal_val.json` 与原始 `qwen_generation_val.json`。评估实现 `code/tools/evaluate_pretrained_grounder.py:136-159,246-290,416-448,644-666,676-748` 按对话中的 `<image>` 顺序送三图、载入指定 LoRA、从 0–1000 框解析为 0–1，再以原始浮点 GT 算 IoU≥0.5。最终 G/U 412 行按 ID 的提示、图像路径和 target **完全相同**；每组 412/412 解析、3 个图像 grid、无生成上限命中，生成最多 25 token。`run_config.json` 的基座、像素上下界、128 生成 token、数值精度设置一致。

进一步把 `results/next_stage_20260925/baseline_m2/predictions.jsonl`、`c_step1500/predictions.jsonl` 与最终云快照 G200/U200 按 412 个 ID 对齐：**四组的 prompt、image、target 逐项完全相同**，`target_source_kind` 全部为 `target_manifest_bbox`，共同指向 `target_v2/qwen_generation_val.json`。例如 `city_000004_018_00000341_008` 四组均评 `The person standing near the central street light`，GT `[0.532813,0.110185,0.543229,0.140741]`。图册 `pre_review_original_query` 记录更早的人工审核前描述（如 `leftmost upper`），不可据此认为本轮 M2/C/G/U 混用旧 Query。此证据只保证四个**落盘评估**公平；不能逆推 M2 早期训练时每条 Query 的版本。

`results/gu_pilot_20260926/final_cloud_snapshot/report_2026/report.md` 以原始 GT 重算：G200 288、U200 283、M2 285、C 292；G→U 是纠正 1、退化 6，mIoU 反而 +0.0018，412 条中只 7 条跨 0.5 阈值。78 图组 bootstrap 的 G→U ACC 差区间为 [-2.75, 0.00] 个百分点。`results/gu_pilot_20260926/atlas_final200/failure_review.md` 记载这 7 条主要是同目标附近的框界变化，未见整批解析或坐标尺度故障；它不证明模型使用了哪种模态。

## 根因判断与最小后续证伪

1. **已排除的常见执行错误**：未见不同初始化/总 LR 计划、二次打乱、图像顺序对调、提示与目标坐标不符、assistant loss 全遮挡、评估 Query/GT 混版、128 token 截断或大规模框解析失败。落盘证据强度如上，正式全部样本的 loss mask 和权重变化没有逐条离线复测。
2. **已证实但未定因果的设计事实**：400/1600 新呈现来自 98 个独立 Query 图组，其中真正区分 U 与 G 的辅助坐标任务仅 183/1600（IR 128、Depth 55）；U 同时换提示模板并去掉深度说明。模型只训练语言 q/k/v/o LoRA，不更新视觉编码器。标准答案 token 交叉熵可以学到回答格式和近似坐标，但 loss/ACC 不能证明建立了 IR/Depth→RGB 互补推理。U 相对 G 的有限差距与这几个因素均相容，不能归咎任一项。
3. **最省成本的进一步鉴别**：先用现有 checkpoint 做 CPU 清单/错例分析，分层量化 7 条阈值翻转的边界与实例差异；若另批 GPU 实验获批准，才在同 98 Query、同标签、同更新顺序下只统一辅助提示措辞与深度说明，再比较固定 200 步。要证实模态利用，需同 checkpoint 的模态遮蔽/交换或坐标输出探针，不能由这份训练日志推断。当前报告不启动新实验，也不建议直接放大至 600/1500 步。
