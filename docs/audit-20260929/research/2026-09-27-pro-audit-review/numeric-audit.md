# GPT Pro 数值精度论点核查（2026-09-27）

本页只核查 GPT Pro 审计第四部分的 BF16/PEFT/AdamW 判断。生产代码、训练任务和历史接手报告均未修改；本轮没有使用 GPU 或启动新训练。证据分为三类：启动脚本与已保存配置、云端既有 checkpoint 的只读 CPU 检查、固定人工梯度的本地 CPU 演示。最后一类不能归因任何比赛成绩。

## 结论

1. **真实 dtype 风险成立，但“整个 LoRA 续训停滞”不成立。**云端既有 M2 checkpoint-928 的 288 个 LoRA 张量全为 FP32；B、G*、U* 的 checkpoint-100 全为 BF16，checkpoint-200 又全为 FP32。对应优化器的一、二阶矩 dtype 与参数同步，只有 `step` 标量为 FP32。对 M2 先舍入到 BF16，再与各臂 checkpoint-100 比较，LoRA A 约 12.1% 元素的存储值改变，B 约 98.3%–98.5%；`BA` **差矩阵**的 Frobenius 范数为初始 `BA` 范数的 11.43%–14.64%。因此“BF16 会吞掉一部分小更新”是可信风险；但 B 和 `BA` 在历史训练中明显改变，不能说整个适配器没有学习，更不能据此解释 U* 成绩。详见 [云端既有 checkpoint 检查结果](./cloud-checkpoint-audit.json)。
2. **同一训练阶段的前后半程发生了已证实的精度策略切换。**[启动脚本](<F:/AIC/code/scripts/run_qwen3vl_native_lora.sh:416>) 初始从 M2 加载时对 `PeftModel.from_pretrained` 传 `autocast_adapter_dtype=False`；续训预建 LoRA 时也对 `get_peft_model` 传 False。[该脚本第 513–540 行](<F:/AIC/code/scripts/run_qwen3vl_native_lora.sh:513>)随后把真实 checkpoint 路径交给 Trainer。但云端安装的 Transformers 5.14.1 `trainer.py:3426` 在读取 checkpoint 时再次调用 `model.load_adapter(..., is_trainable=True)`，没有传该选项；安装的 PEFT 0.20.0 `peft_model.py:1412,1581-1584` 将默认 True 传入升精度流程。其[官方版本源码](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/trainer.py#L3107)与[PEFT 0.20.0 源码](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/peft_model.py#L1293)可复核同一机制。100/200 checkpoint 的 dtype 和文件大小翻倍与该路径吻合。这是**实验数值口径不连续**的明确问题；目前未追溯每一步中哪个具体函数首次改变了所有张量，故“由 Trainer 续训加载触发”仍标记为高度可信的源码归因，而非运行时逐调用跟踪。
3. **不能从 `autocast_adapter_dtype=False` 这个参数本身推定最终 dtype。**它阻止 PEFT 的特定自动升精度，无法保证输入主干是什么 dtype，也无法防止之后另一次 PEFT 加载或显式全模型 cast。PEFT [官方故障排查文档](https://huggingface.co/docs/peft/developer_guides/troubleshooting#selecting-the-dtype-of-the-adapter)说明了“参数存储 dtype”与前向计算 dtype 的区别，[实际升精度函数](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/tuners_utils.py#L2047)在 False 时直接返回。就本项目而言，[Qwen 上游训练脚本快照](<F:/AIC/tmp/source-audits/Qwen3-VL-current-20260922/qwen-vl-finetune/qwenvl/train/train_qwen.py:104>)在 `--bf16` 时加载 BF16 基座；[PEFT 通用适配器迁移代码](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/tuners_utils.py#L1573)把新 LoRA 层移到基座参数 dtype。真实 checkpoint 头部最终证实了本次各阶段的存储类型。

## 口径和原始证据

GPT Pro 文本[第四部分](<F:/Tools/OpenAI/CodexHome/attachments/31dd2787-7afb-4a88-9feb-e77d505076a2/已粘贴的文本.txt>)第 156–197 行提出精度风险，明确说 CPU 示例不是 TriGround 训练复现、没有足够证据断言它是瓶颈。此处同意这一限定。其 `sandbox:/mnt/data/...` 示例文件不在当前工作区，本审计没有把它当作可复核的本地代码；另用[本地演示脚本](./cpu_rounding_probe.py)与[输出](./cpu-rounding-results.json)复测机制。

[启动脚本第 93–123 行](<F:/AIC/code/scripts/run_qwen3vl_native_lora.sh:93>)确实指定 `--bf16`、`adamw_torch_fused`、线性学习率调度和 LoRA r=32、alpha=64；[U* 已保存配置](<F:/AIC/results/gu_diagnosis_20260927/cloud_snapshot/ustar/native_train_config.json>)保存 5e-6、`max_steps=200`、从 M2 初始化、checkpoint-100 后停顿再续训，以及优化器超参数。该配置没有记录运行时 LoRA/优化器 dtype，故配置本身不能替代张量检查。脚本[第 431–437 行](<F:/AIC/code/scripts/run_qwen3vl_native_lora.sh:431>)在验证初始适配器一致性时先把已存权重转为已加载权重的 dtype 再比较；因此校验能检查载入后的数值对应，却**不能证明 FP32 M2 在新阶段仍以 FP32 存储**。第 450–469 行的可训练参数审计只核名字和目标层；第 357–385 行的首步梯度检查只核存在和有限性，不核实际权重变化或 dtype。

既有云端资料经[只读检查脚本](./cloud_numeric_probe.py)检查 10 个 checkpoint；结果见 [JSON](./cloud-checkpoint-audit.json)。云端运行环境实际为 PyTorch 2.8.0+cu128、PEFT 0.20.0、Transformers 5.14.1。检查只读取 safetensors 头、适配器张量及 `optimizer.pt`，未构造 Qwen 8B 模型。每个 checkpoint 都有 288 个 LoRA 张量，共 30,670,848 个元素。BF16 文件约 61.39 MB、FP32 约 122.73 MB；优化器文件分别约 122.93 MB、245.61 MB。下表是**已保存张量的事实**，不是对训练中每个瞬间的追踪：

| checkpoint | LoRA A/B 存储 | `exp_avg`/`exp_avg_sq` | `step` |
|---|---|---|---|
| M2-928 | 全 FP32 | 全 FP32 | FP32 |
| B/G*/U*-100 | 全 BF16 | 全 BF16 | FP32 |
| B/G*/U*-200 | 全 FP32 | 全 FP32 | FP32 |
| C phase1-500 / phase1-1000 | 分别全 BF16 / 全 FP32 | 分别 BF16 / FP32 | FP32 |
| C phase2-500 | 全 BF16 | 全 BF16 | FP32 |

各 checkpoint 的[LoRA 配置补读结果](./cloud-adapter-configs.json)均为 r=32、alpha=64、`use_rslora=False`、`use_dora=False`、PEFT 0.20.0；因此有效层矩阵为 `2 × B@A`。[PEFT 代码](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/lora/layer.py#L250-L253)明确普通 LoRA 的缩放为 alpha/r。本次低秩 Gram 计算没有乘 2，故下表 `BA` 的**相对变化比例不受影响**，其绝对范数如需视为实际 LoRA 权重增量须乘 2。没有显式展开各层很大的 `BA` 矩阵。

| 跨 checkpoint 对比 | A 精确改变比例 | B 精确改变比例 | `BA` 相对变化 |
|---|---:|---:|---:|
| M2→B-100 | 12.17% | 98.34% | 11.43% |
| M2→G*-100 | 12.15% | 98.51% | 14.64% |
| M2→U*-100 | 12.12% | 98.33% | 12.63% |
| B-100→B-200 | 99.9985% | 99.9999% | 4.29% |
| G*-100→G*-200 | 99.9986% | 约 100% | 4.53% |
| U*-100→U*-200 | 99.9985% | 约 100% | 4.31% |
| C phase1-500→1000 | 约 100% | 约 100% | 10.74% |
| C phase1-1000→phase2-500 | 5.86% | 95.70% | 7.47% |

**舍入口径：**每组“精确改变比例”先将旧 checkpoint 的 A/B 各自转为新 checkpoint dtype，再作元素级精确比较；例如 M2→U*-100 必须先把 M2 FP32 转 BF16，防止把初始化舍入误记作训练更新。`BA` 比较也先把旧 A/B 转新 dtype，再对 `B_new A_new − B_old_cast A_old_cast` 求范数；分母是旧 checkpoint 原始 `BA` 范数。这仍只是两端快照差，**不能**得知中间每一步有多少元素被舍入、改变后是否又返回原值，也不能独立证明梯度方向或质量。M2→100 的约 88% A 元素端点未变，只能描述端点；B 与 `BA` 已明显变化。

## AdamW 状态、master 权重与全模型 cast

PyTorch 2.8.0 [Adam/AdamW 源码](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/adam.py#L151-L174)用 `zeros_like(p)` 初始化 `exp_avg` 和 `exp_avg_sq`，仅 fused 的 `step` 标量使用 FP32；[融合内核调用](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/adam.py#L789-L797)直接接收参数和状态张量。源码[第 101–104 行](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/adam.py#L101-L104)仍把低精度参数的高精度副本列为待支持事项。这与真实 `optimizer.pt` 的三类字段及 dtype 完全一致。当前这条原生 `torch.optim.AdamW(fused=True)` 路径没有发现**持久的 FP32 master 参数**；不能把 FP32 `step` 误认为 FP32 master 权重。这里不推断 GPU 内核的瞬时内部累加细节。PyTorch [优化器状态加载源码](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/optimizer.py#L689-L717)还会将非 `step` 的浮点状态转到新参数 dtype，解释 BF16→FP32 续训后动量状态也转 FP32；这不会恢复前半程 BF16 已损失的精度。

是否有全模型 `.to(torch.bfloat16)` 抵消单独升 FP32？[本地 Qwen 训练脚本快照](<F:/AIC/tmp/source-audits/Qwen3-VL-current-20260922/qwen-vl-finetune/qwenvl/train/train_qwen.py:104>)是在建 PEFT 前加载 BF16 基座，建 PEFT 后没有显式全模型 dtype cast；[Transformers 5.14.1 Trainer 设备迁移](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/trainer.py#L4038-L4045)只 `.to(device)`。其 `bf16_full_eval` 的 dtype cast 位于非训练评估分支，[源码第 2435–2441 行](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/trainer.py#L2435-L2441)。当前保存的 checkpoint-200 全 FP32 也直接反驳“之后必然统一转回 BF16”的说法。将来若调整为 FP32 LoRA，仍须在优化器建立前和若干实际 step 后核查 runtime dtype，避免其他包装器或新改动改变口径。

## 本地 CPU 极小演示的边界

[演示脚本](./cpu_rounding_probe.py)在既有 PyTorch 2.8.0+cpu 上，对 8×8 的人工 A/B 使用固定梯度、`AdamW(fused=True)`、5e-6、200 步；分别模拟固定学习率与线性衰减，比较 B≈0.005 的已训练适配器和 B=0 的新适配器。[输出](./cpu-rounding-results.json)显示 BF16 在约 0.02 的间距为 1.22e-4，在约 0.005 的间距为 3.05e-5。B≈0.005 的案例中 FP32 A/B 均改变而 BF16 A/B/BA 端点全未变；B=0 案例中 BF16 A 未变、B 与 BA 却改变。它说明**机制与 LoRA A/B 非对称性**，并与 GPT Pro “不能只看 A”一致；其人工恒定梯度、极小形状和 CPU 融合实现都不等同云端真实训练。真实历史 checkpoint 已经比这个玩具例子更有说服力地显示 B/BA 变化。尤其实际线性调度在 step-100 checkpoint 记录学习率 2.5e-6、step-200 为 0，不能拿 Pro 的恒定学习率 200 步例子当作真实轨迹。

## 下一次真实训练前的最小核验（本轮未执行）

在计划中的下一次训练开始前，先固定要比较的数值策略，并只做最短运行时检查：打印主干、各 `q/k/v/o` 的 LoRA A/B 参数存储 dtype；在 `Trainer` 完成 checkpoint 装载和 optimizer 恢复之后再打印一次，并记录 `exp_avg`、`exp_avg_sq`、`step` dtype 与 `fused` 标志。保存目标 LoRA 在首个及第 3 个优化步前后的精确存储改变比例、梯度非零比例、A/B 范数变化和有效 `2BA` 变化，并确认 step 前后是同一套参数对象。不要只看 loss 或 `allclose`；当前代码的有限梯度回调无法回答这个问题。若需比较 FP32 LoRA 效益，再保持数据、初始 adapter、样本顺序、学习率日程等不变做短对照；**本审计没有授权或执行该训练，也不依据本次检查裁定成绩根因。**
