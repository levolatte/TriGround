# 共享前缀原型的配对对照

日期：2026-10-28。入口为 `tools/benchmark_shared_prefix.py`，模型实现为 `tools/qwen35_shared_prefix.py`。本文件说明可执行对照及其结论边界；不将 CPU 检查等同于真实 27B GPU 结果。

## 对照对象与分母

同一进程只加载一份原模型和 PEFT（参数高效微调）LoRA，兼容原 NF4（4 位量化）设置。每一对比较都从同一适配器权重、新建的原 Swift 优化器和相同随机种子开始。先执行原始逐答路径，再恢复 LoRA 权重、清空梯度与优化器状态，执行共享路径。两者保留三图、G8、原奖励、原 DAPO（动态采样策略优化）损失及原有效回答词元分母。

原始微批为 1，梯度累积为 32。两侧都调用 Swift 原 `compute_loss`，每个回答仍使用原组内有效词元数作为 `num_items_in_batch`。共享路径只将一次共享前向所得的各回答 logp（词元对数概率）送回原损失，八个损失相加后除以 32 并共同反传；没有再乘 8，也没有按组标准差重算优势。

`--groups 1` 是单题三图 G8 正确性探针，保留原四题更新里单题的 `/32` 贡献，然后执行一次参数更新；它不能代表原完整更新吞吐。`--groups 4` 使用原四个题组的 32 个固定回答，累积后只执行一次裁剪和优化器更新，才对应原固定回答完整更新。

## 固定输入与真实链路

默认复用 `benchmark_quant_grounding.py prepare` 的四个真实三图题组与 G8 历史回答，包含命中/未命中混合组、连续奖励差异组和同奖励对照组。也可传入 `--answers` JSONL，每行结构为：

```json
{"id":"正式数据中的id","responses":[{"text":"{\"bbox_2d\":[100,200,300,400]}","finish_reason":"stop"}]}
```

`responses` 必须有八项。脚本从正式数据选择同 id 的题目并重新计算原 A 奖励，不信任外部提供的奖励值。

固定回答只替换采样。Swift 仍执行原始消息预处理、分词、图像预处理、奖励、编码、拼批、优势与损失。编码阶段暂填零 old/ref，以免偷偷执行额外模型前向；完成编码后，两条路径分别计算并填入**自己的原生 old/ref**。它们不会回放或覆盖为基线 old/ref，也不会把候选前向误差额外注入基线旧概率的比率。

脚本在原始旧策略前向中直接捕获实际语言模型接收到的三维/四维 `position_ids`。共享输入使用原 `input_ids`、有效掩码、实际位置、像素与网格，左填充短回答以保持回答尾部对齐。同题只保留唯一三图张量。逐元素检查所有有效提示/回答词元、掩码、位置、完整像素张量和网格；任何不同立即报错。候选还独立重新编码并比对其原位置、掩码与图像。不同图像不会仅因占位词元相同而共享。

回答宽度为 R 时，两侧都保留 R+1 个尾部 logits（未归一化词表分数），用 `[-R-1:-1]` 预测最后 R 个回答词元。`P-1` 行预测第一个回答词元，其梯度必须存在；最后一行预测后续未评分词元，不进入该回答损失。有效掩码排除填充词元。

## GPU 运行

使用现有 `requirements-quant-train.txt` 对应 Linux CUDA 环境：PyTorch 2.12.1、Transformers 5.16.1、ms-swift 4.5.3、TRL 0.29.1、PEFT 0.20.0、bitsandbytes 0.50.2、FLA 0.5.2。生产对照默认 `--gdn-backend fla`；`torch_reference` 是明确的 CPU 小模型参考实现，不能用于宣称生产加速。

先在 CPU 主机准备（也可在 GPU 主机同样执行）：

```bash
python tools/benchmark_shared_prefix.py prepare \
  --formal /path/to/data/quant_grounding_20261026/schedule.jsonl \
  --data-root /path/to/data --output /path/to/fixed_bundle
```

`manifest.json` 的 dataset 和 `swift_dataset.jsonl` 的图像路径是当前主机绝对路径。转移主机后应在目标主机重新运行 prepare，或明确改为目标主机路径，不沿用 Windows 路径。复用默认 fixture 时需要仓库原始/评分 JSONL；`--raw` 和 `--scored` 可指定其位置。

先跑单题：

```bash
python tools/benchmark_shared_prefix.py run \
  --bundle /path/to/fixed_bundle --config /path/to/arm_a.json \
  --adapter /path/to/checkpoint-50 --model /path/to/base_model \
  --precision nf4 --groups 1 --output /path/to/shared_probe_g8
```

单题通过后运行原四题完整固定回答更新；`--repeats 3` 每次恢复初始权重和空优化器，逐次结果和中位数都保留：

```bash
python tools/benchmark_shared_prefix.py run \
  --bundle /path/to/fixed_bundle --config /path/to/arm_a.json \
  --adapter /path/to/checkpoint-50 --model /path/to/base_model \
  --precision nf4 --groups 4 --repeats 3 --output /path/to/shared_full32
```

默认设置 cuBLAS 工作区和 PyTorch/cuDNN 确定性。`--allow-nondeterminism` 只用于进一步定位算子或吞吐，不应与确定性资格结果混作一次结论。两侧都使用实际 BF16（16 位脑浮点）自动混合精度。由于独立加载钩子运行在 `SwiftMixin.train` 之前，脚本显式执行该方法原有的多模态前向钩子安装及梯度检查点准备，确保原路径全部语言层重计算生效；同时调用单卡 `accelerator.prepare_model`，保留原生输出转 FP32 的包装。共享函数绕过顶层包装，因此显式将 logits 转 FP32，再使用同一个 TRL `selective_log_softmax`。报告记录两侧概率计算前实际 logits 精度，任何阶段不是 FP32 都报错。

## 记录与裁决

`result.json` 记录每题每答的完整有效 current/old/ref logp、优势、回答词元、掩码、截断和 DAPO 分母；同时记录损失、裁剪前梯度范数、LoRA 梯度/一次更新的相对 L2、余弦、最大绝对差、符号翻转比例及有限性。首轮应分别看这些指标，不用单个布尔值遮蔽输入、前向与更新的差异。

梯度和参数更新仅以原始路径的一份 CPU 参数字典暂存于内存；候选逐参数拷到 CPU 即累计比较，不写每次上 GB 的 `gradients.pt`/`updates.pt`。模型投影、视觉编码和词表头的实际前向调用数/输入行数分阶段计数，包含激活重计算，另记录共享计划的总词元、紧凑词元、GDN（门控增量网络）分块边界与尾部重放词元。

计时使用 CUDA 同步后的墙钟秒数，分别包括固定 rollout/奖励/编码、共享输入组装、old、ref、当前策略完整前向+反向及裁剪+优化器。`fixed_update_measured_seconds` 为这些真实阶段之和，输入逐元素比较和 CPU 梯度/更新诊断另列并排除；模型加载和在线随机采样也不在其中。显存记录本次峰值分配/保留量、设备总显存与起始非 PyTorch 占用。只有一次的结果仍包含冷启动因素，不能据此宣称稳定吞吐。

该工具同时回答模型数值与固定回答 GRPO 更新的分阶段成本，**不包含在线生成，不能单独证明完整训练端到端达到两倍提速**。真实总收益仍须后续在集成训练入口记录包含生成的更新墙钟；严格保留原算法与新算法结果的区别。若原始重复运行本身波动，应先复测自身基线；若当前 logp 从首答词元即偏离，应先查位置、分支状态和低精度算子，不通过回放基线 old/ref 掩盖或放大问题。

## 首次 GPU 执行与修复

首次 CUDA 小模型 FLA 检查使用 P3500/R23/G8，最大有效 logp 绝对差为 0.004075，梯度相对 L2 为 0.001812。这只覆盖小模型，不代表 27B NF4 对照已通过。

随后真实 27B 基准完成模型加载，在原始路径尚未执行前因 `IndexError` 中止：`shared_logps` 的默认参数曾写成 `packed=packed_groups[group_index]`，定义函数时便索引了尚为空的题组列表。已改为绑定列表本身，在共享函数实际调用时才读取该题组，原失败日志保留。

修复后的 `retry-1/fixed_g1/result.json` 完成单题 176 个有效回答词元对照，但未通过数值门槛：最大 logp 绝对差 0.120575、梯度相对 L2 0.168044/余弦 0.985833、更新相对 L2 0.387969。两侧完整输入、每答 176 的 DAPO 分母和优势均一致；各自 old/ref/current 逐位相同，重要性比率均为 1。近零且相同的标量损失源于组内优势抵消与零 KL，不能证明梯度一致。约 5.37% 梯度元素变号，首步 Adam 更新对这种变化敏感，不应仅因梯度范数接近便放行。

该次固定回答测得 101.63→68.88 秒，仍不包含生成；共享 old 首调用 22.06 秒、相同参考调用 2.42 秒，存在未隔离的首调用额外开销，不能将 1.476 倍视作稳定速度或速度上限。保持失败结果，先定位前向分歧，不调整接受阈值。

## 无更新前向诊断

`diagnose` 沿用同一模型加载和既有 G8 固定回答，只执行一题无梯度前向，不创建优化器、不复制整份 LoRA 初始权重、不执行训练或在线生成：

```bash
python tools/benchmark_shared_prefix.py diagnose \
  --bundle /path/to/fixed_bundle --config /path/to/arm_a.json \
  --adapter /path/to/checkpoint-50 --model /path/to/base_model \
  --precision nf4 --groups 1 --output /path/to/forward_diagnostics
```

结果写入 `diagnostic.json`，路径为 `forward_diagnostics.probes`：

- `native_serial_1/2`：原生逐答同形状复跑，量化本次前向噪声。
- `shared_1/2`：原共享路径同形状复跑，第二次与第一次直接比较。
- `native_batch2`：两答原生语言模型与词表头的执行路径对照，复用首次逐答捕获的实际语言输入嵌入，避免视觉批大小变化成为另一原因。只保留两答嵌入，不构造 G8 整段大批。原 mb1 可能还携带可变长度序列元数据，因此不能将此项全部差异归因于矩阵批形状；实际 `cu_seq_lens_q/k`、`max_length_q/k` 及参数名另存于 `native_call_metadata`。
- `shared_native_gqa`：只将共享前缀注意力的 GQA（分组查询注意力）调度对齐原生 HF 规则，其余保留原共享实现；这是无梯度诊断项，不自动成为训练新默认。

默认只追踪第一个回答对应的少量实际词元行：提示开头/中间、GDN 分块边界、卷积尾部、提示最后一行，以及完整答案。按相同未填充词元位置比较真实视觉注入后的嵌入、三维位置、旋转位置编码、各层输入/输出、最后归一化、词表头输入和原始输出；只即时复制这些行到 CPU，保留逐事件统计，不落盘完整激活。

`shared_1.trace_comparison.first_above_native_repeat_noise_event` 给出采样行中第一个超过原生复跑差异的事件。它不证明所有更早的未采样行完全一致。若为 `layers.N.*`，可只补一轮指定层诊断：

```bash
python tools/benchmark_shared_prefix.py diagnose \
  --bundle /path/to/fixed_bundle --config /path/to/arm_a.json \
  --adapter /path/to/checkpoint-50 --model /path/to/base_model \
  --precision nf4 --trace-layer N --output /path/to/forward_layer_N
```

指定层增加可配对的归一化、投影、混合器与多层感知机事件；不能从原生模块钩子取得的中间量明确列为未配对。还将该层**完全相同的共享归一化输入与旋转编码**交给原生混合器，透传每答实际捕获的 `cu_seq_lens/max_length` 元数据，逐答按原 mb1 序列形状重放，比较全部紧凑行，区分前层已经传播来的误差和当前混合器自身差异。诊断计时包含采样复制，不用于加速比；诊断完成也不自动代表通过任何等价门槛。
