# Qwen3.5 同题共享前缀：模型层实装与接手说明

日期：2026-10-28。实现位于 `tools/qwen35_shared_prefix.py`，专用 CPU 检查位于 `tests/test_qwen35_shared_prefix.py`。这是研究与实现说明，不是开发日志。

## 实现边界

新增一个显式调用的函数入口，不改 Transformers 类、不替换 NF4 或 LoRA 模块、不触碰推理缓存。训练器在开关关闭时继续调用原模型；开关打开后，针对**同一题的完整 G8 组**调用本入口。组身份由上层保证，不能根据相同图像占位词元猜测两个问题共享图像。

语言路径中每层的归一化、投影、门控、残差和 MLP 都作用于 `[1,C,H]` 紧凑词元行，其中 `C=P+sum(R_i)`。这些行包含唯一提示和八个回答。共享语言提示没有 `detach()`；唯一停止梯度的是已冻结视觉塔及其最终合并特征。所有投影仍调用原模块，因此保留 HF 参数、PEFT 适配器和 bitsandbytes NF4 模块的正常调用语义。

针对当前单卡、纯稠密 `model_type=qwen3_5` 的 `Qwen3_5ForConditionalGeneration`。不支持视频、MoE（混合专家）、序列并行、参数卸载/多设备调度或生成缓存。不能把它直接当作生成 `forward` 替代品。上述路径不是静默回退：不满足接口假设就需要在上层关闭开关或明确报错。

## 公共接口

```python
plan = build_shared_prefix_plan(
    input_ids=ids,             # LongTensor [G,L]
    attention_mask=mask,      # [G,L]，或 None 表示全部有效
    position_ids=positions,   # [3,G,L] / [4,G,L]，纯文本也可 [G,L]
    prefix_length=P,          # 有效提示长度，不含左侧 padding
)

outputs = shared_prefix_forward(
    model,                    # 原 HF 模型或 PEFT 包装模型
    plan=plan,
    input_ids=ids,
    attention_mask=mask,
    position_ids=positions,
    pixel_values=unique_pixels,
    image_grid_thw=unique_grid,
    logits_to_keep=R + 1,
    checkpoint_layers=True,
    gdn_backend="fla",
)
# outputs.logits: [G,R+1,V]；回答的预测行是 logits[:,:-1]
```

`input_ids` 和位置编号在同一设备上。有效词元区域必须连续，支持左填充、右填充以及两端都有填充；每个成员必须包含至少一个回答词元。构建计划时明确比较 G 个提示的词元与全部原位置坐标，相同组中的差异会报错。不会自动延长共享前缀到偶然相同的回答开头。

像素和网格只提供**该题唯一三图**，顺序与原问题一致。它们不带 G 份副本。上层若缓存了最终视觉特征，可以改传 `frozen_image_features=[N_image_tokens,H]`，并省去 `pixel_values`。这是视觉塔和合并层之后、原占位顺序下的最终特征。模型层不维护跨组或跨更新缓存。由 `inference_mode` 创建的特征在进入训练图前复制成普通 Tensor（张量）；语言嵌入和后续层继续保留梯度。

便捷入口 `shared_prefix_answer_logps(model, completion_ids=[G,R], **forward_kwargs)` 自动选择 `R+1` 行并返回 `[G,R]` 对数概率。训练器也可以继续使用原有选定词元对数概率函数，以保留其内存与精度策略。调用者必须继续使用原 `completion_mask` 屏蔽无效回答位置。**填充行的输出没有与原生路径逐值对齐的承诺；所有有效预测行需要对齐。**

`SharedPrefixPlan` 公开 `compact_length`、`total_tokens`、`prefix_length`、`suffix_lengths`、`batch_size`、`sequence_length`、`gdn_boundary`、`gdn_replay_tokens`。其中 `total_tokens` 是去除填充但未共享的总词元数，`gdn_replay_tokens=G*(P mod 64)` 是相对紧凑词元数新增的 GDN 尾部重放量。

## 三类序列混合器

### 全注意力

前缀调用一次因果 SDPA（缩放点积注意力）。八个短回答作为一批，查询只包含回答，K/V（键/值）由共享前缀投影和各自回答投影拼接而成。布尔掩码显式采用相对于前缀偏移的因果关系，不能直接对非方阵调用默认 `is_causal=True`。

每个注意力层两次 SDPA。K/V 的拼接和分支视图仍有显存及拷贝成本，但没有重复运行前缀投影、归一化或 MLP。前缀 K/V 上来自各回答的梯度通过 `expand`、拼接和索引反传累加。此实现不要求安装外部 FlashAttention；具体 CUDA SDPA 内核和速度由实际运行确认。

### 因果卷积

共享前缀做一次原始深度卷积，各回答前仅拼接原始卷积输入的末 `kernel_size-1` 行。当前核宽 4，因此重放三行。八个回答补齐后一次批量卷积，丢弃上下文三行的输出和右侧补齐输出。所有历史输入都仍处于计算图中。

### GDN（门控增量网络）

生产入口直接导入 `fla.ops.gated_delta_rule.chunk_gated_delta_rule`，不经过 HF 的自动回退分发器。第一轮对齐到前缀最后的 64 词元块边界 `B=floor(P/64)*64`，获取该边界状态。第二轮使用 FLA 可变长度输入，同时计算：

1. 前缀剩余尾部，产出其唯一隐藏行；
2. 每条回答前重放同一段前缀尾部，再计算该回答。

第二轮的初始状态通过 `index_select` 重复引用第一轮状态。反向过程自动把所有分支的状态梯度求和，并传回第一轮，未使用原地更新或 `detach()`。P 为 64 的倍数时不需要尾部重放；P 小于 64 的 CPU 检查路径也明确支持，初始状态为零。

保留块边界是为了保持与原生长序列相同的分块划分；它不能消除 NF4/BF16 和不同矩阵形状造成的全部舍入差异。真实硬件上仍须对照原生逐答路径比较回答概率、LoRA 梯度和一次更新。

## 位置与第一回答词元

位置编号必须来自原生多图编码，显式传入。四轴输入的第一轴与 HF `TextModel` 一样作为文本位置轴剥离，后三轴进入原 HF `rotary_emb`；三轴输入保持原数值。不会重新用一维 `arange` 替代多图 mRoPE（三维多模态旋转位置编码）。

输出恢复时最后提示隐藏行按每条回答索引复用，随后调用原 `lm_head`。八个回答的第一词元可有不同标签，监督梯度累加到同一提示行。它没有用“从第一回答输入行开始”代替预测首词元所需的最后提示行。

## 激活重计算与集成注意点

使用逐层 `torch.utils.checkpoint(..., use_reentrant=False)`，避免依赖推理缓存和冻结输入是否有 `requires_grad`。默认开启，可以显式关闭。函数直接调用原层内部模块，因此 `mlp.gate_proj`、全注意力 `q_proj` 和 GDN `in_proj_qkv` 的钩子仍可以计数；`DecoderLayer.forward` 本身没有被调用。

PEFT 的 `get_base_model()` 只解除外部包装，原适配器模块仍挂在各投影上；`disable_adapter` 上下文也仍由这些原模块执行。没有修改 LoRA target_modules（目标模块）、秩或损失定义。dropout（随机失活）必须为零，否则各回答原本独立的随机掩码无法由同一共享前缀精确复现。

Swift 需要继续保持题组 G8、原回答权重、有效词元屏蔽、参考模型上下文与原总有效批次。把八答一次前向和八答一次损失误当成只训练一答，会改变梯度缩放；这是训练器侧责任，本模型入口只返回每答每词元分数。

## CPU 检查

使用现有环境：

```powershell
& F:\AIC\.work\grounding_aux_round1\cpu_preencode_env\Scripts\python.exe `
  -m pytest tests/test_qwen35_shared_prefix.py -q
```

该环境为 PyTorch `2.8.0+cpu`、Transformers `5.16.1`、PEFT `0.20.0`。没有安装或升级依赖，没有使用 SSH，也没有运行 CUDA 任务。

CPU 检查包括：

- 显式计划、错误组位置拒绝、填充处理，以及最后提示行对不同第一回答标签的梯度相加。
- 前缀长度 5、64、67 三种情况的卷积与 GDN 独立逐分支输出和输入/权重梯度。
- 批量 SDPA 与独立完整分支的输出以及共享 K/V 梯度。
- 真实 HF 小型两层混合 Qwen3.5：八答、三处最终图像特征、非平凡三维位置；对比原生逐答输出、所有可训练参数梯度和一次 SGD 更新。
- 真实 PEFT LoRA、非零适配器 B、逐层非重入重计算，以及变长回答和两端填充的同类对照。
- 真实冻结小视觉塔对唯一三图只编码一次；缓存推理张量进入语言训练图后仍可反传。

最终本地运行结果为 **11 passed, 1 skipped，17.16 秒**；跳过的是需要显式开关和 CUDA 的长前缀探针。

CPU 参考后端名称是 `torch_reference`，采用独立递归公式，明确只作慢速检查；生产默认 `fla`。CPU 结果**没有证明** 27B NF4、FLA CUDA 最终状态梯度或实际端到端加速。生产依赖沿用项目 `requirements-quant-train.txt`，其中 FLA 为 `0.5.2`。CUDA 门禁与基准由独立基准入口完成，不能以本检查替代。

### 可选 CUDA 小模型探针

在已有 CUDA、FLA、HF 和 PEFT 环境中显式运行：

```bash
AIC_RUN_CUDA_SHARED_PREFIX=1 python -m pytest -s -q \
  tests/test_qwen35_shared_prefix.py -k cuda_fla_long
```

探针对 P=3500、R=23、G=8 使用两层 BF16（脑浮点16）混合 Qwen3.5 和真实 LoRA；原生对照也显式指定实际安装的 FLA，避免自动回退污染结论。除模型概率与 LoRA 梯度外，先独立检查 FLA 最终状态跨可变长度分支的梯度非零，确认当前库支持需要的状态反向接口。共享路径使用逐层激活重计算。

它打印 JSON，包含前缀边界 3456、额外重放 352 词元、回答对数概率最大绝对偏差、LoRA 全局梯度相对 L2 偏差和状态梯度。BF16 冒烟阈值暂设对数概率最大绝对偏差 `<0.02`、全局梯度相对 L2 偏差 `<0.05`；这些只是帮助尽早发现接口/梯度断裂的明确阈值，**不是 27B NF4 的通过标准**。本地没有执行此 CUDA 探针，也没有制造随机小模型 NF4 结果来冒充真实 27B 量化组合。

## 上游归属

GDN 两轮状态分叉与紧凑索引的设计改编自 VeOmni 的 `shared_prefix.py`，其下载证据保存在同目录 `evidence/veomni_shared_prefix.py`。投影、门控、归一化、卷积和 GDN 数学公式对照 Transformers `5.16.1` 的 Qwen3.5 源码。两者原源码均为 Apache License 2.0；本新增模型工具保留其版权与许可声明，并注明本地适配范围，完整许可另存于 [Apache-2.0.txt](evidence/Apache-2.0.txt)。该许可文本来自本地安装的 Transformers `5.16.1` 分发包。仓库根目录的自有代码许可不替换这些上游许可。

接手顺序：先用真实冻结答案完成 CUDA 输出、梯度、更新门禁；再测结构计数和速度；最后才允许在另一个训练命令中显式打开开关。当前 A/B 训练进程无需改动。

## 27B NF4 首次门禁失败后的定位接口

真实结果位于 `evidence/run-20261028/retry-1/fixed_g1/result.json`。本组实际 P=3501、R=22、G=8，紧凑长度 3677，GDN 边界 3456、每分支重放 45 行。共享前向的回答对数概率最大绝对差为 `0.12057495`；裁剪前梯度相对 L2 差为 `0.16804384`、余弦相似度 `0.98583328`；一次更新相对 L2 差为 `0.38796856`。这些数值没有通过既定门禁，不能因固定答案阶段较快就启用正式训练。

证据已逐项确认词元、有效掩码、三图像素和真实 mRoPE 相等，概率计算前均为 FP32。同一路径内 old/current/reference（旧策略/当前策略/参考策略）在 176 个有效词元上相同，比例均为 1，优势相等。因而当前重点是前向算子数值路径，不能把差异解释为奖励归一化、重要性比例裁剪或参考适配器未恢复。

### 尚待实际定位的候选

1. NF4 投影和 BF16 LoRA/MLP 的矩阵行数由原生每答 3523 改为紧凑 3677，以及 `lm_head` 由每答 23 行改为 `[8,23,H]`。即使参数与输入每行相同，矩阵形状和内核选择仍可能产生舍入差异；64 层会放大局部偏差。现有结果尚不能把它确认为主因。
2. HF 5.16.1 原生 SDPA 对无掩码、`key_head_dim == value_head_dim <= 256` 的输入启用 `enable_gqa=True`。当前共享前缀显式展开 K/V 后调用 SDPA，回答则使用短查询与布尔掩码。这是已查明的执行路径差异，但不是已证明的数值主因。
3. 因果卷积的前缀/短回答批量形状和 FLA 的导出状态、可变长度初始状态路径，也需要在完全相同输入下单独比较。小型两层 BF16/FLA 探针通过不能证明 64 层 NF4 组合对齐。

### 逐层追踪

`shared_prefix_forward(..., trace=callback)` 新增可选回调，默认关闭。回调签名为 `callback(name, tensor, metadata)`。模型不会擅自复制、搬到 CPU 或保存整份激活；调用者应只抽取同一分支的少量原始词元行，先查层边界，再针对首个分歧层检查投影/混合器。

- `compact.input_embeds`、`compact.position_ids`、`rope.cos/sin`。
- `layers.N.input/input_layernorm/mixer/after_mixer/post_attention_layernorm/mlp/output`。
- GDN 的 `layers.N.linear_attn.in_proj_qkv/conv/in_proj_z/in_proj_b/in_proj_a/g/beta/core/norm/out_proj`。
- 全注意力的 `layers.N.self_attn.q_proj/q_norm/k_proj/k_norm/v_proj/rotary_query/rotary_key/core/o_proj`。
- `final.norm/selected/logits`。

元数据中的 `seq_dim` 指明词元轴。除末尾已展开的 `final.selected/logits` 外，布局均为 `compact`，需要使用计划的展开索引映射原词元位置；末尾两项布局为 `expanded`，同时提供 `logits_to_keep=K` 和 `sequence_offset=L-K`。带梯度且使用检查点时拒绝启用 trace，避免把重计算事件误计成新层。

`native_sequence_mixer_forward(layer, plan=plan, hidden_states=该层紧凑input_layernorm输出, position_embeddings=(cos,sin))` 提供纯诊断局部重放。在 `torch.no_grad()` 中逐答调用原生 mb1 混合器，返回紧凑输出，不创建完整 G8 长序列批量。把它和 trace 中该层 `mixer` 比较，可以区分本层产生的误差与上游已累积的误差。诊断对象是原生完整混合器，包含其投影形状；进一步的投影与核级隔离由已定位层的输入/输出决定。

调用者可通过 `native_sequence_kwargs` 按回答传入原生语言前向实际携带的 `cu_seq_lens_q/k` 与 `max_length_q/k`；局部重放逐答透传这些参数，保留原生变长序列调用。双样本语言对照单独记录其参数，不把批形状与变长序列路径变化统称为纯矩阵形状误差。

### 单项 GQA 候选

`diagnostic_attention="native_gqa"` 仅在 `no_grad` 中可用：共享前缀调用 HF 的 `use_gqa_in_sdpa(None,key,value)`，据此传入原生 GQA 标志；短回答仍保留正确的分支掩码。默认值始终是 `"current"`，保留首次失败运行中的完整 K/V 展开次序、切片步长及调用布局。

若追踪确认第一处显著分歧发生在全注意力核心，再用完全相同 Q/K/V 比较原路径与此候选。若候选改善仍需继续排除短回答 SDPA 和上游投影差异，不能仅凭无掩码前缀的分发对齐就宣布修复。

本轮新增接口的 CPU 检查为 **12 passed，1 skipped，19.56 秒**：追踪开关不改变默认输出，元数据和两种混合器的同输入原生重放对齐，GQA 候选在 CPU 上保持语义一致。没有运行新的 CUDA、SSH、完整梯度基准，也没有修改正式训练或放宽数值门槛。
