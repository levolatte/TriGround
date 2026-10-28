# Qwen3.6-27B 多图 NF4 QLoRA GRPO：框架、内核与训练前缀共享调查

资料核查日期：2026-10-28。对象为 `model_type=qwen3_5` 的稠密多模态混合模型；不能把纯文本 Qwen3、Qwen3-VL、Qwen3.5-MoE 的支持自动外推到它。当前目标环境：改装 48GB RTX 4090，ms-swift 4.5.3、Transformers 5.16.1、PyTorch 2.12.1+cu130、bitsandbytes 0.50.2、FLA 0.5.2。本文仅联网读文档/源码，没有安装依赖、加载模型或运行 GPU。

已合并主线程取得的[当前A组运行时审计](evidence/a-runtime-audit.json)：FLA 的 chunk/recurrent 均已解析到真实FLA实现；`causal_conv1d_fn` 使用Transformers原生实现；`runtime_optimization=false`；`logits_to_keep_supported=true`；800个`Linear4bit`；视觉线性层保持BF16且冻结；64个语言层逐层启用梯度重计算。参数存储约20.09GB，其中可训练参数约0.934GB，尚不包括bnb的量化状态与优化器/激活。**当前不是“整个GDN都没装加速包”的情况。**

## 1. 直接判断

**想大幅削减当前训练的主要计算，应优先研究“可微的组内提示前缀共享”，而不是先迁移整套推理框架。** 同题 G=8，每答约 P=3500 个提示 token、R=20 个输出 token，逐答训练约处理 `8×3520=28160` 个 token 行；共享提示后的逻辑行数约 `3500+8×20=3660`，比例为 **7.69**。这是投影、归一化、MLP 等重复行计算的理想削减机会，**不是端到端加速预测**；混合注意力的分叉重放、内核利用率、视觉计算和生成阶段都会影响实际时间。

当前 `microbatch=1`、梯度累积 32，把同题答案放在 8 次独立前后向中。仅开启普通序列打包不会消除这些提示副本；仅安装 FlashAttention、Liger 或 vLLM 也不会自动共享训练前缀。需要把一题的 8 个答案组织成一个可微的紧凑分叉图，或使用跨微批的前缀梯度累加调度。

**已有强相关实现，但没有找到“Qwen3.6-27B + 多图 + NF4 + LoRA + GRPO + 单48GB”已经完整验证的即插即用方案。** 最有价值的实现参考是 VeOmni 的稠密 Qwen3.5 共享前缀模块；它已经处理 Gated DeltaNet、因果卷积和全注意力三种状态，但官方当前明确限制为纯文本。HARTS 提供更一般的混合模型树训练方案，仍需适配当前模型、量化和多图输入。[VeOmni 指南](https://veomni.readthedocs.io/en/latest/examples/qwen3_5.html#shared-prefix-training)、[VeOmni 实现](https://github.com/ByteDance-Seed/VeOmni/blob/main/veomni/models/transformers/qwen3_5/shared_prefix.py)、[HARTS 论文](https://arxiv.org/abs/2608.28158)

## 2. 训练前缀共享：最相关的官方实现和真实边界

| 路线 | 确认存在的能力 | 当前任务的缺口 | 工程判断 |
|---|---|---|---|
| **VeOmni shared-prefix** | 稠密 Qwen3.5；投影/MLP 只跑唯一 token；全注意力复用 K/V；GDN 可微分叉；因果卷积重放提示末尾；支持梯度重计算 | 官方限定纯文本；有 image/video 时普通路径运行；未确认 NF4 加载；需要同组答案进入同一微批 | **优先作为移植参考**，不建议为这个功能先整体迁移 |
| **HARTS / AReaL PR #1765** | Qwen3.5 GDN 的树执行、可微边界状态、有界重放、紧凑输出恢复、梯度重计算 | PR 仍 Open；现成集成主要是 35B-A3B-Base + Megatron Bridge，非当前稠密 NF4 VLM | 技术依据强、整体迁移很重；平坦 G=8 用不到其全部树调度 |
| **OpenPipe ART 的 GDN 前缀树** | Qwen3.5 handler 安装 `install_prefix_tree_gdn_hooks`，支持稠密/混合专家语言模型和 LoRA | handler 显式只构建 `provide_language_model`、注册 TextOnlyBridge；没有本项目多图NF4单卡证据 | 又一个GDN训练参考，不能因ART总体能预处理图像就认定该路径支持视觉 |
| **PrefixGrouper** | 作者公开 GRPO 共享前缀工具，有 Qwen2.5-VL 示例；全可微 | 核心接口操作 Q/K/V attention，没有证据覆盖 Qwen3.5 GDN 和卷积状态 | 可参考多图位置和标签恢复；不能只替换 full-attention 后就认为正确 |
| **Flash Preference** | 作者公开共享前后向上下文；测试含 Qwen2-VL/Qwen2.5-VL | README 测试列表无 Qwen3.5；无 GDN 支持证据 | 原理和注意力实现参考，非当前可直接使用的兼容包 |
| **TRL AsyncGRPO tree packing** | 当前实验训练器已实现前缀树紧凑布局和 attention 分段 | 官方明确不支持图像/3D M-RoPE；hybrid recurrent state 的样本边界未正确重置；NVIDIA 默认 FA3；生成服务需另一 GPU | **当前排除直接迁移** |
| **Schedule-Level Shared-Prefix Reuse** | 作者论文实现“提示前向一次 → 各答案微批反向并累加 gK/gV → 提示反向一次” | 论文验证 Llama3/Qwen3/Qwen3-MoE，未证明 hybrid GDN、多图、NF4；需增加 recurrent/conv state 的梯度接口 | 当紧凑单组仍放不下时的后续路线，首版不必先引入复杂调度 |

来源：[VeOmni](https://veomni.readthedocs.io/en/latest/examples/qwen3_5.html#shared-prefix-training)、[HARTS PR](https://github.com/areal-project/AReaL/pull/1765)、[PrefixGrouper 作者仓库](https://github.com/CASIA-IVA-Lab/PrefixGrouper)、[Flash Preference 作者仓库](https://github.com/li-plus/flash-preference)、[TRL AsyncGRPO](https://huggingface.co/docs/trl/async_grpo_trainer)、[跨微批复用作者论文](https://arxiv.org/abs/2606.01143)

ART 支持边界另见[当前Qwen3.5 handler](https://github.com/OpenPipe/ART/blob/main/src/art/megatron/model_support/handlers/qwen3_5.py#L245)：`provide_language_model`、`_art_text_only_language_model=True` 和 `_ArtQwen35DenseTextOnlyBridge` 都是明确的纯文本路径。已只读检查官方发布的0.5.23源码包与当前main，没有安装或执行其训练代码。

### 2.1 VeOmni 为什么特别值得读

已直接阅读其当前 `shared_prefix.py`（367 行）及生成后的 GPU 模型实现。模块不是“脱离梯度的推理缓存”：

- `compact()` 用 `index_select` 取得唯一行；`expand()` 的反向自然把各答案的梯度加回共享行。
- 全注意力用一轮变长 attention 调用，每个答案的 query 只看到自己的提示与自身历史；K/V 按各分支路径收集。
- 因果卷积为每个答案附上提示最后 `kernel_size-1` 行，丢弃其额外输出，保持卷积历史。
- GDN 做两轮变长调用：提示算到最后一个 chunk（分块）边界，输出状态；每个答案从该状态开始，重放少量提示尾部。边界与原始计算一致，避免仅按任意 token 分叉造成分块数值路径变化。

[核心模块](https://github.com/ByteDance-Seed/VeOmni/blob/main/veomni/models/transformers/qwen3_5/shared_prefix.py)、[GPU 模型接入](https://github.com/ByteDance-Seed/VeOmni/blob/main/veomni/models/transformers/qwen3_5/generated/patched_modeling_qwen3_5_gpu.py)、[CPU 输出/梯度等价性测试](https://github.com/ByteDance-Seed/VeOmni/blob/main/tests/models/test_qwen3_5_shared_prefix.py)

**多图限制的根因很具体。** GPU 模型创建 plan 的条件要求 `pixel_values is None` 且 `pixel_values_videos is None`；注释指出，不同图片占位 token 的 ID 相同，因此“token 序列相同”不等于“视觉提示相同”。不能删除这两个判断就算完成适配。若移植到本项目，应只在同一题、同一有序图片列表、同一处理后特征和相同三维位置编号之间共享；先从 GRPO 已知同组成员构建分组，避免用占位 token 猜图像是否相同。[GPU 接入源码](https://github.com/ByteDance-Seed/VeOmni/blob/main/veomni/models/transformers/qwen3_5/generated/patched_modeling_qwen3_5_gpu.py#L2702)

该方案不要求把 8 份长提示全部保留成激活：紧凑输入约等于一份 3500 token 提示加八份短答案，因而**有单48GB尝试价值**。但 NF4/PEFT（参数高效微调）与新模块、FLA 的 `initial_state` 梯度、图像三维位置编号、答案起始预测行、梯度检查点都尚未在当前组合验证；不能据此保证放得下或保证 7.69 倍。

主线程对原B组1888个答案进行CPU统计，逐组完整文本重复仅6个（约0.32%）。因此完整答案去重几乎不能消除当前主体计算；**应共享所有答案都拥有的提示，而不是押注答案恰好相同**。

### 2.2 HARTS 给出的关键经验

HARTS 专门解决 hybrid 模型不能仅靠树 attention mask（掩码）共享的问题：全注意力需要祖先 K/V，而 GDN 要传递祖先 recurrent state（递归状态），短卷积也要传递正确上下文；这些接口必须可微。论文把前缀重放限制在必要的块边界恢复内，并保留原 RL 输出接口与重计算语义。[作者论文](https://arxiv.org/abs/2608.28158)

当前 AReaL PR #1765（2026-09-29 起，核查时仍 Open）记录了一个很有教育意义的例子：8 张 L20X、Qwen3.5-35B-A3B-Base 的三步短跑，训练阶段从 7.81 秒降到 4.44 秒，但端到端反而由 883.18 秒到 899.21 秒，权重交换占绝大部分。**这是阶段加速不能当总加速的直接反例**，而且不是本任务硬件、模型和数据的证据。[PR 记录](https://github.com/areal-project/AReaL/pull/1765)

### 2.3 哪些“缓存”改变了训练定义

在语言 LoRA 仍训练时，先用 `no_grad()` 生成提示的 KV/GDN 状态，再用这些已分离状态训练答案，会删去答案损失通过提示状态返回到提示侧 LoRA 运算的梯度。它是 stop-gradient（停止梯度）近似，不能写成等价 GRPO 加速。

冻结视觉塔与投影层后的视觉特征缓存不同：被缓存模块参数不更新、输入预处理固定且无随机性时，其输出可作为固定输入复用；语言层内部状态仍需可微。缓存图像特征本身不会消除后续 3500 token 在语言网络中的八次重复计算。

跨优化器更新保留语言 KV/GDN 状态也不等价：LoRA 权重改变后，旧提示状态对应旧策略。等价的组内共享应在同一次策略前后向/优化器更新范围内使用同一份当前权重。

### 2.4 本项目多图适配的最小边界

官方 Qwen3.6-27B 配置显示64层中48个GDN、16个全注意力层，视觉 `deepstack_visual_indexes=[]`。Transformers 5.16.1 先运行视觉塔，把合并后的图像特征 `masked_scatter` 到图像占位位置，再进入语言模型。多图不是在每层重新注入视觉分支，因此可以把等价共享的首个适配点设在“已完成视觉嵌入的语言输入”上。[官方模型配置](https://huggingface.co/Qwen/Qwen3.6-27B/blob/main/config.json)、[HF入口实现](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L1507)

建议实现边界如下，属于待验证设计：

1. 从真实processor输入取得同一题的有序图片、`image_grid_thw`、`mm_token_type_ids`，按原实现构造三维 `position_ids`；只用显式GRPO组身份共享，第一版不跨题合并。
2. 若视觉塔及其合并/投影参数全部冻结，计算一次视觉特征；把它注入该题唯一提示的语言嵌入。若词嵌入仍可训练，则词嵌入留在当前计算图，不能跨更新缓存整份嵌入。
3. 语言图同时处理唯一提示和8条答案，按原始每条序列的三维位置编号压紧；传入显式位置编号，不依赖上一轮生成留下的可变 `rope_deltas`。
4. 全注意力分支隔离、因果卷积提示尾重放、GDN块边界状态分叉都使用同一个组内plan。任何一项只按DFS顺序直接串接，都会把别的答案历史混入当前答案。
5. 恢复每条答案的独立预测行、old/ref logps、completion mask和优势。特别是提示最后一个隐藏态要分别预测8个答案的首token；不能把共享行只记一个target（标签）。
6. 仍保留4题×8答的原有效批次和原每答归一化；一次题组紧凑计算不应误变成只训练4个等权样本。若中途改变梯度累积次数，要证明总缩放与原32答等价。

冻结模块缓存与可微语言共享可以同时实施，但应分开计时，避免把省掉视觉重复前向的收益全部归给语言共享。

## 3. vLLM：生成加速有依据，训练主耗时不会自动消失

### 3.1 模型与 LoRA 支持

当前官方支持表明确列出 `Qwen3_5ForConditionalGeneration`，包括文本、图片和视频及 LoRA 能力；源码 `qwen3_5.py` 实现语言骨干 `SupportsLoRA`，并为 GDN 融合投影建立 `in_proj_qkvz`/`in_proj_ba` 映射。因本项目冻结视觉部分，所需重点是语言 LoRA，不需要先引入视觉塔 LoRA 支持。[模型表](https://docs.vllm.ai/en/latest/models/supported_models/)、[模型源码](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_5.py)

必须核对当前适配器的**实际 target_modules（目标模块）**。vLLM PR #47640 于 2026-08-19 合并，修复只覆盖部分 GDN 融合投影（例如有 `in_proj_qkv`、无 `in_proj_z`）时的加载崩溃；这说明“模型支持 LoRA”不足以证明任意目标模块组合都正常。较旧 vLLM 不能直接套用当前主线结论。[修复 PR](https://github.com/vllm-project/vllm/pull/47640)

### 3.2 NF4 是组合门槛

截至核查时，官方文档的 bitsandbytes 支持已迁到官方的独立 `vllm-bnb-plugin`，支持在线量化与预量化模型。它证明 vLLM 有 bnb 加载路径，**没有证明 qwen3_5 多图、GDN 融合映射、NF4、训练中动态 LoRA 更新这四者同时正常**。2024/2025 年的旧 QLoRA 限制和纯 Llama 的演示也不应当当作当前全模型结论。[当前 bnb 文档](https://docs.vllm.ai/en/latest/features/quantization/bnb/)、[官方插件](https://github.com/vllm-project/vllm-bnb-plugin)

量化基础不同会带来 rollout policy（采样策略）与训练策略不一致：训练 NF4，生成 AWQ/GPTQ/FP8，即使适配器相同，也不是数值相同的策略。不能把换量化方法当成完全保持训练定义的实现优化；需记录实际生成 log-probability（对数概率）并评估重要性采样校正是否适用。

### 3.3 同卡与显存

27B 权重仅 BF16 就约 `27e9×2=54GB`，未计视觉塔、LoRA、缓存、激活与运行时空间，已经超过这张48GB卡的容量。因此“NF4 训练器旁边再常驻一个完整 BF16 vLLM”不是可行默认方案。

同卡可以研究**训练与生成交替占用**：vLLM 睡眠、训练模型/优化器卸载到 CPU、唤醒生成、同步适配器。ms-swift 已有 `--use_vllm true --vllm_mode colocate`，并提供 `sleep_level`、`offload_model`、`offload_optimizer`；只有 LoRA 的参数同步也有开关。这是现有框架内比重写训练器低成本的路线，但需要兼容的量化生成模型且转移耗时可能明显。[SWIFT 官方 GRPO](https://swift.readthedocs.io/en/latest/Instruction/GRPO/GetStarted/GRPO.html)

vLLM sleep level 1 将权重移到 CPU 并丢弃 KV cache；level 2 丢弃权重和 KV cache，恢复时需要重新写入权重。`wake_up(tags=["weights"])` 可以在分配 KV 前先同步权重，控制峰值。**睡眠不会让一个54GB BF16模型在醒来后变成能装进48GB的模型**，还需要量化、分片或 CPU 部分常驻。[vLLM 睡眠](https://docs.vllm.ai/en/latest/features/sleep_mode/)

### 3.4 推理前缀缓存的收益和 hybrid 限制

G=8 同题答案确实适合在 rollout 中复用长提示。但 hybrid 模型需要同时命中全注意力 KV 与 GDN 状态，存在块边界和状态检查点限制。当前文档已有共享前缀结点检查点与细粒度匹配单位；旧版本、不同 cache mode（缓存模式）的表现不能混用。**用实际 `num_cached_tokens`/命中率确认功能生效**，不能只看到 `enable_prefix_caching=true` 就认定省掉了7/8提示。[当前缓存说明](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)、[hybrid 部分命中 PR #46384](https://github.com/vllm-project/vllm/pull/46384)

需要同时处理权重更新后的缓存失效；睡眠也会清 KV。这里的前缀缓存加速的是无梯度生成，训练32次微批中的提示前后向仍然存在。32答每答约20 token时，不能照搬长思维链任务主要受 decode（逐 token 生成）支配的经验。

### 3.5 SWIFT 4.5.3 已有的参数，不等于当前组合已支持

直接阅读了 `v4.5.3` 标签，而不只读 latest：参数文档有 `vllm_quantization`、`vllm_enable_prefix_caching`、`vllm_engine_kwargs`、`vllm_limit_mm_per_prompt`、`vllm_max_lora_rank`。rank32 时推理端应允许至少32；多图数也应与实际一致。[4.5.3 参数源码](https://github.com/modelscope/ms-swift/blob/v4.5.3/docs/source_en/Instruction/Command-line-parameters.md)

**建议入口**：仅在独立环境中先证明“同一个 NF4 基座 + rank32 当前适配器 + 真实三图提示”的 vLLM 生成结果正常、加载没有静默遗漏模块，再测一个完整 GRPO 更新周期的同步/睡眠/命中率。此处没有给直接替换现有训练命令，因为尚未建立可用版本交集。

## 4. Unsloth：明确支持视觉 GRPO，但不是已证实的前缀共享替代品

Unsloth 的 Qwen3.6 官方页面将微调指向 Qwen3.5 指南。该指南明确支持 Qwen3.5 vision fine-tuning（视觉微调）、多图与 Vision GRPO，并有 `Qwen3_5_(4B)_Vision_GRPO.ipynb`。但同页明确：Qwen3.5 RL 需 `fast_inference=False`，使用 Unsloth 自身生成，不能把其他模型的 vLLM 快速推理演示套过来。[Qwen3.6 指南](https://unsloth.ai/docs/models/qwen3.6)、[Qwen3.5 微调与 RL](https://unsloth.ai/docs/models/qwen3.5/fine-tune)、[作者 Notebook](https://github.com/unslothai/notebooks/blob/main/nb/Qwen3_5_(4B)_Vision_GRPO.ipynb)

它值得作为独立候选比较，原因是较小的激活占用、编译和融合算子、专用反向实现可能减轻 NF4 LoRA 的开销；释放显存也可能使训练微批增大或减少梯度重计算。但官网模型通用的“2倍/70%节省”等宣传**不是本项目27B、3图、3500/20、G8条件下的结果**。其官方当前指南也承认 Qwen3.5 的 Mamba/Triton 核心使训练较慢。

迁移时应保持冻结视觉层、LoRA rank/alpha/目标模块、量化精度、GRPO 归一化和参考策略相同。默认示例会启用不同微调模块或保存 `lm_head/embed_tokens`，照抄会扩大训练对象。没有从官方材料确认它已通过可微前缀共享消除 G 份语言提示计算，故不应把换 Unsloth 当成这项结构优化本身。

当前同48GB可行性属于**有机会但未验证**：NF4单模型与当前策略相近，没有额外全精度生成副本的硬性需求；实际峰值仍取决于3图 token数、FP32提升位置、重计算和批量。不要在现有训练环境原地升级依赖进行尝试。

## 5. Liger 与 LM head：先确认当前是否已经只算短答案

### 5.1 当前 SWIFT 已有尾部 logits 优化

直接阅读 SWIFT v4.5.3 `swift/rlhf_trainers/grpo_trainer.py`：`_get_logps_via_local_forward()` 在模型支持时传 `logits_to_keep=answer_length+1`。Transformers 5.16.1 的 Qwen3.5 ConditionalGeneration 先切隐藏态再调用 `lm_head`。因此本任务正常路径应只把约21行投到词表，**不是3500行提示都算词表 logits**。[SWIFT 源码](https://github.com/modelscope/ms-swift/blob/v4.5.3/swift/rlhf_trainers/grpo_trainer.py#L1554)、[HF 5.16.1 源码](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L1785)

当前运行时审计已确认模型签名支持该参数，结合上述训练器入口，应按尾部优化存在来评估，而不把整段词表张量当作默认基线。21×248320 个 BF16 logits 约10.4MB，而3520行约1.75GB。**不要以“词表25万，所以 Liger 一定消除整个长提示输出张量”来预计收益**。

### 5.2 `use_liger_kernel` 在 GRPO 中的真实行为

SWIFT v4.5.3 `_prepare_liger_loss()` 将该开关映射到 `LigerFusedLinearGRPOLoss`，设置 `compiled=False`；训练时取模型隐藏态，与 `lm_head.weight`、答案 ID、completion mask、优势、old/ref logps 一同传入融合损失。它同时要求 `padding_free=False` 且 `advantage_estimator='grpo'`。这是**融合 LM head + GRPO 损失**，不等于给交叉熵套个更快函数。[精确版本实现](https://github.com/modelscope/ms-swift/blob/v4.5.3/swift/rlhf_trainers/grpo_trainer.py#L1765)

Liger 当前提供 Qwen3.5 专用补丁，源码明确处理 `Qwen3_5ForConditionalGeneration` 和 `model.model.language_model`，覆盖归一化、SwiGLU、RoPE 与融合交叉熵。早期 PR #1123 的 RoPE 限制已不等于当前 main；应使用与实际安装版本一致的源码判断。[当前补丁](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/transformers/monkey_patch.py#L3171)、[融合 GRPO 实现](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/chunked_loss/grpo_loss.py)

对本任务更可能是显存/小幅算子优化，不能消除64层长提示前后向。需核对 loss_type、token/sequence 重要性采样、KL、completion mask 和优势是否保持一致，不要顺手把 GRPO 改成 DAPO/GSPO。若 `lm_head` 自己做了 LoRA，直接传 `.weight` 是否包括适配器也需要检查；当前语言 LoRA 目标若无 lm_head，这一问题不出现。

有一条当前官方仓库报告：ms-swift 4.5.3 + qwen3_5 + Liger + ZeRO-3 + 混合纯文本/图像数据时，视觉编码器会重复运行导致多卡同步挂起。它**不能外推为当前单卡固定多图必然失败**，但说明需要先做短周期核对，不应直接在长训中切开关。[问题 #10410](https://github.com/modelscope/ms-swift/issues/10410)

## 6. FlashAttention、FLA、卷积与编译

### 6.1 FA2/SDPA 不会加速全部64层

Qwen3.5 的语言层为3:1 GDN/全注意力混合，另有视觉塔 attention。FA2 作用于全注意力与支持该接口的视觉 attention；GDN 需要 FLA 之类的专门内核。[HF 模型说明](https://huggingface.co/docs/transformers/v5.13.0/en/model_doc/qwen3_5)

RTX4090 属于 Ada，FA2 官方支持 Ada 的 FP16/BF16；FA3 公开实现要求 H100/H800，不能把 H100 上的 FA3 路线直接移到4090。PyTorch SDPA 本来就会根据约束选择 FlashAttention 等融合后端，所以 `sdpa→flash_attention_2` 不是“朴素 attention→FlashAttention”的必然转换。需看实际 kernel（内核），不是只看配置字符串。[FA 官方仓库](https://github.com/Dao-AILab/flash-attention)、[PyTorch 2.12 SDPA](https://docs.pytorch.org/docs/2.12/generated/torch.nn.functional.scaled_dot_product_attention.html)

### 6.2 FLA 已安装仍应确认实际分发

精确阅读 Transformers 5.16.1：`torch_chunk_gated_delta_rule` 与 recurrent 路径使用 `use_kernel_func_from_hub_with_fallback`；优先级是请求的 HF kernels → 原始 FLA/causal_conv1d 包 → PyTorch 实现。`hub_kernels.py` 中导入/函数解析异常会落到 PyTorch路径，因此包出现在 `pip list` 不是实际内核生效证明。[模型源码](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen3_5/modeling_qwen3_5.py)、[分发源码](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/integrations/hub_kernels.py#L822)

当前运行时审计进一步给出了函数分发：长序列 `chunk_gated_delta_rule` 来自 `fla.ops.gated_delta_rule.chunk`，单token `fused_recurrent_gated_delta_rule` 来自 `fla.ops.gated_delta_rule.fused_recurrent`，因此不应再把安装FLA列作主要加速建议。因果卷积仍是Transformers原生函数，可在下一轮独立环境中比较专用卷积内核，但它属于局部优化，不能按“修复全部48层纯PyTorch GDN”估计收益。[本地运行时证据](evidence/a-runtime-audit.json)

### 6.3 FlashQLA 的宣传倍数不适用于4090

Qwen 官方 FlashQLA 已提供 GDN 前向和反向，并可作为 FLA 后端，但当前 README 硬件要求 SM90/100/103/120/121；RTX4090 为 SM89，不在要求列表。Hopper/Blackwell 上的2–3倍前向结果不能作为本卡候选承诺。[Qwen 官方 FlashQLA](https://github.com/QwenLM/FlashQLA#installation)

### 6.4 `torch.compile` 放在第二阶段

编译可能融合 pointwise（逐元素）操作、减少 Python 调度，但不会去掉 G 份提示，也不会把 FLA 自定义算子的主体重新优化成另一个算法。动态图像网格、提示长度、答案长度、训练/生成模式切换和 PEFT 都可能造成图中断或重编译。[PyTorch 编译说明](https://docs.pytorch.org/docs/2.12/generated/torch.compile.html)

建议先固定一个小数据子集，分别记录首次编译和稳定迭代；用 `TORCH_LOGS=graph_breaks,recompiles` 定位边界，只编译稳定 decoder block（解码器层）或稳定小函数。没有证据前，不应把全模型 `fullgraph=True` 作为默认，更不应在当前长训中改变编译策略。bitsandbytes 0.50.2 的发布说明主要改进 ROCm/GB10，不能据此推断4090的NF4训练获得新数量级提升。[bnb 0.50.2](https://github.com/bitsandbytes-foundation/bitsandbytes/releases/tag/0.50.2)

## 7. 其他框架：是否值得为当前任务迁移

| 框架 | 官方可核查证据 | 与单48GB多图NF4的关系 | 迁移成本 / 建议 |
|---|---|---|---|
| **继续 SWIFT/HF** | 当前27B NF4训练已运行；4.5.3已有尾部logits与Liger GRPO入口 | 避免额外基座；适合移植紧凑组内前缀共享 | 最低；**结构优化首选承载点** |
| **Unsloth** | Qwen3.5视觉GRPO示例；3.6复用指南；要求fast_inference=False | 单模型NF4有机会；不是已证实的组内前缀共享 | 中；单独环境做同配置候选 |
| **SWIFT + vLLM** | qwen3_5 VLM、语言LoRA、睡眠、前缀缓存分别存在 | NF4/GDN动态LoRA组合待证；BF16模型自身不能完整装卡 | 中至高；生成占比显著时再做 |
| **verl** | 官方27B/35B Qwen3.5 FSDP2脚本，含geo3k图像数据；也有VeOmni后端 | 默认范例为多卡、非NF4；VeOmni共享开关仍纯文本限制 | 高；不能仅因“支持Qwen3.5”迁移 |
| **OpenRLHF** | 官方VLM脚本明确测试Qwen3.5各稠密尺寸，支持多图和冻结视觉；Actor有NF4入口 | VLM Actor明确拒绝use_liger_kernel；现成脚本未证明单卡NF4 rollout组合 | 高；未见针对本瓶颈独有收益 |
| **SGLang** | 原生qwen3_5模型有LoRA分支，Radix缓存与量化能力；可作其他训练器rollout后端 | 服务框架不提供当前训练反向；SWIFT的swift rollout官方目前主支持vLLM | 高；不能直接替代训练器 |
| **AReaL/HARTS** | 树训练、hybrid可微状态与重放的作者实现 | 现成路径主要多卡Megatron/MoE；量化和多图缺口大 | 很高；**读算法和抽取模块比整体换框架更合理** |

证据：[verl 27B脚本](https://github.com/verl-project/verl/blob/main/examples/grpo_trainer/run_qwen3_5_27b_fsdp.sh)、[verl 图像脚本](https://github.com/verl-project/verl/blob/main/examples/grpo_trainer/run_qwen3_5_35b_fsdp.sh)、[OpenRLHF VLM脚本](https://github.com/OpenRLHF/OpenRLHF/blob/main/examples/scripts/train_vlm_math_hybrid_engine.sh)、[OpenRLHF Actor](https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/models/actor.py)、[SGLang Qwen3.5源码](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen3_5.py)、[SGLang LoRA](https://docs.sglang.io/docs/advanced_features/lora)

## 8. 推荐研究顺序与明确停止条件

1. **先确认长提示的主要时间分布。** 训练前后向、生成prefill、视觉塔、逐token生成、old/ref打分、权重同步分别计时。本文不以框架名称猜当前瓶颈。
2. **把等价的组内提示共享列为主实验。** 第一版只针对同一题G=8平坦分叉，保持原答案、奖励、GRPO归一化和32答一次优化器更新。借鉴VeOmni的compact/expand、GDN分块状态分叉、卷积尾部重放；多图用显式组身份和真实三维位置编号，不能按占位token做跨题推断。
3. **先核对输出和LoRA梯度，再计时。** 固定一批真实多图和8条冻结答案，比较独立逐答路径与共享路径的答案logps、损失及LoRA参数梯度；再与“baseline只是改批次”的数值差异比较。只看reward均值不能证明实现等价。此项是后续实验方案，本文未执行。
4. **再考虑冻结视觉特征缓存与较少重计算。** 它们可与前缀共享叠加，但先分开核算收益；关掉梯度检查点是否放得下由峰值决定。
5. **Unsloth/Liger/FA2作为次一级实测候选。** 必须保持算法不变，记录稳定迭代和显存；若词表logits本来已截到短答案，降低对融合LM head的预期。
6. **vLLM只按实际生成阶段占比排序。** 若生成占总时间20%，即使生成时间归零，总加速也只有1.25倍；若主要重复在训练提示前后向，就不应把同卡推理迁移放在首位。

若共享实现只能在提示状态 `detach()` 后运行，或为了兼容被迫禁用图像/冻结全部语言提示路径，应将其标为改变训练梯度的近似实验，而不是成功完成等价加速。若NF4、FLA初始状态梯度、LoRA检查点中的任一项无法对齐，先修根因，不用fallback掩盖。

## 9. 证据可信度与尚未确认事项

- 本文优先使用官方文档、官方源码、作者论文/仓库；issue（问题报告）只证明某组合曾被报告，不代表所有用户或当前环境必然受影响。
- latest/main 页面会变化；报告中对SWIFT和Transformers的核心判断另核对了`v4.5.3`与`v5.16.1`源码。框架“当前支持”不能自动推及用户已装但版本未知的推理框架。
- 未找到当前单卡48GB、27B、真实多图、NF4、rank32 GRPO的共享前缀完整复现记录；不发布未经实测的端到端倍数。
- VeOmni的多图限制是明确事实；把其GDN方案移植到固定组多图的可行性是工程推断，需要验证。
- HARTS公开倍数来自其指定任务/硬件，不能直接用于当前短答案定位任务。
- 当前审计已确认FLA chunk/recurrent分发有效、原生因果卷积、尾部logits参数支持、800个量化线性层、冻结BF16视觉层与逐层重计算；尚未验证的是新共享图在本组合中的输出/梯度与实际端到端收益。`runtime_optimization=false`不应随意等同于所有第三方内核都未启用。

