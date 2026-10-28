# Cambrian-1 / SVA：论文与源码深读

本文为该专题的独立方案，综合架构与实施优先级见[主报告](README.md)。

## 结论

Cambrian-1 值得借鉴的是 **按空间位置融合异构视觉特征**：每个低分辨率查询格点只读取各视觉塔的对应局部区域，保留高分辨率塔的局部细节；后续还可让语言模型内部的图像 token 再次访问原始视觉特征。它证明多塔融合可以在固定图像 token 数下改善通用视觉问答，**没有证明 RGB、红外、原始深度三传感器融合能提高本赛单框 ACC@0.5**。建议先在现役定位模型外做对齐后的局部特征融合或候选框重排对照，验证收益后再考虑修改 27B 的内部层。

## 阅读与证据范围

- 论文：[Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs](https://arxiv.org/abs/2406.16860)，本地 `F:/AIC/tmp/architecture_rethink_20261025/root/cambrian.pdf` 共 51 页；逐页阅读正文第 1–22 页、附录第 30–51 页，参考文献第 23–29 页只核查引用与章节边界。第 13 页图 8、14–15 页表 4–6、第 9 页训练配方图、第 46–47 页实现说明和塔权重表，已用 `pdftoppm` 渲染并目视复核。图在同一临时目录，文件名 `cambrian-p13.png` 等。
- 作者[官方源码](https://github.com/cambrian-mllm/cambrian)，本地 `F:/AIC/tmp/architecture_rethink_20261025/root/cambrian/`，修订号 `539ffc3`。逐段阅读核心文件：`cambrian/model/vision_sampler.py` 全 419 行；`cambrian/model/cambrian_arch.py` 全 653 行；`cambrian/model/language_model/cambrian_llama.py` 全 499 行；语言模型包装 `cambrian_mistral.py`、`cambrian_phi3.py`、`cambrian_gemma.py`、`cambrian_cohere.py`；视觉塔 `multimodal_encoder/{builder,base_encoder,clip_encoder,siglip_encoder,dino_encoder,clip_convnext_encoder,midas_encoder,hybrid_encoder}.py`；投影 `multimodal_projector/{builder,projectors}.py`。
- 训练／推理链进一步核查：`scripts/cambrian/{pretrain,finetune}_cambrian_{8b,13b,34b}.sh`、`cambrian/train/train_tpu.py`、`train_fsdp.py` 中数据、位置掩码、模型初始化和训练入口、`cambrian_trainer.py` 优化器分组；`cambrian/model/builder.py`、`cambrian/mm_utils.py`、`eval/eval/vstar/vstar_eval.py`。这不是对所有第三方视觉塔内部或通用评测脚本的逐行审计。

## 论文方法与源码的数据流

### 1. 不同网格怎样对齐

论文第 13 页设一个 `L×L` 查询网格，视觉塔 `k` 的输出须先变成 `m_k L × m_k L`。查询 `(i,j)` 只能对塔 `k` 的 `[m_k i:m_k(i+1), m_k j:m_k(j+1)]` 区域做交叉注意力，再沿塔拼接键值。论文第 46 页补充：`m_k>1` 的局部特征加可学习位置编码，查询另拼全局池化特征。

公开 8B 训练脚本配置 `image_token_len=576`、`query_num_list=[576]`，所以 `L=24`；四塔输出目标 token 数 `[576,576,576,9216]`，即三个 `24²` 网格和 ConvNeXt 的 `96²` 网格。源码 `cambrian_arch.py:271–287` 先用 `view→permute→flatten` 将每塔改排为 `[B×576, m_k², C]`；三个低分辨率塔各给一个键值，ConvNeXt 给 16 个，**每个查询共读 19 个局部键值**。`vision_sampler.py:155–234` 给每塔独立 K/V 投影，将它们沿局部 token 轴拼接，布尔 mask 限制补边区域，调用 scaled dot product attention。局部格点对齐是硬约束；注意力权重决定该格点内选哪塔和哪一细节。该实现要求塔的输出都是方形、边长能被查询边长整除；并未处理传感器之间的配准误差。

| 阶段 | 公开 8B 配置的典型形状 | 源码 |
|---|---|---|
| 四路视觉塔输入 | 同一张 RGB 图经各自处理器得到四个 `[B,3,R_k,R_k]` | `train_fsdp.py:966–1011`、`mm_utils.py:185–201` |
| 塔输出与投影 | `[B,576,C_1]`、`[B,576,C_2]`、`[B,576,C_3]`、`[B,9216,C_4]` → 各路 `[B,T_k,1024]` | `cambrian_arch.py:48–60,366–379` |
| 每格点交叉注意力 | Q `[B×576,1,1024]`；四路 KV 合计 `[B×576,19,1024]`；输出 Q 同形 | `cambrian_arch.py:382–393`、`vision_sampler.py:248–327` |
| 语言模型图像嵌入 | 查询组拼接、MLP → `[B,576,H_LLM]`；每行追加 newline，成为 600 个位置，GPU 推理可裁掉补边 | `cambrian_arch.py:395–449` |
| LLM 内重复融合 | 从当前层图像 hidden states 抽出 `[B×576,1,H_LLM]`，读**同一批未压缩视觉塔特征**，更新图像位置 | `cambrian_llama.py:168–253` |

投影顺序很关键：`mm_projector_aux_i` 先把异构通道变为统一 1024 维；SVA 做空间融合；`mm_projector` 再把各查询组拼接后映射到 LLM 维度。`num_query_group=1`、`connector_depth=3` 是公开脚本设置。代码中 `VisionTokenSampler` 的 `joint` 分支实际运行；`sep` 分支存在但不是这些脚本的配置。全局上下文在实现中只取 **第 0 路塔**投影后的均值（脚本中为 SigLIP），并非对所有塔取均值（`cambrian_arch.py:375–384`）。

附录第 46 页写“插值到 576×576”，但同页给 `m_k=1,L=24`、脚本 `[576,...]`、源码按 576 **token** 处理；据此应理解为 `24×24=576` 个位置。ConvNeXt 四阶段被插值为 `96×96` 后沿通道拼接（`clip_convnext_encoder.py`），不是把所有塔统一压到 576 再融合。论文方法的关键恰好是保留这一路 9216 个局部特征。

### 2. 梯度和 LLM 内部注入

训练脚本将大多数视觉塔设为冻结；`BaseVisionTower` 子类以 `torch.set_grad_enabled(unfreeze_mm_vision_tower)` 包住前向。连接器预训练使用 `tune_mm_mlp_adapter=True`，`train_fsdp.py:1676–1685` 冻结整体模型，只让 `mm_projector`、各辅助投影、`vision_sampler`、`vision_query`、图像换行向量等学习；微调脚本不设该开关，允许 LLM 和连接器联合更新，视觉塔仍冻结。优化器在 `cambrian_trainer.py:289–318` 为采样器／查询单设学习率。论文第 9、47 页表明连接器预训练有益、解冻视觉塔通常有益但训练约慢 50–55%；这来自其 VQA 数据分布，不能直接迁移为本赛收益。

LLM 内部并非给**所有文本 token** 新增一个视觉交叉注意力模块。Llama 实现先运行普通 decoder layer，再仅抽取图像 token 的 hidden states，做一层新的 `VisionTokenSampler`，写回相同图像 token 位置；后续语言层通过常规自注意力读取被更新的图像位置（`cambrian_llama.py:142–253`）。训练时还保留对补边的布尔 attention mask，GPU 推理时会从图像网格裁掉补边并同步裁视觉特征（`cambrian_arch.py:289–330,422–449`）。输入嵌入和中间注入两处的损失路径都回到连接器；只有显式解冻时才回到塔权重。文中“多层视觉聚合”指多次访问原特征，并不是多轮外部裁剪或主动感知。

该源码将图像 token 起始位置固定在 `config.image_position`；训练脚本 8B/13B/34B 分别写 91/35/87，`train_fsdp.py:1201–1219` 也用它插入无图样本的虚拟图像位置。`generate` 的首轮会缓存视觉特征，随后 token 解码不再重新编码图像（`cambrian_llama.py:437–495`）。要换进 Qwen3.6-27B，必须重新实现适配其 image token 布局、RoPE/位置、缓存及 vLLM 运行时；现有 Llama 类不能作为插件直接挂载。其他 Mistral/Gemma/Cohere 包装仍按旧 6 项返回值解包，而 `cambrian_arch.py` 当前返回 10 项，也没有 Llama 的深层注入实现；因此不能把它们视为已验证的等价移植路径。

## 实验能支持什么

- 第 14 页表 4，在四塔、Vicuna-7B 的 **VQA 类别均分** 中：简单拼接的 Vision-Centric 为 52.6，SVA 无内部多层读取为 52.6，完整 SVA 为 53.2；OCR & Chart 则为 50.1→55.2→55.5。视觉类相对拼接仅 +0.6，主要清晰收益在 OCR。这里“无多层读取”还把查询组改为 `G=3`，并非只移除一个组件的严格单因素比较。
- 第 15 页表 5，单个 CLIP 塔把 576 token 压到 36 token 时，Vision-Centric 的插值+MLP 为 43.7，SVA 为 46.9；这是**压缩场景**的空间归纳偏置证据。表 6 多层读取在 OCR & Chart 52.4→53.3，仅 +0.9；没有 bbox IoU/ACC 消融。
- 第 19 页表 9，Cambrian-1-8B 用 576 视觉 token 报 CV-Bench 2D/3D 72.3/72.0，较同骨干 LLaVA-NeXT-8B 的 62.2/65.3 好；但训练视觉塔、数据、连接器等同时改变，不能单独归因于 SVA。CV-Bench 的 2D 任务是关系/计数选择题，3D 是深度顺序/相对距离选择题，**并不输出单个 RGB 坐标 bbox**。
- 第 47 页表 22 的注意力占比说明任务分布可改变塔的使用：GQA 中 SigLIP/CLIP/DINOv2/ConvNeXt 为 29.7/18.5/24.1/27.7%，DocVQA 为 31.1/13.4/11.0/44.5%。这可启发模态权重诊断，但注意力权重不是因果贡献。第 45–46 页承认 Cambrian-10M 与若干评测图重合；本赛官方测试禁止训练，不能照搬其“同图不同问不算测试训练”的解释。

另有公开材料不一致：论文第 20、46、48 页称最终四塔中的 DINOv2 为 ViT-L/14@518，且 34B 的中层步长为 9；仓库六份脚本使用 `facebook/dinov2-giant-res378`，34B 步长 7。报告保留两种来源原貌，具体复现应以选定 checkpoint 配置及其实际权重为准，不从论文表格反推源码架构。

## 对 RGB／热红外／原始 Depth 的适用边界与新提议

**可借鉴的机制：** 三个模态先在同一 RGB 坐标系下配准，各自提取 `H_k×W_k` 网格特征；再用 SVA 的“每个目标格点只读各模态对应局部区域”融合。可保持固定数目的空间 token，并避免把所有高分辨率特征都塞给 LLM。若只在候选框与参照框周围取局部网格，成本比为全图 27B 增加若干跨层模块低；这属于本项目的新提议，**不是 Cambrian-1 原实现**。

**不应直接照搬：** Cambrian-1 四塔都读取**同一张 RGB 图**；训练整理函数甚至要求每条样本恰有一个 `<image>` 标记（`train_fsdp.py:1089–1101`），多传感器输入并未现成支持。论文第 33 页的 MiDaS 是从 RGB 图估深的预训练视觉表征；正式四塔也没有 MiDaS。它没有读本赛的原始深度图，更没有处理深度缺失值、量纲、噪声或红外强度。直接把单通道 IR/Depth 复制为三通道喂入 CLIP/SigLIP/DINOv2，会遇到显著输入分布偏移，不能视为已有预训练能力。方形补边+固定 24 网格会削弱细长物体、小目标和精细框边界；不同传感器视差/配准误差会使“相同格点”并非同一物理位置。正式方案需每模态独立处理、有效深度 mask、按 RGB 框坐标投影以及可测的缺模态训练/推理规则。

**最小验证路径（本项目假设）：** 保留现役 27B 单框基线 A；在不改 LLM 内部层的情况下，对同一候选框附近的 RGB/IR/Depth 特征做逐格点投影+局部融合得到 B，比较 B 对 A 的 ACC@0.5、IoU、错实例率，并分空间/距离/序数 Query 统计。对照再分别屏蔽 IR、Depth，打乱模态空间位置，以及把局部交叉注意力换成等 token 的插值+拼接，确认收益来自真实传感器信息和空间对齐。只有 B 稳定增益且可负担，再做 C：接入 LLM 中层重复读视觉特征，比较 C 对 B 的单框收益、显存和延迟。试验按图组/序列隔离，仅使用比赛允许的训练来源；官方测试图不回灌训练。

**架构裁决建议：** 先采纳“局部空间对应、各路独立投影、模态消融”这三个低风险思想。暂不移植 Cambrian-1 的四塔大型连接器、固定位置 Llama 内层写回和整套 TPU 训练管线；本赛只有单框指标，现有证据不足以支付其计算与迁移成本。
