# Eagle：多视觉编码器的有效部分与可复用边界

2026-10-25。阅读 [Eagle 论文](https://arxiv.org/abs/2408.15998) 全部 19 页（正文、参考文献、附录），对第 5–6 页的融合比较和预对齐训练图另作图像核对。审读 [官方仓库 Eagle 子目录](https://github.com/NVlabs/Eagle/tree/783f656d127ee498137b5ff52603ce36c292d317/Eagle)，不将同仓其它 Eagle2、Locate 等项目混为这篇论文。

## 能支持什么结论

Eagle 的价值是把“视觉表示更丰富”和“融合器更复杂”分开。它将不同视觉任务的编码器输出调整到同一空间网格，再按通道拼接；语言模型接收的视觉序列不必随编码器数增加。论文表 4 中，两个编码器时序列拼接分数更高但 token 数翻倍，三个编码器时通道拼接更高且保留 1024 token。不能概括成通道拼接在每个设置都胜出。

另一条结果是每个视觉专家先与同一语言模型单独对齐，再联合训练。不过这是增加了独立专家训练阶段，不能把最终收益都归因于“融合算法”。表 7 的延长训练对照支持预对齐有价值，仍不等于严格等总 FLOPs。论文的最佳组合含五个 RGB 视觉专家；继续加入第六个 DINOv2 反而下降。这里的“多模态”主要是不同任务的 RGB 表示，不是 RGB、红外、实采深度三传感器。

解冻结果同样依赖输入设置：原生 336 分辨率的 CLIP 解冻曾退化，插值到 448 后才改善。由此不能直接得出本项目“打开视觉 LoRA 必然涨分”。本文没有本赛实验，论文通用问答成绩也不能换算为 ACC@0.5 的提升。

## 实际源码走向

核心路径是：

```text
同一张RGB图
  → CLIP / ConvNeXt / EVA检测 / SAM分割 / Pix2Struct
  → 每路输出重排或插值为 B×1024×C_m
  → cat(dim=-1)，B×1024×ΣC_m
  → 两层MLP映射到语言维度
  → 替换图像占位token
  → Llama/Vicuna自回归答案，答案token交叉熵
```

`multi_backbone_channel_concatenation_encoder.py` 将统一输入设为 1024，输出网格设为 32×32；不同形状先转换为空间特征图，再双线性插值，最后只沿通道拼接。`multimodal_projector/builder.py` 的 `mlp2x_gelu` 是两层线性层与 GELU。`eagle_arch.py` 负责图像编码、视觉token插入、图像/提示标签置 `IGNORE_INDEX`、padding 和截断；`eagle_llama.py` 调用普通语言模型损失，没有专用框损失、传感器配准或对象关系头。

五路包装器的处理并不完全一样：SAM 用像素重排把空间大小减半、通道增大四倍；EVA 使用检测预训练骨干；Pix2Struct 反归一化后重新预处理，硬编码取 2025 个token再重排为45×45；ConvNeXt取末级特征；HRCLIP自行遍历视觉层。对我们的启示是**保留空间对应并校准每路特征尺度**，不是照搬五个巨型RGB编码器。

发现两个直接影响复用的发布差异：

- `clip-448` 分支在当前源码实际设置 `clip_args.input_image_size=336`；外层 forward 据这个字段缩放，再把 24×24 特征插值到32×32。注释说“actually 448”不能改变执行逻辑。当前发布代码不能直接认作论文448设置。
- README 仍把预对齐专家权重列为 TODO。公开预训练脚本的 `tune_mm_mlp_adapter=True` 会冻结全模型、只训练projector；公开微调脚本再训练全模型。这是两阶段默认路径，不能声称已经取得论文三阶段最佳配方的完整可复现材料。

这两点是静态源码结论；没有加载权重或运行作者模型。

## 对本项目的采用方式

采用**同坐标、多表示、短语条件的特征融合**，不采用五路RGB专家堆叠。我们的三路信息有物理差异：RGB提供外观和语义，IR补轮廓/可见性，原始Depth保留有效距离；因此各自编码后应对齐到RGB空间，交给定位decoder。Eagle固定方形网格不适合直接接本项目长宽比、原图框和不同来源深度。

如果使用Qwen，主方案可以让它保留低细节全图的视觉语义理解，同时让高细节三传感器特征直接进入空间定位头。这样既不用把所有高分辨率patch塞进27B，也不必把定位再压回四个文本数字。它是我们的结构提案，不是Eagle已实现的模型。

第一轮必须让简单的同位置通道拼接作为完整融合结构的对照：两者用同样传感器、骨干、数据、分辨率和框监督。若局部交互没有超过这个对照，就保留通道拼接；不能用更复杂的模块名称替代效果。

## 实读材料

全文材料：`F:/AIC/tmp/architecture_rethink_20261025/root/eagle.pdf` 和逐页文本；渲染图 `eagle-p5.png`、`eagle-p6.png`。

源码完整阅读：`Eagle/train.py`、`train_mem.py`、`eagle/train/eagle_trainer.py`、`model/eagle_arch.py`、`model/language_model/eagle_llama.py`、`model/builder.py`、`model/multimodal_projector/builder.py`、`model/multimodal_encoder/{builder,multi_backbone_channel_concatenation_encoder,hr_clip_encoder,convnext_encoder,sam_encoder,pix2struct_encoder}.py`、`eval/run_eagle.py`；另读 `vision_models/eva_vit.py` 的 `EVAVITVisionTower/build_eva_vit` 包装和初始化，以及 X5 Vicuna-7B 预训练/微调脚本。没有逐行审计第三方ConvNeXt/EVA底层算子、LMMs-Eval全部任务或同仓其它项目，未将这些列为已读源码。

