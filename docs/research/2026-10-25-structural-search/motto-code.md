# Motto 固定提交源码审读：空间锚、解码器和真值信息边界

2026-10-25。审读 [Motto 论文](https://arxiv.org/abs/2607.24407) 与公开仓库固定提交 [`4193cc8`](https://github.com/TG0110/Motto/tree/4193cc896ba36aead364aaadfe08b6ba749c5969) 的三个文件：[模型 1–831 行](https://github.com/TG0110/Motto/blob/4193cc896ba36aead364aaadfe08b6ba749c5969/motto/modeling_motto.py)、[推理 1–310 行](https://github.com/TG0110/Motto/blob/4193cc896ba36aead364aaadfe08b6ba749c5969/motto/inference.py)、[单图 CLI](https://github.com/TG0110/Motto/blob/4193cc896ba36aead364aaadfe08b6ba749c5969/tools/infer_one.py)。这里只从可见源码判断真实执行路径，不推定未公开的训练/评估行为；未装环境、未加载权重、未运行。Motto 是 Qwen3-VL 2B/4B 单图或多图开放式 grounding，现役赛题模型是接口不同的 Qwen3.6-27B `qwen3_5` 三图→单一 RGB BBox，不能直接套类或论文分数。

## 1. 源码里的真实数据流

| 阶段 | 关键实现 | 对“原生定位”的界定 |
| --- | --- | --- |
| 空间 token | `<DET_0..8>` 为 3×3 P 网格；`<EVI_0..24>` 为 R token 候选，最大接 4 P/6 R；`<IMG_0..9>` 选目标图；另有 `<AGG>`、`<VIS_START>`、`<VIS_0..8>`、`<VIS_END>`。[初始化与 token 注册](https://github.com/TG0110/Motto/blob/4193cc896ba36aead364aaadfe08b6ba749c5969/motto/inference.py#L52-L86) | P token id 自带粗格位置。模型输出它们属于自回归分类，再由隐藏态供后续空间头使用，不是由连续坐标头直接预测 P。 |
| 原生图像特征 | 调 `self.visual(pixel_values, grid_thw)` 提取 Qwen3-VL merged visual token，再按 `grid_thw / spatial_merge_size` 重建每图二维网格；缓存、按 `<IMG_i>` 选一张图。[模型约 354–395、513–548 行](https://github.com/TG0110/Motto/blob/4193cc896ba36aead364aaadfe08b6ba749c5969/motto/modeling_motto.py#L354-L395) | 不调用外部检测器视觉 backbone；但重建假定每张静图一个网格。多图选图与本赛三传感共同决定同一 RGB 框是不同问题。 |
| 两次视觉回看 | `ExternalAggregator` 把文本 hidden + 4×4 pooled 图网格变成 `<AGG>` 的输入 embedding；`LocalVisionZoomInV5` 用 P-token 的 3×3 格加权、全局/局部分支和 evidence hidden 生成 9 个 `<VIS_i>` embeddings，再注入随后语言前向。[模型约 176–329、549–631 行](https://github.com/TG0110/Motto/blob/4193cc896ba36aead364aaadfe08b6ba749c5969/motto/modeling_motto.py#L176-L329) | **“视觉网格→文本状态→视觉回看→再入语言模型”已是公开近邻**。其反馈发生在额外 token 的输入 embedding/阶段生成，不是把未生成的完整 Query 后 `[LOC]` 于第 k 语言层内部读出并回注。把后一结构称为首创仍无依据。 |
| BBox 解码 | P/R hidden 经 `VisionConditionedProjector` 与视觉网格 cross-attention，再以 `ModeAwareThoughtProjector` 混到 300 object queries；自建 3 层 `nn.TransformerDecoder` 对图像网格 cross-attention，输出 300 个 sigmoid `cxcywh`、对象 logits 与全局存在性。[模型约 22–136、470–493 行](https://github.com/TG0110/Motto/blob/4193cc896ba36aead364aaadfe08b6ba749c5969/motto/modeling_motto.py#L22-L136) | 这是可训练的 **DETR 式多框解码头**，不是直接从单个 MLLM token 输出四边界。类 docstring 写 “Qwen3-VL + GroundingDINO (Swin backbone ckpt)”，但这个文件未 import GroundingDINO、未加载 Swin ckpt，实际建的是普通 `nn.TransformerDecoder`。不能用 docstring 断言外部 DINO 参与推理。 |
| 自由推理 | `generate_with_boxes` 第一阶段自回归到 `<VIS_START>`，拼入视觉占位 token 后第二阶段继续；缓存生成的 P/R hidden，遇设定的 `.` trigger 才调用空间 decoder。随后过滤分数、NMS、top-k，单图工具绘图。[模型约 755–831 行](https://github.com/TG0110/Motto/blob/4193cc896ba36aead364aaadfe08b6ba749c5969/motto/modeling_motto.py#L755-L831)、[推理约 167–310 行](https://github.com/TG0110/Motto/blob/4193cc896ba36aead364aaadfe08b6ba749c5969/motto/inference.py#L167-L310) | 公开推理按生成 token 决定锚，不在函数参数上传 GT box。结果可能有多个框；本赛需要唯一 RGB 框及有效归一化格式，不能用 NMS 输出列表直接提交。 |

`ReasoningGuidedDecoder.obj_query_embed` 和 `weights` 在当前 `forward` 里定义/算出却未用于输出；300 queries 实际来自 P/R token 的线性混合，不应说“300 学习 object query embedding”。`GroundingOutput.loss` 直接取基座 `vlm_outputs.loss`；本模型文件**没有 Hungarian matching、L1/GIoU、focal detection loss 的训练实现**。公开源只能证明 decoder 可前向，不能复核论文检测损失的执行细节。[模型输出与 loss](https://github.com/TG0110/Motto/blob/4193cc896ba36aead364aaadfe08b6ba749c5969/motto/modeling_motto.py#L620-L685)

## 2. GT 空间锚是否泄漏到推理

论文 §3.1 明写：训练用**每个 Query 的 GT boxes**与网格单元 IoU 排序，选 P/R 的空间 token 序列；随后 §3.3 对这些序列做教师强制语言交叉熵，同时训练 detector 的匹配框损失。[论文](https://arxiv.org/abs/2607.24407)。这给 detector 训练输入提供了 GT 导出的**粗格位置先验**，不能把其单臂成绩解释成“没有目标位置监督，仅靠隐向量读出”。这本身是正常的训练标签构造，不等于测试集泄漏。

公开单图推理调用 `generate_with_boxes`，从**模型实际生成**的 `<DET_i>` 隐藏态缓存取 P，`grid_ids` 由生成 token id 映射；`LocalVisionZoomInV5.build_weight_map` 据此给相应 3×3 区块权重。单图入口没有 GT bbox 参数，也没看到在推理中把标注 P token 塞进去。[P 提取与局部权重](https://github.com/TG0110/Motto/blob/4193cc896ba36aead364aaadfe08b6ba749c5969/motto/modeling_motto.py#L249-L285)、[生成入口](https://github.com/TG0110/Motto/blob/4193cc896ba36aead364aaadfe08b6ba749c5969/motto/modeling_motto.py#L755-L831)。**因此公开推理路径不支持“直接喂测试 GT 框”的指控**。但训练与正式 benchmark eval 源码、权重尚未发布，无法独立确认表格是否全以自由生成 P 评估，也无法量化教师 P 与生成 P 的分布差。必要的验证应同时报自由生成 P 和 oracle GT P 的差距；前者才能与现役直接框公平比较。

额外需注意 `build_question` 默认措辞“在最后一张图找 TARGET”，但公开 `build_messages` 只放一张图；多图选择逻辑虽在模型中，单图 demo 不验证它。[推理提示](https://github.com/TG0110/Motto/blob/4193cc896ba36aead364aaadfe08b6ba749c5969/motto/inference.py#L13-L27)、[图像消息](https://github.com/TG0110/Motto/blob/4193cc896ba36aead364aaadfe08b6ba749c5969/motto/inference.py#L129-L145)。模型以第一个出现的 `<IMG_i>` 选目标图，未出现则取首图，这不是按 RGB/IR/Depth 物理含义融合三幅图。[目标图选择](https://github.com/TG0110/Motto/blob/4193cc896ba36aead364aaadfe08b6ba749c5969/motto/modeling_motto.py#L430-L447)

## 3. 对本项目方案的原创边界与取舍

Motto 已覆盖**空间锚 token、自回归 P/R 隐向量、目标条件视觉网格 cross-attention、再次视觉回看并注入语言 token、DETR 式连续框输出**，这些不能作为本项目的新机制宣传。原生 Qwen 特征接 DETR decoder 也不是无需检测结构；Motto 的 paper/result 是原生 MLLM 语义驱动但包含完整检测头。

[本项目完整结构草案](native-grounding.md)能保持可检验区别的部分是：现役 `qwen3_5` 的 **RGB/IR/Depth 同场景传感器**共同辅助单一 RGB 坐标；完整 Query 条件下的可变角色与软实例集合保留位置、内容及传感证据，经成对关系更新，在**同一次首轮前向的中间语言层**回注对应证据位置，再由共享单框空间模块输出；不先用 GT 派生的离散格锚训练 token 生成，也不再跑“先生成 P→停机→局部 zoom→继续 R”的协议。最终边界在小模块内交互，使用有序连续区间与整框损失；它不声称建模完整四维坐标概率。各组成仍可能有其他先例，真正要验证的是这组连接的净效果，不能只凭结构图宣称新颖。

不建议照搬 Motto 的 300-query 多目标 decoder 或双模式路由：赛题需要唯一 RGB 框，多框/NMS 输出不契合其评测形式；合法外部训练数据可扩充，当前 157 图只是专项池，不能由此断言结构训练总体数据不足。可吸收**原生视觉网格的 Query 条件读取**及“训练/推理 P 是否一致”的审计原则，不复刻 P/R token 协议。完整模型应直接与同数据原生 AR、删关系、删主体回注、同状态 AR 出口比较；末层简单头可以诊断，不作为完整结构的准入。Motto 文献分数不能与本赛 .8012 直接对齐。

## 4. 当前公开仓库的复现边界

- 固定提交的 [推理文件](https://github.com/TG0110/Motto/blob/4193cc896ba36aead364aaadfe08b6ba749c5969/motto/inference.py#L15) 导入 `motto.modeling_qwen3.Qwen3VLGroundingPrompt`，实际公开模型文件叫 `motto/modeling_motto.py`；对前一路径查询返回 404。按当前提交原样运行 CLI 会先遇到模块导入错误，不能把它当已验证的复现管线。
- 模型 `forward` 除真实 VLM 前向外还无梯度地额外跑一次 `super().forward` 以取得上下文 hidden；若 `<AGG>` 预填充，又有一次无梯度前向。初次 `self.visual` 也另跑一次。这样的成本与论文所述速度指标需要在可运行代码/权重发布后实测，不能由源码静态推出。[模型前向](https://github.com/TG0110/Motto/blob/4193cc896ba36aead364aaadfe08b6ba749c5969/motto/modeling_motto.py#L504-L634)
- `forward` 触发空间头的缓存/标志由 `generate_with_boxes` 置位；没有公开 trainer 来展示 detection loss 如何反传，也没有 README 所称训练权重可供检验。因此源码支持**结构取证**，不支持复现实验结论或在本项目直接加载使用。[触发与输出](https://github.com/TG0110/Motto/blob/4193cc896ba36aead364aaadfe08b6ba749c5969/motto/modeling_motto.py#L632-L680)
