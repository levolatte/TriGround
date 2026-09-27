# EGM LocateAnything 与 Rex Omni 源码审计

日期：2026-09-22。范围仅为公开的作者论文、官方 GitHub 与官方 Hugging Face 模型页；没有下载权重、数据或执行外部代码，也没有启动 GPU。本文把“官方文档声称”“源码可见”和“本机未复现”严格分开。

## 可直接用于复赛路线的结论

首批单张 L40 推理评估应先做 **EGM-Qwen3-VL-8B** 和 **LocateAnything-3B**，均只作为 RGB 外部专家，而不是替换现有 RGB+IR+Depth 主线。前者最接近当前 Qwen3-VL-8B 的复杂指代/序数问题；后者的并行框解码适合作为强定位与候选框质量对照。**Rex-Omni 不放进首批全量直接评分**：在其公开 wrapper（封装器）中没有独立的 referring task，主要接口是“图片 + 类别/短语列表”的 detection，因此更适合在强基线错误上按名词生成候选，先报告 `OracleRecall@K`，再决定是否训练重排。

这不是已证明的 L40 可用性或得分承诺。EGM 的官方评估启动脚本把 `dp-size`/`data-parallel-size` 固定为 8；LocateAnything 的公开高吞吐结果是 H100，`la_flash` 的公开内存探针是 A100。两项目都没有在官方材料中给出“单张 L40 跑完整 RefCOCO 或本赛数据”的实测日志。因此首批应先做 8--16 条 smoke test（冒烟测试），记录加载、峰值显存、真实生成 token 数、可解析率与每样本时延；通过后才跑相同的 119 条验证集。

竞赛官方 PDF 的主指标是 `ACC@0.5 = count(IoU >= 0.5) / all queries`，输出为 RGB 坐标系的归一化 `xyxy` 框。三套公开模型都没有原生的红外或深度语义输入接口；把 IR/Depth 灰度图拼进 RGB、假装多图输入，或用其论文分数证明三模态收益，均没有一手依据。

## 已核实的权重与原生输入输出

| 项目 | 已发布权重与来源 | 原生输入 | 原生输出及接入含义 | 对本赛的边界 |
| --- | --- | --- | --- | --- |
| EGM | 官方仓库明确提供 RL checkpoint：[`nvidia/EGM-8B`](https://huggingface.co/nvidia/EGM-8B) 与 [`nvidia/EGM-4B`](https://huggingface.co/nvidia/EGM-4B)，并给出 `hf download` 命令；8B 建在 Qwen3-VL-8B-Thinking 之上。见[仓库评估段](https://github.com/NVlabs/EGM#evaluation)。 | 官方 `sglang_infer.py` 对每条样本读取一个 `image` 文件与 `sent`，发送一张 image URL 和文本提示 `Locate {sent}, output its bbox coordinates using JSON format`。[源码](https://github.com/NVlabs/EGM/blob/main/verl/scripts/sglang_infer.py)。 | 模型返回任意文本；解析器优先找 JSON 的 `bbox_2d:[x1,y1,x2,y2]`，否则找任意四数数组，最后取文本末尾四个数字。默认 `scale` 将 0--1000 坐标变为像素框。[同一源码](https://github.com/NVlabs/EGM/blob/main/verl/scripts/sglang_infer.py)。 | 可以将 RGB + 英文 Query 送入，最后将框除以图宽高并用赛题评测器计分。不能声称它原生消费 IR/Depth；输出解析必须保留原始文本并审计，不应悄悄选择更有利的四个数字。 |
| LocateAnything | 官方 README 用 [`nvidia/LocateAnything-3B`](https://huggingface.co/nvidia/LocateAnything-3B) 的下载和 worker API；公共权重明确**不**支持 visual-prompt inference（视觉提示推理）。见[README](https://github.com/NVlabs/Eagle/blob/main/Embodied/README.md)。 | `LocateAnythingWorker` 的 quick start 接受一张 RGB `PIL.Image` 和类别列表或指代表达；`ground_multi(image, phrase)` 是公开短语定位接口。[接口示例](https://github.com/NVlabs/Eagle/blob/main/Embodied/README.md#-quick-start)。 | 框为 `<ref>label</ref><box><x1><y1><x2><y2></box>`，整数 0--1000；`none` 为无目标。README 给出了除以 1000 后映射回图像的解析代码。[格式与解析](https://github.com/NVlabs/Eagle/blob/main/Embodied/README.md#-output-format)。 | 可直接以 `ground_multi` 做单框/短语对照，或 `detect` 生成候选。没有 RGB-T/RGB-D 训练或输入约定；不得把“多张图占位符”误表述为已训练的传感器融合。 |
| Rex-Omni | README 指定 [`IDEA-Research/Rex-Omni`](https://huggingface.co/IDEA-Research/Rex-Omni) 为 `model_path`，另提供 AWQ 量化版本。见[官方 quick start](https://github.com/IDEA-Research/Rex-Omni#2-quick-start-using-rex-omni-for-detection)。 | `RexOmniWrapper.inference` 接受单个或批量 `PIL.Image`、`task`、类别/短语列表；公开任务枚举是 detection、pointing、visual_prompting、keypoint、OCR、GUI grounding/pointing，没有单列的 `referring` task。[wrapper](https://github.com/IDEA-Research/Rex-Omni/blob/master/rex_omni/wrapper.py)；[任务提示](https://github.com/IDEA-Research/Rex-Omni/blob/master/rex_omni/tasks.py)。 | 返回每图的 `raw_output` 与 `extracted_predictions`；detection 的结构化结果为 `{category: [{"type":"box", "coords":[x0,y0,x1,y1]}, ...]}`。官方 README 明确该结构。[返回约定](https://github.com/IDEA-Research/Rex-Omni#initialization-parameters-rexomniwrapper)。 | 优先当“按类别生成多个 RGB 候选”的组件。复杂 Query 若直接塞进类别字符串，是否保留序数、关系、否定等语义必须实测；不能因 README 中有 object referring cookbook 就假定其独立接口已按本赛格式验证。 |

## EGM：长生成是方法本体，不是把当前上限从 40 调大就能宣布复现

论文的因果链是：先用推理轨迹做 SFT（监督微调），再做 GRPO（组相对策略优化）；推理时让模型生成辨别目标的过程后给出框。论文附录的**统一评估参数**是 `temperature=0`、`top_p=1`、`top_k=20`、`max length=4096`；仓库的 SGLang/vLLM 评估脚本也硬编码 `--max_tokens 4096`。[论文表 8](https://arxiv.org/html/2601.13633v3#A2.SS0.T8)；[SGLang 启动/评估脚本](https://github.com/NVlabs/EGM/blob/main/verl/scripts/sglang_infer.sh)；[vLLM 脚本](https://github.com/NVlabs/EGM/blob/main/verl/scripts/vllm_infer.sh)。

因此，当前项目生成上限 40 token 不能被称为 EGM 原型，更不能用其低分否定 EGM。最小、可归因的推理原型应是：固定 RGB、Query、图像像素预算、bbox 后处理与赛题 `ACC@0.5`；仅将 EGM 的输出上限设为官方 4096，记录实际 completion token、是否被截断、原始 reasoning（推理文本）、解析框、IoU、时延和峰值显存。先在 8--16 条做通过门，再跑 119 条；若显存或时延不可接受，可以把 512/1024 当作**新消融**，不可再叫论文同设置。对照至少包括当前原生 Qwen3-VL-8B RGB 和 EGM-8B，二者都必须接受相同的最终赛题评测器。

EGM 的官方 RL 启动配置也不是单卡范例：它启动 Ray 时写死 `--num-gpus=8`，训练 batch=256、每 prompt 采样 16 条，最大 response length=2048；论文报告 SFT 在 8 张 A100、RL batch=256，Qwen-8B 的 RL 数据为 83,989 条。[训练脚本](https://github.com/NVlabs/EGM/blob/main/verl/scripts/grounding_qwen.sh)；[论文训练设置](https://arxiv.org/html/2601.13633v3#S5.SS1)。故本轮只评价预训练权重的**推理**，不把 EGM 的 SFT/GRPO 当作单张 L40 的低成本迁移方案。

## 两个不同来源的 88.6：都不可直接移植为本赛分数

GPT Pro 所引的 LocateAnything 论文事实是正确的：其表 5 的 RefCOCOg val `F1@IoU=0.5` 列中，LocateAnything-3B 与 Qwen3-VL-8B 均为 **88.6**。这是同一数据切分、同一阈值下的同分；不能据此推出 LocateAnything 在完整表中没有差异：该切分的 `F1@IoU=0.95` 分别为 41.5 与 33.4，mean 分别为 76.7 与 74.9，RefCOCOg test 的 `F1@IoU=0.5` 也分别为 88.8 与 88.6。[LocateAnything 论文表 5](https://arxiv.org/html/2605.27365v1#S4.T5)。该 F1 指标与本赛单框 `ACC@0.5` 不同，仍须在本赛评测器上复算；但不能拿 EGM 的同名数字反驳这条 LocateAnything 论文结论。

**另一个独立事实**是：EGM 论文表 4 中的 88.6 对应 `Qwen3-VL-8B-Thinking + EGM w/o R` 的八个 RefCOCO/RefCOCO+/RefCOCOg 切分平均值。表题和最后列都写明 `Accuracy` / `Avg. Acc`；论文将成功定义为预测框 `IoU > 0.5`，评价是成功样本比例。它不是 F1，也不是平均 IoU，更不能与前述 LocateAnything 表 5 的 88.6 互相替代或反驳。[成功定义](https://arxiv.org/html/2601.13633v3#S3.E1)；[评价定义](https://arxiv.org/html/2601.13633v3#S5.SS2)；[88.6 所在表](https://arxiv.org/html/2601.13633v3#S5.T4)。

原论文在摘要中把 91.4 称作 “IoU”，官方模型卡也写 “average IoU”；但其正文表 4、评价定义及发布评估源码并不支持把 88.6 解释为 mean IoU：评估源码同时输出 `mean_iou` 和 `pass_rate@0.5`，后者才对应论文的 Accuracy。今后报告应写为“RefCOCO 八切分 Avg. Acc@IoU>0.5 = 88.6”，不应复述为 F1 或 mean IoU。[评估汇总源码](https://github.com/NVlabs/EGM/blob/main/verl/scripts/sglang_infer.py)。

它与本赛 `ACC@0.5` 在“单框成功率”定义上接近，但仍不构成同数据口径：EGM 使用 RefCOCO 系列 RGB 数据、575,208 条 vanilla-grounding 训练样本，并以 Qwen3-VL-8B-**Thinking** 为基座；本项目是城市 RGB/红外/深度、923 条目标域训练、现有基座是 8B-**Instruct**。此外 EGM 用 `IoU > 0.5`，赛题是 `IoU >= 0.5`；必须以比赛评测器重算。

## LocateAnything 的 `la_flash`：L40 是目标平台之一，但性能数字不是 L40 实测

官方 README 的表述很明确：MagiAttention 只支持 Hopper/Blackwell；对于 **non-Hopper/Blackwell** 的 A100、L40 等，HF 权重仓内含 `la_flash` batch runtime，使用 FlashAttention varlen 稀疏 range plan，避免 stock path 的稠密 SDPA mask。[架构限制与 L40 说明](https://github.com/NVlabs/Eagle/blob/main/Embodied/README.md#%EF%B8%8F-magi-attention-hopper--blackwell-only)。其调用是 `--attn la_flash --vision-attn flash_attention_2 --scheduler pipeline`，并保留 MTP/NTP hybrid decode（混合并行/自回归解码）路径。[批量运行命令](https://github.com/NVlabs/Eagle/blob/main/Embodied/README.md#-batch-inference-release)。

结论应写成：`la_flash` 确实是给 L40 这类非 Hopper/Blackwell 卡准备的推理路径，**不是** Magi 的 Hopper 路径；但官方唯一给出的详细显存探针是 A100、3840x2160、batch 4、25,600 输入 token，`la_flash` 为 8.0314 秒/11.71 GB，对照 SDPA 为 8.2600 秒/35.12 GB。它证明作者报告过稀疏 attention 的 A100 内存下降，不能证明 L40 同样的时延、显存或吞吐。README 的 12.7 BPS 则标注为单张 H100，也不可作为 L40 预算。应在 L40 先实际测目标分辨率与 batch 1/2/4。

### LocateAnything 单卡训练的证据边界

官方“Full fine-tuning”启动命令是 `torchrun --nproc_per_node=8`，`--attn_implementation magi`、16K sequence、DeepSpeed ZeRO-2；这不是 L40 单卡训练证据，而且 Magi 的公开支持范围排除了 L40。[训练入口](https://github.com/NVlabs/Eagle/blob/main/Embodied/README.md#%EF%B8%8F-training-continual-sft)。README 另有 `bash shell/locate-anything-lora-visual-prompt.sh 1 ...` 的单进程 LoRA 示例，默认冻结 LLM 与视觉主干、训练 MLP projector（投影器）；但该示例的任务是 visual prompt，且同一 README 明确公共 3B checkpoint 未支持 visual-prompt inference。[LoRA 段](https://github.com/NVlabs/Eagle/blob/main/Embodied/README.md#lora-fine-tuning)。

所以“单张 L40 能训练 LocateAnything”目前最多有**单进程 LoRA 脚本存在**这一证据，尚无作者公开的 L40 显存/成功日志，也无同等 text-referring 训练 recipe。复赛首轮不应安排全参训练，更不应把 visual-prompt LoRA 的脚本当作文本指代微调已证实可行。

## 对既有 GPT Pro 建议的必要修正

1. LocateAnything 论文表 5 的 RefCOCOg val `F1@IoU=0.5` 中，LocateAnything-3B 与 Qwen3-VL-8B 均为 88.6，这一引述应保留。EGM 论文表 4 的 88.6 则是另一项 `Avg. Acc` 消融事实，必须分开叙述；只有把 **EGM** 的 88.6 写成 F1、平均 IoU，或直接同当前 Qwen3-VL-8B-Instruct 数值比较时，才需要纠正。
2. 若建议以 H100 的 12.7 BPS 或 A100 的 `la_flash` 数据承诺 L40 速度/显存，应降级为待测假设。可采纳的是 L40 走 `la_flash` 的官方接口，不是性能数字的迁移。
3. 若建议立即在单 L40 上全参训练 LocateAnything，应拒绝：公开 full-SFT recipe 是 8 进程 Magi，L40 不属于 Magi 支持架构。单进程 LoRA 的公开任务还不是本赛所需的文本指代。
4. 若建议把 Rex-Omni 当作所有 Query 的最终答案器，应先改为候选组件。它的类别式公开任务接口与唯一目标、关系/序数 Query 不同；先测候选 `OracleRecall@K` 才能知道它是否给选择器留下可提升空间。

## 建议执行顺序与停止条件

1. **EGM-8B RGB 评估**：8--16 条确认官方 4096 上限能加载、输出可解析后，再以当前原生 Qwen RGB 相同清单跑 119 条；只报告赛题 `ACC@0.5`、mIoU、解析率、真实输出 token、延迟与峰值显存。因其官方评估脚本默认 8 卡，单 L40 连加载是否成功仍是第一道门。
2. **LocateAnything-3B RGB 评估**：先以 worker 的 `ground_multi` 做直接 referring 对照，并单独以 `detect` 做候选。L40 先用 `la_flash`，batch 1 开始逐步增大；记录 MTP 回退到 NTP 的比例、解析格式、候选数和 `OracleRecall@1/5/10`。如果候选 oracle 在当前 RGB 错例上没有明显余量，停止候选重排支线。
3. **Rex-Omni 按需**：只有当 LocateAnything 的候选对类别/小目标明显漏检，或需要第二个候选来源时，才以同一名词集合跑 Rex；合并候选后仍先测 oracle。不训练 Rex，也不将其外部检测基准混入本赛 ACC 表。

这条顺序把两个最有信息量的“直接定位专家”和“候选能力”拆开，避免同时更换模型、token 长度、解析器与三模态融合而无法归因。最终是否纳入 ensemble（集成）以固定选择/融合规则在同一验证集上的实际成对净 `ACC@0.5` 增益为准，而非单模型分数；单分较低的外部模型若能补足当前强 RGB 的错误，仍可能产生净增益。

## 一手来源清单

- [EGM 官方仓库](https://github.com/NVlabs/EGM)、[EGM 论文 HTML v3](https://arxiv.org/html/2601.13633v3)、[EGM-8B 模型卡](https://huggingface.co/nvidia/EGM-8B)。
- [EGM SGLang 评估启动](https://github.com/NVlabs/EGM/blob/main/verl/scripts/sglang_infer.sh)、[EGM 推理/解析实现](https://github.com/NVlabs/EGM/blob/main/verl/scripts/sglang_infer.py)、[EGM GRPO 启动](https://github.com/NVlabs/EGM/blob/main/verl/scripts/grounding_qwen.sh)。
- [LocateAnything 官方 README](https://github.com/NVlabs/Eagle/blob/main/Embodied/README.md)、[模型页](https://huggingface.co/nvidia/LocateAnything-3B)、[worker 实现](https://github.com/NVlabs/Eagle/blob/main/Embodied/locateanything_worker.py)、[并行/混合生成实现](https://github.com/NVlabs/Eagle/blob/main/Embodied/eaglevl/utils/locany/generate_utils.py)。
- [Rex-Omni 官方仓库](https://github.com/IDEA-Research/Rex-Omni)、[wrapper](https://github.com/IDEA-Research/Rex-Omni/blob/master/rex_omni/wrapper.py)、[task templates](https://github.com/IDEA-Research/Rex-Omni/blob/master/rex_omni/tasks.py)。
- 本地竞赛规则：[赛题 PDF](../../contest/基于大模型的多模态视觉理解与推理.pdf)，第 4、6 页。
