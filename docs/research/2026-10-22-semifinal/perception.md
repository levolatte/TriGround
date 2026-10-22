# 半决赛视觉感知、分辨率与定位模型调研

2026-10-22；只做文献、源码和仓库既有实验审阅，未训练或下载权重。比赛任务、评价和输入约束依据本地 `contest/基于大模型的多模态视觉理解与推理.pdf`；项目现状及数字依据 `code/docs/research/2026-10-22-takeover/` 三文档及其指向的原始实验。以下外部基准均不能替代 AIC ACC@0.5。

## 决策

1. **先做同一基座的分辨率/原生局部信息配对实验，再独立试新基座**。当前三图直接输出框的已验证强提交路径用 Qwen3.6-27B、2 倍 LANCZOS 放大、每图 `max_pixels=8,294,400`、总上下文 32,768（`code/tools/predict_submission_vllm.py:67-113,143-159,269-277`）。优先比较RGB高分辨率、IR/D低分辨率。裁剪与插值均不能创造未采样的信息，裁剪可能改变已有细节的token分配；预测框决定裁剪范围会强化错实例，故固定重叠局部图块排在非对称分辨率之后。另以 Qwen3.8-27B 原生三图零样本作为独立基座对照，不能把 3.6 LoRA 迁过去直接解释为模型升级。
2. **Rex-Omni 是最值得试的专用定位模型，但应作为直接框的独立一臂**。它的自回归四坐标定位不受旧候选池 339/412 覆盖上限束缚，也不要求 Qwen 在候选编号间选择；官方代码有 vLLM 推理。先独立跑RGB原图与完整原Query的单图零样本，再将短语化作为单独消融，不追加旧多步 Agent。论文与模型许可证均需在正式参赛前确认适用范围。
3. **Grounding DINO / SAM 3 / Florence-2 暂作诊断或候选源**。Grounding DINO 的核心任务是短语条件检测，关系与序数选择不是其强项；SAM 3 官方是短概念短语的全实例分割。若要接入，先看候选 oracle 覆盖和 *同一个* 直接框基线上的净纠错，不能再把候选覆盖误当最终分数。

## 论文一：Grounding DINO 与源码

论文 [Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection](https://arxiv.org/abs/2303.05499)，阅读 arXiv v5 完整 33 页；源码 [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO/tree/856dde20aee659246248e20734ef9ba5214f5e44)，本地浅克隆 commit `856dde20aee659246248e20734ef9ba5214f5e44`，Apache-2.0 [`LICENSE`](https://github.com/IDEA-Research/GroundingDINO/blob/856dde20aee659246248e20734ef9ba5214f5e44/LICENSE)。论文第 5–9 页和第 12–14 页是模型与 REC（指代表达理解）结论；第 26 页是吞吐表。

模型用图文特征增强器、语言引导的 900 个视觉查询选择、跨模态解码器，将开放词表文本与图像区域对齐。它不是以复杂指代表达做单框决策为主的模型；原论文不加 RefCOCO 训练的 REC 与专门训练版本有明显差距。论文的 COCO/RefCOCO 数字与本赛三模态长 Query 不等价。官方 [预处理与推理源码](https://github.com/IDEA-Research/GroundingDINO/blob/856dde20aee659246248e20734ef9ba5214f5e44/groundingdino/util/inference.py#L39-L97)：输入通常短边 800、长边上限 1333，按框/token 阈值输出 `cxcywh` 归一化框；[模型 forward](https://github.com/IDEA-Research/GroundingDINO/blob/856dde20aee659246248e20734ef9ba5214f5e44/groundingdino/models/GroundingDINO/groundingdino.py#L227-L349) 输出 `pred_boxes`，再做坐标变换。其 Swin-T 论文表列 172M 参数、464 GFLOPs、8.37 FPS，硬件/图像条件不能直接推到当前集群。官方仓库的权重需单独确认许可证。

项目历史已测试 Grounding DINO 等构候选：C 基线 292/412；候选选择 273/412（修 10、毁 29），再局部细化 221/412（再修 1、毁 53）；新 Agent 未训练 217/412（修 11、毁 86）。候选池初始 oracle 覆盖 339/412，仍有 73 道池外目标全失败，搜索仅多覆盖 1 道。因而再接另一个检测器，必须证明它**独有覆盖 C 错题**并能被下游稳定选中，不能只看 mAP 或 oracle。历史 `code/docs/research/2026-10-22-takeover/experiments-archive.md:24-62` 已按源码与逐题结果核对；未评价的居中裁剪修正版不能冒充 221 的实测改进。

## 论文二：Rex-Omni 与源码

论文 [Detect Anything via Next Point Prediction](https://arxiv.org/abs/2510.12798)（Rex-Omni），阅读 arXiv v1 的 48 页（方法、数据、训练、评估、消融与附录实验第 1–32 页完整；其余参考文献及可视化附录已浏览）。源码 [IDEA-Research/Rex-Omni](https://github.com/IDEA-Research/Rex-Omni/tree/6508981c1e0c3fbb2dbe7b962a4bb745005f3e2e)，本地浅克隆 commit `6508981c1e0c3fbb2dbe7b962a4bb745005f3e2e`。其 [IDEA License 1.0](https://github.com/IDEA-Research/Rex-Omni/blob/6508981c1e0c3fbb2dbe7b962a4bb745005f3e2e/LICENSE#L7-L13) 第 9 行只授权**非商业研究**；README 第 254 行另说明基座 Qwen2.5-VL-3B 的 Qwen Research License。竞赛能否用，必须按官方比赛及许可证确认，不能默认等同 Apache-2.0。

论文第 4–6 页：基于 Qwen2.5-VL-3B，将词表末尾 1000 个 token 用作 `<0>`…`<999>` 坐标，逐目标自回归生成 `x1,y1,x2,y2`，对当前单框任务是直接输出，不必由外部检测器先限定可选框。第 7–10 页：22M SFT 数据中约 3M grounding，目标短语来自 Qwen2.5-VL 描述、spaCy 名词短语与检测伪标注；修饰词常被过滤以适配 DINO-X，复杂关系原句训练支持因而不应高估。第 11–13 页：66k 样本 GRPO；表中 8×A100 约 24h，而 SFT 是 64×A100 约 8 天，不能视为本项目可轻易复现。第 14–18 页 RefCOCOg 与 COCO/HumanRef 成绩显著，但任务、提示、数据及指标均不与 AIC 可比。第 26–31 页的消融表明 GRPO 主要修重复、遗漏和过大框，坐标精度改善较小；A100/vLLM 少目标输出约 2 秒，密集数百框可 >16 秒，AIC 单目标需另测。

官方 [wrapper](https://github.com/IDEA-Research/Rex-Omni/blob/6508981c1e0c3fbb2dbe7b962a4bb745005f3e2e/rex_omni/wrapper.py#L23-L180) 提供 Transformers/vLLM 两个 backend；`:30-31` 默认为 `min_pixels=16*28²`、`max_pixels=2560*28²≈2.01M`，`:449-520` 做 28 倍数 resize、生成与解析；`:646-740` 构提示、以 `process_vision_info` 和 chat template 进入 vLLM。正式对照须记录实际像素网格，不要假设当前 8.29M 图会原样进 Rex。官方 [parser](https://github.com/IDEA-Research/Rex-Omni/blob/6508981c1e0c3fbb2dbe7b962a4bb745005f3e2e/rex_omni/parser.py#L34-L130) 把 `/999` 坐标乘输入宽高；[训练 grounding 转换](https://github.com/IDEA-Research/Rex-Omni/blob/6508981c1e0c3fbb2dbe7b962a4bb745005f3e2e/finetuning/dataset/task_fns/grounding_task.py#L67-L136) 将 xyxy 离散到 0–999。该 parser 没有保证 `x1<x2,y1<y2` 或限制越界：输出适配本赛时应在最终提交边界验证，而非盲接。

一个需要先核实的源码/论文差异：论文称 GRPO 抑制重复框，但公开 [reward_func.py:197-224](https://github.com/IDEA-Research/Rex-Omni/blob/6508981c1e0c3fbb2dbe7b962a4bb745005f3e2e/finetuning/verl/configs/reward_func.py#L197-L224) 对每个 GT、每个预测分别独立取最大 IoU；**两个相同正确预测框对一个 GT，precision 和 recall 仍可同为 1**，该函数本身不罚重复。这是对开源训练脚本的可复算判断，不推断论文实际训练一定用了同一实现。`reward_func.py:81-85` 的归一化是 `/1000`，与推理 parser `/999` 有 0.1% 量级差异；跨阈值样本须按真实脚本检查。

## 分辨率与局部区域：信息、token 和坐标

项目在 `upscale_images` 中先把每个 1920×1080 图用 LANCZOS 放大 2 倍，再把 `max_pixels` 同步调到 8,294,400。**插值图是原图的确定性函数，不能恢复未采样到的真实细节。** 它可能提分，是因为后续视觉 patch 的尺度、抗混叠/边缘频率、相对目标占据的 token 数、注意力分配和坐标量化误差都变了。处理器在 `[min_pixels,max_pixels]` 内不主动放大；只调上限无效，放大而不调上限则被缩回。这是 [项目代码](../../../tools/predict_submission_vllm.py) `:67-113` 可直接复查的事实。

当前 Qwen3.6-27B 官方 [`config.json`](https://huggingface.co/Qwen/Qwen3.6-27B/blob/main/config.json) 与本地下载配置表明 `model_type=qwen3_5`，视觉 `patch_size=16, spatial_merge_size=2`；[Transformers Qwen3VL processor](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/qwen3_vl/processing_qwen3_vl.py#L79-L81)按视觉网格除以 merge² 展开占位，[Qwen2-VL image processor](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py#L56-L81) 的`smart_resize`默认签名为28，但本模型处理器按patch16×merge2传入**32**，实际尺寸按32对齐并受像素上限约束。因此粗估有效 LLM 视觉 token ≈像素/`(16×2)²`：单图原生约 2,025，三图约 6,075；2 倍后单图约 8,100、三图约 24,300（实际以 `image_grid_thw` 为准）。相同总图token的局部tile可以改变小区域的token分配，但仍需resize，不能保证增加真实信息，全球关系则由低成本全图保留。2 倍图还使视觉预填充/内存与空间 token 近 4 倍，32k 上下文剩余空间少；GPU 峰值、延迟必须实测，不可将参数量或论文 FPS当显存预测。

**最小可证伪对照**：以当前27B、2×三图为冻结A，先做B1（RGB2×、IR/D1×），这是减少辅助预算；再做B2（把节省的预算转给RGB），这是预算重分配。统一Query、图组、权重、输出协议，记录实际网格/视觉token及全分母改对改错。仅增大上限但原图未触顶的变化已知是空操作，不浪费完整模型评估。独立Qwen3.8零样本与3.6零样本并列筛选，再与微调Arm 2比较实际价值。City412只作历史诊断，独立结论用未参与旧选模的合法外部图组。

局部图块后置：优先固定覆盖、保留全图，不由GT选图块；若仍研究预测驱动裁剪，须把错误实例自我强化列为首要风险。所有tile-local坐标按真实整数裁切边界映射，不能与全图归一化协议混用。历史旧8B裁剪细化221/412包含坐标协议问题，但不能把其全部退化归因于坐标：裁掉关系背景、搜索覆盖不足与错误目标同样可能造成失败。新路线要对当前27B重新建立证据。

## 备选模型的使用边界

| 模型/来源 | 已核对的能力和工程状态 | 本项目排序 |
|---|---|---|
| [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | 官方多模态、Apache-2.0，模型卡有 vLLM 图像请求。官网 `config.json` 与 Qwen3.6-27B 的文本/视觉配置及图像预处理配置逐字段相同，除 Transformers 版本元数据；这只说明推理代码易适配，不证明 3.6 LoRA 可迁移。 | P0：独立同提示零样本，不与旧 LoRA 混为一组。 |
| [Rex-Omni](https://huggingface.co/IDEA-Research/Rex-Omni) | 3B 直接坐标模型，官方 vLLM/Transformers wrapper，许可证非商业研究。单图为主、默认像素上限远低当前三图 2×，长关系/多模态需实测。 | P1：完整Query的RGB独立臂，短语化另作消融，先测合法框与长 Query 失效；再考虑蒸馏或其训练代码。 |
| [SAM 3](https://github.com/facebookresearch/sam3) | [论文](https://arxiv.org/abs/2511.16719) 和官方 README 是短名词概念的全实例分割/检测，848M 参数，支持文本/盒提示；[SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE)，权重取得另有门槛。不能默认懂“第二个/左边的人背后”。 | P2：查 C 错题中独有候选覆盖；只有覆盖和选择均有净增才接。 |
| [Florence-2-large](https://huggingface.co/microsoft/Florence-2-large) | MIT，模型卡提供 `CAPTION_TO_PHRASE_GROUNDING`、`REGION_PROPOSAL` 任务及 xyxy 输出；官方 [论文](https://arxiv.org/abs/2311.06242) 为多任务表示，不是三模态关系单框专用。官方卡示例需 `trust_remote_code`，对当前 vLLM 主路径不能假定无缝。 | P2：轻量诊断/候选补漏，与 DINO 重叠率先测。 |
| [InternVL](https://github.com/OpenGVLab/InternVL/tree/main/internvl_chat) | 官方 README/代码使用 448² 动态 tiles，高分辨率思想可借鉴，但替换整个大模型与训练链成本高。 | P3：借鉴 tile 设计，不先移植模型。 |
| [Gemma 4 31B IT](https://huggingface.co/google/gemma-4-31B-it) | 官方模型卡确认 31B、图像输入、Apache-2.0；不同处理器/提示协议。 | P2：若 Qwen3.8 不增，再作独立零样本对照。 |
| [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | 官方卡称多模态、MIT、vLLM，但约 321B 总参数/18B 激活，参数驻留和部署资源远大当前 27B。 | P3：当前资源条件不优先，不能因“Flash”误判成小模型。 |

这些路线均保留现有直接框 Baseline 与逐题预测，评分用相同原浮点 GT、相同 ACC@0.5 和图组切分。模型基准分数、候选 oracle、无效框率、显存与实际吞吐分别记录；不能相互替代。
