# 强监督训练、关系专家与 SFT→RL 接入（2026-10-23）

## 当前选择

优先直接使用 **nvidia/EGM-8B 与 nvidia/EGM-8B-SFT**。前者作为独立的 RGB 关系定位专家进入候选缓存，后者作为强化学习冷启动或小量域适配起点。两个 Hugging Face 模型 API 本轮均实际返回 200、gated=false，并列出完整模型文件。这样可以利用已经完成的大规模监督训练，而不在昂贵的单张 96GB 云卡上重做一遍。较小的 nvidia/EGM-4B 也真实存在；不能仅因参数更少就假定其对 AIC 小目标更差或更好。

本地已实现数据准备和两个明确训练入口，但未启动 GPU。100 步默认值是检查一条训练臂能否正常学习的低成本起跑值，不是声称已经检验“大规模 SFT”。真正检验规模必须增加独立训练图组和有效监督任务，并按完整 epoch 或明确呈现预算运行。

## 既有证据不能排除什么

- Arm 2 的当前最佳记录是 ck500、k2.0、官方记录 0.8012；这次适配训练每步 8 条，500 步约 4,000 次呈现，视觉塔与连接层冻结。不能把呈现数当独立新 Query 数，也不能把这次任务微调等同于基座全部训练历史。
- Arm 4/5 同时改变数据、深度处理或训练范围；Arm 5 k2.0 没有本地可核实的官方实测成绩。其一次失败不能单独证明“视觉适配无用”或“扩充 SFT 无用”。
- 旧 G/U 的新增监督仅 98 个 RGBDT 图组/不同 Query，181 是分模态输出任务数。重复小池的负结果不等于十万级真实指代表达训练的负结果。
- 本项目真正需要改变的预测决定是：同类竞争实例的身份选择，以及低光/小目标的视觉区分。只增加普通对象框、重新措辞或给几十条示例反复加权，不能自动补上这两种能力。

以上来自 `code/docs/research/2026-10-22-takeover/current-pipeline.md`、`code/docs/audit-20260927/DATASETS.md` 与实际训练脚本；本轮未复算官方成绩。

## 一手研究筛选

| 工作 | 已核实内容 | 本项目裁决 |
|---|---|---|
| [EGM，2026](https://arxiv.org/html/2601.13633v3) | Qwen3-VL-8B 的 Base→SFT→RL 为 87.8→88.8→91.4；是 RefCOCO 系列八切分平均成功率，正文阈值为 IoU>0.5，摘要的“IoU”措辞不能当 mIoU。论文训练池 575,208 样本、24,407 图，针对复杂表达与竞争实例。 | 最接近本任务且有现成专家；AIC 用自己的 IoU≥0.5 重新评分。论文本身支持 SFT 和 RL 分别有收益，不支持只做 SFT 足够。 |
| [EGM 官方工程](https://github.com/NVlabs/EGM) | 提供 8B/4B、SFT 训练与 RL 接口；完整推理需保留思考。公开完整 CoT 数据不是 575K，README 明示仅 1K 示例。 | 优先预训练好的 SFT/RL 权重；不把 1K 示例当完整大训练集。 |
| [Rex-Omni](https://arxiv.org/abs/2510.12798)；[官方微调工程](https://github.com/IDEA-Research/Rex-Omni/tree/master/finetuning) | 3B 基座经过 22M SFT 再 GRPO；检测、指代、视觉提示等任务共享训练。官方微调样例只有 1K，TSV 包含图像、boxes/phrase，支持负类。 | 利用成品权重作为异构感知教师或框候选；没有证据表明其完整 22M 池已开放，不能规划全量直接下载。 |
| [DiG，CVPR2026](https://github.com/gulizhoutao/DiG) | 已公开 Blender 差异图生成与 EasyR1 训练。真实方法是差异定位代理任务+分阶段 GRPO，包含增删、颜色、大小、位置改变。 | 是改变细粒度感知的结构性任务，不是现成强 SFT 配方；移交 RL 方向作后备，当前不为了做“不同结构”而先搭 Blender 全链。 |
| [TreeVGR，ICLR2026](https://github.com/Haochen-Wang409/TreeVGR) | 发布 TreeVGR-SFT-35K、RL-37K 和模型；研究可追踪视觉证据与推理。 | 可补中间实体/关系证据，但它的 VQA/框联合目标不同于 AIC 单目标框；只选能映射为明确单目标定位的数据。 |
| [GroundingSuite](https://github.com/hustvl/GroundingSuite) | README 宣称 9.56M 训练样本，但公开下载入口主要指向评测，训练数据请求 [issue #3](https://github.com/hustvl/GroundingSuite/issues/3) 仍未给出可核实完整训练包。 | 不列入近期可直接开跑的数据量。 |
| [Ferret](https://arxiv.org/abs/2310.07704)；[Groma](https://github.com/FoundationVision/Groma) | 大量区域指代/定位、多任务和困难负例提供成熟先例；Groma还提供不同训练阶段检查点。 | 支持“任务覆盖和区域对齐值得扩大”，但完整复制老模型架构相对直接用 EGM/Rex-Omni 成本更高。 |
| [RGBT-GroundBench](https://github.com/crazyxiaoxi/RGBT-GroundBench)；[RoboRefIt](https://luyh20.github.io/RoboRefIt.github.io/) | 前者真实 RGB/热红外+Query+RGB框，后者真实 RGBD 指代。 | 两者直接覆盖传感器域适配，优先于没有 Query 的 RGBDT 追踪帧。已有转换器可复用。 |

## 读过的重要工程细节

1. `NVlabs/EGM/sft/qwen-vl-finetune/scripts/sft_qwen3_8b_grounding.sh` 是冻结视觉塔、训练语言模型和连接层的全参数方案，不是语言 LoRA；脚本设 1 epoch、lr=1e-5、8卡×8×2 全局批量128。`max_pixels=50176` 很低，不应复制到 AIC 小目标。此次只迁移方法与已训练权重。源码：[SFT脚本](https://raw.githubusercontent.com/NVlabs/EGM/main/sft/qwen-vl-finetune/scripts/sft_qwen3_8b_grounding.sh)。
2. EGM 的 `sft/data_generation/convert_gpt_to_qwen3_sft.py` 产出 `image` 字符串、`conversations`，assistant 为推理文本+`</think>`+带 `bbox_2d` 的 JSON 列表；不是项目原有 `image` 列表结构。若自己生成 ms-swift 推理监督，采用明确的 `<think>...</think>` 和最终答案，不照抄依赖上游模板补开标签的格式。源码：[转换器](https://raw.githubusercontent.com/NVlabs/EGM/main/sft/data_generation/convert_gpt_to_qwen3_sft.py)。
3. EGM `verl/examples/data_preprocess/grounding_all.py` 读取 conversations、sent、image、height/width，将 `<box>` 目标变为 `reward_model.ground_truth`，只在用户提示里放 Query。其源码注释写 IoU 选样，但该文件实际主体只是转换，不应声称它已完成困难样本筛选。源码：[RL预处理](https://raw.githubusercontent.com/NVlabs/EGM/main/verl/examples/data_preprocess/grounding_all.py)。
4. 当前 [ms-swift 文档](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html) 已有 `lora_llm` 新语义，但仓库历史环境是 4.5.3。此次读取 [v4.5.3 tuner_args](https://raw.githubusercontent.com/modelscope/ms-swift/v4.5.3/swift/arguments/tuner_args.py)，确认 all-linear 下 freeze_vit/aligner 控制是否添加 LoRA，不能用最新文档选项冒充历史已支持。新脚本仅使用现有常规 LoRA 配置。

## 本机实际数据与权重库存

数量由本轮 Python 读取文件/标注统计；本机与旧云端分开。

| 来源 | 本机实际量与路径 | 现在能否直接训练 |
|---|---|---|
| RoboRefIt 已取子集 | `F:/AIC/data/external/RoboRefIt/subset/converted/train_depth_visual_fixed_p01_p99.jsonl`：2,000 条/2,000 RGB图组；对应 RGB+原深度共4,000图，预渲染深度另在 `subset/depth_visual_fixed_p01_p99/`。 | **可以**。本轮实际生成 1,800训练+200留出。保留原有 `depth_policy=visual`、`depth_unit=unknown`，不虚构毫米含义。 |
| RoboRefIt 全训练标注 | `.../raw/final_dataset/train/roborefit_train.json`：36,915条、7,929不同RGB路径；raw目录实际仅10张图。 | 只能先用已取得的2K；全标注不能当全图已下载。其他36K应补媒体后再训练。 |
| RGBT-GroundBench | `.../rgbtvg_flir/*train.pth` 7,000；m3fd 3,604；mfad 16,000，总26,604。`image_data/`本机只有20图。 | **本机全池不就绪**。旧HANDOFF记云端6K已解包，但需待云机恢复后检查原路径，避免重下。 |
| RGBDT500 | `F:/AIC/data/external/RGBDT500/Train.zip` 26,447,029,453字节、14,000归档项；400训练序列×10个标注帧。 | 框和三模态在包内，但没有原生语言Query；不能把4K框直接算4K指代任务。已有181任务只来自98独立图组。 |
| City | 仓内快照记录3,707训练Query/681图组、412开发Query/78图组；完整媒体未随代码保存。 | 清单能读，媒体另查；长期反复使用City412，不能视为新独立评测。 |
| RefCOCO/+/g | 本机只查到 `tools/prepare_refcoco.py` 与测试，无完整数据。 | 转换器可复用，仍需真实COCO媒体。无需下载TB级资源。 |
| 当前最优适配器 | `F:/AIC/deliverables/作品材料/模型文件/adapter_model.safetensors` 933,974,032字节，语言LoRA。 | 本地有适配器；Qwen3.6-27B基座未随材料打包。 |

RoboRefIt 的 scene_id 按原清单是**每张图的身份**，1,800/200 图组隔离不等于物理布景、对象或拍摄序列严格独立。将来获取真实序列标识后应以该标识分组。没有把目录名如 chair 这种大场景类别误当唯一拍摄序列。

## 可获取数据的真实格式和最短转换

- **RoboRefIt**：本地原始 JSON 每行有 `text,bbox,rgb_path,depth_path,scene,class`，bbox 是像素 xyxy；原图路径带 Windows 反斜线。已有 `tools/prepare_roborefit.py` 完成转归一化 xyxy 与路径解析。本轮直接复用已渲染 JSONL，避免重复处理深度。
- **RGBT**：`.pth` 每项前四位为 `[filename,{height,width},[x,y,w,h],query]`，框是像素 xywh；`tools/prepare_rgbt_groundbench.py` 输出 `rgb,aux,aux_type='ir',query,bbox,scene_id`。新准备器支持这一形态。公开 [HF仓库](https://huggingface.co/datasets/JiawenXi/RGBT-Ground-Dataset) API确有 `data_flir.tar,data_m3fd.tar,data_mfad.tar` 与三标注tar，未见gated；优先复用云端已下载6K，再按需要补源。
- **RefCOCO**：已有转换器从 `jxu124/refcoco,refcocoplus,refcocog` parquet 读 `bbox,raw_image_info,sentences`，必须用 raw_image_info 的原始 COCO 文件名，不能把裁剪图文件名拼到全图目录。每表达转一行。该转换器约定这些镜像的 bbox 是 xyxy；不是泛化为所有 RefCOCO 原始标注都为 xyxy。
- **EGM**：HF API实际列出 `vanilla_grounding_reasoning_training_dataset_raw.jsonl`、`...cot_subset.jsonl`、`metadata/train/refcoco_train_egm8b.jsonl`、训练/验证parquet和COCO分卷。只需要格式/少量样例时用 allow_patterns 定点下载；不要不加过滤下载整个仓库。公开文件字段由作者转换源码核对，本轮部分 HF 原始内容请求网络重置，**未冒称已逐条审阅或核算全文件条数**。
- **TreeVGR**：HF API实际列出 `TreeVGR-SFT-35K.json`，卡片为Apache-2.0；并不是内嵌全部媒体包。需核对引用图源和任务后才混入，当前不把35K算进本机可用量。

## 两个保留的大训练臂

### S1：扩大真实指代覆盖，联合语言/视觉 LoRA 适配

**机制假设**：现有小池语言LoRA能记住任务输出，但没有足够独立实例监督来适应热红外、深度可视化与细粒度目标区分。扩大真实监督，同时让视觉编码器/连接层能够变化，可能提高新场景实例辨别。

完整数据目标：最多26.6K RGBT +36.9K RoboRefIt +约30–50K RefCOCO关系/属性样本，按真实可取得图像和去除评测交集后的数量计算；保留City目标域训练回放。规模是计划，不是本机已有池。首个完整配方可用“目标域回放30%、RGBT30%、RGBD20%、RGB指代20%”的呈现预算；不是数据来源天然等权，更不是同义改写凑十万。先保证每源多实例、尺度与句长覆盖，完整Query不截断到100字符。

从同一个完整SFT/base checkpoint，各开新LoRA：`ARM=language` 与 `ARM=vision_language`，相同数据、步数、分辨率与种子。两个臂分别冻结/开放视觉塔及连接层LoRA。先用现成2K数据和100步定位实现/遗忘问题，数据补齐后用 `MAX_STEPS=-1 NUM_TRAIN_EPOCHS=1` 检验完整配方，或设置相同明确总呈现数。100步不会用于得出“强SFT无效”的结论。

这个比较把**新数据是否有效**与**视觉训练范围是否有增量**分开：先对照同基座的语言臂，再看视觉臂；与现有最佳27B最终比较另报。若只是训练源拟合改善、独立外部图组和目标域配对净翻转下降，应否定该配方或处理明显领域分布失衡，不能继续用训练损失下降解释成功。

### S2：现成关系专家冷启动→困难但可学样本→RL

**机制假设**：复杂关系失败主要来自遗漏约束/参照物，而非所有候选不可见。EGM已具备关系推理形式，让它做冷启动或教师，比用小样本从通用27B重新教长推理更划算。

优先直接比较 EGM-8B-SFT、EGM-8B 与原生27B的逐题翻转，得到专家互补事实。训练仅使用合法外部训练图组：将基线失败但EGM/Rex-Omni正确的例子与基线已正确例子混合；框标签仍用数据GT，教师只提供可验证的身份/关系解释。对象属性或关系不受标注支持时，不把“教师最终框正确”偷换成中间解释已真。使用本地开源模型生成，不调用闭源在线API。

若采用额外SFT：先准备10–30K不同目标/场景关系例，短证据加最终框，混入普通直接框回放；保持短框臂和推理臂可分开评价。随后接本项目GRPO，直接优化最终框的IoU与命中阈值。若预算紧，**跳过重复SFT，从EGM-8B-SFT直接启动RL**，已有SFT完成态本身就是冷启动；EGM-8B则是更强但已RL过的对照/专家。

多模型蒸馏只对外部训练样本做：教师一致且与GT相符可作为优先监督；分歧样本保留并按真实GT判断，不能把一致性当真值。先用专家缓存确认有净救回能力，再决定是否值得蒸馏；无需第一轮搭复杂门控网络。

## 已完成的本地接入

新增且只改这三个业务文件：

- `code/tools/prepare_semifinal_sft.py`：支持项目归一化框清单、native conversations、swift messages；保留原query、原推理答案、真实图片顺序，输出 `messages/images/id/source/scene_id/query/modalities/bbox`。已有source/scene_id优先，没有scene_id时使用该来源的完整RGB路径。没有虚构黑模态、没有按不同源的同名文件全局去重。权重是来源池的呈现配额，报告另外给唯一ID与图组数。
- `code/scripts/train_grounding_sft.sh`：4.5.3常规LoRA两臂，默认100步，支持完整MODEL、ADAPTER仅载权重续训、RESUME恢复完整训练状态、DRY_RUN打印参数。视觉臂拒绝直接加载旧语言ADAPTER，因为其旧target_modules可能阻止新增视觉LoRA；需要完整/已合并MODEL和新LoRA。
- `code/tests/test_semifinal_sft.py`：覆盖图组隔离、精确配比/复现、来源命名空间排除、官方结构精确路径排除、原深度显式渲染、原推理答案保持、双图语义、禁止把val输入作为训练源。

实际输出：`F:/AIC/tmp/structural_20261023/sft_data/train.jsonl` 1,800条、`val.jsonl` 200条，`summary.json` 有统计。相对路径已转当前电脑绝对路径，上传云端后必须重新映射或在云端重新执行准备器，不能把F盘路径直接送Linux训练。

```powershell
# 已实际运行；2K现成RGBD图组，保持原可视深度
.\.venv\Scripts\python.exe tools/prepare_semifinal_sft.py `
  --source roborefit=../data/external/RoboRefIt/subset/converted/train_depth_visual_fixed_p01_p99.jsonl `
  --val-ratio 0.1 --output-dir ../tmp/structural_20261023/sft_data
```

```bash
# 云机恢复后才执行；当前只核查DRY_RUN。MODEL也可设Qwen3.6完整基座。
MODEL=/models/EGM-8B-SFT DATASET=/data/train.jsonl \
OUTPUT_DIR=/runs/sft_language ARM=language MAX_STEPS=100 DRY_RUN=1 \
bash scripts/train_grounding_sft.sh

# 同基座、同数据、同预算的视觉训练范围对照
MODEL=/models/EGM-8B-SFT DATASET=/data/train.jsonl \
OUTPUT_DIR=/runs/sft_vision ARM=vision_language MAX_STEPS=100 DRY_RUN=1 \
bash scripts/train_grounding_sft.sh
```

混合示例：`--source city=/data/city_train.jsonl --source rgbt=/data/rgbt_train.jsonl --source roborefit=/data/robo_train.jsonl --weight city=3 --weight rgbt=3 --weight roborefit=2 --train-samples 32000`。实际权重按需要改；准备器不会把短源先耗尽当训练结束。

排除清单可用 `{"source":"roborefit","scene_id":"原场景ID"}` 或 `{"source":"roborefit","id":"某题ID"}`，后者会排除该题所属整个图组；也可读已有开发/官方JSON的真实RGB路径。搬过盘的同图必须用明确的source+scene映射排除，脚本不会凭basename猜测。没有哈希与自动相似查重。

8项针对性CPU测试通过，脚本两臂参数打印和bash语法通过；未做GPU训练、未宣称显存或AIC收益已验证。Robo图组读取有实际磁盘开销，应在开云机前完成CPU准备。

## 合规与下步交接

依据 `contest/基于大模型的多模态视觉理解与推理.pdf` 第7–8页：外部公开训练数据不设范围限制，但需注明来源/处理；官方初赛、复赛、半决赛数据只能推理，不得训练或人工标注；禁止商业闭源在线推理API。现成公开训练数据曾使用商业教师不等同于本项目可以再调用该接口。所有自建教师流程仅用合规开源模型和外部训练图。

下一步只需主线程把同一 `val.jsonl` 接到Qwen/EGM/Rex候选缓存与ACC评价；RL分支复用训练清单并保留GT作为奖励侧数据。恢复云机后先核云端旧RGBT6K和模型库存，做专家零样本/已有SFT评估，再根据配对净救回决定S1或S2预算。完整研究目标保留，昂贵计算不因准备器已完成而自动启动。
