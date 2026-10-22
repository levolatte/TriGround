# 半决赛方案如何接入现有训练链

2026-10-23。这是源码层面的接入设计，尚未实现辅助任务/奖励插件或运行GPU实验。当前项目训练脚本使用历史ms-swift4.5.3；本轮阅读上游固定commit `d5253cbc5fe3258f143d1826d4078471fa0fc5dc`，不能默认新接口与历史安装完全相同。保留原环境，实验环境先确认版本与模型模板，不无条件升级主线。

## 当前可直接复用的边界

| 项目文件 | 职责 | 半决赛改动位置 |
|---|---|---|
| `tools/prepare_qwen3vl_native_sft.py` | 模态提示、深度渲染、0–1000框编码 | 复用图像顺序/坐标约定；新增候选判别样本生成器，独立于原普通样本 |
| `scripts/train_qwen36_27b_swift.sh` | 27B语言LoRA、冻结视觉塔/连接层 | 继续作为SFT起点，首先只换数据任务混合，不同时解冻视觉塔 |
| `tools/predict_submission_vllm.py` | 三图预处理、vLLM、短框解析与写回 | 第一轮继续直接框；非对称缩放按模态指定，不改RGB坐标空间 |
| `tools/verify_acc05_dev.py` | 框合法性及归一化IoU | 评分与新奖励共用该约定，`>=0.5`，不使用像素面积`+1` |
| `tools/prepare_roborefit.py`、`tools/depth_relation_tasks.py` | RGBD表达转换、原始深度统计 | 优先复用数据接口与已有深度判据，核对各源物理单位 |

## 判别辅助任务：先用现有多任务SFT

输入仍是`messages`和顺序一致的`images`列表。直接框样本保留；候选判别样本在用户提示中增加RGB全图归一化候选A/B，assistant只输出固定标签`A/B/both/neither`。GT仅用于生成训练标签，不放进模型可见的提示。候选顺序随机，正负覆盖多个来源与独立场景。

概念损失是两项分别平均后的`L_box + λ L_choice`。最小实现可使用交替任务批次及明确任务权重；若仅混合JSONL而沿用默认token平均，应计算实际监督token权重，不能声称已精确实现该式。确认当前ms-swift已有loss扩展/缩放接口后再决定是否需要很小的插件，避免另造训练器。记录每任务样本/视觉token/监督token总数和训练实际权重。

上游[Qwen模板](https://github.com/modelscope/ms-swift/blob/d5253cbc5fe3258f143d1826d4078471fa0fc5dc/swift/template/templates/qwen.py)中，`Qwen3VLTemplate._encode`（约560–605行）遍历多张images，以`image_grid_thw/merge_size²`扩展视觉占位；Qwen3.5与3.8注册项（约710–735行）使用不同模板类。多图输入链可以复用，但不能因为模型config相似就跳过chat template（对话模板）检查。多任务训练也不等于推理要多轮对话。

候选判别标签精确取决于每个候选对该Query真框的IoU。构造时保留原浮点GT；若候选编码为0–1000整数，先还原实际模型看到的框再算标签，避免编码前后在0.5附近标签相反。both/neither是真实标签，不是解析兜底。

## GRPO：复用上游扩展接口，不搬旧训练器

已读[官方奖励文档](https://github.com/modelscope/ms-swift/blob/d5253cbc5fe3258f143d1826d4078471fa0fc5dc/docs/source_en/Instruction/GRPO/DeveloperGuide/reward_function.md)、[插件示例](https://github.com/modelscope/ms-swift/blob/d5253cbc5fe3258f143d1826d4078471fa0fc5dc/examples/train/grpo/plugin/plugin.py)、[GRPO训练器](https://github.com/modelscope/ms-swift/blob/d5253cbc5fe3258f143d1826d4078471fa0fc5dc/swift/rlhf_trainers/grpo_trainer.py)以及[Qwen加载器](https://github.com/modelscope/ms-swift/blob/d5253cbc5fe3258f143d1826d4078471fa0fc5dc/swift/model/models/qwen.py)。

接线约定：

```text
数据：messages + images[RGB, IR, D] + solution[原浮点RGB xyxy]
插件：继承 swift.rewards.ORM
调用：__call__(completions, solution, trainer_state, **kwargs)
注册：orms['acc05_bbox'] = Acc05BBoxReward
参数：--external_plugins <插件文件> --reward_funcs acc05_bbox
```

数据原assistant答案在GRPO处理时会被移除，真框放独立`solution`列。生成框按统一短JSON协议解析并从0–1000转换到0–1，奖励使用同一IoU函数。模型输出非法是外部边界，可以给低奖；图像/GT数据损坏应直接报错，不静默吞掉样本。奖励不需要额外语言模型或在线API。

内置`accuracy`主要调用数学答案验证，`format`检查think/answer标签，都不是本赛的bbox正确率；不能直接套用。上游训练器约343–386行将样本、reward函数与trainer_state送入统一评分；约627–719行的动态采样依据组奖励方差补采样。先使用最小固定采样预算观察分布，不能靠无限补采样掩盖几何奖励无信号。

拟比较三种奖励：纯连续IoU、纯ACC阈值、阈值加小连续项。保持相同起点、数据、rollout（采样生成）与训练预算。记录每组框身份/IoU/奖励方差、解析率、净新增过线题及GPU小时；不因生成更长解释而视为进步。Qwen3.5系列的混合注意力加载补丁与多图处理继续交给原生ms-swift，外部Qwen2.5单图训练器只作机制参考。

## 分辨率与融合接口

现有推理对所有模态统一`upscale`。非对称版本只需在图像准备处为RGB/IR/D分别设缩放与预算，并在run config保存每路实际尺寸/网格；训练与推理都要核对处理器是否再次缩放。patch16×merge2给32的尺寸因子；默认函数签名中的28不是这个模型实际使用的因子。

真正新增模态专属LoRA/融合残差，需要处理同一共享视觉塔在不同图像上的路由、批处理和vLLM模型实现，不是修改一个freeze开关。残差零初始化只保证退回**它附加的那个基线**：若基线仍输入三图，就不是RGB-only。因此先在普通三图或明确RGB-only之一上定义好基线，再决定该复杂度是否值得投入。
