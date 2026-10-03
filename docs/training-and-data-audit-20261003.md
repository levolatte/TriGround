# 训练策略与数据审计（2026-10-03）

> 起因：用户质疑"我们直接开始训练了，但不能保证训练方法和数据集都完美"。
> 本文是据此做的实查，**所有数字都来自本机/云端的实际文件与进程**，不是推断。
> 结论分三部分：训练策略、数据血缘、待补的缺口。

---

## 一、训练策略审计

### 1.1 实际生效的参数（从运行中的进程 cmdline 读出，非配置推测）

| 参数 | 值 | 与基线 A 是否一致 |
|---|---|---|
| `--max_steps` | 600 | ✅ |
| `--per_device_train_batch_size` × `--gradient_accumulation_steps` | 1 × 8 | ✅ |
| `--learning_rate` | 5e-6 | ✅ |
| `--lr_scheduler_type` / `--warmup_ratio` | linear / 0 | ✅ |
| `--lora_rank` / `--target_modules` | 32 / all-linear | ✅ |
| `--freeze_vit` / `--freeze_aligner` | true / true | ✅（正确隔离了"可训练范围"这个变量） |
| `--max_pixels` / `--max_length` | 2073600 / 8192 | **有意改动**（本轮主变量） |
| `--gradient_checkpointing` | true | ✅ |
| `--save_steps` / `--save_total_limit` | 100 / **2** | ⚠️ 见下 |
| `--num_train_epochs` | 1 | ✅ |

### 1.2 训练/推理一致性（已逐项核实，**没有静默不一致**）

这是最容易出问题、也最容易无人察觉的地方，因此单独查了三条：

1. **提示词逐字符一致** ✅
   训练样本的 `messages[0].content` 与推理的 `run_config.json.prompt_template` 完全是
   同一段文本（含 `<image>` 三连、深度读法说明、"normalized to 0-1000" 要求）。
2. **深度图处理是共享代码** ✅
   训练用的 `depth_rgb/*.png` 由 `prepare_qwen3vl_native_sft.depth_mm_to_grayscale_rgb()` 生成；
   推理端 `predict_native_submission.py` **直接 import 了同一个函数**（第 33–36 行）并在
   第 199 行调用。官方复赛的深度是 `I;16` uint16（0–19999），推理时现渲染，路径一致。
   *（这一条一度看起来像严重 bug——训练是 RGB uint8 可视化、官方是 uint16 原始——
   查到共享函数后才确认无问题。）*
3. **答案格式一致** ✅ 训练目标是 `{"bbox_2d":[801,111,869,781]}`，推理用 `stop=["]"` 截断后解析。

### 1.3 可以合理改进的地方（按性价比排序）

| # | 问题 | 影响 | 成本 | 建议 |
|---|---|---|---|---|
| **1** | **`save_total_limit=2`**：600 步只保留最后两个检查点（500/600）。若最佳在 300 步，**永远看不到** | 中高（小数据 + 大模型，过拟合风险真实存在） | 低（只改一个参数） | 下一臂改成 6 |
| **2** | **训练中无验证**：`--split_dataset_ratio 0`，无法在线发现过拟合 | 中 | 中（需接 City412 评测回调） | 用离线检查点评测替代 |
| **3** | **只有 1 个 epoch、2250 条唯一样本**：对 27B 可能偏少。loss 到 190 步仍在降（0.357） | 中 | 高（12h/epoch） | 第二臂可试 2 epoch |
| **4** | **53% 数据重复**：4800 行里 2250 条唯一，等于一半算力花在重复样本上 | 中 | 低（换清单） | 第二臂用去重清单 |
| **5** | **LR 沿用 8B 的 5e-6**：未针对 27B 调过 | 未知 | 高（需扫参） | 时间不允许，记录为未验证 |
| **6** | `warmup_ratio=0`：27B 从第 0 步就是满学习率 | 低（实测 grad_norm 稳定在 0.7–2.9） | — | 可不动 |
| **7** | `weight_decay=0`：无正则 | 低 | — | 可不动 |

**结论**：训练策略本身**没有发现会毁掉结果的问题**，主要缺陷是**可观测性**——
保存点太少、训练中无验证，导致"训完才知道好不好，且只能看最后一步"。

---

## 二、数据血缘审计（回答"手动加的数据在不在管线里"）

### 2.1 结论：**完全不在。**

当前训练清单是

```
/root/autodl-tmp/rematch_20260922/results/triground_abv_execution_20260928/
    deployment_600_seed2026/manifests/A.json
```

实查该目录：**只有 `A.json` 一个文件**，且

- `task_category` **100% 为 `old_city`**
- 4800 行、**2250 个唯一查询**、660 组三图
- 12 个类别：person / animal / street light / car / sign / seat / garbage can /
  bicycle / boat / uav / tricycle / ball

### 2.2 A/B/V 三个臂的设计（读 `tools/prepare_triground_abv_data.py:757-810`）

脚本为每 80 个呈现（steps//10 块）这样组批：

| 区间 | A 臂 | B 臂 |
|---|---|---|
| `local < 48`（60%） | old_city | **同一条** old_city |
| `local 48–63`（20%） | old_city | `depth_relation` + `robo_competition` |
| `local 64–71`（10%） | old_city | `ir_complement` |
| `local 72–79`（10%） | old_city | `reliability` |

即：
- **A = 纯 City**
- **B = 60% City + 40% 新数据**（RoboRefIt、深度关系、红外互补、可靠性）
- **V = B 的副本**

**我们训练的正是 A —— 唯一不含任何外部/人工新增数据的臂。**
B/V 从未在云端生成（部署目录里不存在）。

### 2.3 外部数据现在在哪

| 数据 | 云端 | 本地 | 可用性 |
|---|---|---|---|
| **RoboRefIt** | ❌ **不存在**（`data/external/` 都没有） | ✅ **7.96 GB / 6024 文件 / 6010 张图** + 已转换标注 | **可直接用** |
| **RGBT-GroundBench** | ❌ 不存在 | ⚠️ 仅 0.03 GB —— 只有 3 个 `ann_*.tar` 标注包，`image_data/` 只有 20 个文件 | **图像未下载，不可用** |

### 2.4 RoboRefIt 能补什么（已实查转换产物）

`data/external/RoboRefIt/subset/converted/train.jsonl`：

- **2000 条 / 1913 个唯一查询 / 2000 张唯一图**
- 字段：`rgb`、`depth`、`query`、`bbox`、`class_name`、`scene`
- 查询风格为机器人指令式：`"please bring me the yellow cone"`（长度 13–115 字符）
- **只有 RGB + depth 两模态，没有红外**
- **65 个类别，与 City 的 12 类零重叠**

官方复赛查询中出现这些新类别的次数（词面匹配）：
`orange` 116、`pear` 76、`glass` 58、`tiger` 53、`umbrella` 39、`rectangular box` 24、
`camel` 21、`elephant` 20、`lion` 20、`cone` 18 …… 共 **482 条查询（8.5%）** 命中。

**⚠️ 但要谨慎**：RoboRefIt 是**室内桌面/货架**场景（facial cleanser、toothpaste、teddy bear），
与官方**室外街景**域差距明显；而 `orange` 这类词在街景里多半是**颜色**而非水果。
所以 8.5% 这个数字是**词面匹配的上界**，真实收益会低不少。

### 2.5 训练清单与官方清单的差距（**已更正归因**）

| | 数值 |
|---|---|
| 官方训练清单 `qwen_generation_train_100.json` | 3707 行 / **3144 个唯一查询** |
| A.json | 4800 行 / **2250 个唯一查询** |
| **A 从未让模型见过的问题** | **894 个（28.4%）**，涉及 270 组三图 |

A.json 是官方清单的**严格子集**（A 有而官方没有 = 0 条）。

**⚠️ 但这不是"漏用"，而是有意的场景上限抽样。** 见 `prepare_triground_abv_data.py:642-771`：

```python
common = _draw_old(city, 2880, scene_cap=6, rng=rng)    # 每场景最多抽 6 条
extra  = _draw_old(city, 1920, scene_cap=11, rng=rng)   # 每场景最多抽 11 条
...
if scene_counters[scene] >= scene_cap:
    continue        # 触顶的场景直接跳过
```

**已用现有数据验证该机制**（`audit_894_scenes.py`）：

| | 场景数 | A 中已用条数**中位** |
|---|---|---|
| 有缺失的场景 | 282 | **6** ← 正好等于 `scene_cap=6` |
| 零缺失的场景 | 399 | 2 |

**缺失的场景×问题对 924 条中，795 条（86%）来自 A 已抽满 ≥6 条的场景。**

**结论**：这是**采样纪律**（防止查询密集场景主导分布），不是疏漏。
代价是在总语料仅 3144 个问题的前提下砍掉 28%，**相当激进**。

**因此"用满清单"不是零风险改动**——取消上限会改变训练分布（少数场景权重上升）。
**更稳妥的是折中：把 `scene_cap` 从 6 提到 8~10。**

### 2.6 数据与官方分布的三个缺口（量化）

| 维度 | 训练集 | 官方复赛 |
|---|---|---|
| 查询长度 >100 字符 | **0.0%** | **7.5%** |
| 查询长度 0–20 | **0.0%** | 4.9% |
| 序数词（first/second/third） | 6.7% | **21.4%** |
| 目标物体在训练 12 类内 | — | **仅 31.0%** |

---

## 三、可执行的改进（按风险从低到高）

| 优先级 | 动作 | 风险 | 预期 |
|---|---|---|---|
| **1** | 第二臂把 `scene_cap` 从 6 提到 **8~10**（而非取消） | **低**——多用数据但保留防场景过采样 | 唯一查询 2250 → 约 2700+ |
| **2** | 第二臂把 `save_total_limit` 提到 6 | 极低 | 可挑选最佳检查点 |
| **3** | **接入 RGBT-GroundBench**（39,442 条英文查询，10.39 GB，已实测可下） | **中**——只有 4 类（person/car/truck/bicycle），补的是**长查询与空间关系**而非物体类型；无 depth 模态，**必须同时对 depth 做随机 dropout** | >100 字符查询 0% → 23.85% |
| **4** | 抽样核对 **RGBDT500** 的类表（唯一真三模态源，官方称 ≥66 类） | 中——需勾选协议、类表未公开 | 可能补 planter/umbrella/wall 等缺口 |
| **5** | 序数词（官方 21.4%）**必须另开 RGB/RGB-D REC 路线** | 高 | 红外数据 39,442 条里序数词仅 9 条（0.02%），此路无效 |

**明确不建议**：改 LoRA 秩、换学习率、改 epoch 数——这些都会破坏与 A 臂的可比性，
而当前证据不支持它们能带来确定收益。

---

## 四、数据集调研结论（红外/热成像路线，2026-10-03）

完整报告见 `docs/dataset_survey_thermal_rgb.md`（由子 Agent 完成，**结论均经实测解包验证**）。

### 4.1 三个决定性发现

1. **「RGB + 热成像 + 英文指代查询 + bbox」的数据集确实存在**：**RGBT-GroundBench**
   （39,442 条查询 / 17,335 对图 / 全英文 / 10.39 GB / HF 免登录直链）。
2. **序数词缺口在热成像路线上无解**：39,442 条查询里 `first` 3 条、`second` 6 条、
   `third` **0 条**（**0.02%**），而官方测试集 **21.4%** 的查询含序数词。
   **必须另开 RGB/RGB-D 的 REC 数据集路线（gRefCOCO / Ref-ZOM 等）。**
3. **不存在「RGB + 热成像 + 深度 + 指代查询」的四元数据集**。三模态齐全的只有
   RGBDT500（跟踪框，无查询）与 PST900（894 对，隧道域，5 类分割）。
   → **所有 RGBT 数据都没有 depth，训练时必须对 depth 做随机 dropout**，
   否则模型会学到"depth 一定存在"的捷径，反而在只有 RGB 的场景崩掉。

### 4.2 两个会导致直接 bug 的实测细节（RGBT-GroundBench）

- **bbox 是 `[x, y, w, h]` 绝对像素，不是 xyxy**。子 Agent 边界校验：23,000 条样本按 xywh
  解释 **0 条越界**，按 xyxy 解释则大量越界。转换必须 `xyxy = [x, y, x+w, y+h]`。
- **类别 id 每个源数据集独立，不可跨源复用**：FLIR 子集 = `{1:truck, 2:car, 3:person, 4:bicycle}`，
  MFAD 子集几乎只有 `{2:car, 3:truck}`。

### 4.3 必须诚实说明的限制

**RGBT-GroundBench 的类别空间（4 类）比我方现有 12 类更窄。** 它补的是
**语言复杂度**（>100 字符占 23.85%、含 "near" 57.4%、"under" 42.5%）与
**跨模态对齐监督**，**不是物体类型**。不要指望它解决"官方 69% 查询不在 12 类内"的问题。

### 4.4 已劝退的候选（看起来相关但实际无用）

LLVIP（仅 person 类）、KAIST（仅行人/骑行者）、LasHeR / RGBT234 / RGBT210 / GTOT
（单目标跟踪框，**无类别语义**）、DroneVehicle（5 类全是车 + 航拍域）、
RGB-Th-Bench（**完全没有 bbox**）、SGRT-140（140 对 + 登录墙）、
TartanRGBT（三模态齐全但**无任何目标框**）、MFNet（本轮未确证下载入口）。

### 4.5 已采取的行动

**已启动 RGBT-GroundBench 下载**（`scripts/fetch_rgbt_groundbench.sh`，`screen -dmS rgbt`）。
云端 HF 直连被墙，**改用 `hf-mirror.com`**（实测 1.6 MB/s，10.39 GB 约 1.8 小时）。
下载在 training 之外并行进行，不占 GPU。
