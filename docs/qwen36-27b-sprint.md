# Qwen3.6-27B 原生分辨率冲刺：实施记录

2026-10-02。分支 `sprint/qwen36-27b-native-resolution`。本文记录**无卡阶段（阶段 A）已完成的事实**，
以及开卡后（阶段 B）要做什么。所有数字都来自实查，未实测的明确标注为估算。

---

## 1. 目标与判据

把基线从 `Qwen3-VL-8B + 语言 LoRA`（City412 = **296/412 = 71.8447%**）换成
`Qwen3.6-27B + 原生分辨率`，为半决赛（10-10 前开始）产出可提交的模型/代码/报告。

**为什么是这两件事**（10-02 实测）：

| 事实 | 数值 | 含义 |
|---|---|---|
| 116 个失败题的目标面积中位数 | **0.88 个视觉 token** | 目标在缩放后不到一个 token |
| 命中题的目标面积中位数 | 3.33 token | 对比 |
| 30 个"框住同图另一实例"案例中本题-邻居中心距 | 中位 **2.31 token** | 要从相距 2 token 的实例里挑对 |
| 模型默认像素预算 vs 项目实际 | 16,777,216 vs **602,112** | 只用了默认的 **1/28** |
| 602112 的来源 | 9-18 从 802816 OOM 后的应急降档 | 不是设计选择 |
| 11 个历史臂合并后的 oracle 上限 | 306/412（+8 题） | 在现有模型族里加臂/挑检查点已无空间 |
| A 的 116 个失败中"别的臂做对过"的 | 仅 10 个 | 是能力问题，不是选择问题 |

**成功判据**：City412 命中 **> 296**，且 `tools/verify_acc05_dev.py --compare` 的 **78 图组聚类配对
95% 区间下界 > 0**。否则回退到 A，不追加种子。

---

## 2. 旧 → 新：变化清单（逐项）

| 维度 | 旧（已验证基线 A） | 新（Qwen3.6-27B） | 处置 |
|---|---|---|---|
| 实例 | 原 4090 机 `westc:46057` | **RTX PRO 6000 96G** `weste:47809` | 新机全空，需重建 |
| 架构 | Ada sm_89 / 24G | **Blackwell sm_120 / 96G** | 用 sdpa，不装 flash-attn |
| torch | 2.8.0+cu128 | **2.12.1+cu130（镜像自带）** | 未被 pip 改动（已加护栏） |
| transformers | 5.14.1 | **5.16.1**（`>=5.2.0` 硬要求） | 4.57.3 连 config 都读不了 |
| 新增依赖 | — | `qwen_vl_utils 0.0.14` / `decord 0.6.0` / **`ms-swift 4.5.3`** | qwen3_5 必需 |
| 数据盘 | 150G/67G free | **200G/194.6G free** | 用户已扩容 |
| 模型类 | `Qwen3VLForConditionalGeneration` | **`Qwen3_5ForConditionalGeneration`** | 3 处写死 → 全部改动态解析 |
| `model_type` | `qwen3_vl` | `qwen3_5` | — |
| 注意力 | 全注意力 36 层 | **混合：Gated DeltaNet+全注意力，64 层** | LoRA 目标层名不同，必须核实 |
| DeepStack | 4 路视觉出口 | **无（`deepstack_visual_indexes: []`）** | 旧结构实验路线不适用 |
| 视觉塔 | patch16/merge2/1152/27 层 | **同族同参数** | pixel 参数语义不变 |
| processor | `Qwen3VLProcessor` | **仍是 `Qwen3VLProcessor`** | 可直接换 |
| thinking | 无 | **默认开启** | 必须 `enable_thinking=False` |
| max_pixels | 602112（588 tok/图） | **探测 {602112, 1204224, 2073600}** | 2073600 = 原生 |
| model_max_length | 4096 | **4096（1.2M 档）/ 8192（原生档）** | 三图 6075 token |
| 训练框架 | `qwen-vl-finetune`（只支持 qwen3_vl） | **ms-swift**（官方支持 Qwen3.6） | 数据格式随之改变 |
| 数据格式 | conversations(human/gpt) | **messages + images** | 新增转换器 |
| 图像路径 | 绝对路径 `/root/autodl-tmp/rematch_20260922/...` | **完全一致** | 新机复刻旧布局，manifest 零改动 |
| 超参 | 语言 LoRA r32/a64/drop0.05、lr5e-6、accum8、seed2026、600 步 | **逐项不变** | 单变量可比 |

---

## 3. 阶段 A 已完成事项（实查证据）

### 3.1 云端对传通道
旧机恢复可达后，在新机生成一次性 ed25519 传输密钥，把**公钥**追加到旧机
`authorized_keys`，新机直连旧机验证通过（`TRANSFER_OK`）。旧机私钥未离开本机。

### 3.2 数据与权重迁移（`scripts/migrate_from_old.sh`，ALL_DONE）
云端对传 **~52 MB/s**，全程约 6 分钟：

| 项 | 结果 |
|---|---|
| `data/city/train` | **5.4G**，visible / infrared / depth_rgb **各 759 个文件** |
| `city_val.json` + `city_gt.json` | 已就位 |
| 训练清单 `A.json` | 4,860,610 B，已就位 |
| 基线 A 语言 LoRA | `adapter_model.safetensors` 122,726,608 B（已排除中间 checkpoint） |
| `Qwen3-VL-8B-Instruct` | 17G，已就位（用于第一跳探针与兜底） |

### 3.3 环境（`scripts/cloud_setup_qwen36.sh`，SETUP_DONE）
`transformers 5.16.1 / peft 0.20.0 / accelerate 1.15.0 / decord 0.6.0 / qwen_vl_utils 0.0.14 / ms-swift 4.5.3`；
**关键自检通过**：`from transformers import Qwen3_5ForConditionalGeneration` → OK。
`torch 2.12.1+cu130` 前后一致（ms-swift 曾把 transformers 从 5.18.0 收敛到 5.16.1，仍满足 ≥5.2.0）。

### 3.4 权重下载（`scripts/fetch_model_modelscope_curl.sh`）

**踩到并修掉的两个部署坑**：

1. **HF 镜像卡死**：`hf-mirror` 跑到 0.27GB 后 20 秒零增长，进程仍在但速率恒为 0。改走 **ModelScope**（同仓库 `Qwen/Qwen3.6-27B`）。
2. **ModelScope CLI 同样卡死，且根因是磁盘写满**：在约 13.2GB 处停住；查下来是
   **系统的 30GB overlay 盘 100% 写满** —— `/root/rematch_models` 位于系统盘而非 200GB 数据盘，
   8B(17G) + 27B 部分(13G) 把它填满，curl 报 `No space left on device` 而下载静默停滞。
   **修复**：把模型目录整体移到 `/root/autodl-tmp/rematch_models`（200GB 数据盘），
   并在 `/root/rematch_models` 留**同名符号链接**，使所有既有绝对路径（脚本、manifest）继续可用。
   修复后：系统盘 2.4G/28G 可用，数据盘 35G/166G 可用。

同时改为**清单式 curl 下载**：官方 API 取文件清单与字节大小 → 把遗留 `*.incomplete` 改回正式名交给
`curl -C -` 续传 → 多轮扫描直到大小吻合 → 按 `model.safetensors.index.json` 校验分片集合。
**并加了磁盘守卫**：目标文件系统可用空间小于「模型体积 + 5GiB」时直接退出，不再静默跑一半。

### 3.4.1 下载源与体积
ModelScope 上 29 个文件、合计 **55.6 GB**（15 个分片各约 3.9GB）。HF 镜像在本机不可用。

### 3.5 代码适配（已提交）
- `tools/native_model_loading.py`：按 `config.json` 的 `architectures[0]` 解析模型类，一份代码兼容 Qwen3-VL / 3.5 / 3.6。
- `tools/evaluate_pretrained_grounder.py`、`tools/predict_native_submission.py`：改用解析器；新增
  `--model-max-length`、`--no-thinking`；`verify_run_config` 把新键当作带默认值的可选键，历史 run 仍可 `--package-only`。
- `tools/convert_manifest_to_swift.py` + `tests/test_convert_manifest_to_swift.py`。
- `scripts/{migrate_from_old,cloud_setup_qwen36,download_qwen36_27b,train_qwen36_27b_swift,cloud_cpu_precheck}.sh`。

### 3.6 清单转换与测试（本地与云端一致）
`A.json` → ms-swift 格式：**4800 条 / 660 个唯一图组 / 每条 3 图**，保序保重数、文本逐字不变。
`pytest tests/test_convert_manifest_to_swift.py` **7 项全过**，含：
- 真实清单无损（首末条逐字比对）
- **训练图组 ∩ 开发集 78 图组 = 0**（防泄漏硬门槛）
- 跨平台绝对路径判断（Windows 上 `/root/...` 曾被误判为相对路径，已修）

### 3.7 云端 CPU 预检（`scripts/cloud_cpu_precheck.sh`）
- 数据完整性：759 文件/模态、A 适配器、8B 基座 ✓
- 清单转换：4800 / 660 / 3 ✓
- 探针 dry-run：**588 / 1176 / 2025 token 每图 → 1764 / 3528 / 6075 token 每条** ✓
- ms-swift 已按 `MAX_PIXELS=1204224` 推出 `image_max_token_num: 1176`，与预测一致 ✓
- 像素覆盖实测生效 ✓（这是项目历史上踩过的坑）

---

## 4. 阶段 B：开卡后执行顺序

| 步 | 动作 | 预计 | 通过条件 |
|---|---|---|---|
| B0 | `preflight_new_base.py` 复核（`capability=sm_120`、96G、权重可载） | 10 min | 三个 problem 清零 |
| B1 | **第一跳探针**：冻结 A(8B) × 三档像素，City412 | ~1 h | 错误实例数下降即强信号 |
| B2 | **第二跳探针**：Qwen3.6-27B 零样本 × 三档，City412 | 2–4 h | 看错误实例题与小目标子集 |
| B3 | 4 步训练预检：s/step、峰值显存、LoRA 张量有梯度 | 0.5 h | 显存 < 90G |
| B4 | 正式训练 600 步（`train_qwen36_27b_swift.sh`） | 8–16 h（估算） | 检查点 100/200/400/600 |
| B5 | City412 评估 + 聚类配对判决 | 2 h | 下界 > 0 且命中 > 296 |
| B6 | 半决赛包（`predict_native_submission.py` 已适配新基座） | 按赛程 | 格式校验通过 |

**命令**：
```bash
# B1 第一跳（冻结 A，只改像素）
python tools/probe_native_resolution.py \
  --model /root/rematch_models/Qwen3-VL-8B-Instruct \
  --adapter /root/autodl-tmp/rematch_20260922/results/triground_abv_execution_20260928/runs/seed2026_600_deterministic/A/main \
  --manifest .../inputs/city_val.json --target-manifest .../inputs/city_gt.json \
  --data-root .../data/city/train --output-dir <out>/probe_A \
  --pixels 602112 1204224 2073600 --model-max-length 16384

# B2 第二跳（新基座零样本）
python tools/probe_native_resolution.py --model /root/rematch_models/Qwen3.6-27B --no-thinking ...

# B4 训练
MAX_PIXELS=1204224 MAX_LENGTH=4096 OUTPUT_DIR=/root/runs/q36_27b_1m2 \
  bash scripts/train_qwen36_27b_swift.sh
```

---

## 5. 风险与回退

| 风险 | 触发信号 | 处置 |
|---|---|---|
| 显存不足 | B3 峰值 > 92G | 降到 1204224 档；或改 9B |
| LoRA 打错模块 | 可训练参数 ≈ 0 / 无梯度 | 用 `target_modules.log` 里打印的真实层名显式指定 |
| ms-swift 模板不匹配 | 编码后 `<image>` 未替换 / 标签缺失 | 已被 3.7 的模板编码检查覆盖；必要时注册自定义模板 |
| Blackwell 算子问题 | 报 unsupported | 退 `attn_impl eager`；仍失败则换 vGPU-48G + 9B |
| 训练退化 | 200 步 City412 < 250 | 立即停，回 A；不调 lr 重试 |
| 探针无信号 | 错误实例数不降 | 停止训练，A 交半决赛 |

**回退**：A 的 adapter、8B 权重、历史预测全部保留在旧机与新机；分支独立，
`audit/gpt-pro-20260927` 未受影响。

---

## 6. 关键路径与当前状态

- ✅ 阶段 A 全部完成（除 27B 权重仍在下载）
- ⏳ 27B 权重（ModelScope，15 分片，约 1 小时）
- ⏸ 阶段 B 等用户挂载 GPU（当前实例为无卡模式）
