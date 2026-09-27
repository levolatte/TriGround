# 原生 Qwen3-VL-8B RGB / 三图语言 LoRA 接入准备

日期：2026-09-22。本文只记录数据转换和官方训练入口接入；未启动 GPU、未下载权重、未安装训练依赖，也不声称 8B 训练能放入 24 GB 显存。

> 20:51主Agent实测补充：上句及下文“未验证”表述为初始源码准备时的边界。随后已在云端TF5.14.1/PEFT0.20/4090D上完成新数据全量转换、正确602112像素预算的2B三图8微批/1更新、检查点保存和LoRA重载推理；11项相关测试通过。2B训练11.98秒、峰值allocated约7.77GiB，loss1.448/grad_norm15.42有限，仅语言q/k/v/o LoRA可训练。真实1859输入token、3×[1,36,64]、23完整监督token；重载2条均解析。**8B本身仍未验证，2条不构成质量结论。** 详见[4090D准备与证据](2026-09-22-4090d-preparation.md)。

## 结论与交付

本次新增：

- `code/tools/prepare_qwen3vl_native_sft.py`：读取新标签压缩包，生成官方 Qwen3-VL 微调格式的 RGB 和原生 RGB+IR+Depth 两套训练/验证清单；
- `code/tests/test_prepare_qwen3vl_native_sft.py`：覆盖深度映射边界、BBox 量化边界、split/ID 保留、多图顺序和两路线监督一致性；
- `code/scripts/run_qwen3vl_native_lora.sh`：调用官方 `train_qwen.py` 内的同一个 `train()`，显式选择单进程 SDPA，并在 PEFT 返回模型后检查只有语言层 `q_proj/k_proj/v_proj/o_proj` LoRA 可训练。

转换只读取 `qwen_generation_train_100.json`（3,707 条）和 `qwen_generation_val.json`（412 条）。它不会读取旧融合标签，也不会把验证集并入训练集。RGB 与三图清单保持相同 ID、顺序、split 和 assistant 监督，因此在 `batch=1, accumulation=8, epochs=2, seed=2026` 下样本数和优化器更新节奏一致。

## 官方源码核验

核验对象是 QwenLM/Qwen3-VL 官方仓库当前 HEAD `96588727e44c78b25ba03ea03b8e12f7e64fd0da`（提交时间 2026-01-30）：

- 官方 [qwen-vl-finetune README](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/qwen-vl-finetune/README.md) 明确给出单图、多图和 Grounding（目标定位）格式。多图是 `"image": [path1, path2, ...]`，用户文本中必须有等量、同序的 `<image>`。
- [`data_processor.py`](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/qwen-vl-finetune/qwenvl/data/data_processor.py) 的 `_build_messages()` 按占位符顺序依次弹出图像，并在图像剩余或占位符过多时抛错。它从 `qwenvl.data.data_dict` 读取 `dataset_use`；不能直接把 JSON 路径传给 CLI。
- [`train_qwen.py`](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/qwen-vl-finetune/qwenvl/train/train_qwen.py) 的 LoRA 分支先冻结全部基础参数，再以 `q_proj/k_proj/v_proj/o_proj` 为 PEFT 目标。当前 Qwen3-VL 视觉注意力使用 `qkv/proj`，因此这四个尾名应只命中语言注意力；这是依赖当前模块命名的事实，仍需在实际加载 8B 权重后验收参数名。
- 同一文件的 `__main__` 把 `flash_attention_2` 写死，没有 attention implementation（注意力实现）CLI 参数。`train(attn_implementation=...)` 本身接受实现名，所以包装器导入并调用官方函数，显式传 `sdpa`。
- 仅传 `sdpa` 仍不足以在无 `flash_attn` 环境启动：`train_qwen.py` 无条件导入同目录 `trainer.py`，而该模块第 4 行无条件导入 `flash_attn_varlen_func`。`trainer.py` 还依赖 Transformers 内部 API，并全局替换 `Trainer.create_optimizer`。
- 包装器在导入 `train_qwen.py` 前注册一个只包含 `replace_qwen2_vl_attention_class` 的 `trainer` shim（薄适配层），从而不导入上述 FlashAttention 模块。该 shim 的函数一旦被调用便直接报错；包装器把 `data_flatten` 和 `data_packing` 固定为 `False`，正常 SDPA 路线不会调用它。LoRA 路线没有设置 `mm_projector_lr` 或 `vision_tower_lr`，因此使用原生 Hugging Face `Trainer` 的默认优化器分组，不需要官方 `trainer.py` 的全局优化器替换。这不是新的 Trainer 或 loss。
- `make_supervised_data_module()` 固定返回 `eval_dataset=None`。官方入口当前不能消费这次的 412 条验证清单；包装器使用 `--eval_strategy no`，验证清单留给后续独立生成评估。若要训练时评估，至少要改官方数据参数和数据模块，这已超出本轮“薄接入、不写新 Trainer”的范围。

## 转换格式

每个输出条目保留：

- `id`：原始 Query ID；
- `split`：严格为 `train` 或 `val`；
- `class_name`：源标签类别，仅作审计，官方加载器忽略未知字段；
- `image`：RGB 路线是一个路径字符串；三图路线按 RGB、红外、深度可视化顺序给三个路径；
- `conversations`：一轮 human/gpt 对话，答案只有 `{"bbox_2d":[x1,y1,x2,y2]}`。

源框是 `[0,1]` 的 `xyxy`。工具逐坐标乘 1000 并使用十进制 half-up（五入）量化，输出 0--1000 整数；越界、反向框以及量化后塌缩的框直接报错。

图片路径以 `DATA_ROOT` 为基准。RGB 与 IR 直接引用原 PNG，不复制。深度图输出到：

```text
DATA_ROOT/target_v2/qwen3vl_native_sft/depth_rgb/
```

默认输出为：

```text
rgb_train.json
rgb_val.json
trimodal_train.json
trimodal_val.json
conversion_report.json
depth_rgb/*.png
```

## 深度毫米可视化

官方赛题 PDF 说明深度是单通道 16 位、单位毫米、0/过小代表无效、数值越小越近，相机可感知范围约为 0.3--20 m；现有训练图抽样的非零值 `p1=1596, p50=6687, p99=19197`，抽样最大值不超过 19999，零值平均占 24.72%。官方材料没有给出“过小”的精确阈值，因此工具没有从验证集估计阈值，也没有自行引入 300 mm 或 1 m 的截断。

固定映射为：

```text
d == 0 mm       -> RGB(0, 0, 0)，无效
1 <= d <= 19999 -> 线性映射 RGB(255..1)，近亮、远暗
d > 19999 mm    -> RGB(1, 1, 1)，截到最远有效灰度
```

三个通道始终相同。该表示保留了全数据统一的距离方向，黑色只表示明确的 0 无效值；超过相机标称量程的正值仍保留“很远”的含义。它不逐图归一化，也不使用旧的 green-mask（绿色掩码）编码。映射参数会写进 `conversion_report.json`。

## 云端 CPU 转换

在 `F:/AIC/code` 对应的云端代码目录执行：

```bash
python tools/prepare_qwen3vl_native_sft.py \
  --manifest-zip /path/to/qwen_generation_train_val_manifests.zip \
  --data-root /root/autodl-tmp/rematch_20260922/data/city/train
```

成功报告应为 `train_samples=3707`、`val_samples=412`、`rendered_depth_images=759`。这是由 759 组互不重叠的 RGB 场景推得的预期值；实际命令输出是验收依据。

转换后建议先运行：

```bash
python -m pytest tests/test_prepare_qwen3vl_native_sft.py -q
```

本地静态/CPU 验证结果：

- 对真实标签压缩包读取并遍历全部 BBox：3,707 train、412 val、ID 交集 0，全部可完成 0--1000 量化，唯一 RGB/Depth 路径均为 759；
- `pytest tests/test_prepare_qwen3vl_native_sft.py -q`：3 passed；
- Ruff：通过；`bash -n scripts/run_qwen3vl_native_lora.sh`：通过；
- 在 Transformers 4.57.3、未安装 `flash_attn` 的本地环境中，shim 后可成功导入官方 `qwenvl.train.train_qwen`；该环境未安装 PEFT，也没有权重，因此这只证明 FlashAttention 导入阻塞已被绕开，不证明可训练。

## 单卡 SDPA LoRA 启动

包装器默认参数与首轮约定一致：LoRA `r=32, alpha=64, dropout=0.05`，学习率 `1e-5`，batch 1，梯度累积 8，2 epoch，seed/data_seed 2026。它使用一个普通 `python` 进程，无 `torchrun`、无 DeepSpeed、无 FlashAttention；视觉编码器、Merger（合并器）和基础语言模型权重均应冻结。

官方微调 README 指定 `torch==2.6.0`、`torchvision==0.21.0`、`transformers==4.57.0.dev0`、`accelerate==1.7.0`、`peft==0.17.1`。云端现有推理环境的 Transformers 5.14.1 与这份训练代码尚未做 import、单步和保存兼容性验证。训练应创建独立 venv 并按官方组合安装，保留当前推理环境；本轮没有执行安装。SDPA 包装器移除了对 `flash_attn` 的启动依赖，但没有把 5.14.1 宣称为兼容版本。

包装器默认每图 `max_pixels=602112`、`min_pixels=200704`，与本项目当前原生推理尺度一致。

### Transformers 5 图像上限兼容修复

云端首次 2B 三图单步虽然成功，但官方数据模块首样本实际达到 6,251 token。根因是 Transformers 5.14.1 中 `processor.image_processor.size` 是 `SizeDict`，不是 `dict`；官方 `update_processor_pixels()` 的 `isinstance(ip.size, dict)` 分支被跳过，且该处理器实例没有可供旧兼容分支修改的 `min_pixels/max_pixels` 属性。于是 CLI 的 602112 没有进入实际 resize（缩放）配置。官方脚本另外创建的 tokenizer 只交给 Trainer，数据集实际使用 `processor.tokenizer`；后者也没有被设置为 4096。第一次 2B 显存结果因此对应错误的大图 token 输入，不能用于后续 8B 推断。

包装器现在通过受支持的公开入口，在官方 `AutoProcessor.from_pretrained()` 构造时传入 `min_pixels` 与 `max_pixels`，并显式设置 `processor.tokenizer.model_max_length=4096`。它随后包装官方 `make_supervised_data_module()`，只额外读取并 collate（组批）第一个训练样本，打印并检查：

- 实际 `image_grid_thw`、每图缩放像素数和合并视觉 token；
- 总输入 token 和 `processor.tokenizer.model_max_length`；
- 有效监督 token 数与解码文本；
- 图像数与 grid 数一致、每图不超过 `max_pixels`、输入不超过 4096，且完整 BBox JSON 确实存在于监督文本。

这里不依赖 tokenizer 自动截断。任何超长输入或被截断的答案都会在 Trainer 构造前直接报错。

云端已用官方 `make_supervised_data_module -> dataset[0] -> collator` 路径验证该构造器修复：输入 1,859 token，三张图均为 `[1,36,64]`，`pixel_values` 为 `[6912,1536]`，有效监督 23 token，内容为完整 BBox JSON 加终止符。每图对应 589,824 像素、576 个合并视觉 token，符合 602112 上限。这是输入正确性证据，不是 8B 显存可容纳证明。证据保存在 `.work/rematch_20260922/official_training_processor_probe.txt`。

先对 RGB 做一步显存验收：

```bash
QWEN_FINETUNE_DIR=/root/src/Qwen3-VL/qwen-vl-finetune \
DATA_ROOT=/root/autodl-tmp/rematch_20260922/data/city/train \
MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct \
DATASET_VARIANT=rgb \
MAX_STEPS=1 \
OUTPUT_DIR=/root/autodl-tmp/rematch_20260922/outputs/native_rgb_smoke \
bash scripts/run_qwen3vl_native_lora.sh
```

随后以完全相同设置测试三图，只改 `DATASET_VARIANT` 和输出目录：

```bash
QWEN_FINETUNE_DIR=/root/src/Qwen3-VL/qwen-vl-finetune \
DATA_ROOT=/root/autodl-tmp/rematch_20260922/data/city/train \
MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct \
DATASET_VARIANT=trimodal \
MAX_STEPS=1 \
OUTPUT_DIR=/root/autodl-tmp/rematch_20260922/outputs/native_trimodal_smoke \
bash scripts/run_qwen3vl_native_lora.sh
```

启动日志必须列出实际可训练参数并以 `language_model...q_proj/k_proj/v_proj/o_proj...lora_A/lora_B` 为限。发现视觉 LoRA、Merger 可训练、缺失任一目标投影或没有可训练参数时，包装器会在训练前失败。

一步成功后去掉 `MAX_STEPS=1`（默认 `-1`）启动完整 2 epoch。RGB 与三图应分别从同一基础 8B 权重开始，不应从另一条路线的 LoRA 接续。验证集不传给脚本。

## L40 / 4090D 接续边界

L40 48 GB 和 4090D 24 GB 都应先运行一次真实的 forward/backward/optimizer step，并记录峰值显存和单步时间。尤其 24 GB 不能根据 LoRA 参数量推断必然可放下，因为冻结视觉权重仍需视觉前向，三图还把约三倍视觉 token 送进语言层。

若 602112 的三图一步在 4090D 上 OOM，可将 `MAX_PIXELS=200704` 后重新做 RGB 与三图两条路线的一步测试；正式对照也必须让两条路线使用同一个每图上限。降低分辨率会改变实验问题，不能只降低三图后与高分辨率 RGB 直接归因比较。L40 也不能跳过一步验收。

仍未验证的项目：

- 8B 权重加载后的实际 PEFT 参数名和冻结状态；
- SDPA 下 602112 每图上限的一步峰值显存、吞吐和是否 OOM；
- 独立训练 venv 中官方 pinned 依赖组合的 import、单步和保存兼容性；
- 官方入口保存的最终 PEFT adapter 是否能被当前推理脚本直接加载；
- 412 条验证集的独立生成、BBox 解析和 ACC@0.5 评估入口。

这些都需要权重或 GPU 才能验证。本轮没有为它们增加 fallback，也没有复制或改写官方 Trainer。
