# RoboRefIt 原始 Depth 编码审计（2026-09-25）

## 结论与证据边界

- 已下载的 RoboRefIt 训练子集深度文件是 **640×480、单通道 16 位 PNG**；Pillow 读取为 `I;16`，NumPy 为 `uint16`。这是文件编码结论，**不是物理单位结论**。
- 目前查到的[作者数据说明](https://luyh20.github.io/RoboRefIt.github.io/)、[作者仓库 README](https://github.com/luyh20/VL-Grasp/blob/main/RoboRefIt/README.md)、[作者读取代码](https://github.com/luyh20/VL-Grasp/blob/main/RoboRefIt/datasets/grounding_datasets/refer_dataset.py#L229-L267)及[论文](https://arxiv.org/pdf/2308.00640)均未说明发布的深度 PNG 是毫米、厘米或其他单位，也未定义 0、65535 的语义。论文说明采集使用 Intel RealSense D435i/D455，并不能据此反推导出发布文件的比例尺；保存时是否做过转换及相机 `depth_scale` 均未知。因此 **不应把 raw/1000 当作经证实的米数，也不应生成数值距离标签**。
- 作者官网说数据还含对应的 `depth_colormap`，但没有给出色图生成参数。作者数据集读取代码用 `cv2.imread(depth_path)` 默认方式读深度，RGBD 分支只取其第一个通道拼到 RGB 后；这段代码没有提供原始 16 位值的尺度或无效值定义。仓库 README 也注明其公开的预训练模型以 RGB 为输入。

## 本地核验

输入来自 [`selection_report.json`](../../data/external/RoboRefIt/subset/selection_report.json) 与 [`selection_index.jsonl`](../../data/external/RoboRefIt/subset/selection_index.jsonl)：2000 张不同训练图，覆盖 chair、shelf、sofa、table、wash table 五种场景。按各场景索引等间隔抽 10 张，共 50 张、15,360,000 个像素，用 `np.asarray(PIL.Image.open(path))` 对原 PNG 计数；没有把已渲染的色图当原始数据。

| 指标 | 50 张抽样结果 |
| --- | ---: |
| 文件模式/数组类型/尺寸 | 全部 `I;16` / `uint16` / 480×640 |
| 值为 0 的像素 | 2,334,270（15.197%）；各场景 10 图平均约 3.73%–25.41% |
| 值为 65535 的像素 | 176（0.00115%） |
| 排除 0、65535 后的 raw 分位数 | P1=333，P50=896，P95=2253，P99=4124；最大 65303 |

这些数字只描述**原始整数分布**，不表示毫米或距离。零值在[示例 RGB](../../data/external/RoboRefIt/subset/final_dataset/train/image/0000000.png)对应的[原始深度](../../data/external/RoboRefIt/subset/final_dataset/train/depth/0000000.png)中呈大片缺口、边缘和遮挡附近的空洞，适合先作为“无可用深度”的候选掩码；但作者未明确定义该哨兵。65535 在抽样中稀少，部分图像还出现 6 万以上但小于 65535 的值，因此不能仅凭上界断言“65535 是唯一无效值”或将所有高值解释成远距离。先前[执行记录](2026-09-25-rgbdt-execution.md)对另 200 张图的零值估计约 25.1%；两次抽样范围不同，不应合并成全量比例。

直接执行 `PIL.Image.open(depth).convert("RGB")` 会造成严重信息损失：首张 `0000000.png` 的原始深度有 1634 个不同值，转换后的 71.1% 像素成为 RGB `(255,255,255)`，恰好对应原图 raw 值大于 255 的像素。该显示结果的大片纯白不是可靠的“最远区域”。

## 最小处理建议与验证点

1. **保留原始 16 位 PNG**，解码后先在 `uint16` 数组上做统计，不通过 `convert("RGB")`、8 位图或 `cv2.imread` 默认结果反推单位。[当前项目原深度编码函数](../../code/src/mm_grounding/data.py)使用 `depth_scale=1000`、`depth_clip=20`；这一参数组合不能无证据地套用于 RoboRefIt。[后续数据准备代码](../../code/tools/prepare_next_stage_data.py)已经将缺少单位说明的 RoboRefIt 默认设为 `visual`，应维持该来源独立策略。
2. **可视化使用固定、可记录的整数映射**：先从训练子集代表样本统计非零值的稳健分位点，固定上下界，对原值截断并线性或对数映射到 1–255；把 raw=0 显示为 0/单独掩码。50 图得到的 P1=333、P99=4124 可作初步诊断参数，正式参数应在 2000 张训练子集上重新计算并记录。不要逐图独立拉伸作为模型输入，否则跨图亮度不可比较。65535 及其他极高值先统计并靠上界截断，不静默宣称它们全是无效测量；若决定另设掩码，要记录判据。
3. **验证视觉方向与无效区**：在每个场景抽 RGB、raw 深度、上述固定映射图、零值掩码并排目检；核对遮挡、桌面/地面与对象轮廓是否对齐，再与作者提供的 `depth_colormap` 抽样对照。这个步骤只能检验映射是否保留有用结构，仍不能确证物理单位。若要使用“近/远”语义或米制距离，须取得作者保存代码、相机标定/深度比例尺或可独立标定的场景证据后另行验证。

本次仅审计原文件及公开一手资料；未修改训练代码、未触碰云端 GPU、未下载全包，也未生成数值距离标签。
