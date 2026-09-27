# RGBDT500 深度语义：原始来源证据（2026-09-25）

## 结论

公开原始资料确认 RGBDT500 的 RGB 与 Depth 来自 ZED 立体相机；论文称两者时间同步、像素对齐。红外图来自另一台 LGCS121 相机，作者以人工特征点映射对齐各模态。论文同时说明有少数序列存在模态视角不一致。资料没有说明归档 Depth PNG 的物理单位、数值比例/偏移、无效值码或 raw 数值的近远方向。故目前不能把 RGBDT500 的 16 位 PNG 当作毫米图，也不能直接由灰阶显示或数值相关性给出近远标签。

## 来源明确写出的事实

| 主题 | 原始来源明确写出的内容 | 边界 |
|---|---|---|
| 采集设备 | RGBDT500 论文 §3.1（第4页）称 RGB 和 Depth 由 ZED 立体相机采集；TIR 由独立 LGCS121 热红外相机采集。 | 论文没有写 ZED 的具体型号、SDK 版本或导出 API。 |
| 对齐设计 | §3.1 称 ZED 输出时间同步、像素对齐的 RGB/Depth 图像对。TIR 与 RGB/Depth 存在分辨率差异，作者以人工特征点矩阵映射对齐目标区域及周围像素。三种图像均以 PNG 存储，分辨率统一为 1920×1080。 | 同尺寸本身不证明变换无误。论文还称某些序列有 RGB、Depth、TIR 视角不一致的模态扰动；因此不要把全数据集理解成逐像素处处可靠。 |
| 作者训练输入预处理 | 官方 RDTTrack 的 `get_rgbdt_frame` 以 `cv2.imread(depth_path, -1)` 读 Depth；启用 `depth_clip` 时把高于 `min(当前图中位数×3, 10000)` 的值截到阈值，然后逐图 `cv2.normalize(..., 0, 255, NORM_MINMAX)`、转为 uint8、应用 JET 色图，并与 RGB、IR 通道合并。 | 这是作者基线读取/可视化输入流程，不是原始 PNG 的编码规范；代码没有声明毫米、无效值规则或 raw 极性。逐图归一化后的色图不能保留可跨帧比较的尺度。 |

可核实的原始来源：[NeurIPS 论文 PDF（§3.1，第4页）](https://papers.nips.cc/paper_files/paper/2025/file/b4962fcd5d4410a9f43ef70f528eedd8-Paper-Datasets_and_Benchmarks_Track.pdf)、[RGBDT500 作者主页](https://xuefeng-zhu5.github.io/RGBDT500/)、[作者 RDTTrack 读取代码（`rgbdt.py` 第606–643行）](https://github.com/xuefeng-zhu5/RDTTrack/blob/main/lib/train/dataset/rgbdt.py#L606-L643)。主页只复述三模态数据和对齐概况，未补充编码单位。

## SDK 背景、推断与未知项

- **ZED SDK 一般行为（不是 RGBDT500 文件约定）**：[官方深度 API 文档](https://docs.stereolabs.com/docs/development/zed-sdk/modules/depth-sensing/using-the-api)说明 `retrieveMeasure(..., MEASURE::DEPTH)` 返回与左目图像对齐的深度测量；深度单位由 `InitParameters::coordinate_units` 配置，文档示例用毫米。文档还把 `VIEW::DEPTH` 的 8 位灰度显示定义为近处亮、远处暗，并明确该归一化图只用于显示。由于 RGBDT500 论文未给出具体 ZED 型号、SDK 版本、初始化参数或保存代码，不能将 SDK 的默认单位、左目参考或显示方向直接套到数据集 PNG。
- **合理推断**：作者把数据称作 ZED 深度，并由官方代码当作一个可排序的数值阵列截顶后归一化；这说明训练流程消费了像素强度差异。它不能证明归档值等于未经变换的 SDK 深度测量，也不能证明值更大就是更远，或零就是无效。
- **仍未知**：确切 ZED 型号及 SDK 版本；输出的 PNG 来自 `MEASURE::DEPTH` 还是显示视图/其他导出；单位、比例和偏移；数值随距离增大还是减小；0、饱和值及空洞的定义；保存图像以哪一路作像素参考；红外映射矩阵及每帧对齐误差。论文的“像素对齐”是数据集设计主张，无法排除其同时承认的扰动场景或单帧局部误差。

## 本地数据与处理代码的边界

本地采集记录显示抽样 RGBDT500 Depth PNG 可读为 `uint16`，抽样值域为 0–19999；这只是文件与样本观察，不是作者对单位或 sentinel 的说明。[采集脚本](</F:/AIC/code/tools/acquire_rgbdt500.py>)也将 `depth_units` 记作 `unverified`，`depth_visual` 是本地人为设定的固定反向灰度映射（raw 0 显黑、较小正数更亮），只用于查看。该显示映射不能作为原始传感器远近方向证据。

代码入口核对后需限定上述风险：[`native_prompt`](</F:/AIC/code/tools/prepare_qwen3vl_native_sft.py:94>) 有 millimeter、visual 和 `sensor_linear_20000` 三种提示分支；RGBDT 的实际候选入口 [`prepare_next_stage_data`](</F:/AIC/code/tools/prepare_next_stage_data.py:59>) 从记录读取 `depth_policy`，并在逐样本生成时传入提示函数。以当前 RGBDT provisional candidate（`depth_policy=sensor_linear_20000`）做 CPU 格式探针，三图提示确实没有毫米或近远声明。因此没有发现这条候选进入 D/T 的 bbox 路径时被默认误判为毫米的证据。`_nearfar_task` 只接受 `millimeter`/`meter`，不会把 sensor-linear 候选当作度量近远题。

需要区分的是：`prepare_qwen3vl_native_sft.convert_split` 仍直接调用固定的 `_trimodal_prompt`，并按固定毫米范围渲染；该脚本 CLI 明确面向 City manifest。不能把它用于 RGBDT500，但它不是当前 `prepare_next_stage_data` 的 RGBDT 提示入口。City 的 bbox 入口还会保留其已有 conversations，而不会调用 `native_prompt`；所以 City millimeter 参考提示只是在 `native_prompt(..., "millimeter")` 上做的对照，不代表当前 City 样本必然使用这句新提示。格式探针结果详见 [`depth_prompt_audit`](</F:/AIC/results/rgbdt500_annotation_20260925/depth_prompt_audit/audit_report.json>)；未运行训练、未改代码。

上述结论与另一条线的 6 图配准目视诊断互补：论文提供整体采集设计，逐图目视只能帮助标记明显的局部错位/视角扰动；两者都不能据此确定深度单位或 raw 数值方向。该六图盲审中，语义旗标通过不等于框通过最终 IoU 门槛；根 Agent 汇总的最终 IoU/歧义/证据门槛通过数为 4 条（外观 2、实例 2）。

## 最小待确认项与不依赖绝对单位的验证

若以后要启用深度近远问题，最少需要一份可追溯的采集/导出说明，回答：

1. 使用的 ZED 型号、SDK 版本、深度导出 API，以及原始 PNG 是否经过裁剪、归一化、反转或类型转换。
2. 存储值到实际深度的单位/比例/偏移、数值方向，以及 0、无效/空洞和饱和值规则。
3. Depth 与 RGB 的参考图像坐标系、TIR 映射方式；哪些序列或帧有已知视角扰动。

在这些事实未确认前，只能做**受限的经验性相对关系检查**：选几组视觉上有明确前后关系且对齐良好的对象，在原始 Depth 数组上手工取对象内部样本，把 raw=0 单独标记为未解析并暂不纳入有效深度统计，再比较稳健统计量，检查 raw 值顺序是否在多例中一致。此类检查只能提出“这些样例中的 raw 值与相对顺序一致/不一致”的经验结论；不确定全局极性、无效值语义或绝对尺度，也不能单独升级为深度 Query。若关系仅由 RGB 的遮挡或投影位置支持，应标成 RGB 关系，而非深度证据。对方向、配准或有效区域含糊的样本保留 unknown/skip。

与现有本地记录对照：[数据获取状态](</F:/AIC/docs/research/2026-09-25-data-acquisition-status.md>)、[标注工作流](</F:/AIC/docs/research/2026-09-25-rgbdt-annotation-workflow.md>)。本次只查阅公开资料与本地代码/记录；未联系作者、未重下数据、未训练。
