# A/B/V 候选人审备注与代表图像复核（2026-09-28）

本记录只分析审核反馈，不修改 `decisions.csv`、候选题、框或训练清单。数据来自 `F:/AIC/results/triground_abv_depthv2_20260928/review/decisions.csv`、同包 `data/candidates.jsonl` 与审核图册。CSV 共 327 题：302 保留、25 拒绝；30 条有备注（5 保留、25 拒绝），所有 `approved_query` 为空，所有这 30 条的 `proposed_query` 与 `original_query` 相同。下文“用户判断”指备注和决定；“图册目视”仅用于我实际打开的图，未打开的题不冒称独立确认。保留也只是本次快速人审结论，不能据此宣称已通过 RGB 置空/IR 置空的因果对照。

## Query 和框的真实来源

- RoboRefIt 条目的英文 Query 与框来自 `data/external/RoboRefIt/subset/final_dataset/train/roborefit_train.json` 中的作者行，经 `data/external/RoboRefIt/subset/converted/train_depth_visual_fixed_p01_p99.jsonl` 转换后，`code/tools/prepare_triground_abv_data.py` 的 `robo_candidates` 直接复制 `row["query"]` / `row["bbox"]`。例如 `robo:0002937:reliability` 原作者行 `num=14507` 的文本就是 `pick up the yellow pear`，原框像素 `[62,147,228,329]`。这个原始标注与图像冲突，不能说是本项目写错“梨”。
- FLIR、M3FD、MFAD 条目的英文 Query 与框来自 RGBT-GroundBench 官方 train `.pth` 标注。`code/tools/prepare_rgbt_groundbench.py` 读取 `(filename, size, xywh, query)`，输出到 `results/multimodal_data_20260925/next_sources/rgbt_cloud_train6000.jsonl`；`code/tools/prepare_triground_abv_data.py` 的 `rgbt_candidates` 又原样取 `row["query"]` 与 `row["bbox"]`。例如 `rgbt_mfad_train_007411` 的 `AF1818Z`、FLIR 的 `parked` 都已存在源行。中文审核译文或候选类别是本项目流程，**这 30 条英文题意不是本轮本项目生成/扩写**。这不降低筛除错误源标注的必要性。
- `reliability`、`diag_ir`、`ir_complement`、`reserve_ir` 是本轮分配的用途标签，不等于源标注准确，也不单独证明 IR 真正必要。修订后重新放行应另做人审；空 `approved_query` 不提供任何已批准改写。

## 直接目视结果

| ID | 我打开的图册文件 | 图册目视与用户判断的边界 |
|---|---|---|
| `robo:0002937:reliability` | `assets/bundle_0078_00_answer.jpg` | **图册目视确认**红框是左侧斜放的条纹盒，旁边另有香蕉和浅黄色圆形物；红框不是梨。与用户“盒子不是梨”一致。类别/目标绑定错误，删词不能修好。 |
| `robo:0002006:diag_aux` | `assets/bundle_0098_00_answer.jpg` | **图册目视确认**框在白色细长电子体温计上，左边才有一只小玩具狮子；与用户判断一致。目标绑定/框错误。 |
| `rgbt_flir_train_000108:reliability` | `assets/bundle_0170_00_answer.jpg`、`bundle_0170_infrared.jpg` | **图册目视确认**是靠墙的自行车；标框围住灯杆左侧前部，没有包含灯杆右侧清楚可见的后轮和座架。与用户“不是摩托车、框只覆盖部分”一致。若要重做，需同时重写类名和重新标完整框，不能沿用源框。 |
| `rgbt_m3fd_train_002847:reliability` | `assets/bundle_0230_rgb.jpg`、`bundle_0230_infrared.jpg`、`bundle_0230_00_answer.jpg` | **图册目视确认**红框是路尽头极小的人形，RGB 难辨，IR 同位置亮人形更清楚；用户明确批准并称“很好的红外数据例子”。这符合后续 IR 正常输入候选的方向，却与本轮 `reliability` 所需“RGB 单独已足够”的用途冲突。 |
| `rgbt_m3fd_train_002480:ir_complement` | `assets/bundle_0194_00_answer.jpg`、`bundle_0194_infrared.jpg` | **图册目视确认**RGB 中标注的路边远处人形极小，IR 同一位置有亮目标；支持用户正面评价。框非常小，后续若检验模型收益须保留原浮点框。 |
| `rgbt_m3fd_train_002659:diag_ir` | `assets/bundle_0205_00_answer.jpg`、`bundle_0205_infrared.jpg` | **图册目视确认**浓雾 RGB 几乎遮住框内人，IR 可见前景明亮完整人形。与用户“极其好的红外数据例子”一致。该例还显示 IR 中有另一名远处人，保留时 Query 唯一性应按现有目标框/场景核对。 |
| `rgbt_mfad_train_001861:diag_ir` | `assets/bundle_0276_00_answer.jpg`、`bundle_0276_infrared.jpg` | RGB 里的车身受夜间尾灯与湿路反光影响呈红色外观，**图册目视不能据此稳定确认本色**，IR 灰度也无法证实红色。用户拒绝并称“看不出来是不是红色”，应照此剔除；不要把主观“看着偏红”当真值。 |
| `rgbt_mfad_train_007411:reliability` | `assets/bundle_0280_00_answer.jpg` | 图册牌照可辨末尾 `F18182`（前面还有地区/字母前缀），而源 Query 写 `AF1818Z`；**图册目视确认**尾字符与源文本不符。用户备注是“车牌号是F18182”，不擅自扩写完整号码；车牌改写还需用原分辨率复核。 |

图册文件都位于 `F:/AIC/results/triground_abv_depthv2_20260928/review/assets/`。直接目视只证明上述可见事实，不能从单帧严格判定车辆是否正在运动；下表的“行驶/停泊”按用户人审判断处理。

## 5 条保留且有备注：优先保留的 IR 例子

| ID（当前类别） | 用户判断 | 本次建议 |
|---|---|---|
| `rgbt_m3fd_train_002480:ir_complement` | “非常好的红外互补训练数据例子” | 保留；图册目视同向。后续可作为 RGB 单图与 RGB+IR 成对验证样本。 |
| `rgbt_m3fd_train_003321:ir_complement` | “很好的红外互补训练数据例子” | 保留；本次未独立目视，实验或抽检时优先复看。 |
| `rgbt_m3fd_train_000836:ir_complement` | “很好的例子” | 保留；本次未独立目视。 |
| `rgbt_m3fd_train_002659:diag_ir` | “极其好的红外数据例子” | 保留作高价值留出诊断图，图册目视支持；保持现有诊断用途。 |
| `rgbt_m3fd_train_002847:reliability` | “这其实是很好的红外数据例子” | **保留用户批准及原 CSV/train 分组**，但标记本轮用途冲突：`reliability` 明确要求 RGB 单独足以作答，此图目标主要靠 IR 显现，不应当作 IR 置空可靠性训练样本。后续可另整为 IR 正常输入候选并重新审核。 |

## 25 条拒绝备注：逐项归因和修订候选

这里的“可修订”仅指未来新候选，不代表用户批准了新 Query；本次原候选仍按 `reject`。除上节列出的图册目视项，其余具体视觉判断均仅来自用户备注，未由我逐图复核。

### 目标类别、目标绑定或框错误（5）

| ID | 源 Query 关键表述；用户判断 | 建议 |
|---|---|---|
| `robo:0002937:reliability` | `yellow pear`；框中是盒子 | **图册目视确认**。原题丢弃；若想取梨，重新定位真实目标和框，需重新人审。 |
| `robo:0002006:diag_aux` | `light brown lion`；框中是体温计 | **图册目视确认**。原题丢弃；原图另一处有狮子玩具，但不得拿当前框训练“狮子”。 |
| `rgbt_flir_train_000108:reliability` | `silver motorcycle ... near a pole`；实为自行车且框残缺 | **图册目视确认**。原题丢弃；若重做，需核准自行车整物框与遮挡约定。 |
| `rgbt_m3fd_train_000401:diag_ir` | `small ... lamp`；用户称是摩托车 | 原题丢弃；仅用户判断，重做前看图并重标类别/框。 |
| `rgbt_mfad_train_000306:diag_ir` | `red car follows behind another vehicle`；用户称“图文不符” | 原题丢弃；备注未指出究竟是目标、颜色、关系或框错，需回看图才可归入精确根因。暂列“未定位的图文不符”，不能自作修句。 |

### 动作或状态与画面不符（8）

| ID | 源 Query 与用户判断 | 建议 |
|---|---|---|
| `rgbt_flir_train_004556:ir_complement` | 摩托车 `parked`；用户见有人骑行 | 原题丢弃；可另写只描述目标及可见位置的候选，重新审框和唯一性。 |
| `rgbt_flir_train_006127:ir_complement` | 摩托车 `parked`；用户见有人骑行 | 同上。 |
| `rgbt_flir_train_001988:ir_complement` | 源写 `motorcycle ... parked`；用户称实为**自行车**且有人骑行 | 兼有类别与动作错，改写须两者一起处理，不能只删 `parked`。 |
| `rgbt_flir_train_006096:diag_ir` | `Person walking`；用户称人在骑车 | 删原题；如作骑车人新题，应复核框究竟标人还是人车整体。 |
| `rgbt_mfad_train_008915:ir_complement` | `silver sedan parked`；用户称在行驶 | 删原题；未来仅能用可由画面稳定判断的描述，需新审。 |
| `rgbt_mfad_train_005708:diag_ir` | `red sedan parked`；用户称正在路中间行驶 | 删原题；另需核颜色是否可靠，不能只换状态。 |
| `rgbt_mfad_train_002164:diag_ir` | `silver sedan parked`；用户称在行驶 | 删原题；状态和银色都需复核。 |
| `rgbt_mfad_train_010853:diag_ir` | `red sedan parked`；用户称在行驶 | 删原题；状态和红色都需复核。 |

### 不能从图可靠读出的附加属性（8）

| ID | 源 Query 与用户判断 | 建议 |
|---|---|---|
| `robo:0001950:reliability` | `vitamins in white bottle`；用户无法看清是否维生素 | 原题丢弃；若白瓶对象和框确凿，可新审只保留外观/位置，不借“维生素”语义。 |
| `robo:0006964:reliability` | `baking soda toothpaste`；用户认为不一定是牙膏 | 原题丢弃；若重写，去掉未证实商品用途，重新核目标唯一性。 |
| `rgbt_mfad_train_001861:diag_ir` | `red sedan`；用户无法确认红色 | 原题丢弃；图册目视也无法稳定确认本色。 |
| `rgbt_mfad_train_011152:reliability` | `red sedan`；用户无法确认红色 | 原题丢弃；不把红色作为辨别词。 |
| `rgbt_mfad_train_007411:reliability` | 牌照 `AF1818Z`；用户读为 `F18182` | 原题丢弃；图册目视确认源串错误。新题应优先避免模糊牌照；若牌照是唯一定位线索，需原图分辨率复核字符后再审。 |
| `rgbt_mfad_train_015352:reliability` | `red sedan`；用户无法确认红色，并指出“每次都是红色” | 原题丢弃；对 MFAD `red` 表达做批量抽检，查是否源标注系统性受尾灯/色偏影响。 |
| `rgbt_mfad_train_002003:reliability` | `red sedan`；用户无法辨颜色 | 原题丢弃；颜色不可用。 |
| `rgbt_mfad_train_010539:reliability` | `dark-colored SUV`；用户无法区分深浅 | 原题丢弃；新题不得凭夜景亮度推车身原色。 |

### IR 证据弱或题目过于刁钻（4）

| ID | 用户判断 | 建议 |
|---|---|---|
| `rgbt_mfad_train_005875:ir_complement` | IR 看不清、太刁钻 | 原题丢弃；源还写 `red vehicle`，颜色和小目标需一起复核。 |
| `rgbt_mfad_train_005207:ir_complement` | 太刁钻 | 原题丢弃；框触及图像最右边界，若重做先核是否部分截断及目标唯一性。 |
| `rgbt_mfad_train_015289:reserve_ir` | 太牵强 | 原题丢弃；源写 `red semi-truck`，框极小，先核类别/颜色/框再考虑重做。 |
| `rgbt_mfad_train_015217:reserve_ir` | IR 看不清 | 原题丢弃；若辅助模态不能看清，不宜作 IR 互补例子。 |

## 给下一步处理的明确建议

1. 保持 25 条原题拒绝，不从备注自动产出已批准改写；`approved_query` 全空。可修订项应建立新的 Query/框候选，经用户重新看图，尤其目标类别/框错题不能用纯文本修复。
2. 优先保留 M3FD 的 5 条正面反馈作为分层样例，尤其 `002847`、`002659`、`002480`。`002659` 保持留出诊断用途；`002847` 只作为后续 IR 正常输入候选，勿用于本轮 IR 置空可靠性训练。在最终实验报告中分别列“用户认可”“图册目视”“模型 IR 消融获益”，避免把三者混为一项证据。
3. 对 RGBT-GroundBench MFAD 源标注做有针对性的预筛：`parked`/`walking` 等动作状态，`red`/`silver`/`dark-colored` 等夜间颜色，精细车牌，小目标边界。当前拒绝说明的是**源 Query 和标注质量问题**，不是本轮候选生成器凭空加词；本项目的问题是未在入选前筛掉这些源错误。
4. `rgbt_m3fd_train_002847:reliability` 的用户批准、原 CSV 与 train 分组均无需动，但其**用途与本轮可靠性目标冲突**：该目标要求 RGB 单独已足够、去掉 IR 仍能定位，图册所见却是 IR 明显提供目标证据。放行前的用途筛选应将它排除于本轮 IR 置空可靠性训练；后续仅作为 IR 正常输入候选重新整理和审核，不改动现有诊断/训练 split。
