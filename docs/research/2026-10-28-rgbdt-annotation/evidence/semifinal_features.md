# 半决赛Query语言现象与RGBDT补标边界

2026-10-28。来源：`F:/AIC/data/半决赛数据集-基于大模型的多模态视觉理解与推理/queries/queries.json`。仅阅读英文与字段，没有推断或标注官方目标框，没有改动官方数据。以下15例是目的性挑选的语言现象，不能据此估计难例比例、标签质量或模态必要性；本次没有逐题检查对应图像，尤其不把语言看似复杂解释为已证实视觉困难。原文仅作本地内部研究，不复制进训练标注。

已有全量词面口径保存在`F:/AIC/code/docs/research/2026-10-28-rgbdt-annotation/evidence/style_comparison.json`：5691条、1290图组，序数/极值词1943条，明确first–tenth1160条，近远/距离词684条，红外/热/温度词180条。类别可重叠，也有词面误计；例如`light`可能是灯名而非明暗描述。它们只支持“需要覆盖这些表达形式”，不能直接规定最优训练配额。

## 15个代表现象

| ID | 官方Query原文 | 语言现象与新增标注要点 |
|---|---|---|
| 000633_001 | The third vertical railing on the left on the nearest fence | 先选择最近围栏，再限定左侧栏杆并取第三；候选集合不是整图所有栏杆。`on the left`本身没有把起数方向完全写清，不能机械照抄成模板。 |
| 001305_005 | The second small white bird to the right of the boat is swimming alone on the open water near the right side of the lake. | 以船限定子集，再对右侧小白鸟取第二；不是整图第二只鸟。新题应明确从哪个方向或按什么轴排序，不借整图right代替子集定义。 |
| 000430_002 | The second protrusion from left to right on the middle iron pillar | 先找中间铁柱，再在该柱上数凸起；同类序数与部件组合。新题需要柱与凸起的对应，不能把柱的跟踪框用作凸起框。 |
| 003027_005 | The second person counting from the nearest to the farthest | 序数不是横向计数，而是相对相机的近远排序。新题必须有可靠排序依据，不能把画面纵坐标当距离。 |
| 000724_003 | The black swan closest to the ship | 距离参照是船而非相机。新题应区分画面间距、真实空间距离和一般邻接；不能用主体深度排名直接回答“离船最近”。 |
| 000458_003 | The stone with thermal radiation information closest to the white crane nearest to the camera | 先选离相机最近的白鹤，再选与它最近且有热线索的石头；包含参照链、两种距离参照和热属性。只凭文字不能认定IR或Depth都必要，也不能认定“热辐射信息”具有可操作的阈值。 |
| 000857_004 | The right foot of the bear closest to the camera. | 最近熊的右脚，目标是部件；`right`可能指动物身体侧或画面侧。新题要明确视角约定，无法判断时改用可见位置描述；不能整熊框代替脚。 |
| 001004_002 | The middle blade of the three-leaf device at the upper right corner of the environmental air quality monitoring box | 装置依附参照物，目标又是装置三片叶片中的中间片；表现“参照物→装置→部件”的层级。新题核对每层是否唯一及部件是否足够可辨。 |
| 000444_003 | The two white cranes on the left lowering their heads to peck | 两个对象共同构成一个目标单位；需要组合框，不能只选其中一只。新题明确成员和候选范围，框包含两者所有可见部分，夹入背景不等于选错。 |
| 000381_004 | The dark shadow of a tree on the red brick pavement in the foreground | 目标是投影区域，不是树本身或物理物体。新题必须独立定义可见区域边界；原生单目标跟踪框不能直接承担。 |
| 000829_002 | The white line in the middle of the road | 线状区域的范围可能是可见连续段或整条线，文字本身未规定端点。新题限定可见连通段或两个实际端点，避免框边界纯凭标注者习惯。 |
| 000794_005 | The reflection of the fourth wooden stake from the left sticking into the water | 先对实体木桩排序，定位的是相应倒影；实体与倒影不是同一框。新题需明确影像目标，检查倒影是否独立可辨，不能简单上下翻转实体框。 |
| 001487_002 | The black keyboard held in the person's left hand, which appears dark and cold in the thermal image. | RGB颜色、手部参照与IR表现并列；热图中的暗与物理冷之间还隔着编码/标定证据。新题可写可证实的IR暗区域，不直接用灰度宣称绝对低温。 |
| 001529_008 | The peacock whose body appears as the brightest region in the thermal spectrum indicating highest temperature | “最亮”被语言进一步解释为“最高温”。本次不裁决原测试真值；新增RGBDT不能照搬这一推断，需先核实IR显示极性、相对响应及相关候选是否可比。 |
| 001508_005 | The warmest region in the thermal image corresponding to the peacock's body | 目标可能是热区域而非整只鸟，反映“模态定义区域”与物体定义框的区别。新标注必须先定单位、再定RGB对应范围；跨模态区域无法对应时跳过，不能自动套整鸟跟踪框。 |

## 可落实到RGBDT500的内容

1. **优先补可见实例、局部集合与顺序。** 从合法训练图选同类多实例、设施重复结构、密集人物和小部件；记录必要的候选集合、参照对象与排序方向。每图第二条题优先换真实实例或目标单位，而不是近义改写。序数题可以包含颜色，但保留一部分确需子集选择与计数才能唯一回答的题；不为“难”而删掉必要条件。
2. **补层级目标单位，而非只加长句。** 整伞/伞布、人物/人物连器具、装置/其中一个部件、实体/倒影、对象/影子分别标；新目标自行画像素框，原生跟踪框仅供参考。组合要求成员明确；区域要求边界可复核。找不到足够清晰素材则报告缺口，不把所有跟踪对象都拆成三条题。
3. **关系标注只记录必要对象。** 常规题一两项限定即可；真正两层参照才保存参照链，不要求每条都制作复杂图结构。盲复核者必须能单靠原图与Query恢复同一目标和目标单位，不能靠生成者解释补完不清楚的英文。
4. **近远分开处理。** 相对相机的层级可在来源方向、主体有效值和配准确认后制作；相对另一个物体的距离不能只比较两者深度值，也不应含糊混用图像平面邻接与真实三维距离。可以优先做`to the left of / between / beside`等可见位置题；不具备距离证据时不要硬换成closest。
5. **IR从可核查现象起步。** 可标“RGB较难分、IR中主体轮廓/对比更清晰”的实例及对应范围，分别记`rgb_sufficient / ir_helpful / depth_helpful / uncertain`等人工观察；若说最亮，先确认一致显示规则与候选可比，不能把最亮自动改叫最热。绝对温度、人体/动物体温排名、真实热辐射强度缺证据时不做。

## 不能从半决赛语言直接推出或硬补的内容

- “出现closest”不等于Depth必需；“出现thermal”不等于IR决定答案；多层修饰也可能被独特颜色单独解出。必要性应另做证据标记，不能用关键词替代。
- 地面阴影、道路标线、倒影、细叶片、人体脚等不是RGBDT原生4000个跟踪框自带的真值。可以人工补，但要能看清并统一单位；无法复核边界的区域不能为了匹配分布而强行入库。
- 官方措辞中的歧义、重复冗余、冷/亮/温度跳跃不应作为新数据规范。可覆盖相同语言能力，但用独立合法训练图、事实明确的新英文描述。
- 测试Query不提供框定义真值。本次的语言归纳不能证明任何官方题错误，也不能推出哪些具体题视觉困难或哪个模态缺失。
- RGBDT能支持哪些室内、道路、细设施、热或距离案例，要由排除最新测试重叠后的素材实数决定。当前仅34个“未在已恢复旧标注/旧保留/风险序列中出现”的候选上界，不等于34个已获准使用的新场景，更不等于可完整承载所有专项。

建议把“300题”保留为工作预算，前三类（同类集合/参照/目标单位）先落地；距离与热题设上限、无证据转移预算。新增规范应吸收半决赛的目标粒度与关系形式，避免把官方测试句子本身变为训练样本。

## 媒体格式补充

主Agent系统抽样证据在`F:/AIC/code/docs/research/2026-10-28-rgbdt-annotation/evidence/city_semifinal_comparison.json`的`semifinal_systematic_media_sample`。当前65组样本中61组为1920×1080 RGB/IR及16位Depth，4组为640×360且三路均8位RGB；此为抽样数量，等待全1290组读取文件头后的正式总数。IR仅3/65三通道数值完全相等，不能把所有文件当单通道灰度读取；但“通道不完全相等”本身也**不证明采用伪彩编码**，微小通道差异仍可呈现近灰度外观。

实际查看了`Images/infrared/000003.png`、`Images/depth/004133.jpg`及`Images/depth/004413.jpg`，均无目标框或答案推断。000003红外外观近灰度，左右有黑边、场景轮廓偏软；004133深度是8位灰度外观的显示图；004413深度显示为全黑，抽样统计存储值全0。上述只能说明所发媒体的表示差异，不能推出PNG物理距离单位、远近极性、IR温度关系或全黑图的传感故障原因。

因此新标准应增加两项最小记录：原始媒体的尺寸/模式，以及制作时使用的固定预览规则。不能对所有Depth文件统一套16位原始数值映射；已有8位显示图应与原始16位分开处理。盲复核应保留全图与必要局部放大、注意黑边/软轮廓和配准限制；图像在某模态看不到可分离主体时允许记证据不足，不强制制造距离/热监督。源图尺寸与模式不同本身也不能当作场景难度或新旧数据来源的判定依据。
