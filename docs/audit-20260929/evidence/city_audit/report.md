# City412 A/B 逐题翻转审计

原始 GT、A/B 单文件预测各有完整 412 个相同 ID；在归一化 xyxy 框上重算 IoU，命中条件为 IoU≥0.5。

A 296/412（0.7184），B 289/412（0.7015），净差 -7 题；同对 286、同错 113、B 救回 3、B 损害 10。A/B mIoU 分别 0.6178/0.6055；mIoU 是 412 题平均，不能归因于仅 13 道翻转。

| 地点序列分组 | 样本 | 同对 | 同错 | 救回 | 损害 | 净差 |
|---|---:|---:|---:|---:|---:|---:|
| known_reuse_47 | 47 | 27 | 18 | 2 | 0 | +2 |
| other_365 | 365 | 259 | 95 | 1 | 10 | -9 |

13 道翻转集中于 11 个三模态图像组。以下数值保留 6 位小数，完整浮点框见 [flips.csv](flips.csv)，原图叠框见 [图册](atlas/cases.html)。

| A→B | ID | 英文 Query | A IoU | B IoU | 目视判别 |
|---|---|---|---:|---:|---|
| 损害 | city_000012_022_00000090_001 | The dark object in the bushes on the right | 0.584411 | 0.336258 | same_object_bad_box |
| 损害 | city_000012_022_00000090_002 | The small object in the bushes on the left | 0.669885 | 0.000000 | wrong_object |
| 损害 | city_000081_006 | The small boat in the distance near the shore | 0.594687 | 0.482763 | same_object_bad_box |
| 损害 | city_000190_005 | The small rectangular sign mounted on the pole to the right | 0.519044 | 0.498298 | unknown |
| 损害 | city_001229_004 | The ceiling light on the right side | 0.875542 | 0.000000 | wrong_object |
| 损害 | city_001897_008 | The small object on the far left floor | 0.717413 | 0.375490 | same_object_bad_box |
| 损害 | city_001897_013 | The leftmost toy car in the background row | 0.566674 | 0.445571 | same_object_bad_box |
| 损害 | city_001927_011 | The fourth robot chassis in the cluster behind the person | 0.616464 | 0.000000 | wrong_object |
| 救回 | city_002022_004 | The ceiling light below the center top one | 0.000000 | 0.920362 | wrong_object |
| 损害 | city_002509_001 | The red garbage can on the far left | 0.516591 | 0.493812 | same_object_bad_box |
| 救回 | city_hehe_224_000002_041_00000197_005 | The gray garbage can on the left | 0.000000 | 0.851419 | wrong_object |
| 救回 | city_hehe_230_000002_043_00000061_013 | The person standing next to the bicycle | 0.000000 | 0.608023 | wrong_object |
| 损害 | city_shuming_215_00000189_004 | The blue plastic seat back visible above the chair stack | 0.880303 | 0.147064 | same_object_bad_box |

## 图组聚集

- 000012_022_00000090.png：救回 0、损害 2；city_000012_022_00000090_001, city_000012_022_00000090_002。
- 000081.png：救回 0、损害 1；city_000081_006。
- 000190.png：救回 0、损害 1；city_000190_005。
- 001229.png：救回 0、损害 1；city_001229_004。
- 001897.png：救回 0、损害 2；city_001897_008, city_001897_013。
- 001927.png：救回 0、损害 1；city_001927_011。
- 002022.png：救回 1、损害 0；city_002022_004。
- 002509.png：救回 0、损害 1；city_002509_001。
- hehe_224_000002_041_00000197.png：救回 1、损害 0；city_hehe_224_000002_041_00000197_005。
- hehe_230_000002_043_00000061.png：救回 1、损害 0；city_hehe_230_000002_043_00000061_013。
- shuming_215_00000189.png：救回 0、损害 1；city_shuming_215_00000189_004。

## 目视证据和局限

- city_000012_022_00000090_001：RGB中三框聚在右侧灌木同一暗目标；B框比GT小且右移，目标细节暗，按框位置仅能判断边界偏差。
- city_000012_022_00000090_002：GT/A框在画面中部灌木的小物体，B框跳到画面最左侧的另一丛灌木；相对GT明显换目标。
- city_000081_006：三框都围住远岸同一艘小船，B框的边缘位置略偏，IoU从0.594687降至0.482763。
- city_000190_005：本地未找到原RGB，B的0.498298只是阈值边缘；无图不能判定是否换对象。
- city_001229_004：A与GT围住较下方的右侧顶灯，B围住其上方另一盏顶灯；Query只说right side，两灯均位于右侧，存在指代歧义。
- city_001897_008：GT/A指向最左侧地面黄色小物体；B仍覆盖该物体，但框向右扩张并包含旁边地面。
- city_001897_013：三框仍在背景排最左侧玩具车，B框收窄偏左，未见换到另一辆车。
- city_001927_011：GT/A位于人右侧背景排的一台底盘，B跳到人左侧另一台底盘；相对GT明显换对象。
- city_002022_004：A框住上方顶灯，B改为下方对应GT的顶灯；B救回的是目标选择。
- city_002509_001：A/B仍框在左远处同一组红色垃圾桶位置，框范围小幅变化导致0.516591→0.493812；原图中相邻红桶增加指代歧义。
- city_hehe_224_000002_041_00000197_005：A框在骑车人处，B框在左下角GT灰色垃圾桶，B救回目标对象。
- city_hehe_230_000002_043_00000061_013：A框在候车牌附近，B框在GT所标的自行车旁人物；B救回目标对象。
- city_shuming_215_00000189_004：1000像素RGB预览显示GT/A围住椅子堆上方蓝色椅背，B只框其顶部窄条；同物但框高度过小。

图册的 GT 框及预测框只叠加在 RGB 原图；IR/Depth 原图供参考。对象级判断以 RGB 上的 Query、GT 和 A/B 框共同核查；图像缺失或遮挡难辨时标未知。

以下只作为后续实验假设：B 的专项训练可能改变了指代对象选择或框边界。需要固定检查点的输入模态干预、逐题检查及独立场景复验，不能由这 13 个 IoU 翻转直接推断因果。
