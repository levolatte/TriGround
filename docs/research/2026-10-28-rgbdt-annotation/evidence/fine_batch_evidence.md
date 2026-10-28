# RGBDT500 精标批次溯源与风格证据

2026-10-28。仅本地只读盘点；下文“本人精标”指用户组织、亲自验收的批次，不等于每句英文由用户亲手撰写。

## 来源与数量

- 原始完整对象证据：`F:/AIC/code/docs/audit-20260927/datasets/rgbdt_trial200/reviewed_evidence.jsonl`，200图组、209对象、138条自然Query。独立Agent复核后252输出项通过、67待复核；这份快照的`human_approved=0`，不能整份称人审已通过。
- 真正人工决策：`F:/AIC/code/docs/audit-20260927/datasets/human_decisions/natural100_seed2026_human_decisions_2026-09-26.csv`（自然描述100项：98正确唯一、1不确定、1错）与同目录`correspondence100_seed2026_human_decisions_2026-09-26.csv`（对应100项：97接受、1不确定、2拒绝）。
- 实际放行：`F:/AIC/code/docs/audit-20260927/datasets/released181/native_sft.jsonl`、`metadata.jsonl`、`summary.json`。181是输出任务数：RGB98、IR58、Depth25；独立自然Query与图组均98，原候选另5拒绝、66未决。
- `F:/AIC/data/external/RGBDT500/converted`目前不存在。归档`released181`为可直接引用的权威训练内容副本。
- 生产来源：`F:/AIC/code/docs/audit-20260927/research/2026-09-26-object-evidence-trial200.md`明确“LUNA-MAX制作、SOL独立盲审”；`2026-09-26-human-review-depth-and-errors.md`明确收到用户两份CSV并导出181任务。正确表述应为“Agent看图制作、独立复核、用户人工验收”。
- 后续`F:/AIC/data/grounding_aux_round1_20261025/inventory/rgbdt_train.jsonl`有98行/98序列，`label_source=released181_rgb`，其98条英文与人审通过98条逐字相同。013一行ID由`::q01`展开为`::rgbdt500_013_00000160_q01`，不属新增。
- 后续`bundle/annotations/pairs.jsonl`共266行，含226训练对与40评估对。RGBDT训练148对（错对象74/37图，完整范围74/74图），并集79图/79序列，全部来自上述98图组。148行含144条不同英文Query，其中66条与旧Query逐字相同，其余含重写与第二实例；不是148个新图组，也不能与旧98相加。
- 完整范围初稿76条，74接受、2拒绝（217和219攀爬架部件范围不清）；错对象初稿74条全部接受。旧98中有19序列没进入最终两个训练池，不等于被判错。
- `F:/AIC/code/docs/research/2026-10-25-grounding-aux-round1/pairs.jsonl`与`F:/AIC/data/grounding_aux_round1_20261025/bundle/annotations/pairs.jsonl`逐行内容完全相同，是同批副本。

## 旧98条风格

按英文词正则`[A-Za-z]+(?:'[A-Za-z]+)?`统计：6–20词，中位11，均值11.51。句首Where47、The32、Which8、Locate9、Find2。词面含left/right22条，near/beside/next to/by39条，front/behind/foreground/background25条，明确first/second/third/fourth/leftmost/rightmost1条；infrared/thermal/hot/heat为0，depth/nearer/farther/closest/farthest仅1（076的nearer）。这是词面口径，不是推理能力真值。

对应对象字段`extent`：whole85、part11、group2。`part`既包含明确部件（伞布），也包含被遮挡/截断的整对象（大象、鹦鹉、坐着的人），不能直接解释成11条专门部件定位题。两个group为簸箕+扫帚、动物群。

核心风格是简短自然英文，先对象名/外观，再1–2个可见动作、位置或参照限定；以单目标自然定位和跨模态对象对应为主。并未充分覆盖密集序数、参照链与真正IR/Depth必要题。

证据字段保留`objects.semantic_label/boxes/confirmed_modalities/extent/evidence/uncertainties`；Query有`target_object_id/answerable_modalities/augmentation_eligible/review_status/modality_reviews`，逐模态保留独立预测框、IoU和语义意见。不能只复制一列Query而丢掉目标单位与证据链。

## 可引用例子

以下行号均指自然100人工CSV，除另注外均`correct_unique`。

| 行 | 图组 | 原句 | 体现/问题 |
|---:|---|---|---|
| 33 | 211/00000184 | The leftmost of the three large birds standing by the back fence. | 同类+最左+后方围栏；少量显式序数型 |
| 42 | 114/00000159 | Where is the pedestrian by the railing who stands to the left of the other two people on the walkway? | 同类人物参照关系，较长20词 |
| 55 | 013/00000160 | The red dustpan and broom being used together by the cleaner in gray. | 明确组合目标，不能只框簸箕或扫帚 |
| 62 | 310/00000337 | Where is the umbrella canopy over the outdoor seating area? | 明确伞布部件，不能套整把伞/支撑架 |
| 38 | 076/00000093 | Where is the white animal facing away behind the nearer white animal? | 前后参照，但nearer是图像可见关系，非已证实深度单位监督 |
| 27 | 005/00000198 | Where is the person in dark clothing standing beside an upright skateboard on the plaza? | 原句通过；后续复核改为long dark coat bending over...。已查看三模态复核图，人物确有弯腰，说明人审可用仍可精化动作 |
| 65 | 087/00000167 | Locate the dark bear standing beside the large rock in the enclosure. | uncertain；历史报告记用户指出只框右半边，未放行 |
| 81 | 229/00000082 | The partly visible person standing by the doorway on the right. | wrong；历史报告明确实际鹦鹉误描述成人，关联RGB/IR均排除 |

后续005复核图：`F:/AIC/data/grounding_aux_round1_20261025/review/rgbdt_extent/001_rgbdt500_005_00000198__q01__extent_01.jpg`。最终Query在`annotations/rgbdt_extent_review.jsonl:1`，描述从standing精化为bending over，并校验完整人物可见范围。

注意自然CSV中文备注有编码损坏，错误原因引用上述历史人审报告更稳；英文、ID、决策可直接使用。即使`human_accepted`也不证明多模态是必要条件。
