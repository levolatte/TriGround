# C0/A/B/V 原始 GT 重算

总分母 70，图像组 67；已知地点序列 0，其余 70。
完整单文件预测才参加排序、成对比较与入围判断；partial/pending 仅为进度预览。

| 臂 | 状态 | 样本/分母 | ACC@0.5 | mIoU | ACC@0.7 | 已知复用命中 | 其余命中 | 排名 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | complete | 70/70 | 0.8143 | 0.7173 | 0.6857 | 0 | 57 | 1 |
| A | complete | 70/70 | 0.8143 | 0.7121 | 0.6857 | 0 | 57 | 2 |
| B | complete | 70/70 | 0.7714 | 0.6769 | 0.6429 | 0 | 54 | 4 |
| V | complete | 70/70 | 0.7714 | 0.6792 | 0.6571 | 0 | 54 | 3 |

排序：ACC@0.5 → mIoU → ACC@0.7；相对 C0 净增至少 4 命中且救回覆盖至少 3 图像组才入围。

- A：救回 0、伤害 0、净增 +0，救回 0 图像组；未入围。
  救回 IDs：无；伤害 IDs：无；图像组 bootstrap 95% CI [0.0000, 0.0000]。
- B：救回 0、伤害 3、净增 -3，救回 0 图像组；未入围。
  救回 IDs：无；伤害 IDs：abv:diag:robo:0004596:diag_aux, abv:diag:robo:0007227:diag_aux, abv:diag:robo:0007353:diag_aux；图像组 bootstrap 95% CI [-0.0986, 0.0000]。
- V：救回 0、伤害 3、净增 -3，救回 0 图像组；未入围。
  救回 IDs：无；伤害 IDs：abv:diag:robo:0004596:diag_aux, abv:diag:robo:0007227:diag_aux, abv:diag:robo:0007353:diag_aux；图像组 bootstrap 95% CI [-0.0986, 0.0000]。

## 同一检查点的输入模态干预

每条比较固定模型权重、Query 和 GT；左侧为缺失输入，右侧为正常全输入。
- C0:ir_missing：正常全输入救回 0、伤害 0，净效用 +0。
  图组/场景簇 bootstrap 95% CI [0.0000, 0.0000]；救回 IDs：无；伤害 IDs：无。
  depth（N=6）：不足 16，不作类级结论。
  ir（N=32）：救回 0、伤害 0、净 +0。
  rgb_sufficient（N=32）：救回 0、伤害 0、净 +0。
- C0:depth_missing：正常全输入救回 0、伤害 0，净效用 +0。
  图组/场景簇 bootstrap 95% CI [0.0000, 0.0000]；救回 IDs：无；伤害 IDs：无。
  depth（N=6）：不足 16，不作类级结论。
  ir（N=32）：救回 0、伤害 0、净 +0。
  rgb_sufficient（N=32）：救回 0、伤害 0、净 +0。
- C0:both_missing：正常全输入救回 0、伤害 0，净效用 +0。
  图组/场景簇 bootstrap 95% CI [0.0000, 0.0000]；救回 IDs：无；伤害 IDs：无。
  depth（N=6）：不足 16，不作类级结论。
  ir（N=32）：救回 0、伤害 0、净 +0。
  rgb_sufficient（N=32）：救回 0、伤害 0、净 +0。
- A:ir_missing：正常全输入救回 0、伤害 0，净效用 +0。
  图组/场景簇 bootstrap 95% CI [0.0000, 0.0000]；救回 IDs：无；伤害 IDs：无。
  depth（N=6）：不足 16，不作类级结论。
  ir（N=32）：救回 0、伤害 0、净 +0。
  rgb_sufficient（N=32）：救回 0、伤害 0、净 +0。
- A:depth_missing：正常全输入救回 1、伤害 0，净效用 +1。
  图组/场景簇 bootstrap 95% CI [0.0000, 0.0488]；救回 IDs：abv:diag:robo:0004596:diag_aux；伤害 IDs：无。
  depth（N=6）：不足 16，不作类级结论。
  ir（N=32）：救回 0、伤害 0、净 +0。
  rgb_sufficient（N=32）：救回 1、伤害 0、净 +1。
- A:both_missing：正常全输入救回 1、伤害 0，净效用 +1。
  图组/场景簇 bootstrap 95% CI [0.0000, 0.0488]；救回 IDs：abv:diag:robo:0004596:diag_aux；伤害 IDs：无。
  depth（N=6）：不足 16，不作类级结论。
  ir（N=32）：救回 0、伤害 0、净 +0。
  rgb_sufficient（N=32）：救回 1、伤害 0、净 +1。
- B:ir_missing：正常全输入救回 0、伤害 0，净效用 +0。
  图组/场景簇 bootstrap 95% CI [0.0000, 0.0000]；救回 IDs：无；伤害 IDs：无。
  depth（N=6）：不足 16，不作类级结论。
  ir（N=32）：救回 0、伤害 0、净 +0。
  rgb_sufficient（N=32）：救回 0、伤害 0、净 +0。
- B:depth_missing：正常全输入救回 0、伤害 2，净效用 -2。
  图组/场景簇 bootstrap 95% CI [-0.0746, 0.0000]；救回 IDs：无；伤害 IDs：abv:diag:robo:0007227:diag_aux, abv:diag:robo:0007353:diag_aux。
  depth（N=6）：不足 16，不作类级结论。
  ir（N=32）：救回 0、伤害 0、净 +0。
  rgb_sufficient（N=32）：救回 0、伤害 2、净 -2。
- B:both_missing：正常全输入救回 0、伤害 2，净效用 -2。
  图组/场景簇 bootstrap 95% CI [-0.0746, 0.0000]；救回 IDs：无；伤害 IDs：abv:diag:robo:0007227:diag_aux, abv:diag:robo:0007353:diag_aux。
  depth（N=6）：不足 16，不作类级结论。
  ir（N=32）：救回 0、伤害 0、净 +0。
  rgb_sufficient（N=32）：救回 0、伤害 2、净 -2。
- V:ir_missing：正常全输入救回 0、伤害 1，净效用 -1。
  图组/场景簇 bootstrap 95% CI [-0.0441, 0.0000]；救回 IDs：无；伤害 IDs：abv:diag:rgbt_m3fd_train_000698:diag_ir。
  depth（N=6）：不足 16，不作类级结论。
  ir（N=32）：救回 0、伤害 1、净 -1。
  rgb_sufficient（N=32）：救回 0、伤害 0、净 +0。
- V:depth_missing：正常全输入救回 0、伤害 1，净效用 -1。
  图组/场景簇 bootstrap 95% CI [-0.0484, 0.0000]；救回 IDs：无；伤害 IDs：abv:diag:robo:0007227:diag_aux。
  depth（N=6）：不足 16，不作类级结论。
  ir（N=32）：救回 0、伤害 0、净 +0。
  rgb_sufficient（N=32）：救回 0、伤害 1、净 -1。
- V:both_missing：正常全输入救回 0、伤害 2，净效用 -2。
  图组/场景簇 bootstrap 95% CI [-0.0714, 0.0000]；救回 IDs：无；伤害 IDs：abv:diag:rgbt_m3fd_train_000698:diag_ir, abv:diag:robo:0007227:diag_aux。
  depth（N=6）：不足 16，不作类级结论。
  ir（N=32）：救回 0、伤害 1、净 -1。
  rgb_sufficient（N=32）：救回 0、伤害 1、净 -1。

B/V 相对 A 的模态输入效用差（描述性）：
- B:ir_missing：救回差 +0、伤害差 +0、净差 +0。
- B:depth_missing：救回差 -1、伤害差 +2、净差 -3。
- B:both_missing：救回差 -1、伤害差 +2、净差 -3。
- V:ir_missing：救回差 +0、伤害差 +1、净差 -1。
- V:depth_missing：救回差 -1、伤害差 +1、净差 -2。
- V:both_missing：救回差 -1、伤害差 +2、净差 -3。

所有 IoU 均从原始 manifest 的浮点框重新计算。翻转 ID、图像组、图像组 bootstrap 区间、47/365 和类别切片详见 summary.json。
