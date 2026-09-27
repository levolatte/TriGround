# 复赛重算报告

完整 manifest 样本数：412。所有 IoU 与 ACC 均由原始 manifest bbox 重新计算。
图像组 bootstrap：10000 次，seed=2026。

## 单模型

| run | 状态 | 样本 | mean IoU | ACC@0.5 | ACC@0.7 | 解析率 |
|---|---|---:|---:|---:|---:|---:|
| M2 | complete | 412 | 0.5983 | 0.6917 | 0.5194 | 1.0000 |
| C | complete | 412 | 0.6136 | 0.7087 | 0.5437 | 1.0000 |
| G_old | complete | 412 | 0.5944 | 0.6990 | 0.4927 | 1.0000 |
| U_old | complete | 412 | 0.5962 | 0.6869 | 0.4903 | 1.0000 |
| B | complete | 412 | 0.5978 | 0.6966 | 0.5049 | 1.0000 |
| Gstar | complete | 412 | 0.5914 | 0.6893 | 0.4927 | 1.0000 |
| Ustar | complete | 412 | 0.5995 | 0.6942 | 0.5073 | 1.0000 |

## 成对比较

仅完整 run 进入成对比较和门槛；partial run 只报告自身重算结果。

### M2 vs C

wrong_to_right：9 IDs，涉及 9 个图像组。
right_to_wrong：2 IDs，涉及 2 个图像组。
ACC@0.5 delta=0.0170，图像组 bootstrap 95% CI=[0.0024, 0.0329]。
mean IoU delta=0.0153，95% CI=[0.0084, 0.0230]。

wrong_to_right IDs：city_000021_004_00000065_005, city_000023_001_00000001_001, city_001229_004, city_001685_008, city_001879_011, city_002509_001, city_003595_005, city_shuming_1132_00000385_005, city_shuming_910_00000290_005
right_to_wrong IDs：city_000004_018_00000341_010, city_001029_003

### M2 vs G_old

wrong_to_right：6 IDs，涉及 6 个图像组。
right_to_wrong：3 IDs，涉及 3 个图像组。
ACC@0.5 delta=0.0073，图像组 bootstrap 95% CI=[-0.0073, 0.0216]。
mean IoU delta=-0.0039，95% CI=[-0.0090, 0.0012]。

wrong_to_right IDs：city_000006_011_00000307_006, city_000019_038_00000205_010, city_000023_001_00000001_001, city_000081_004, city_hehe_230_000002_043_00000061_013, city_shuming_910_00000290_005
right_to_wrong IDs：city_000012_022_00000090_001, city_001029_003, city_001927_011

### M2 vs U_old

wrong_to_right：3 IDs，涉及 3 个图像组。
right_to_wrong：5 IDs，涉及 5 个图像组。
ACC@0.5 delta=-0.0049，图像组 bootstrap 95% CI=[-0.0187, 0.0080]。
mean IoU delta=-0.0021，95% CI=[-0.0068, 0.0029]。

wrong_to_right IDs：city_000006_011_00000307_006, city_hehe_230_000002_043_00000061_013, city_shuming_910_00000290_005
right_to_wrong IDs：city_000081_006, city_001029_003, city_001782_001, city_001927_011, city_shuming_1062_00000100_005

### M2 vs B

wrong_to_right：5 IDs，涉及 5 个图像组。
right_to_wrong：3 IDs，涉及 3 个图像组。
ACC@0.5 delta=0.0049，图像组 bootstrap 95% CI=[-0.0068, 0.0169]。
mean IoU delta=-0.0005，95% CI=[-0.0052, 0.0047]。

wrong_to_right IDs：city_000023_001_00000001_001, city_000081_004, city_001685_008, city_hehe_230_000002_043_00000061_013, city_shuming_910_00000290_005
right_to_wrong IDs：city_000004_018_00000341_010, city_000081_006, city_001927_011

### M2 vs Gstar

wrong_to_right：4 IDs，涉及 4 个图像组。
right_to_wrong：5 IDs，涉及 5 个图像组。
ACC@0.5 delta=-0.0024，图像组 bootstrap 95% CI=[-0.0173, 0.0118]。
mean IoU delta=-0.0069，95% CI=[-0.0135, -0.0009]。

wrong_to_right IDs：city_000023_001_00000001_001, city_001685_008, city_hehe_230_000002_043_00000061_013, city_shuming_910_00000290_005
right_to_wrong IDs：city_000012_022_00000090_001, city_001029_003, city_001927_011, city_hehe_224_000002_041_00000197_005, city_shuming_1062_00000100_002

### M2 vs Ustar

wrong_to_right：4 IDs，涉及 4 个图像组。
right_to_wrong：3 IDs，涉及 3 个图像组。
ACC@0.5 delta=0.0024，图像组 bootstrap 95% CI=[-0.0100, 0.0152]。
mean IoU delta=0.0011，95% CI=[-0.0041, 0.0075]。

wrong_to_right IDs：city_000019_038_00000205_010, city_000081_004, city_003595_005, city_shuming_910_00000290_005
right_to_wrong IDs：city_000012_022_00000090_001, city_001029_003, city_001927_011

### C vs G_old

wrong_to_right：5 IDs，涉及 5 个图像组。
right_to_wrong：9 IDs，涉及 9 个图像组。
ACC@0.5 delta=-0.0097，图像组 bootstrap 95% CI=[-0.0280, 0.0073]。
mean IoU delta=-0.0192，95% CI=[-0.0278, -0.0110]。

wrong_to_right IDs：city_000004_018_00000341_010, city_000006_011_00000307_006, city_000019_038_00000205_010, city_000081_004, city_hehe_230_000002_043_00000061_013
right_to_wrong IDs：city_000012_022_00000090_001, city_000021_004_00000065_005, city_001229_004, city_001685_008, city_001879_011, city_001927_011, city_002509_001, city_003595_005, city_shuming_1132_00000385_005

### C vs U_old

wrong_to_right：3 IDs，涉及 3 个图像组。
right_to_wrong：12 IDs，涉及 12 个图像组。
ACC@0.5 delta=-0.0218，图像组 bootstrap 95% CI=[-0.0404, -0.0048]。
mean IoU delta=-0.0174，95% CI=[-0.0263, -0.0089]。

wrong_to_right IDs：city_000004_018_00000341_010, city_000006_011_00000307_006, city_hehe_230_000002_043_00000061_013
right_to_wrong IDs：city_000021_004_00000065_005, city_000023_001_00000001_001, city_000081_006, city_001229_004, city_001685_008, city_001782_001, city_001879_011, city_001927_011, city_002509_001, city_003595_005, city_shuming_1062_00000100_005, city_shuming_1132_00000385_005

### C vs B

wrong_to_right：3 IDs，涉及 3 个图像组。
right_to_wrong：8 IDs，涉及 8 个图像组。
ACC@0.5 delta=-0.0121，图像组 bootstrap 95% CI=[-0.0267, 0.0000]。
mean IoU delta=-0.0158，95% CI=[-0.0252, -0.0070]。

wrong_to_right IDs：city_000081_004, city_001029_003, city_hehe_230_000002_043_00000061_013
right_to_wrong IDs：city_000021_004_00000065_005, city_000081_006, city_001229_004, city_001879_011, city_001927_011, city_002509_001, city_003595_005, city_shuming_1132_00000385_005

### C vs Gstar

wrong_to_right：2 IDs，涉及 2 个图像组。
right_to_wrong：10 IDs，涉及 10 个图像组。
ACC@0.5 delta=-0.0194，图像组 bootstrap 95% CI=[-0.0362, -0.0045]。
mean IoU delta=-0.0222，95% CI=[-0.0308, -0.0141]。

wrong_to_right IDs：city_000004_018_00000341_010, city_hehe_230_000002_043_00000061_013
right_to_wrong IDs：city_000012_022_00000090_001, city_000021_004_00000065_005, city_001229_004, city_001879_011, city_001927_011, city_002509_001, city_003595_005, city_hehe_224_000002_041_00000197_005, city_shuming_1062_00000100_002, city_shuming_1132_00000385_005

### C vs Ustar

wrong_to_right：3 IDs，涉及 3 个图像组。
right_to_wrong：9 IDs，涉及 9 个图像组。
ACC@0.5 delta=-0.0146，图像组 bootstrap 95% CI=[-0.0315, 0.0000]。
mean IoU delta=-0.0141，95% CI=[-0.0213, -0.0075]。

wrong_to_right IDs：city_000004_018_00000341_010, city_000019_038_00000205_010, city_000081_004
right_to_wrong IDs：city_000012_022_00000090_001, city_000021_004_00000065_005, city_000023_001_00000001_001, city_001229_004, city_001685_008, city_001879_011, city_001927_011, city_002509_001, city_shuming_1132_00000385_005

### G_old vs U_old

wrong_to_right：1 IDs，涉及 1 个图像组。
right_to_wrong：6 IDs，涉及 5 个图像组。
ACC@0.5 delta=-0.0121，图像组 bootstrap 95% CI=[-0.0275, 0.0000]。
mean IoU delta=0.0018，95% CI=[-0.0015, 0.0049]。

wrong_to_right IDs：city_000012_022_00000090_001
right_to_wrong IDs：city_000019_038_00000205_010, city_000023_001_00000001_001, city_000081_004, city_000081_006, city_001782_001, city_shuming_1062_00000100_005

### G_old vs B

wrong_to_right：3 IDs，涉及 3 个图像组。
right_to_wrong：4 IDs，涉及 4 个图像组。
ACC@0.5 delta=-0.0024，图像组 bootstrap 95% CI=[-0.0148, 0.0101]。
mean IoU delta=0.0034，95% CI=[-0.0000, 0.0067]。

wrong_to_right IDs：city_000012_022_00000090_001, city_001029_003, city_001685_008
right_to_wrong IDs：city_000004_018_00000341_010, city_000006_011_00000307_006, city_000019_038_00000205_010, city_000081_006

### G_old vs Gstar

wrong_to_right：1 IDs，涉及 1 个图像组。
right_to_wrong：5 IDs，涉及 5 个图像组。
ACC@0.5 delta=-0.0097，图像组 bootstrap 95% CI=[-0.0216, 0.0000]。
mean IoU delta=-0.0030，95% CI=[-0.0093, 0.0018]。

wrong_to_right IDs：city_001685_008
right_to_wrong IDs：city_000006_011_00000307_006, city_000019_038_00000205_010, city_000081_004, city_hehe_224_000002_041_00000197_005, city_shuming_1062_00000100_002

### G_old vs Ustar

wrong_to_right：1 IDs，涉及 1 个图像组。
right_to_wrong：3 IDs，涉及 3 个图像组。
ACC@0.5 delta=-0.0049，图像组 bootstrap 95% CI=[-0.0148, 0.0047]。
mean IoU delta=0.0051，95% CI=[0.0003, 0.0106]。

wrong_to_right IDs：city_003595_005
right_to_wrong IDs：city_000006_011_00000307_006, city_000023_001_00000001_001, city_hehe_230_000002_043_00000061_013

### U_old vs B

wrong_to_right：6 IDs，涉及 6 个图像组。
right_to_wrong：2 IDs，涉及 2 个图像组。
ACC@0.5 delta=0.0097，图像组 bootstrap 95% CI=[-0.0026, 0.0240]。
mean IoU delta=0.0016，95% CI=[-0.0022, 0.0053]。

wrong_to_right IDs：city_000023_001_00000001_001, city_000081_004, city_001029_003, city_001685_008, city_001782_001, city_shuming_1062_00000100_005
right_to_wrong IDs：city_000004_018_00000341_010, city_000006_011_00000307_006

### U_old vs Gstar

wrong_to_right：5 IDs，涉及 5 个图像组。
right_to_wrong：4 IDs，涉及 4 个图像组。
ACC@0.5 delta=0.0024，图像组 bootstrap 95% CI=[-0.0102, 0.0154]。
mean IoU delta=-0.0048，95% CI=[-0.0107, 0.0002]。

wrong_to_right IDs：city_000023_001_00000001_001, city_000081_006, city_001685_008, city_001782_001, city_shuming_1062_00000100_005
right_to_wrong IDs：city_000006_011_00000307_006, city_000012_022_00000090_001, city_hehe_224_000002_041_00000197_005, city_shuming_1062_00000100_002

### U_old vs Ustar

wrong_to_right：6 IDs，涉及 5 个图像组。
right_to_wrong：3 IDs，涉及 3 个图像组。
ACC@0.5 delta=0.0073，图像组 bootstrap 95% CI=[-0.0074, 0.0234]。
mean IoU delta=0.0033，95% CI=[-0.0025, 0.0095]。

wrong_to_right IDs：city_000019_038_00000205_010, city_000081_004, city_000081_006, city_001782_001, city_003595_005, city_shuming_1062_00000100_005
right_to_wrong IDs：city_000006_011_00000307_006, city_000012_022_00000090_001, city_hehe_230_000002_043_00000061_013

### B vs Gstar

wrong_to_right：2 IDs，涉及 2 个图像组。
right_to_wrong：5 IDs，涉及 5 个图像组。
ACC@0.5 delta=-0.0073，图像组 bootstrap 95% CI=[-0.0190, 0.0024]。
mean IoU delta=-0.0064，95% CI=[-0.0132, -0.0005]。

wrong_to_right IDs：city_000004_018_00000341_010, city_000081_006
right_to_wrong IDs：city_000012_022_00000090_001, city_000081_004, city_001029_003, city_hehe_224_000002_041_00000197_005, city_shuming_1062_00000100_002

### B vs Ustar

wrong_to_right：4 IDs，涉及 4 个图像组。
right_to_wrong：5 IDs，涉及 5 个图像组。
ACC@0.5 delta=-0.0024，图像组 bootstrap 95% CI=[-0.0169, 0.0118]。
mean IoU delta=0.0017，95% CI=[-0.0047, 0.0088]。

wrong_to_right IDs：city_000004_018_00000341_010, city_000019_038_00000205_010, city_000081_006, city_003595_005
right_to_wrong IDs：city_000012_022_00000090_001, city_000023_001_00000001_001, city_001029_003, city_001685_008, city_hehe_230_000002_043_00000061_013

### Gstar vs Ustar

wrong_to_right：5 IDs，涉及 5 个图像组。
right_to_wrong：3 IDs，涉及 3 个图像组。
ACC@0.5 delta=0.0049，图像组 bootstrap 95% CI=[-0.0084, 0.0185]。
mean IoU delta=0.0081，95% CI=[0.0020, 0.0152]。

wrong_to_right IDs：city_000019_038_00000205_010, city_000081_004, city_003595_005, city_hehe_224_000002_041_00000197_005, city_shuming_1062_00000100_002
right_to_wrong IDs：city_000023_001_00000001_001, city_001685_008, city_hehe_230_000002_043_00000061_013

## 候选并集

候选按固定 run 次序 round-robin 合流，候选间 IoU≥0.9 去重；pool 保留全部去重候选，pool_top8 是前 8 个。metadata 中 candidate_cap=8 是选择上限，Recall@all 使用全量候选；GT 只用于离线覆盖评分。

| 候选数 | 覆盖样本 | 覆盖率 |
|---:|---:|---:|
| 1 | 285 | 0.6917 |
| 4 | 296 | 0.7184 |
| 8 | 296 | 0.7184 |
| all | 296 | 0.7184 |

最好单模型：C；相对其完整候选流新增覆盖样本：4，新增图像组：4。
相对其首选框新增覆盖样本：4，新增图像组：4。
required_selection_rate=(B+8)/C，其中 B=292 为最好单模型首选框命中数、C=296 为前 8 个候选覆盖数：1.0135135135135136。

## Run metadata

相邻的 summary/metadata JSON 只被读取并原样嵌入 summary.json；本报告不会修改源文件。

`M2`：`/root/autodl-tmp/rematch_20260922/results/next_stage_20260925/baseline_m2/summary.json`

`C`：`/root/autodl-tmp/rematch_20260922/results/next_stage_20260925/c_step1500/summary.json`

`G_old`：`/root/autodl-tmp/rematch_20260922/results/gu_pilot_20260926/g_2026_city200/summary.json`

`U_old`：`/root/autodl-tmp/rematch_20260922/results/gu_pilot_20260926/u_2026_city200/summary.json`

`B`：`/root/autodl-tmp/rematch_20260922/results/gu_diagnosis_20260927/b_city412/summary.json`

`Gstar`：`/root/autodl-tmp/rematch_20260922/results/gu_diagnosis_20260927/gstar_city412/summary.json`

`Ustar`：`/root/autodl-tmp/rematch_20260922/results/gu_diagnosis_20260927/ustar_city412/summary.json`
