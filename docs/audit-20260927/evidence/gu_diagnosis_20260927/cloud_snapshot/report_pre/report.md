# G/U 诊断结果

原始 GT 重算；City 96 四输入共享同一 ID/GT，fit canonical83 与 legacy181 只在共同 83 ID 上成对比较。输入增益和提示敏感性是描述性差异，不代表已识别因果机制。

fit：181 输出任务，98 个唯一 Query（按 query_key 去重）。

- m2: ACC@0.5=167/181 (0.9227)，mIoU=0.7900，ACC@0.7=151/181 (0.8343)；Query 宏平均 ACC@0.5=0.9337。
- g_old: ACC@0.5=170/181 (0.9392)，mIoU=0.8152，ACC@0.7=152/181 (0.8398)；Query 宏平均 ACC@0.5=0.9456。
- u_old: ACC@0.5=170/181 (0.9392)，mIoU=0.8115，ACC@0.7=154/181 (0.8508)；Query 宏平均 ACC@0.5=0.9456。

City96 输入模式（每模态同 96 ID）

- m2: rgb ACC@0.5=71/96 (0.7396); rgb_ir ACC@0.5=71/96 (0.7396); rgb_depth ACC@0.5=73/96 (0.7604); trimodal ACC@0.5=70/96 (0.7292)
- c: rgb ACC@0.5=72/96 (0.7500); rgb_ir ACC@0.5=73/96 (0.7604); rgb_depth ACC@0.5=70/96 (0.7292); trimodal ACC@0.5=73/96 (0.7604)
- g_old: rgb ACC@0.5=71/96 (0.7396); rgb_ir ACC@0.5=70/96 (0.7292); rgb_depth ACC@0.5=70/96 (0.7292); trimodal ACC@0.5=70/96 (0.7292)
- u_old: rgb ACC@0.5=71/96 (0.7396); rgb_ir ACC@0.5=73/96 (0.7604); rgb_depth ACC@0.5=72/96 (0.7500); trimodal ACC@0.5=68/96 (0.7083)

成对图像组 10,000 次 bootstrap 区间、ACC@0.7、逐模态和逐 Query 统计见 summary.json；完整 City412 比较见 report_city412/report.md。
