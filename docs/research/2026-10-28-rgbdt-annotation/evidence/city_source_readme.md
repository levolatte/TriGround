# Qwen City Query 数据划分

本包包含 Qwen 生成并通过本地规则校验的 City 数据集训练/验证清单。

## 文件

- `qwen_generation_train_100.json`：训练集，3,707 条 Query、681 个场景。
- `qwen_generation_val.json`：验证集，412 条 Query、78 个场景。
- `qwen_generation_split_report.json`：划分参数和样本数。
- `qwen_generation_overlap_report.json`：训练/验证重叠检查结果。

`train_100` 表示使用划分后训练分区的 100%，并不是 100 条样本。

## 划分规则

- 源数据：4,119 条可用 Query。
- 训练/验证比例约为 90%/10%。
- 固定随机种子：`5843`。
- 按 RGB 场景整体划分，同一张图的多个 Query 不会跨集合。
- 该种子从候选划分中按类别比例误差选择，并确保 12 个类别在验证集均有覆盖。
- 训练/验证的 Query ID、RGB 场景和源序列重叠均为 0。

## 放置位置与图片要求

两个 manifest 内的图片路径是相对路径。请把 manifest 放在：

`city_detection_prepared/train/target_v2/`

并确保以下三个同级数据目录存在：

- `city_detection_prepared/train/visible/`
- `city_detection_prepared/train/infrared/`
- `city_detection_prepared/train/depth/`

本 manifest 包不包含图片。若接收方没有同一版本的三模态 City 图片，只发送本压缩包不足以开始训练。

## 本地验证结果

- 训练集：3,707 条、681 场景。
- 验证集：412 条、78 场景。
- 12 个类别均在训练集和验证集中出现。
- 4,119 条记录的 Query、归一化 `xyxy` 框和三模态路径全部有效。
- 2,277 个唯一图片文件均可打开，无损坏。
