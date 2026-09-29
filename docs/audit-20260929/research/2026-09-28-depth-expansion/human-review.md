# 2026-09-28 深度扩充包人审回收

43题全部收到决定：42保留、1剔除，无新增备注、无英文改写。已逐行核对完整ID及7个来源/Query字段；原CSV移动到`results/triground_depth_expansion_20260928/review/decisions.csv`，不覆盖其他包。正式accepted_candidates返回42，配对检查没有额外丢题。

累计368条批准：Depth训练56题/25图组，Depth诊断6题/3图组；其他类别为IR训练90、可靠性75、普通竞争56、IR诊断49、RGB足够诊断32、IR候补4。原326条记录保持原样。按已有scene/source+location/sequence检查无train/diagnostic交叉；不代表所有未知地点重用已排除。

唯一剔除题为`city:shuming_443_00000026:depthv2:b_rain_person_car:nearer`，即共撑黑伞右侧人物与黑车的相机距离比较。用户未写原因，不推断其意图。已重新目视RGB与模型Depth，保留旧题/原证据供未来修订判断，本次不改写、恢复批准或加入训练。

600步的_select_release纯数量检查已通过，当前可选训练Depth56/竞争56/IR60/可靠性60，诊断Depth6/IR32/RGB足够32。只是数量检查，不是已经冻结或导出训练release。深度训练距64目标差8；诊断距32差26，距类别总体分析最低16差10。

下一步整理正式合并候选与采样前，仍须落实M3FD002847改为正常IR用途；把全部诊断City地点从A/B/V原City池中排除。6道Depth诊断仅能个案分析，批准不等于模态必要性已验证。未启动GPU/训练，无提交推送。

审计产物：`verification/human_review_final/summary.json`、`accepted_expansion.jsonl`、`rejected_expansion.jsonl`及最新完整`combined_approved_candidates.jsonl`。
