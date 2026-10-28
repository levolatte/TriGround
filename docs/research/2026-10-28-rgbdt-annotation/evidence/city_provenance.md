# City 训练与开发标签来源

## 三种数据身份

| 数据 | 本地直接证据 | 数量与用途 |
|---|---|---|
| City 外部训练原标签 | `F:/AIC/data/quant_grounding_source/original_city_labels.zip` 中 `qwen_generation_train_100.json` | 3,707 Query、681 RGB 图组；不是复赛官方测试标签 |
| City 原开发标签 | 同 ZIP 的 `qwen_generation_val.json` | 412 Query、78 RGB 图组；与训练同一批来源和生产流程，历史开发集 |
| City 局部修订开发标签 | `F:/AIC/data/quant_grounding_source/city_val_corrected.json`；原成果位于 `F:/AIC/docs/research/2026-10-24-label-fix/` | 仍保留全部412个ID；只对91个历史错例开展本轮审查 |
| 复赛官方测试输入 | 历史本地路径 `F:/AIC/data/复赛数据集-基于大模型的多模态视觉理解与推理/queries/queries.json` | 历史盘点为5,690 Query/1,295图组，无公开GT；本次未读取或标注该数据 |

不要因 City 图像与部分官方输入同源或重合，就把 City 清单当成官方测试GT；也不要把旧 City `target_v2.zip` 的923/119人工修订划分、992弱监督划分与本批3707/412混成一个版本。相关旧批次出处为 `docs/research/2026-09-18-target-v2-labels.md`。

## 原 City 批次的生产方式

ZIP 内还保留 `qwen_generation_DATASET_README.md`、`qwen_generation_split_report.json` 与 `qwen_generation_overlap_report.json`。报告的源清单为 `D:/AIC/city_detection_prepared/train/target_v2/qwen_generation_reviewed.json`，seed=5843、validation_fraction=0.1、group_key=null；README记述“Qwen生成并通过本地规则校验”，按RGB图组整组拆分，确保12个旧类别均有覆盖。`train_100` 指训练分区100%，不是100条数据。

本次直接解析确认：训练3707、开发412，合计4119，全部 `review_decision=query_corrected`，都保留 `original_query`。训练有3144种不同Query、开发398种；两分区所有句子均以“The ”开头，词数中位9/P90=11。训练Query最大18词，开发最大14词。旧类别字段仅12种，但并不可靠对应修订Query的目标类别。

`F:/AIC/code/tools/enrich_city_queries.py:57` 按框中心x排序生成同类左右极值和序数，`:84` 将单类目标配最近异类参照物，否则配画面区域。这解释了许多 `original_query` 中的几何模板。**现有包没有Qwen调用提示、实际输入模态、逐条人工复核日志，不能因为文件名reviewed或字段query_corrected就宣称4119条已逐条人审、已检查三模态或已核定框范围。** 该脚本与旧Query形式吻合，也不等于拿到了这批Qwen调用本身。

历史原图使用位置为 `/root/autodl-tmp/rematch_20260922/data/city/train/{visible,infrared,depth}`，清单路径相对 `target_v2/` 解析。本批每图平均约5.4条Query，与队友RGBDT批次“一帧一目标Query、每序列约1–3帧”的结构不同，不能只比总Query数。

## 412、400、378是同一开发集逐步净化

`F:/AIC/docs/research/2026-10-24-label-fix/README.md:3` 明确：审查对象是此前91个未命中，旧321个命中项本轮没有人工复查。`corrections.json` 直接记录51 corrected、28 keep、11 deferred、1 partial。本次对比原/新清单确认52条记录变化：27条改Query、27条改框，二者重叠2条。其余360条记录不变。

| 名称 | 构成 | 仍有的边界 |
|---|---|---|
| City412 | 原412个ID，应用已确定的52条修改 | 321个旧命中未重审；11暂缓、1部分修正保留原值或只改文案 |
| City400 | 412去除11 deferred和1 partial | 不是400条全量新人工精标；仍含321个未重审旧命中 |
| City378 | 在400上再剔除官方测试重叠4帧、22条有效题 | 74帧；仍是开发诊断集，不能称最终独立测试集 |

构造代码直接见 `F:/AIC/code/tools/prepare_quant_grounding.py:266`（排除未决）、`:271`（按官方重叠帧过滤）；正式口径见 `F:/AIC/code/docs/research/2026-10-26-quant-rl/README.md:123`、`:137`。本次仅用本地原/新清单、corrections与 `data/quant_grounding_source/recovered_city_test_exclusions.json` 重新计算，得到412→400→378、最终74帧，与报告一致；未读取官方测试像素。

原训练3707按同份已恢复排除清单，净量3178条/571帧，剔529条/110帧。正式训练再按组封存后City训练2882、封存296；这是后续量化实验的数据净量，不代表历史复赛每次运行都用了2882。正式数见同 README:121。

**本地 `data/quant_grounding_20261026/city400_dev.jsonl` 实为0行**，对应summary mode=`nonformal_local_pool`、dev_rows.city400=0；该目录只具有本地可用外部数据。正式实验开发文件在云端 `data/quant_grounding_20261026_formal/city400_dev.jsonl`，文件名沿用400而实际378。本地完整开发标签来源应引用上面的412原清单及过滤规则，不能把空JSONL当成378条。

## 五个本地真实Query及其意义

1. **训练，city_000002_082_00000121_001**：`The red and white sign on the blue tricycle`。原Query为 `The sign, above the nearby tricycle`，class_name仍是tricycle。说明改写增加外观，但旧类别元数据不能直接用于类别平衡或同类负例。
2. **训练，city_000003_002_00000178_003**：`The blue bicycle parked behind the white scooter`。原Query为 `The 6th bicycle from the left, near the center of the image`。说明生产方式从几何序数转向外观关系，不等于原序数已被独立确认。
3. **训练，city_000127_007**：`The set of three silver bins on the left`。目标单位明确是一组三个桶，不能在后续训练中擅自改为单个实例。
4. **开发修订，city_000004_018_00000341_009**：旧 `The leftmost person in the group on the bridge`；新 `The leftmost person in the cluster immediately to the right of the central double-headed street light on the bridge`。修订限定候选群体，消除“哪一群人”的歧义；框保持原目标。
5. **开发修订，city_hehe_230_000002_043_00000061_002**：旧 `The blue rectangular sign on the far left`；新 `The pair of blue rectangular signs mounted on the far left pole`。修订明确成对牌面，不应把支撑杆或整组道路设施混入目标。

前三条直接来自训练JSON；后两条来自原/新开发JSON逐ID对照。这五例均非本次对官方输入新增的标注。

## 已有局限与新标准的直接启示

`F:/AIC/docs/research/2026-09-25-training-data-deep-audit.md:39` 记录原批Query虽有外观多样性，但没有证据证明生成时实际查看IR/Depth；训练显式红外/热词0、显式depth/米词0。该报告还记录目标框内深度全无效713/3707，训练/开发至少7组已目视确认的同物理场景不同帧跨划分；这些是历史证据边界，不等于本次重新验证了全部图像。

10月24日局部修订直接暴露的可确认问题包括：参照群体未限定、类别与所框实体不一致、同一人物可见范围被截断、单灯头/完整双头灯、牌面/含杆、单件/成组目标，以及同类计数候选集合不清。暂缓项也证明部分微小/遮挡目标不能可靠确定，不能靠润色英语消除。

因此半决赛新增标注标准应先审定**目标单位与框范围→同图候选集合→唯一指代证据→辅助模态是否真的消歧**，再写自然英语。继承City的同图多目标覆盖，补上可靠实例ID和参照对象ID；继承RGBDT精标的实际图像审核，避免把Qwen自审当独立复核。根据半决赛官方输入的实际词长、序数、遮挡、小目标与模态特征调整配比，不以City或队友批次的短句形式直接充当统一标准。按场景/序列分割训练和开发，保留412/400/378不同版本名称及来源，不混比分母。
