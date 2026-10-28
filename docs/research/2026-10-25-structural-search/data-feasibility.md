# 软实例关系记忆：公开标注可行性核对

## 1. RefCOCO / RefCOCO+ / RefCOCOg → COCO 多实例掩码

[REFER 官方代码](https://github.com/lichengunc/refer/blob/master/refer.py)把一条指代表达的 `ann_id`、`image_id` 读入索引，直接以 `ann_id` 查该数据集 `instances.json` 中的实例；实例有 `bbox` 和 `segmentation`，`getMask` 会解码目标掩码。同一 `image_id` 的其他实例可由 `imgToAnns` 取得；若训练要用该图像**全部 COCO 已标注实例**作候选或遮挡约束，应再按 `image_id` 关联原版 COCO `instances_train2014.json`，并以 `ann_id` 校对目标实例，不能把同图全部 `refs` 当成全部实例。可直接监督“表达→目标框/掩码”和 COCO 类别内多实例竞争；用同类其他 COCO 实例作对比候选是可行的，但其是否被文本明确排除并无逐条标签。“左边的人”等参照物在句中没有独立 `ann_id`，须另外匹配候选，关系类型、完整可见物体和深度/红外均无真值。三个数据集各有独立的 `instances.json` 与 `refs(...).p`，图像另取 COCO 2014 train；见[官方目录及下载说明](https://github.com/lichengunc/refer/blob/master/data/README.md)。跨数据集合训时应按 `image_id`、`ann_id` 核对评测重合，不能仅凭文件名或 split 名认定无泄漏。

## 2. Visual Genome / GQA → 实体关系与二维框

[Visual Genome 官方场景图接口](https://visualgenome.org/api/v0/api_endpoint_reference)给每张图的实体 `id`、`x,y,width,height`，关系为 `subject` 实体 ID、`predicate`、`object` 实体 ID，属性也指向实体 ID；可直接监督“实体框—关系边—实体框”的图结构。[GQA 官方下载与格式页](https://cs.stanford.edu/people/dorarad/gqa/download.html)给 `scene_graphs.json`：实体 ID 对应框、属性及指向另一实体 ID 的关系；问题还设有词语到实体 ID 的视觉指针及 `select/filter/relate` 等语义步骤，适合检验指代与关系检索。两者的关系边能训候选间的相对位置和动作谓词，却不能直接训像素级遮挡。但 GQA 的问题任务是回答，不是每条问题都提供本赛题式唯一目标框；Visual Genome/GQA 均无实例掩码、深度和红外，也不能把未记录的关系当负例。GQA 官方场景图约 42.7 MB，问题文件约 1.4 GB；图像需另下。图像域与比赛数据不同，宜做关系预训练或诊断，不能称作比赛样本的完整参照标注。

## 3. PACO → 部件及所属物体

[PACO 官方格式说明](https://github.com/facebookresearch/paco/blob/main/docs/PACO_DATASET.md)的每条物体或部件标注均有 `bbox`、RLE `segmentation`、`category_id`、`image_id`；`obj_ann_id` 可关联部件所属物体实例，另有属性 ID。可直接监督部件框/掩码、部件—整体关联和部分属性；PACO-LVIS 训练集有 45,790 张图、395,071 个部件掩码。其测试集另列结构化 `queries` 与正负实例 ID，但不能等同于任意自由文本的目标及参照物监督；数据只覆盖所选 75 类物体，属性标注并非每框都有，也无深度/红外。[官方仓库说明](https://github.com/facebookresearch/paco/blob/main/README.md)提供 PACO-LVIS 与 PACO-Ego4D 标注包；前者图像另需 COCO 2017，后者帧需接受 Ego4D 使用协议并取得下载凭据。优先用 PACO-LVIS 检验部件模块，不假设它具备全场景关系图。

**组合边界：** 三类数据可分别监督“目标实例”“实体关系”“部件层级”，却不存在现成的“比赛 RGB/红外/深度＋每个候选掩码＋每个参照实体与关系＋部件”联合真值。软记忆中的未标注槽位只能以弱监督或模型预测训练，并单独评估其误差。
