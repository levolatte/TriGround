# 第一轮 A/B 运行命令

以下命令在 Linux 的现有 27B 训练/推理环境执行。只需按实际位置设置路径；`BUNDLE` 同时包含 `annotations/`、`media/`、`evaluation/`。`CITY_ROOT` 下应有完整 `visible/`、`infrared/`、`depth/`。`ARM2` 固定为既有 checkpoint-500，两臂从它初始化并新建优化器。

```bash
set -euo pipefail
REPO=/root/AIC_code
PY=/root/miniconda3/bin/python
SWIFT_CLI=/root/miniconda3/bin/swift
VLLM_PY=/root/vllm_env/bin/python
MODEL=/root/rematch_models/Qwen3.6-27B
ARM2=/root/autodl-tmp/runs/q36_arm2_capability/archived/checkpoint-500
BUNDLE=/root/autodl-tmp/grounding_aux_round1/bundle
PREP=/root/autodl-tmp/grounding_aux_round1/prepared
RUNS=/root/autodl-tmp/runs/grounding_aux_round1
CITY_ROOT=/root/autodl-tmp/rematch_20260922/data/city/train
export PY SWIFT_CLI
cd "$REPO"
mkdir -p "$PREP" "$RUNS"
```

## 1. 构造目标机清单与模板预编码

包内标注使用相对 `media/...` 路径。以下步骤生成目标机的绝对媒体路径，不直接使用Windows生成的A/B文件。

```bash
"$PY" tools/prepare_grounding_aux.py \
  --pairs "$BUNDLE/annotations/pairs.jsonl" --image-root "$BUNDLE" \
  --output-dir "$PREP" --seed 2026

DATA_DIR="$PREP" MODEL="$MODEL" PREFLIGHT_ONLY=1 \
  bash scripts/train_grounding_aux_ab.sh
```

预编码使用真实 `qwen3_5` 模板，`max_length=None`统计未经截断的全长，再核对8192上限、有效回答和结束符。完整训练入口也执行这一步。

## 2. 最小GPU冒烟：两任务反向传播、保存、重载

取B臂固定前16行，包含12条框、4条辅助，执行2个优化步。目录应为空，避免覆盖旧运行。

```bash
DATA_DIR="$PREP" MODEL="$MODEL" ADAPTER="$ARM2" ARM=B \
  OUTPUT_ROOT="$RUNS/smoke" SMOKE=1 \
  bash scripts/train_grounding_aux_ab.sh

SMOKE_CK="$(find "$RUNS/smoke/b-smoke" -type d -name checkpoint-2 -print)"
test -f "$SMOKE_CK/adapter_config.json"
head -n 2 "$PREP/aux_eval.jsonl" > "$RUNS/aux_smoke2.jsonl"
"$VLLM_PY" tools/evaluate_grounding_aux_ab.py infer-aux \
  --model "$MODEL" --adapter "$SMOKE_CK" \
  --manifest "$RUNS/aux_smoke2.jsonl" --output "$RUNS/aux_smoke2_predictions.jsonl" \
  --upscale 1 --max-pixels 2073600 --model-max-length 8192 \
  --max-num-seqs 2 --chunk-size 2
```

这是检查训练后适配器可重载的最小运行，不用于报告准确率。检查训练日志中的有效损失、实际LoRA命中模块、两步保存记录，以及重载推理输出；不要把仅生成文件当作已经证明模型学习有效。

## 3. 固定240步的正式对照

```bash
for ARM in A B; do
  DATA_DIR="$PREP" MODEL="$MODEL" ADAPTER="$ARM2" ARM="$ARM" \
    OUTPUT_ROOT="$RUNS" SMOKE=0 \
    bash scripts/train_grounding_aux_ab.sh
done

A240="$(find "$RUNS/a" -type d -name checkpoint-240 -print)"
B240="$(find "$RUNS/b" -type d -name checkpoint-240 -print)"
test -f "$A240/adapter_config.json"
test -f "$B240/adapter_config.json"
```

每臂应只有一份新运行；若找到多个路径，`test -f`会失败，应明确选择本轮目录。第80、160步仅保留诊断，主比较不挑最佳步数。可用本机已有适配器目录 `F:/AIC/deliverables/作品材料/模型文件` 作为上传到目标机的Arm2来源。

## 4. 两臂辅助40题

```bash
for ARM in A B; do
  if [ "$ARM" = A ]; then CK="$A240"; else CK="$B240"; fi
  "${VLLM_PY}" tools/evaluate_grounding_aux_ab.py infer-aux \
    --model "$MODEL" --adapter "$CK" --manifest "$PREP/aux_eval.jsonl" \
    --output "$RUNS/aux_${ARM}_predictions.jsonl" \
    --upscale 2 --max-pixels 8294400 --model-max-length 32768 \
    --max-num-seqs 8 --chunk-size 8
  "$PY" tools/evaluate_grounding_aux_ab.py aux \
    --manifest "$PREP/aux_eval.jsonl" --predictions "$RUNS/aux_${ARM}_predictions.jsonl" \
    --output-dir "$RUNS/aux_${ARM}_score"
done
```

## 5. City412与历史外部94题：单次直接定位

两臂共用现役City提示、关闭思考、确定性生成和k2输入。`--limit`等于全量，仅避免自动生成比赛提交压缩包，仍处理集合全部题目。外部集使用自身的深度说明。

```bash
for ARM in A B; do
  if [ "$ARM" = A ]; then CK="$A240"; else CK="$B240"; fi
  "$VLLM_PY" tools/predict_submission_vllm.py \
    --model "$MODEL" --adapter "$CK" --modality trimodal \
    --queries "$BUNDLE/evaluation/city/queries.json" --data-root "$CITY_ROOT" \
    --expected-query-count 412 --limit 412 --single-pass \
    --min-pixels 200704 --max-pixels 8294400 --model-max-length 32768 --upscale 2 \
    --max-num-seqs 8 --chunk-size 8 --max-lora-rank 32 \
    --output-dir "$RUNS/eval/city/$ARM"

  "$VLLM_PY" tools/predict_submission_vllm.py \
    --model "$MODEL" --adapter "$CK" --modality trimodal \
    --queries "$BUNDLE/evaluation/external/queries.json" --data-root "$BUNDLE" \
    --expected-query-count 94 --limit 94 --single-pass --depth-policy sensor_linear_20000 \
    --min-pixels 200704 --max-pixels 8294400 --model-max-length 32768 --upscale 2 \
    --max-num-seqs 8 --chunk-size 8 --max-lora-rank 32 \
    --output-dir "$RUNS/eval/external/$ARM"
done
```

## 6. 比较与判读

```bash
for SET in city external; do
  "$PY" tools/evaluate_grounding_aux_ab.py compare \
    --gt "$BUNDLE/evaluation/$SET/gt.json" \
    --queries "$BUNDLE/evaluation/$SET/queries.json" \
    --groups "$BUNDLE/evaluation/$SET/groups.json" \
    --subsets "$BUNDLE/evaluation/$SET/subsets.json" \
    --arm "A=$RUNS/eval/$SET/A/predictions.jsonl" \
    --arm "B=$RUNS/eval/$SET/B/predictions.jsonl" \
    --output-dir "$RUNS/compare/$SET"
done
```

阅读 `report.md`、`summary.json` 和逐题 `cases.jsonl`。City主指标分母400，另看321回归与79专项；12未决和412全量视图只提供预测诊断。核对缺失/非法/回退计数、救回和损害图组、配对区间，再结合外部诊断与辅助准确率判断。辅助题提升不能替代直接定位改善。
