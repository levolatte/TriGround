# T线独立SFT对接审查（2026-09-29）

本审查检查训练、轨迹导出、在线运行和离线评分入口，并补齐离线失败终局标签；未占用GPU，未修改公共控制器或评价核心。v4主管已停止，dev-fast子批次可自行完成；capacity与T250不得沿用偏RGB的机制自动启动。下列命令只在跨模态取证机制修订、重新冻结并生成完整T250真实轨迹后执行，不代表已有SFT成绩。

## 已确认的接口

- 冻结清单 `t_train200.jsonl` / `t_holdout50.jsonl` 本地分别为200/50题，ID与RGB图组交集均为0。导出器还与 `train32.jsonl` 做图组隔离；在线运行不接收GT。
- `export_trajectories.py` 只把训练分区逐决策写到 `train.jsonl`，留出分区写到 `holdout.jsonl`。终局标签只从当时可见的合法候选池和GT构造；KEEP、候选未覆盖的情况分开处理。训练入口只读 `train.jsonl`。
- `train.py` 从原生基座创建独立语言注意力 q/k/v/o 的 rank32 LoRA；没有加载C适配器。真实Processor的标签掩码6/6检查已由主线完成，此审查没有重复该检查。
- `evaluate.py` 对 `--manifest` 要求完整、无额外ID的预测集合。T250原生轨迹的250行输出不能直接用于50题留出评分。

## 实际T执行顺序（等待修订后的T250）

先核对T250原生运行的 `run_config.json` 为修订版、native/D/sft/latest，`execution.json.complete=true` 且预测恰有250题；核对原始轨迹确有IR隐藏热源、模糊RGB和可靠Depth前中后层级的取证行为。然后把 `CODE`、`T_RUN` 设为**实际**部署目录与完整运行目录。以下输出目录应是新目录；先前 preflight 的 `training_config.json` 会锁定配置，不能复用于两步训练。

```bash
set -e
R=/root/autodl-tmp/rematch_20260922
V="$R/results/visual_agent/implementation_20260928"
PY="$R/.venvs/aux_selection/bin/python"
MODEL=/root/rematch_models/Qwen3-VL-8B-Instruct
CODE=...   # 修订版的实际部署代码绝对路径
T_RUN=...  # 修订版完整native D/sft/latest T250运行目录
EXPORT="$V/t_sft_export_revised"
PREFLIGHT="$V/t_preflight_revised"
SMOKE="$V/t_smoke_revised"
FULL="$V/t_train_revised"
EVAL="$V/t_eval_revised"
cd "$CODE"
record_t() {
  "$PY" - "$V/t_budget.jsonl" "$1" "$2" "$3" <<'PY'
import json, sys
ledger, task, kind, report = sys.argv[1:]
with open(report, encoding="utf-8") as handle:
    data = json.load(handle)
seconds = data["invocations"][-1]["gpu_stage_seconds"] if kind == "train" else data["invocation_wall_seconds"]
with open(ledger, encoding="utf-8") as handle:
    entries = [json.loads(line) for line in handle if line.strip()]
if any(entry.get("task") == task for entry in entries):
    raise SystemExit(f"T budget task already recorded: {task}")
with open(ledger, "a", encoding="utf-8") as handle:
    handle.write(json.dumps({"phase": "t-sft", "task": task, "elapsed_seconds": seconds,
                             "budget_hours": 8, "source": report}) + "\n")
print(f"T GPU budget used: {(sum(row['elapsed_seconds'] for row in entries) + seconds)/3600:.3f}/8 h")
PY
}

"$PY" -m experiments.visual_agent.export_trajectories \
  --traces "$T_RUN/traces" --train-manifest "$V/manifests/t_train200.jsonl" \
  --holdout-manifest "$V/manifests/t_holdout50.jsonl" \
  --debug-manifest "$V/manifests/train32.jsonl" \
  --gt "$V/scoring/t250_gt.json" --output-dir "$EXPORT"
"$PY" -c 'import json,sys; x=json.load(open(sys.argv[1])); assert (x["collected_train_trajectories"],x["collected_holdout_trajectories"],x["missing_train_starts"],x["missing_holdout_starts"])==(200,50,0,0), x' "$EXPORT/inventory.json"
"$PY" -m experiments.visual_agent.train --train "$EXPORT/train.jsonl" \
  --model "$MODEL" --output-dir "$PREFLIGHT" --preflight-only

"$PY" -m experiments.visual_agent.train --train "$EXPORT/train.jsonl" \
  --model "$MODEL" --output-dir "$SMOKE" --epochs 2 --max-steps 2 \
  --save-steps 1 --stop-after-step 1
record_t sft_smoke_step1 train "$SMOKE/train_summary.json"
test -s "$SMOKE/checkpoint-1/trainer_state.json"
test -s "$SMOKE/checkpoint-1/optimizer.pt"
"$PY" -m experiments.visual_agent.train --train "$EXPORT/train.jsonl" \
  --model "$MODEL" --output-dir "$SMOKE" --epochs 2 --max-steps 2 \
  --save-steps 1 --resume "$SMOKE/checkpoint-1"
record_t sft_smoke_step2 train "$SMOKE/train_summary.json"
"$PY" -c 'import json,sys; a,b=json.load(open(sys.argv[1]))["invocations"][-2:]; assert (a["final_global_step"],a["training_complete"],b["resume_start_step"],b["optimizer_steps_this_invocation"],b["final_global_step"],b["training_complete"])==(1,False,1,1,2,True), (a,b)' "$SMOKE/train_summary.json"
```

导出后先读 `inventory.json`：`collected_train_trajectories=200`、`collected_holdout_trajectories=50`、两个 `missing_*_starts=0`。`train_trajectories` 是 KEEP 上限和合法监督筛选后的实际数量，可能小于200。CPU preflight 必须全部通过4096 token检查。两次 smoke 使用同一数据、目录、`epochs/max_steps/save_steps`；第二次必须从 `checkpoint-1` 恢复，不能从第一步导出的 `SMOKE/adapter` 初始化。检查 `SMOKE/train_summary.json` 两个 invocation：第一条 `final_global_step=1, training_complete=false`，第二条 `resume_start_step=1, optimizer_steps_this_invocation=1, final_global_step=2, training_complete=true`。`Trainer.train(resume_from_checkpoint=...)` 是恢复接口；检查点的 `optimizer.pt` 与训练后FP32优化器审计提供状态证据。

两步通过后，在**同一 Bash 会话**和**全新** `FULL` 目录执行正式训练；若重新登录，先恢复上段变量与 `record_t` 函数。按共享T预算先为模型加载、最终保存和留出50题推理预留时间，再从剩余额度确定 `TRAIN_SECONDS`；墙钟到限会在优化器步后保存检查点。若 `training_complete=false`，用 `train_summary.json` 的 `last_checkpoint` 在同一 `FULL` 目录续跑，继续保持 `--epochs 2 --save-steps 50`、同一训练文件与基座。只有最后一次 `training_complete=true` 才用于评价。`train.py` 保存的推理适配器固定在 `FULL/adapter`；`run.py` 的 t-lora 接口以原生基座和这个目录加载 PEFT 适配器。

```bash
TRAIN_SECONDS=...  # 根据V/t_budget.jsonl的余额、实测装载/保存和留出推理成本填写
"$PY" -m experiments.visual_agent.train --train "$EXPORT/train.jsonl" \
  --model "$MODEL" --output-dir "$FULL" --epochs 2 --save-steps 50 \
  --max-run-seconds "$TRAIN_SECONDS"
record_t sft_full_attempt1 train "$FULL/train_summary.json"
"$PY" -c 'import json,sys; x=json.load(open(sys.argv[1]))["invocations"][-1]; assert x["training_complete"] and x["max_steps"]==-1, x' "$FULL/train_summary.json"

"$PY" -m experiments.visual_agent.subset_predictions \
  --manifest "$V/manifests/t_holdout50.jsonl" \
  --predictions "$T_RUN/predictions.jsonl" --output "$EVAL/native_holdout50.jsonl"
"$PY" -m experiments.visual_agent.subset_predictions \
  --manifest "$V/manifests/t_holdout50.jsonl" \
  --predictions "$V/baseline250/predictions.jsonl" --output "$EVAL/c_holdout50.jsonl"
"$PY" -m experiments.visual_agent.run --mechanism D --controller t-lora \
  --profile sft --memory latest --model "$MODEL" --adapter "$FULL/adapter" \
  --manifest "$V/manifests/t_holdout50.jsonl" \
  --candidate-cache "$V/t250/candidates.jsonl" \
  --dino-model "$R/models/grounding-dino-tiny" \
  --sam-model "$R/models/sam2.1-hiera-tiny" \
  --output-dir "$EVAL/trained_holdout50"
record_t sft_holdout50 run "$EVAL/trained_holdout50/execution.json"
"$PY" -m experiments.visual_agent.evaluate \
  --gt "$V/scoring/t250_gt.json" --manifest "$V/manifests/t_holdout50.jsonl" \
  --baseline "$EVAL/c_holdout50.jsonl" \
  --run "N=$EVAL/native_holdout50.jsonl" \
  --run "T=$EVAL/trained_holdout50/predictions.jsonl" \
  --pair C:N --pair N:T --output-dir "$V/scores/t_holdout50_revised"
```

`subset_predictions.py` 只按冻结50题清单筛选原生T250和C250预测，不读取GT；留出评价的分母始终是完整50题，训练标签筛选不影响分母。正式训练和留出必须沿用同一修订版控制器与 `sft/latest` 输入；不要混入旧v3/v4偏RGB轨迹。

共享 `V/t_budget.jsonl` 中已有T250 `batch.py` 运行记录，每条 `elapsed_seconds` 已含该次模型装载与工具等待，**不要再为T250追加一条队列总时长**。每次 smoke、正式训练及其续跑，在完成后各追加一条 `budget_hours:8` 记录，`elapsed_seconds` 取对应 `train_summary.json` 新 invocation 的 `gpu_stage_seconds`（包括模型装载、训练、保存；CPU preflight不计）。留出推理单独追加 `execution.json.invocation_wall_seconds`，评分与子集筛选不计GPU。每次启动前累加全 ledger 的 `elapsed_seconds`，确认加预留成本不超过8小时；若GPU任务在写 summary 前失败，也要按实际占卡墙钟记入，不因失败漏账。

## 导出清单中的轨迹统计

`inventory.json` 现区分读入的全部原始轨迹 `input_trajectories`、属于冻结T训练/留出清单的 `collected_train_trajectories` / `collected_holdout_trajectories`，以及经监督标签和KEEP上限筛选后真正进入JSONL的 `train_trajectories` / `holdout_trajectories`。每个分区的 `origin` 与在线 `final_status` 均按轨迹计数；旧的模糊 `train_origins` 改为明确的 `train_origin_decisions`，同时有 `train_origin_trajectories`。原始轨迹文件不被修改或丢弃。

训练来源数据构成分别见 `collected_train_composition_trajectories`（已采集的冻结训练来源）和 `train_composition_trajectories`（实际导出的训练来源）。五个可重叠计数依次表示：UNKNOWN/EMPTY真实观察后又调用不同真实工具（finish不算）、真实搜索增加候选ID、初始C正确且在线最终KEEP、初始C错误且在线最终框命中GT、离线终局目标相对原模型动作发生修订。这些计数不包含留出50题，也不改变轨迹或标签选择。

## 失败终局的离线监督

本地v3调试轨迹中可见失败终局事件的 `raw_output`、`usage.output_tokens`、`messages` 与 `candidates_before`。导出器现对 `INVALID_FINAL_ACTION` 且末次真实生成非空输出的轨迹，尝试把**末次决策**的监督标签修订为当时KEEP或合法目标候选的 `finish`；既不读取末次决策后的候选，也不生成新框。原始 `original_action`、`original_raw_output`、`online_final_status=INVALID_FINAL_ACTION` 和单列的 `label_source=gt_keep_repaired_invalid_final` / `gt_candidate_repaired_invalid_final` 保存在导出行中；在线预测仍是失败，不计为成功。GT未被当时候选覆盖、空输出、零输出token或输入预算耗尽时，不补终局标签，已发生的合法工具动作仍可导出。

用本地v3 train32调试轨迹做只读兼容检查：128条轨迹中有9条 `INVALID_FINAL_ACTION`，修订逻辑可为其中4条构造当时合法的终局监督，另5条因无合法GT命中保持未修订。这是导出行为检查，不是T线训练数据或成绩。

本地子集入口两项测试已通过；`python -m unittest experiments.visual_agent.tests.test_sft -v` 为7通过、1项真实模型环境测试跳过，覆盖失败终局候选/KEEP修订、未来候选排除、空生成与预算耗尽排除、两类轨迹计数及UNKNOWN/EMPTY工具切换。本次 `train.py` / `export_trajectories.py` / `subset_predictions.py` 通过 `py_compile`；真实两步恢复、正式训练和T50推理仍须在重新分配GPU后执行。
