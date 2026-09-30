# GPU恢复后的执行入口

> **2026-10-01用户明确要求停止，当前禁止接续。** aic定时任务PAUSED，教师采集进程已停止，27135 GPU实查空闲。下列命令仅作为历史记录保留，不能自行执行。城市412未训练对照已412/412完成，不得重跑；正式训练未启动，最终80未评分。

仅使用27135。工作目录为 `/root/autodl-tmp/rematch_20260922/results/visual_agent/evidence_decision_20260930/code`，Python为 `/root/autodl-tmp/rematch_20260922/.venvs/aux_selection/bin/python`。下列命令均使用这个Python。2026-10-01 GPU已恢复；直接C120及未训练120已全部完成，城市412未训练对照分段接续中，正式训练还未开始。已完成阶段不能重复启动，先查看其state、execution和账本。

```bash
cd /root/autodl-tmp/rematch_20260922/results/visual_agent/evidence_decision_20260930/code
source /root/autodl-tmp/rematch_20260922/.venvs/aux_selection/bin/activate
```

## 数据冻结

重启停点见结果目录 `restart_status.json` 和 `teachers/blocked_search_jobs.jsonl`。最新合并为865行，仍未冻结。先恢复真实教师：已执行待审步骤先盲审；待执行的保存动作仍按原参数执行一次，不能把已执行步骤重做。首35相机题中的0178深度角色错误已修复，0162/0157有待审实际步骤；三个搜索待GPU/内存变化后执行。继续补预测框和UNKNOWN/ERROR/EMPTY恢复等缺口，之后才走下述冻结与GPU阶段。不得把865行或旧843行预检当作完整课程已验收。

使用 `collect_object_teachers` 将真实episode与逐步盲审导出为每个bucket的records/decisions；GT仅用于筛掉错误finish。用 `merge_reviewed_data --collect-dir ...` 合并到 `data/train_full.jsonl`，同样本用新轨迹整体替换旧teacher轨迹，不混不同前缀。合理证据动作不会因终局失败被丢弃。检查合并summary的真实动作、工具观察、图组和留出排除，缺的能力继续收集，不以静态finish补足4200。

最终文件必须重新运行全量CPU预检：

```bash
python -m experiments.evidence_decision.train --train ../data/train_full.jsonl --model /root/rematch_models/Qwen3-VL-8B-Instruct --init-adapter /root/autodl-tmp/rematch_20260922/results/next_stage_20260925/c_phase2/checkpoint-500 --output-dir ../cpu_preflight_full_8192 --max-length 8192 --loss-mode token-mean --preflight-only
```

现有已通过的长度文件只对应843行种子，不能用于追加后的数据。保存冻结的 `train_full.jsonl`、长度文件及实际统计，不添加文件哈希。

## GPU阶段

`run_gpu_phase` 默认只打印命令，加 `--execute` 才执行，执行前要求CUDA可用。每个阶段经 `run_capability_stage` 统一结算原20小时预算，计入旧implementation与旧capability ledger，不能另开账。

在教师轨迹仍收集时，可先完成独立holdout的未训练对照。它只依赖120组冻结清单、fresh C预测及对齐后的候选缓存，不依赖不断扩充的`train_full`或其最终长度文件；保持8192上下文，完成后保留这份预测用于后续配对，不重复运行：

```bash
python -m experiments.evidence_decision.run_gpu_phase --phase reference_holdout --execute
# reference_holdout 完成后，以此轮直接C输出固定C初始框；新文件保留所有既有非C候选，不覆盖来源未知的缓存。
python -m experiments.evidence_decision.align_baseline_candidates --candidates ../data/holdout120_candidates.jsonl --baseline-predictions ../reference_holdout/predictions.jsonl --model /root/rematch_models/Qwen3-VL-8B-Instruct --adapter /root/autodl-tmp/rematch_20260922/results/next_stage_20260925/c_phase2/checkpoint-500 --output ../data/holdout120_candidates_frozenC.jsonl
python -m experiments.evidence_decision.run_gpu_phase --phase untrained_holdout --execute
```

如需在最终数据冻结前提早检查训练显存，可从当前已审阅的真实教师决策行建立独立快照（不含合成bbox排练行），在CPU上对这个快照完整做长度预检，再将其最长行用于一次不更新权重的capacity探针。该探针只是当前已收集轨迹的早期信号，不能替代下节最终`train_full`冻结后的全量CPU预检与正式capacity；额外720秒仅在20小时总账仍能保留正式capacity、训练和两次完整412评测预算时运行。

```bash
python - <<'PY'
import json
from pathlib import Path
source = Path('../data/train_full.jsonl')
snapshot = Path('../data/train_capacity_probe_snapshot.jsonl')
rows = [json.loads(line) for line in source.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
teachers = [row for row in rows if row.get('origin') == 'teacher']
if not teachers:
    raise SystemExit('no reviewed real teacher decisions in the current snapshot')
snapshot.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in teachers), encoding='utf-8')
print(json.dumps({'snapshot': str(snapshot), 'teacher_decisions': len(teachers)}, ensure_ascii=False))
PY
python -m experiments.evidence_decision.train --train ../data/train_capacity_probe_snapshot.jsonl --model /root/rematch_models/Qwen3-VL-8B-Instruct --init-adapter /root/autodl-tmp/rematch_20260922/results/next_stage_20260925/c_phase2/checkpoint-500 --output-dir ../cpu_preflight_teacher_probe_8192 --max-length 8192 --loss-mode token-mean --preflight-only
python -m experiments.evidence_decision.run_capability_stage --root .. --old-budget-root ../../implementation_20260928 --prior-ledger ../../capability_rebuild_20260929/ledger.jsonl --name capacity_teacher_probe_8192 --seconds 720 --cwd . -- python -m experiments.evidence_decision.train --train ../data/train_capacity_probe_snapshot.jsonl --model /root/rematch_models/Qwen3-VL-8B-Instruct --init-adapter /root/autodl-tmp/rematch_20260922/results/next_stage_20260925/c_phase2/checkpoint-500 --output-dir ../capacity_teacher_probe_8192 --max-length 8192 --validated-lengths ../cpu_preflight_teacher_probe_8192/preflight_lengths.json --loss-mode token-mean --capacity-only
```

最终`train_full.jsonl`冻结并完成上节全量CPU预检后，再运行正式GPU阶段：

```bash
python -m experiments.evidence_decision.run_gpu_phase --phase capacity --execute
python -m experiments.evidence_decision.run_gpu_phase --phase untrained_city412 --execute
python -m experiments.evidence_decision.run_gpu_phase --phase train --execute
python -m experiments.evidence_decision.run_gpu_phase --phase trained_holdout --execute
python -m experiments.evidence_decision.run_gpu_phase --phase trained_city412 --execute
```

正式capacity对冻结数据最长真实行前后向但不更新权重；训练使用冻结C、语言LoRA r32/alpha64/dropout0.05、lr1e-5、microbatch1/acc8、一遍混合训练、token均值CE，前20个正式步测吞吐。保留8192配置；只有正式全量capacity或训练实际显存OOM才统一改6144、重新全量CPU预检并重跑未训练与训练后接口评测，各阶段加 `--context-tokens 6144`。

阶段硬限只是预算上限。读取每次execution.json及账本的实际完成数、耗时与余量；如果评价达到上限但未跑齐，使用run_objects原配置的 `--resume` 接续并单独结算，减少尚未使用的训练/余量预算，不能把部分样本成绩作为完整412结果。训练前先按未训练实际每题耗时保留完整评价预算。

当前用户已启用本对话每30分钟定时接续。城市412第四段从143题接续，根持久会话的task_gpu_watch_future负责有限后台等待进程退出并释放设备调度锁；watch future尚未完成时，检查只读stage状态，不并发调用read_gpu_phase。每次核对完成即可结束当前agent回合，等待下次定时任务；不再以短间隔主动轮询维持回合。

## 评分与诊断

三臂评分源：

| 队列 | 直接C | 新接口未训练 | 新接口训练后 |
|---|---|---|---|
| 独立120组（40开发+80最终） | ../reference_holdout/predictions.jsonl | ../untrained_holdout/predictions.jsonl | ../trained_holdout/predictions.jsonl |
| 开发412 | ../../implementation_20260928/manifests/city412_c_baseline.jsonl | ../untrained_city412/predictions.jsonl | ../trained_city412/predictions.jsonl |

旧 `capability_rebuild_20260929/c_reference412/predictions.jsonl` 是225/412的旧接口控制器结果，不能当直接C；直接C292/412来自上表implementation中的baseline。120组旧候选缓存的C权重来源未知，因此直接C120须本轮实际重跑，并按上面步骤把新预测对齐到新的 `holdout120_candidates_frozenC.jsonl`。`untrained_holdout` 和 `trained_holdout` 共用该候选池。

`evaluate`要求每个输入预测文件与评分manifest的ID完全一致。先按manifest筛取120组预测；缺少的题目写成 `bbox:null`（不合成坐标），仍在完整分母中计0，重复输入ID会报错。训练前只生成40开发集子集评分；最终80子集只能在训练和配置冻结后生成。

```bash
python -m experiments.evidence_decision.prepare_evaluation_subset --manifest ../data/dev40_manifest.jsonl --run C=../reference_holdout/predictions.jsonl --run untrained=../untrained_holdout/predictions.jsonl --output-dir ../score_inputs_dev40_pretrain
python -m experiments.evidence_decision.evaluate --gt ../data/dev40_gt.json --manifest ../data/dev40_manifest.jsonl --baseline ../score_inputs_dev40_pretrain/C.jsonl --run untrained=../score_inputs_dev40_pretrain/untrained.jsonl --pair C:untrained --output-dir ../score_dev40_pretrain
python -m experiments.evidence_decision.prepare_evaluation_subset --manifest ../data/dev40_manifest.jsonl --run C=../reference_holdout/predictions.jsonl --run untrained=../untrained_holdout/predictions.jsonl --run trained=../trained_holdout/predictions.jsonl --output-dir ../score_inputs_dev40_frozen
python -m experiments.evidence_decision.evaluate --gt ../data/dev40_gt.json --manifest ../data/dev40_manifest.jsonl --baseline ../score_inputs_dev40_frozen/C.jsonl --run untrained=../score_inputs_dev40_frozen/untrained.jsonl --run trained=../score_inputs_dev40_frozen/trained.jsonl --pair C:trained --pair untrained:trained --output-dir ../score_dev40_frozen
# 只在训练和配置确定后运行以下最终80评分：
python -m experiments.evidence_decision.prepare_evaluation_subset --manifest ../data/final80_manifest.jsonl --run C=../reference_holdout/predictions.jsonl --run untrained=../untrained_holdout/predictions.jsonl --run trained=../trained_holdout/predictions.jsonl --output-dir ../score_inputs_final80
python -m experiments.evidence_decision.evaluate --gt ../data/final80_gt.json --manifest ../data/final80_manifest.jsonl --baseline ../score_inputs_final80/C.jsonl --run untrained=../score_inputs_final80/untrained.jsonl --run trained=../score_inputs_final80/trained.jsonl --pair C:trained --pair untrained:trained --output-dir ../score_final80
python -m experiments.evidence_decision.evaluate --gt /root/autodl-tmp/rematch_20260922/results/aux_selection_refine_20260927/manifests/scoring/city412_gt.json --manifest /root/autodl-tmp/rematch_20260922/results/visual_agent/capability_rebuild_20260929/data/city412.jsonl --baseline ../../implementation_20260928/manifests/city412_c_baseline.jsonl --run untrained=../untrained_city412/predictions.jsonl --run trained=../trained_city412/predictions.jsonl --pair C:trained --pair untrained:trained --output-dir ../score_city412
```

完整分母包含非法框、无终局和未完成题。报告ACC@0.5、mIoU、纠正/破坏、合法终局、耗时；用diagnose_capabilities区分初始覆盖、search新增覆盖及predicted_bbox独有救回。

约32个开发状态的证据遮蔽先CPU准备，再单独结算GPU推理：

```bash
python -m experiments.evidence_decision.probe_decisions --traces ../trained_holdout/traces --development-manifest ../data/dev40_manifest.jsonl --count 32 --output-dir ../evidence_probe_inputs
python -m experiments.evidence_decision.run_capability_stage --root .. --old-budget-root ../../implementation_20260928 --prior-ledger ../../capability_rebuild_20260929/ledger.jsonl --name evidence_probe --seconds 1260 --cwd . -- python -m experiments.evidence_decision.probe_decisions --paired-decisions ../evidence_probe_inputs/paired_decisions.jsonl --model /root/rematch_models/Qwen3-VL-8B-Instruct --adapter ../model/adapter --context-tokens 8192 --max-run-seconds 1160 --output-dir ../evidence_probe
```

实际需要新观察的开发状态不足32时记录实际数量，不填静态状态。遮蔽仅作证据依赖诊断，不能混入正式定位分数。所有新成绩和GPU账更新RESULTS/HANDOFF。
