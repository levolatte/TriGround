"""Read-only teacher-forced action-token audit on real multimodal decisions."""
import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
import json
from pathlib import Path
import time

from .model import QwenBackend
from .train import action_token_indices, encode_decision, read_decisions


def select_rows(rows):
    controls = Counter()
    selected = []
    for index, row in enumerate(rows):
        kind = json.loads(row['target_action'])['action']
        if kind in ('search_candidates', 'measure_depth') or controls[kind] < 8:
            selected.append((index, row))
            controls[kind] += 1
    return selected


def adapter_stats(model):
    stats = defaultdict(lambda: {'tensors': 0, 'nonzero_elements': 0, 'elements': 0, 'squared_norm': 0.0})
    for name, tensor in model.named_parameters():
        if 'lora_' not in name:
            continue
        adapter = 'comparison' if '.comparison.' in name else 'default'
        family = 'A' if 'lora_A' in name else 'B'
        item = stats[adapter + '_' + family]
        item['tensors'] += 1
        item['elements'] += tensor.numel()
        item['nonzero_elements'] += int(tensor.count_nonzero().item())
        item['squared_norm'] += float(tensor.float().square().sum().item())
    return dict(stats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decisions', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--comparison-adapter')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    rows = read_decisions(args.decisions)
    selected = select_rows(rows)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    backend = QwenBackend(args.model, args.adapter)
    torch, model, tokenizer = backend.torch, backend.model, backend.processor.tokenizer
    if args.comparison_adapter:
        model.load_adapter(args.comparison_adapter, adapter_name='comparison',
                           is_trainable=False, autocast_adapter_dtype=False, local_files_only=True)
        model.to(device=backend.device, dtype=torch.bfloat16)
    model.eval()
    metadata = {'diagnostic_only': True, 'input': str(args.decisions),
                'selected_row_indices': [i for i, _ in selected],
                'target_action_counts': dict(Counter(json.loads(r['target_action'])['action'] for _, r in selected)),
                'adapters': {'T3': args.adapter, 'T2': args.comparison_adapter},
                'adapter_tensor_stats': adapter_stats(model),
                'interpretation': 'Teacher-forced probabilities on training states, not autonomous task accuracy.'}
    (args.output_dir/'config.json').write_text(json.dumps(metadata, indent=2))
    records = []
    modes = [('base', None)] + ([('T2', 'comparison')] if args.comparison_adapter else []) + [('T3', 'default')]
    with (args.output_dir/'rows.jsonl').open('w') as handle:
        for ordinal, (index, row) in enumerate(selected):
            encoded = encode_decision(backend.processor, row)
            labels = encoded.pop('labels')[0].to(backend.device)
            positions = (labels != -100).nonzero(as_tuple=True)[0]
            targets = labels[positions]
            action_indices = action_token_indices(tokenizer, row['target_action'], targets.tolist())
            other_indices = [i for i in range(len(targets)) if i not in action_indices]
            inputs = {k: value.to(backend.device) for k, value in encoded.items()}
            record = {'row_index': index, 'id': row['id'], 'target_action': row['target_action'],
                      'target_kind': json.loads(row['target_action'])['action'], 'origin': row.get('origin'),
                      'mask': {'input_tokens': len(labels), 'supervised_tokens': len(targets),
                               'action_value_indices': action_indices,
                               'action_value_tokens': tokenizer.convert_ids_to_tokens(targets[action_indices].tolist()),
                               'first_supervised_position': int(positions[0]),
                               'last_supervised_position': int(positions[-1]),
                               'final_supervised_token': tokenizer.decode(targets[-1:])}, 'models': {}}
            for name, adapter in modes:
                model.set_adapter(adapter or 'default')
                context = model.disable_adapter() if adapter is None else nullcontext()
                with context, torch.inference_mode():
                    logits = model(**inputs, logits_to_keep=positions-1).logits[0].float()
                    losses = torch.nn.functional.cross_entropy(logits, targets, reduction='none')
                    first = action_indices[0]
                    probabilities = logits[first].softmax(-1)
                    top = probabilities.topk(5)
                    result = {'supervised_ce': float(losses.mean()),
                              'action_value_ce': float(losses[action_indices].mean()),
                              'other_supervised_ce': float(losses[other_indices].mean()),
                              'first_action_target_probability': float(probabilities[targets[first]]),
                              'first_action_target_token': tokenizer.decode(targets[first:first+1]),
                              'top5': [{'token': tokenizer.decode([int(token)]), 'probability': float(prob)}
                                       for token, prob in zip(top.indices, top.values)],
                              'active_adapters': list(model.active_adapters), 'adapters_disabled': adapter is None}
                    if ordinal == 0 and name == 'T3':
                        full = model(**inputs).logits[0]
                        projected = full[positions-1].float()
                        full_ce = torch.nn.functional.cross_entropy(projected, targets)
                        result['real_qwen_full_vs_sparse'] = {
                            'max_logit_difference': float((projected-logits).abs().max()),
                            'mean_logit_difference': float((projected-logits).abs().mean()),
                            'full_masked_ce': float(full_ce), 'sparse_ce': float(losses.mean()),
                            'absolute_ce_difference': float((full_ce-losses.mean()).abs())}
                        del full, projected
                    record['models'][name] = result
                    del logits, losses, probabilities
            records.append(record)
            handle.write(json.dumps(record)+'\n')
            handle.flush()
            print(json.dumps({'completed': len(records), 'total': len(selected)}), flush=True)
    grouped = {}
    for mode, _ in modes:
        grouped[mode] = {}
        for kind in metadata['target_action_counts']:
            subset = [r['models'][mode] for r in records if r['target_kind'] == kind]
            grouped[mode][kind] = {'n': len(subset), **{key: sum(r[key] for r in subset)/len(subset)
                for key in ('supervised_ce', 'action_value_ce', 'other_supervised_ce', 'first_action_target_probability')}}
    summary = {**metadata, 'complete': len(records) == len(selected), 'rows': len(records),
               'elapsed_seconds': time.perf_counter()-started, 'groups': grouped,
               'real_qwen_full_vs_sparse': records[0]['models']['T3']['real_qwen_full_vs_sparse']}
    (args.output_dir/'summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
