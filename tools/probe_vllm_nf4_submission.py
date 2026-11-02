"""用官方vLLM bitsandbytes插件实测纠错适配器，不修改正式预测。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.quant_grounding_postprocess import normalize_bbox_completion
from tools.quant_grounding_reward import parse_bbox


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-num-seqs', type=int, default=8)
    parser.add_argument('--quantization', choices=['bitsandbytes', 'fp8_per_tensor'], default='bitsandbytes')
    args = parser.parse_args()
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    config = json.loads((args.run_dir / 'full/run_config.json').read_text())
    selected = json.loads((args.run_dir / 'batch_benchmark/report.json').read_text())['sample_ids']
    dataset = {row['id']: row for row in map(json.loads, (args.run_dir / 'full/unlabeled.jsonl').read_text().splitlines())}
    rows = [dataset[sample_id] for sample_id in selected]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {'status': 'loading', 'quantization': args.quantization,
              'sample_ids': selected, 'max_num_seqs': args.max_num_seqs}
    report_path = args.output_dir / 'report.json'

    def save(**values):
        report.update(values)
        report_path.write_text(json.dumps(report, indent=2) + '\n')

    save()
    processor = AutoProcessor.from_pretrained(config['model'])
    requests = []
    for row in rows:
        images = []
        for path in row['images']:
            with Image.open(path) as image:
                images.append(image.convert('RGB').copy())
        chunks = row['messages'][0]['content'].split('<image>')
        if len(chunks) != len(images) + 1:
            raise ValueError('image placeholders differ from image count')
        content = []
        for index, chunk in enumerate(chunks):
            if chunk:
                content.append({'type': 'text', 'text': chunk})
            if index < len(images):
                content.append({'type': 'image', 'image': images[index]})
        prompt = processor.apply_chat_template([{'role': 'user', 'content': content}],
                                               tokenize=False, add_generation_prompt=True, enable_thinking=False)
        requests.append({'prompt': prompt, 'multi_modal_data': {'image': images}})
    options = dict(model=config['model'], dtype='bfloat16', max_model_len=config['max_length'],
                   max_num_seqs=args.max_num_seqs, max_num_batched_tokens=8192,
                   gpu_memory_utilization=.90, enable_lora=True, max_lora_rank=32,
                   enable_prefix_caching=False, mamba_cache_mode='none', enforce_eager=True,
                   limit_mm_per_prompt={'image': 3}, seed=config['seed'],
                   mm_processor_kwargs={'min_pixels': config['min_pixels'], 'max_pixels': config['max_pixels']},
                   quantization=args.quantization)
    if args.quantization == 'bitsandbytes':
        options.update(load_format='bitsandbytes', hf_overrides={'quantization_config': {
            'quant_method': 'bitsandbytes', 'load_in_4bit': True, 'load_in_8bit': False,
            'bnb_4bit_compute_dtype': 'bfloat16', 'bnb_4bit_quant_type': 'nf4',
            'bnb_4bit_use_double_quant': True, 'llm_int8_skip_modules': ['visual', 'lm_head', 'in_proj_ba'],
        }})
    else:
        options.update(quantization_config={'ignore': ['*visual*', '*in_proj_ba*', '*lm_head*']})
    save(options=options)
    started = time.monotonic()
    try:
        llm = LLM(**options)
        save(status='generating', loading_seconds=time.monotonic() - started)
        lora = LoRARequest('correction1600', 1, config['adapter'])
        sampling = SamplingParams(temperature=0, max_tokens=config['max_tokens'], seed=config['seed'])
        # Keep first-use setup separate from steady throughput.
        llm.generate(requests[:1], sampling, lora_request=lora, use_tqdm=False)
        started = time.monotonic()
        outputs = llm.generate(requests, sampling, lora_request=lora, use_tqdm=False)
        elapsed = time.monotonic() - started
        raw = []
        for sample_id, output in zip(selected, outputs):
            sample = output.outputs[0]
            raw.append({'id': sample_id, 'prompt_token_ids': list(output.prompt_token_ids),
                        'samples': [{'text': sample.text, 'tokens': len(sample.token_ids),
                                     'finish_reason': sample.finish_reason, 'seconds': elapsed / len(outputs)}]})
        (args.output_dir / 'predictions_raw.jsonl').write_text(
            ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in raw))
        reference = [json.loads(line) for line in (args.run_dir / 'batch_benchmark/batch2/predictions_raw.jsonl').read_text().splitlines()]
        texts = [row['samples'][0]['text'] for row in raw]
        reference_text = [row['samples'][0]['text'] for row in reference]
        boxes = [parse_bbox(normalize_bbox_completion(text, 'short', config['format_policy']), 'short') for text in texts]
        reference_boxes = [parse_bbox(normalize_bbox_completion(text, 'short', config['format_policy']), 'short') for text in reference_text]
        save(status='completed', generation_seconds=elapsed, seconds_per_query=elapsed / len(outputs),
             valid_boxes=sum(box is not None for box in boxes), text_equal=texts == reference_text,
             boxes_equal=boxes == reference_boxes,
             changed_ids=[sample_id for sample_id, x, y in zip(selected, texts, reference_text) if x != y])
        print(json.dumps(report), flush=True)
    except Exception as error:
        save(status='failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
