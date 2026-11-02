"""Run Arm2 with the completed correction run's vLLM configuration and prepared inputs."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.predict_vllm_quant_submission import main as predict, build_llm_options
from tools.predict_quant_submission import load_queries, package_predictions
from tools.eval_quant_grounding import read_jsonl
from tools.quant_grounding_reward import parse_bbox
from tools.quant_grounding_postprocess import normalize_bbox_completion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--reference-run', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    ref = json.loads((args.reference_run / 'full/run_config.json').read_text())
    previous = [x for x in ref['resume_history'] if x.get('engine') == 'vllm'][-1]
    assert build_llm_options(ref, 8) == previous['options']
    state = dict(status='running', stage='loading', pid=os.getpid(), expected_queries=5691,
                 reference_run=str(args.reference_run), adapter=str(args.adapter))
    def save(**values):
        state.update(values, updated_at=datetime.now(timezone.utc).isoformat())
        (args.output_dir / 'state.json').write_text(json.dumps(state, indent=2)+'\n')
    save()
    try:
        argv = ['--dataset', str(args.reference_run / 'full/unlabeled.jsonl'),
                '--output-dir', str(args.output_dir / 'full'), '--adapter', str(args.adapter),
                '--chunk-size', str(previous['chunk_size']), '--max-num-seqs', '8', '--upscale', '1.0']
        for key in ('data_root', 'model', 'min_pixels', 'max_pixels', 'max_length', 'max_tokens', 'seed'):
            argv += ['--'+key.replace('_','-'), str(ref[key])]
        if args.resume:
            argv.append("--resume")
        summary = predict(argv)
        save(stage='packaging')
        full = args.output_dir / 'full'
        actual = json.loads((full / 'run_config.json').read_text())
        assert actual['llm_options'] == previous['options']
        source = load_queries(Path(ref['queries']), 5691)
        raw = read_jsonl(full / 'predictions_raw.jsonl')
        assert [x['id'] for x in raw] == list(source)
        normalized = []
        wrapped_arrays = 0
        for row in raw:
            sample = row['samples'][0]
            text = sample['text']
            if parse_bbox(normalize_bbox_completion(text, 'short', ref['format_policy']), 'short') is None:
                payload = json.loads(text)
                if not isinstance(payload, list) or len(payload) != 4:
                    raise ValueError(f"{row['id']}: expected one complete four-coordinate array")
                canonical = json.dumps({'bbox_2d': payload})
                if parse_bbox(canonical, 'short') is None:
                    raise ValueError(f"{row['id']}: invalid coordinate array")
                sample = {**sample, 'text': canonical}
                wrapped_arrays += 1
            normalized.append({**row, 'samples': [sample]})
        summary.update(package_predictions(source, normalized, full, ref['format_policy']))
        summary['bare_coordinate_arrays_wrapped'] = wrapped_arrays
        summary['packaging_policy'] = 'punctuation plus a complete bare four-coordinate JSON array; coordinates unchanged' 
        summary.update(reference_run=str(args.reference_run), only_model_change='Arm2 checkpoint-500 adapter',
                       comparison_note='Reference official package uses HF for its first 1720 queries; this run uses vLLM for all queries.')
        (full / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        save(status='completed', stage='completed', completed_queries=len(raw), submission=str(full / 'submission.zip'))
    except Exception as error:
        save(status='failed', stage='failed', error=f'{type(error).__name__}: {error}')
        raise

if __name__ == '__main__':
    main()
