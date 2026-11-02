"""Read-only lexical comparison; word matches are not semantic labels."""
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path('F:/AIC')
OUT = Path(__file__).resolve().parent

def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

fine = jsonl(ROOT / 'data/quant_grounding_source/rgbdt_train.jsonl')
teammate = jsonl(OUT / 'teammate_recovered919.jsonl')
pairs = [r for r in jsonl(ROOT / 'data/grounding_aux_round1_20261025/bundle/annotations/pairs.jsonl')
         if r['source'] == 'rgbdt500' and r['split'] == 'train']
semifinal_path = ROOT / 'data/半决赛数据集-基于大模型的多模态视觉理解与推理/queries/queries.json'
semifinal = [dict(id=k, **v) for k, v in json.loads(semifinal_path.read_text(encoding='utf-8')).items()]
patterns = {
    'color_brightness': r'\b(?:white|black|red|green|blue|yellow|orange|brown|gray|grey|pink|purple|teal|beige|dark|light|bright)\b',
    'left_right': r'\b(?:left|right|leftmost|rightmost)\b',
    'ordinal_extreme': r'\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|leftmost|rightmost|nearest|farthest|furthest|closest)\b',
    'ordinal_only': r'\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b',
    'spatial_relation': r'\b(?:near|beside|next to|in front of|behind|between|above|below|under|over|foreground|background|back)\b',
    'depth_distance': r'\b(?:depth|nearer|closer|farther|further|nearest|closest|farthest|furthest)\b',
    'infrared_thermal': r'\b(?:infrared|thermal|temperature|temperatures|hot|heat|hotter|hottest|warm|warmer|warmest|cold|colder|coldest)\b',
    'part_words': r'\b(?:head|face|arm|leg|tail|canopy|wheel|handle|roof|door|window|sign|panel|board|lamp|light|hat)\b',
}

def stats(rows):
    queries = [r['query'] for r in rows]
    lengths = sorted(len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", q)) for q in queries)
    return {
        'n': len(rows), 'unique_queries': len(set(queries)),
        'words': {'min': min(lengths), 'median': statistics.median(lengths),
                  'mean': statistics.mean(lengths), 'p90_nearest_rank': lengths[math.ceil(.9*len(lengths))-1],
                  'max': max(lengths)},
        'no_english_words': [{'id': r['id'], 'query': r['query']} for r in rows if not re.search('[A-Za-z]', r['query'])],
        'starts': dict(Counter((re.findall(r'[A-Za-z]+', q) or ['<no English word>'])[0].lower() for q in queries).most_common()),
        'lexical': {key: sum(bool(re.search(pattern, q, re.I)) for q in queries) for key, pattern in patterns.items()},
    }

fine_keys = {(r['scene_id'],r['image_id']) for r in fine}
team_keys = {(r['sequence_id'],r['original_id'].split('_')[-1]) for r in teammate}
fine_seq = {k[0] for k in fine_keys}
team_seq = {k[0] for k in team_keys}
pair_seq = {r['scene_id'] for r in pairs}
excluded = json.loads((ROOT / 'data/grounding_aux_round1_20261025/inventory/exclusions.json').read_text(encoding='utf-8'))
reserved = set(excluded['reserved_sequences'])
risk = set(excluded['known_risk_sequences'])
pool400 = {f'{i:03d}' for i in range(1, 401)}
fine_by_frame = {(r['scene_id'], r['image_id']): r for r in fine}
same_frame_pairs = [dict(sequence=r['sequence_id'], frame=r['original_id'].split('_')[-1],
                         fine_query=fine_by_frame[(r['sequence_id'],r['original_id'].split('_')[-1])]['query'],
                         teammate_query=r['query'])
                    for r in teammate if (r['sequence_id'], r['original_id'].split('_')[-1]) in fine_keys]
comparison = {
    'method': {'word_regex': r"[A-Za-z]+(?:'[A-Za-z]+)?", 'patterns': patterns,
               'limitation': 'Overlapping lexical proxies only, not hand-reviewed capability, correctness or modality-necessity labels. Teammate subset misses 73 historical records.'},
    'fine98': stats(fine), 'teammate919': stats(teammate), 'auxiliary148': stats(pairs),
    'semifinal5691': stats(semifinal),
    'semifinal_image_groups': len({r['visible'] for r in semifinal}),
    'overlap': {'fine_sequences': len(fine_seq), 'teammate_sequences': len(team_seq),
                'shared_sequences': len(fine_seq & team_seq),
                'shared_frames': len(fine_keys & team_keys),
                'auxiliary_sequences': len(pair_seq), 'auxiliary_subset_of_fine': pair_seq <= fine_seq,
                'combined_fine_teammate_sequences': len(fine_seq | team_seq),
                'team_reserved_overlap': sorted(team_seq & reserved),
                'team_known_risk_overlap': sorted(team_seq & risk),
                'not_in_fine_team_reserved_risk': sorted(pool400-fine_seq-team_seq-reserved-risk),
                'train_eligible_upper_bound_before_new_test_scene_audit': len(pool400-reserved-risk)},
    'same_frame_queries': same_frame_pairs,
    'semifinal_examples': {
        key: [{'id': r['id'], 'query': r['query']} for r in semifinal if re.search(pattern, r['query'], re.I)][:8]
        for key, pattern in patterns.items()
    },
}
(OUT / 'style_comparison.json').write_text(json.dumps(comparison, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps({k: v for k,v in comparison.items() if k not in ('method', 'semifinal_examples')}, ensure_ascii=False, indent=2))
print(json.dumps(comparison['semifinal_examples'], ensure_ascii=False, indent=2))
