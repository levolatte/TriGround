from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ORDINAL_QUERY = re.compile(r"^The \d+(?:st|nd|rd|th) (.+) from the left$")
EDGE_QUERY = re.compile(r"^The (?:leftmost|rightmost) (.+)$")
PLAIN_QUERY = re.compile(r"^The (.+)$")
PROCESSED_WIDTH, PROCESSED_HEIGHT = 1176, 672


def object_name(query: str) -> str:
    for pattern in (ORDINAL_QUERY, EDGE_QUERY, PLAIN_QUERY):
        match = pattern.match(query.strip())
        if match:
            return match.group(1)
    raise ValueError(f"Cannot extract class from weak query: {query!r}")


def center(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def scale_properties(box):
    short_side = min((box[2] - box[0]) * PROCESSED_WIDTH, (box[3] - box[1]) * PROCESSED_HEIGHT)
    area = (box[2] - box[0]) * (box[3] - box[1])
    scale = "tiny" if short_side < 8 else "small" if short_side < 16 else "medium" if short_side < 32 else "large"
    return scale, short_side, area


def region_phrase(box):
    x, y = center(box)
    horizontal = "left" if x < 1 / 3 else "right" if x > 2 / 3 else "center"
    vertical = "upper" if y < 1 / 3 else "lower" if y > 2 / 3 else "middle"
    if horizontal == "center" and vertical == "middle": return "near the center of the image"
    if horizontal == "center": return f"in the {vertical} part of the image"
    if vertical == "middle": return f"on the {horizontal} side of the image"
    return f"in the {vertical} {horizontal} part of the image"


def relation_phrase(target, reference):
    tx, ty = center(target["bbox"]); rx, ry = center(reference["bbox"])
    dx, dy = tx - rx, ty - ry
    if abs(dx) >= abs(dy): return "to the right of" if dx > 0 else "to the left of"
    return "below" if dy > 0 else "above"


def dhash(path):
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    value = 0
    for bit in (gray[:, 1:] > gray[:, :-1]).ravel(): value = (value << 1) | int(bit)
    return value


class UnionFind:
    def __init__(self, values): self.parent = {value: value for value in values}
    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]; value = self.parent[value]
        return value
    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right: self.parent[right] = left


def build_components(root, scenes, hash_distance):
    stems = sorted(scenes); union = UnionFind(stems)
    groups = defaultdict(list)
    for stem in stems: groups[str(scenes[stem][0]["source_group"])].append(stem)
    for values in groups.values():
        for stem in values[1:]: union.union(values[0], stem)
    hashes = {stem: dhash(root / scenes[stem][0]["visible"]) for stem in stems}
    near_pairs = 0
    for index, left in enumerate(stems):
        for right in stems[index + 1:]:
            if (hashes[left] ^ hashes[right]).bit_count() <= hash_distance:
                near_pairs += 1; union.union(left, right)
    components = defaultdict(list)
    for stem in stems: components[union.find(stem)].append(stem)
    return list(components.values()), near_pairs


def split_components(components, scenes, val_fraction, seed):
    total_scenes = sum(map(len, components)); total_classes = Counter(r["class_name"] for v in scenes.values() for r in v)
    target_scenes = total_scenes * val_fraction; targets = {n: c * val_fraction for n, c in total_classes.items()}
    stats = [(c, Counter(r["class_name"] for s in c for r in scenes[s])) for c in components]
    rng = random.Random(seed); best = None
    for _ in range(1500):
        shuffled = stats[:]; rng.shuffle(shuffled); selected = []; nscene = 0; counts = Counter()
        for component, classes in shuffled:
            before = abs(nscene-target_scenes) + .08*sum(abs(counts[n]-t)/max(t,1) for n,t in targets.items())
            after = abs(nscene+len(component)-target_scenes) + .08*sum(abs(counts[n]+classes[n]-t)/max(t,1) for n,t in targets.items())
            if after <= before: selected.append(component); nscene += len(component); counts.update(classes)
        score = abs(nscene-target_scenes)/max(target_scenes,1) + .08*sum(abs(counts[n]-t)/max(t,1) for n,t in targets.items())
        if best is None or score < best[0]: best = (score, selected)
    val = {s for c in best[1] for s in c}; return set(scenes)-val, val


def identity_for(item, same_class):
    name = item["class_name"]
    if len(same_class) == 1: return f"The {name}"
    ordered = sorted(same_class, key=lambda r: (center(r["bbox"])[0], center(r["bbox"])[1])); rank = ordered.index(item)
    if rank == 0: return f"The leftmost {name}"
    if rank == len(ordered)-1: return f"The rightmost {name}"
    if rank == 1: return f"The second {name} from the left"
    if rank == 2: return f"The third {name} from the left"
    return None


def optimize_scene(rows, class_frequency, max_per_scene):
    by_class = defaultdict(list)
    for row in rows: by_class[row["class_name"]].append(row)
    candidates = []
    for row in rows:
        identity = identity_for(row, by_class[row["class_name"]])
        if identity is None: continue
        others = [r for r in rows if r["class_name"] != row["class_name"]]
        if len(by_class[row["class_name"]]) == 1 and others:
            ref = min(others, key=lambda r: math.dist(center(row["bbox"]), center(r["bbox"])))
            query = f"{identity} {relation_phrase(row, ref)} the nearby {ref['class_name']}"
        elif len(by_class[row["class_name"]]) == 1: query = f"{identity} {region_phrase(row['bbox'])}"
        else: query = identity
        scale, short_side, area = scale_properties(row["bbox"]); candidate = dict(row)
        candidate.update(query=query, scale=scale, short_side=short_side, area=area); candidates.append(candidate)
    priority = {"large":0,"medium":1,"small":2,"tiny":3}
    candidates.sort(key=lambda r: (class_frequency[r["class_name"]], priority[r["scale"]], r["sample_id"]))
    selected=[]; per_class=Counter()
    for row in candidates:
        if len(selected) >= max_per_scene: break
        if per_class[row["class_name"]] >= 4: continue
        selected.append(row); per_class[row["class_name"]] += 1
    return selected


def serialize(rows):
    return {r["sample_id"]: {"visible":r["visible"],"infrared":r["infrared"],"depth":r["depth"],"query":r["query"],"bbox":[round(float(v),6) for v in r["bbox"]],"class_name":r["class_name"],"scale_bin":r["scale"],"bbox_area":round(r["area"],8),"processed_short_side":round(r["short_side"],3),"weak_label":True} for r in rows}


def summarize(rows):
    return {"queries":len(rows),"scenes":len({r['stem'] for r in rows}),"classes":dict(sorted(Counter(r['class_name'] for r in rows).items())),"scales":dict(Counter(r['scale'] for r in rows)),"max_queries_per_scene":max(Counter(r['stem'] for r in rows).values(),default=0),"queries_with_comma":sum(',' in r['query'] for r in rows),"max_ordinal":3}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--data-root',type=Path,required=True); parser.add_argument('--source',type=Path); parser.add_argument('--val-fraction',type=float,default=.2); parser.add_argument('--max-per-scene',type=int,default=12); parser.add_argument('--hash-distance',type=int,default=3); parser.add_argument('--seed',type=int,default=2026); args=parser.parse_args()
    root=args.data_root.resolve(); source=(args.source or root/'queries'/'weak_grounding.json').resolve(); raw=json.loads(source.read_text(encoding='utf-8-sig'))
    scenes=defaultdict(list); excluded=set()
    for sample_id, record in raw.items():
        stem=Path(record['visible']).stem
        with Image.open(root/record['depth']) as depth: metric=depth.mode in {'I;16','I;16L','I;16B','I'}
        if not metric: excluded.add(stem); continue
        box=[float(v) for v in record['bbox']]
        if not (0<=box[0]<box[2]<=1 and 0<=box[1]<box[3]<=1): raise ValueError(f'Invalid bbox {sample_id}')
        scenes[stem].append({'sample_id':sample_id,'stem':stem,'visible':record['visible'],'infrared':record['infrared'],'depth':record['depth'],'bbox':box,'class_name':object_name(record['query']),'source_group':record.get('_group',stem)})
    frequencies=Counter(r['class_name'] for rows in scenes.values() for r in rows)
    optimized={s:optimize_scene(rows,frequencies,args.max_per_scene) for s,rows in scenes.items()}; optimized={s:r for s,r in optimized.items() if r}
    components,near_pairs=build_components(root,optimized,args.hash_distance); train_stems,val_stems=split_components(components,optimized,args.val_fraction,args.seed)
    train=[r for s in sorted(train_stems) for r in optimized[s]]; val=[r for s in sorted(val_stems) for r in optimized[s]]
    (root/'grounding_final_train.json').write_text(json.dumps(serialize(train),ensure_ascii=False,indent=2),encoding='utf-8'); (root/'grounding_final_val.json').write_text(json.dumps(serialize(val),ensure_ascii=False,indent=2),encoding='utf-8')
    tg={optimized[s][0]['source_group'] for s in train_stems}; vg={optimized[s][0]['source_group'] for s in val_stems}
    report={'source':str(source),'weak_labels':True,'policy':{'metric_depth_only':True,'max_queries_per_scene':args.max_per_scene,'max_same_class_per_scene':4,'max_ordinal':3,'small_targets':'retained and tagged by processed short side; not removed by area','split':f'source-group components plus RGB dHash <= {args.hash_distance}'},'excluded_visualized_depth_scenes':len(excluded),'content_near_pairs_joined':near_pairs,'split_components':len(components),'source_group_overlap':len(tg&vg),'scene_overlap':len(train_stems&val_stems),'train':summarize(train),'val':summarize(val),'total_queries':len(train)+len(val)}
    (root/'grounding_final_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
