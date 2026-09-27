"""Prepare paired modality interventions; keep prompts, RGB and annotations unchanged."""
import argparse, copy, json, random
from collections import defaultdict
from pathlib import Path
from PIL import Image

ARMS = ("normal", "both_shuffle", "both_black", "ir_shuffle", "depth_shuffle", "ir_black", "depth_black")

def prepare(source, data_root, output, seed=2026):
    rows = json.loads(source.read_text())
    groups = {}
    for r in rows:
        paths = r["image"]
        assert len(paths) == 3
        groups.setdefault(paths[0], paths)
        assert groups[paths[0]] == paths
    # Permute whole image groups within identical image dimensions.
    blocks = defaultdict(list)
    for key, paths in groups.items():
        sizes = tuple(Image.open(data_root/p).size for p in paths)
        blocks[sizes].append(key)
    rng = random.Random(seed)
    donor = {}
    for block in blocks.values():
        assert len(block) >= 2, "Cannot make a same-size non-self permutation"
        ordered = sorted(block)
        rng.shuffle(ordered)
        donor.update({x: ordered[(i+1)%len(ordered)] for i,x in enumerate(ordered)})
    output.mkdir(parents=True, exist_ok=True)
    blank = output/"black"; blank.mkdir(exist_ok=True)
    black_paths = {}
    for sizes in blocks:
        for width, height in sizes:
            p = blank/f"{width}x{height}.png"
            if not p.exists(): Image.new("RGB",(width,height),(0,0,0)).save(p)
            black_paths[(width,height)] = str(p.resolve())
    for arm in ARMS:
        dest = []
        for src in rows:
            row = copy.deepcopy(src)
            original = src["image"]
            paths = [str((data_root/p).resolve()) for p in original]
            key = original[0]
            if arm != "normal":
                slots = [1,2] if arm.startswith("both") else [1] if arm.startswith("ir") else [2]
                for slot in slots:
                    if arm.endswith("shuffle"):
                        paths[slot] = str((data_root/groups[donor[key]][slot]).resolve())
                    else:
                        with Image.open(paths[slot]) as im:
                            paths[slot] = black_paths[im.size]
            row["image"] = paths
            assert row["conversations"] == src["conversations"]
            assert row["id"] == src["id"] and paths[0] == str((data_root/original[0]).resolve())
            dest.append(row)
        (output/f"{arm}.json").write_text(json.dumps(dest,ensure_ascii=False,indent=2))
    meta = {"seed":seed, "samples":len(rows), "image_groups":len(groups), "arms":ARMS,
            "donor_by_rgb":donor,
            "design":"Same M2 checkpoint, RGB/prompt/decode fixed; black images retain dimensions; derangement by image group with matched sizes. Both shuffled modalities share a donor. Full cohort; no GT-based arm selection."}
    (output/"design.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in meta.items() if k!="donor_by_rgb"},ensure_ascii=False))

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True);p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    a=p.parse_args()
    prepare(a.source,a.data_root,a.output_dir)
