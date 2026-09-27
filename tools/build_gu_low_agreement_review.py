"""Build CPU-only visual sheets for previously unreviewed C/G/U box disagreements."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw

from analyze_gu_outcome_20260926 import ROOT, SOURCES, iou, load


OUT = ROOT / "results/gu_pilot_20260926/low_agreement_review12"
SRC = ROOT / "results/failure_analysis_20260926/source_images"
COLORS = {"GT": "#24d64d", "C": "#3680ff", "G200": "#ff9d30", "U200": "#ff4da0"}


def crop_box(rows, width, height):
    boxes = [row for row in rows]
    x0 = min(b[0] for b in boxes) * width
    y0 = min(b[1] for b in boxes) * height
    x1 = max(b[2] for b in boxes) * width
    y1 = max(b[3] for b in boxes) * height
    side = max(x1 - x0, y1 - y0, 260)
    x0 = max(0, min(width-side, (x0+x1-side)/2))
    y0 = max(0, min(height-side, (y0+y1-side)/2))
    return tuple(map(round, (x0,y0,min(width,x0+side),min(height,y0+side))))


def panel(image, rect, label, boxes=None):
    cropped = image.convert("RGB").crop(rect).resize((600,600), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(cropped)
    if boxes:
        left, top, right, bottom = rect
        sx, sy = 600/(right-left), 600/(bottom-top)
        for name, box in boxes.items():
            xy = [(box[0]*image.width-left)*sx, (box[1]*image.height-top)*sy,
                  (box[2]*image.width-left)*sx, (box[3]*image.height-top)*sy]
            draw.rectangle(xy, outline=COLORS[name], width=4)
    draw.rectangle((0,0,600,27), fill="#101820")
    draw.text((8,6), label, fill="white")
    return cropped


def main():
    runs, ids = load()
    prior = {json.loads(s)["id"] for s in (ROOT / "results/gu_pilot_20260926/atlas_final200/failure_review.jsonl").open(encoding="utf-8")}
    low = {id_ for id_ in ids if min(iou(runs["C"][id_]["prediction"], runs[b][id_]["prediction"]) for b in ("G200","U200")) < .5}
    pending = sorted(low - prior)
    assert len(low) == 20 and len(pending) == 12
    with zipfile.ZipFile("F:/Downloads/qwen_generation_train_val_manifests.zip") as z:
        original = json.loads(z.read("qwen_generation_val.json"))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "visuals").mkdir(exist_ok=True)
    rows = []
    for id_ in pending:
        rr = {name:runs[name][id_] for name in ("C","G200","U200")}
        gt = rr["C"]["target"]
        prompt = rr["C"]["prompt"]
        query = prompt.split("Locate the object described by this query: ",1)[1].split("\n",1)[0]
        assert original[id_]["bbox"] == gt and original[id_]["query"] == query
        filename = Path(rr["C"]["image"][0]).name
        paths = {m:SRC/m/filename for m in ("visible","infrared","depth_rgb","depth")}
        available = {m:path.is_file() for m,path in paths.items()}
        result = {
            "id":id_, "group":Path(filename).stem, "query":query, "gt":gt,
            "predictions":{name:{"bbox":r["prediction"],"iou":r["iou"]} for name,r in rr.items()},
            "box_iou_c_g":iou(rr["C"]["prediction"],rr["G200"]["prediction"]),
            "box_iou_c_u":iou(rr["C"]["prediction"],rr["U200"]["prediction"]),
            "local_modalities":available,
            "visual_sheet":str(OUT/"visuals"/f"{id_}.jpg") if all(available.values()) else None,
        }
        rows.append(result)
        if not all(available.values()):
            continue
        images = {m:Image.open(p) for m,p in paths.items()}
        w,h = images["visible"].size
        assert all(images[m].size == (w,h) for m in images)
        boxes = {"GT":gt,**{name:r["prediction"] for name,r in rr.items()}}
        rect = crop_box(boxes.values(),w,h)
        collage = Image.new("RGB",(1200,1200),"#1b2430")
        for i,(m,label) in enumerate((("visible","RGB boxes: GT green C blue G orange U pink"),
                                     ("visible","RGB original"),("infrared","IR original"),
                                     ("depth_rgb","model depth input"))):
            pic = panel(images[m],rect,label,boxes if i==0 else None)
            collage.paste(pic,((i%2)*600,(i//2)*600))
        collage.save(OUT/"visuals"/f"{id_}.jpg",quality=94)
    with (OUT/"case_inputs.jsonl").open("w",encoding="utf-8",newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row,ensure_ascii=False)+"\n")
    print(f"pending={len(pending)} image_complete={sum(bool(r['visual_sheet']) for r in rows)} missing={len(rows)-sum(bool(r['visual_sheet']) for r in rows)}")


if __name__ == "__main__":
    main()
