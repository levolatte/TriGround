from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw


root = Path(r"D:\AIC\city_detection_prepared\train")
records = json.loads((root / "stage4_train_v2.json").read_text(encoding="utf-8"))
ranked = sorted(records.items(), key=lambda item: (item[1]["bbox"][2] - item[1]["bbox"][0]) * (item[1]["bbox"][3] - item[1]["bbox"][1]))
rng = random.Random(2026)
chosen = ranked[:4] + rng.sample(ranked[len(ranked)//3:2*len(ranked)//3], 4)
tiles = []
for sample_id, record in chosen:
    modalities = []
    for name in ("visible", "infrared", "depth"):
        with Image.open(root / record[name]) as source:
            image = source.convert("RGB").copy()
        x1, y1, x2, y2 = record["bbox"]
        draw = ImageDraw.Draw(image)
        draw.rectangle((x1 * image.width, y1 * image.height, x2 * image.width, y2 * image.height), outline="red", width=max(3, image.width // 300))
        image.thumbnail((480, 270))
        modalities.append(image)
    tile = Image.new("RGB", (1440, 320), "white")
    for index, image in enumerate(modalities):
        tile.paste(image, (index * 480, 50))
    ImageDraw.Draw(tile).text((10, 10), f"{sample_id}: {record['query']}", fill="black")
    tiles.append(tile)
canvas = Image.new("RGB", (1440, 320 * len(tiles)), "white")
for index, tile in enumerate(tiles):
    canvas.paste(tile, (0, index * 320))
output = Path("audit_city_samples.jpg").resolve()
canvas.save(output, quality=92)
print(output)
