import json
from pathlib import Path
from PIL import Image, ImageDraw

root = Path(r"D:\AIC\city_detection_prepared\train")
report = json.loads(Path("audit_city_report.json").read_text(encoding="utf-8-sig"))
pairs = report["leakage"]["dhash_examples"][:12]
canvas = Image.new("RGB", (960, len(pairs) * 300), "white")
for index, (distance, left, right) in enumerate(pairs):
    for column, path in enumerate((left, right)):
        with Image.open(root / path) as source:
            image = source.convert("RGB")
        image.thumbnail((470, 250))
        canvas.paste(image, (column * 480, index * 300 + 40))
    ImageDraw.Draw(canvas).text((5, index * 300 + 5), f"d={distance}  {left}  |  {right}", fill="black")
output = Path("audit_leakage_pairs.jpg").resolve()
canvas.save(output, quality=92)
print(output)
