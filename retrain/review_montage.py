"""Листы глазной проверки псевдо-разметки: по строке на кроп.

Индекс в левом углу — позиция в сортировке по conf; fixups ссылаются на
полный ключ: retrain/review_index.json {index: key}. Тёмные кропы (цены)
для показа эквализуются (на разметку не влияет).

Запуск из retrain/:  python review_montage.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import paths  # noqa: E402
import segment_v2  # noqa: E402

PER_PAGE = 12
FONT = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 22)
FONT_SMALL = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 15)


def norm(crop_l):
    """Эквализация для показа тёмных кропов."""
    import numpy as np

    a = crop_l
    if a.mean() < 90:
        lo, hi = np.percentile(a, 1), np.percentile(a, 99)
        if hi - lo > 15:
            a = ((a.astype(np.float32) - lo) * 255.0 / (hi - lo)).clip(0, 255).astype("uint8")
    return a


def main() -> None:
    preds = json.loads((ROOT / "preds_new.json").read_text(encoding="utf-8"))
    cells = segment_v2.segment_all(paths.SCREENSHOTS_DIR)
    unique: dict[str, object] = {}
    for c in cells:
        unique.setdefault(c.key, c)

    items = sorted(preds.items(), key=lambda kv: kv[1]["conf"])
    (ROOT / "review_index.json").write_text(
        json.dumps([k for k, _ in items], ensure_ascii=False), encoding="utf-8")

    pages = [items[i:i + PER_PAGE] for i in range(0, len(items), PER_PAGE)]
    for pi, page in enumerate(pages):
        row_h = 78
        sheet = Image.new("RGB", (2200, len(page) * row_h + 10), "white")
        draw = ImageDraw.Draw(sheet)
        for i, (key, p) in enumerate(page):
            gi = pi * PER_PAGE + i
            y0 = i * row_h + 5
            if unique[key].image.size == 0 or 0 in unique[key].image.shape:
                continue
            import numpy as np

            crop = Image.fromarray(norm(np.asarray(unique[key].image)))
            if crop.mode != "L":
                crop = crop.convert("L")
            h = row_h - 12
            w = int(crop.width * h / crop.height)
            if w > 1300:
                w = 1300
                h = int(crop.height * w / crop.width)
            crop = crop.resize((w, h), Image.NEAREST)
            sheet.paste(crop.convert("RGB"), (90, y0 + 2))
            color = "black" if p["conf"] >= 0.9 else "red"
            label = f"{p['text']!r}  ({p['conf']})  {p['layout']}/{p['col']} x{p['count']}"
            draw.text((1420, y0 + 26), label, font=FONT, fill=color)
            draw.text((10, y0 + 26), f"#{gi}", font=FONT, fill="blue")
            draw.text((90, y0 + row_h - 16), key[:10], font=FONT_SMALL, fill="gray")
        out = ROOT / f"review_{pi:02d}.png"
        sheet.save(out)
        print(out.name, len(page))


if __name__ == "__main__":
    main()
