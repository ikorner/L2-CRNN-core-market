"""Дозум-выгрузка отдельных кропов для точного чтения.

Запуск из retrain/:  python zoom_dump.py 3 4 7 ...
Индексы — позиции в review_index.json. Кропы усиливаются (percentile 0.5..99.5,
усиление) и увеличиваются x2, сохраняются в zoom_<i>.png рядом.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import paths  # noqa: E402
import segment_v2  # noqa: E402


def norm(a: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(a, 0.5), np.percentile(a, 99.5)
    if hi - lo < 10:
        return a
    out = (a.astype(np.float32) - lo) * (255.0 / (hi - lo))
    out = np.clip(out, 0, 255).astype("uint8")
    # дополнительное усиление контраста: растяжение к полному диапазону
    lo2, hi2 = np.percentile(out, 0.5), np.percentile(out, 99.5)
    if hi2 - lo2 > 10:
        out = np.clip((out.astype(np.float32) - lo2) * (255.0 / (hi2 - lo2)), 0, 255).astype("uint8")
    return out


def main() -> None:
    indices = [int(x) for x in sys.argv[1:]]
    idx = json.loads((ROOT / "review_index.json").read_text(encoding="utf-8"))
    cells = segment_v2.segment_all(paths.SCREENSHOTS_DIR)
    unique: dict[str, object] = {}
    for c in cells:
        unique.setdefault(c.key, c)
    for gi in indices:
        key = idx[gi]
        img = np.asarray(unique[key].image)
        if img.size == 0:
            print(f"#{gi}: empty")
            continue
        crop = Image.fromarray(norm(img)).resize(
            (img.shape[1] * 2, img.shape[0] * 2), Image.LANCZOS)
        out = ROOT / f"zoom_{gi:03d}.png"
        crop.save(out)
        print(f"#{gi} -> {out.name} {crop.size}")


if __name__ == "__main__":
    main()
