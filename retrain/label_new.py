"""Псевдо-разметка текущих скриншотов базовой моделью (этап 1 ретрейна).

Сегментация — segment_v2.py (две раскладки окна). Дедуп по shape key,
инференс базовой моделью (вендоренный код ядра — тот же, что в проде).

Выход:
- retrain/preds_new.json: shape key -> {text, conf, col, row, layout, count};
- retrain/review_NN.png: листы для глазной проверки (запуск review_montage.py).

Запуск из retrain/:  python label_new.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import paths  # noqa: E402
import segment_v2  # noqa: E402

sys.path.insert(0, str(ROOT.parent))
from l2core.adapters.ocr_vendor import (  # noqa: E402
    ctc_greedy_decode,
    load_model,
    preprocess,
)


def cv2_cvt(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    import cv2

    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def main() -> None:
    cells = segment_v2.segment_all(paths.SCREENSHOTS_DIR)
    unique: dict[str, object] = {}
    counts: dict[str, int] = {}
    for c in cells:
        unique.setdefault(c.key, c)
        counts[c.key] = counts.get(c.key, 0) + 1
    print(f"cells: {len(cells)}, unique shapes: {len(unique)}")

    model, alphabet, _ = load_model(paths.BASE_MODEL)
    model = model.to(torch.device("cpu")).eval()

    preds: dict[str, dict] = {}
    with torch.no_grad():
        for i, (key, c) in enumerate(unique.items()):
            if c.image.size == 0 or 0 in c.image.shape:
                continue
            x = preprocess(cv2_cvt(c.image))[None]
            logits = model(torch.from_numpy(x))
            text = ctc_greedy_decode(logits[0], alphabet)
            probs = logits[0].softmax(dim=-1).max(dim=-1).values
            conf = round(float(probs.mean()), 4)
            preds[key] = {"text": text, "conf": conf, "col": c.meta["col"],
                          "row": c.meta["row"], "layout": c.meta.get("layout"),
                          "count": counts[key], "rect": c.meta["rect"]}
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(unique)}")

    (ROOT / "preds_new.json").write_text(
        json.dumps(preds, ensure_ascii=False, indent=1), encoding="utf-8")
    confs = sorted(p["conf"] for p in preds.values())
    print(f"conf: min={confs[0]} p25={confs[len(confs)//4]} "
          f"med={confs[len(confs)//2]} max={confs[-1]}")


if __name__ == "__main__":
    main()
