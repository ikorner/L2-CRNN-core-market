"""Сбор смешанного датасета: старый (ручная разметка) + новые скрины
(псевдо-разметка с глазной верификацией) + golden-кропы core-market.

Выход: mixed_dataset/images/*.png, labels.csv (image_path,text),
cells.csv (pixel_key,col,source). Статистика и сверка алфавита — в stdout.

Запуск из retrain/:  python build_mixed.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).parent
CORE = ROOT.parent
sys.path.insert(0, str(ROOT))
import paths  # noqa: E402
import segment_v2  # noqa: E402

OUT = ROOT / "mixed_dataset"
MIN_AUTO_CONF = 0.95


def load_old(out_images: Path, rows: list, cells_rows: list) -> int:
    """Старый датасет: только строки с существующими файлами."""
    n = 0
    for r in csv.DictReader((ROOT / "dataset" / "labels.csv").open(encoding="utf-8")):
        text = r["text"].strip()
        name = Path(r["image_path"]).name
        src = ROOT / "dataset" / "images" / name
        if not text or not src.exists():
            continue
        (out_images / name).write_bytes(src.read_bytes())
        rows.append((f"images/{name}", text))
        cells_rows.append((Path(name).stem, "old", "old"))
        n += 1
    return n


def load_new(out_images: Path, rows: list, cells_rows: list) -> tuple[int, int]:
    """Новые скрины: fixups (глазная проверка) > pred при conf>=0.95; skips пропуск."""
    fixups = json.loads((ROOT / "fixups.json").read_text(encoding="utf-8"))
    skips = set(json.loads((ROOT / "skips.json").read_text(encoding="utf-8")))
    preds = json.loads((ROOT / "preds_new.json").read_text(encoding="utf-8"))
    cells = segment_v2.segment_all(paths.SCREENSHOTS_DIR)
    unique: dict[str, object] = {}
    for c in cells:
        unique.setdefault(c.key, c)
    n_fix = n_auto = 0
    for key, cell in unique.items():
        if key in skips:
            continue
        if key in fixups:
            text = fixups[key]
            n_fix += 1
        else:
            p = preds.get(key)
            if not p or p["conf"] < MIN_AUTO_CONF:
                continue
            text = p["text"]
            n_auto += 1
        name = f"{key}.png"
        # важно: cell.image — BGR (cv2), продакшен перед OCR делает
        # cv2.cvtColor(BGR2GRAY) (l2core/adapters/ocr.py::_to_gray);
        # сохранять нужно именно такое grayscale, иначе цветной текст
        # (синие/фиолетовые цены) обучается с неверной яркостью
        import cv2  # noqa

        arr = np.asarray(cell.image)
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        Image.fromarray(arr).save(out_images / name)
        rows.append((f"images/{name}", text))
        cells_rows.append((key, getattr(cell, "col", "name"), "new"))
    return n_fix, n_auto


def load_golden(out_images: Path, rows: list, cells_rows: list) -> int:
    """Golden-кропы core-market: тексты из index.yaml (verified)."""
    import yaml  # noqa

    golden = CORE / "assets" / "ocr_golden"
    meta = yaml.safe_load((golden / "index.yaml").read_text(encoding="utf-8"))
    n = 0
    for card in meta["cards"]:
        src = golden / card["file"]
        if not src.exists() or not str(card.get("text", "")).strip():
            continue
        # оверсэмплинг: эталонные кропы критичны для приёмки (spec 4.6),
        # по одной копии за прогон модель их не закрепляет; каждая копия —
        # отдельный файл (load_samples дедуплицирует по имени файла)
        for copy in range(4):
            suffix = f"_c{copy}" if copy else ""
            name = f"golden_{card['id']}{suffix}.png"
            (out_images / name).write_bytes(src.read_bytes())
            rows.append((f"images/{name}", str(card["text"]).strip()))
            cells_rows.append((f"golden_{card['id']}{suffix}",
                               str(card.get("grammar", "text")), "golden"))
            n += 1
    return n


def main() -> None:
    out_images = OUT / "images"
    out_images.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, str]] = []
    cells_rows: list[tuple[str, str, str]] = []

    n_old = load_old(out_images, rows, cells_rows)
    n_fix, n_auto = load_new(out_images, rows, cells_rows)
    n_golden = load_golden(out_images, rows, cells_rows)

    with (OUT / "labels.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "text"])
        w.writerows(rows)
    with (OUT / "cells.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pixel_key", "col", "source"])
        w.writerows(cells_rows)

    # статистика
    from collections import Counter
    by_src = Counter(c[2] for c in cells_rows)
    print(f"old={n_old} new_fix={n_fix} new_auto={n_auto} golden={n_golden}")
    print(f"всего samples={len(rows)} по источникам: {dict(by_src)}")

    # сверка алфавита со старым чекпоинтом
    import torch
    ck = torch.load(ROOT / "base_model.pt", map_location="cpu", weights_only=False)
    old_alpha = set(ck["alphabet"])
    new_chars = set()
    for _, text in rows:
        new_chars.update(text)
    new_chars.discard(" ")
    print(f"алфавит старого чекпоинта: {len(old_alpha)} символов")
    print(f"алфавит смешанных данных: {len(new_chars)} символов")
    print(f"новых символов нет в старом: {''.join(sorted(new_chars - old_alpha)) or '—'}")
    print(f"символы старого, не встречающиеся в данных: {''.join(sorted(old_alpha - new_chars)) or '—'}")


if __name__ == "__main__":
    main()
