#!/usr/bin/env python3
"""Генерация эталонов guard-тестов вендоренной сегментации таблицы (задача 4.3).

Запускается ВРУЧНУЮ, вне CI, один раз (и после каждого обновления вендоринга
из нового коммита источника):

    python tools/gen_table_segmentation_fixtures.py

Условия: окружение l2core (та же версия numpy, что у guard-тестов),
исходный репозиторий L2-CRNN-Dataset-Builder доступен read-only (путь —
константа SOURCE_DIR ниже или переменная окружения SEGMENTATION_VENDOR_SOURCE).

Детерминизм: входы синтетические (фиксированные константы, без rng-зависимости
от платформы), CPU, без меток времени. Входы сохраняются в эталонах —
guard-тесты гоняют ВЕНДОРЕННЫЙ код на тех же байтах входов и сравнивают
exact (структуры, хэши, байты кропов). Выход —
tests/fixtures/table_segmentation/: segmentation_inputs.npz (win_gray, color
image, trim-кропы), rows_and_columns.json, extract_cells.json (+ те же кропы
в npz), trim_to_text в npz. Эталоны коммитятся в git; для запуска guard-тестов
исходный репозиторий НЕ нужен.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

SOURCE_DIR = Path(os.environ.get("SEGMENTATION_VENDOR_SOURCE", r"E:\L2-CRNN-Dataset-Builder"))
FIXTURES = Path(__file__).parent.parent / "tests" / "fixtures" / "table_segmentation"
COMMIT = "606a65ebb474a1b8ea13ad2483aaa915b89a168d"  # фиксируем источник эталонов

sys.path.insert(0, str(SOURCE_DIR))
import build_dataset as bd  # noqa: E402  (оригинальная сегментация)


# --- Синтетические входы (детерминированные, без rng) ---

def synth_win_gray() -> np.ndarray:
    """Grayscale «окно» под константы оригинала: h=680, w=1000.

    Полоса заголовка 242..260, строки 261+56k высотой 55; строки 0..4
    непустые (метки в зоне имени), строки 5..6 пустые; нижняя кромка
    TABLE_BOTTOM=660 ограничивает сканирование (h=680 не биндинг).
    """
    h, w = 680, 1000
    img = np.full((h, w), 60, dtype=np.uint8)
    img[242:260, 14:986] = 100  # полоса заголовка
    for k in range(5):
        y0 = bd.ROW_Y0 + k * bd.ROW_PITCH
        zone = img[y0:y0 + bd.ROW_H, bd.NAME_TEXT_X[0]:bd.NAME_TEXT_X[1]]
        for i in range(8):  # «текст»: 8 светлых меток
            zone[20:34, 10 + i * 40:22 + i * 40] = 220
    return img


def synth_color() -> np.ndarray:
    """BGR «скриншот» для extract_cells: фон + маркер в каждой ячейке.

    Маркеры разного цвета по (row, col) — кропы различимы, trim детерминирован.
    Цвета укладываются в валюту uint8 без переполнений.
    """
    h, w = 680, 1000
    img = np.empty((h, w, 3), dtype=np.uint8)
    img[:, :] = (40, 60, 80)  # фоновый цвет окна
    structure = bd.rows_and_columns(synth_win_gray(), origin=(7, 9))

    def marker(col_idx: int, row_idx: int, rect) -> None:
        x0, y0, x1, y1 = rect
        cx0, cy0 = x0 + 6, y0 + 6
        cx1, cy1 = min(x1 - 6, cx0 + 20), min(y1 - 6, cy0 + 10)
        color = ((100 + 37 * col_idx + row_idx) % 256,
                 (50 + 11 * row_idx) % 256,
                 (200 - 13 * col_idx) % 256)
        img[cy0:cy1, cx0:cx1] = color

    for col_idx, (col, rect) in enumerate(structure["header"].items()):
        marker(col_idx, -1, rect)
    for row_idx, (y0, y1) in enumerate(structure["rows"]):
        for col_idx, (col, (x0, x1)) in enumerate(structure["cols"].items()):
            marker(col_idx, row_idx, (x0, y0, x1, y1))
    return img


def synth_trim_inputs() -> dict[str, np.ndarray]:
    """Кропы для trim_to_text: текст белый, пустой, цветной."""
    base = np.full((40, 120, 3), 30, dtype=np.uint8)
    text_white = base.copy()
    text_white[15:25, 30:90] = 210  # светлый текст на тёмном фоне
    empty = base.copy()  # равномерный — trim возвращает вход как есть
    colored = np.full((40, 120, 3), 50, dtype=np.uint8)
    colored[12:26, 20:100] = (50, 50, 200)  # красный текст (канал B отклоняется)
    return {"plain": text_white, "empty": empty, "colored": colored}


def main() -> None:
    assert (SOURCE_DIR / ".git").exists(), f"нет репозитория: {SOURCE_DIR}"
    head = os.popen(f'git -C "{SOURCE_DIR}" rev-parse HEAD').read().strip()
    assert head == COMMIT, \
        f"источник сдвинулся: {head} (ожидался {COMMIT}) — обнови COMMIT и вендоринг"

    FIXTURES.mkdir(parents=True, exist_ok=True)

    gray = synth_win_gray()
    color = synth_color()
    trims = synth_trim_inputs()

    # --- rows_and_columns: вход и эталонный вывод ---
    structure = bd.rows_and_columns(gray, origin=(7, 9))
    rows_expected = {
        "header": {name: list(rect) for name, rect in structure["header"].items()},
        "rows": [list(r) for r in structure["rows"]],
        "cols": {name: list(rect) for name, rect in structure["cols"].items()},
        "origin": list(structure["origin"]),
    }
    (FIXTURES / "rows_and_columns.json").write_text(
        json.dumps(rows_expected, ensure_ascii=False, indent=1), encoding="utf-8")

    # --- extract_cells: эталонные ячейки (мета + байты кропов) ---
    cells = bd.extract_cells(color, structure)
    cells_meta = []
    arrays = {"win_gray": gray, "color": color}
    arrays.update({f"trim_in_{k}": v for k, v in trims.items()})
    for i, cell in enumerate(cells):
        name = f"cell_{i:03d}"
        arrays[name] = cell.image
        cells_meta.append({
            "key": cell.key,
            "col": cell.meta["col"],
            "row": cell.meta["row"],
            "rect": list(cell.meta["rect"]),
            "pixel_key": cell.meta["pixel_key"],
            "image": name,
            "shape": list(cell.image.shape),
        })
    (FIXTURES / "extract_cells.json").write_text(
        json.dumps(cells_meta, ensure_ascii=False, indent=1), encoding="utf-8")

    # --- trim_to_text: эталонные выходы ---
    trim_shapes = {}
    for k, crop in trims.items():
        out = bd.trim_to_text(crop)
        arrays[f"trim_out_{k}"] = out
        trim_shapes[k] = list(out.shape)

    np.savez(FIXTURES / "segmentation_cases.npz", **arrays)
    print(f"эталоны записаны в {FIXTURES} (источник {COMMIT}, "
          f"{len(cells)} ячеек, строк: {len(structure['rows'])})")


if __name__ == "__main__":
    main()
