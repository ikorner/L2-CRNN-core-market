"""
Вендоринг: сегментация таблицы «Обзор лавок».

Источник: L2-CRNN-Dataset-Builder/build_dataset.py, коммит
606a65ebb474a1b8ea13ad2483aaa915b89a168d.
Копия без редизайна; правки вендоренного кода запрещены (менять в источнике,
затем перевендоривать, обновив эталоны tools/gen_table_segmentation_fixtures.py).

Скопирован блок сегментации целиком: Cell, константы геометрии (дефолт —
калибровка датасета 1136x833, ruling R5: для рантайма устарела),
_column_ranges, _row_is_empty, rows_and_columns, _text_mask, _trim,
trim_to_text, _cell_key, _pixel_key, extract_cells.

НЕ скопировано (вне задачи 4.3): detect_window (в ядре — свой детектор
якоря, l2core/detector/anchor.py), разметка/сборка датасета, CLI.

Адаптации (минимально необходимые, импортный уровень): импорты собраны
заново — только нужные скопированному блоку (hashlib/namedtuple/numpy);
argparse/csv/json/sys/Pathlib/cv2 относятся к невендоренным частям.
Тело функций и значения констант — байт-идентичны источнику; guard-тесты
tests/test_table_segmentation_vendor.py сравнивают их с эталонами,
сгенерированными оригиналом. Константы для рантайма подменяет wrapper
(segmentation.py) из table_geometry UI-модели — ядро не использует
дефолты этого модуля.
"""
from __future__ import annotations

import hashlib
from collections import namedtuple

import numpy as np

Cell = namedtuple("Cell", ["key", "image", "meta"])

# --- Relative geometry inside the window frame (origin = first interior pixel) ---
# Reference: 1136x833 screenshots, frame at x56..1059 y70..740.
CONTENT_X0 = 14          # left edge of table content (abs 71)
COL_SEPS = (72, 536, 764, 912, 966)  # abs 129, 593, 821, 969, 1023
CONTENT_X1 = 986         # right edge before scrollbar (abs 1043)
HEADER_Y0, HEADER_Y1 = 242, 260      # abs 313..331
ROW_Y0 = 261             # first data row top (abs 332)
ROW_PITCH = 56
ROW_H = 55
TABLE_BOTTOM = 660     # abs 731: bottom border of the table area
MAX_ROWS = 8

COLUMNS = ("type", "name", "price", "seller", "deal")

# zones for text extraction inside a row (relative to window interior)
NAME_TEXT_X = (119, 534)   # exclude item icon (abs ~139..172)
PRICE_TEXT_X = (556, 736)  # exclude seal icon (abs ~594-612) and gold scroll icon (abs ~799)

def _column_ranges():
    bounds = [CONTENT_X0] + list(COL_SEPS) + [CONTENT_X1]
    return {name: (bounds[i], bounds[i + 1]) for i, name in enumerate(COLUMNS)}


def _row_is_empty(win_gray, y0, y1):
    zone = win_gray[y0:y1, NAME_TEXT_X[0]:NAME_TEXT_X[1]]
    bg = np.median(zone)
    return int((zone > bg + 25).sum()) < 30


def rows_and_columns(win_gray: np.ndarray, origin=(0, 0)) -> dict:
    """Segment the table: header + non-empty rows.

    Coordinates in header/rows/cols are relative to win_gray; origin is the
    (x, y) of win_gray's top-left pixel in the full screenshot.
    """
    cols = _column_ranges()
    header = {name: (x0, HEADER_Y0, x1, HEADER_Y1) for name, (x0, x1) in cols.items()}
    h = win_gray.shape[0]
    rows = []
    for k in range(MAX_ROWS):
        y0 = ROW_Y0 + k * ROW_PITCH
        y1 = y0 + ROW_H
        if y1 > min(TABLE_BOTTOM, h):
            break
        if not _row_is_empty(win_gray, y0, y1):
            rows.append((y0, y1))
    return {"header": header, "rows": rows, "cols": cols, "origin": origin}


def _text_mask(crop: np.ndarray) -> np.ndarray:
    """Binary text mask: 255 where a pixel deviates from the background color."""
    bg_color = np.median(crop.reshape(-1, 3), axis=0)
    return (np.abs(crop.astype(np.int16) - bg_color).max(axis=2) > 25).astype(np.uint8) * 255


def _trim(crop: np.ndarray, pad: int = 4):
    """Crop to the text bounding box plus padding. Returns (crop, dx, dy).

    Mask is color-based (max per-channel deviation from the background color),
    so colored text (blue/red/green) is detected as reliably as white.
    """
    mask = _text_mask(crop)
    if not mask.any():
        return crop, 0, 0
    ys, xs = np.where(mask)
    y0, y1 = max(0, ys.min() - pad), min(crop.shape[0], ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(crop.shape[1], xs.max() + 1 + pad)
    return crop[y0:y1, x0:x1], x0, y0


def trim_to_text(crop: np.ndarray, pad: int = 4) -> np.ndarray:
    """Crop to the text bounding box plus padding. Falls back to the input."""
    return _trim(crop, pad)[0]


def _cell_key(image: np.ndarray) -> str:
    """Shape key: hash of the binary text mask.

    Same text on alternating row backgrounds or in different colors hashes
    identically, so each unique text is labeled once. The original pixels are
    kept in Cell.image and in meta['pixel_key'] for the dataset output.
    """
    return hashlib.sha1(_text_mask(image).tobytes()).hexdigest()


def _pixel_key(image: np.ndarray) -> str:
    return hashlib.sha1(image.tobytes()).hexdigest()


def extract_cells(img: np.ndarray, structure: dict) -> list:
    """Cut all labelable cells: 5 header labels + rows x 5 columns.

    Returns a list of Cell(key, image, meta); key deduplicates identical crops.
    meta: col, row (-1 = header), rect (absolute x0,y0,x1,y1).
    """
    ox, oy = structure["origin"]
    cells = []

    def cut(col, row, rect):
        x0, y0, x1, y1 = rect
        dx0 = dy0 = 0
        crop = img[oy + y0 : oy + y1, ox + x0 : ox + x1]
        if col == "name":
            tx0, tx1 = NAME_TEXT_X
            dx0 = max(0, tx0 - x0)
            crop = crop[:, dx0 : tx1 - x0]
        elif col == "price":
            tx0, tx1 = PRICE_TEXT_X
            dx0 = max(0, tx0 - x0)
            crop = crop[:, dx0 : tx1 - x0]
        crop, tx, ty = _trim(crop)
        abs_rect = (int(ox + x0 + dx0 + tx), int(oy + y0 + ty),
                    int(ox + x0 + dx0 + tx + crop.shape[1]), int(oy + y0 + ty + crop.shape[0]))
        cells.append(Cell(_cell_key(crop), crop,
                          {"col": col, "row": row, "rect": abs_rect, "pixel_key": _pixel_key(crop)}))

    for col, rect in structure["header"].items():
        cut(col, -1, rect)
    for row, (y0, y1) in enumerate(structure["rows"]):
        for col, (x0, x1) in structure["cols"].items():
            cut(col, row, (x0, y0, x1, y1))
    return cells
