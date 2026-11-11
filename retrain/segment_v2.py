"""Сегментация текущих скриншотов: две раскладки окна «Обзор лавок».

Раскладка LIST — классический список (колонки Тип/Предмет/Цена/Персонаж/
Сделка): геометрия констант build_dataset.py подходит (окно того же размера).

Раскладка TOP5 — «Топ-5 в личной торговой лавке» (две таблицы Хит продаж):
внутренняя геометрия другая, константы измерены со scr_153551:
  заголовки таблиц y 330..355; строки: 5 полос y 363..395 + k*60;
  левая таблица: имя x 105..425, кол-во x 430..495;
  правая таблица: имя x 610..855, цена x 860..995.

Детекция раскладки: баннер «Пять самых популярных товаров» (y 205..240,
текст есть только в TOP5).

API: segment_all() -> list[bd.Cell] (meta: col, row, rect, layout).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import build_dataset as bd

# --- TOP5 константы (окно-относительные) ---
T5_HEADER_Y = (330, 355)
T5_ROW_Y0, T5_ROW_PITCH, T5_ROW_H, T5_ROWS = 363, 60, 32, 5
T5_ZONES = {
    "l_name": (105, 425),
    "l_qty": (430, 495),
    "r_name": (610, 855),
    "r_price": (860, 995),
}
T5_HEADER_ZONES = {
    "l_name": (105, 425),
    "l_qty": (430, 495),
    "r_name": (610, 855),
    "r_price": (860, 995),
}

BANNER_ZONE = (100, 900, 243, 261)  # x0, x1, y0, y1 (окно-относительные)


def is_top5(win_gray: np.ndarray) -> bool:
    """TOP5-раскладка: под баннером пусто, а декор панелей даёт много ярких
    пикселей; в LIST там текстовая шапка таблицы (~3040 пикселей).

    Порог 3500 отделил единственный top5-скрин (scr_153551, 4104) от всех
    list-скринов (3040..3045) на текущей сырьевой базе."""
    x0, x1, y0, y1 = BANNER_ZONE
    zone = win_gray[y0:y1, x0:x1]
    bg = np.median(zone)
    return int((zone > bg + 25).sum()) > 3500


def segment_top5(win_bgr: np.ndarray, origin: tuple[int, int]) -> list:
    """Клетки TOP5-раскладки. win_bgr — кроп окна (цветной)."""
    cells = []
    ox, oy = origin

    def cut(col, row, rect):
        x0, y0, x1, y1 = rect
        crop = win_bgr[y0:y1, x0:x1]
        if crop.size == 0 or 0 in crop.shape:
            return
        if col in ("l_name", "r_name"):
            crop = _split_left_cluster(crop)
        elif col in ("r_price",):
            crop = _split_right_cluster(crop)
        abs_rect = (int(ox + x0), int(oy + y0),
                    int(ox + x0 + crop.shape[1]), int(oy + y0 + crop.shape[0]))
        cells.append(bd.Cell(bd._cell_key(crop), crop,
                             {"col": col, "row": row, "rect": abs_rect,
                              "layout": "top5"}))

    for col, (x0, x1) in T5_HEADER_ZONES.items():
        cut(col, -1, (x0, T5_HEADER_Y[0], x1, T5_HEADER_Y[1]))
    for k in range(T5_ROWS):
        y0 = T5_ROW_Y0 + k * T5_ROW_PITCH
        y1 = y0 + T5_ROW_H
        # пустая строка: нет текста в зоне имени левой таблицы
        zone = win_bgr[y0:y1, 105:425]
        bg = np.median(zone.reshape(-1, 3), axis=0)
        if (np.abs(zone.astype(np.int16) - bg).max(axis=2) > 25).sum() < 30:
            continue
        for col, (x0, x1) in T5_ZONES.items():
            cut(col, k, (x0, y0, x1, y1))
    return cells


# --- LIST константы (окно-относительные, измерены со scr_153611) ---
# type x50..82, icon x124..160, name x164..433, price x531..681 (+иконка
# свитка x701..770 — за пределами зоны), seller x892..917, deal x938..966
# (+печать x972..986 — за пределами зоны). Строки: шаг 56, высота полосы
# 33 (иконка), шапка y240..258. Смещения строк плавают между скринами
# (шапка встречалась на y225..243) -> поиск полос динамический.
LIST_ZONES = {
    "type": (40, 95),
    "name": (160, 493),     # до разделителя колонок (~497)
    "price": (510, 695),
    "seller": (700, 878),   # имена прижаты вправо, кончаются ~870
    "deal": (885, 925),     # «Аден» x892..917; печать-свиток x938+ — вне зоны
}
LIST_PROFILE_X = (160, 500)   # зона профиля строк (имя)
LIST_Y_RANGE = (225, 660)
BAND_MERGE_GAP = 6
ROW_MIN_H = 24              # строки с иконкой ≥ 24px; шапка ~18px
BAND_MIN_H = 8


def _text_mask(win_bgr: np.ndarray) -> np.ndarray:
    bg = np.median(win_bgr.reshape(-1, 3), axis=0)
    return (np.abs(win_bgr.astype(np.int16) - bg).max(axis=2) > 25)


def _bands(profile: np.ndarray, thr: int) -> list[tuple[int, int]]:
    """Слияние подряд идущих полос текста в полосы строк/шапки."""
    runs, cur = [], None
    for y, v in enumerate(profile > thr):
        if v and cur is None:
            cur = y
        elif not v and cur is not None:
            runs.append([cur, y])
            cur = None
    if cur is not None:
        runs.append([cur, len(profile)])
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= BAND_MERGE_GAP:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [(a, b) for a, b in merged if b - a >= BAND_MIN_H]


def _split_left_cluster(crop: np.ndarray, min_gap: int = 16) -> np.ndarray:
    """Отрезать правый текстовый кластер (протёкшую соседнюю колонку).

    Маска текста -> профиль чернил по x -> самый широкий внутренний
    полностью пустой зазор; если есть — кроп до него. Иконка грейда
    (справа, близко к тексту) не отрезается: зазор мал.
    """
    if crop.ndim == 3:
        bg = np.median(crop.reshape(-1, crop.shape[2]), axis=0)
        mask = (np.abs(crop.astype(np.int16) - bg).max(axis=2) > 25)
    else:
        bg = np.median(crop)
        mask = np.abs(crop.astype(np.int16) - bg) > 25
    ink = mask.sum(axis=0)
    best = None
    run = None
    for x, v in enumerate(ink):
        if v == 0:
            run = x if run is None else run
        else:
            if run is not None and x - run >= min_gap and run > 8 \
                    and crop.shape[1] - x > 8:
                if best is None or x - run > best[1] - best[0]:
                    best = (run, x)
            run = None
    if best:
        return crop[:, :best[0] + 1]
    return crop


def _split_right_cluster(crop: np.ndarray, min_gap: int = 14) -> np.ndarray:
    """Зеркало _split_left_cluster: оставить правый текстовый кластер.

    Для ценовой зоны: соседняя колонка (продавец) «протекает» справа в
    узких раскладках, цена прижата вправо.
    """
    if crop.ndim == 3:
        bg = np.median(crop.reshape(-1, crop.shape[2]), axis=0)
        mask = (np.abs(crop.astype(np.int16) - bg).max(axis=2) > 25)
    else:
        bg = np.median(crop)
        mask = np.abs(crop.astype(np.int16) - bg) > 25
    ink = mask.sum(axis=0)
    best = None
    run = None
    for x, v in enumerate(ink):
        if v == 0:
            run = x if run is None else run
        else:
            if run is not None and x - run >= min_gap and run > 8 \
                    and crop.shape[1] - x > 8:
                if best is None or x - run > best[1] - best[0]:
                    best = (run, x)
            run = None
    if best:
        return crop[:, best[1]:]
    return crop


def segment_list(win_bgr: np.ndarray, origin: tuple[int, int]) -> list:
    """Клетки списочной раскладки: динамический поиск шапки и строк.

    Кроп = полоса строки (текст ±3px) на всю ширину зоны колонки — как
    продовые OCR-регионы (fixed zones), НЕ подрезка по маске: маска ловит
    только яркие глифы и обрезает тусклый фиолетовый текст цен.
    """
    mask = _text_mask(win_bgr)
    x0, x1 = LIST_PROFILE_X
    y0, y1 = LIST_Y_RANGE
    profile = mask[y0:y1, x0:x1].sum(axis=1)
    bands = _bands(profile, thr=4)

    cells = []
    ox, oy = origin

    def cut(col, row, band):
        a, b = band
        cy0, cy1 = y0 + a - 3, y0 + b + 3
        for zx0, zx1 in [LIST_ZONES[col]]:
            crop = win_bgr[cy0:cy1, zx0:zx1]
            if col == "name":
                # цена «протекает» в зону имени в узких раскладках колонок
                crop = _split_left_cluster(crop)
            elif col == "price":
                # продавец «протекает» справа в ценовую зону
                crop = _split_right_cluster(crop)
            if crop.size == 0 or 0 in crop.shape:
                return
            cells.append(bd.Cell(bd._cell_key(crop), crop,
                                 {"col": col, "row": row,
                                  "rect": (int(ox + zx0), int(oy + cy0),
                                           int(ox + zx0 + crop.shape[1]),
                                           int(oy + cy0 + crop.shape[0])),
                                  "layout": "list"}))

    kept = [(a, b) for a, b in bands if b - a >= 10]
    for bi, (a, b) in enumerate(kept):
        row = -1 if bi == 0 else bi - 1  # первая полоса — шапка таблицы
        for col in LIST_ZONES:
            cut(col, row, (a, b))
    return cells


def cv2_cvt(img):
    import cv2

    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def segment_all(screenshots_dir: Path) -> list:
    """Все клетки из всех BMP скриншотов каталога."""
    import cv2

    cells = []
    for bmp in sorted(Path(screenshots_dir).glob("*.bmp")):
        img = cv2.imread(str(bmp))
        if img is None:
            continue
        gray = cv2_cvt(img)
        w = bd.detect_window(gray)
        if not w:
            print(f"пропуск {bmp.name}: окно не найдено")
            continue
        x, y, ww, wh = w
        win = img[y:y + wh, x:x + ww]
        layout = "top5" if is_top5(cv2_cvt(win)) else "list"
        found = (segment_top5(win, (x, y)) if layout == "top5"
                 else segment_list(win, (x, y)))
        rows = {c.meta["row"] for c in found if c.meta["row"] >= 0}
        print(f"{bmp.name}: {layout}, строк: {len(rows)}, ячеек: {len(found)}")
        cells.extend(found)
    return cells


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    import paths

    cells = segment_all(paths.SCREENSHOTS_DIR)
    print("всего ячеек:", len(cells))
