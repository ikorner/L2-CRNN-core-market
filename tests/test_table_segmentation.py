"""Тесты wrapper'а сегментации таблицы (задача 4.3).

Wrapper (l2core.adapters.table_segmentation.TableSegmentation) подменяет
module-level константы вендоренного блока значениями table_geometry модели
(единственная адаптация — инъекция; тело вендора байт-идентично, см.
tests/test_table_segmentation_vendor.py). Здесь: маппинг констант модели на
константы вендора, валидация геометрии, поведение на синтетике и
интеграционная нарезка реальных скриншотов меты (base_screenshot/scr/,
калибровка 2.4: col_seps/row_y0/row_pitch/max_rows, зоны колонок).
"""
from pathlib import Path

import cv2
import numpy as np
import pytest

from l2core.adapters.table_segmentation import TableSegmentation
from l2core.adapters.table_segmentation import build_dataset as vendor
from l2core.detector.anchor import detect_window
from l2core.model import TableGeometry, load_model

REPO_ROOT = Path(__file__).resolve().parent.parent
MARKET_YAML = REPO_ROOT / "ui_model" / "market.yaml"
SCR_DIR = REPO_ROOT / "base_screenshot" / "scr"
WINDOW_SIZE = (1008, 674)  # размер окна market из модели

# Скрины меты с устойчиво найденным якорем «Обзора лавок» (две позиции окна).
SCREENSHOTS = ["scr_153600.png", "scr_154134.png"]


@pytest.fixture()
def geometry() -> TableGeometry:
    return load_model(MARKET_YAML).table_geometry


@pytest.fixture()
def vendor_state():
    """Снапшот module-level состояния вендора: wrapper меняет его
    процесс-wide, после теста возвращаем дефолт датасета."""
    snapshot = {k: getattr(vendor, k) for k in _VENDOR_CONSTANTS}
    yield snapshot
    for k, v in snapshot.items():
        setattr(vendor, k, v)


_VENDOR_CONSTANTS = (
    "CONTENT_X0", "COL_SEPS", "CONTENT_X1", "COLUMNS",
    "HEADER_Y0", "HEADER_Y1", "ROW_Y0", "ROW_PITCH", "ROW_H",
    "TABLE_BOTTOM", "MAX_ROWS", "NAME_TEXT_X", "PRICE_TEXT_X",
)


def _synth_window(geometry: TableGeometry):
    """Grayscale «окно» под константы МОДЕЛИ: заголовок + 3 непустые строки
    из max_rows (маркеры в зоне item), остальные пустые."""
    h, w = 700, 1000
    img = np.full((h, w), 60, dtype=np.uint8)
    img[geometry.row_y0 - 18:geometry.row_y0, :] = 100  # полоса заголовка
    item_zone = geometry.columns["item"].zone
    for k in range(3):
        y0 = geometry.row_y0 + k * geometry.row_pitch
        zone = img[y0:y0 + geometry.row_pitch,
                   item_zone.x:item_zone.x + item_zone.width]
        for i in range(6):
            zone[20:34, 10 + i * 30:20 + i * 30] = 220
    return img


# --- Маппинг констант модели -> вендора ---

def test_wrapper_injects_model_constants(geometry, vendor_state):
    TableSegmentation(geometry)
    assert vendor.COLUMNS == ("name", "price", "seller", "deal")
    assert vendor.CONTENT_X0 == geometry.col_seps[0] == 117
    assert vendor.COL_SEPS == tuple(geometry.col_seps[1:]) == (417, 648, 778, 967)
    assert vendor.CONTENT_X1 == geometry.col_seps[-1] == 967
    assert (vendor.HEADER_Y0, vendor.HEADER_Y1) == (245, 263)
    assert vendor.HEADER_Y1 == geometry.row_y0
    assert vendor.ROW_Y0 == 263
    assert vendor.ROW_PITCH == 56
    assert vendor.MAX_ROWS == 8
    assert vendor.ROW_H == geometry.row_pitch
    assert vendor.TABLE_BOTTOM == geometry.row_y0 + geometry.max_rows * geometry.row_pitch
    item, price = geometry.columns["item"].zone, geometry.columns["price"].zone
    assert vendor.NAME_TEXT_X == (item.x, item.x + item.width) == (161, 417)
    assert vendor.PRICE_TEXT_X == (price.x, price.x + price.width) == (437, 618)


def test_wrapper_restores_after_test(vendor_state):
    # Контроль фикстуры: дефолт вендора — калибровка датасета.
    assert vendor.COLUMNS == ("type", "name", "price", "seller", "deal")


# --- Валидация геометрии ---

def test_wrapper_requires_item_and_price_zones(geometry, vendor_state):
    from l2core.model import ColumnZone, GrammarSpec, Rect
    grammar = GrammarSpec("text", "text")
    base = {"seller": ColumnZone("seller", Rect(655, 10, 118, 38), grammar),
            "deal": ColumnZone("deal", Rect(790, 10, 140, 38), grammar)}
    g = TableGeometry(col_seps=(648, 778, 967), row_y0=263, row_pitch=56,
                      max_rows=8, columns=base)
    with pytest.raises(ValueError, match="item"):
        TableSegmentation(g)


def test_wrapper_requires_columns_match_seps(geometry, vendor_state):
    from l2core.model import ColumnZone, GrammarSpec, Rect
    grammar = GrammarSpec("text", "text")
    cols = {"item": ColumnZone("item", Rect(161, 10, 256, 38), grammar),
            "price": ColumnZone("price", Rect(437, 10, 181, 38), grammar)}
    g = TableGeometry(col_seps=(117, 417, 648, 778, 967), row_y0=263, row_pitch=56,
                      max_rows=8, columns=cols)
    with pytest.raises(ValueError, match="col_seps"):
        TableSegmentation(g)


# --- Поведение на синтетике (константы модели) ---

def test_rows_and_columns_on_model_constants(geometry, vendor_state):
    seg = TableSegmentation(geometry)
    gray = _synth_window(geometry)
    structure = seg.rows_and_columns(gray, origin=(0, 0))
    # Заголовок: полоса row_y0-18..row_y0, границы колонок — из col_seps.
    for rect in structure["header"].values():
        assert rect[1] == 245 and rect[3] == 263
    assert structure["cols"]["name"] == (117, 417)
    assert structure["cols"]["price"] == (417, 648)
    assert structure["cols"]["seller"] == (648, 778)
    assert structure["cols"]["deal"] == (778, 967)
    # 3 непустые строки из max_rows, высота строки = pitch, тайлинг вплотную.
    assert structure["rows"] == [(263, 319), (319, 375), (375, 431)]
    # TABLE_BOTTOM не биндинг (нижняя кромка = низ «окна»): max_rows не достигается.
    assert len(structure["rows"]) < geometry.max_rows


def test_extract_cells_trims_to_zones(geometry, vendor_state):
    seg = TableSegmentation(geometry)
    gray = _synth_window(geometry)
    structure = seg.rows_and_columns(gray, origin=(0, 0))
    # BGR «окно»: маркер в зоне текста каждой ячейки.
    color = np.full((gray.shape[0], gray.shape[1], 3), (40, 60, 80), np.uint8)
    for col, rect in structure["header"].items():
        x0, y0, x1, y1 = rect
        color[y0 + 5:y1 - 5, x0 + 5:x1 - 5] = (220, 220, 220)
    for row, (y0, y1) in enumerate(structure["rows"]):
        for col, (x0, x1) in structure["cols"].items():
            color[y0 + 20:y1 - 20, x0 + 5:x1 - 5] = (220, 220, 220)
    cells = seg.extract_cells(color, structure)
    assert len(cells) == 4 + 3 * 4
    by_col = {}
    for c in cells:
        by_col.setdefault(c.meta["col"], []).append(c)
    # Обрезка по зонам: name обрезан до зоны item (без иконки), price — до
    # зоны price; полные границы колонок шире зон — trim виден по ширине.
    name_w = geometry.columns["item"].zone.width      # 256 против колонки 300
    price_w = geometry.columns["price"].zone.width    # 181 против колонки 231
    for c in by_col["name"]:
        assert c.image.shape[1] <= name_w, c.image.shape
    for c in by_col["price"]:
        assert c.image.shape[1] <= price_w, c.image.shape
    # Ключи дедупликации стабильны (mask-hash, не пиксели).
    assert all(len(c.key) == 40 for c in cells)


# --- Интеграция: нарезка реальных скринов константами модели ---

@pytest.mark.parametrize("shot", SCREENSHOTS)
def test_real_screenshot_segmentation(shot, geometry, vendor_state):
    """Скрин меты -> якорь (детектор ядра) -> нарезка константами модели.

    Сверка с калибровкой 2.4 (tests/test_table_geometry.py): 7 строк,
    row_y0=263, pitch=56, границы колонок = col_seps, зоны текстов уважаются.
    """
    frame = cv2.imread(str(SCR_DIR / shot))
    assert frame is not None, shot
    at = detect_window(frame, WINDOW_SIZE)
    assert at is not None, shot
    x, y = at
    win_bgr = frame[y:y + WINDOW_SIZE[1], x:x + WINDOW_SIZE[0]]
    win_gray = cv2.cvtColor(win_bgr, cv2.COLOR_BGR2GRAY)

    seg = TableSegmentation(geometry)
    structure = seg.rows_and_columns(win_gray, origin=(x, y))
    # Строки: 7 штук, тайлинг от row_y0 с шагом pitch; row 7 упирается в низ окна.
    assert structure["rows"] == [
        (263 + 56 * k, 263 + 56 * (k + 1)) for k in range(7)
    ], shot
    # Заголовок — полоса над первой строкой.
    for rect in structure["header"].values():
        assert rect == (rect[0], 245, rect[2], 263), shot
    # Колонки — пары col_seps; ключ первой колонки маппится на «name».
    seps = geometry.col_seps
    names = ("name", "price", "seller", "deal")
    for i, name in enumerate(names):
        assert structure["cols"][name] == (seps[i], seps[i + 1]), (shot, name)

    cells = seg.extract_cells(frame, structure)
    assert len(cells) == 4 + 7 * 4, shot
    rows = {c.meta["row"] for c in cells}
    assert rows == {-1, 0, 1, 2, 3, 4, 5, 6}, shot
    # Зоны текстов не превышены (иконки исключены из name/price).
    name_w = geometry.columns["item"].zone.width
    price_w = geometry.columns["price"].zone.width
    for c in cells:
        if c.meta["col"] == "name":
            assert c.image.shape[1] <= name_w, (shot, c.meta)
        if c.meta["col"] == "price":
            assert c.image.shape[1] <= price_w, (shot, c.meta)


# --- Разделительные линии на кромках строк (живой кейс 2026-11-12) ---

def test_extract_cells_strips_rule_line_at_row_edge(geometry, vendor_state):
    """Яркая линия-разделитель на верхней кромке строки (y=row_y0, граница
    заголовка) не должна входить в кроп ячейки: она протяжённая, тянет
    bbox триммера на всю высоту строки, препроцесс OCR сжимает текст
    (чтение деградирует до уверенного мусора). Глифы текста не тронуты."""
    seg = TableSegmentation(geometry)
    h, w = 700, 1000
    color = np.full((h, w, 3), (45, 48, 52), np.uint8)
    # разделитель заголовок/строка 0: яркая линия на всю ширину таблицы
    color[geometry.row_y0, 117:967] = (200, 200, 200)
    # текст первой строки: маркер в зоне price
    pz = geometry.columns["price"].zone
    y0 = geometry.row_y0 + pz.y
    color[y0:y0 + 12, 437 + 20:437 + 90] = (230, 230, 230)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    structure = seg.rows_and_columns(gray, origin=(0, 0))
    cells = seg.extract_cells(color, structure)
    row0_price = [c for c in cells
                  if c.meta["col"] == "price" and c.meta["row"] == 0]
    assert row0_price, "строка 0 с текстом обязана нарезаться"
    cell = row0_price[0]
    # высота кропа — около текста (+pad триммера), а не всей строки с линией
    assert cell.image.shape[0] <= 24, cell.image.shape
