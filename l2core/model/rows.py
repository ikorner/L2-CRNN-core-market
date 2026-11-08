"""Чтение строк таблицы списка: Row, pHash иконки, дедупликация
(задача 6.1, spec ocr-reading «Чтение строк таблицы с Row и дедупликацией»).

`read_rows()` возвращает строки как `Row`: словарь ячеек по именам колонок
модели (`cells`) + `screen_y` + `icon_hash` (pHash иконки предмета) +
`partial: bool` для обрезанных краевых строк; результат `RowsResult`
включает `snapshot_id`. Дедупликация одинаковых строк между кадрами
скролла — `dedup_rows()` по icon_hash + цене + y.

КООРДИНАТНАЯ СИСТЕМА: `screen_y` — y верхней кромки строки в координатах
ОКНА (от якоря, 0..высота окна), не абсолютный экранный пиксель: дедуп
между кадрами имеет смысл в координатах окна — окно может сдвинуться
между кадрами, а слот строки в нём остаётся тем же. Абсолютный экранный
y = anchor.y + screen_y.

СНИМОК (snapshot_id): детерминированный идентификатор кадра-источника —
sha1 по байтам кадра (hex). Рынок меняется во время сканирования:
стратегия сверяет, что строки одного агрегата взяты из разных снимков.
Идентичные кадры дают одинаковый id; можно передать свой (запись сессии).

pHash ИКОНКИ: классический перцептивный хэш (DCT, как imagehash.phash):
серый -> ресайз 32x32 (ближайший сосед, numpy-only) -> DCT-II -> низкие
8x8 частоты -> биты «>= медианы» -> int64. Детерминирован (без cv2/BLAS-
зависимостей). Сравнение — расстоянием Хэмминга; порог по умолчанию
DEFAULT_HAMMING_THRESHOLD=6 из 64 бит: допуск к мелким различиям отрисовки
одной иконки (альтернирующий фон строк, сжатие), зафиксирован тестами
tests/test_rows.py (параметр открыт, строгое сравнение — threshold=0).

ЧТЕНИЕ ЯЧЕЕК: нарезка TableSegmentation (4.3) на константах table_geometry;
каждая непустая ячейка читается read_confident (4.5) с грамматикой колонки
(ColumnZone.grammar). Пустая ячейка (нет пикселей текста) — честная
CellValue("", None) без вызова OCR. Исчерпание перечитываний — OcrReadError
стратегии с диагностикой (не маскируется).

PARTIAL: краевые слоты вьюпорта (первый и последний из max_rows) помечаются
partial=True: без информации о позиции скролла консервативно считаются
потенциально обрезанными краем окна (честная граница: вызывающий слой при
полностью видимом списке может игнорировать флаг). Внутренние слоты
геометрически внутри окна — всегда полные.

Модуль — слой чтения модели (как reading.py): сегментация и OCR —
инъекции/контракты, без HID и времени (инварианты 1/9).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Tuple, Union

import numpy as np

from l2core.adapters.table_segmentation import TableSegmentation
from l2core.detector.anchor import Anchor
from l2core.model import ColumnZone, OcrRegionSpec, Rect, UiModel
from l2core.model.reading import read_confident

__all__ = [
    "DEFAULT_HAMMING_THRESHOLD",
    "CellValue",
    "Row",
    "RowsError",
    "RowsResult",
    "dedup_rows",
    "hamming_distance",
    "phash",
    "read_rows",
]

# Порог сравнения иконок по Хэммингу (из 64 бит): одинаковые предметы на
# разных кадрах/фонах дают близкие хэши; разные — далёкие. Происхождение:
# допуск к мелким пережаткам отрисовки, поведение закреплено пинами
# tests/test_rows.py (включая threshold=0 для строгого равенства).
DEFAULT_HAMMING_THRESHOLD = 6

# Размеры pHash: входной ресайз и сторона квадрата низких частот.
_PHASH_INPUT = 32
_PHASH_SIZE = 8

# Маппинг ключей колонок вендора обратно на ключи модели (только «name»).
_VENDOR_TO_MODEL_COLUMN = {"name": "item"}

# Минимум пикселей текста в обрезанной ячейке: ниже — ячейка считается пустой
# (вендорная эвристика _text_mask: отклонение от фона > 25 на канал).
_TEXT_MIN_PIXELS = 1


class RowsError(Exception):
    """Ошибка чтения строк (модель/аргументы), не эскалация рантайма."""

    def __init__(self, detail: str):
        super().__init__(f"read_rows: {detail}")
        self.detail = detail


@dataclass(frozen=True)
class CellValue:
    """Ячейка строки: сырое прочитанное text и разобранное value.

    value — результат parse_value по грамматике колонки (int для price/
    digits, str для text); None — ячейка пустая (честное значение, OCR
    не вызывался).
    """

    text: str
    value: Optional[Union[int, str]] = None


@dataclass(frozen=True)
class Row:
    """Строка таблицы списка (spec ocr-reading).

    cells — ячейки по именам колонок модели; screen_y — y верхней кромки
    строки в координатах окна (от якоря); icon_hash — pHash иконки предмета
    (None — зона иконки не выражена геометрией); partial — краевая строка
    за пределами safe-зоны (потенциально обрезана краем окна).
    """

    cells: Mapping[str, CellValue]
    screen_y: int
    icon_hash: Optional[int]
    partial: bool


@dataclass(frozen=True)
class RowsResult:
    """Строки одного кадра + идентификатор снимка-источника."""

    rows: Tuple[Row, ...]
    snapshot_id: str


# --- pHash ---

_DCT_CACHE: dict = {}


def _dct_matrix(n: int) -> np.ndarray:
    """Ортонормированная матрица DCT-II (кэшируется, детерминирована)."""
    if n not in _DCT_CACHE:
        k = np.arange(n)[:, None]
        i = np.arange(n)[None, :]
        d = np.cos(np.pi * (2 * i + 1) * k / (2 * n))
        d[0] *= np.sqrt(1.0 / n)
        d[1:] *= np.sqrt(2.0 / n)
        _DCT_CACHE[n] = d
    return _DCT_CACHE[n]


def _resize_nearest(gray: np.ndarray, size: int) -> np.ndarray:
    """Ресайз ближайшим соседом (numpy-only, детерминирован)."""
    ys = (np.arange(size) * (gray.shape[0] / size)).astype(int)
    xs = (np.arange(size) * (gray.shape[1] / size)).astype(int)
    ys = np.clip(ys, 0, gray.shape[0] - 1)
    xs = np.clip(xs, 0, gray.shape[1] - 1)
    return gray[ys][:, xs]


def phash(image: np.ndarray, hash_size: int = _PHASH_SIZE,
          input_size: int = _PHASH_INPUT) -> int:
    """Перцептивный хэш изображения (imagehash.phash-совместимая схема).

    Серый (BT.601) -> ресайз input_size x input_size -> DCT -> низкие
    hash_size x hash_size частоты -> биты «>= медианы» -> int (hash_size² бит).
    Детерминирован, чистая функция (инвариант 7).
    """
    img = np.asarray(image)
    if img.ndim == 3:
        gray = img[..., 0] * 0.114 + img[..., 1] * 0.587 + img[..., 2] * 0.299
    else:
        gray = img.astype(np.float64)
    small = _resize_nearest(np.asarray(gray, dtype=np.float64), input_size)
    d = _dct_matrix(input_size)
    low = (d @ small @ d.T)[:hash_size, :hash_size]
    bits = low >= np.median(low)
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return value


def hamming_distance(a: int, b: int) -> int:
    """Расстояние Хэмминга между двумя хэшами (bit_count, Python 3.10+)."""
    return (a ^ b).bit_count()


# --- read_rows ---

def _to_gray(crop: np.ndarray) -> np.ndarray:
    """BGR-кроп -> серый (BT.601, как в detector.anchor/scroll)."""
    return crop[..., 0] * 0.114 + crop[..., 1] * 0.587 + crop[..., 2] * 0.299


def _has_text(crop: np.ndarray) -> bool:
    """Есть ли в кропе пиксели текста (эвристика вендора _text_mask)."""
    img = np.asarray(crop)
    if img.ndim == 3:
        bg = np.median(img.reshape(-1, 3), axis=0)
        mask = (np.abs(img.astype(np.int16) - bg).max(axis=2) > 25)
    else:
        bg = np.median(img)
        mask = np.abs(img.astype(np.int16) - int(bg)) > 25
    return int(mask.sum()) >= _TEXT_MIN_PIXELS


def _default_snapshot_id(frame: np.ndarray) -> str:
    """Детерминированный id кадра: sha1 по байтам (идентичные кадры — тот же id)."""
    return hashlib.sha1(np.ascontiguousarray(frame).tobytes()).hexdigest()


def _icon_hash(frame: np.ndarray, anchor: Anchor, y0: int, y1: int,
               x0: int, text_x0: int) -> Optional[int]:
    """pHash иконки предмета: левая часть ячейки item до зоны текста.

    Зона иконки не выражена геометрией (зона текста начинается с левого
    края ячейки) — None (дедуп опирается на цену+y).
    """
    if text_x0 <= x0:
        return None
    crop = frame[anchor.y + y0:anchor.y + y1, anchor.x + x0:anchor.x + text_x0]
    if crop.size == 0:
        return None
    return phash(crop)


def read_rows(
    frame: np.ndarray,
    *,
    model: UiModel,
    ocr,
    anchor: Anchor,
    threshold: float,
    reread_count: int,
    segmentation: Optional[TableSegmentation] = None,
    snapshot_id: Optional[str] = None,
    pad: int = 8,
) -> RowsResult:
    """Прочитать строки таблицы списка с кадра (только чтение, без HID).

    Нарезка TableSegmentation на константах table_geometry модели; каждая
    непустая ячейка читается read_confident с грамматикой колонки; иконка
    предмета — pHash левой части ячейки колонки item. Окно вне кадра —
    пустой RowsResult (читать нечего). table_geometry обязательна
    (RowsError, fail fast). Окно скролла/клики — НЕ здесь: read_rows только
    читает переданный кадр (обзор — разведка цен, чужие лавки не открываются).
    """
    frame = np.asarray(frame)
    if model.table_geometry is None:
        raise RowsError(
            f"окно {model.window.id!r}: table_geometry отсутствует — "
            f"строки таблицы без неё не выражены")
    sid = snapshot_id or _default_snapshot_id(frame)
    seg = segmentation or TableSegmentation(model.table_geometry)
    geo = model.table_geometry

    w, h = model.window.size
    crop = frame[anchor.y:anchor.y + h, anchor.x:anchor.x + w]
    if crop.shape[0] < h or crop.shape[1] < w:
        return RowsResult(rows=(), snapshot_id=sid)

    structure = seg.rows_and_columns(_to_gray(crop), origin=(anchor.x, anchor.y))
    slots = structure["rows"]
    if not slots:
        return RowsResult(rows=(), snapshot_id=sid)
    cells = seg.extract_cells(frame, structure)
    by_slot: dict = {}
    for cell in cells:
        if cell.meta["row"] < 0:
            continue
        by_slot.setdefault(cell.meta["row"], []).append(cell)

    item_cell_x0 = structure["cols"]["name"][0]
    item_text_x0 = geo.columns["item"].zone.x

    rows = []
    for slot, (y0, y1) in enumerate(slots):
        # Слот в сетке (не индекс среди непустых): краевость — от слота.
        grid_k = round((y0 - geo.row_y0) / geo.row_pitch)
        partial = grid_k == 0 or grid_k == geo.max_rows - 1
        row_cells: dict = {}
        for cell in sorted(by_slot.get(slot, ()), key=lambda c: c.meta["rect"]):
            col = _VENDOR_TO_MODEL_COLUMN.get(cell.meta["col"], cell.meta["col"])
            column: ColumnZone = geo.columns[col]
            if not _has_text(cell.image):
                row_cells[col] = CellValue(text="", value=None)
                continue
            cx0, cy0, cx1, cy1 = cell.meta["rect"]
            region = OcrRegionSpec(
                id=f"{model.window.id}:row{grid_k}:{col}",
                rect=Rect(cx0, cy0, cx1 - cx0, cy1 - cy0),
                grammar=column.grammar,
            )
            result = read_confident(
                ocr, frame, region,
                threshold=threshold, reread_count=reread_count, pad=pad,
            )
            row_cells[col] = CellValue(text=result.text, value=result.value)
        icon = _icon_hash(frame, anchor, y0, y1, item_cell_x0, item_text_x0)
        rows.append(Row(cells=row_cells, screen_y=y0,
                        icon_hash=icon, partial=partial))
    return RowsResult(rows=tuple(rows), snapshot_id=sid)


# --- Дедупликация ---

def _price_key(row: Row, price_column: str):
    """Ключ цены строки: разобранное значение колонки цен, иначе текст.

    Колонки цен в модели нет — ключом служат тексты всех ячеек строки
    (дедуп остаётся детерминированным и не схлопывает разные строки).
    """
    cell = row.cells.get(price_column)
    if cell is not None:
        return cell.value if cell.value is not None else cell.text
    return tuple(sorted((col, c.text) for col, c in row.cells.items()))


def _icons_match(a: Optional[int], b: Optional[int],
                 hamming_threshold: int) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return hamming_distance(a, b) <= hamming_threshold


def dedup_rows(
    rows: Iterable[Row],
    *,
    price_column: str = "price",
    hamming_threshold: int = DEFAULT_HAMMING_THRESHOLD,
) -> Tuple[Row, ...]:
    """Сквозной дедуп строк между кадрами скролла (spec ocr-reading).

    Ключ строки: icon_hash (по Хэммингу <= threshold) + цена + y (все три
    совпали — та же строка). Порядок — по первому появлению. Порог —
    параметр (происхождение дефолта — в докстринге модуля).
    """
    accepted: list = []
    for row in rows:
        price = _price_key(row, price_column)
        duplicate = any(
            _icons_match(prev.icon_hash, row.icon_hash, hamming_threshold)
            and _price_key(prev, price_column) == price
            and prev.screen_y == row.screen_y
            for prev in accepted
        )
        if not duplicate:
            accepted.append(row)
    return tuple(accepted)
