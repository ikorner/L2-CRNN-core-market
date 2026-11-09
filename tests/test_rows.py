"""Тесты чтения строк таблицы: Row/read_rows/pHash/дедуп (задача 6.1,
spec ocr-reading «Чтение строк таблицы с Row и дедупликацией»: Row = ячейки
по колонкам + screen_y + icon_hash (pHash) + partial; дедуп между кадрами
по icon_hash+цена+y; результат включает snapshot_id).

Детерминизм (инвариант 7): кадры синтетические (маркеры-цвета в зонах
колонок), OCR — фейк по доминантному нефоновому цвету кропа (контракт
OcrClient), vendor-константы сегментации снапшотятся/восстанавливаются.
"""
import hashlib

import numpy as np
import pytest

from l2core.detector.anchor import Anchor
from l2core.model import (
    ColumnZone,
    GrammarSpec,
    Rect,
    TableGeometry,
    UiModel,
    VariantSpec,
    WindowSpec,
)
from l2core.model.reading import OcrReadError
from l2core.model.rows import (
    DEFAULT_HAMMING_THRESHOLD,
    CellValue,
    Row,
    RowsResult,
    dedup_rows,
    hamming_distance,
    phash,
    read_rows,
)

ANCHOR = Anchor(window_id="market", x=0, y=0, score=0.99, method="template")
OFFSET_ANCHOR = Anchor(window_id="market", x=5, y=7, score=0.99, method="template")
WINDOW_SIZE = (160, 100)
FRAME_SIZE = (200, 120)  # (w, h)

# Цвета-маркеры: (текст, цена, продавец) -> цвет BGR; цвет иконки отдельно.
COLOR_ITEM_A = (250, 120, 40)
COLOR_ITEM_B = (40, 220, 60)
COLOR_ITEM_C = (60, 80, 230)
COLOR_PRICE_A = (200, 200, 200)
COLOR_PRICE_B = (180, 160, 255)
COLOR_SELLER = (90, 140, 240)
ICON_A = (10, 90, 90)
ICON_B = (90, 10, 90)
ICON_C = (90, 90, 10)
ICON_A_NEAR = (10, 90, 90)  # почти та же иконка (2 пикселя отличаются)

TEXTS_BY_COLOR = {
    COLOR_ITEM_A: ("Меч", 0.99),
    COLOR_ITEM_B: ("Щит", 0.99),
    COLOR_ITEM_C: ("Зелье", 0.99),
    COLOR_PRICE_A: ("1500000", 0.99),
    COLOR_PRICE_B: ("25000", 0.99),
    COLOR_SELLER: ("Seller", 0.99),
}

# Геометрия таблицы: 3 слота (row_y0=20, pitch=10), колонки
# item(10..50, текст 30..48, иконка 10..30) / price(50..90) / seller(90..130).
GEOMETRY = TableGeometry(
    col_seps=(10, 50, 90, 130),
    row_y0=20,
    row_pitch=10,
    max_rows=3,
    columns={
        "item": ColumnZone(id="item", zone=Rect(30, 2, 18, 6),
                           grammar=GrammarSpec(name="text", kind="text")),
        "price": ColumnZone(id="price", zone=Rect(54, 2, 20, 6),
                            grammar=GrammarSpec(name="price", kind="price")),
        "seller": ColumnZone(id="seller", zone=Rect(94, 2, 20, 6),
                             grammar=GrammarSpec(name="text", kind="text")),
    },
)

# Зоны текстов колонок: (x0, y0, x1, y1) от верха строки (как zone колонок).
# Маркер закрашивает 4 из 10 строк зоны — меньше половины: медиана фона
# вендорной эвристики _row_is_empty остаётся фоном (как в tests/test_scroll).
TEXT_ZONES = {
    "item": (30, 3, 48, 7),
    "price": (54, 3, 74, 7),
    "seller": (94, 3, 114, 7),
}
ITEM_CELL_X = (10, 50)
ICON_X = (10, 30)  # левая часть ячейки item — зона иконки


@pytest.fixture(autouse=True)
def vendor_state():
    """Снапшот/восстановление module-level констант вендора сегментации."""
    from l2core.adapters.table_segmentation import build_dataset as vendor

    keys = (
        "CONTENT_X0", "COL_SEPS", "CONTENT_X1", "COLUMNS",
        "HEADER_Y0", "HEADER_Y1", "ROW_Y0", "ROW_PITCH", "ROW_H",
        "TABLE_BOTTOM", "MAX_ROWS", "NAME_TEXT_X", "PRICE_TEXT_X",
    )
    snapshot = {k: getattr(vendor, k) for k in keys}
    yield
    for k, v in snapshot.items():
        setattr(vendor, k, v)


def make_model():
    return UiModel(
        path="test",
        window=WindowSpec(id="market", title="Обзор лавок", size=WINDOW_SIZE),
        elements={},
        states={},
        transitions=(),
        ocr_regions={},
        table_geometry=GEOMETRY,
    )


class ColorOcr:
    """Фейковый OcrClient: текст по доминантному нефоновому цвету кропа.

    Расширение региона (перечитывание 4.4/4.5) добавляет фоновые пиксели —
    доминантный цвет маркера сохраняется, ключ устойчив.
    """

    def __init__(self, texts_by_color=None):
        self.texts_by_color = dict(texts_by_color or TEXTS_BY_COLOR)

    def read(self, region_img):
        img = np.asarray(region_img)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        px = img.reshape(-1, 3).astype(np.int16)
        bg = np.median(px, axis=0)
        mask = np.abs(px - bg).max(axis=1) > 25
        if not mask.any():
            return ("", 0.5)
        colors, counts = np.unique(px[mask], axis=0, return_counts=True)
        color = tuple(int(c) for c in colors[counts.argmax()])
        return self.texts_by_color.get(color, ("???", 0.1))


def make_frame(anchor=ANCHOR, rows=None):
    """Синтетический кадр: rows = {slot: {"texts": {col: color}, "icon": color}}.

    Маркер — заливка зоны текста колонки цветом; иконка — заливка левой
    части ячейки item. Незакрашенная зона = пустая ячейка.
    """
    frame = np.full((FRAME_SIZE[1], FRAME_SIZE[0], 3), 30, dtype=np.uint8)
    for slot, spec in (rows or {}).items():
        y0 = GEOMETRY.row_y0 + slot * GEOMETRY.row_pitch
        for col, color in spec.get("texts", {}).items():
            zx0, zy0, zx1, zy1 = TEXT_ZONES[col]
            frame[anchor.y + y0 + zy0:anchor.y + y0 + zy1,
                  anchor.x + zx0:anchor.x + zx1] = color
        icon = spec.get("icon")
        if icon is not None:
            frame[anchor.y + y0:anchor.y + y0 + GEOMETRY.row_pitch,
                  anchor.x + ICON_X[0]:anchor.x + ICON_X[1]] = icon
    return frame


def read(frame, ocr=None, anchor=ANCHOR, **kwargs):
    return read_rows(
        frame,
        model=make_model(),
        ocr=ocr or ColorOcr(),
        anchor=anchor,
        threshold=0.85,
        reread_count=1,
        **kwargs,
    )


def textured_icon(seed_shift=0):
    """Текстурная иконка (детерминированный паттерн) — как реальная."""
    base = (np.arange(200, dtype=np.int64).reshape(10, 20) * 7 + seed_shift) % 256
    return np.repeat(base[:, :, None], 3, axis=2).astype(np.uint8)


class TestPhash:
    def test_determinism(self):
        img = np.full((20, 20, 3), 100, dtype=np.uint8)
        assert phash(img) == phash(img.copy())

    def test_same_pixels_any_channel_order_of_calls(self):
        a = np.zeros((16, 16, 3), dtype=np.uint8)
        a[4:12, 4:12] = (200, 100, 50)
        assert phash(a) == phash(np.ascontiguousarray(a))

    def test_different_images_differ(self):
        a = np.full((20, 20, 3), 100, dtype=np.uint8)
        b = np.full((20, 20, 3), 100, dtype=np.uint8)
        b[0:10] = 200
        assert phash(a) != phash(b)

    def test_hamming_distance(self):
        assert hamming_distance(0, 0) == 0
        assert hamming_distance(0, (1 << 64) - 1) == 64
        assert hamming_distance(0b1010, 0b0011) == 2

    def test_near_icons_close_hashes(self):
        a = textured_icon()
        b = a.copy()
        b[3, 7] = (150, 150, 150)
        b[8, 15] = (150, 150, 150)
        d = hamming_distance(phash(a), phash(b))
        assert 0 < d <= DEFAULT_HAMMING_THRESHOLD


class TestReadRows:
    def test_cells_by_columns_and_values(self):
        frame = make_frame(rows={
            0: {"texts": {"item": COLOR_ITEM_A, "price": COLOR_PRICE_A,
                          "seller": COLOR_SELLER}, "icon": ICON_A},
            1: {"texts": {"item": COLOR_ITEM_B, "price": COLOR_PRICE_B,
                          "seller": COLOR_SELLER}, "icon": ICON_B},
        })
        result = read(frame)
        assert isinstance(result, RowsResult)
        assert len(result.rows) == 2
        first, second = result.rows
        assert set(first.cells) == {"item", "price", "seller"}
        assert first.cells["item"] == CellValue(text="Меч", value="Меч")
        assert first.cells["price"] == CellValue(text="1500000", value=1500000)
        assert second.cells["price"].value == 25000

    def test_screen_y_in_window_coords(self):
        frame = make_frame(rows={
            1: {"texts": {"item": COLOR_ITEM_A}, "icon": ICON_A},
        })
        (row,) = read(frame).rows
        assert row.screen_y == GEOMETRY.row_y0 + GEOMETRY.row_pitch

    def test_anchor_offset_does_not_shift_window_coords(self):
        frame = make_frame(anchor=OFFSET_ANCHOR, rows={
            0: {"texts": {"item": COLOR_ITEM_A}, "icon": ICON_A},
        })
        (row,) = read(frame, anchor=OFFSET_ANCHOR).rows
        assert row.screen_y == GEOMETRY.row_y0
        assert row.cells["item"].value == "Меч"

    def test_icon_hash_present_and_stable_across_frames(self):
        f1 = make_frame(rows={
            0: {"texts": {"item": COLOR_ITEM_A}, "icon": ICON_A}})
        f2 = make_frame(rows={
            1: {"texts": {"item": COLOR_ITEM_A}, "icon": ICON_A}})
        r1 = read(f1).rows[0]
        r2 = read(f2).rows[0]
        assert r1.icon_hash is not None
        assert r1.icon_hash == r2.icon_hash

    def test_different_icons_different_hashes(self):
        f = make_frame(rows={
            0: {"texts": {"item": COLOR_ITEM_A}, "icon": ICON_A},
            1: {"texts": {"item": COLOR_ITEM_B}, "icon": ICON_B},
        })
        rows = read(f).rows
        assert rows[0].icon_hash != rows[1].icon_hash

    def test_partial_edge_slots_only(self):
        frame = make_frame(rows={
            0: {"texts": {"item": COLOR_ITEM_A}, "icon": ICON_A},
            1: {"texts": {"item": COLOR_ITEM_B}, "icon": ICON_B},
            2: {"texts": {"item": COLOR_ITEM_C}, "icon": ICON_C},
        })
        rows = read(frame).rows
        assert [r.partial for r in rows] == [True, False, True]

    def test_snapshot_id_is_frame_sha1(self):
        frame = make_frame(rows={
            0: {"texts": {"item": COLOR_ITEM_A}, "icon": ICON_A}})
        result = read(frame)
        expected = hashlib.sha1(np.ascontiguousarray(frame).tobytes())
        assert result.snapshot_id == expected.hexdigest()

    def test_snapshot_id_differs_between_frames(self):
        f1 = make_frame(rows={
            0: {"texts": {"item": COLOR_ITEM_A}, "icon": ICON_A}})
        f2 = make_frame(rows={
            0: {"texts": {"item": COLOR_ITEM_B}, "icon": ICON_B}})
        assert read(f1).snapshot_id != read(f2).snapshot_id

    def test_explicit_snapshot_id_wins(self):
        frame = make_frame(rows={
            0: {"texts": {"item": COLOR_ITEM_A}, "icon": ICON_A}})
        assert read(frame, snapshot_id="snap-42").snapshot_id == "snap-42"

    def test_empty_price_cell_reads_as_none(self):
        frame = make_frame(rows={
            0: {"texts": {"item": COLOR_ITEM_A, "seller": COLOR_SELLER},
                "icon": ICON_A},
        })
        (row,) = read(frame).rows
        assert row.cells["price"] == CellValue(text="", value=None)

    def test_window_outside_frame_gives_empty_result(self):
        frame = np.full((10, 10, 3), 30, dtype=np.uint8)
        result = read(frame)
        assert result.rows == ()

    def test_low_confidence_raises(self):
        frame = make_frame(rows={
            0: {"texts": {"item": (250, 250, 20)}, "icon": ICON_A},
        })
        with pytest.raises(OcrReadError):
            read(frame, ocr=ColorOcr({}))

    def test_model_without_geometry_fails_fast(self):
        model = make_model()
        bare = UiModel(
            path=model.path, window=model.window, elements=model.elements,
            states=model.states, transitions=model.transitions,
            ocr_regions=model.ocr_regions, table_geometry=None,
        )
        from l2core.model.rows import RowsError

        with pytest.raises(RowsError):
            read_rows(np.zeros((10, 10, 3), dtype=np.uint8), model=bare,
                      ocr=ColorOcr(), anchor=ANCHOR,
                      threshold=0.85, reread_count=1)


class TestDedup:
    def _row(self, icon, price, y, color=COLOR_ITEM_A):
        return Row(
            cells={
                "item": CellValue(text="X", value="X"),
                "price": CellValue(text=str(price), value=price),
            },
            screen_y=y,
            icon_hash=phash(np.full((10, 20, 3), icon, dtype=np.uint8)),
            partial=False,
        )

    def test_same_row_on_two_frames_deduped(self):
        icon = np.full((10, 20, 3), ICON_A, dtype=np.uint8)
        frame_rows = [
            Row(cells={"price": CellValue("1500000", 1500000)},
                screen_y=20, icon_hash=phash(icon), partial=True),
            Row(cells={"price": CellValue("25000", 25000)},
                screen_y=30, icon_hash=phash(icon + 1), partial=False),
            Row(cells={"price": CellValue("1500000", 1500000)},
                screen_y=20, icon_hash=phash(icon), partial=True),
        ]
        merged = dedup_rows(frame_rows)
        assert len(merged) == 2
        # Порядок первого появления сохранён.
        assert [r.cells["price"].value for r in merged] == [1500000, 25000]

    def test_same_icon_and_price_but_moved_row_kept(self):
        rows = [self._row(ICON_A, 100, 20), self._row(ICON_A, 100, 30)]
        assert len(dedup_rows(rows)) == 2

    def test_same_icon_and_y_but_other_price_kept(self):
        rows = [self._row(ICON_A, 100, 20), self._row(ICON_A, 200, 20)]
        assert len(dedup_rows(rows)) == 2

    def test_other_icon_same_price_and_y_kept(self):
        rows = [self._row(ICON_A, 100, 20), self._row(ICON_B, 100, 20)]
        assert len(dedup_rows(rows)) == 2

    def test_near_icons_deduped_by_default_threshold(self):
        a = textured_icon()
        b = a.copy()
        b[3, 7] = (150, 150, 150)
        b[8, 15] = (150, 150, 150)
        rows = [
            Row(cells={"price": CellValue("100", 100)}, screen_y=20,
                icon_hash=phash(a), partial=False),
            Row(cells={"price": CellValue("100", 100)}, screen_y=20,
                icon_hash=phash(b), partial=True),
        ]
        assert len(dedup_rows(rows)) == 1
        # Строгое сравнение хэшей: те же строки не дедупятся.
        assert len(dedup_rows(rows, hamming_threshold=0)) == 2

    def test_missing_price_uses_text_key(self):
        rows = [
            Row(cells={"item": CellValue("a", "a")}, screen_y=20,
                icon_hash=1, partial=False),
            Row(cells={"item": CellValue("a", "a")}, screen_y=20,
                icon_hash=1, partial=False),
            Row(cells={"item": CellValue("b", "b")}, screen_y=20,
                icon_hash=1, partial=False),
        ]
        assert len(dedup_rows(rows)) == 2
