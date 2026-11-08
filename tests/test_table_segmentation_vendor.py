"""Guard-тесты вендоренной сегментации таблицы (задача 4.3, design.md
решение 12, spec ui-model «Геометрия таблицы …»).

Сравнивают вендоренный блок (l2core.adapters.table_segmentation.build_dataset)
с эталонами, сгенерированными ОРИГИНАЛЬНЫМ кодом L2-CRNN-Dataset-Builder/
build_dataset.py (коммит 606a65e, см. шапку вендоренного модуля). Эталоны
закоммичены в tests/fixtures/table_segmentation/ — для запуска guard-тестов
исходный репозиторий НЕ нужен.

Сравнение exact: входы сохранены в эталонах (те же байты), выходы —
структуры/хэши/байты кропов. Эталоны сгенерированы тем же окружением (numpy),
операции детерминированы (пороги, median, sha1). Обоснование —
отчёт task-4.3-report.md.

Перегенерация эталонов (вручную, вне CI):
    python tools/gen_table_segmentation_fixtures.py
"""
import json
from pathlib import Path

import numpy as np

from l2core.adapters.table_segmentation import build_dataset as vendor

FIXTURES = Path(__file__).parent / "fixtures" / "table_segmentation"


def _meta(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_constants_frozen():
    # Дефолтные константы вендора — калибровка датасета (референс 1136x833,
    # ruling R5: устарела для рантайма, но тело вендора к ней привязано).
    # Guard: правка констант вендора (как и тела) запрещена — ловим drift.
    assert vendor.CONTENT_X0 == 14
    assert vendor.COL_SEPS == (72, 536, 764, 912, 966)
    assert vendor.CONTENT_X1 == 986
    assert (vendor.HEADER_Y0, vendor.HEADER_Y1) == (242, 260)
    assert vendor.ROW_Y0 == 261
    assert vendor.ROW_PITCH == 56
    assert vendor.ROW_H == 55
    assert vendor.TABLE_BOTTOM == 660
    assert vendor.MAX_ROWS == 8
    assert vendor.COLUMNS == ("type", "name", "price", "seller", "deal")
    assert vendor.NAME_TEXT_X == (119, 534)
    assert vendor.PRICE_TEXT_X == (556, 736)


def test_rows_and_columns_reference():
    # Эталон: нарезка структуры на фиксированном grayscale-входе ==
    # вывод оригинала (header/rows/cols/origin), точь-в-точь.
    expected = _meta("rows_and_columns.json")
    with np.load(FIXTURES / "segmentation_cases.npz") as npz:
        got = vendor.rows_and_columns(npz["win_gray"], origin=(7, 9))
    assert got["origin"] == tuple(expected["origin"])
    assert [list(r) for r in got["rows"]] == expected["rows"]
    assert {k: list(v) for k, v in got["cols"].items()} == expected["cols"]
    assert {k: list(v) for k, v in got["header"].items()} == expected["header"]


def test_extract_cells_reference():
    # Эталон: вырезание ячеек (key/pixel_key/мета/байты кропов) == оригинал.
    expected = _meta("extract_cells.json")
    with np.load(FIXTURES / "segmentation_cases.npz") as npz:
        structure = vendor.rows_and_columns(npz["win_gray"], origin=(7, 9))
        got = vendor.extract_cells(npz["color"], structure)
        assert len(got) == len(expected)
        for cell, exp in zip(got, expected):
            assert cell.key == exp["key"], exp["image"]
            assert cell.meta["col"] == exp["col"], exp["image"]
            assert cell.meta["row"] == exp["row"], exp["image"]
            assert list(cell.meta["rect"]) == exp["rect"], exp["image"]
            assert cell.meta["pixel_key"] == exp["pixel_key"], exp["image"]
            assert list(cell.image.shape) == exp["shape"], exp["image"]
            assert np.array_equal(cell.image, npz[exp["image"]]), exp["image"]


def test_trim_to_text_reference():
    # Эталон: обрезка до текста (bbox + pad=4; пустой вход — без изменений).
    with np.load(FIXTURES / "segmentation_cases.npz") as npz:
        for name in (n for n in npz.files if n.startswith("trim_in_")):
            case = name[len("trim_in_"):]
            got = vendor.trim_to_text(npz[name])
            assert np.array_equal(got, npz[f"trim_out_{case}"]), case
