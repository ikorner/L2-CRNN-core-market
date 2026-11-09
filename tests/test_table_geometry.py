"""Golden-тесты констант геометрии таблицы (задача 2.4).

Константы секции table_geometry откалиброваны из меты (base_screenshot/scr/,
39 скриншотов 1920x1080, метод — в отчёте задачи 2.4). Этот тест фиксирует
итоговые значения: любое изменение модели без осознанного обновления
golden-теста падает. Инварианты зон (вписывание в колонки/строку) —
защита от опечаток разметчика поверх golden-значений.

Референс 1136x833 (L2-CRNN-Dataset-Builder, COL_SEPS=(72,536,764,912,966)
и т.д.) УСТАРЕЛ (ruling R5): патч игры изменил геометрию таблицы, значения
ниже выведены заново из скриншотов меты.
"""

from pathlib import Path

import pytest

from l2core.model import load_model

REPO_ROOT = Path(__file__).resolve().parent.parent
MARKET_YAML = REPO_ROOT / "ui_model" / "market.yaml"

# --- Итоговая калибровка под 1920x1080 (метод см. отчёт задачи 2.4) ---

# Разделители колонок заголовка таблицы (1px вертикальные линии,
# скриншоты scr_153600/153611/153619/154134, window-relative):
#   117  Тип|Предмет, 417  Предмет|Цена, 648  Цена|Персонаж,
#   778  Персонаж|Сделка, 967 — правый край контента перед скроллбаром
#   (панель строки заканчивается на 968, скроллбар с 973; старое 966).
COL_SEPS = (117, 417, 648, 778, 967)
ROW_Y0 = 263       # верх первой строки данных (переход фона после заголовка)
ROW_PITCH = 56     # шаг строк: (657-264)/7 = 56.1, линии строк ровно 56
MAX_ROWS = 8       # потолок сканирования (источник); помещается 7 строк

# Зоны текста колонок: x — от якоря окна, y — от верха строки.
# Выведены из пиксельных кластеров текста строк (скрины меты,
# полоса текста y+20..+34 для имени): иконка 127..159 (рамка квадрата
# 32px), текст имени начинается с x=164 (зона берётся с 161 — запас
# после кромки иконки), цена право-выравнена и заканчивается на x=616
# во всех строках (иконка свитка с 625), продавец 703..772, сделка
# «Аден» 894..918 (иконка монеты с ~935).
COLUMNS = {
    "item": {"zone": (161, 10, 256, 38), "grammar": "text"},
    "price": {"zone": (437, 10, 181, 38), "grammar": "price"},
    "seller": {"zone": (655, 10, 118, 38), "grammar": "text"},
    "deal": {"zone": (790, 10, 140, 38), "grammar": "text"},
}


def _geometry():
    return load_model(MARKET_YAML).table_geometry


def test_market_has_table_geometry():
    geometry = _geometry()
    assert geometry is not None, "у market (окно со списком) обязана быть table_geometry"


def test_golden_constants():
    """Итоговые значения констант — под любое изменение модели тест красный."""
    geometry = _geometry()
    assert geometry.col_seps == COL_SEPS
    assert geometry.row_y0 == ROW_Y0
    assert geometry.row_pitch == ROW_PITCH
    assert geometry.max_rows == MAX_ROWS


def test_golden_column_zones():
    geometry = _geometry()
    assert set(geometry.columns) == set(COLUMNS)
    for name, expected in COLUMNS.items():
        zone = geometry.columns[name]
        assert tuple(zone.zone) == expected["zone"], name
        assert zone.grammar.name == expected["grammar"], name


def test_col_seps_strictly_increasing():
    geometry = _geometry()
    assert list(geometry.col_seps) == sorted(geometry.col_seps)
    assert len(set(geometry.col_seps)) == len(geometry.col_seps)


@pytest.mark.parametrize("name", list(COLUMNS))
def test_zone_fits_inside_its_column(name):
    """Зона колонки вписывается между её разделителями (x) и в строку (y)."""
    geometry = _geometry()
    order = list(COLUMNS)
    index = order.index(name)
    left, right = geometry.col_seps[index], geometry.col_seps[index + 1]
    zone = geometry.columns[name].zone
    assert left <= zone.x, name
    assert zone.x + zone.width <= right, name
    assert 0 <= zone.y, name
    assert zone.y + zone.height <= geometry.row_pitch, name


# Пиксельные экстенты текста, измеренные на мете (скрины scr_153600/153611/
# 154134, полоса строки y+20..+34): имя начинается с x=164 (иконка 127..159),
# цена право-выравнена и заканчивается на x=616 (иконка свитка с 625).
# Зоны обязаны покрывать текст целиком — иначе нарезка для OCR обрезает
# символы (баги ревью 2.4: префикс «+10 » имени, последняя буква «аден»).
TEXT_EXTENTS = {"item": (164, 412), "price": (544, 616)}


@pytest.mark.parametrize("name", list(TEXT_EXTENTS))
def test_zone_covers_observed_text(name):
    geometry = _geometry()
    zone = geometry.columns[name].zone
    text_x0, text_x1 = TEXT_EXTENTS[name]
    assert zone.x <= text_x0, f"зона {name} обрезает начало текста ({zone.x} > {text_x0})"
    assert zone.x + zone.width > text_x1, (
        f"зона {name} обрезает конец текста ({zone.x + zone.width} <= {text_x1})"
    )
