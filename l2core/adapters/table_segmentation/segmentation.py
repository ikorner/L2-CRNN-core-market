"""Wrapper сегментации таблицы: константы table_geometry модели ->
вендоренный блок (задача 4.3, design.md решение 12, spec ui-model
«Геометрия таблицы вынесена в модель и обща с пайплайном обучения»).

Вендор (build_dataset.py в этом пакете) байт-идентичен источнику и читает
геометрию из module-level globals — это его замороженный интерфейс. Единственная
адаптация — эта: TableSegmentation при создании подменяет глобалы константами
TableGeometry (объектная модель l2core.model, YAML сам не парсится). Эффект
процесс-wide (модуль-одиночка): не создавать два экземпляра с разной
геометрией одновременно; в рантайме экземпляр один на окно market.

Маппинг (модель -> вендор; см. отчёт task-4.3-report.md):

    col_seps[0]                  -> CONTENT_X0   (левый край контента)
    col_seps[1:]                 -> COL_SEPS     (линии колонок + правый край)
    col_seps[-1]                 -> CONTENT_X1   (мёртвая в теле, нужна для bounds)
    ключи columns (по порядку)   -> COLUMNS      (порядок = порядку пар col_seps)
    «item» -> «name»             -> ключ первой колонки модели: у вендора обрезка
                                     зоны текста (без иконки) привязана к имени
                                     «name», роль зоны идентична (зона item)
    row_y0 - 18 .. row_y0        -> HEADER_Y0/Y1 (полоса заголовка 18px:
                                     элементы заголовков модели занимают
                                     y 245..263 при row_y0=263, калибровка 2.4;
                                     в модели отдельной константы нет)
    row_y0, row_pitch, max_rows  -> ROW_Y0, ROW_PITCH, MAX_ROWS (1:1)
    row_pitch                    -> ROW_H        (строки без зазора: в новой
                                     калибровке нет разделительных линий между
                                     строками — профиль меты, отчёт 4.3)
    row_y0 + max_rows*pitch      -> TABLE_BOTTOM (модель нижнюю кромку не хранит;
                                     при кропе по якорю биндинг-ограничение —
                                     высота окна, это значение её не перекрывает)
    зона item (x, x+w)           -> NAME_TEXT_X  (зона текста имени, без иконки)
    зона price (x, x+w)          -> PRICE_TEXT_X (зона текста цены, без иконок)

Проверки геометрии (падаем рано, с причиной): колонок ровно len(col_seps)-1,
зоны item и price обязаны присутствовать (вендор использует их глобалы).

Координатная система: константы модели от якоря окна — caller передаёт
изображение, начинающееся в якоре (кроп окна по результату детектора якоря),
origin = абсолютный (x, y) якоря на кадре (как в источнике). Вырезание ячеек
(extract_cells) работает в координатах КАДРА: img — полный кадр (либо кроп,
начинающийся в якоре, с origin=(0, 0)); структура от rows_and_columns несёт
origin и смещает вырезание на него.
"""
from __future__ import annotations

import numpy as np

from l2core.adapters.table_segmentation import build_dataset as _vendor
from l2core.model import TableGeometry

# Высота полосы заголовка (модель отдельной константы не хранит): элементы
# заголовков колонок market (type/predmet/price/personaj/gorod) занимают
# y 245..263 при row_y0=263 — полоса 18px непосредственно над первой строкой
# (та же смежность, что в исходной калибровке датасета 242..260/261).
HEADER_BAND = 18

# Маппинг ключей колонок модели на ключи вендора: только «item» -> «name»
# (обрезка зоны текста без иконки в вендоре привязана к литералу «name»).
_MODEL_TO_VENDOR_COLUMN = {"item": "name"}


class TableSegmentation:
    """Нарезка таблицы «Обзор лавок» на константах table_geometry модели.

    Методы — делегаты к байт-идентичному вендору с подмененными константами:
    rows_and_columns(win_gray, origin), extract_cells(img, structure),
    trim_to_text(crop, pad=4). trim_to_text констант не использует.
    """

    def __init__(self, geometry: TableGeometry) -> None:
        if len(geometry.columns) != len(geometry.col_seps) - 1:
            raise ValueError(
                "table_geometry: колонок должно быть len(col_seps) - 1 "
                f"({len(geometry.col_seps) - 1}), получено {len(geometry.columns)}"
            )
        missing = [c for c in ("item", "price") if c not in geometry.columns]
        if missing:
            raise ValueError(
                "table_geometry: для нарезки нужны зоны колонок item и price "
                f"(обрезка текста без иконок), отсутствуют: {', '.join(missing)}"
            )
        self._geometry = geometry
        self._apply(geometry)

    @staticmethod
    def _apply(g: TableGeometry) -> None:
        """Подмена module-level констант вендора (процесс-wide, документировано)."""
        seps = tuple(g.col_seps)
        item = g.columns["item"].zone
        price = g.columns["price"].zone
        _vendor.CONTENT_X0 = seps[0]
        _vendor.COL_SEPS = seps[1:]
        _vendor.CONTENT_X1 = seps[-1]
        _vendor.COLUMNS = tuple(
            _MODEL_TO_VENDOR_COLUMN.get(name, name) for name in g.columns
        )
        _vendor.HEADER_Y0 = g.row_y0 - HEADER_BAND
        _vendor.HEADER_Y1 = g.row_y0
        _vendor.ROW_Y0 = g.row_y0
        _vendor.ROW_PITCH = g.row_pitch
        _vendor.ROW_H = g.row_pitch
        _vendor.TABLE_BOTTOM = g.row_y0 + g.max_rows * g.row_pitch
        _vendor.MAX_ROWS = g.max_rows
        _vendor.NAME_TEXT_X = (item.x, item.x + item.width)
        _vendor.PRICE_TEXT_X = (price.x, price.x + price.width)

    @property
    def geometry(self) -> TableGeometry:
        return self._geometry

    def rows_and_columns(self, win_gray: np.ndarray, origin=(0, 0)) -> dict:
        """Структура таблицы: header + непустые строки (как в источнике)."""
        return _vendor.rows_and_columns(win_gray, origin=origin)

    def extract_cells(self, img: np.ndarray, structure: dict) -> list:
        """Все ячейки (header + строки x колонки), обрезанные до текстов."""
        return _vendor.extract_cells(img, structure)

    def trim_to_text(self, crop: np.ndarray, pad: int = 4) -> np.ndarray:
        """Обрезка кропа до bbox текста (+pad); пустой вход — без изменений."""
        return _vendor.trim_to_text(crop, pad=pad)
