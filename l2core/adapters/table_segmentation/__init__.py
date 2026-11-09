"""Сегментация таблицы «Обзор лавок» (задача 4.3).

Вендор — байт-идентичный блок rows_and_columns / extract_cells / trim_to_text
из L2-CRNN-Dataset-Builder/build_dataset.py (коммит 606a65e, см. шапку
build_dataset.py); guard-тесты на эталонах оригинала —
tests/test_table_segmentation_vendor.py. Правки вендора запрещены: менять
в источнике, затем перевендоривать (эталоны перегенерирует
tools/gen_table_segmentation_fixtures.py).

Потребители ядра работают ТОЛЬКО через TableSegmentation: дефолтные константы
вендора — устаревшая калибровка датасета (ruling R5), рантайм нарезает
константами table_geometry модели (design.md решение 12). Прямые импорты
build_dataset вне l2core/adapters/ запрещены guard-тестом
tests/test_table_segmentation_contract.py.

Потребитель (read_rows) — задача 6.1, здесь его нет (YAGNI).
"""
from l2core.adapters.table_segmentation.build_dataset import Cell
from l2core.adapters.table_segmentation.segmentation import (
    HEADER_BAND,
    TableSegmentation,
)

__all__ = ["Cell", "HEADER_BAND", "TableSegmentation"]
