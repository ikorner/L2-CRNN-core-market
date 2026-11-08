"""Тесты экспорта констант геометрии в датасет-пайплайн (задача 2.4).

Сценарий спеки ui-model «Экспорт констант в датасет-пайплайн»: выгрузка
содержит пометку версии модели и отчёт об отличиях от предыдущей выгрузки.
Формат выгрузки — Python-модуль констант (зафиксирован в отчёте 2.4):
L2-CRNN-Dataset-Builder импортирует его напрямую. Детерминизм: выгрузка не
содержит меток времени, diff-отчёт стабилен.
"""

from pathlib import Path

import pytest

from l2core.model.export_geometry import export_geometry

REPO_ROOT = Path(__file__).resolve().parent.parent
MARKET_YAML = REPO_ROOT / "ui_model" / "market.yaml"


def test_export_writes_python_module_with_version_mark(tmp_path):
    output = tmp_path / "table_geometry.py"
    report = export_geometry(MARKET_YAML, output, version="test-1")

    text = output.read_text(encoding="utf-8")
    # Пометка версии модели и источника — обязательны (спека ui-model).
    assert f"MODEL_VERSION = {'test-1'!r}" in text
    assert "market" in text
    # Константы, совпадающие с golden-тестами table_geometry.
    assert "COL_SEPS = (117, 417, 648, 778, 967)" in text
    assert "ROW_Y0 = 263" in text
    assert "ROW_PITCH = 56" in text
    assert "MAX_ROWS = 8" in text
    assert "'price'" in text
    # Модуль исполняем: константы читаются импортом.
    namespace = {}
    exec(text, namespace)
    assert namespace["COL_SEPS"] == (117, 417, 648, 778, 967)
    assert namespace["COLUMNS"]["price"]["grammar"] == "price"
    # Первой выгрузке не с чем сравнивать — отчёт это фиксирует.
    assert "Предыдущая выгрузка" in report


def test_reexport_without_changes_reports_no_diff(tmp_path):
    output = tmp_path / "table_geometry.py"
    export_geometry(MARKET_YAML, output, version="test-1")
    report = export_geometry(MARKET_YAML, output, version="test-1")

    assert "Отличий нет" in report
    assert "было" not in report


def test_diff_reports_changed_constants(tmp_path):
    output = tmp_path / "table_geometry.py"
    export_geometry(MARKET_YAML, output, version="test-1")

    # Меняем калибровку: модель с другим row_y0.
    model = tmp_path / "market_shifted.yaml"
    text = MARKET_YAML.read_text(encoding="utf-8")
    shifted = text.replace("row_y0: 263", "row_y0: 265", 1)
    assert shifted != text, "ожидаемое значение row_y0 не найдено в модели"
    model.write_text(shifted, encoding="utf-8")

    report = export_geometry(model, output, version="test-2")
    assert "ROW_Y0" in report
    assert "263" in report and "265" in report
    assert "test-1" in report and "test-2" in report
    # diff-отчёт детерминирован и дублируется в файл рядом с выгрузкой.
    diff_file = tmp_path / "table_geometry.py.diff"
    assert diff_file.read_text(encoding="utf-8") == report


def test_diff_is_deterministic(tmp_path):
    output = tmp_path / "table_geometry.py"
    export_geometry(MARKET_YAML, output, version="v1")
    first = export_geometry(MARKET_YAML, output, version="v2")
    export_geometry(MARKET_YAML, output, version="v1")
    second = export_geometry(MARKET_YAML, output, version="v2")
    assert first == second
