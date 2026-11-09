"""Экспорт констант геометрии таблицы в датасет-пайплайн (задача 2.4).

Сценарий спеки ui-model «Экспорт констант в датасет-пайплайн»: константы
table_geometry модели ядра выгружаются для L2-CRNN-Dataset-Builder копией
с явной пометкой версии модели и отчётом об отличиях от предыдущей
выгрузки. Формат выгрузки (зафиксирован в отчёте 2.4) — Python-модуль
констант: пайплайн импортирует его напрямую (build_dataset.py читает
COL_SEPS/ROW_Y0/ROW_PITCH/MAX_ROWS и зоны текста).

Детерминизм (инвариант 7): выгрузка не содержит меток времени и случайных
данных; diff-отчёт стабилен при повторном запуске.

CLI:
    python -m l2core.model.export_geometry <модель.yaml> --version <версия> \
        --output <файл.py>

Версия модели передаётся явно (модели не несут собственного поля версии);
вызывающий берёт её из git-описания ui_model/ или назначает вручную.
Рядом с выгрузкой пишется <файл.py.diff> с отчётом об отличиях.
"""

import argparse
import sys
from pathlib import Path
from typing import Union

from .loader import load_model

# Ключи выгрузки в каноническом (детерминированном) порядке.
_CONSTANT_KEYS = ("COL_SEPS", "ROW_Y0", "ROW_PITCH", "MAX_ROWS", "COLUMNS")

_HEADER = """\
# Сгенерировано автоматически: python -m l2core.model.export_geometry.
# Источник: {source}
# Окно: {window}
# Версия модели: {version}
# Не править вручную — правится модель ui_model/ и экспорт запускается заново.

WINDOW = {window!r}
MODEL_VERSION = {version!r}
COL_SEPS = {col_seps!r}
ROW_Y0 = {row_y0!r}
ROW_PITCH = {row_pitch!r}
MAX_ROWS = {max_rows!r}
COLUMNS = {columns!r}
"""


def export_geometry(
    model_path: Union[str, Path],
    output_path: Union[str, Path],
    version: str,
) -> str:
    """Выгрузить table_geometry модели в Python-модуль констант.

    Возвращает diff-отчёт об отличиях от предыдущей выгрузки (тот же
    текст пишется рядом в <output>.diff). Предыдущая выгрузка — файл
    output_path, если он существует; иначе отчёт это фиксирует.
    """
    if not version or not isinstance(version, str):
        raise ValueError("version: ожидается непустая строка (пометка версии модели)")
    model = load_model(model_path)
    if model.table_geometry is None:
        raise ValueError(
            f"{Path(model_path).name}: секция table_geometry отсутствует, "
            "выгружать нечего"
        )
    geometry = model.table_geometry

    output = Path(output_path)
    previous = _load_previous(output)

    module_text = _HEADER.format(
        source=str(model_path),
        window=model.window.id,
        version=version,
        col_seps=tuple(geometry.col_seps),
        row_y0=geometry.row_y0,
        row_pitch=geometry.row_pitch,
        max_rows=geometry.max_rows,
        columns={
            name: {"zone": tuple(zone.zone), "grammar": zone.grammar.name}
            for name, zone in geometry.columns.items()
        },
    )

    current = _constants_of(model.window.id, version, geometry)
    report = build_diff_report(previous, current)

    output.write_text(module_text, encoding="utf-8")
    Path(str(output) + ".diff").write_text(report, encoding="utf-8")
    return report


def build_diff_report(previous: Union[dict, None], current: dict) -> str:
    """Детерминированный отчёт об отличиях выгрузок (ключи по алфавиту).

    previous — None (выгрузки не было) или константы прошлой выгрузки
    (_load_previous); current — константы новой выгрузки.
    """
    if previous is None:
        return (
            f"Предыдущая выгрузка не найдена ({current['OUTPUT']}).\n"
            f"Зафиксирована первая выгрузка, версия модели: {current['MODEL_VERSION']}."
        )

    lines = [
        f"Отчёт об отличиях: {previous['MODEL_VERSION']} -> {current['MODEL_VERSION']}"
    ]
    changed = False
    for key in sorted(set(previous) | set(current)):
        if key in ("OUTPUT",):
            continue
        was = previous.get(key, "<отсутствует>")
        now = current.get(key, "<отсутствует>")
        if was != now:
            lines.append(f"{key}: было {was!r}, стало {now!r}")
            changed = True
    if not changed:
        lines.append("Отличий нет.")
    return "\n".join(lines)


def _constants_of(window_id: str, version: str, geometry) -> dict:
    """Константы выгрузки как словарь (для сравнения с предыдущей)."""
    return {
        "OUTPUT": f"{window_id}@{version}",
        "MODEL_VERSION": version,
        "COL_SEPS": tuple(geometry.col_seps),
        "ROW_Y0": geometry.row_y0,
        "ROW_PITCH": geometry.row_pitch,
        "MAX_ROWS": geometry.max_rows,
        "COLUMNS": {
            name: {"zone": tuple(zone.zone), "grammar": zone.grammar.name}
            for name, zone in geometry.columns.items()
        },
    }


def _load_previous(output: Path) -> Union[dict, None]:
    """Прочитать константы предыдущей выгрузки; None — файла нет/не читается.

    Файл — сгенерированный плоский модуль констант; читается exec'ом текста
    в пустом namespace. БЕЗ importlib: source loader кэширует pyc по
    (mtime секунды, размер) — перезаписи выгрузки с той же длиной текста
    в пределах секунды отдают stale-кэш и diff-отчёт становится недетерминирован
    (вскрыто задачей 4.3: test_diff_is_deterministic флакал при другом
    тайминге прогона). Exec по тексту детерминирован всегда.
    """
    if not output.is_file():
        return None
    namespace: dict = {}
    try:
        exec(output.read_text(encoding="utf-8"), namespace)  # noqa: S102
    except Exception:
        return None
    constants = {}
    for key in ("MODEL_VERSION",) + _CONSTANT_KEYS:
        if key not in namespace:
            return None
        constants[key] = namespace[key]
    return constants


def main(argv=None) -> int:
    """CLI: выгрузить константы; отчёт об отличиях — на stdout и в .diff."""
    parser = argparse.ArgumentParser(
        prog="python -m l2core.model.export_geometry",
        description="Экспорт table_geometry в Python-модуль для датасет-пайплайна",
    )
    parser.add_argument("model", help="YAML-файл модели окна")
    parser.add_argument(
        "--version",
        required=True,
        help="пометка версии модели (git-описание ui_model/ или ручная)",
    )
    parser.add_argument("--output", required=True, help="путь выгрузки (Python-модуль)")
    args = parser.parse_args(argv)

    try:
        report = export_geometry(args.model, args.output, args.version)
    except (ValueError, OSError) as exc:
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        return 2
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
