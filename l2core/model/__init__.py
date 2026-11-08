"""Интерпретатор UI-модели: формат YAML и объектная модель.

UI-модель описывает окна торговли декларативно: «где и как выглядит»
(design.md, решение 1). Формат файлов задокументирован в
ui_model/FORMAT.md; поставляемые файлы — ui_model/common.yaml (общий
словарь) и ui_model/<window>.yaml (одна модель на окно).

Публичный API:
- load_common(path) -> CommonDicts — общий словарь (варианты, роли, грамматики);
- load_model(path, common=None) -> UiModel — модель окна; common=None —
  встроенный словарь, совпадающий с поставляемым common.yaml;
- ModelFormatError — ошибка формата с именем файла и секцией;
- validate_model(model, assets_dir) -> list[Violation] — семантические
  инварианты спеки ui-model (задача 2.3); CLI для людей:
  python -m l2core.model.validator <файлы...>.

Пакет отвечает за формат, объектную модель и семантическую валидацию
инвариантов: детекция состояний и исполнение переходов — слоями выше.
"""

from .loader import ModelFormatError, load_common, load_model
from .validator import (
    Gap,
    GapCheckResult,
    Violation,
    check_against_gaps,
    format_report,
    load_gaps,
    validate_model,
)
from .types import (
    ActionSpec,
    Check,
    ColumnZone,
    CommonDicts,
    ElementSpec,
    GrammarSpec,
    OcrRegionSpec,
    ProbePoint,
    Rect,
    StateSpec,
    TableGeometry,
    TransitionSpec,
    UiModel,
    VariantSpec,
    WindowSpec,
)

__all__ = [
    "ActionSpec",
    "Check",
    "ColumnZone",
    "CommonDicts",
    "ElementSpec",
    "Gap",
    "GapCheckResult",
    "GrammarSpec",
    "ModelFormatError",
    "OcrRegionSpec",
    "ProbePoint",
    "Rect",
    "StateSpec",
    "TableGeometry",
    "TransitionSpec",
    "UiModel",
    "VariantSpec",
    "Violation",
    "WindowSpec",
    "check_against_gaps",
    "format_report",
    "load_common",
    "load_gaps",
    "load_model",
    "validate_model",
]
