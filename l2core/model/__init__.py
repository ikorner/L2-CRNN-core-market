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
from .navigation import (
    EXPECT_ABSENT,
    NavigationModel,
    Recipe,
    RecipeStep,
    load_navigation,
)
from .grammars import (
    DEFAULT_DIGITS_MAX,
    DEFAULT_DIGITS_MIN,
    DEFAULT_PRICE_MAX,
    DEFAULT_PRICE_MIN,
    GrammarError,
    GrammarMismatch,
    RegionReadResult,
    ValueOutOfRange,
    expand_rect,
    parse_value,
    read_region,
)
from .validator import (
    Gap,
    GapCheckResult,
    KNOWN_ON_FAIL_STEPS,
    Violation,
    check_against_gaps,
    format_report,
    load_gaps,
    validate_model,
    validate_navigation,
)
from .reading import (
    OcrReadError,
    ReadAttempt,
    StableReadResult,
    read_confident,
    read_stable,
)
from .types import (
    ActionSpec,
    Check,
    ColumnZone,
    CommonDicts,
    ElementSpec,
    GrammarSpec,
    GridGeometry,
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
    "DEFAULT_DIGITS_MAX",
    "DEFAULT_DIGITS_MIN",
    "DEFAULT_PRICE_MAX",
    "DEFAULT_PRICE_MIN",
    "ElementSpec",
    "EXPECT_ABSENT",
    "Gap",
    "GapCheckResult",
    "GrammarError",
    "GrammarMismatch",
    "GrammarSpec",
    "GridGeometry",
    "KNOWN_ON_FAIL_STEPS",
    "ModelFormatError",
    "OcrReadError",
    "NavigationModel",
    "OcrRegionSpec",
    "ProbePoint",
    "ReadAttempt",
    "Recipe",
    "RecipeStep",
    "Rect",
    "RegionReadResult",
    "StateSpec",
    "StableReadResult",
    "TableGeometry",
    "TransitionSpec",
    "UiModel",
    "ValueOutOfRange",
    "VariantSpec",
    "Violation",
    "WindowSpec",
    "check_against_gaps",
    "expand_rect",
    "format_report",
    "load_common",
    "load_gaps",
    "load_model",
    "load_navigation",
    "load_navigation",    "parse_value",
    "read_confident",
    "read_region",
    "read_stable",
    "validate_model",
    "validate_navigation",
]
