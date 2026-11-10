"""Объектная модель UI-модели: типы для окон, элементов, состояний,
переходов, OCR-регионов и геометрии таблицы.

Формат YAML задокументирован в ui_model/FORMAT.md. Это только структуры
данных; разбор — в l2core.model.loader, семантическая валидация —
задача 2.3.
"""

from dataclasses import dataclass, field
from typing import NamedTuple, Optional, Tuple


class Rect(NamedTuple):
    """Прямоугольник [x, y, w, h] в координатах своей системы отсчёта."""

    x: int
    y: int
    width: int
    height: int


class ProbePoint(NamedTuple):
    """Пиксель-проба: точка и ожидаемый цвет."""

    at: Tuple[int, int]
    color: Tuple[int, int, int]


# Значение поля expect template-проверки: элемент обязан ОТСУТСТВОВАТЬ
# на кадре (отрицание; используется навигационной моделью — X toggle).
EXPECT_ABSENT = "absent"


@dataclass(frozen=True)
class GrammarSpec:
    """Грамматика OCR-региона: вид и правила разбора.

    min_value/max_value — опциональный диапазон допустимых значений
    (видов digits и price; None — дефолт диапазона вида, применяется
    при разборе, задача 4.4). units — опциональные суффиксы-единицы
    измерения, отсекаемые перед разбором (только price).
    """

    name: str
    kind: str  # digits | price | text
    separators: frozenset = frozenset()  # разделители разрядов (price)
    suffixes: Tuple[Tuple[str, int], ...] = ()  # суффикс -> множитель (price)
    min_value: Optional[int] = None  # None — дефолт диапазона вида
    max_value: Optional[int] = None  # None — дефолт диапазона вида
    units: Tuple[str, ...] = ()  # единицы измерения, отсекаются (price)


@dataclass(frozen=True)
class VariantSpec:
    """Визуальный вариант элемента: кроп-шаблон и точка клика."""

    name: str
    template: str  # путь относительно assets/
    click_point: Optional[Tuple[int, int]] = None


@dataclass(frozen=True)
class ElementSpec:
    """Элемент окна или HUD-элемент навигации.

    Обычный элемент задан rect от якоря окна; HUD-элемент (hud=True) —
    областью поиска search_area в клиентской области игры.
    """

    id: str
    role: str
    rect: Optional[Rect] = None
    hud: bool = False
    search_area: Optional[Rect] = None
    variants: dict = field(default_factory=dict)  # имя варианта -> VariantSpec


@dataclass(frozen=True)
class Check:
    """Признак-сигнатура или проверка verify.

    kind: pixel_probe (points), template (element+variant),
    ocr (region+expect). Поля, не нужные виду, равны None.
    """

    kind: str
    points: Tuple[ProbePoint, ...] = ()
    element: Optional[str] = None
    variant: Optional[str] = None
    region: Optional[str] = None
    expect: Optional[str] = None


@dataclass(frozen=True)
class WindowSpec:
    id: str
    title: str
    size: Tuple[int, int]  # размер окна в референсе


@dataclass(frozen=True)
class StateSpec:
    """Параметризованное состояние: значения params + сигнатуры."""

    id: str
    params: dict
    signatures: Tuple[Check, ...]


@dataclass(frozen=True)
class ActionSpec:
    """Действие перехода: click по элементу или key."""

    type: str  # click | key
    element: Optional[str] = None
    key: Optional[str] = None


@dataclass(frozen=True)
class TransitionSpec:
    """Переход (state, action) -> (expected, verify, timeout, on_fail[])."""

    source: str
    action: ActionSpec
    expected: str
    verify: Tuple[Check, ...]
    timeout: Optional[float]  # None — брать из конфига при исполнении
    on_fail: Tuple[str, ...]


@dataclass(frozen=True)
class OcrRegionSpec:
    """OCR-регион: rect от якоря окна, грамматика ожидаемого значения.

    placeholder — опциональный regex-паттерн текста-плейсхолдера поля
    ввода (пустое поле); стабилизатор (задача 4.5) отличает его от
    введённого текста. None — паттерн не задан (нет данных в модели).
    """

    id: str
    rect: Rect
    grammar: GrammarSpec
    placeholder: Optional[str] = None


@dataclass(frozen=True)
class ColumnZone:
    """Зона текста колонки внутри ячейки таблицы."""

    id: str
    zone: Rect
    grammar: GrammarSpec


@dataclass(frozen=True)
class TableGeometry:
    """Константы нарезки таблицы (общие с пайплайном обучения)."""

    col_seps: Tuple[int, ...]
    row_y0: int
    row_pitch: int
    max_rows: int
    columns: dict  # имя колонки -> ColumnZone


@dataclass(frozen=True)
class GridGeometry:
    """Геометрия иконной сетки слотов окна (лавка: ячейки под предметы).

    Координаты — от якоря окна (как rect'ы элементов). Сетка параметрична:
    origin — верхний левый угол ячейки (0, 0), pitch — шаг между соседними
    ячейками по x/y, cell — размер ячейки, cols/rows — число колонок/строк.
    Нумерация слотов row-major: slot = row * cols + col. Центр ячейки
    вычисляет генератор слоя сценариев (setup_sell, 6.2) — в формате
    только константы, вычислений нет (как у table_geometry).
    """

    origin: Tuple[int, int]
    pitch: Tuple[int, int]
    cell: Tuple[int, int]
    cols: int
    rows: int

    @property
    def capacity(self) -> int:
        """Число слотов сетки."""
        return self.cols * self.rows


@dataclass(frozen=True)
class UiModel:
    """Модель одного окна: результат load_model()."""

    path: str
    window: WindowSpec
    elements: dict  # id -> ElementSpec
    states: dict  # id -> StateSpec
    transitions: Tuple[TransitionSpec, ...]
    ocr_regions: dict  # id -> OcrRegionSpec
    table_geometry: Optional[TableGeometry]
    grid_geometry: Optional[GridGeometry] = None  # иконная сетка слотов


@dataclass(frozen=True)
class CommonDicts:
    """Общий словарь: варианты, роли, грамматики OCR."""

    variants: frozenset
    roles: frozenset
    grammars: dict  # имя -> GrammarSpec
