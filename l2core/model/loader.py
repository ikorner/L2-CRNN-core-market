"""Парсер YAML UI-модели: load_model() / load_common() с диагностикой.

Разбирает файлы формата ui_model/FORMAT.md в объектную модель
(l2core.model.types). Проверяет только формат: структуру, типы, словари
(common) и ссылки на грамматики. Семантические инварианты (число
сигнатур на состояние, вписывание rect'ов в окно, существование
шаблонов в assets/, наличие table_geometry у окна со списком) —
валидатор, задача 2.3.

Любая ошибка формата -> ModelFormatError с именем файла, секцией
(с путём до проблемного узла) и деталью на русском.
"""

from pathlib import Path
import re
from typing import Any, Optional, Union

import yaml

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

# Модель окна: одна на окно, id окна фиксированы спекой.
WINDOW_IDS = ("market", "shop_buy", "shop_sell")

# Виды грамматик OCR.
GRAMMAR_KINDS = ("digits", "price", "text")

# Действия переходов.
ACTION_TYPES = ("click", "key")

# Виды проверок (сигнатур и verify): иерархия pixel_probe -> template -> ocr.
CHECK_KINDS = ("pixel_probe", "template", "ocr")


class ModelFormatError(Exception):
    """Некорректный файл UI-модели: формат, типы или словари."""

    def __init__(self, file: Union[str, Path], section: str, detail: str):
        self.file = str(file)
        self.section = section
        self.detail = detail
        super().__init__(f"{Path(self.file).name}, секция «{section}»: {detail}")


# --- Общий словарь (common.yaml) ---

_DEFAULT_VARIANTS = (
    "normal",
    "hover",
    "pressed",
    "disabled",
    "active",
    "placeholder",
    "focused",
    "filled",
)

_DEFAULT_ROLES = (
    "click_target",
    "ocr_region",
    "state_signature",
    "icon_sample",
)


def _default_grammars() -> dict:
    return {
        "digits": GrammarSpec(name="digits", kind="digits"),
        "price": GrammarSpec(
            name="price",
            kind="price",
            separators=frozenset({",", "."}),
            suffixes=(
                ("к", 1000),
                ("кк", 1000000),
                ("тыс", 1000),
                ("млн", 1000000),
                ("млрд", 1000000000),
            ),
            units=("аден",),
        ),
        "text": GrammarSpec(name="text", kind="text"),
    }


def _builtin_common() -> CommonDicts:
    """Встроенный словарь, совпадающий с поставляемым ui_model/common.yaml."""
    return CommonDicts(
        variants=frozenset(_DEFAULT_VARIANTS),
        roles=frozenset(_DEFAULT_ROLES),
        grammars=_default_grammars(),
    )


def load_common(path: Union[str, Path]) -> CommonDicts:
    """Загрузить общий словарь (variants, roles, grammars)."""
    file = Path(path)
    data = _load_yaml(file)

    allowed = {"variants", "roles", "grammars"}
    _reject_unknown(file, "common", data, allowed)

    variants = _string_list(file, "variants", data.get("variants"))
    roles = _string_list(file, "roles", data.get("roles"))
    grammars_raw = data.get("grammars", {})
    if not isinstance(grammars_raw, dict):
        raise ModelFormatError(file, "grammars", "ожидается mapping имя: спека")

    grammars = {}
    for name, spec in grammars_raw.items():
        grammars[name] = _parse_grammar(file, f"grammars.{name}", name, spec)

    return CommonDicts(
        variants=frozenset(variants),
        roles=frozenset(roles),
        grammars=grammars,
    )


def _parse_grammar(file: Path, section: str, name: str, raw: Any) -> GrammarSpec:
    if not isinstance(raw, dict):
        raise ModelFormatError(file, section, "ожидается mapping ключ: значение")
    _reject_unknown(
        file, section, raw, {"kind", "separators", "suffixes", "min", "max", "units"}
    )

    kind = raw.get("kind")
    if kind not in GRAMMAR_KINDS:
        raise ModelFormatError(
            file, section, f"неизвестный вид грамматики {kind!r} (допустимы {list(GRAMMAR_KINDS)})"
        )

    separators = frozenset()
    suffixes: tuple = ()
    min_value = None
    max_value = None
    units: tuple = ()
    if kind == "price":
        if "separators" in raw:
            separators = frozenset(
                _string_list(file, f"{section}.separators", raw["separators"])
            )
        if "suffixes" in raw:
            raw_suffixes = raw["suffixes"]
            if not isinstance(raw_suffixes, dict):
                raise ModelFormatError(
                    file, f"{section}.suffixes", "ожидается mapping суффикс: множитель"
                )
            pairs = []
            for suffix, multiplier in raw_suffixes.items():
                if (
                    not isinstance(suffix, str)
                    or not suffix
                    or isinstance(multiplier, bool)
                    or not isinstance(multiplier, int)
                    or multiplier <= 0
                ):
                    raise ModelFormatError(
                        file,
                        f"{section}.suffixes",
                        f"суффикс {suffix!r}: ожидается непустая строка и "
                        f"целый положительный множитель (получено {multiplier!r})",
                    )
                pairs.append((suffix, multiplier))
            suffixes = tuple(pairs)

    if kind in ("price", "digits"):
        min_value = _parse_range_bound(file, f"{section}.min", raw.get("min"), "min")
        max_value = _parse_range_bound(file, f"{section}.max", raw.get("max"), "max")
        if min_value is not None and max_value is not None and min_value > max_value:
            raise ModelFormatError(
                file,
                section,
                f"min ({min_value}) больше max ({max_value})",
            )
    elif "min" in raw or "max" in raw:
        raise ModelFormatError(
            file, section, f"диапазон min/max неприменим к грамматике вида {kind!r}"
        )

    if kind == "price" and "units" in raw:
        units = tuple(
            _string_list(file, f"{section}.units", raw["units"])
        )
    elif "units" in raw:
        raise ModelFormatError(
            file, section, f"units неприменимы к грамматике вида {kind!r}"
        )

    return GrammarSpec(
        name=name,
        kind=kind,
        separators=separators,
        suffixes=suffixes,
        min_value=min_value,
        max_value=max_value,
        units=units,
    )


def _parse_placeholder(file: Path, section: str, raw: Any) -> Optional[str]:
    """Regex-паттерн плейсхолдера поля ввода или None (не задан).

    Должен компилироваться как regex (re.search против прочитанного
    текста, задача 4.5); некомпилируемый — ошибка формата.
    """
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw:
        raise ModelFormatError(
            file, section, "placeholder: ожидается непустая строка-регулярное выражение"
        )
    try:
        re.compile(raw)
    except re.error as exc:
        raise ModelFormatError(
            file, section, f"placeholder: некорректное regex-выражение: {exc}"
        ) from exc
    return raw


def _parse_range_bound(file: Path, section: str, raw: Any, bound: str) -> Optional[int]:
    """Целое неотрицательное значение границы диапазона или None (не задано)."""
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise ModelFormatError(
            file, section, f"{bound}: ожидается целое неотрицательное (получено {raw!r})"
        )
    return raw


# --- Модель окна ---

def load_model(
    path: Union[str, Path], common: Optional[CommonDicts] = None
) -> UiModel:
    """Загрузить модель окна.

    common — общий словарь (load_common); None — встроенный, совпадающий
    с поставляемым ui_model/common.yaml.
    """
    file = Path(path)
    if common is None:
        common = _builtin_common()
    data = _load_yaml(file)

    required = {"window", "elements", "states", "transitions", "ocr_regions"}
    missing = required - set(data)
    if missing:
        raise ModelFormatError(
            file, "корень", f"отсутствуют обязательные секции {sorted(missing)}"
        )
    _reject_unknown(
        file,
        "корень",
        data,
        required | {"table_geometry", "grid_geometry"},
    )

    window = _parse_window(file, data["window"])
    elements = _parse_elements(file, data["elements"], common)
    states = _parse_states(file, data["states"])
    transitions = _parse_transitions(file, data["transitions"])
    ocr_regions = _parse_ocr_regions(file, data["ocr_regions"], common)
    table_geometry = None
    if "table_geometry" in data:
        table_geometry = _parse_table_geometry(file, data["table_geometry"], common)
    grid_geometry = None
    if "grid_geometry" in data:
        grid_geometry = _parse_grid_geometry(file, data["grid_geometry"])

    return UiModel(
        path=str(file),
        window=window,
        elements=elements,
        states=states,
        transitions=transitions,
        ocr_regions=ocr_regions,
        table_geometry=table_geometry,
        grid_geometry=grid_geometry,
    )


def _parse_window(file: Path, raw: Any) -> WindowSpec:
    section = "window"
    if not isinstance(raw, dict):
        raise ModelFormatError(file, section, "ожидается mapping ключ: значение")
    _reject_unknown(file, section, raw, {"id", "title", "size"})

    window_id = raw.get("id")
    if window_id not in WINDOW_IDS:
        raise ModelFormatError(
            file, section, f"неизвестное окно {window_id!r} (допустимы {list(WINDOW_IDS)})"
        )
    title = raw.get("title")
    if not isinstance(title, str) or not title.strip():
        raise ModelFormatError(file, section, "title: ожидается непустая строка")
    size = _parse_size(file, f"{section}.size", raw.get("size"))
    return WindowSpec(id=window_id, title=title, size=size)


def _parse_elements(file: Path, raw: Any, common: CommonDicts) -> dict:
    section = "elements"
    if not isinstance(raw, dict) or not raw:
        raise ModelFormatError(file, section, "ожидается непустой mapping id: элемент")

    elements = {}
    for element_id, spec in raw.items():
        sec = f"{section}.{element_id}"
        if not isinstance(spec, dict):
            raise ModelFormatError(file, sec, "ожидается mapping ключ: значение")
        _reject_unknown(file, sec, spec, {"rect", "role", "hud", "search_area", "variants"})

        role = spec.get("role")
        if role not in common.roles:
            raise ModelFormatError(
                file, sec, f"неизвестная роль {role!r} (допустимы {sorted(common.roles)})"
            )

        hud = spec.get("hud", False)
        if not isinstance(hud, bool):
            raise ModelFormatError(file, sec, "hud: ожидается true/false")

        rect = spec.get("rect")
        search_area = spec.get("search_area")
        if hud:
            if search_area is None:
                raise ModelFormatError(
                    file, sec, "HUD-элемент обязан задавать область поиска search_area"
                )
            if rect is not None:
                raise ModelFormatError(
                    file, sec, "HUD-элемент не задаёт rect, только search_area"
                )
            search_area = _parse_rect(file, f"{sec}.search_area", search_area)
        else:
            if rect is None:
                raise ModelFormatError(file, sec, "отсутствует обязательный rect")
            if search_area is not None:
                raise ModelFormatError(
                    file,
                    sec,
                    "search_area допустимо только у HUD-элементов (hud: true)",
                )
            rect = _parse_rect(file, f"{sec}.rect", rect)

        elements[element_id] = ElementSpec(
            id=element_id,
            role=role,
            rect=rect,
            hud=hud,
            search_area=search_area,
            variants=_parse_variants(file, sec, spec.get("variants"), common),
        )
    return elements


def _parse_variants(
    file: Path, section: str, raw: Any, common: CommonDicts
) -> dict:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ModelFormatError(file, f"{section}.variants", "ожидается mapping вариант: спека")

    variants = {}
    for name, spec in raw.items():
        sec = f"{section}.variants.{name}"
        if name not in common.variants:
            raise ModelFormatError(
                file,
                sec,
                f"неизвестный вариант {name!r} (допустимы {sorted(common.variants)})",
            )
        if not isinstance(spec, dict):
            raise ModelFormatError(file, sec, "ожидается mapping ключ: значение")
        _reject_unknown(file, sec, spec, {"template", "click_point"})

        template = spec.get("template")
        if not isinstance(template, str) or not template.strip():
            raise ModelFormatError(
                file, sec, "template: ожидается непустая строка (путь относительно assets/)"
            )
        click_point = None
        if "click_point" in spec:
            click_point = _parse_point(file, f"{sec}.click_point", spec["click_point"])

        variants[name] = VariantSpec(name=name, template=template, click_point=click_point)
    return variants


def _parse_states(file: Path, raw: Any) -> dict:
    section = "states"
    if not isinstance(raw, dict):
        raise ModelFormatError(file, section, "ожидается mapping id: состояние")
    if not raw:
        # Пусто допустимо: начальный импорт конвертером (2.2) до разметки.
        return {}

    states = {}
    for state_id, spec in raw.items():
        sec = f"{section}.{state_id}"
        if not isinstance(spec, dict):
            raise ModelFormatError(file, sec, "ожидается mapping ключ: значение")
        _reject_unknown(file, sec, spec, {"params", "signatures"})

        params = spec.get("params", {})
        if not isinstance(params, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in params.items()
        ):
            raise ModelFormatError(
                file, f"{sec}.params", "ожидается mapping строка: строка"
            )

        signatures = spec.get("signatures")
        if not isinstance(signatures, list) or not signatures:
            raise ModelFormatError(
                file, f"{sec}.signatures", "ожидается непустой список признаков-сигнатур"
            )
        checks = tuple(
            _parse_check(file, f"{sec}.signatures[{i}]", item)
            for i, item in enumerate(signatures)
        )
        states[state_id] = StateSpec(id=state_id, params=dict(params), signatures=checks)
    return states


def _parse_transitions(file: Path, raw: Any) -> tuple:
    section = "transitions"
    if not isinstance(raw, list):
        raise ModelFormatError(file, section, "ожидается список переходов")

    transitions = []
    for index, spec in enumerate(raw):
        sec = f"{section}[{index}]"
        if not isinstance(spec, dict):
            raise ModelFormatError(file, sec, "ожидается mapping ключ: значение")
        _reject_unknown(
            file,
            sec,
            spec,
            {"from", "action", "element", "key", "expected", "verify", "timeout", "on_fail"},
        )

        source = spec.get("from")
        if not isinstance(source, str) or not source:
            raise ModelFormatError(file, sec, "from: ожидается имя состояния")

        action_type = spec.get("action")
        if action_type not in ACTION_TYPES:
            raise ModelFormatError(
                file, sec, f"action: допустимы {list(ACTION_TYPES)} (получено {action_type!r})"
            )
        if action_type == "click":
            element = spec.get("element")
            if not isinstance(element, str) or not element:
                raise ModelFormatError(file, sec, "click требует непустой element")
            if "key" in spec:
                raise ModelFormatError(file, sec, "key недопустим при action: click")
            action = ActionSpec(type="click", element=element)
        else:
            key = spec.get("key")
            if not isinstance(key, str) or not key:
                raise ModelFormatError(file, sec, "key-действие требует непустой key")
            if "element" in spec:
                raise ModelFormatError(file, sec, "element недопустим при action: key")
            action = ActionSpec(type="key", key=key)

        expected = spec.get("expected")
        if not isinstance(expected, str) or not expected:
            raise ModelFormatError(file, sec, "expected: ожидается имя состояния")

        verify = spec.get("verify")
        if not isinstance(verify, list) or not verify:
            raise ModelFormatError(file, sec, "verify: ожидается непустой список проверок")
        checks = tuple(
            _parse_check(file, f"{sec}.verify[{i}]", item) for i, item in enumerate(verify)
        )

        timeout = spec.get("timeout")
        if timeout is not None and (
            isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0
        ):
            raise ModelFormatError(
                file, sec, f"timeout: ожидается положительное число (получено {timeout!r})"
            )

        on_fail = spec.get("on_fail")
        if not isinstance(on_fail, list) or not on_fail or not all(
            isinstance(step, str) and step for step in on_fail
        ):
            raise ModelFormatError(
                file, sec, "on_fail: ожидается непустой список имён шагов recover"
            )

        transitions.append(
            TransitionSpec(
                source=source,
                action=action,
                expected=expected,
                verify=checks,
                timeout=float(timeout) if timeout is not None else None,
                on_fail=tuple(on_fail),
            )
        )
    return tuple(transitions)


def _parse_ocr_regions(file: Path, raw: Any, common: CommonDicts) -> dict:
    section = "ocr_regions"
    if not isinstance(raw, dict):
        raise ModelFormatError(file, section, "ожидается mapping id: регион")
    if not raw:
        # Пусто допустимо: начальный импорт конвертером (2.2) до разметки.
        return {}

    regions = {}
    for region_id, spec in raw.items():
        sec = f"{section}.{region_id}"
        if not isinstance(spec, dict):
            raise ModelFormatError(file, sec, "ожидается mapping ключ: значение")
        _reject_unknown(file, sec, spec, {"rect", "grammar", "placeholder"})

        rect = _parse_rect(file, f"{sec}.rect", spec.get("rect"))
        regions[region_id] = OcrRegionSpec(
            id=region_id,
            rect=rect,
            grammar=_resolve_grammar(file, sec, spec.get("grammar"), common),
            placeholder=_parse_placeholder(file, f"{sec}.placeholder", spec.get("placeholder")),
        )
    return regions


def _parse_table_geometry(file: Path, raw: Any, common: CommonDicts) -> TableGeometry:
    section = "table_geometry"
    if not isinstance(raw, dict):
        raise ModelFormatError(file, section, "ожидается mapping ключ: значение")
    _reject_unknown(file, section, raw, {"col_seps", "row_y0", "row_pitch", "max_rows", "columns"})

    col_seps_raw = raw.get("col_seps")
    if (
        not isinstance(col_seps_raw, list)
        or len(col_seps_raw) < 2
        or not all(_is_int(v) for v in col_seps_raw)
    ):
        raise ModelFormatError(
            file,
            f"{section}.col_seps",
            "ожидается список из не менее двух целых x-координат",
        )
    row_y0 = raw.get("row_y0")
    if not _is_int(row_y0):
        raise ModelFormatError(file, f"{section}.row_y0", "ожидается целое число")
    row_pitch = raw.get("row_pitch")
    if not _is_int(row_pitch) or row_pitch <= 0:
        raise ModelFormatError(
            file, f"{section}.row_pitch", "ожидается целое число > 0"
        )
    max_rows = raw.get("max_rows")
    if not _is_int(max_rows) or max_rows <= 0:
        raise ModelFormatError(file, f"{section}.max_rows", "ожидается целое число > 0")

    columns_raw = raw.get("columns", {})
    if not isinstance(columns_raw, dict):
        raise ModelFormatError(file, f"{section}.columns", "ожидается mapping колонка: спека")
    columns = {}
    for column_id, spec in columns_raw.items():
        sec = f"{section}.columns.{column_id}"
        if not isinstance(spec, dict):
            raise ModelFormatError(file, sec, "ожидается mapping ключ: значение")
        _reject_unknown(file, sec, spec, {"zone", "grammar"})
        columns[column_id] = ColumnZone(
            id=column_id,
            zone=_parse_rect(file, f"{sec}.zone", spec.get("zone")),
            grammar=_resolve_grammar(file, sec, spec.get("grammar"), common),
        )

    return TableGeometry(
        col_seps=tuple(col_seps_raw),
        row_y0=row_y0,
        row_pitch=row_pitch,
        max_rows=max_rows,
        columns=columns,
    )


def _parse_grid_geometry(file: Path, raw: Any) -> GridGeometry:
    """Геометрия иконной сетки слотов (опциональная секция, 6.2).

    origin — точка [x, y] (от якоря окна), pitch — шаг [dx, dy] (оба > 0),
    cell — размер ячейки [w, h] (положительный), cols/rows — целые > 0.
    """
    section = "grid_geometry"
    if not isinstance(raw, dict):
        raise ModelFormatError(file, section, "ожидается mapping ключ: значение")
    _reject_unknown(file, section, raw, {"origin", "pitch", "cell", "cols", "rows"})

    origin = _parse_point(file, f"{section}.origin", raw.get("origin"))
    pitch = _parse_point(file, f"{section}.pitch", raw.get("pitch"))
    if pitch[0] <= 0 or pitch[1] <= 0:
        raise ModelFormatError(
            file, f"{section}.pitch",
            f"ожидаются положительные шаги [dx, dy] (получено {list(pitch)})",
        )
    cell = _parse_size(file, f"{section}.cell", raw.get("cell"))
    cols = raw.get("cols")
    rows = raw.get("rows")
    for name, value in (("cols", cols), ("rows", rows)):
        if not _is_int(value) or value <= 0:
            raise ModelFormatError(
                file, f"{section}.{name}",
                f"ожидается целое число > 0 (получено {value!r})",
            )
    return GridGeometry(origin=origin, pitch=pitch, cell=cell,
                        cols=cols, rows=rows)


# --- Проверки (сигнатуры и verify) ---

def _parse_check(file: Path, section: str, raw: Any) -> Check:
    if not isinstance(raw, dict):
        raise ModelFormatError(file, section, "ожидается mapping ключ: значение")
    kind = raw.get("kind")
    if kind not in CHECK_KINDS:
        raise ModelFormatError(
            file,
            section,
            f"неизвестный вид проверки {kind!r} (допустимы {list(CHECK_KINDS)})",
        )

    if kind == "pixel_probe":
        _reject_unknown(file, section, raw, {"kind", "points"})
        points = raw.get("points")
        if not isinstance(points, list) or not points:
            raise ModelFormatError(
                file, section, "pixel_probe требует непустой список points"
            )
        probes = []
        for i, point in enumerate(points):
            psec = f"{section}.points[{i}]"
            if not isinstance(point, dict):
                raise ModelFormatError(file, psec, "ожидается mapping at/color")
            _reject_unknown(file, psec, point, {"at", "color"})
            at = _parse_point(file, f"{psec}.at", point.get("at"))
            color = point.get("color")
            if (
                not isinstance(color, list)
                or len(color) != 3
                or not all(_is_int(c) and 0 <= c <= 255 for c in color)
            ):
                raise ModelFormatError(
                    file, psec, f"color: ожидается [r, g, b] в 0..255 (получено {color!r})"
                )
            probes.append(ProbePoint(at=at, color=tuple(color)))
        return Check(kind=kind, points=tuple(probes))

    if kind == "template":
        _reject_unknown(file, section, raw, {"kind", "element", "variant", "expect"})
        element = raw.get("element")
        if not isinstance(element, str) or not element:
            raise ModelFormatError(file, section, "template требует непустой element")
        # Опциональное отрицание (навигационная модель): элемент обязан
        # отсутствовать на кадре. Другие значения недопустимы.
        from .types import EXPECT_ABSENT
        expect = raw.get("expect")
        if expect is not None and expect != EXPECT_ABSENT:
            raise ModelFormatError(
                file, section,
                f"template: допустимое значение expect — {EXPECT_ABSENT!r} "
                f"(получено {expect!r})")
        variant = raw.get("variant")
        # Variant обязателен для проверки присутствия; для отрицания
        # (expect: absent) допустимо опустить — оценщик возьмёт первый
        # вариант элемента.
        if (not isinstance(variant, str) or not variant) and expect is None:
            raise ModelFormatError(file, section, "template требует непустой variant")
        return Check(kind=kind, element=element, variant=variant, expect=expect)

    # ocr
    _reject_unknown(file, section, raw, {"kind", "region", "expect"})
    region = raw.get("region")
    if not isinstance(region, str) or not region:
        raise ModelFormatError(file, section, "ocr требует непустой region")
    expect = raw.get("expect")
    if not isinstance(expect, str) or not expect:
        raise ModelFormatError(file, section, "ocr требует непустой expect")
    return Check(kind=kind, region=region, expect=expect)


# --- Базовые разборы и помощники ---

def _load_yaml(file: Path) -> dict:
    """Прочитать YAML; битый файл -> ModelFormatError с именем файла."""
    try:
        with open(file, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ModelFormatError(file, "корень", f"битый YAML: {exc}") from exc

    if not isinstance(data, dict):
        raise ModelFormatError(file, "корень", "ожидается mapping секций")
    return data


def _reject_unknown(file: Path, section: str, raw: dict, allowed: set) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise ModelFormatError(
            file, section, f"неизвестные ключи {sorted(unknown)}"
        )


def _is_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int)


def _parse_rect(file: Path, section: str, raw: Any) -> Rect:
    if (
        not isinstance(raw, list)
        or len(raw) != 4
        or not all(_is_int(v) for v in raw)
    ):
        raise ModelFormatError(
            file,
            section,
            f"rect: ожидается [x, y, w, h] из целых чисел (получено {raw!r})",
        )
    x, y, width, height = raw
    if width <= 0 or height <= 0:
        raise ModelFormatError(
            file,
            section,
            f"rect: размеры должны быть положительными (получено {raw!r})",
        )
    if x < 0 or y < 0:
        raise ModelFormatError(
            file,
            section,
            f"rect: координаты не должны быть отрицательными (получено {raw!r})",
        )
    return Rect(x=x, y=y, width=width, height=height)


def _parse_point(file: Path, section: str, raw: Any) -> tuple:
    if not isinstance(raw, list) or len(raw) != 2 or not all(_is_int(v) for v in raw):
        raise ModelFormatError(
            file, section, f"ожидается [x, y] из целых чисел (получено {raw!r})"
        )
    if raw[0] < 0 or raw[1] < 0:
        raise ModelFormatError(
            file, section, f"координаты не должны быть отрицательными (получено {raw!r})"
        )
    return (raw[0], raw[1])


def _parse_size(file: Path, section: str, raw: Any) -> tuple:
    if not isinstance(raw, list) or len(raw) != 2 or not all(_is_int(v) for v in raw):
        raise ModelFormatError(
            file, section, f"size: ожидается [w, h] из целых чисел (получено {raw!r})"
        )
    if raw[0] <= 0 or raw[1] <= 0:
        raise ModelFormatError(
            file, section, f"size: ожидаются положительные размеры (получено {raw!r})"
        )
    return (raw[0], raw[1])


def _string_list(file: Path, section: str, raw: Any) -> list:
    if not isinstance(raw, list) or not raw:
        raise ModelFormatError(file, section, "ожидается непустой список строк")
    if not all(isinstance(item, str) and item for item in raw):
        raise ModelFormatError(file, section, "ожидается список непустых строк")
    if len(set(raw)) != len(raw):
        raise ModelFormatError(file, section, "повторы в списке недопустимы")
    return list(raw)


def _resolve_grammar(
    file: Path, section: str, raw: Any, common: CommonDicts
) -> GrammarSpec:
    if not isinstance(raw, str) or not raw:
        raise ModelFormatError(file, section, "grammar: ожидается имя грамматики")
    grammar = common.grammars.get(raw)
    if grammar is None:
        raise ModelFormatError(
            file,
            section,
            f"неизвестная грамматика {raw!r} (известны {sorted(common.grammars)})",
        )
    return grammar
