"""Навигационная модель «Меню/quick-slot» (menu-quickslot-navigation).

Файл `ui_model/navigation.yaml` — НЕ модель окна: секций `window`/
`states`/`transitions` в нём нет, парсер моделей окон (`load_model`) его
не принимает. Содержит:

- `elements` — элементы навигации в формате элементов окон (HUD —
  `search_area` в клиентской области игры; элементы внутри «Меню» —
  `rect` 1:1 от HUD-элемента-якоря `menu_anchor`);
- `signatures` — именованные наборы проверок (минимум два признака, как
  у состояний окон; `menu_open` — сигнатура открытого «Меню»);
- `recipes` — рецепты открытия окон: шаги `key`/`click` с обязательными
  `verify`/`on_fail`, опциональными `when`/`skip_if` (условные пометки,
  проверяемые на текущем кадре до шага) и `alternate` (резервный рецепт).

Проверки рецептов и сигнатур — `template` (с опциональным отрицанием
`expect: absent`) и `pixel_probe`; `ocr` в навигации не поддержан (меню
подтверждается визуально, без чтения текста). Элемент списка
`verify`/`when`/`skip_if` — mapping проверки или строка-ссылка на
сигнатуру (разворачивается в её проверки).

Координатных действий в рецептах нет: точки кликов происходят из
`rect`/`click_point` элементов и template-поиска по `search_area`
(инвариант 3). Разбор — здесь; семантические инварианты —
`validate_navigation` (валидатор), статический обход — `trace_navigation`
(трассировка 6.5).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

from .loader import (
    ModelFormatError,
    _load_yaml,
    _parse_check,
    _parse_elements,
    _reject_unknown,
)
from .types import Check, CommonDicts, ElementSpec, EXPECT_ABSENT

__all__ = [
    "EXPECT_ABSENT",
    "NavigationModel",
    "Recipe",
    "RecipeStep",
    "load_navigation",
]

# Значение поля expect template-проверки: элемент обязан ОТСУТСТВОВАТЬ
# на кадре (отрицание; нужно для нормализации «Меню»: X — toggle).
# (Константа живёт в types.py — парсер проверок loader.py её потребляет.)

# Действия шагов рецепта.
_STEP_ACTIONS = ("key", "click")


@dataclass(frozen=True)
class RecipeStep:
    """Шаг рецепта: нажатие клавиши или клик по элементу навигации.

    verify — обязательные наблюдаемые признаки результата шага; on_fail —
    именованные шаги исполнителя (тот же словарь, что в transitions окон);
    when — выполнять шаг только если проверки сошлись на текущем кадре;
    skip_if — пропускать шаг, если сошлись. when/skip_if пусты — шаг
    выполняется всегда.
    """

    action: str  # key | click
    element: Optional[str]  # action=click
    key: Optional[str]  # action=key (имя из KEYCODES)
    verify: Tuple[Check, ...]
    on_fail: Tuple[str, ...]
    when: Tuple[Check, ...] = ()
    skip_if: Tuple[Check, ...] = ()


@dataclass(frozen=True)
class Recipe:
    """Рецепт открытия окна: opens — целевое окно; steps — непустой список;
    on_fail — шаги при исчерпании рецепта; alternate — опциональный
    резервный рецепт (пробуется после on_fail)."""

    id: str
    opens: str
    steps: Tuple[RecipeStep, ...]
    on_fail: Tuple[str, ...]
    alternate: Optional[str] = None


@dataclass(frozen=True)
class NavigationModel:
    """Навигационная модель: результат load_navigation()."""

    path: str
    elements: dict  # id -> ElementSpec
    signatures: dict  # id -> Tuple[Check, ...]
    recipes: dict  # id -> Recipe


def load_navigation(path: Union[str, Path],
                    common: Optional[CommonDicts] = None) -> NavigationModel:
    """Загрузить навигационную модель (формат — в docstring модуля).

    common — общий словарь (load_common); None — встроенный, совпадающий
    с поставляемым ui_model/common.yaml.
    """
    file = Path(path)
    if common is None:
        from .loader import _builtin_common
        common = _builtin_common()
    data = _load_yaml(file)

    required = {"elements", "signatures", "recipes"}
    missing = required - set(data)
    if missing:
        raise ModelFormatError(
            file, "корень",
            f"отсутствуют обязательные секции {sorted(missing)}")
    _reject_unknown(file, "корень", data, required)

    elements = _parse_elements(file, data["elements"], common)
    signatures = _parse_signatures(file, data["signatures"])
    recipes = _parse_recipes(file, data["recipes"], signatures)
    return NavigationModel(
        path=str(file), elements=elements,
        signatures=signatures, recipes=recipes)


def _parse_signatures(file: Path, raw: object) -> dict:
    section = "signatures"
    if not isinstance(raw, dict) or not raw:
        raise ModelFormatError(file, section, "ожидается непустой mapping id: список проверок")
    signatures = {}
    for sig_id, items in raw.items():
        sec = f"{section}.{sig_id}"
        if not isinstance(items, list) or not items:
            raise ModelFormatError(
                file, sec, "ожидается непустой список проверок")
        signatures[sig_id] = tuple(
            _parse_check(file, f"{sec}[{i}]", item)
            for i, item in enumerate(items))
        _reject_ocr(file, sec, signatures[sig_id])
    return signatures


def _reject_ocr(file: Path, section: str, checks: Tuple[Check, ...]) -> None:
    """OCR-проверки в навигации не поддержаны: меню подтверждается
    визуально (template/pixel_probe), чтения текста в рецептах нет."""
    for check in checks:
        if check.kind == "ocr":
            raise ModelFormatError(
                file, section,
                "ocr-проверки в навигационной модели не поддержаны "
                "(допустимы template с опциональным expect: absent "
                "и pixel_probe)")


def _resolve_checks(file: Path, section: str, raw: object,
                    signatures: dict) -> Tuple[Check, ...]:
    """Список проверок: mapping -> Check, строка -> проверки сигнатуры."""
    if not isinstance(raw, list) or not raw:
        raise ModelFormatError(file, section, "ожидается непустой список проверок")
    resolved = []
    for i, item in enumerate(raw):
        if isinstance(item, str):
            checks = signatures.get(item)
            if checks is None:
                raise ModelFormatError(
                    file, f"{section}[{i}]",
                    f"неизвестная сигнатура {item!r}")
            resolved.extend(checks)
        else:
            resolved.append(_parse_check(file, f"{section}[{i}]", item))
    result = tuple(resolved)
    _reject_ocr(file, section, result)
    return result


def _parse_recipes(file: Path, raw: object,
                   signatures: dict) -> dict:
    section = "recipes"
    if not isinstance(raw, dict) or not raw:
        raise ModelFormatError(file, section, "ожидается непустой mapping id: рецепт")
    recipes = {}
    for recipe_id, spec in raw.items():
        sec = f"{section}.{recipe_id}"
        if not isinstance(spec, dict):
            raise ModelFormatError(file, sec, "ожидается mapping ключ: значение")
        _reject_unknown(
            file, sec, spec, {"opens", "steps", "on_fail", "alternate"})

        opens = spec.get("opens")
        if not isinstance(opens, str) or not opens:
            raise ModelFormatError(file, sec, "opens: ожидается непустая строка")
        steps_raw = spec.get("steps")
        if not isinstance(steps_raw, list) or not steps_raw:
            raise ModelFormatError(
                file, f"{sec}.steps", "ожидается непустой список шагов")
        steps = tuple(
            _parse_step(file, f"{sec}.steps[{i}]", item, signatures)
            for i, item in enumerate(steps_raw))
        on_fail = _parse_on_fail(file, f"{sec}.on_fail", spec.get("on_fail"))
        alternate = spec.get("alternate")
        if alternate is not None and (
                not isinstance(alternate, str) or not alternate):
            raise ModelFormatError(
                file, sec, "alternate: ожидается непустая строка-имя рецепта")
        recipes[recipe_id] = Recipe(
            id=recipe_id, opens=opens, steps=steps,
            on_fail=on_fail, alternate=alternate)
    return recipes


def _parse_step(file: Path, section: str, raw: object,
                signatures: dict) -> RecipeStep:
    if not isinstance(raw, dict):
        raise ModelFormatError(file, section, "ожидается mapping ключ: значение")
    _reject_unknown(
        file, section, raw,
        {"action", "element", "key", "verify", "on_fail", "when", "skip_if"})

    action = raw.get("action")
    if action not in _STEP_ACTIONS:
        raise ModelFormatError(
            file, section,
            f"неизвестное действие {action!r} (допустимы {list(_STEP_ACTIONS)})")
    element = raw.get("element")
    key = raw.get("key")
    if action == "key":
        if not isinstance(key, str) or not key:
            raise ModelFormatError(
                file, section, "шаг key требует непустой key")
        if element is not None:
            raise ModelFormatError(
                file, section, "шаг key не принимает element")
    else:
        if not isinstance(element, str) or not element:
            raise ModelFormatError(
                file, section, "шаг click требует непустой element")
        if key is not None:
            raise ModelFormatError(
                file, section, "шаг click не принимает key")

    verify = _resolve_checks(file, f"{section}.verify",
                             raw.get("verify"), signatures)
    on_fail = _parse_on_fail(file, f"{section}.on_fail", raw.get("on_fail"))
    when = _resolve_checks(file, f"{section}.when",
                           raw.get("when"), signatures) if "when" in raw else ()
    skip_if = _resolve_checks(file, f"{section}.skip_if",
                              raw.get("skip_if"), signatures) if "skip_if" in raw else ()
    return RecipeStep(
        action=action, element=element, key=key,
        verify=verify, on_fail=on_fail, when=when, skip_if=skip_if)


def _parse_on_fail(file: Path, section: str, raw: object) -> Tuple[str, ...]:
    """on_fail — непустой список именованных шагов исполнителя; семантику
    шагов проверяет валидатор/трассировка (паритет со transitions окон)."""
    from .loader import _string_list
    if raw is None:
        return ()
    return tuple(_string_list(file, section, raw))
