"""Исполнитель рецептов навигации (menu-quickslot-navigation).

run_recipe исполняет декларативный рецепт открытия окна из
навигационной модели (l2core.model.navigation): шаги key/click с
verify/when/skip_if, on_fail-цепочки и alternate (резервный рецепт).
Весь внешний мир — инъекции (тот же паттерн, что у click_element 5.3 и
press_key 5.9): HID, экран, активация окна игры, оценка проверок
(EvaluateChecks 5.9 — l2core.detector.navchecks), hud_locator (точки
HUD-элементов), rng, часы.

УСЛОВНЫЕ ПОМЕТКИ: перед шагом на свежем кадре оцениваются skip_if
(сошлись — шаг пропускается целиком, считается успешным) и when
(задан и не сошёлся — пропуск). Это нормализация «Меню»: X — toggle,
слепое нажатие закрыло бы открытое меню (recipes лавок: when=[menu_open]
закрыть; open_market: skip_if=[menu_open] не закрывать).

ШАГ key — делегируется примитиву press_key 5.9 (retry -> on_fail ->
recover-сигнал внутри; UNKNOWN прерывает рецепт — инвариант 4).

ШАГ click — паттерн открытия через HUD (l2core/api/core.py, открытие при
закрытом окне, пролог click_element 5.3 недостижим): точку даёт
hud_locator (HUD-элементы) или якорь «Меню» + rect элемента (элементы
внутри меню 1:1 от menu_anchor, инвариант 3); иконка не локализована ->
попытка потрячена, КЛИКА НЕТ (слепых кликов нет, инвариант 9); далее
один клик -> блокирующий wait_until verify на свежих кадрах; retry из
конфига; HID-сбой -> UNKNOWN (unconfirmed), рецепт прерван; исчерпание ->
on_fail шага (после повторной активации) -> следующий шаг.

ИТОГ: все шаги успешны -> OK. Были провалы -> on_fail рецепта ->
alternate (рекурсивно; итог alternate с префиксом failures основного) ->
иначе FAILED, needs_recover (исполнение recover — вызывающий слой,
5.8/6.4). Отмена (kill switch 8.1) — cancelled, HID чист.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from l2core.adapters.hid import BTN_LEFT, HidError
from l2core.adapters.screen import Screen
from l2core.config import CoreConfig
from l2core.model.navigation import NavigationModel, Recipe
from l2core.model.types import ElementSpec
from l2core.primitives.activate import ActivateResult
from l2core.primitives.click import (
    VerifySpec,
    _move_to,
    _run_on_fail,
)
from l2core.primitives.key import EvaluateChecks, press_key
from l2core.primitives.wait import wait_until

__all__ = [
    "RecipeError",
    "RecipeResult",
    "run_recipe",
]

# Соглашение id якоря меню — как в l2core.detector.navchecks.
MENU_ANCHOR_ID = "menu_anchor"


@dataclass(frozen=True)
class RecipeResult:
    """Итог исполнения рецепта; трёхзначная обёртка OK/FAILED/UNKNOWN —
    адаптер outcome.py (5.7) поверх success/unconfirmed (как у ClickResult;
    отмена — outcome=None). attempts — сумма попыток по шагам."""

    recipe_id: str
    success: bool
    attempts: int
    cancelled: bool = False
    unconfirmed: bool = False
    on_fail_executed: bool = False
    failures: Tuple[str, ...] = ()

    @property
    def needs_recover(self) -> bool:
        """Сигнал общего recover: неуспех без отмены."""
        return not self.success and not self.cancelled


class RecipeError(Exception):
    """Ошибка исполнения рецепта (нет рецепта/элемента/якоря меню —
    ошибка модели, fail fast), не эскалация рантайма."""

    def __init__(self, detail: str):
        super().__init__(f"run_recipe: {detail}")
        self.detail = detail


def _hold_ms(config: CoreConfig, rng) -> int:
    return rng.randint(config.behavior.press_delay_min_ms,
                       config.behavior.press_delay_max_ms)


def _menu_click_point(element: ElementSpec, anchor_point: Tuple[int, int]
                      ) -> Tuple[int, int]:
    """Точка клика элемента меню: якорь + rect + click_point варианта
    (первого заданного, иначе центр rect) — как _click_point у 5.3."""
    rect = element.rect
    click_point = next(
        (v.click_point for v in element.variants.values()
         if v.click_point is not None),
        (rect.width // 2, rect.height // 2),
    )
    return (anchor_point[0] + rect.x + click_point[0],
            anchor_point[1] + rect.y + click_point[1])


def _run_click_step(step, *, nav_model: NavigationModel,
                    hud_locator, evaluate: EvaluateChecks, screen: Screen,
                    hid, cursor_pos, activate, config,
                    free_area_point, cancelled, rng, clock, sleep,
                    failures: list) -> Tuple[bool, bool, int]:
    """Шаг click: (успех, отмена, число попыток). UNKNOWN — отдельный
    прерыв всего рецепта, здесь не возвращается (сигнал через исключение
    внутреннее — нет, см. ниже: возвращаем третьим флагом)."""
    element = nav_model.elements.get(step.element)
    if element is None:
        raise RecipeError(
            f"элемент {step.element!r} отсутствует в навигационной модели")

    anchor_element = None
    if not element.hud:
        anchor_element = nav_model.elements.get(MENU_ANCHOR_ID)
        if anchor_element is None or not anchor_element.hud:
            raise RecipeError(
                f"элемент {step.element!r} — элемент меню, а HUD-якоря "
                f"{MENU_ANCHOR_ID!r} в модели нет")

    max_attempts = 1 + config.timing.click_retry_count
    spent = 0
    for attempt in range(1, max_attempts + 1):
        if cancelled is not None and cancelled():
            return False, True, spent
        spent = attempt
        frame = np.asarray(screen.grab())
        if element.hud:
            point = hud_locator(element, None, frame)
            if point is None:
                failures.append(
                    f"попытка {attempt}: HUD-элемент {element.id!r} не "
                    f"локализован в search_area — клик отменён (слепых "
                    f"кликов нет)")
                continue
        else:
            anchor_point = hud_locator(anchor_element, None, frame)
            if anchor_point is None:
                failures.append(
                    f"попытка {attempt}: якорь меню не локализован — меню "
                    f"не открыто, клик по {element.id!r} отменён")
                continue
            point = _menu_click_point(element, anchor_point)
        try:
            _move_to(hid, cursor_pos, point)
            hid.click(BTN_LEFT, 1, _hold_ms(config, rng))
        except HidError as exc:
            # Клик мог уйти, могло нет — подтвердить невозможно: UNKNOWN.
            raise _StepUnconfirmed(
                f"HID-сбой при клике по {element.id!r}: {exc}") from exc

        def _settled() -> bool:
            fresh = np.asarray(screen.grab())
            return evaluate(step.verify, fresh).passed

        waited = wait_until(
            _settled,
            timeout=config.timing.wait_default_timeout,
            interval=config.timing.verify_interval,
            cancelled=cancelled, clock=clock, sleep=sleep)
        if waited.cancelled:
            return False, True, spent
        if waited.success:
            return True, False, spent
        failures.append(
            f"попытка {attempt}: verify шага не подтверждён за "
            f"{config.timing.wait_default_timeout} с")

    # Исчерпание: повторный пролог перед on_fail, затем on_fail шага.
    reactivated = activate()
    if reactivated.cancelled:
        return False, True, spent
    _run_on_fail(step.on_fail, hid=hid, cursor_pos=cursor_pos,
                 free_area_point=free_area_point, config=config, rng=rng)
    failures.append(f"шаг click {element.id!r}: on_fail {list(step.on_fail)} "
                    f"исполнен, переход к следующему шагу")
    return False, False, spent


class _StepUnconfirmed(Exception):
    """Шаг click прерван UNKNOWN (HID-сбой): клик мог уйти."""


def run_recipe(recipe_id: str, *,
               nav_model: NavigationModel,
               evaluate: EvaluateChecks,
               hud_locator,
               screen: Screen,
               hid,
               cursor_pos: Callable[[], Tuple[int, int]],
               activate: Callable[[], ActivateResult],
               config: CoreConfig,
               free_area_point: Optional[Tuple[int, int]] = None,
               cancelled: Optional[Callable[[], bool]] = None,
               rng: Optional[object] = None,
               clock: Callable[[], float] = time.monotonic,
               sleep: Callable[[float], None] = time.sleep,
               ) -> RecipeResult:
    """Исполнить рецепт открытия окна (контракт — в docstring модуля).
    Рецепта нет — RecipeError (fail fast)."""
    recipe = nav_model.recipes.get(recipe_id)
    if recipe is None:
        raise RecipeError(
            f"рецепт {recipe_id!r} отсутствует в навигационной модели "
            f"{nav_model.path!r}")
    return _run_recipe_obj(
        recipe, nav_model=nav_model, evaluate=evaluate,
        hud_locator=hud_locator, screen=screen, hid=hid,
        cursor_pos=cursor_pos, activate=activate, config=config,
        free_area_point=free_area_point, cancelled=cancelled, rng=rng,
        clock=clock, sleep=sleep, prefix_failures=())


def _run_recipe_obj(recipe: Recipe, *, nav_model, evaluate, hud_locator,
                    screen, hid, cursor_pos, activate, config,
                    free_area_point, cancelled, rng, clock, sleep,
                    prefix_failures: Tuple[str, ...]) -> RecipeResult:
    rng = rng or random.Random()
    failures: list = list(prefix_failures)
    attempts = 0
    on_fail_executed = False
    failed = False

    for i, step in enumerate(recipe.steps):
        frame = np.asarray(screen.grab())
        if step.skip_if:
            report = evaluate(step.skip_if, frame)
            if report.passed:
                continue
        if step.when:
            report = evaluate(step.when, frame)
            if not report.passed:
                continue

        if step.action == "key":
            result = press_key(
                step.key,
                verify=VerifySpec(checks=step.verify),
                evaluate=evaluate, screen=screen, hid=hid,
                cursor_pos=cursor_pos, activate=activate, config=config,
                on_fail=step.on_fail, free_area_point=free_area_point,
                cancelled=cancelled, rng=rng, clock=clock, sleep=sleep)
            attempts += result.attempts
            if result.cancelled:
                return RecipeResult(
                    recipe.id, success=False, attempts=attempts,
                    cancelled=True, unconfirmed=result.unconfirmed,
                    failures=tuple(failures))
            if result.unconfirmed:
                return RecipeResult(
                    recipe.id, success=False, attempts=attempts,
                    unconfirmed=True,
                    failures=tuple(failures) + result.failures)
            if not result.success:
                failures.extend(result.failures)
                on_fail_executed = on_fail_executed or result.on_fail_executed
                failed = True
                continue
        else:
            try:
                ok, was_cancelled, spent = _run_click_step(
                    step, nav_model=nav_model, hud_locator=hud_locator,
                    evaluate=evaluate, screen=screen, hid=hid,
                    cursor_pos=cursor_pos, activate=activate, config=config,
                    free_area_point=free_area_point, cancelled=cancelled,
                    rng=rng, clock=clock, sleep=sleep, failures=failures)
            except _StepUnconfirmed as exc:
                return RecipeResult(
                    recipe.id, success=False, attempts=attempts + 1,
                    unconfirmed=True,
                    failures=tuple(failures) + (str(exc),))
            attempts += spent
            if was_cancelled:
                return RecipeResult(
                    recipe.id, success=False, attempts=attempts,
                    cancelled=True, failures=tuple(failures))
            if not ok:
                on_fail_executed = True
                failed = True
                continue

    # Итог: все непропущенные шаги успешны?
    if not failed:
        return RecipeResult(
            recipe.id, success=True, attempts=attempts,
            on_fail_executed=on_fail_executed, failures=tuple(failures))

    # Неуспех: on_fail рецепта -> alternate -> FAILED с сигналом recover.
    reactivated = activate()
    if reactivated.cancelled:
        return RecipeResult(
            recipe.id, success=False, attempts=attempts, cancelled=True,
            failures=tuple(failures))
    _run_on_fail(recipe.on_fail, hid=hid, cursor_pos=cursor_pos,
                 free_area_point=free_area_point, config=config, rng=rng)
    on_fail_executed = True
    if recipe.alternate is not None:
        alternate = nav_model.recipes.get(recipe.alternate)
        if alternate is None:
            raise RecipeError(
                f"alternate {recipe.alternate!r} рецепта {recipe.id!r} "
                "отсутствует в навигационной модели")
        result = _run_recipe_obj(
            alternate, nav_model=nav_model, evaluate=evaluate,
            hud_locator=hud_locator, screen=screen, hid=hid,
            cursor_pos=cursor_pos, activate=activate, config=config,
            free_area_point=free_area_point, cancelled=cancelled, rng=rng,
            clock=clock, sleep=sleep,
            prefix_failures=tuple(failures))
        return RecipeResult(
            recipe.id, success=result.success,
            attempts=attempts + result.attempts,
            cancelled=result.cancelled, unconfirmed=result.unconfirmed,
            on_fail_executed=on_fail_executed or result.on_fail_executed,
            failures=result.failures)
    return RecipeResult(
        recipe.id, success=False, attempts=attempts,
        on_fail_executed=on_fail_executed, failures=tuple(failures))
