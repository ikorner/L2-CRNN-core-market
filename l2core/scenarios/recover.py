"""Сценарийная точка входа recover (задача 6.4, spec market-scenarios:
«Сценарий recover как композиция примитивов без внутреннего состояния»).

scenario_recover — ТОНКАЯ ОБВЯЗКА поверх общей recover-цепочки 5.8
(l2core/primitives/recover.py): механика цепочки (шаги, пролог активации,
release_all, эскалация) живёт там и здесь не дублируется. Сценарий решает
только контракт вызова для вызывающего слоя (стратегии/API 7.1), получившего
needs_recover от любого примитива 5.x или сценария 6.1–6.3:

    cause — результат примитива/сценария-причины. БЕЗ needs_recover
    (успех или отмена) — ScenarioRecoverError (fail fast): recover от
    не-причины — не ритуал, а ошибка вызывающего слоя. Отменённый cause —
    домен kill switch (8.1), стратегия обязана ветвить cancelled до вызова.
    cause протягивается в цепочку: при needs_recover recover делает
    release_all первым HID-действием (залипшая кнопка из 5.6).

    target — id окна, которое восстанавливаем (market / shop_buy /
    shop_sell); переоткрытие — рецепт навигации open_<target>
    (menu-quickslot-navigation) через инъекцию recipe_runner; runner не
    задан — шаг переоткрытия пропускается с диагностикой.

    Провайдеры (screen/hid/cursor_pos/activate/window_status/find_anchor/
    goal_reached/current_window/close_point/recipe_runner/...) — ГОТОВЫЕ
    провайдеры от вызывающего слоя (тот же паттерн, что 6.1): сценарий
    адаптеры не создаёт и wiring на детекторы 3.4/3.5 не собирает — это
    ответственность стратегии/API (7.1). БЕЗ внутреннего состояния (spec):
    сценарий не кэширует ничего, каждый вызов — независимая композиция.

РЕЗУЛЬТАТ — RecoverResult цепочки 5.8 как есть: success (восстановлен),
escalated (цепочка исчерпана/окно не открылось — событие уже отправлено
push-callback'ом on_escalation), cancelled (kill switch). Эскалация и
отмена сценарием не перехватываются и не превращаются в исключения.

ПАТТЕРН ИНТЕГРАЦИИ 6.1–6.3: сценарии фиксируют needs_recover и обрывают
работу, сами recover не вызывают. Штатный цикл стратегии:

    result = scan_market(...) / setup_sell(...) / ...
    if result.cancelled:  # kill switch — домен 8.1
        ...
    if result.needs_recover:  # примитив вернул needs_recover
        recovered = scenario_recover(result, target, ...провайдеры)
        # стратегия решает: повторить действие, эскалация, пауза...

ГРАНИЦЫ (YAGNI): никакого event dispatcher (7.3), публичного API (7.1),
kill switch (8.1) — здесь только точка входа вызывающего слоя.
"""
from __future__ import annotations

import time
from typing import Callable, Optional, Protocol, Tuple

import numpy as np

from l2core.adapters.screen import Screen
from l2core.config import CoreConfig
from l2core.detector.anchor import Anchor
from l2core.detector.status import WindowStatus
from l2core.model import ElementSpec, UiModel
from l2core.primitives.activate import ActivateResult
from l2core.primitives.click import JitterRng
from l2core.primitives.recover import RecoverResult, recover

__all__ = ["RecoverCause", "ScenarioRecoverError", "scenario_recover"]


class RecoverCause(Protocol):
    """Причина recover: результат примитива 5.x / сценария 6.x с сигналом
    needs_recover (PrimitiveResult + свойство; все ClickResult/EnterResult/
    ScrollResult/DragResult/ScanResult/SetupSellResult* им)."""

    success: bool
    cancelled: bool
    unconfirmed: bool

    @property
    def needs_recover(self) -> bool: ...


class ScenarioRecoverError(Exception):
    """Ошибка контракта вызова сценария (cause без needs_recover)."""

    def __init__(self, detail: str):
        super().__init__(f"scenario_recover: {detail}")
        self.detail = detail


def scenario_recover(
    cause: RecoverCause,
    target: str,
    *,
    recipe_runner: Optional[Callable[[str], RecipeResult]] = None,
    screen: Screen,
    hid,
    cursor_pos: Callable[[], Tuple[int, int]],
    activate: Callable[[], ActivateResult],
    window_status: Callable[[np.ndarray], WindowStatus],
    find_anchor: Callable[[np.ndarray], Optional[Anchor]],
    goal_reached: Callable[[np.ndarray], bool],
    config: CoreConfig,
    current_window: Optional[Callable[[np.ndarray], Optional[str]]] = None,
    close_point: Optional[Callable[[str, np.ndarray],
                                   Optional[Tuple[int, int]]]] = None,
    on_escalation: Optional[Callable[[RecoverResult], None]] = None,
    resume: Optional[Callable[[], bool]] = None,
    pause_timeout: Optional[float] = None,
    capture_state: Optional[Callable[[], None]] = None,
    stability_frames: int = 2,
    free_area_point: Optional[Tuple[int, int]] = None,
    cancelled: Optional[Callable[[], bool]] = None,
    rng: Optional[JitterRng] = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> RecoverResult:
    """Восстановление состояния после needs_recover: валидация cause →
    recover-цепочка 5.8 с протянутыми провайдерами. Контракт — в
    докстринге модуля. Возвращает RecoverResult как есть (success /
    escalated / cancelled)."""
    if not cause.needs_recover:
        raise ScenarioRecoverError(
            f"recover вызывается только по сигналу needs_recover: cause "
            f"success={cause.success} cancelled={cause.cancelled} — это "
            f"ошибка вызывающего слоя (отмену ветвить до вызова, kill "
            f"switch — 8.1), цепочка не запускается")
    return recover(
        target,
        recipe_runner=recipe_runner,
        screen=screen,
        hid=hid,
        cursor_pos=cursor_pos,
        activate=activate,
        window_status=window_status,
        find_anchor=find_anchor,
        goal_reached=goal_reached,
        config=config,
        current_window=current_window,
        close_point=close_point,
        on_escalation=on_escalation,
        resume=resume,
        pause_timeout=pause_timeout,
        capture_state=capture_state,
        stability_frames=stability_frames,
        cause=cause,
        free_area_point=free_area_point,
        cancelled=cancelled,
        rng=rng,
        clock=clock,
        sleep=sleep,
    )
