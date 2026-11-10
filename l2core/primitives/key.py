"""Примитив нажатия клавиши (5.9, menu-quickslot-navigation): key_press с
verify и трёхзначным исходом (spec action-primitives; инварианты 1, 2, 4).

press_key — оркестратор, внешний мир — инъекции (HID, экран, activate,
оценка проверок, rng, часы):

    пролог активации -> цикл попыток (retry из конфига) -> on_fail[]

Пролог — только активация окна игры (5.2): в отличие от click_element 5.3,
здесь нет пролога статуса окна/якоря — клавиша X открывает «Меню» при
закрытом окне торговли, якоря на кадре нет (тот же аргумент, что у
открытия через HUD в l2core/api/core.py). Verify оценивается инъекцией
evaluate(checks, frame) -> CheckReport БЕЗ якоря окна: наблюдаемые
признаки навигации (меню, HUD-иконки quick-slot) — объекты клиентской
области игры, а не окна торговли (оценщик — l2core.detector.navchecks).

ЦИКЛ ПОПЫТОК (попыток = 1 + timing.click_retry_count): нажатие ->
блокирующий wait_until по свежим кадрам (verify на первом же кадре,
где признаки сошлись; инвариант 1 — никаких sleep). Исходы:
нажатие ушло, verify не сошёлся за таймаут -> следующая попытка
(эффект отсутствует, повтор безопасен); HID-сбой -> KeyResult UNKNOWN
(unconfirmed=True, нажатие могло уйти, могло нет — инвариант 4),
прерыв; отмена (kill switch 8.1) -> cancelled. Исчерпание -> повторный
activate -> on_fail[] (общий словарь шагов с click_element 5.3) ->
KeyResult FAILED, сигнал needs_recover (исполнение recover — вызывающий
слой, 5.8/6.4).
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from l2core.adapters.hid import HidError
from l2core.adapters.keycodes import KEYCODES
from l2core.adapters.screen import Screen
from l2core.config import CoreConfig
from l2core.model.types import Check
from l2core.primitives.activate import ActivateResult
from l2core.primitives.click import CheckReport, VerifySpec, _run_on_fail
from l2core.primitives.wait import wait_until

__all__ = [
    "EvaluateChecks",
    "KeyResult",
    "PressKeyError",
    "press_key",
]

# Оценка проверок по кадру без якоря окна (навигационные признаки — HUD).
EvaluateChecks = Callable[[Tuple[Check, ...], np.ndarray], CheckReport]


@dataclass(frozen=True)
class KeyResult:
    """Итог нажатия клавиши (успех/отмена/сигнал recover); трёхзначная
    обёртка OK/FAILED/UNKNOWN — адаптер outcome.py (5.7) поверх полей
    success/unconfirmed (как у ClickResult)."""

    key_id: str
    success: bool
    attempts: int
    cancelled: bool = False
    unconfirmed: bool = False
    on_fail_executed: bool = False
    failures: Tuple[str, ...] = ()

    @property
    def needs_recover(self) -> bool:
        """Сигнал общего recover (5.8): неуспех без отмены."""
        return not self.success and not self.cancelled


class PressKeyError(Exception):
    """Ошибка исполнения примитива (неизвестная клавиша), не эскалация."""

    def __init__(self, detail: str):
        super().__init__(f"press_key: {detail}")
        self.detail = detail


def press_key(
    key_id: str,
    *,
    verify: VerifySpec,
    evaluate: EvaluateChecks,
    screen: Screen,
    hid,
    cursor_pos: Callable[[], Tuple[int, int]],
    activate: Callable[[], ActivateResult],
    config: CoreConfig,
    on_fail: Tuple[str, ...] = (),
    free_area_point: Optional[Tuple[int, int]] = None,
    cancelled: Optional[Callable[[], bool]] = None,
    rng: Optional[object] = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> KeyResult:
    """Нажать клавишу и подтвердить результат verify (контракт — в
    docstring модуля). verify обязателен: нажатия без verify-спецификации
    запрещены (инвариант 2). Клавиша разрешается по словарю KEYCODES;
    неизвестное имя — PressKeyError (fail fast)."""
    code = KEYCODES.get(key_id)
    if code is None:
        raise PressKeyError(
            f"неизвестная клавиша {key_id!r} (допустимы {sorted(KEYCODES)})")
    rng = rng or random.Random()
    failures: list = []

    activated = activate()
    if activated.cancelled:
        return KeyResult(key_id, success=False, attempts=0, cancelled=True)

    max_attempts = 1 + config.timing.click_retry_count
    for attempt in range(1, max_attempts + 1):
        if cancelled is not None and cancelled():
            return KeyResult(key_id, success=False, attempts=attempt - 1,
                             cancelled=True)
        try:
            hid.key_press(code)
        except HidError as exc:
            # Транспортный сбой: нажатие могло уйти, могло нет —
            # подтвердить невозможно -> UNKNOWN (инвариант 4).
            return KeyResult(
                key_id, success=False, attempts=attempt, unconfirmed=True,
                failures=(f"HID-сбой при нажатии {key_id!r}: {exc}",))

        def _settled() -> bool:
            frame = np.asarray(screen.grab())
            return evaluate(verify.checks, frame).passed

        waited = wait_until(
            _settled,
            timeout=config.timing.wait_default_timeout,
            interval=config.timing.verify_interval,
            cancelled=cancelled, clock=clock, sleep=sleep)
        if waited.cancelled:
            # Нажатие, скорее всего, ушло; подтвердить нельзя.
            return KeyResult(key_id, success=False, attempts=attempt,
                             cancelled=True, unconfirmed=True)
        if waited.success:
            return KeyResult(key_id, success=True, attempts=attempt)
        failures.append(
            f"попытка {attempt}: verify не подтверждён за "
            f"{config.timing.wait_default_timeout} с")

    # Исчерпание retry: повторный пролог перед on_fail (как в 5.3).
    reactivated = activate()
    if reactivated.cancelled:
        return KeyResult(key_id, success=False, attempts=max_attempts,
                         cancelled=True)
    _run_on_fail(on_fail, hid=hid, cursor_pos=cursor_pos,
                 free_area_point=free_area_point, config=config, rng=rng)
    return KeyResult(key_id, success=False, attempts=max_attempts,
                     on_fail_executed=bool(on_fail), failures=tuple(failures))
