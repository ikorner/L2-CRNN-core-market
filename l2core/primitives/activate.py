"""Примитив активации окна игры (задача 5.2, spec core-api-events: «Фокус и
активность окна перед каждым действием»; сценарий «Окно не активно»; safety:
осмысленная ошибка/эскалация при невозможности активации — не слепые клики).

Контракт вызова: примитив вызывается перед каждым действием (дисциплина
«фокус перед кликом» оформляется в задаче 5.3+, здесь — готовый примитив и
сигнал эскалации).

Проверка перекрытия — честный минимум: фокус ввода (GetForegroundWindow ==
hwnd окна игры). Надёжной дешёвой проверки перекрытия клиентской области
другим окном Win32-шлюз не даёт (нужен перебор z-order с пересечением
rect'ов, а оверлеи внутри окна игры Win32 вообще не видит). Перекрытие
покрывается статус-детектором (3.5, occluded_or_unstable — нестабильность
между кадрами) и перехватчиком модалок (3.6) — вне этого примитива.

Семантика:
- окно ищется через GameWindowFinder (критерии — config.window); окно не
  найдено/свёрнуто/без клиентской области → ActivateGameError с reason из 3.2,
  SetForeground не вызывается (инвариант 9);
- фокус уже у окна → результат с already_active=True, SetForeground не
  вызывается;
- иначе SetForeground (через шлюз), затем verify: wait_until по предикату
  «фокус получен» с таймингами из конфига (инвариант 2: verify на действие);
  verify-fail → retry, число попыток = 1 + timing.click_retry_count;
- исчерпание retry → ActivateGameError (reason=focus_not_acquired) — сигнал
  эскалации; recover-цепочка (5.8) здесь не исполняется, решение за
  вызывающим слоем;
- cancelled (kill switch 8.1) прерывает ожидание verify: результат с флагом
  cancelled=True (не исключение — консистентно с wait.py), решение за
  вызывающим слоем.

Детерминизм (инвариант 7): вся недетерминированность — за инъекциями
(шлюз, clock/sleep для wait_until); в проде — time.monotonic / time.sleep.

Retry-тайминг: таймаут verify каждой попытки — timing.wait_default_timeout,
интервал — timing.verify_interval (точка синхронизации дефолтов одна,
l2core/config.py).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

from l2core.adapters.win32 import GameWindowFinder, Win32Gateway
from l2core.config import CoreConfig
from l2core.primitives.wait import wait_until

__all__ = ["ActivateGameError", "ActivateResult", "REASON_FOCUS_NOT_ACQUIRED", "activate_game"]

# Причина эскалации: SetForeground исчерпал retry, фокус не подтверждён
REASON_FOCUS_NOT_ACQUIRED = "focus_not_acquired"


@dataclass(frozen=True)
class ActivateResult:
    """Итог активации (успех или отмена; эскалация — исключением).

    hwnd — окно игры; already_active — фокус уже был, SetForeground не
    вызывался; attempts — число вызовов SetForeground (0 при already_active);
    cancelled — ожидание прервано через cancelled, фокус не подтверждён.
    """

    hwnd: int
    already_active: bool
    attempts: int
    cancelled: bool


class ActivateGameError(Exception):
    """Эскалация/отказ активации: слепые действия запрещены (инвариант 9).

    reason — focus_not_acquired либо причина отказа находилки (3.2:
    window_not_found / window_minimized / no_client_area); detail — человеку
    читаемая диагностика.
    """

    def __init__(self, reason: str, detail: str):
        super().__init__(f"activate_game: {reason}: {detail}")
        self.reason = reason
        self.detail = detail


def activate_game(
    gateway: Win32Gateway,
    *,
    config: CoreConfig,
    cancelled: Optional[Callable[[], bool]] = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> ActivateResult:
    """Активировать окно игры: фокус через SetForeground с retry + эскалацией.

    gateway — шлюз Win32 (в тестах fake); config — критерии окна
    (config.window) и тайминги (timing); cancelled — инъекция kill switch
    (8.1); clock/sleep — инъекции времени для wait_until.

    Возврат ActivateResult при успехе/отмене; ActivateGameError при отказе
    находилки или исчерпании retry (сигнал эскалации).
    """
    finder = GameWindowFinder(gateway, title=config.window.title, class_name=config.window.class_name)
    window = finder.find()
    if not window.ok:
        raise ActivateGameError(
            reason=window.reason or "unknown",
            detail="окно игры недоступно, SetForeground не выполнялся",
        )
    hwnd = window.hwnd

    def _focused() -> bool:
        return gateway.get_foreground_window() == hwnd

    if _focused():
        return ActivateResult(hwnd=hwnd, already_active=True, attempts=0, cancelled=False)

    max_attempts = 1 + config.timing.click_retry_count
    for attempt in range(1, max_attempts + 1):
        gateway.set_foreground(hwnd)
        wait = wait_until(
            _focused,
            config=config,
            cancelled=cancelled,
            clock=clock,
            sleep=sleep,
        )
        if wait.success:
            return ActivateResult(hwnd=hwnd, already_active=False, attempts=attempt, cancelled=False)
        if wait.cancelled:
            return ActivateResult(hwnd=hwnd, already_active=False, attempts=attempt, cancelled=True)

    raise ActivateGameError(
        reason=REASON_FOCUS_NOT_ACQUIRED,
        detail=(
            f"SetForeground ×{max_attempts} не дал фокуса: "
            f"foreground={gateway.get_foreground_window()!r}, hwnd={hwnd}"
        ),
    )
