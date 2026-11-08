"""Примитив ожидания по предикату (задача 5.1, spec action-primitives:
«Ожидание только по предикату, без sleep», инвариант 1 AGENTS.md).

`wait_until` — единственная точка блокирующего ожидания в ядре: poll
predicate каждые `interval` до `timeout`. Прямые вызовы sleep запрещены
guard-тестом tests/test_sleep_contract.py; сам модуль спит только через
инъецируемый `sleep` (дефолт — time.sleep как значение, не вызов).

Подстановка таймингов (защита от рассинхрона дефолтов, договорённость 4.5):
явные `timeout`/`interval` побеждают; если None — значения берутся из
переданного `config` (timing.wait_default_timeout / timing.verify_interval).
Литеральных дефолтов в модуле нет — точка синхронизации одна (l2core/config.py).

Семантика (спека: «не ждёт дольше таймаута»):
- первый poll — немедленно (t=0), интервал — между poll'ами, а не один
  сон до дедлайна;
- сон перед следующим poll ограничен остатком до дедлайна: последний poll
  происходит не позже deadline (timeout=1.0, interval=0.3 → poll'ы в
  0, 0.3, 0.6, 0.9 и финальный ровно на 1.0);
- `cancelled` (точка инъекции kill switch, 8.1 — сам kill switch здесь не
  реализован) проверяется до первого poll'а и перед каждым сном (после
  неуспешного poll'а); отмена — определённый результат (флаг), не
  исключение: решение за вызывающим слоем.

Результат — WaitResult (успех/отмена/число poll'ов/elapsed/последнее
значение): диагностика для примитивов 5.x; обёртка в трёхзначный
OK/FAILED/UNKNOWN — задача 5.7, здесь не строится (YAGNI), но структура
ей не мешает.

Детерминизм (инвариант 7): часы и сон — инъекции (`clock`, `sleep`),
в проде — time.monotonic / time.sleep; тесты на фейковом времени.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from l2core.config import CoreConfig

__all__ = ["WaitResult", "wait_until"]


@dataclass(frozen=True)
class WaitResult:
    """Итог ожидания: факты для диагностики и решения вызывающего слоя.

    success — предикат выполнился; cancelled — остановлен через `cancelled`
    (success при этом False — флаги взаимоисключающие); polls — число
    вызовов predicate; elapsed — секунд по `clock` от старта до возврата;
    last_value — последнее значение predicate (диагностика при таймауте).
    """

    success: bool
    cancelled: bool
    polls: int
    elapsed: float
    last_value: Any


def _resolve(value: Optional[float], config: Optional[CoreConfig], attr: str) -> float:
    if value is not None:
        return value
    if config is None:
        raise ValueError(
            f"{attr}: не передано явно и config=None — подставьте значение "
            f"из конфига вызывающим слоем (timing.{attr})"
        )
    return getattr(config.timing, attr)


def _check_positive(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name}: ожидается число > 0 (получено {value!r})")


def wait_until(
    predicate: Callable[[], Any],
    *,
    timeout: Optional[float] = None,
    interval: Optional[float] = None,
    config: Optional[CoreConfig] = None,
    cancelled: Optional[Callable[[], bool]] = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> WaitResult:
    """Ждать выполнения predicate: poll каждые `interval` до `timeout`.

    predicate — callable без аргументов; truthy-результат = выполнено.
    timeout/interval — явно или из config.timing (см. шапку модуля).
    cancelled — опциональный callable [[], bool], проверяется до первого
    poll'а и после каждого неуспешного (перед сном); отмена возвращает
    WaitResult(cancelled=True), не исключение.
    clock/sleep — инъекции времени (тесты без реального ожидания).

    Возврат: WaitResult. Дольше timeout не ждёт: последний poll — не позже
    дедлайна (сон перед ним урезан остатком).
    """
    timeout = _resolve(timeout, config, "wait_default_timeout")
    interval = _resolve(interval, config, "verify_interval")
    _check_positive(timeout, "timeout")
    _check_positive(interval, "interval")

    start = clock()
    deadline = start + timeout
    polls = 0
    last_value: Any = None

    def _cancelled_result(now: float) -> WaitResult:
        return WaitResult(False, True, polls, now - start, last_value)

    # Отмена проверяется до первого poll'а и после каждого неуспешного
    # poll'а (перед сном): не спим, если cancelled сработал.
    if cancelled is not None and cancelled():
        return _cancelled_result(clock())
    while True:
        last_value = predicate()
        polls += 1
        if last_value:
            return WaitResult(True, False, polls, clock() - start, last_value)
        if cancelled is not None and cancelled():
            return _cancelled_result(clock())
        remaining = deadline - clock()
        # EPS: float-дрейф накопленного времени не должен порождать
        # sleep(~0) без прогресса часов; доли наносекунды физически
        # незначимы — это всё ещё «не дольше timeout». Elapsed на таймауте
        # — ровно timeout: дедлайн достигнут (в пределах EPS) по построению.
        if remaining <= 1e-9:
            return WaitResult(False, False, polls, timeout, last_value)
        # Округление убирает float-шум длительности (1.0-0.8 != 0.2);
        # точность выше наносекунд для сна не значима.
        sleep(round(min(interval, remaining), 9))
