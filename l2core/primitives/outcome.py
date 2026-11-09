"""Трёхзначный результат критичных действий OK/FAILED/UNKNOWN (задача 5.7,
spec action-primitives: «Критичные действия SHALL возвращать OK/FAILED/
UNKNOWN, где UNKNOWN = эффект не подтверждён (лаг, перекрытие)»; «ядро
MUST NOT повторять потенциально необратимые действия (начать продажу,
подтвердить) при UNKNOWN — решение передаётся стратегии»).

Слой — адаптер ПОВЕРХ результатов примитивов 5.1–5.6: сами примитивы не
переписываются (YAGNI-отложение 5.7 закрыто здесь). Общий контракт результата
примитива — PrimitiveResult (success/cancelled/unconfirmed).

ДОМЕНЫ (README-замечание 1.4: CANCELLED ≠ результат действия):

    ActionOutcome = {OK, FAILED, UNKNOWN}   — итог завершённого действия;
    отмена (cancelled)                      — домен kill switch (8.1).

CANCELLED в enum не входит: отмена — прерывание исполнения, а не его итог,
и смешивать её с трёхзначной семантикой означало бы, что каждый потребитель
обязан разбирать четыре значения вместо трёх + одного флага до адаптации.
to_outcome отменённый результат не пропускает (ValueError, fail fast):
стратегия обязана ветвить cancelled ДО адаптации. Поведение kill switch
(release_all, увод курсора, событие killed) — задача 8.1, здесь только
место в типах.

СЕМАНТИКА (честное различие FAILED vs UNKNOWN):

    OK      — success: verify подтвердил эффект.
    FAILED  — действие завершено, verify выполнялся и провалился, либо
              HID-эффект не был применён вовсе (якорь потерян до клика,
              hover re-verify не сошёлся): эффект отсутствует или опровергнут
              → recover-сигнал, повтор решения стратегии (для обратимых —
              обычно можно).
    UNKNOWN — HID-эффект, скорее всего, уже применён (клик ушёл / текст
              набран / wheel послан / drag отпущен), но подтвердить его
              невозможно: якорь потерян на свежем кадре ПОСЛЕ действия
              (окно деградировало между действием и verify), OCR-чтение
              поля невозможно после набора, HID-сбой оставил состояние
              неизвестным. Эффект мог свершиться, но не доказано.

Различие несёт поле unconfirmed результатов примитивов (5.7): False по
умолчанию, True выставляется только в точках «эффект ушёл, подтверждение
невозможно» (click.py/enter.py/scroll.py/drag.py, пины — tests/test_outcome.py).

СТРАЖ НЕОБРАТИМЫХ ДЕЙСТВИЙ (IrreversibleGuard):

    Спека запрещает ЯДРУ повторять необратимое при UNKNOWN; «решение за
    стратегией» — это и выбор момента повтора, и ответственность за него.
    Страж — механизм ядра, делающий запрет исполняемым: действие,
    помеченное необратимым, при UNKNOWN внутренне блокируется от повторного
    исполнения до явного снятия стратегией (confirm_after_unknown с
    диагностикой разборок). Страж НЕ исполняет действие и не решает за
    стратегию — он только не даёт молча повторить то, что могло уже
    свершиться. recover-исполнитель — 5.8, здесь нет (YAGNI).

    Поток стратегии:

        guard.before(action_id)                  # IrreversibleRepeatError?
        result = primitive(...)                  # примитив 5.x
        if result.cancelled: ...                 # домен kill switch (8.1)
        outcome = to_outcome(result)             # OK/FAILED/UNKNOWN
        guard.after(action_id, outcome, detail)  # UNKNOWN -> блок
        ...
        # после разборок (ручная проверка состояния игры):
        guard.confirm_after_unknown(action_id, detail)
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Tuple, runtime_checkable
from typing import Protocol as _Protocol


__all__ = [
    "ActionOutcome",
    "GuardResolution",
    "IrreversibleGuard",
    "IrreversibleRepeatError",
    "PrimitiveResult",
    "to_outcome",
]


class ActionOutcome(str, Enum):
    """Трёхзначный итог критичного действия (spec action-primitives).

    str-mixin — для логирования/диагностики (f"{outcome}"). CANCELLED сюда
    не входит — домен kill switch (8.1), см. докстринг модуля.
    """

    OK = "OK"          # эффект подтверждён verify
    FAILED = "FAILED"  # эффект опровергнут или не применён (verify провалился)
    UNKNOWN = "UNKNOWN"  # эффект не подтверждён: мог свершиться, доказать нельзя


@runtime_checkable
class PrimitiveResult(_Protocol):
    """Общий контракт результата примитива 5.1–5.6 для адаптера to_outcome.

    success — эффект подтверждён; cancelled — прервано kill switch'ем (8.1),
    до to_outcome не доходит; unconfirmed — эффект, скорее всего, применён,
    но подтвердить невозможно (UNKNOWN-земля).
    """

    success: bool
    cancelled: bool
    unconfirmed: bool


def to_outcome(result: PrimitiveResult) -> ActionOutcome:
    """Адаптер результата примитива в ActionOutcome (диагностика — в поле
    failures самого результата, здесь не дублируется).

    Отменённый результат не адаптируется: ValueError (fail fast) — отмена —
    домен kill switch (8.1), стратегия обязана ветвить cancelled до вызова.
    """
    if result.cancelled:
        raise ValueError(
            "to_outcome: отменённый результат (cancelled=True) — домен "
            "kill switch (8.1), не ActionOutcome; обработайте отмену до "
            "адаптации (cancelled ≠ OK/FAILED/UNKNOWN)"
        )
    if result.success:
        return ActionOutcome.OK
    if result.unconfirmed:
        return ActionOutcome.UNKNOWN
    return ActionOutcome.FAILED


class IrreversibleRepeatError(Exception):
    """Повтор необратимого действия заблокирован стражем (UNKNOWN не
    разрешён стратегией): исполнение не подтверждено, повтор запрещён до
    явного снятия — confirm_after_unknown."""

    def __init__(self, action_id: str, reason: str):
        self.action_id = action_id
        self.reason = reason
        super().__init__(
            f"необратимое действие {action_id!r} заблокировано: исход "
            f"предыдущего исполнения UNKNOWN ({reason}) — повтор запрещён "
            f"до confirm_after_unknown (решение за стратегией)"
        )


@dataclass(frozen=True)
class GuardResolution:
    """Запись явного снятия блокировки стратегией (диагностика/аудит)."""

    action_id: str
    detail: str


class IrreversibleGuard:
    """Страж необратимых действий: UNKNOWN блокирует повтор до явного снятия.

    Страж инфраструктурный (не исполняет действия): стратегия помечает
    необратимые действия идентификатором (например, "start_sell") и обязует
    себя вызывать before/after вокруг каждого исполнения. Декоратор-обёртка
    wrap() — защита от пропуска before.
    """

    def __init__(self) -> None:
        # action_id -> причина блокировки (диагностика последнего UNKNOWN).
        self._blocked: Dict[str, str] = {}
        self._resolutions: list[GuardResolution] = []

    @property
    def resolutions(self) -> Tuple[GuardResolution, ...]:
        """Журнал снятий блокировок стратегией (не стирается)."""
        return tuple(self._resolutions)

    def is_blocked(self, action_id: str) -> bool:
        return action_id in self._blocked

    def before(self, action_id: str) -> None:
        """Точка входа перед исполнением необратимого действия: повтор при
        неподтверждённом исходе — IrreversibleRepeatError (действие не
        доходит до исполнителя)."""
        if action_id in self._blocked:
            raise IrreversibleRepeatError(action_id, self._blocked[action_id])

    def after(self, action_id: str, outcome: ActionOutcome,
              detail: str = "") -> None:
        """Фиксация итога исполнения: UNKNOWN блокирует повтор (снятие —
        только confirm_after_unknown). OK/FAILED не блокируют."""
        if not isinstance(outcome, ActionOutcome):
            raise TypeError(
                f"after: ожидается ActionOutcome, получено "
                f"{type(outcome).__name__!r} — отмена/результат примитива "
                f"(домен kill switch) сюда не передаются: адаптируйте через "
                f"to_outcome после ветвления cancelled"
            )
        if outcome is ActionOutcome.UNKNOWN:
            reason = detail or "эффект не подтверждён (диагностика не передана)"
            self._blocked[action_id] = reason

    def confirm_after_unknown(self, action_id: str,
                              detail: str = "") -> GuardResolution:
        """Явное снятие блокировки стратегией после разборок (ручная
        проверка состояния игры). Снятие без блокировки — ValueError
        (fail fast: ошибка протокола вызывающего слоя). Возвращает запись
        снятия (журналируется в resolutions)."""
        if action_id not in self._blocked:
            raise ValueError(
                f"confirm_after_unknown: действие {action_id!r} не "
                f"заблокировано — снимать нечего (пропуск before/after?)"
            )
        del self._blocked[action_id]
        resolution = GuardResolution(action_id=action_id, detail=detail)
        self._resolutions.append(resolution)
        return resolution

    def wrap(self, action_id: str,
             fn: Callable[[], ActionOutcome]) -> Callable[[], ActionOutcome]:
        """Обёртка исполнителя: before → fn → after (UNKNOWN блокирует).
        Исключения fn пролетают мимо стража (страж фиксирует только итоги)."""

        @functools.wraps(fn)
        def runner() -> ActionOutcome:
            self.before(action_id)
            outcome = fn()
            self.after(action_id, outcome)
            return outcome

        return runner
