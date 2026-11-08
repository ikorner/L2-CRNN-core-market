"""Примитив drag-переноса предметов (задача 5.6, spec action-primitives:
«Каждое действие верифицируется, при неудаче — retry → on_fail → recover»;
market-scenarios/design: «Перенос предметов в лавку и обратно — drag»;
инвариант 9: слепых действий нет).

drag_item — оркестратор по паттерну 5.3-5.5, весь внешний мир — инъекции
(HID, экран/транзакция кадра, activate, статус окна, поиск якоря, оценка
проверок, rng, часы):

    пролог → цикл попыток (retry из конфига) → on_fail[] → сигнал recover

ГРАНИЦА ОТВЕТСТВЕННОСТИ (кто считает точки): drag_item принимает ГОТОВЫЕ
экранные координаты source_point/target_point. Почему не элементы модели,
как в click_element: точки переноса — это ячейки сеток инвентаря/лавки,
координаты которых ГЕНЕРИРУЮТСЯ из геометрии сетки (контракт design.md:
«slot: координаты ячейки генерируются из геометрии сетки лавки, не
размечаются вручную» — генерация принадлежит вызывающему слою, сценарию
6.2, у ядра-интерпретатора модели таких элементов нет). Якорь свежий
вызывающий слой берёт непосредственно перед вызовом; drag сам пересчитывает
точки внутри попытки не может (курсор уже тащит предмет), поэтому за сдвиг
окна между расчётом точек и drop отвечает verify: эффект не подтвердился →
retry с пересчитанными точками (вызывающий слой решает). Verify-контракт
drag — типизированная VerifySpec, как у клика: наблюдаемые признаки цели
(иконка предмета в ячейке-цели и пр.), оценка — через инъекцию run_checks.

ПРОЛОГ (как 5.3-5.5): drag допустим только при activate_game OK (5.2, иначе
ActivateGameError проброшена — эскалация) И статусе окна found (3.5). Иначе
— исключение, HID не трогается (no blind drag). Точки переданы вызывающим
слоем, но слепым действие не становится: окно проверено по свежему кадру,
перед press на каждой попытке есть свежий якорь.

МЕХАНИКА HID-DRAG: готовой команды drag в протоколе вендора нет — собирается
из низкоуровневых примитивов HidDevice (вендорить нельзя, правки
запрещены): move_rel к source, mouse_down (press), удержание, пакетная
траектория traj ((dx, dy, delay_ms) за точку, исполнение на плате), пауза
на цели, mouse_up (release). Одна траектория — один пакет CMD_TRAJ
(до TRAJ_MAX_POINTS точек; поведенческий профиль ограничен fail-fast
проверкой на планировании).

ТРАЕКТОРИЯ (человеческая, behavior-профиль): source и target получают
джиттер click_jitter_px (как точка клика, 5.3); между ними — 1..3
промежуточные точки (drag_midpoints_min/max) с джиттером move_jitter_px на
отрезке; последний шаг — ровно в target (джиттер только на точке назначения,
не за целью). Пауза каждого шага — move_step_delay_min/max_ms через rng.
Порядок rng-розыгрышей на попытку зафиксирован пинами тестов
(tests/test_drag.py::planned): джиттер source, джиттер target, hold,
число середин, на каждую середину (джиттер + задержка), задержка финала.

ПАУЗЫ БЕЗ SLEEP (инвариант 1): удержание после press (press_delay из
профиля, мс → сек) и пауза на цели перед release (timing.hover_pause —
как hover в 5.3: UI успевает подсветить цель drop) — фиксированные
задержки через _fixed_pause по инъецированным часам (механизм дедлайна
wait_until, см. click.py). Прямых вызовов sleep здесь нет.

ОТМЕНА МЕЖДУ PRESS И RELEASE: если cancelled сработал в удержании или в
паузе на цели, кнопка ОТПУЩЕНА (mouse_up) до возврата — мышь зажатой не
остаётся (инвариант 9: эскалация не оставляет опасного состояния).

ОБРАТИМОСТЬ И REVERTIFY (3.6): перенос предмета меняет состояние, но
обратим — предмет можно перетащить обратно. Re-verify «перед необратимым»
применяется к подтверждению продажи start_sell (6.2), НЕ к drag: drag
обязан лишь в verify ПОСЛЕ release (инвариант 2). Механика drag не мешает
reverify сценария: транзакция кадра (3.6) не вложенная — drag открывает
свою FrameTransaction, reverify start_sell исполняется вне примитива.

ГРАНИЦЫ: якорь проверяется перед press каждой попытки; сдвиг окна ПОСЛЕ
press поймает verify (см. границу ответственности за точки). Залипшая
кнопка при HID-сбое (HidError из транспорта) — эскалация рантайма, очистку
делает владелец сессии HID (release_all в __exit__ HidDevice, 5.8).

РЕЗУЛЬТАТ: DragResult (source_point/target_point/success/attempts/cancelled/
on_fail_executed/failures) — без трёхзначной обёртки (5.7, YAGNI).
needs_recover — сигнал общего recover (5.8).
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from l2core.adapters.hid import BTN_LEFT
from l2core.adapters.screen import Screen
from l2core.config import CoreConfig
from l2core.detector.anchor import Anchor
from l2core.detector.status import STATUS_FOUND, WindowStatus
from l2core.detector.transaction import FrameTransaction
from l2core.primitives.activate import ActivateResult
from l2core.primitives.click import (
    CheckRunner,
    ClickElementError,
    JitterRng,
    VerifySpec,
    _fixed_pause,
    _jitter,
    _move_to,
    _run_on_fail,
)

__all__ = [
    "DragElementError",
    "DragPreconditionError",
    "DragResult",
    "drag_item",
]


@dataclass(frozen=True)
class DragResult:
    """Итог drag (обёртка OK/FAILED/UNKNOWN — задача 5.7, YAGNI).

    success — verify подтверждён; attempts — число потраченных попыток;
    cancelled — прервано через cancelled (kill switch 8.1), verify не
    подтверждён; on_fail_executed — шаги on_fail[] исполнены; failures —
    диагностика провалов по попыткам.
    """

    source_point: Tuple[int, int]
    target_point: Tuple[int, int]
    success: bool
    attempts: int
    cancelled: bool = False
    on_fail_executed: bool = False
    failures: Tuple[str, ...] = ()

    @property
    def needs_recover(self) -> bool:
        """Сигнал общего recover (5.8): неуспех без отмены."""
        return not self.success and not self.cancelled


class DragElementError(Exception):
    """Ошибка исполнения примитива (профиль/аргументы), не эскалация рантайма."""

    def __init__(self, detail: str):
        super().__init__(f"drag_item: {detail}")
        self.detail = detail


class DragPreconditionError(DragElementError):
    """Пролог не пройден: статус окна не found — drag запрещён (инвариант 9).

    activate_game не удался — отдельно: ActivateGameError (5.2) проброшена.
    """

    def __init__(self, source_point, target_point, status: WindowStatus):
        self.status = status.status
        super().__init__(
            f"{source_point!r} -> {target_point!r}: статус окна "
            f"{status.status!r} ({status.reason or status.content_detail}) "
            f"— drag не выполняется"
        )


@dataclass(frozen=True)
class _DragPlan:
    """План одной попытки drag (rng розыгран до первого HID-вызова).

    source/target — точки с джиттером click_jitter_px; hold_s — удержание
    после press (press_delay из профиля, секунды); traj_points — точки
    пакета traj: (dx, dy, delay_ms) за шаг, последний шаг — ровно в target.
    """

    source: Tuple[int, int]
    target: Tuple[int, int]
    hold_s: float
    traj_points: Tuple[Tuple[int, int, int], ...]


def _plan_attempt(source: Tuple[int, int], target: Tuple[int, int],
                  config: CoreConfig, rng: JitterRng,
                  traj_max_points: int) -> _DragPlan:
    """Спланировать попытку: джиттеры, удержание, человеческая траектория.

    Порядок rng-розыгрышей зафиксирован пинами тестов (tests/test_drag.py::
    planned) — при фиксированном seed план и весь HID-трафик детерминированы
    (инвариант 7). Число промежуточных точек из профиля не влезает в пакет
    траектории HID-протокола — ошибка профиля до HID (fail fast).
    """
    behavior = config.behavior
    sjx, sjy = _jitter(rng, behavior.click_jitter_px)
    tjx, tjy = _jitter(rng, behavior.click_jitter_px)
    hold_ms = rng.randint(behavior.press_delay_min_ms,
                          behavior.press_delay_max_ms)
    n_mid = rng.randint(behavior.drag_midpoints_min,
                        behavior.drag_midpoints_max)

    cur = (source[0] + sjx, source[1] + sjy)
    points: list[Tuple[int, int, int]] = []
    for i in range(1, n_mid + 1):
        frac = i / (n_mid + 1)
        jx, jy = _jitter(rng, behavior.move_jitter_px)
        px = round(source[0] + (target[0] - source[0]) * frac) + jx
        py = round(source[1] + (target[1] - source[1]) * frac) + jy
        delay = rng.randint(behavior.move_step_delay_min_ms,
                            behavior.move_step_delay_max_ms)
        points.append((px - cur[0], py - cur[1], delay))
        cur = (px, py)
    final_delay = rng.randint(behavior.move_step_delay_min_ms,
                              behavior.move_step_delay_max_ms)
    points.append((target[0] + tjx - cur[0], target[1] + tjy - cur[1],
                   final_delay))

    if len(points) > traj_max_points:
        raise DragElementError(
            f"траектория drag: {len(points)} точек (drag_midpoints_max="
            f"{behavior.drag_midpoints_max}) превышает ёмкость пакета "
            f"HID-протокола ({traj_max_points}) — скорректируйте профиль")
    return _DragPlan(
        source=(source[0] + sjx, source[1] + sjy),
        target=(target[0] + tjx, target[1] + tjy),
        hold_s=hold_ms / 1000.0,
        traj_points=tuple(points),
    )


def drag_item(
    source_point: Tuple[int, int],
    target_point: Tuple[int, int],
    *,
    verify: VerifySpec,
    run_checks: CheckRunner,
    screen: Screen,
    hid,
    cursor_pos: Callable[[], Tuple[int, int]],
    activate: Callable[[], ActivateResult],
    window_status: Callable[[np.ndarray], WindowStatus],
    find_anchor: Callable[[np.ndarray], Optional[Anchor]],
    config: CoreConfig,
    on_fail: Tuple[str, ...] = (),
    free_area_point: Optional[Tuple[int, int]] = None,
    cancelled: Optional[Callable[[], bool]] = None,
    rng: Optional[JitterRng] = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> DragResult:
    """Drag-перенос из source_point в target_point: пролог → press →
    траектория → release → verify → retry → on_fail → сигнал recover.
    Контракт — в докстринге модуля; точки передаёт вызывающий слой (граница
    ответственности — там же).

    verify — обязателен: drag без verify-спецификации запрещён (spec
    action-primitives; guard — tests/test_drag.py::TestVerifyContract).
    """
    rng = rng or random.Random()
    traj_max_points = int(getattr(hid, "TRAJ_MAX_POINTS", 60))
    failures: list = []

    with FrameTransaction(screen) as tx:
        # --- ПРОЛОГ: activate OK (5.2) + статус found (3.5), иначе drag нет.
        activated = activate()
        if activated.cancelled:
            return DragResult(source_point, target_point, success=False,
                              attempts=0, cancelled=True)
        status = window_status(tx.frame)
        if status.status != STATUS_FOUND:
            raise DragPreconditionError(source_point, target_point, status)

        max_attempts = 1 + config.timing.click_retry_count
        for attempt in range(1, max_attempts + 1):
            if cancelled is not None and cancelled():
                return DragResult(source_point, target_point, success=False,
                                  attempts=attempt - 1, cancelled=True)

            frame = tx.refresh()
            anchor = find_anchor(frame)
            if anchor is None:
                failures.append(f"попытка {attempt}: якорь не найден на кадре")
                continue

            # План (rng) строится до первого HID-вызова: детерминизм
            # траектории и таймингов при фиксированном seed (инвариант 7).
            plan = _plan_attempt(source_point, target_point, config, rng,
                                 traj_max_points)

            _move_to(hid, cursor_pos, plan.source)
            hid.mouse_down(BTN_LEFT)
            _fixed_pause(plan.hold_s, config=config, cancelled=cancelled,
                         clock=clock, sleep=sleep)
            if cancelled is not None and cancelled():
                # Отмена в удержании: кнопку отпускаем, drag не продолжаем.
                hid.mouse_up(BTN_LEFT)
                return DragResult(source_point, target_point, success=False,
                                  attempts=attempt - 1, cancelled=True)
            hid.traj(list(plan.traj_points))
            _fixed_pause(config.timing.hover_pause, config=config,
                         cancelled=cancelled, clock=clock, sleep=sleep)
            if cancelled is not None and cancelled():
                # Отмена на цели: траектория прошла, кнопку отпускаем.
                hid.mouse_up(BTN_LEFT)
                return DragResult(source_point, target_point, success=False,
                                  attempts=attempt - 1, cancelled=True)
            hid.mouse_up(BTN_LEFT)

            verify_frame = tx.refresh()
            verify_anchor = find_anchor(verify_frame)
            if verify_anchor is None:
                failures.append(f"попытка {attempt}: якорь потерян перед verify")
                continue
            report = run_checks(verify.checks, verify_frame, verify_anchor)
            if report.passed:
                return DragResult(source_point, target_point, success=True,
                                  attempts=attempt)
            failures.append(
                f"попытка {attempt}: verify не подтверждён "
                f"({'; '.join(report.failures) or 'признаки не совпали'})")

        # Исчерпание retry: ПОВТОРНЫЙ ПРОЛОГ перед on_fail — за цикл окно
        # могло деградировать, а click_free_area кликает в область экрана:
        # без activate OK + статуса found это слепой клик (инвариант 9),
        # on_fail пропускаем и сразу отдаём сигнал recover с причиной.
        reactivated = activate()
        if reactivated.cancelled:
            return DragResult(source_point, target_point, success=False,
                              attempts=max_attempts, cancelled=True)
        status = window_status(tx.refresh())
        if status.status != STATUS_FOUND:
            failures.append(
                f"on_fail пропущен: статус окна {status.status!r} "
                f"({status.reason or status.content_detail}) — слепые "
                f"HID-шаги запрещены")
            return DragResult(source_point, target_point, success=False,
                              attempts=max_attempts, on_fail_executed=False,
                              failures=tuple(failures))
        try:
            _run_on_fail(on_fail, hid=hid, cursor_pos=cursor_pos,
                         free_area_point=free_area_point, config=config, rng=rng)
        except ClickElementError as exc:
            # Общий исполнитель шагов — из click.py; контракт drag — своя
            # ошибка модели (например, неизвестный шаг on_fail).
            raise DragElementError(exc.detail) from exc
        return DragResult(source_point, target_point, success=False,
                          attempts=max_attempts, on_fail_executed=bool(on_fail),
                          failures=tuple(failures))
