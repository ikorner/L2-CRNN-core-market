"""Модель потоков ядра: L2CoreBot — orchestrator потоков (задача 7.2,
spec core-api-events: «Модель потоков: executor, observer, dispatcher»).

ТРИ ПОТОКА (design.md, решение 8):

- action executor — ЕДИНСТВЕННЫЙ владелец HID. Выбран вариант Б: executor —
  это поток стратегии. API синхронен и блокирующий (7.1) и по спеке
  вызывается только из потока стратегии (thread-owner enforcement 7.1) —
  отдельный executor-поток с очередью команд (вариант А) при однопоточной
  стратегии не имел бы потребителя, только async-слой. Владение делается
  ИСПОЛНЯЕМЫМ: wiring.hid оборачивается OwnerGuardedHid (hid_owner.py) с
  потоком-владельцем = потоку конструктора бота (поток стратегии); HID-
  команда из любого другого потока — HidOwnershipError до железа
  (сценарий спеки «Observer не отправляет действия» — enforcement,
  не дисциплина). Многопоточные стратегии — не-цель v1.

- state observer — фоновый поток, ТОЛЬКО ЧТЕНИЕ: grab, детекция окна/
  статуса/состояния (3.4/3.5), перехватчик модалок (3.6), монитор лавки
  (6.3). HID observer'у не передаётся ВООБЩЕ (в конструктор не принимается)
  — отправить действие из observer-потока невозможно по построению.
  Публикует события в EventQueue: window_closed (переход «окно было ->
  окна нет»), modal (перехватчик сработал), sold (монитор детектировал
  продажу), escalation (устойчивая деградация occluded_or_unstable /
  устойчивая модалка — safety 8.4, одно событие за эпизод). Внутреннее
  состояние observer'а — допустимое исключение
  (монитор/наблюдатель, design 8): последний снапшот и факт открытого
  окна для детекции закрытия. Ожидание между итерациями — wait_until по
  инъецированным часам с предикатом stop/cancelled (инвариант 1: сна нет;
  kill switch 8.1 — через cancelled).

- event dispatcher — поток доставки: EventQueue -> callbacks подписчиков.
  СЕМАНТИКА ДОСТАВКИ (отклонение от буквы design 8, обоснование — в
  отчёте task-7.2-report.md): callbacks исполняются В ПОТОКЕ DISPATCHER'А,
  а не в observer-потоке. «Доставка в поток стратегии» буквально требовала
  бы event pump'а в потоке стратегии — это противоречит блокирующей модели
  API (поток стратегии спит в вызовах). Приёмлемая трактовка: push-callback
  не исполняется в producer-потоке (observer) и не блокирует его; точка
  исполнения — dispatcher-поток, она зафиксирована публично. Pull-форма,
  исполняемая строго в потоке стратегии, — wait_event (задача 7.3) поверх
  той же EventQueue. Callback, бросающий исключение, не роняет доставку
  остальных (исключение глушится с записью в лог). Callback НЕ должен
  вызывать блокирующий API: его поток — dispatcher, API поднимет
  RuntimeError (thread-owner 7.1) — граница зафиксирована.

L2CoreBot — orchestrator: создаётся в потоке стратегии с готовым ApiWiring
(7.1), start() порождает observer- и dispatcher-потоки, stop(timeout) —
корректная остановка (stop-флаги + join). Повторный start — EngineError
(жизненный цикл один; restart — не-цель). Публичная точка push(type,
payload) — для внешних источников событий (escalation от recover через
on_escalation wiring'а, killed от kill switch 8.1).

ГРАНИЦА GUARD: владение HID гарантируется для доступа ЧЕРЕЗ guard-
сборку — `bot.wiring.hid` (OwnerGuardedHid): любая HID-команда не из
потока стратегии отклоняется до железа, публичного обхода у обёртки нет
(fail closed, см. hid_owner.py). ВТОРОЙ КАНАЛ (исходный wiring): экземпляр
ApiWiring, переданный в конструктор, — неизменяемый (frozen dataclass),
бот работает с КОПИЕЙ (dataclasses.replace); исходный wiring хранит сырой
HidDevice, и зона ответственности за него — у вызывающего слоя: передал
wiring (или сырой hid из него) третьей стороне — обязан сначала обернуть
hid в OwnerGuardedHid. Эталонная сборка, при которой сырой HidDevice за
пределами конструктора бота не используется: create_api (живые адаптеры)
-> wiring -> L2CoreBot (bot.api, bot.wiring — guard-копия); ручная сборка
ApiWiring(...)/build_wiring(...) — точка расширения тестов и нестандартных
сборок, её дисциплина — контракт вызывающего слоя, не исполняемый guard
(глубокая инкапсуляция wiring — YAGNI). Доступ к блокирующему API — только
через bot.api.

YAGNI (задача 7.2): kill switch подключается параметром kill_switch
(8.1, l2core/api/killswitch.py: release_all при kill — исключение guard'а
владельца, событие killed — push в очередь этой сборки); журнал событий
и API-справочник payload'ов — 7.3 (реализовано: EventAPI в
bot.event_api, журнал ведётся на push очереди).
"""
from __future__ import annotations

import dataclasses
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Tuple

import numpy as np

from l2core.api.core import L2CoreAPI
from l2core.api.events import (
    EVENT_ESCALATION,
    EVENT_KILLED,
    EVENT_MODAL,
    EVENT_SOLD,
    EVENT_WINDOW_CLOSED,
    Event,
    EventAPI,
    EventJournal,
    EventQueue,
)
from l2core.api.hid_owner import OwnerGuardedHid
from l2core.detector.status import STATUS_NOT_FOUND, STATUS_OCCLUDED
from l2core.primitives.wait import wait_until

__all__ = [
    "EngineError",
    "EngineShutdown",
    "EventDispatcher",
    "L2CoreBot",
    "ObserverSnapshot",
    "StateObserver",
]

logger = logging.getLogger(__name__)

# Дефолт интервала опроса очереди dispatcher'ом (сек): потребитель должен
# замечать stop-флаг быстро; не конфигурируется отдельно (poll-петля).
_DISPATCH_POLL_INTERVAL = 0.1

# Safety (8.4, spec «Осмысленные ошибки при блокировках экрана и оверлеях»):
# пороги устойчивой деградации observer'а (подряд кадров/циклов) — превышение
# порога превращается в одно событие escalation за эпизод (дедуп: стратегия
# получает сигнал «экран заблокирован/перекрыт, действия остановлены», а не
# спам каждый цикл). Единичные всплески (межкадровый джиттер тултипа) —
# штатно, эскалации нет. Константы, не конфиг: события редки, требования
# настраивать порог нет (YAGNI; параметры конструктора — точка подмены тестов
# и нестандартных сборок).
_OCCLUDED_ESCALATION_FRAMES = 3     # кадров подряд occluded_or_unstable
_MODAL_ESCALATION_CYCLES = 3        # циклов подряд с modal.unknown


class EngineError(Exception):
    """Нарушение жизненного цикла движка (повторный start и пр.)."""

    def __init__(self, detail: str):
        super().__init__(f"L2CoreBot: {detail}")
        self.detail = detail


@dataclass(frozen=True)
class ObserverSnapshot:
    """Последнее наблюдение observer'а (снапшот, не запрос к экрану).

    window — открытое окно торговли на последнем кадре (None — ни одного);
    status — статус 3.5 этого окна (not_found, когда окна нет); state —
    подтверждённое состояние 3.4 (None — unknown/детектор не подключён);
    detail — диагностика детектора. Чтение из потока стратегии: снапшот —
    замена дорогому grab+детекции для «что сейчас на экране» между
    блокирующими вызовами API.
    """

    window: Optional[str] = None
    status: str = STATUS_NOT_FOUND
    state: Optional[str] = None
    detail: str = ""


@dataclass(frozen=True)
class EngineShutdown:
    """Итог stop(): stopped — оба потока завершились (join уложился)."""

    observer_joined: bool
    dispatcher_joined: bool

    @property
    def stopped(self) -> bool:
        return self.observer_joined and self.dispatcher_joined


class StateObserver:
    """Фоновый наблюдатель: читает экран, публикует события, HID не трогает.

    Принимает только провайдеров чтения (screen/game_window/current_window/
    find_anchor/window_status/detect_state) + компоненты наблюдения
    (monitor 6.3, modal 3.6) + очередь событий. HID в конструктор НЕ
    принимается — сценарий спеки «Observer не отправляет действия»
    выполнен по построению (intеграционно: wiring движка оборачивает hid
    guard'ом, observer его не получает).

    Один полный цикл наблюдения — _observe(): кадр -> открытое окно ->
    (смена: публикация window_closed) -> статус/состояние -> снапшот ->
    модалки -> монитор лавки (sold). Пауза между циклами — wait_until с
    предикатом «stop запрошен» (инвариант 1: сна нет; cancelled — kill
    switch 8.1). Исключение итерации (сбой детектора/экрана) не роняет
    поток: фиксируется в лог, цикл продолжается — наблюдатель обязан
    переживать кратковременные деградации (деградация 3.5/6.3 — штатный
    режим, не повод умирать).

    Safety (8.4, spec «Осмысленные ошибки при блокировках экрана и
    оверлеях»): устойчивая деградация статуса 3.5 (occluded_or_unstable
    N кадров подряд — anchor_lost/anchor_moved/anchor_appeared, перекрытие
    тултипом/окном) и устойчивая неизвестная модалка (3.6, N циклов подряд)
    публикуются одним событием escalation за эпизод (payload —
    {"target", "detail", "steps": ()} в формате 7.3; target=None при
    модалке). Единичные всплески — не эскалация; возврат к устойчивому
    состоянию переводит счётчик (новый эпизод — новое событие). Это
    наблюдательный канал «заблокированный экран»: действия при этом не
    выполняются — примитивы 5.x отклоняют действие прологом (activate +
    status found), observer лишь даёт стратегии диагностируемый сигнал
    между блокирующими вызовами.
    """

    def __init__(
        self,
        *,
        screen,
        game_window: Callable[[], Any],
        current_window: Callable[[np.ndarray], Optional[str]],
        find_anchor: Callable[[np.ndarray, str], Any],
        window_status: Callable[[np.ndarray, str], Any],
        queue: EventQueue,
        interval: float,
        monitor: Optional[Any] = None,
        modal: Optional[Any] = None,
        detect_state: Optional[Callable[[np.ndarray, Any], Any]] = None,
        cancelled: Optional[Callable[[], bool]] = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        window_ids: Tuple[str, ...] = ("market", "shop_buy", "shop_sell"),
        occluded_escalation_frames: int = _OCCLUDED_ESCALATION_FRAMES,
        modal_escalation_cycles: int = _MODAL_ESCALATION_CYCLES,
    ):
        if interval <= 0:
            raise ValueError(f"interval observer'а должен быть > 0 "
                             f"(получено {interval!r})")
        if occluded_escalation_frames <= 0:
            raise ValueError(f"порог устойчивой деградации должен быть > 0 "
                             f"(получено {occluded_escalation_frames!r})")
        if modal_escalation_cycles <= 0:
            raise ValueError(f"порог устойчивой модалки должен быть > 0 "
                             f"(получено {modal_escalation_cycles!r})")
        self._screen = screen
        self._game_window = game_window
        self._current_window = current_window
        self._find_anchor = find_anchor
        self._window_status = window_status
        self._queue = queue
        self._interval = interval
        self._monitor = monitor
        self._modal = modal
        self._detect_state = detect_state
        self._cancelled = cancelled
        self._clock = clock
        self._sleep = sleep
        self._window_ids = window_ids
        self._occluded_escalation_frames = occluded_escalation_frames
        self._modal_escalation_cycles = modal_escalation_cycles
        # Safety (8.4): счётчики/флаги эпизодов устойчивой деградации
        # (внутреннее состояние наблюдателя — монитору разрешено, design 8).
        self._occluded_streak = 0
        self._occluded_escalated = False
        self._modal_streak = 0
        self._modal_escalated = False
        self._stop = threading.Event()
        self._snapshot = ObserverSnapshot(detail="наблюдение ещё не выполнялось")
        self._iterations = 0

    @property
    def interval(self) -> float:
        """Пауза между циклами наблюдения, секунды."""
        return self._interval

    @property
    def snapshot(self) -> ObserverSnapshot:
        """Последнее наблюдение (актуально на момент последнего цикла)."""
        return self._snapshot

    @property
    def iterations(self) -> int:
        """Число выполненных циклов наблюдения (диагностика/тесты)."""
        return self._iterations

    def request_stop(self) -> None:
        """Запросить остановку цикла (idempotent)."""
        self._stop.set()

    def run(self) -> None:
        """Тело потока observer'а: цикл наблюдения до stop/cancelled."""
        prev_window: Optional[str] = None
        while not self._stop.is_set():
            if self._cancelled is not None and self._cancelled():
                break   # отмена до чтения кадра (kill switch 8.1)
            try:
                prev_window = self._observe(prev_window)
            except Exception:
                logger.exception(
                    "observer: итерация наблюдения завершилась ошибкой "
                    "(цикл продолжается)")
            self._iterations += 1
            waited = wait_until(
                self._stop.is_set,
                timeout=self._interval,
                interval=min(self._interval, 0.05),
                cancelled=self._cancelled,
                clock=self._clock, sleep=self._sleep)
            if waited.success or waited.cancelled:
                break

    # --- один цикл наблюдения ---

    def _observe(self, prev_window: Optional[str]) -> Optional[str]:
        frame = np.asarray(self._screen.grab())
        window = self._current_window(frame)
        if window != prev_window:
            if prev_window is not None:
                # Открытое окно исчезло со всех кадров: событие ядру.
                self._queue.push(EVENT_WINDOW_CLOSED, {"window": prev_window})
            prev_window = window
        status = STATUS_NOT_FOUND
        state: Optional[str] = None
        detail = ""
        if window is not None:
            report = self._window_status(frame, window)
            status = report.status
            detail = report.reason or report.content_detail or ""
            if self._detect_state is not None and report.anchor is not None:
                detected = self._detect_state(frame, report.anchor)
                state = detected.state
            self._track_occluded(window, report)
        else:
            self._reset_occluded()
        self._snapshot = ObserverSnapshot(
            window=window, status=status, state=state, detail=detail)
        if self._modal is not None:
            game = self._game_window()
            anchors = {window_id: self._find_anchor(frame, window_id)
                       for window_id in self._window_ids}
            modal_report = self._modal.check(frame, game, anchors)
            self._track_modal(modal_report)
        if self._monitor is not None:
            poll = self._monitor.poll()
            if poll.sold is not None:
                self._queue.push(EVENT_SOLD, poll.sold)
        return prev_window

    # --- safety: устойчивая деградация -> escalation (8.4) ---

    def _track_occluded(self, window: str, report: Any) -> None:
        """Счётчик эпизода occluded_or_unstable: порог подряд кадров ->
        одно событие escalation за эпизод (дедуп флагом)."""
        if report.status != STATUS_OCCLUDED:
            self._reset_occluded()
            return
        self._occluded_streak += 1
        if (not self._occluded_escalated
                and self._occluded_streak >= self._occluded_escalation_frames):
            self._queue.push(EVENT_ESCALATION, {
                "target": window,
                "detail": report.reason or "окно нестабильно или перекрыто",
                "steps": (),
            })
            self._occluded_escalated = True
            logger.warning(
                "observer: устойчивая деградация окна %s (%s кадров подряд): "
                "escalation", window, self._occluded_streak)

    def _reset_occluded(self) -> None:
        """Снятие эпизода деградации (found/not_found/нет окна)."""
        self._occluded_streak = 0
        self._occluded_escalated = False

    def _track_modal(self, modal_report: Any) -> None:
        """Счётчик эпизода modal.unknown: порог подряд циклов -> одно
        событие escalation за эпизод (модалка не уходит сама)."""
        if modal_report.detected:
            self._queue.push(EVENT_MODAL, modal_report)
            self._modal_streak += 1
            if (not self._modal_escalated
                    and self._modal_streak >= self._modal_escalation_cycles):
                reason = modal_report.reason or "кандидат не уходит"
                self._queue.push(EVENT_ESCALATION, {
                    "target": None,
                    "detail": f"устойчивая неизвестная модалка: {reason}",
                    "steps": (),
                })
                self._modal_escalated = True
                logger.warning(
                    "observer: устойчивая модалка (%s циклов подряд): "
                    "escalation", self._modal_streak)
        else:
            self._modal_streak = 0
            self._modal_escalated = False


class EventDispatcher:
    """Поток доставки событий: EventQueue -> callbacks подписчиков.

    Callbacks исполняются В ЭТОМ потоке (не в observer-потоке — см.
    семантику доставки в докстринге модуля; не в потоке стратегии — она
    блокирована в API; pull-форма для потока стратегии — wait_event 7.3).
    Callback получает Event (type/payload/seq). Исключение callback'а
    изолируется: доставка остальных продолжается, сбой — в лог (диспетчер
    не должен умирать от чужого кода). Остановка — request_stop() +
    poll-цикл: get с конечным таймаутом замечает флаг.
    """

    def __init__(self, queue: EventQueue, *,
                 poll_interval: float = _DISPATCH_POLL_INTERVAL):
        if poll_interval <= 0:
            raise ValueError(f"poll_interval dispatcher'а должен быть > 0 "
                             f"(получено {poll_interval!r})")
        self._queue = queue
        self._poll_interval = poll_interval
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._callbacks: Mapping[str, list] = {}

    def subscribe(self, event_type: str, callback: Callable[[Event], None]
                  ) -> None:
        """Подписать callback на тип события (повтор — допустим, вызовет
        callback дважды за событие — забота подписчика)."""
        with self._lock:
            self._callbacks.setdefault(event_type, []).append(callback)

    def unsubscribe(self, event_type: str,
                    callback: Callable[[Event], None]) -> None:
        """Отписать callback (нет подписки — no-op)."""
        with self._lock:
            callbacks = self._callbacks.get(event_type)
            if callbacks and callback in callbacks:
                callbacks.remove(callback)

    def request_stop(self) -> None:
        """Запросить остановку цикла доставки (idempotent)."""
        self._stop.set()

    def run(self) -> None:
        """Тело потока dispatcher'а: доставка до stop-флага."""
        while not self._stop.is_set():
            event = self._queue.get(timeout=self._poll_interval)
            if event is None:
                continue
            with self._lock:
                callbacks = tuple(self._callbacks.get(event.type, ()))
            for callback in callbacks:
                try:
                    callback(event)
                except Exception:
                    logger.exception(
                        "dispatcher: callback события %r (seq=%s) бросил "
                        "исключение — доставка остальных продолжается",
                        event.type, event.seq)


class L2CoreBot:
    """Orchestrator потоков ядра: strategy(executor) + observer + dispatcher.

    Создаётся в потоке стратегии с готовым ApiWiring (7.1). Делает wiring
    guard-сборкой (hid -> OwnerGuardedHid, владелец — поток-конструктор =
    action executor) и отдаёт фасад bot.api, привязанный к тому же потоку.
    start() порождает observer- и dispatcher-потоки; stop(timeout) —
    stop-флаги + join; повторный start — EngineError (цикл жизни один).

    События: observer публикует в bot.events (EventQueue); внешние
    источники (recover on_escalation 5.8) — через bot.push(type, payload);
    kill switch (8.1) подключается параметром kill_switch: release_all при
    kill — исключение guard'а владельца, событие killed — push этой
    очереди, а wiring.cancelled движок подставляет ks.is_killed сам
    (явный wiring.cancelled сборки побеждает). Доставка —
    bot.dispatcher.subscribe(type, cb).
    """

    def __init__(
        self,
        wiring,
        *,
        monitor: Optional[Any] = None,
        modal: Optional[Any] = None,
        poll_interval: Optional[float] = None,
        dispatch_interval: float = _DISPATCH_POLL_INTERVAL,
        cancelled: Optional[Callable[[], bool]] = None,
        kill_switch: Optional[Any] = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._owner = threading.current_thread()
        # Kill switch (8.1): без явного cancelled (параметр, wiring) сборка
        # проверяет kill-флаг — headline-гарантия спеки «прерывание
        # блокирующих вызовов» не должна зависеть от ручной передачи
        # cancelled в build_wiring. Приоритет: явный wiring.cancelled
        # (сборка вызывающего слоя) > параметр cancelled > kill_switch.
        cancelled_default = (cancelled if cancelled is not None
                             else (kill_switch.is_killed
                                   if kill_switch is not None else None))
        # release_all при kill — санкционированное исключение guard'а
        # владельца (слушатель — чужой поток, см. hid_owner.py/
        # killswitch.py); событие killed публикуется в очередь этой сборки.
        guarded = OwnerGuardedHid(wiring.hid, owner_thread=self._owner,
                                  kill_switch=(kill_switch.is_killed
                                               if kill_switch is not None
                                               else None))
        self._wiring = dataclasses.replace(
            wiring, hid=guarded,
            cancelled=(wiring.cancelled if wiring.cancelled is not None
                       else cancelled_default))
        # Журнал ведётся на push (до потребления): pull-форма wait_event
        # не теряет события, забранные dispatcher'ом (7.3).
        self._queue = EventQueue(journal=EventJournal())
        if kill_switch is not None:
            kill_switch.attach(
                hid=guarded,
                on_kill=lambda reason: self._queue.push(
                    EVENT_KILLED, {"reason": reason}))
        self._event_api = None   # создан после dispatcher'а (см. ниже)
        interval = (poll_interval if poll_interval is not None
                    else wiring.config.timing.poll_interval)
        # Id окон наблюдения — из сборки моделей (дедупликация: модели
        # в тестах могут совпадать).
        seen: list = []
        for window_id in (wiring.market.window.id, wiring.shop_buy.window.id,
                          wiring.shop_sell.window.id):
            if window_id not in seen:
                seen.append(window_id)
        self._observer = StateObserver(
            screen=wiring.screen,
            game_window=wiring.game_window,
            current_window=wiring.current_window,
            find_anchor=wiring.find_anchor,
            window_status=wiring.window_status,
            queue=self._queue, interval=interval, monitor=monitor,
            modal=modal, detect_state=wiring.detect_state,
            cancelled=cancelled_default, clock=clock, sleep=sleep,
            window_ids=tuple(seen))
        self._dispatcher = EventDispatcher(self._queue,
                                           poll_interval=dispatch_interval)
        # API событий (7.3): push-подписки — через dispatcher этой сборки;
        # pull/journal — поверх очереди с журналом; владелец — поток
        # стратегии (R14: wait_event строго из него).
        self._event_api = EventAPI(self._queue, self._dispatcher,
                                   owner_thread=self._owner)
        self._api = L2CoreAPI(self._wiring, owner_thread=self._owner,
                              events=self._event_api)
        self._cancelled = cancelled
        self._observer_thread: Optional[threading.Thread] = None
        self._dispatcher_thread: Optional[threading.Thread] = None
        self._started = False
        self._stopped = False

    # ------------------------------------------------------------------
    # Доступные точки сборки

    @property
    def api(self) -> L2CoreAPI:
        """Блокирующий фасад: вызывается только из потока-владельца
        (поток конструктора бота)."""
        return self._api

    @property
    def wiring(self):
        """Guard-сборка провайдеров (hid — OwnerGuardedHid)."""
        return self._wiring

    @property
    def events(self) -> EventQueue:
        """Очередь событий ядра (pull-форма 7.3 строится поверх неё)."""
        return self._queue

    @property
    def event_api(self) -> EventAPI:
        """События двух форм (7.3): on (push, dispatcher-поток, R14),
        wait_event (pull, строго из потока стратегии), get_events (журнал),
        push (внешним источникам, 8.1). Тот же объект виден через
        bot.api.events."""
        return self._event_api

    @property
    def observer(self) -> StateObserver:
        """Компонент наблюдателя (снапшот/диагностика)."""
        return self._observer

    @property
    def dispatcher(self) -> EventDispatcher:
        """Компонент доставки (подписка callback'ов)."""
        return self._dispatcher

    @property
    def observer_thread(self) -> threading.Thread:
        """Поток observer'а (None до start())."""
        return self._observer_thread

    @property
    def dispatcher_thread(self) -> threading.Thread:
        """Поток dispatcher'а (None до start())."""
        return self._dispatcher_thread

    def push(self, event_type: str, payload: Any = None) -> Event:
        """Опубликовать событие внешнего источника (escalation от recover
        5.8 через on_escalation wiring'а, killed от kill switch 8.1)."""
        return self._queue.push(event_type, payload)

    # ------------------------------------------------------------------
    # Жизненный цикл

    def start(self) -> None:
        """Запустить observer- и dispatcher-потоки.

        Повторный вызов — EngineError (в т.ч. после stop: restart — не-цель,
        жизненный цикл бота один)."""
        if self._started:
            raise EngineError(
                "повторный запуск: бот уже запущен" if not self._stopped
                else "повторный запуск после stop не поддерживается "
                     "(жизненный цикл один)")
        self._started = True
        self._observer_thread = threading.Thread(
            target=self._observer.run, name="l2core-observer")
        self._dispatcher_thread = threading.Thread(
            target=self._dispatcher.run, name="l2core-dispatcher")
        self._observer_thread.start()
        self._dispatcher_thread.start()

    def stop(self, timeout: Optional[float] = None) -> EngineShutdown:
        """Остановить потоки: stop-флаги + join с таймаутом (idempotent).

        Возвращает факты join'а; stopped=False — поток жив после таймаута
        (вызывающий слой решает, эскалировать ли; HID при этом не тронут —
        guard владельца никто не обходит). Вызывать можно из callback'а
        dispatcher'а (сценарий «killed останавливает бота»): собственный
        поток не join'ится (join current thread — RuntimeError), stop-флаг
        гарантирует его завершение после текущей итерации."""
        if not self._started:
            self._stopped = True
            return EngineShutdown(observer_joined=True,
                                  dispatcher_joined=True)
        self._stopped = True
        self._observer.request_stop()
        self._dispatcher.request_stop()
        current = threading.current_thread()
        observer_joined = self._join(self._observer_thread, timeout, current)
        dispatcher_joined = self._join(self._dispatcher_thread, timeout,
                                       current)
        return EngineShutdown(observer_joined=observer_joined,
                              dispatcher_joined=dispatcher_joined)

    @staticmethod
    def _join(thread: threading.Thread, timeout: Optional[float],
              current: threading.Thread) -> bool:
        """Join не-собственного потока; собственный — считается завершённым
        (stop-флаг выставлен до join, поток выйдет после итерации)."""
        if thread is current:
            return True
        thread.join(timeout=timeout)
        return not thread.is_alive()

    def __enter__(self) -> "L2CoreBot":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop(timeout=5.0)
