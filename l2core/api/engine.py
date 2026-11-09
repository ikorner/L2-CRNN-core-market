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
  продажу). Внутреннее состояние observer'а — допустимое исключение
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

ГРАНИЦА GUARD: доступ к блокирующему API — только через bot.api (его wiring
— guard-обёртка). Прямое использование исходного ApiWiring с сырым
HidDevice обходит guard — живой HidDevice после сборки доступен только
здесь (create_api отдаёт wiring наружу конструктору бота; дальше — дисциплина
сборки, YAGNI: глубокая инкапсуляция wiring — не-цель).

YAGNI (задача 7.2): kill switch (8.1), on()/wait_event() для стратегий,
журнал событий и API-справочник payload'ов — 7.3; подключение
on_escalation wiring'а к очереди — точка push уже здесь, wiring решает
вызывающий слой.
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
    EVENT_MODAL,
    EVENT_SOLD,
    EVENT_WINDOW_CLOSED,
    Event,
    EventQueue,
)
from l2core.api.hid_owner import OwnerGuardedHid
from l2core.detector.status import STATUS_NOT_FOUND
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
    ):
        if interval <= 0:
            raise ValueError(f"interval observer'а должен быть > 0 "
                             f"(получено {interval!r})")
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
        self._snapshot = ObserverSnapshot(
            window=window, status=status, state=state, detail=detail)
        if self._modal is not None:
            game = self._game_window()
            anchors = {window_id: self._find_anchor(frame, window_id)
                       for window_id in self._window_ids}
            modal_report = self._modal.check(frame, game, anchors)
            if modal_report.detected:
                self._queue.push(EVENT_MODAL, modal_report)
        if self._monitor is not None:
            poll = self._monitor.poll()
            if poll.sold is not None:
                self._queue.push(EVENT_SOLD, poll.sold)
        return prev_window


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
    источники (recover on_escalation 5.8, kill switch 8.1) — через
    bot.push(type, payload). Доставка — bot.dispatcher.subscribe(type, cb).
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
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._owner = threading.current_thread()
        guarded = OwnerGuardedHid(wiring.hid, owner_thread=self._owner)
        self._wiring = dataclasses.replace(wiring, hid=guarded)
        self._api = L2CoreAPI(self._wiring, owner_thread=self._owner)
        self._queue = EventQueue()
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
            cancelled=cancelled, clock=clock, sleep=sleep,
            window_ids=tuple(seen))
        self._dispatcher = EventDispatcher(self._queue,
                                           poll_interval=dispatch_interval)
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
