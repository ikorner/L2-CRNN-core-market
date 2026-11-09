"""Инфраструктура очереди событий ядра (задача 7.2, spec core-api-events:
«Модель потоков: executor, observer, dispatcher»).

EventQueue — потокобезопасная очередь событий между producer'ами
(state observer 7.2, recover-цепочка 5.8 через push, kill switch 8.1)
и consumer'ом (event dispatcher 7.2; pull-форма wait_event — 7.3 поверх
этой же очереди). Порядок событий фиксируется монотонным seq (счётчик
на очереди): порядок доставки == порядок публикации, диагностика
потоковых гонок — по seq.

ТИПЫ СОБЫТИЙ (spec core-api-events, раздел событий): sold, window_closed,
modal, recovered, escalation, killed. Инфраструктура типы ЗНАЕТ
(неизвестный тип — ValueError, fail fast: опечатка типа не должна
потеряться молча), но НЕ ПОРОЖДАЕТ: источники подключаются —
- sold / window_closed / modal — state observer (7.2, engine.py);
- escalation / recovered — recover-цепочка/стратегия через L2CoreBot.push
  (точка входа on_escalation wiring'а подключается вызывающим слоем);
- killed — kill switch (8.1).

YAGNI (задача 7.2): публичный API событий для стратегий (on(callback),
wait_event(type, timeout), журнал с полезной нагрузкой, API-справочник) —
задача 7.3; здесь только очередь с типизированным событием.
"""
from __future__ import annotations

import itertools
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

__all__ = [
    "EVENT_ESCALATION",
    "EVENT_KILLED",
    "EVENT_MODAL",
    "EVENT_RECOVERED",
    "EVENT_SOLD",
    "EVENT_WINDOW_CLOSED",
    "EVENT_TYPES",
    "Event",
    "EventQueue",
]

# Типы событий (spec core-api-events: «типы событий SHALL включать ...»).
EVENT_SOLD = "sold"
EVENT_WINDOW_CLOSED = "window_closed"
EVENT_MODAL = "modal"
EVENT_RECOVERED = "recovered"
EVENT_ESCALATION = "escalation"
EVENT_KILLED = "killed"
EVENT_TYPES = frozenset({
    EVENT_SOLD, EVENT_WINDOW_CLOSED, EVENT_MODAL, EVENT_RECOVERED,
    EVENT_ESCALATION, EVENT_KILLED,
})


@dataclass(frozen=True)
class Event:
    """Событие ядра: тип, полезная нагрузка, порядковый номер.

    payload — объект, определённый источником (SoldPayload 6.3 для sold,
    ModalReport 3.6 для modal, dict для window_closed и пр.); формат
    полезной нагрузки фиксирует API-справочник событий (задача 7.3).
    """

    type: str
    payload: Any
    seq: int


class EventQueue:
    """Потокобезопасная FIFO-очередь событий с монотонным seq.

    push — из любого потока (observer, recover, kill switch); get —
    блокирующее ожидание с таймаутом (None по таймауту) для потока
    dispatcher'а. Один счётчик seq под lock'ом: события строго
    упорядочены моментом публикации, гонок пушей нет.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._items: list[Event] = []
        self._counter = itertools.count(1)

    def push(self, event_type: str, payload: Any = None) -> Event:
        """Опубликовать событие; вернуть его (с присвоенным seq)."""
        if event_type not in EVENT_TYPES:
            raise ValueError(
                f"неизвестный тип события {event_type!r}: допустимы "
                f"{sorted(EVENT_TYPES)} (опечатка типа — fail fast, "
                f"событие не должно потеряться молча)")
        with self._cond:
            event = Event(type=event_type, payload=payload,
                          seq=next(self._counter))
            self._items.append(event)
            self._cond.notify()
            return event

    def get(self, timeout: Optional[float] = None) -> Optional[Event]:
        """Взять событие; None — таймаут ожидания (событий не было).

        timeout=None — ждать бесконечно (для будущего wait_event 7.3);
        конечный таймаут — poll-цикл dispatcher'а (stop-флаг снаружи).
        """
        with self._cond:
            if timeout is None:
                while not self._items:
                    self._cond.wait()
            else:
                deadline = time.monotonic() + timeout
                while not self._items:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    self._cond.wait(remaining)
            return self._items.pop(0)

    def __len__(self) -> int:
        with self._cond:
            return len(self._items)
