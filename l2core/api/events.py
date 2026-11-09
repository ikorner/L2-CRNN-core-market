"""Инфраструктура очереди событий ядра (задача 7.2) и события двух форм —
push (on) и pull (wait_event) — с журналом (задача 7.3, spec core-api-events:
«События в двух формах», «Журнал событий ведётся ядром»; ruling R14, см. ниже).

EventQueue — потокобезопасная очередь событий между producer'ами
(state observer 7.2, recover-цепочка 5.8, kill switch 8.1)
и consumer'ом (event dispatcher 7.2; pull-форма wait_event — 7.3).
Порядок событий фиксируется монотонным seq (счётчик
на очереди): порядок доставки == порядок публикации, диагностика
потоковых гонок — по seq.

ТИПЫ СОБЫТИЙ (spec core-api-events, раздел событий): sold, window_closed,
modal, recovered, escalation, killed. Инфраструктура типы ЗНАЕТ
(неизвестный тип — ValueError, fail fast: опечатка типа не должна
потеряться молча), но НЕ ПОРОЖДАЕТ: источники подключаются —
- sold / window_closed / modal — state observer (7.2, engine.py);
- recovered / escalation — api.recover() по итогу recover-цепочки 5.8/6.4
  (успех -> recovered, эскалация -> escalation);
- killed — kill switch (8.1) через bot.push / EventAPI.push.

СОБЫТИЯ ДВУХ ФОРМ (задача 7.3) — EventAPI поверх (queue, dispatcher):

- PUSH: on(type, callback) -> EventSubscription (отмена idempotent,
  контекстный менеджер). Точка исполнения callback'ов — dispatcher-поток
  (ruling R14, ратификация отклонения от буквы design 8 «доставка в поток
  стратегии»: буквальная доставка требовала бы event pump'а в потоке
  стратегии — противоречие блокирующей модели API 7.1; ключевое требование
  выполняется иначе — не в producer-потоке, с изоляцией исключений).
  Блокирующие вызовы API из callback недопустимы: dispatcher-поток ≠ поток
  владельца, L2CoreAPI поднимет RuntimeError (thread-owner 7.1).
- PULL: wait_event(type, timeout) -> Event | None — блокирующее ожидание
  СТРОГО из потока стратегии (owner_thread, enforcement R14);
  возвращает первое событие данного типа, опубликованное ПОСЛЕ вызова;
  таймаут — None. Ожидание строится по ЖУРНАЛУ, а не по очереди: журнал
  пишется на push (до всякого потребления), поэтому pull не теряет
  события, забранные dispatcher'ом из очереди.
- ЖУРНАЛ: EventJournal — кольцевой буфер (лимит DEFAULT_JOURNAL_LIMIT,
  константа: события ядра редки — десятки в час; 1000 записей — сутки+
  диагностики; конфигурация не требуется — YAGNI, добавить ключ поздно
  тривиально). Записи JournalEntry(event, timestamp): timestamp —
  реальное время в проде (time.time), в тестах — инъекция часов. Доступ:
  EventAPI.get_events(since_seq). Журнал ведёт ядро (spec), подключён к
  очереди: EventQueue(journal=...).
- ЕДИНЫЙ КАНАЛ ПУБЛИКАЦИИ: всё проходит через EventQueue.push (observer,
  api.recover, bot.push, EventAPI.push) — журнал и подписчики видят
  одни и те же события.

Форматы полезной нагрузки по типам — API-справочник docs/API.md.
"""
from __future__ import annotations

import itertools
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

__all__ = [
    "DEFAULT_JOURNAL_LIMIT",
    "EVENT_ESCALATION",
    "EVENT_KILLED",
    "EVENT_MODAL",
    "EVENT_RECOVERED",
    "EVENT_SOLD",
    "EVENT_WINDOW_CLOSED",
    "EVENT_TYPES",
    "Event",
    "EventAPI",
    "EventJournal",
    "EventQueue",
    "EventSubscription",
    "JournalEntry",
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

    journal (опционально) — EventJournal: запись выполняется на push,
    ДО потребления очереди (dispatcher/pull): журнал — полный след
    публикаций, вытеснение старых записей — по лимиту буфера.
    """

    def __init__(self, journal: Optional["EventJournal"] = None) -> None:
        self._cond = threading.Condition()
        self._items: list[Event] = []
        self._counter = itertools.count(1)
        self._journal = journal

    @property
    def journal(self) -> Optional["EventJournal"]:
        """Журнал этой очереди (None — сборка без журнала, совместимость
        7.2; EventAPI требует журнал — fail fast)."""
        return self._journal

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
        if self._journal is not None:
            self._journal.record(event)
        return event

    def get(self, timeout: Optional[float] = None) -> Optional[Event]:
        """Взять событие; None — таймаут ожидания (событий не было).

        timeout=None — ждать бесконечно; конечный таймаут — poll-цикл
        dispatcher'а (stop-флаг снаружи).
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


# --- Журнал событий и API событий (задача 7.3) ---

# Лимит кольцевого журнала по умолчанию (записей). Константа, не конфиг:
# события ядра редки (десятки в час — продажи/закрытия/модалки/recover);
# 1000 записей — сутки+ диагностики; требования настраивать лимит нет
# (YAGNI: добавить ключ конфига поздно тривиально).
DEFAULT_JOURNAL_LIMIT = 1000


@dataclass(frozen=True)
class JournalEntry:
    """Запись журнала: событие + момент публикации (диагностика).

    timestamp — время из часов журнала (в проде реальное time.time;
    в тестах инъекция) — диагностика потоковых гонок, не часть контракта
    payload'ов.
    """

    event: Event
    timestamp: float


class EventJournal:
    """Кольцевой журнал событий ядра (spec: журнал ведётся ядром).

    Ведётся на push очереди (до потребления): полный след публикаций
    с вытеснением старых записей сверх лимита. Потокобезопасен; Condition
    — точка ожидания pull-формы (wait_event ищет по записям, поэтому
    события, забранные dispatcher'ом из очереди, pull'у не теряются).
    """

    def __init__(self, limit: int = DEFAULT_JOURNAL_LIMIT, *,
                 clock: Callable[[], float] = time.time) -> None:
        if limit <= 0:
            raise ValueError(f"лимит журнала должен быть > 0 "
                             f"(получено {limit!r})")
        self._limit = limit
        self._clock = clock
        self._cond = threading.Condition()
        self._entries: deque[JournalEntry] = deque()

    @property
    def limit(self) -> int:
        """Глубина журнала (записей; старые вытесняются)."""
        return self._limit

    def record(self, event: Event) -> None:
        """Записать событие (вызывается EventQueue.push)."""
        with self._cond:
            self._entries.append(JournalEntry(event=event,
                                              timestamp=self._clock()))
            while len(self._entries) > self._limit:
                self._entries.popleft()
            self._cond.notify_all()

    def get_events(self, since_seq: int = 0) -> Tuple[JournalEntry, ...]:
        """Записи с seq >= since_seq (0 — весь журнал), по порядку seq."""
        with self._cond:
            return tuple(entry for entry in self._entries
                         if entry.event.seq >= since_seq)

    def wait(self, event_type: str, after_seq: int, timeout: Optional[float],
             *, clock: Callable[[], float] = time.monotonic
             ) -> Optional[Event]:
        """Блокировка до первого события типа с seq > after_seq; таймаут —
        None. Ожидание по записям журнала (события живут здесь до
        вытеснения, независимо от dispatcher'а)."""
        with self._cond:
            if timeout is None:
                while True:
                    found = self._find(event_type, after_seq)
                    if found is not None:
                        return found
                    self._cond.wait()
            deadline = clock() + timeout
            while True:
                found = self._find(event_type, after_seq)
                if found is not None:
                    return found
                remaining = deadline - clock()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)

    def _find(self, event_type: str, after_seq: int) -> Optional[Event]:
        """Первая запись типа со свежим seq (под lock'ом вызывающего)."""
        for entry in self._entries:
            if entry.event.seq > after_seq and entry.event.type == event_type:
                return entry.event
        return None


class EventSubscription:
    """Регистрация push-подписки: отмена (idempotent) + контекстный
    менеджер (with-подписка с гарантированной отпиской)."""

    def __init__(self, dispatcher: Any, event_type: str,
                 callback: Callable[[Event], None]) -> None:
        self._dispatcher = dispatcher
        self._event_type = event_type
        self._callback = callback
        self._active = True

    def unsubscribe(self) -> bool:
        """Отписать callback; True — отписка выполнена, False — уже
        отписан (idempotent)."""
        if not self._active:
            return False
        self._active = False
        self._dispatcher.unsubscribe(self._event_type, self._callback)
        return True

    def __enter__(self) -> "EventSubscription":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.unsubscribe()


class EventAPI:
    """События двух форм для стратегии поверх EventQueue + EventDispatcher.

    PUSH (on): подписка callback'ов, доставка — dispatcher-поток (R14,
    семантика — в докстринге модуля); отмена — subscription.unsubscribe().
    PULL (wait_event): блокирующее ожидание строго из потока стратегии
    (owner_thread; чужой поток — RuntimeError). ЖУРНАЛ (get_events):
    записи ядра с seq >= since_seq. PUSH ВНЕШНИХ ИСТОЧНИКОВ (push):
    единый канал публикации (killed от kill switch 8.1).

    owner_thread=None — проверка потока не выполняется (сборки без
    владельца, тесты); движок всегда передаёт поток стратегии.
    """

    def __init__(self, queue: EventQueue, dispatcher: Any = None, *,
                 owner_thread: Optional[threading.Thread] = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._queue = queue
        self._dispatcher = dispatcher
        self._owner = owner_thread
        self._clock = clock

    # ------------------------------------------------------------------
    # push-форма

    def on(self, event_type: str, callback: Callable[[Event], None]
           ) -> EventSubscription:
        """Подписать callback на тип события; доставка — dispatcher-поток
        (R14). Callback НЕ должен вызывать блокирующий API (thread-owner).
        Подписка без dispatcher'а — RuntimeError (сборка движка всегда
        имеет его)."""
        if event_type not in EVENT_TYPES:
            raise ValueError(
                f"неизвестный тип события {event_type!r}: допустимы "
                f"{sorted(EVENT_TYPES)}")
        if self._dispatcher is None:
            raise RuntimeError(
                "EventAPI: подписка on() невозможна — сборка без "
                "event dispatcher'а (push-форма требует доставщика)")
        self._dispatcher.subscribe(event_type, callback)
        return EventSubscription(self._dispatcher, event_type, callback)

    # ------------------------------------------------------------------
    # pull-форма

    def wait_event(self, event_type: str, timeout: Optional[float] = None
                   ) -> Optional[Event]:
        """Блокировка до события ЗАДАННОГО типа, опубликованного ПОСЛЕ
        вызова; None по таймауту. Строго из потока стратегии (R14)."""
        if event_type not in EVENT_TYPES:
            raise ValueError(
                f"неизвестный тип события {event_type!r}: допустимы "
                f"{sorted(EVENT_TYPES)}")
        self._check_thread()
        journal = self._require_journal()
        return journal.wait(event_type, after_seq=self._queue_tip(),
                            timeout=timeout, clock=self._clock)

    def _queue_tip(self) -> int:
        """seq на момент вызова: события с seq <= tip — «прошлое»,
        wait_event их не возвращает (только свежие)."""
        entries = self._require_journal().get_events()
        return entries[-1].event.seq if entries else 0

    def _check_thread(self) -> None:
        """Pull-форма — строго из потока стратегии (spec core-api-events,
        R14): блокирующий вызов из чужого потока — RuntimeError."""
        if self._owner is not None \
                and threading.current_thread() is not self._owner:
            raise RuntimeError(
                f"EventAPI: wait_event вызван из чужого потока "
                f"({threading.current_thread().name!r}): pull-форма "
                f"исполняется строго в потоке стратегии (R14)")

    # ------------------------------------------------------------------
    # журнал

    def get_events(self, since_seq: int = 0) -> Tuple[JournalEntry, ...]:
        """Записи журнала с seq >= since_seq (0 — весь журнал)."""
        return self._require_journal().get_events(since_seq=since_seq)

    def _require_journal(self) -> EventJournal:
        journal = self._queue.journal
        if journal is None:
            raise RuntimeError(
                "EventAPI: очередь без журнала (EventQueue(journal=...) "
                "не передан): pull-форма и журнал недоступны — fail fast "
                "(сборка движка всегда ведёт журнал)")
        return journal

    # ------------------------------------------------------------------
    # публикация (единый канал внешних источников)

    def push(self, event_type: str, payload: Any = None) -> Event:
        """Опубликовать событие внешнего источника (killed 8.1): тот же
        канал, что и у движка — журнал и подписчики видят его наряду с
        событиями observer'а."""
        return self._queue.push(event_type, payload)
