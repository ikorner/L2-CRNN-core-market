"""События двух форм — push (on) и pull (wait_event), журнал событий,
payload по API-справочнику (задача 7.3, spec core-api-events «События в двух
формах», «Журнал событий ведётся ядром»; ruling R14 — ратификация отклонения
от design 8, см. отчёт и правку spec.md).

Контракт, проверяемый здесь:
- EventJournal: кольцевой буфер с лимитом (старые вытесняются), timestamp из
  инъецированных часов, get_events(since_seq);
- EventQueue(journal=...): журнал пишется на push (до потребления: pull не
  теряет события, забранные dispatcher'ом);
- EventAPI.on: доставка в dispatcher-потоке, отписка (idempotent), изоляция
  исключений callback'а;
- EventAPI.wait_event: блокируется до события ЗАДАННОГО типа (другой тип не
  будит), таймаут -> None, события, опубликованные ДО вызова, не возвращаются,
  чужой поток -> RuntimeError (pull — строго из потока стратегии, R14),
  неизвестный тип -> ValueError;
- EventAPI.push: внешним источникам (killed 8.1) — единый канал;
- payload по справочнику docs/API.md: sold -> SoldPayload (поля 6.3),
  window_closed -> {"window": str}, modal -> ModalReport (поля 3.6),
  recovered -> {"target", "steps"}, escalation -> {"target", "detail",
  "steps"}; recovered/escalation публикуются api.recover() по итогу цепочки
  (успех -> recovered, эскалация -> escalation, отмена -> событий нет);
- L2CoreAPI: on/wait_event/get_events делегируют EventAPI; сборка без
  событий — fail fast;
- L2CoreBot: event_api создан, подключён к фасаду (api.events), bot.push
  виден в журнале.

Детерминизм (инвариант 7): потоки короткоживущие (join в каждом тесте),
синхронизация через wait_for-предикаты, часы журнала инъецируются.
"""
import threading
import time

import pytest

from l2core.api.engine import EventDispatcher, L2CoreBot
from l2core.api.events import (
    DEFAULT_JOURNAL_LIMIT,
    EVENT_ESCALATION,
    EVENT_KILLED,
    EVENT_MODAL,
    EVENT_RECOVERED,
    EVENT_SOLD,
    EVENT_WINDOW_CLOSED,
    EventAPI,
    EventJournal,
    EventQueue,
)
from l2core.api.hid_owner import OwnerGuardedHid
from l2core.detector.modal import ModalReport
from l2core.scenarios.monitor import SoldPayload

from tests.test_api import Cause, FakeClock, World, make_api
from tests.test_engine import RecordingHid, make_wiring, wait_for

POLL_INTERVAL = 0.01
DISPATCH_INTERVAL = 0.01

SOLD = SoldPayload(sold_count=3, sold_delta=1, revenue=1500,
                   revenue_delta=500, snapshot_id="abc123")


def make_journaled_queue(**journal_kwargs):
    """Очередь с журналом (сборка движка: журнал обязателен)."""
    return EventQueue(journal=EventJournal(**journal_kwargs))


def make_event_api(queue, **kwargs):
    dispatcher = kwargs.pop("dispatcher", EventDispatcher(
        queue, poll_interval=DISPATCH_INTERVAL))
    return EventAPI(queue, dispatcher, **kwargs)


# --- EventJournal ---


class TestEventJournal:
    def test_records_events_with_timestamp(self):
        clock = FakeClock()
        journal = EventJournal(clock=lambda: clock.monotonic())
        queue = EventQueue(journal=journal)
        event = queue.push(EVENT_SOLD, SOLD)

        entries = journal.get_events()
        assert len(entries) == 1
        assert entries[0].event is event
        assert entries[0].timestamp == clock.now

    def test_ring_buffer_drops_oldest(self):
        journal = EventJournal(limit=3)
        queue = EventQueue(journal=journal)
        for index in range(5):
            queue.push(EVENT_SOLD, index)

        entries = journal.get_events()
        assert [entry.event.payload for entry in entries] == [2, 3, 4]
        assert [entry.event.seq for entry in entries] == [3, 4, 5]

    def test_since_seq_filters(self):
        journal = EventJournal()
        queue = EventQueue(journal=journal)
        events = [queue.push(EVENT_SOLD, index) for index in range(4)]

        entries = journal.get_events(since_seq=events[2].seq)
        assert [entry.event for entry in entries] == events[2:]

    def test_default_limit_is_documented_constant(self):
        journal = EventJournal()
        queue = EventQueue(journal=journal)
        for index in range(DEFAULT_JOURNAL_LIMIT + 10):
            queue.push(EVENT_SOLD, index)
        assert len(journal.get_events()) == DEFAULT_JOURNAL_LIMIT

    def test_no_journal_no_recording(self):
        queue = EventQueue()
        queue.push(EVENT_SOLD, SOLD)   # журнал необязателен (совместимость 7.2)
        assert queue.journal is None


# --- EventAPI: push-форма ---


class TestEventAPIPush:
    def _make(self):
        queue = make_journaled_queue()
        api = make_event_api(queue)
        dispatcher = api._dispatcher
        thread = threading.Thread(target=dispatcher.run, name="test-dispatcher")
        thread.start()

        def stop():
            dispatcher.request_stop()
            thread.join(timeout=2.0)
            assert not thread.is_alive(), "dispatcher-поток не завершился"

        return api, queue, stop

    def test_on_delivers_payload_in_dispatcher_thread(self):
        api, queue, stop = self._make()
        received = []
        api.on(EVENT_MODAL, received.append)
        try:
            modal = ModalReport(detected=True, state="modal.unknown",
                                region=(10, 10, 40, 40),
                                reason="кандидат", snapshot_path=None)
            queue.push(EVENT_MODAL, modal)
            assert wait_for(lambda: len(received) == 1)
            # Payload по справочнику: ModalReport целиком (3.6).
            assert received[0].payload is modal
            assert received[0].payload.state == "modal.unknown"
        finally:
            stop()

    def test_unsubscribe_stops_delivery_idempotent(self):
        api, queue, stop = self._make()
        received = []
        subscription = api.on(EVENT_SOLD, received.append)
        assert subscription.unsubscribe() is True
        assert subscription.unsubscribe() is False   # idempotent
        try:
            queue.push(EVENT_SOLD, SOLD)
            time.sleep(0.05)
            assert received == []
        finally:
            stop()

    def test_subscription_context_manager(self):
        api, queue, stop = self._make()
        received = []
        with api.on(EVENT_SOLD, received.append):
            queue.push(EVENT_SOLD, SOLD)
            assert wait_for(lambda: len(received) == 1)
        queue.push(EVENT_SOLD, SOLD)
        time.sleep(0.05)
        assert len(received) == 1
        stop()

    def test_callback_exception_isolated(self):
        api, queue, stop = self._make()
        calls = []

        def bad(event):
            raise RuntimeError("сломанный callback")

        api.on(EVENT_SOLD, bad)
        api.on(EVENT_SOLD, calls.append)
        try:
            queue.push(EVENT_SOLD, SOLD)
            assert wait_for(lambda: len(calls) == 1)
        finally:
            stop()

    def test_on_unknown_type_rejected(self):
        api, queue, stop = self._make()
        try:
            with pytest.raises(ValueError, match="unknown_type"):
                api.on("unknown_type", lambda event: None)
        finally:
            stop()


# --- EventAPI: pull-форма ---


class TestEventAPIPull:
    def test_wait_event_returns_matching_type(self):
        queue = make_journaled_queue()
        api = EventAPI(queue, None)

        def producer():
            # пауза: wait_event должен захватить tip ДО публикации
            # (события, опубликованные до вызова, pull не возвращает).
            time.sleep(0.05)
            queue.push(EVENT_MODAL, "иное событие")
            queue.push(EVENT_SOLD, SOLD)

        thread = threading.Thread(target=producer)
        thread.start()
        event = api.wait_event(EVENT_SOLD, timeout=5.0)
        thread.join(timeout=2.0)
        assert event is not None
        assert event.type == EVENT_SOLD
        assert event.payload is SOLD

    def test_wait_event_timeout_returns_none(self):
        queue = make_journaled_queue()
        api = EventAPI(queue, None)
        started = time.monotonic()
        assert api.wait_event(EVENT_SOLD, timeout=0.05) is None
        assert time.monotonic() - started < 1.0   # реально вернулся по таймауту

    def test_wait_event_ignores_events_published_before_call(self):
        queue = make_journaled_queue()
        api = EventAPI(queue, None)
        queue.push(EVENT_SOLD, SOLD)   # до вызова: не возвращается
        assert api.wait_event(EVENT_SOLD, timeout=0.05) is None

        def producer():
            time.sleep(0.05)   # wait_event уже ждёт: захватил tip
            queue.push(EVENT_SOLD, "свежее")

        thread = threading.Thread(target=producer)
        thread.start()
        assert api.wait_event(EVENT_SOLD, timeout=5.0).payload == "свежее"
        thread.join(timeout=2.0)

    def test_wait_event_finds_event_dispatcher_consumed(self):
        """Журнал пишется на push (до потребления): pull находит событие,
        даже если dispatcher уже забрал его из очереди."""
        queue = make_journaled_queue()
        api = make_event_api(queue)
        dispatcher = api._dispatcher
        received = []
        dispatcher.subscribe(EVENT_SOLD, received.append)
        thread = threading.Thread(target=dispatcher.run, name="test-dispatcher")
        thread.start()

        def producer():
            time.sleep(0.05)   # wait_event уже ждёт
            queue.push(EVENT_SOLD, SOLD)

        producer_thread = threading.Thread(target=producer)
        producer_thread.start()
        # Pull ждёт в потоке стратегии; журнал не теряет событие, даже
        # когда dispatcher забирает его из очереди.
        event = api.wait_event(EVENT_SOLD, timeout=5.0)
        assert event is not None and event.payload is SOLD
        try:
            producer_thread.join(timeout=2.0)
            # Dispatcher обязательно забрал событие из очереди (доставка
            # push-подписчику) — pull при этом его не потерял.
            assert wait_for(lambda: len(received) == 1)
        finally:
            producer_thread.join(timeout=2.0)
            dispatcher.request_stop()
            thread.join(timeout=2.0)
            assert not thread.is_alive()

    def test_wait_event_foreign_thread_rejected(self):
        queue = make_journaled_queue()
        api = EventAPI(queue, None, owner_thread=threading.current_thread())
        outcome = {}

        def foreign():
            try:
                api.wait_event(EVENT_SOLD, timeout=5.0)
            except RuntimeError as exc:
                outcome["error"] = exc

        thread = threading.Thread(target=foreign)
        thread.start()
        thread.join(timeout=2.0)
        assert isinstance(outcome.get("error"), RuntimeError)
        assert "поток" in str(outcome["error"])

    def test_wait_event_unknown_type_rejected(self):
        queue = make_journaled_queue()
        api = EventAPI(queue, None)
        with pytest.raises(ValueError, match="unknown_type"):
            api.wait_event("unknown_type", timeout=0.01)


# --- push: единый канал внешних источников ---


class TestEventAPIPushExternal:
    def test_push_external_source_goes_to_journal(self):
        queue = make_journaled_queue()
        api = EventAPI(queue, None)

        event = api.push(EVENT_KILLED, {"reason": "kill switch"})

        assert event.seq == 1
        assert [entry.event for entry in queue.journal.get_events()] == [event]


# --- payload по справочнику (источники ядра) ---


class TestPayloadContract:
    def test_sold_payload_fields(self):
        # Справочник docs/API.md: SoldPayload 6.3.
        assert SOLD.sold_count == 3
        assert SOLD.sold_delta == 1
        assert SOLD.revenue == 1500
        assert SOLD.revenue_delta == 500
        assert SOLD.snapshot_id == "abc123"
        revenue_less = SoldPayload(sold_count=3, sold_delta=1, revenue=None,
                                   revenue_delta=None, snapshot_id="x")
        assert revenue_less.revenue is None   # честная деградация 6.3

    def test_window_closed_payload_fields(self):
        queue = EventQueue()
        event = queue.push(EVENT_WINDOW_CLOSED, {"window": "market"})
        assert set(event.payload) == {"window"}

    def test_modal_payload_fields(self):
        modal = ModalReport(detected=True, state="modal.unknown",
                            region=(10, 10, 40, 40), reason="r",
                            snapshot_path=None)
        for field in ("detected", "state", "region", "reason",
                      "snapshot_path"):
            assert hasattr(modal, field)


# --- L2CoreAPI: события + recovered/escalation от recover ---


class TestApiEvents:
    def test_on_and_wait_event_delegated(self):
        queue = make_journaled_queue()
        api = make_api(events=make_event_api(queue))
        received = []
        api.on(EVENT_SOLD, received.append)
        dispatcher = api.events._dispatcher
        thread = threading.Thread(target=dispatcher.run, name="test-dispatcher")
        thread.start()
        try:
            queue.push(EVENT_SOLD, SOLD)
            assert wait_for(lambda: len(received) == 1)

            def producer():
                time.sleep(0.05)   # wait_event уже ждёт
                queue.push(EVENT_MODAL, None)

            producer_thread = threading.Thread(target=producer)
            producer_thread.start()
            assert api.wait_event(EVENT_MODAL, timeout=5.0) is not None
            producer_thread.join(timeout=2.0)
            # sold был ДО вызова: pull его не возвращает (свежие только).
            assert api.wait_event(EVENT_SOLD, timeout=0.05) is None
        finally:
            dispatcher.request_stop()
            thread.join(timeout=2.0)
            assert not thread.is_alive()

    def test_get_events_returns_journal_entries(self):
        queue = make_journaled_queue()
        api = make_api(events=make_event_api(queue))
        event = queue.push(EVENT_SOLD, SOLD)
        entries = api.get_events(since_seq=event.seq)
        assert [entry.event for entry in entries] == [event]

    def test_events_absent_fail_fast(self):
        api = make_api()
        with pytest.raises(Exception, match="событий"):
            api.on(EVENT_SOLD, lambda event: None)
        with pytest.raises(Exception, match="событий"):
            api.wait_event(EVENT_SOLD, timeout=0.01)
        with pytest.raises(Exception, match="событий"):
            api.get_events()

    def test_recover_success_publishes_recovered(self):
        queue = make_journaled_queue()
        world, clock = World(), FakeClock()
        world.goal = True
        api = make_api(world, clock=clock, events=make_event_api(queue))

        result = api.recover(Cause(), "shop_sell")

        assert result.success
        recovered = [entry.event for entry in queue.journal.get_events()
                     if entry.event.type == EVENT_RECOVERED]
        assert len(recovered) == 1
        # Справочник: {"target": str, "steps": tuple имён шагов}.
        assert recovered[0].payload["target"] == "shop_sell"
        assert recovered[0].payload["steps"] == ()

    def test_recover_escalation_publishes_escalation(self):
        queue = make_journaled_queue()
        world, clock = World(), FakeClock()
        world.current = "market"   # чужое окно; закрытие не работает ->
                                   # цепочка исчерпывается -> эскалация
        api = make_api(world, clock=clock, events=make_event_api(queue))

        result = api.recover(Cause(), "shop_sell", pause_timeout=0.01)

        assert result.escalated
        escalations = [entry.event for entry in queue.journal.get_events()
                       if entry.event.type == EVENT_ESCALATION]
        assert len(escalations) == 1
        # Справочник: {"target", "detail", "steps"}.
        assert escalations[0].payload["target"] == "shop_sell"
        assert "detail" in escalations[0].payload
        assert isinstance(escalations[0].payload["steps"], tuple)
        assert not [entry for entry in queue.journal.get_events()
                    if entry.event.type == EVENT_RECOVERED]

    def test_recover_cancelled_publishes_nothing(self):
        queue = make_journaled_queue()
        world, clock = World(), FakeClock()
        api = make_api(world, clock=clock, cancelled=lambda: True,
                       events=make_event_api(queue))

        result = api.recover(Cause(), "shop_sell")

        assert result.cancelled
        assert queue.journal.get_events() == ()


# --- L2CoreBot: интеграция ---


class TestBotEventAPI:
    def test_event_api_wired_push_visible_in_journal(self):
        bot = L2CoreBot(make_wiring(RecordingHid()),
                        poll_interval=POLL_INTERVAL,
                        dispatch_interval=DISPATCH_INTERVAL)
        received = []
        bot.event_api.on(EVENT_ESCALATION, received.append)
        bot.start()
        try:
            bot.push(EVENT_ESCALATION, {"target": "market", "detail": "x",
                                        "steps": ()})
            assert wait_for(lambda: len(received) == 1)
            # bot.push виден в журнале той же сборки.
            assert [entry.event.payload for entry in bot.event_api.get_events()
                    if entry.event.type == EVENT_ESCALATION] == \
                   [{"target": "market", "detail": "x", "steps": ()}]
        finally:
            shutdown = bot.stop(timeout=2.0)
        assert shutdown.stopped

    def test_api_facade_exposes_events(self):
        bot = L2CoreBot(make_wiring(RecordingHid()),
                        poll_interval=POLL_INTERVAL,
                        dispatch_interval=DISPATCH_INTERVAL)
        assert isinstance(bot.wiring.hid, OwnerGuardedHid)
        # Фасад API событий: тот же объект, что у бота.
        assert bot.api.events is bot.event_api
