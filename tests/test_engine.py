"""Модель потоков: state observer, event dispatcher, HID-owner guard,
lifecycle движка (задача 7.2, spec core-api-events «Модель потоков:
executor, observer, dispatcher»; сценарий «Observer не отправляет действия»).

Контракт, проверяемый здесь:
- EventQueue: FIFO, монотонный seq, доставка между потоками, неизвестный
  тип события отклоняется (fail fast);
- OwnerGuardedHid: HID-команды только из потока-владельца (action executor),
  вызов из чужого потока -> HidOwnershipError, железо не тронуто;
- StateObserver: публикует события sold/window_closed/modal в очередь,
  HID не получает ВООБЩЕ (observer'у hid не передаётся; интеграционно —
  wiring.hid движка после прогона чист), снапшот состояния обновляется;
- EventDispatcher: доставка callbacks НЕ в observer-потоке, порядок
  событий сохранён, исключение callback'а не роняет доставку остальных;
- L2CoreBot: start/stop (join потоков, idempotent stop), повторный
  start — EngineError, контекстный менеджер, api привязан к потоку
  владельца (guard поверх wiring.hid), push внешних источников событий.

Детерминизм (инвариант 7): реальные потоки короткоживущие (интервалы
0.01–0.05 с), join в каждом тесте; синхронизация — через очередь событий и
wait_for-предикаты, без привязки к стенным часам в утверждениях.
"""
import threading
import time

import numpy as np
import pytest

from l2core.adapters.win32 import GameWindow
from l2core.api.engine import (
    EngineError,
    EngineShutdown,
    EventDispatcher,
    L2CoreBot,
    StateObserver,
)
from l2core.api.events import (
    EVENT_ESCALATION,
    EVENT_MODAL,
    EVENT_SOLD,
    EVENT_WINDOW_CLOSED,
    Event,
    EventQueue,
)
from l2core.api.hid_owner import HidOwnershipError, OwnerGuardedHid
from l2core.api.wiring import ApiWiring
from l2core.config import CoreConfig, TimingConfig
from l2core.detector.modal import ModalReport
from l2core.detector.status import (
    STATUS_FOUND,
    STATUS_NOT_FOUND,
    WindowStatus,
)
from l2core.model import UiModel, WindowSpec
from l2core.scenarios.monitor import MonitorPollResult, SoldPayload

POLL_INTERVAL = 0.01   # интервал observer'а/диспетчера в тестах
DISPATCH_INTERVAL = 0.01


def wait_for(predicate, timeout=5.0):
    """Ожидание предиката с таймаутом (синхронизация тестов, не утверждения
    по часам): poll каждые 5 мс до дедлайна."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


class RecordingHid:
    """HID-заглушка: записывает вызовы команд (эффекты не исполняет)."""

    TRAJ_MAX_POINTS = 60

    def __init__(self):
        self.calls = []
        self.version = (2, 0)

    def _rec(self, name, args, kwargs):
        self.calls.append((name, args, kwargs))

    def click(self, *args, **kwargs):
        self._rec("click", args, kwargs)

    def move_rel(self, *args, **kwargs):
        self._rec("move_rel", args, kwargs)

    def mouse_down(self, *args, **kwargs):
        self._rec("mouse_down", args, kwargs)

    def mouse_up(self, *args, **kwargs):
        self._rec("mouse_up", args, kwargs)

    def traj(self, *args, **kwargs):
        self._rec("traj", args, kwargs)

    def scroll(self, *args, **kwargs):
        self._rec("scroll", args, kwargs)

    def key_press(self, *args, **kwargs):
        self._rec("key_press", args, kwargs)

    def key_down(self, *args, **kwargs):
        self._rec("key_down", args, kwargs)

    def key_up(self, *args, **kwargs):
        self._rec("key_up", args, kwargs)

    def key_combo(self, *args, **kwargs):
        self._rec("key_combo", args, kwargs)

    def type_char(self, *args, **kwargs):
        self._rec("type_char", args, kwargs)

    def release_all(self):
        self.calls.append(("release_all", (), {}))

    def is_connected(self):
        return True


class FakeScreen:
    """Экран с фиксированным кадром."""

    def __init__(self):
        self.frame = np.zeros((64, 64, 3), dtype=np.uint8)
        self.grabs = 0

    def grab(self):
        self.grabs += 1
        return self.frame


class ScriptedWindows:
    """current_window по сценарию (последнее значение повторяется)."""

    def __init__(self, sequence):
        self._sequence = list(sequence)
        self._index = 0
        self.calls = 0

    def __call__(self, frame):
        self.calls += 1
        value = self._sequence[min(self._index, len(self._sequence) - 1)]
        self._index += 1
        return value


class ScriptedMonitor:
    """Монитор-заглушка: poll() отдаёт скриптованные результаты."""

    def __init__(self, results):
        self._results = list(results)
        self._index = 0
        self.polls = 0

    def poll(self):
        self.polls += 1
        if self._index < len(self._results):
            result = self._results[self._index]
            self._index += 1
            return result
        return MonitorPollResult()


class ScriptedModal:
    """Перехватчик-заглушка: срабатывает, пока включён флаг."""

    def __init__(self, detected):
        self._detected = detected
        self.calls = 0

    def check(self, frame, game, anchors):
        self.calls += 1
        if self._detected.is_set():
            return ModalReport(
                detected=True, state="modal.unknown",
                region=(10, 10, 40, 40), reason="тестовый кандидат",
                snapshot_path=None)
        return ModalReport(
            detected=False, state=None, region=None, reason="нет",
            snapshot_path=None)


SOLD = SoldPayload(sold_count=3, sold_delta=1, revenue=1500,
                   revenue_delta=500, snapshot_id="abc123")


def make_observer(queue, *, windows=(None,), monitor=None, modal=None,
                  cancelled=None):
    """Observer на скриптованных провайдерах (interval тестовый)."""
    return StateObserver(
        screen=FakeScreen(),
        game_window=lambda: GameWindow.found(1, (0, 0, 64, 64)),
        current_window=ScriptedWindows(windows),
        find_anchor=lambda frame, window_id: None,
        window_status=lambda frame, window_id: WindowStatus(
            status=STATUS_FOUND if window_id is not None
            else STATUS_NOT_FOUND,
            reason=None, anchor=None, content_loaded=True,
            content_detail="ok"),
        queue=queue, interval=POLL_INTERVAL, monitor=monitor, modal=modal,
        cancelled=cancelled,
    )


def run_observer(observer):
    """Запустить observer в потоке; вернуть (thread, stop+join)."""
    thread = threading.Thread(target=observer.run, name="test-observer")
    thread.start()

    def stop():
        observer.request_stop()
        thread.join(timeout=2.0)
        assert not thread.is_alive(), "observer-поток не завершился за 2 с"

    return thread, stop


def make_model():
    return UiModel(
        path="market", window=WindowSpec(id="market", title="m",
                                         size=(100, 100)),
        elements={}, states={}, transitions=(), ocr_regions={},
        table_geometry=None)


def make_wiring(hid, *, current_window=None, config=None):
    model = make_model()
    return ApiWiring(
        config=config or CoreConfig(
            timing=TimingConfig(poll_interval=POLL_INTERVAL)),
        screen=FakeScreen(),
        hid=hid,
        ocr=None,
        market=model,
        shop_buy=model,
        shop_sell=model,
        hud_model=model,
        cursor_pos=lambda: (0, 0),
        activate=lambda: None,
        find_anchor=lambda frame, window_id: None,
        window_status=lambda frame, window_id: WindowStatus(
            status=STATUS_NOT_FOUND, reason=None, anchor=None,
            content_loaded=None, content_detail=None),
        run_checks=None,
        current_window=current_window or (lambda frame: None),
        close_point=lambda window_id, frame: None,
        hud_locator=lambda element, variant, frame: None,
        goal_reached=lambda target: (lambda frame: False),
        game_window=lambda: GameWindow.found(1, (0, 0, 64, 64)),
        ocr_health=lambda: True,
        hid_health=lambda: True,
    )


# --- EventQueue ---


class TestEventQueue:
    def test_push_assigns_monotonic_seq_fifo(self):
        queue = EventQueue()
        first = queue.push(EVENT_SOLD, {"a": 1})
        second = queue.push(EVENT_MODAL, "x")
        assert (first.seq, second.seq) == (1, 2)
        assert queue.get(timeout=0) is first
        assert queue.get(timeout=0) is second

    def test_get_timeout_returns_none(self):
        queue = EventQueue()
        started = time.monotonic()
        assert queue.get(timeout=0.02) is None
        assert time.monotonic() - started < 1.0   # реально вернулся по таймауту

    def test_unknown_event_type_rejected(self):
        queue = EventQueue()
        with pytest.raises(ValueError, match="unknown_type"):
            queue.push("unknown_type")

    def test_concurrent_push_unique_seq(self):
        queue = EventQueue()
        seqs = []
        seqs_lock = threading.Lock()

        def worker():
            local = []
            for _ in range(50):
                local.append(queue.push(EVENT_SOLD).seq)
            with seqs_lock:
                seqs.extend(local)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2.0)
            assert not thread.is_alive()
        assert len(seqs) == 200
        assert len(set(seqs)) == 200


# --- OwnerGuardedHid ---


class TestOwnerGuardedHid:
    def test_delegates_in_owner_thread(self):
        hid = RecordingHid()
        guarded = OwnerGuardedHid(hid)
        guarded.click(1, 1, 50)
        guarded.move_rel(3, -2)
        guarded.release_all()
        assert [call[0] for call in hid.calls] == [
            "click", "move_rel", "release_all"]

    def test_rejects_foreign_thread(self):
        hid = RecordingHid()
        guarded = OwnerGuardedHid(hid)
        outcome = {}

        def foreign():
            try:
                guarded.click(1, 1, 50)
            except HidOwnershipError as exc:
                outcome["error"] = exc

        thread = threading.Thread(target=foreign)
        thread.start()
        thread.join(timeout=2.0)
        assert not thread.is_alive()
        assert isinstance(outcome.get("error"), HidOwnershipError)
        assert hid.calls == []   # железо не тронуто

    def test_explicit_owner_thread(self):
        hid = RecordingHid()
        allowed = threading.Event()
        outcome = {}

        def owner():
            guarded = OwnerGuardedHid(hid, owner_thread=threading.current_thread())
            allowed.wait(timeout=2.0)
            try:
                guarded.key_press(0x1E)
            except HidOwnershipError as exc:
                outcome["error"] = exc

        thread = threading.Thread(target=owner)
        thread.start()
        # Из потока конструктора (не владельца) — отказ.
        with pytest.raises(HidOwnershipError):
            OwnerGuardedHid(hid, owner_thread=thread).key_press(0x1E)
        allowed.set()
        thread.join(timeout=2.0)
        assert not thread.is_alive()
        assert "error" not in outcome
        assert [call[0] for call in hid.calls] == ["key_press"]

    def test_attribute_passthrough(self):
        hid = RecordingHid()
        guarded = OwnerGuardedHid(hid)
        assert guarded.TRAJ_MAX_POINTS == 60
        assert guarded.version == (2, 0)

    def test_no_public_raw_hid_access(self):
        """Публичного люка к сырому HidDevice без проверки владельца нет:
        аксессора `inner` нет, команды вне явного списка недоступны
        (__getattr__ fail closed). Сырой адаптер остаётся только под
        приватным именем `_hid` — конвенция Python, вне поддерживаемой
        границы guard'а (зафиксовано в докстринге hid_owner.py)."""
        hid = RecordingHid()
        guarded = OwnerGuardedHid(hid)
        assert "inner" not in vars(type(guarded))
        with pytest.raises(AttributeError):
            guarded.inner

    def test_unlisted_method_blocked_by_getattr(self):
        """Метод, появившийся у оборачиваемого объекта вне явного списка
        guard'а, НЕ доступен через __getattr__-проскок: обхода проверки
        владельца нет (fail closed)."""

        class ExtendedHid(RecordingHid):
            def future_command(self):
                self.calls.append(("future_command", (), {}))

        hid = ExtendedHid()
        guarded = OwnerGuardedHid(hid)
        with pytest.raises(AttributeError, match="future_command"):
            guarded.future_command()
        assert hid.calls == []   # железо не тронуто

    def test_private_names_blocked_by_getattr(self):
        """Приватные атрибуты оборачиваемого объекта снаружи недоступны
        (доступ к ним — вне поддерживаемой границы guard'а)."""

        class WithPrivate(RecordingHid):
            def __init__(self):
                super().__init__()
                self._serial = object()

        hid = WithPrivate()
        guarded = OwnerGuardedHid(hid)
        with pytest.raises(AttributeError):
            guarded._serial


class TestHidOwnerContract:
    """Guard-тест паритета guard'а и вендорного HidDevice: публичные методы
    вендора ⊆ явный список обёрнутых команд. Новая команда в вендоре не
    проскочит молча — расхождение падает с именем метода."""

    # Явный список команд, обёрнутых OwnerGuardedHid (дублирует список
    # методов guard'а намеренно: расхождение списков — тоже находка).
    _WRAPPED = frozenset({
        "connect", "close", "is_connected", "ping", "get_version",
        "release_all", "move_rel", "click", "mouse_down", "mouse_up",
        "traj", "scroll", "key_press", "key_down", "key_up", "key_combo",
        "type_char",
    })

    def test_vendor_public_methods_match_guard_list(self):
        from l2core.adapters.hid import HidDevice

        vendor_public = frozenset(
            name for name, value in vars(HidDevice).items()
            if not name.startswith("_") and callable(value))
        missing = vendor_public - self._WRAPPED
        extra = self._WRAPPED - vendor_public
        assert not missing, (
            f"публичные методы вендорного HidDevice вне guard'а "
            f"(обход проверки владельца): {sorted(missing)} — добавьте "
            f"обёртку с _check() в OwnerGuardedHid")
        assert not extra, (
            f"guard оборачивает методы, которых нет в вендоре "
            f"(вендор изменён?): {sorted(extra)}")

    def test_each_vendor_method_wrapped_explicitly(self):
        """Каждая команда вендора — собственный метод OwnerGuardedHid
        (проверка владельца), а не __getattr__-проброс."""
        for name in sorted(self._WRAPPED):
            assert name in vars(OwnerGuardedHid), (
                f"{name}: не обёрнут явно в OwnerGuardedHid — доступен "
                f"только через __getattr__ без проверки владельца")


# --- EventDispatcher ---


class TestEventDispatcher:
    def _run_dispatcher(self, dispatcher):
        thread = threading.Thread(
            target=dispatcher.run, name="test-dispatcher")
        thread.start()

        def stop():
            dispatcher.request_stop()
            thread.join(timeout=2.0)
            assert not thread.is_alive(), (
                "dispatcher-поток не завершился за 2 с")

        return thread, stop

    def test_delivers_in_order(self):
        queue = EventQueue()
        dispatcher = EventDispatcher(queue, poll_interval=DISPATCH_INTERVAL)
        received = []
        dispatcher.subscribe(EVENT_SOLD, received.append)
        thread, stop = self._run_dispatcher(dispatcher)
        try:
            first = queue.push(EVENT_SOLD, SOLD)
            second = queue.push(EVENT_SOLD, SOLD)
            assert wait_for(lambda: len(received) == 2)
            assert [event.seq for event in received] == [first.seq, second.seq]
            assert all(isinstance(event, Event) for event in received)
            assert received[0].payload is SOLD
        finally:
            stop()
        assert not thread.is_alive()

    def test_callback_exception_isolated(self):
        queue = EventQueue()
        dispatcher = EventDispatcher(queue, poll_interval=DISPATCH_INTERVAL)
        calls = []

        def bad(event):
            raise RuntimeError("сломанный callback")

        dispatcher.subscribe(EVENT_SOLD, bad)
        dispatcher.subscribe(EVENT_SOLD, calls.append)
        thread, stop = self._run_dispatcher(dispatcher)
        try:
            queue.push(EVENT_SOLD, SOLD)
            assert wait_for(lambda: len(calls) == 1)
        finally:
            stop()
        # Поток завершился штатно: исключение callback'а не роняет цикл.

    def test_unsubscribe_stops_delivery(self):
        queue = EventQueue()
        dispatcher = EventDispatcher(queue, poll_interval=DISPATCH_INTERVAL)
        received = []
        dispatcher.subscribe(EVENT_SOLD, received.append)
        dispatcher.unsubscribe(EVENT_SOLD, received.append)
        thread, stop = self._run_dispatcher(dispatcher)
        try:
            queue.push(EVENT_SOLD, SOLD)
            time.sleep(0.05)
            assert received == []
        finally:
            stop()

    def test_stop_joins_thread(self):
        queue = EventQueue()
        dispatcher = EventDispatcher(queue, poll_interval=DISPATCH_INTERVAL)
        thread, stop = self._run_dispatcher(dispatcher)
        stop()


# --- StateObserver ---


class TestStateObserver:
    def test_publishes_window_closed(self):
        queue = EventQueue()
        observer = make_observer(queue, windows=("market", None))
        _, stop = run_observer(observer)
        try:
            assert wait_for(lambda: len(queue) >= 1)
            event = queue.get(timeout=0)
            assert event.type == EVENT_WINDOW_CLOSED
            assert event.payload == {"window": "market"}
            assert observer.snapshot.window is None
            assert observer.snapshot.status == STATUS_NOT_FOUND
            assert observer.iterations >= 2
        finally:
            stop()

    def test_stable_state_publishes_nothing(self):
        queue = EventQueue()
        observer = make_observer(queue, windows=(None,))
        _, stop = run_observer(observer)
        try:
            assert wait_for(lambda: observer.iterations >= 3)
            assert len(queue) == 0
        finally:
            stop()

    def test_publishes_sold_from_monitor(self):
        queue = EventQueue()
        monitor = ScriptedMonitor([
            MonitorPollResult(sold=SOLD),
            MonitorPollResult(),
        ])
        observer = make_observer(queue, monitor=monitor)
        _, stop = run_observer(observer)
        try:
            assert wait_for(lambda: len(queue) >= 1)
            event = queue.get(timeout=0)
            assert event.type == EVENT_SOLD
            assert event.payload is SOLD
        finally:
            stop()

    def test_publishes_modal(self):
        queue = EventQueue()
        detected = threading.Event()
        modal = ScriptedModal(detected)
        observer = make_observer(queue, modal=modal)
        _, stop = run_observer(observer)
        try:
            detected.set()
            assert wait_for(lambda: len(queue) >= 1)
            event = queue.get(timeout=0)
            assert event.type == EVENT_MODAL
            assert event.payload.state == "modal.unknown"
        finally:
            stop()

    def test_cancelled_stops_before_first_frame(self):
        queue = EventQueue()
        screen = FakeScreen()
        observer = StateObserver(
            screen=screen,
            game_window=lambda: GameWindow.found(1, (0, 0, 64, 64)),
            current_window=ScriptedWindows(("market",)),
            find_anchor=lambda frame, window_id: None,
            window_status=lambda frame, window_id: WindowStatus(
                status=STATUS_FOUND, reason=None, anchor=None,
                content_loaded=True, content_detail="ok"),
            queue=queue, interval=POLL_INTERVAL,
            cancelled=lambda: True,
        )
        thread, stop = run_observer(observer)
        stop()
        assert screen.grabs == 0   # отмена до чтения кадра (паттерн 6.3)
        assert observer.iterations == 0


# --- L2CoreBot ---


class TestL2CoreBot:
    def test_start_stop_joins_threads_and_hid_untouched(self):
        hid = RecordingHid()
        bot = L2CoreBot(make_wiring(hid),
                        poll_interval=POLL_INTERVAL,
                        dispatch_interval=DISPATCH_INTERVAL)
        bot.start()
        assert bot.observer_thread.is_alive()
        assert bot.dispatcher_thread.is_alive()
        shutdown = bot.stop(timeout=2.0)
        assert isinstance(shutdown, EngineShutdown)
        assert shutdown.stopped
        assert not bot.observer_thread.is_alive()
        assert not bot.dispatcher_thread.is_alive()
        assert hid.calls == []   # «Observer не отправляет действия»

    def test_stop_idempotent(self):
        bot = L2CoreBot(make_wiring(RecordingHid()),
                        poll_interval=POLL_INTERVAL,
                        dispatch_interval=DISPATCH_INTERVAL)
        bot.start()
        assert bot.stop(timeout=2.0).stopped
        again = bot.stop(timeout=2.0)
        assert again.stopped

    def test_start_twice_rejected(self):
        bot = L2CoreBot(make_wiring(RecordingHid()),
                        poll_interval=POLL_INTERVAL,
                        dispatch_interval=DISPATCH_INTERVAL)
        bot.start()
        try:
            with pytest.raises(EngineError, match="уже запущен"):
                bot.start()
        finally:
            bot.stop(timeout=2.0)

    def test_restart_after_stop_rejected(self):
        bot = L2CoreBot(make_wiring(RecordingHid()),
                        poll_interval=POLL_INTERVAL,
                        dispatch_interval=DISPATCH_INTERVAL)
        bot.start()
        bot.stop(timeout=2.0)
        with pytest.raises(EngineError, match="повторный запуск"):
            bot.start()

    def test_context_manager(self):
        hid = RecordingHid()
        with L2CoreBot(make_wiring(hid),
                       poll_interval=POLL_INTERVAL,
                       dispatch_interval=DISPATCH_INTERVAL) as bot:
            assert bot.observer_thread.is_alive()
        assert not bot.observer_thread.is_alive()
        assert not bot.dispatcher_thread.is_alive()

    def test_wiring_hid_wrapped_guard(self):
        bot = L2CoreBot(make_wiring(RecordingHid()),
                        poll_interval=POLL_INTERVAL,
                        dispatch_interval=DISPATCH_INTERVAL)
        assert isinstance(bot.wiring.hid, OwnerGuardedHid)
        outcome = {}

        def foreign():
            try:
                bot.wiring.hid.click(1, 1, 50)
            except HidOwnershipError as exc:
                outcome["error"] = exc

        thread = threading.Thread(target=foreign)
        thread.start()
        thread.join(timeout=2.0)
        assert isinstance(outcome.get("error"), HidOwnershipError)

    def test_api_enforces_owner_thread(self):
        bot = L2CoreBot(make_wiring(RecordingHid()),
                        poll_interval=POLL_INTERVAL,
                        dispatch_interval=DISPATCH_INTERVAL)
        outcome = {}

        def foreign():
            try:
                bot.api.health()
            except RuntimeError as exc:
                outcome["error"] = exc

        thread = threading.Thread(target=foreign)
        thread.start()
        thread.join(timeout=2.0)
        assert isinstance(outcome.get("error"), RuntimeError)
        assert "поток" in str(outcome["error"])

    def test_api_usable_in_owner_thread(self):
        bot = L2CoreBot(make_wiring(RecordingHid()),
                        poll_interval=POLL_INTERVAL,
                        dispatch_interval=DISPATCH_INTERVAL)
        health = bot.api.health()
        assert health.ok

    def test_observer_feeds_engine_queue(self):
        hid = RecordingHid()
        wiring = make_wiring(hid, current_window=ScriptedWindows(
            ("market", None)))
        bot = L2CoreBot(wiring, poll_interval=POLL_INTERVAL,
                        dispatch_interval=DISPATCH_INTERVAL)
        received = []
        bot.dispatcher.subscribe(EVENT_WINDOW_CLOSED, received.append)
        bot.start()
        try:
            assert wait_for(lambda: len(received) == 1)
            assert received[0].payload == {"window": "market"}
        finally:
            bot.stop(timeout=2.0)
        assert hid.calls == []

    def test_push_external_event_delivered(self):
        bot = L2CoreBot(make_wiring(RecordingHid()),
                        poll_interval=POLL_INTERVAL,
                        dispatch_interval=DISPATCH_INTERVAL)
        received = []
        bot.dispatcher.subscribe(EVENT_ESCALATION, received.append)
        bot.start()
        try:
            event = bot.push(EVENT_ESCALATION, {"detail": "тест"})
            assert wait_for(lambda: len(received) == 1)
            assert received[0] is event
            assert received[0].payload == {"detail": "тест"}
        finally:
            bot.stop(timeout=2.0)

    def test_stop_from_dispatcher_callback(self):
        """Остановка из callback'а (сценарий «killed останавливает бота»):
        dispatcher не join'ит сам себя, оба потока завершаются."""
        bot = L2CoreBot(make_wiring(RecordingHid()),
                        poll_interval=POLL_INTERVAL,
                        dispatch_interval=DISPATCH_INTERVAL)
        outcome = {}

        def shutdown(event):
            outcome["shutdown"] = bot.stop(timeout=2.0)

        bot.dispatcher.subscribe(EVENT_ESCALATION, shutdown)
        bot.start()
        bot.push(EVENT_ESCALATION, {"detail": "killed"})
        assert wait_for(lambda: not bot.dispatcher_thread.is_alive())
        assert wait_for(lambda: not bot.observer_thread.is_alive())
        assert outcome["shutdown"].stopped
        assert bot.stop(timeout=2.0).stopped

    def test_poll_interval_defaults_from_config(self):
        config = CoreConfig(timing=TimingConfig(poll_interval=0.123))
        bot = L2CoreBot(make_wiring(RecordingHid(), config=config),
                        dispatch_interval=DISPATCH_INTERVAL)
        assert bot.observer.interval == 0.123
