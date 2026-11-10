"""Kill switch: локальный хоткей + таймаут неактивности (задача 8.1,
spec safety: «Аварийная остановка хоткеем», «Таймаут неактивности»).

Контракт, проверяемый здесь:
- хоткей (инъецированный источник) -> kill: cancelled выставлен, hid.
  release_all() вызван, курсор уведён в безопасную точку, on_kill получил
  причину; повторный kill идемпотентен;
- таймаут неактивности: note_activity() продлевает линию, превышение ->
  kill с reason="inactivity" (детерминированно на фейковых часах через
  poll_once + живой прогон run-цикла);
- блокирующий вызов прерывается: wait_until с cancelled=killswitch.
  is_killed возвращает результат с cancelled=True; полный прогон
  api.open_market() прерван kill -> NavigationResult(cancelled=True,
  outcome=None, unconfirmed=True) — CANCELLED-домен, не UNKNOWN;
- активность: L2CoreAPI через wiring.note_activity продлевает таймер;
- OwnerGuardedHid: release_all при сработавшем kill — санкционированное
  исключение guard'а владельца (любая другая команда из чужого потока
  по-прежнему отклоняется);
- L2CoreBot: kill_switch подключён — событие killed в очереди/журнале
  (payload {"reason": ...}), observer получает cancelled по умолчанию и
  завершается сам, dispatcher жив, stop() штатный;
- конфиг: kill_switch.safe_point (пара неотрицательных int; None —
  курсор не уводится);
- Win32HotkeySource: разбор хоткея (без Win32 — парсинг в __init__),
  протокол Win32Gateway объявляет set_cursor_pos.

Детерминизм (инвариант 7): источник клавиш — инъекция, часы — инъекция;
поток слушателя короткоживущий (poll_interval 0.01), join в каждом тесте.
"""
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from l2core.adapters.win32 import GameWindow, Win32Gateway, Win32HotkeySource
from l2core.api.core import L2CoreAPI
from l2core.api.engine import L2CoreBot
from l2core.api.events import EVENT_KILLED
from l2core.api.hid_owner import HidOwnershipError, OwnerGuardedHid
from l2core.api.killswitch import (
    REASON_HOTKEY,
    REASON_INACTIVITY,
    KillSwitch,
    KillSwitchError,
)
from l2core.api.wiring import ApiWiring
from l2core.config import (
    CoreConfig,
    KillSwitchConfig,
    TimingConfig,
    load_config,
)
from l2core.detector.status import STATUS_NOT_FOUND, WindowStatus
from l2core.model import (
    ActionSpec,
    Check,
    ElementSpec,
    TransitionSpec,
    UiModel,
    WindowSpec,
)
from l2core.primitives.click import CheckReport
from l2core.primitives.wait import wait_until

POLL = 0.01   # poll_interval слушателя в потоковых тестах


def wait_for(predicate, timeout=5.0):
    """Ожидание предиката с таймаутом (синхронизация тестов)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


class FakeClock:
    """Инъецированные часы: ручной перевод времени."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class FakeHotkeySource:
    """Источник хоткея-инъекция: press() имитирует нажатие."""

    def __init__(self):
        self._pressed = threading.Event()
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def pressed(self):
        return self._pressed.is_set()

    def press(self):
        self._pressed.set()


class RecordingHid:
    """HID-заглушка: записывает команды (эффекты не исполняет)."""

    TRAJ_MAX_POINTS = 60

    def __init__(self):
        self.calls = []

    def release_all(self):
        self.calls.append("release_all")

    def click(self, *args, **kwargs):
        self.calls.append("click")

    def key_press(self, *args, **kwargs):
        self.calls.append("key_press")

    def move_rel(self, *args, **kwargs):
        self.calls.append("move_rel")

    def is_connected(self):
        return True


class FailingHid(RecordingHid):
    """HID, чей release_all падает: kill не должен прерываться."""

    def release_all(self):
        raise RuntimeError("транспортный сбой HID")


def make_switch(**overrides):
    """KillSwitch с тестовыми инъекциями (по умолчанию источника нет)."""
    params = dict(hotkey_source=None, poll_interval=POLL)
    params.update(overrides)
    return KillSwitch(**params)


# --- Срабатывание по хоткею и по таймауту неактивности ---


class TestKillTrigger:
    def test_hotkey_poll_once_kills_with_full_effects(self):
        source = FakeHotkeySource()
        hid = RecordingHid()
        cursor = []
        reasons = []
        ks = make_switch(
            hotkey_source=source, hid=hid, safe_point=(8, 8),
            set_cursor_pos=lambda x, y: cursor.append((x, y)),
            on_kill=reasons.append)

        assert ks.poll_once() is False
        assert ks.is_killed() is False

        source.press()
        assert ks.poll_once() is True

        assert ks.is_killed() is True
        assert ks.reason == REASON_HOTKEY
        assert ks.cancelled() is True      # контракт cancelled-callable
        assert hid.calls == ["release_all"]
        assert cursor == [(8, 8)]
        assert reasons == [REASON_HOTKEY]

    def test_kill_is_idempotent(self):
        hid = RecordingHid()
        reasons = []
        ks = make_switch(hid=hid, on_kill=reasons.append)

        assert ks.kill(REASON_HOTKEY) is True
        assert ks.kill(REASON_INACTIVITY) is False
        assert ks.kill(REASON_HOTKEY) is False

        assert hid.calls == ["release_all"]   # ровно один раз
        assert reasons == [REASON_HOTKEY]     # событие — одно
        assert ks.reason == REASON_HOTKEY     # причина первого kill

    def test_hid_failure_does_not_block_kill(self):
        hid = FailingHid()
        cursor = []
        reasons = []
        ks = make_switch(hid=hid, safe_point=(4, 4),
                         set_cursor_pos=lambda x, y: cursor.append((x, y)),
                         on_kill=reasons.append)

        assert ks.kill(REASON_INACTIVITY) is True
        assert ks.is_killed() is True
        assert cursor == [(4, 4)]
        assert reasons == [REASON_INACTIVITY]

    def test_cursor_not_moved_without_safe_point(self):
        cursor = []
        ks = make_switch(safe_point=None,
                         set_cursor_pos=lambda x, y: cursor.append((x, y)))
        ks.kill(REASON_HOTKEY)
        assert ks.is_killed() is True
        assert cursor == []   # безопасная точка не задана — курсор не трогаем

    def test_safe_point_from_config(self):
        cfg = KillSwitchConfig(hotkey="ctrl+shift+k", inactivity_timeout=60.0,
                               safe_point=(16, 16))
        cursor = []
        ks = make_switch(config=cfg,
                         set_cursor_pos=lambda x, y: cursor.append((x, y)))
        ks.kill(REASON_HOTKEY)
        assert cursor == [(16, 16)]

    def test_explicit_safe_point_wins_over_config(self):
        cfg = KillSwitchConfig(safe_point=(16, 16))
        cursor = []
        ks = make_switch(config=cfg, safe_point=(1, 2),
                         set_cursor_pos=lambda x, y: cursor.append((x, y)))
        ks.kill(REASON_HOTKEY)
        assert cursor == [(1, 2)]

    def test_inactivity_timeout_deterministic(self):
        clock = FakeClock()
        ks = make_switch(inactivity_timeout=10.0, clock=clock)

        ks.note_activity()
        clock.advance(9.9)
        assert ks.poll_once() is False
        ks.note_activity()               # активность продлевает линию
        clock.advance(9.9)
        assert ks.poll_once() is False
        clock.advance(0.2)               # суммарно 10.1 с тишины
        assert ks.poll_once() is True
        assert ks.reason == REASON_INACTIVITY

    def test_inactivity_disabled_without_timeout(self):
        clock = FakeClock()
        ks = make_switch(inactivity_timeout=None, clock=clock)
        ks.note_activity()
        clock.advance(10 ** 9)
        assert ks.poll_once() is False
        assert ks.is_killed() is False

    def test_inactivity_from_config(self):
        clock = FakeClock()
        cfg = KillSwitchConfig(inactivity_timeout=5.0)
        ks = make_switch(config=cfg, inactivity_timeout=None, clock=clock)
        ks.note_activity()
        clock.advance(5.0)
        assert ks.poll_once() is True
        assert ks.reason == REASON_INACTIVITY


# --- Жизненный цикл слушателя ---


class TestListenerLifecycle:
    def test_run_loop_hotkey_threaded(self):
        source = FakeHotkeySource()
        ks = make_switch(hotkey_source=source)

        ks.start()
        assert source.started is True
        source.press()
        assert wait_for(ks.is_killed)
        assert ks.reason == REASON_HOTKEY

        assert ks.stop(timeout=2.0) is True
        assert source.stopped is True

    def test_run_loop_inactivity_threaded(self):
        ks = make_switch(inactivity_timeout=0.05)

        ks.start()
        assert wait_for(ks.is_killed)
        assert ks.reason == REASON_INACTIVITY
        assert ks.stop(timeout=2.0) is True

    def test_start_twice_raises(self):
        ks = make_switch(hotkey_source=FakeHotkeySource())
        ks.start()
        try:
            with pytest.raises(KillSwitchError):
                ks.start()
        finally:
            assert ks.stop(timeout=2.0) is True

    def test_start_after_stop_raises(self):
        ks = make_switch(hotkey_source=FakeHotkeySource())
        ks.start()
        assert ks.stop(timeout=2.0) is True
        with pytest.raises(KillSwitchError):
            ks.start()

    def test_stop_without_start_is_noop(self):
        ks = make_switch()
        assert ks.stop(timeout=2.0) is True

    def test_note_activity_from_listener_thread_keeps_alive(self):
        """Активность, поданная во время работы слушателя, продлевает
        линию: kill по таймауту не срабатывает, пока поток активности жив."""
        clock = FakeClock()
        ks = make_switch(inactivity_timeout=1.0, clock=clock)

        ks.start()
        try:
            for _ in range(5):
                clock.advance(0.5)
                ks.note_activity()
                time.sleep(0.02)   # слушатель успевает сделать цикл
            assert ks.is_killed() is False
        finally:
            assert ks.stop(timeout=2.0) is True


# --- Прерывание блокирующих вызовов ---


def _blocked_wait(cancelled):
    """Блокирующее ожидание без шанса на успех: прерывается только kill."""
    return wait_until(lambda: False, timeout=30.0, interval=0.02,
                      cancelled=cancelled)


class TestBlockingInterruption:
    def test_wait_until_interrupted_by_kill(self):
        """Прямой контракт spec: блокирующий вызов прерывается kill с
        флагом cancelled (CANCELLED-домен, не UNKNOWN)."""
        ks = make_switch()
        result = {}

        def worker():
            result["wait"] = _blocked_wait(ks.is_killed)

        thread = threading.Thread(target=worker, name="strategy")
        thread.start()
        time.sleep(0.05)          # worker входит в wait_until
        assert ks.kill(REASON_HOTKEY) is True
        thread.join(timeout=5.0)
        assert not thread.is_alive()

        wait = result["wait"]
        assert wait.cancelled is True
        assert wait.success is False

    def test_api_open_market_interrupted_by_kill(self):
        """Полный прогон: поток стратегии блокирован в api.open_market,
        хоткей -> вызов прерывается, результат CANCELLED."""
        source = FakeHotkeySource()
        ks = make_switch(hotkey_source=source)
        hid = RecordingHid()
        outcome = {}

        def strategy():
            # Владелец API — поток стратегии (thread-owner 7.1).
            api = L2CoreAPI(wiring,
                            owner_thread=threading.current_thread())
            outcome["result"] = api.open_market()

        thread = threading.Thread(target=strategy, name="strategy")
        # HID сборки — под guard'ом владельца (владелец — поток стратегии):
        # kill приходит из потока слушателя — санкционированный обход.
        guarded = OwnerGuardedHid(hid, owner_thread=thread,
                                  kill_switch=ks.is_killed)
        ks.attach(hid=guarded)
        wiring = _blocking_wiring(guarded, cancelled=ks.is_killed)
        ks.start()
        try:
            thread.start()
            assert wait_for(lambda: "key_press" in hid.calls)  # X ушла, ждём verify
            source.press()                      # пользователь жмёт хоткей
            thread.join(timeout=5.0)
            assert not thread.is_alive()
        finally:
            assert ks.stop(timeout=2.0) is True

        result = outcome["result"]
        assert result.cancelled is True
        assert result.outcome is None        # CANCELLED не адаптируется (5.7)
        assert result.unconfirmed is True    # клик ушёл, эффект не подтверждён
        assert "release_all" in hid.calls    # HID освобождён


def _blocking_wiring(hid, *, cancelled=None, note_activity=None):
    """Сборка wiring, где open_market блокируется в ожидании verify шага
    рецепта: нажатие X уходит, признаки не сходятся никогда."""
    from l2core.model.navigation import (
        NavigationModel,
        Recipe,
        RecipeStep,
    )

    never_checks = (Check(kind="template", element="never"),)
    key_step = RecipeStep(action="key", element=None, key="X",
                          verify=never_checks, on_fail=())
    navigation = NavigationModel(
        path="navigation",
        elements={},
        signatures={},
        recipes={
            "open_market": Recipe(id="open_market", opens="market",
                                  steps=(key_step,), on_fail=()),
            "open_shop_buy": Recipe(id="open_shop_buy", opens="shop_buy",
                                    steps=(key_step,), on_fail=()),
            "open_shop_sell": Recipe(id="open_shop_sell", opens="shop_sell",
                                     steps=(key_step,), on_fail=()),
        })

    def nav_evaluate(checks, frame):
        return CheckReport(passed=False, failures=("never",))

    model = UiModel(
        path="market",
        window=WindowSpec(id="market", title="m", size=(100, 100)),
        elements={},
        states={},
        transitions=(),
        ocr_regions={}, table_geometry=None)
    return ApiWiring(
        config=CoreConfig(
            timing=TimingConfig(wait_default_timeout=30.0,
                                verify_interval=0.02)),
        screen=_FakeScreen(),
        hid=hid,
        ocr=None,
        market=model, shop_buy=model, shop_sell=model, hud_model=model,
        cursor_pos=lambda: (0, 0),
        activate=lambda: SimpleNamespace(cancelled=False),
        find_anchor=lambda frame, window_id: None,
        window_status=lambda frame, window_id: WindowStatus(
            status=STATUS_NOT_FOUND, reason=None, anchor=None,
            content_loaded=None, content_detail=None),
        run_checks=lambda checks, frame, anchor: CheckReport(passed=False),
        current_window=lambda frame: None,
        close_point=lambda window_id, frame: None,
        hud_locator=lambda element, variant, frame: (10, 10),
        goal_reached=lambda target: (lambda frame: False),
        game_window=lambda: GameWindow.found(1, (0, 0, 64, 64)),
        ocr_health=lambda: True,
        hid_health=lambda: True,
        cancelled=cancelled,
        note_activity=note_activity,
        navigation=navigation,
        nav_evaluate=nav_evaluate)


class _FakeScreen:
    def __init__(self):
        self.frame = np.zeros((32, 32, 3), dtype=np.uint8)

    def grab(self):
        return self.frame


# --- Активность через API ---


class TestActivity:
    def test_api_call_notes_activity(self):
        """Блокирующий вызов API продлевает таймер неактивности
        (wiring.note_activity — канал активности executor'а)."""
        clock = FakeClock()
        ks = make_switch(inactivity_timeout=10.0, clock=clock)
        wiring = _blocking_wiring(RecordingHid(),
                                  note_activity=ks.note_activity)
        api = L2CoreAPI(wiring)

        ks.note_activity()           # старт линии на старте цикла
        clock.advance(9.9)
        api.get_state()              # вызов из потока стратегии — активность
        clock.advance(9.9)           # суммарно 19.8 с тишины — не сработало
        assert ks.poll_once() is False
        clock.advance(0.2)
        assert ks.poll_once() is True
        assert ks.reason == REASON_INACTIVITY


# --- Guard владельца и kill ---


class TestGuardKillException:
    def test_release_all_allowed_from_foreign_thread_after_kill(self):
        """Решение 8.1: release_all при сработавшем kill — санкционированное
        исключение guard'а владельца (слушатель живёт в своём потоке)."""
        hid = RecordingHid()
        ks = make_switch()
        owner = threading.Thread(target=lambda: None)   # фиктивный владелец
        guarded = OwnerGuardedHid(hid, owner_thread=owner,
                                  kill_switch=ks.is_killed)

        with pytest.raises(HidOwnershipError):
            guarded.release_all()
        assert hid.calls == []

        ks.kill(REASON_HOTKEY)
        guarded.release_all()          # из чужого потока, но kill активен
        assert hid.calls == ["release_all"]

    def test_other_commands_still_blocked_after_kill(self):
        """Kill открывает только release_all: клик из чужого потока при
        активном kill по-прежнему отклоняется (guard fail closed)."""
        hid = RecordingHid()
        ks = make_switch()
        owner = threading.Thread(target=lambda: None)
        guarded = OwnerGuardedHid(hid, owner_thread=owner,
                                  kill_switch=ks.is_killed)
        ks.kill(REASON_HOTKEY)
        with pytest.raises(HidOwnershipError):
            guarded.click(1, 1, 50)
        with pytest.raises(HidOwnershipError):
            guarded.key_press(0x1E)
        assert hid.calls == []   # отклонённые команды железо не достигли

    def test_without_kill_switch_guard_unchanged(self):
        hid = RecordingHid()
        owner = threading.Thread(target=lambda: None)
        guarded = OwnerGuardedHid(hid, owner_thread=owner)
        with pytest.raises(HidOwnershipError):
            guarded.release_all()
        assert hid.calls == []


# --- Интеграция с L2CoreBot ---


class TestEngineIntegration:
    def test_killed_event_published_to_bot_queue(self):
        ks = make_switch()
        hid = RecordingHid()
        bot = L2CoreBot(_blocking_wiring(hid),
                        kill_switch=ks, poll_interval=POLL)

        assert ks.kill(REASON_HOTKEY) is True

        entries = bot.event_api.get_events()
        killed = [e for e in entries if e.event.type == EVENT_KILLED]
        assert len(killed) == 1
        assert killed[0].event.payload == {"reason": REASON_HOTKEY}
        # release_all дошёл до железа через guard-сборку бота (из потока
        # kill-слушателя — санкционированное исключение guard'а 8.1).
        assert hid.calls == ["release_all"]

    def test_observer_cancelled_defaults_to_kill_switch(self):
        """Движок с kill_switch и без явного cancelled: observer проверяет
        kill — после kill цикл наблюдения завершается сам, dispatcher жив,
        stop() штатный."""
        ks = make_switch()
        bot = L2CoreBot(_blocking_wiring(RecordingHid()),
                        kill_switch=ks, poll_interval=POLL)
        bot.start()
        try:
            assert wait_for(lambda: bot.observer.iterations > 0)
            ks.kill(REASON_INACTIVITY)
            assert wait_for(lambda: not bot.observer_thread.is_alive())
            assert bot.dispatcher_thread.is_alive()
        finally:
            shutdown = bot.stop(timeout=2.0)
        assert shutdown.stopped is True


# --- Асимметрия авто-подключения (ревью 8.1) ---


class TestBotCancelledDefault:
    """L2CoreBot(kill_switch=ks) дефолтит wiring.cancelled = ks.is_killed
    (явный wiring.cancelled побеждает). Иначе headline-гарантия спеки
    «прерывание блокирующих вызовов» молча не работает: wait_until
    стратегии идёт до таймаута, поток-владелец HID шлёт команды."""

    def test_wiring_cancelled_defaults_to_kill_switch(self):
        ks = make_switch()
        bot = L2CoreBot(_blocking_wiring(RecordingHid()),
                        kill_switch=ks, poll_interval=POLL)

        assert bot.wiring.cancelled is not None
        assert bot.wiring.cancelled() is False
        ks.kill(REASON_HOTKEY)
        assert bot.wiring.cancelled() is True

    def test_explicit_wiring_cancelled_wins(self):
        ks = make_switch()
        custom = lambda: False   # noqa: E731
        bot = L2CoreBot(_blocking_wiring(RecordingHid(), cancelled=custom),
                        kill_switch=ks, poll_interval=POLL)
        assert bot.wiring.cancelled is custom

    def test_api_call_interrupted_without_explicit_wiring_cancelled(self):
        """Интеграционный прогон: wiring собран БЕЗ cancelled, движок
        подставил ks.is_killed — блокирующий api.open_market из потока
        стратегии прерывается kill'ом (CANCELLED-домен)."""
        import dataclasses

        ks = make_switch()
        hid = RecordingHid()
        wiring = _blocking_wiring(hid)
        wiring = dataclasses.replace(
            wiring,
            config=CoreConfig(timing=TimingConfig(
                wait_default_timeout=5.0, verify_interval=0.02)))
        outcome = {}

        def strategy():
            # Бот создаётся в потоке стратегии: владелец guard-сборки —
            # этот поток (эталонная сборка).
            bot = L2CoreBot(wiring, kill_switch=ks, poll_interval=POLL)
            api = L2CoreAPI(bot.wiring,
                            owner_thread=threading.current_thread())
            outcome["result"] = api.open_market()

        thread = threading.Thread(target=strategy, name="strategy")
        thread.start()
        assert wait_for(lambda: "key_press" in hid.calls)  # X ушла, ждём verify
        ks.kill(REASON_HOTKEY)
        thread.join(timeout=5.0)
        assert not thread.is_alive()

        result = outcome["result"]
        assert result.cancelled is True
        assert result.outcome is None
        assert result.unconfirmed is True
        assert "release_all" in hid.calls


# --- Конфиг: safe_point ---


class TestConfigSafePoint:
    def test_defaults(self):
        cfg = load_config(None)
        assert cfg.kill_switch.safe_point == (8, 8)
        assert cfg.kill_switch.hotkey == "ctrl+shift+k"
        assert cfg.kill_switch.inactivity_timeout == 300.0

    def test_safe_point_from_yaml(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("kill_switch:\n  safe_point: [8, 8]\n",
                        encoding="utf-8")
        cfg = load_config(path)
        assert cfg.kill_switch.safe_point == (8, 8)

    def test_safe_point_none_explicit(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("kill_switch:\n  safe_point: null\n",
                        encoding="utf-8")
        assert load_config(path).kill_switch.safe_point is None

    @pytest.mark.parametrize("raw", [
        "[8]",          # не пара
        "[8, 8, 8]",    # тройка
        "['a', 8]",     # не числа
        "[-1, 8]",      # отрицательная
        "8",            # не список
    ])
    def test_safe_point_invalid_rejected(self, tmp_path, raw):
        from l2core.config import ConfigError

        path = tmp_path / "config.yaml"
        path.write_text(f"kill_switch:\n  safe_point: {raw}\n",
                        encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(path)


# --- Win32 hotkey source (парсинг без Win32; hook — живой код) ---


class TestWin32HotkeySource:
    def test_parse_ctrl_shift_k(self):
        source = Win32HotkeySource("ctrl+shift+k")
        assert source.vk == 0x4B                 # 'K'
        assert source.modifiers == {"ctrl", "shift"}

    def test_parse_letter_and_digit_and_fkeys(self):
        assert Win32HotkeySource("alt+f4").vk == 0x73
        assert Win32HotkeySource("ctrl+0").vk == 0x30
        with pytest.raises(ValueError):
            Win32HotkeySource("ctrl+f13")

    def test_parse_unknown_and_empty_rejected(self):
        for bad in ("ctrl+qq", "ctrl+", "+k", ""):
            with pytest.raises(ValueError):
                Win32HotkeySource(bad)

    def test_gateway_protocol_declares_set_cursor_pos(self):
        import inspect

        assert hasattr(Win32Gateway, "set_cursor_pos")
        sig = inspect.signature(Win32Gateway.set_cursor_pos)
        assert list(sig.parameters) == ["self", "x", "y"]
