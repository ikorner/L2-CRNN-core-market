"""Тесты drag_item (задача 5.6, spec action-primitives: «Каждое действие
верифицируется, при неудаче — retry → on_fail → recover»; market-scenarios/
design: «Перенос предметов в лавку и обратно — drag»; инвариант 9: слепых
действий нет — без activate OK + статуса found HID не трогается).

Детерминизм (инвариант 7): ScriptedHid фиксирует действия с метками фейкового
времени, скриптованные run_checks/find_anchor/status, rng — random.Random с
фиксированным seed, часы — FakeClock (sleep продвигает время). Никаких
обращений к живому экрану/окнам/HID.

Порядок rng-розыгрышей на попытку (зафиксирован пинами): джиттер source,
джиттер target, hold_ms (press_delay), число промежуточных точек, на каждую
середину (джиттер move + задержка шага), задержка финального шага.
"""
import ast
import random
from pathlib import Path

import numpy as np
import pytest

from l2core.config import (
    BehaviorConfig,
    CoreConfig,
    ResolutionConfig,
    TimingConfig,
)
from l2core.detector.anchor import Anchor
from l2core.detector.status import (
    STATUS_FOUND,
    STATUS_NOT_FOUND,
    STATUS_OCCLUDED,
    WindowStatus,
)
from l2core.model import Check
from l2core.primitives.activate import ActivateGameError, ActivateResult
from l2core.primitives.click import CheckReport, VerifySpec

SOURCE = (300, 400)
TARGET = (700, 500)
ON_FAIL = ("press_esc", "click_free_area")
VERIFY = VerifySpec(
    checks=(Check(kind="template", element="slot", variant="filled"),),
    expected="item_placed",
)

# Параметры профиля make_config(): пины ниже считаются от этих значений.
CLICK_JITTER = 3
MOVE_JITTER = 8
PRESS_MIN, PRESS_MAX = 40, 120
STEP_MIN, STEP_MAX = 8, 24
MID_MIN, MID_MAX = 1, 3


class FakeClock:
    """Фейковые часы: sleep продвигает время."""

    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeScreen:
    """Экран со скриптованной последовательностью кадров (последний — по кругу)."""

    def __init__(self, frames=None):
        self._frames = list(frames or [np.zeros((1080, 1920, 3), dtype=np.uint8)])
        self._index = 0

    def grab(self):
        frame = self._frames[min(self._index, len(self._frames) - 1)]
        if self._index < len(self._frames) - 1:
            self._index += 1
        return frame


class ScriptedHid:
    """HID с записью действий: (время, действие, args)."""

    def __init__(self, clock):
        self.clock = clock
        self.log = []

    def move_rel(self, dx, dy):
        self.log.append((self.clock.now, "move_rel", (dx, dy)))

    def click(self, button=1, count=1, hold_ms=None):
        self.log.append((self.clock.now, "click", (button, count, hold_ms)))

    def key_press(self, code):
        self.log.append((self.clock.now, "key_press", (code,)))

    def mouse_down(self, button=1):
        self.log.append((self.clock.now, "mouse_down", (button,)))

    def mouse_up(self, button=1):
        self.log.append((self.clock.now, "mouse_up", (button,)))

    def traj(self, points):
        self.log.append((self.clock.now, "traj", tuple(points)))


class FakeCursor:
    """Позиция курсора: move_rel и traj двигают."""

    def __init__(self, pos=(500, 300)):
        self.pos = pos

    def get(self):
        return self.pos

    def track(self, hid):
        original_move = hid.move_rel
        original_traj = hid.traj

        def move_rel(dx, dy):
            self.pos = (self.pos[0] + dx, self.pos[1] + dy)
            original_move(dx, dy)

        def traj(points):
            self.pos = (
                self.pos[0] + sum(p[0] for p in points),
                self.pos[1] + sum(p[1] for p in points),
            )
            original_traj(points)

        hid.move_rel = move_rel
        hid.traj = traj


class FakeChecker:
    """Скриптованная оценка verify-проверок."""

    def __init__(self, verify=None):
        self._verify = list(verify or [])
        self.calls = []

    def __call__(self, checks, frame, anchor):
        self.calls.append(tuple(checks))
        passed = self._verify.pop(0) if self._verify else True
        failures = () if passed else tuple(f"fail:{c.element}/{c.variant}" for c in checks)
        return CheckReport(passed=passed, failures=failures)


def make_config(**timing_kwargs):
    timing = TimingConfig(verify_interval=0.05, hover_pause=0.2, **timing_kwargs)
    behavior = BehaviorConfig(
        click_jitter_px=CLICK_JITTER,
        move_jitter_px=MOVE_JITTER,
        press_delay_min_ms=PRESS_MIN,
        press_delay_max_ms=PRESS_MAX,
        move_step_delay_min_ms=STEP_MIN,
        move_step_delay_max_ms=STEP_MAX,
        drag_midpoints_min=MID_MIN,
        drag_midpoints_max=MID_MAX,
    )
    return CoreConfig(
        timing=timing,
        behavior=behavior,
        resolution=ResolutionConfig(width=1920, height=1080),
    )


def found_status(frame):
    return WindowStatus(
        status=STATUS_FOUND, reason=None, anchor=None,
        content_loaded=None, content_detail="",
    )


def run_drag(hid, clock, checker, source=SOURCE, target=TARGET, verify=None,
             activate=None, status=None, find_anchor=None, on_fail=(),
             rng_seed=7, **kwargs):
    from l2core.primitives.drag import drag_item

    activate = activate or (
        lambda: ActivateResult(hwnd=1, already_active=True, attempts=0, cancelled=False)
    )
    return drag_item(
        source,
        target,
        verify=verify or VERIFY,
        run_checks=checker,
        screen=kwargs.pop("screen", FakeScreen()),
        hid=hid,
        cursor_pos=kwargs.pop("cursor_pos", FakeCursor().get),
        activate=activate,
        window_status=status or found_status,
        find_anchor=find_anchor or (lambda frame: ANCHOR),
        on_fail=on_fail,
        config=kwargs.pop("config", make_config()),
        rng=random.Random(rng_seed),
        clock=clock.monotonic,
        sleep=clock.sleep,
        **kwargs,
    )


ANCHOR = Anchor(window_id="shop_sell", x=202, y=103, score=0.99, method="template")


def planned(seed=7):
    """Ожидаемый rng-план попытки: джиттеры, hold, середины, задержки."""
    rng = random.Random(seed)
    sjx, sjy = rng.randint(-CLICK_JITTER, CLICK_JITTER), rng.randint(-CLICK_JITTER, CLICK_JITTER)
    tjx, tjy = rng.randint(-CLICK_JITTER, CLICK_JITTER), rng.randint(-CLICK_JITTER, CLICK_JITTER)
    hold = rng.randint(PRESS_MIN, PRESS_MAX)
    n_mid = rng.randint(MID_MIN, MID_MAX)
    mids = []
    for _ in range(n_mid):
        jx, jy = rng.randint(-MOVE_JITTER, MOVE_JITTER), rng.randint(-MOVE_JITTER, MOVE_JITTER)
        mids.append((jx, jy, rng.randint(STEP_MIN, STEP_MAX)))
    d_last = rng.randint(STEP_MIN, STEP_MAX)
    return sjx, sjy, tjx, tjy, hold, mids, d_last


def expected_traj_points(seed=7):
    """Точки traj (dx, dy, delay_ms), посчитанные тем же алгоритмом, что ядро."""
    sjx, sjy, tjx, tjy, _hold, mids, d_last = planned(seed)
    cur = (SOURCE[0] + sjx, SOURCE[1] + sjy)
    points = []
    n = len(mids)
    for i, (jx, jy, delay) in enumerate(mids, start=1):
        frac = i / (n + 1)
        px = round(SOURCE[0] + (TARGET[0] - SOURCE[0]) * frac) + jx
        py = round(SOURCE[1] + (TARGET[1] - SOURCE[1]) * frac) + jy
        points.append((px - cur[0], py - cur[1], delay))
        cur = (px, py)
    points.append((TARGET[0] + tjx - cur[0], TARGET[1] + tjy - cur[1], d_last))
    return points


class TestSuccessfulDrag:
    def test_drag_sequence_and_verify(self):
        # Успешный перенос: move -> press -> hold -> traj -> пауза -> release
        # -> verify OK; одна попытка.
        clock = FakeClock()
        hid = ScriptedHid(clock)
        FakeCursor().track(hid)
        result = run_drag(hid, clock, FakeChecker())

        assert result.success is True
        assert result.attempts == 1
        assert result.cancelled is False
        assert result.on_fail_executed is False
        assert result.needs_recover is False
        assert result.source_point == SOURCE and result.target_point == TARGET

        actions = [a for _, a, _ in hid.log]
        assert actions == ["move_rel", "mouse_down", "traj", "mouse_up"]
        assert hid.log[1][2] == (0x01,)  # BTN_LEFT
        assert hid.log[3][2] == (0x01,)
        assert hid.log[2][2] == tuple(expected_traj_points())

    def test_move_to_jittered_source(self):
        # Курсор наводится на source + джиттер клика (click_jitter_px, seed=7).
        clock = FakeClock()
        hid = ScriptedHid(clock)
        FakeCursor().track(hid)
        run_drag(hid, clock, FakeChecker())

        sjx, sjy, _, _, _, _, _ = planned()
        assert hid.log[0][2] == (SOURCE[0] + sjx - 500, SOURCE[1] + sjy - 300)

    def test_press_hold_pause_before_traj(self):
        # Удержание после press (press_delay из профиля) — между mouse_down
        # и traj, без sleep: traj не раньше, чем часы дошли до hold.
        clock = FakeClock()
        hid = ScriptedHid(clock)
        FakeCursor().track(hid)
        run_drag(hid, clock, FakeChecker())

        down_t = next(t for t, a, _ in hid.log if a == "mouse_down")
        traj_t = next(t for t, a, _ in hid.log if a == "traj")
        _, _, _, _, hold, _, _ = planned()
        assert down_t == 0.0
        assert traj_t >= hold / 1000.0 - 1e-9  # float-дрейф фейковых часов

    def test_target_pause_before_release(self):
        # Пауза на цели (timing.hover_pause) — между traj и release.
        clock = FakeClock()
        hid = ScriptedHid(clock)
        FakeCursor().track(hid)
        run_drag(hid, clock, FakeChecker())

        traj_t = next(t for t, a, _ in hid.log if a == "traj")
        up_t = next(t for t, a, _ in hid.log if a == "mouse_up")
        assert up_t - traj_t >= 0.2 - 1e-9  # hover_pause

    def test_traj_lands_on_jittered_target(self):
        # Сумма дельт траектории — ровно до target + джиттер: отпускаем над
        # целью, а не где придётся.
        clock = FakeClock()
        hid = ScriptedHid(clock)
        cursor = FakeCursor()
        cursor.track(hid)
        run_drag(hid, clock, FakeChecker())

        _, _, tjx, tjy, _, _, _ = planned()
        assert cursor.pos == (TARGET[0] + tjx, TARGET[1] + tjy)

    def test_delays_and_midpoints_from_profile(self):
        # Число точек traj = 1 + число промежуточных из профиля; задержки шагов
        # внутри [move_step_delay_min_ms, move_step_delay_max_ms].
        clock = FakeClock()
        hid = ScriptedHid(clock)
        FakeCursor().track(hid)
        run_drag(hid, clock, FakeChecker())

        points = next(args for _, a, args in hid.log if a == "traj")
        _, _, _, _, _, mids, _ = planned()
        assert len(points) == len(mids) + 1
        for _, _, delay in points:
            assert STEP_MIN <= delay <= STEP_MAX

    def test_rng_seed_deterministic(self):
        # Траектория/тайминги детерминированы при фиксированном seed: два
        # прогона с одним seed дают одинаковую последовательность действий.
        logs = []
        for _ in range(2):
            clock = FakeClock()
            hid = ScriptedHid(clock)
            FakeCursor().track(hid)
            run_drag(hid, clock, FakeChecker())
            logs.append(hid.log)
        assert logs[0] == logs[1]

    def test_verify_called_with_spec(self):
        # Verify оценивается через run_checks на свежем кадре с переданной
        # спецификацией.
        clock = FakeClock()
        hid = ScriptedHid(clock)
        FakeCursor().track(hid)
        checker = FakeChecker()
        run_drag(hid, clock, checker)

        assert len(checker.calls) == 1
        assert checker.calls[0] == VERIFY.checks

    def test_cancelled_flag_result(self):
        # Отмена (kill switch 8.1) до начала drag — флаг в результате, не
        # исключение; HID не трогали.
        clock = FakeClock()
        hid = ScriptedHid(clock)
        result = run_drag(hid, clock, FakeChecker(), cancelled=lambda: True)
        assert result.cancelled is True
        assert result.success is False
        assert result.attempts == 0
        assert result.needs_recover is False
        assert hid.log == []

    def test_cancel_during_hold_releases_button(self):
        # Отмена после press (в удержании): кнопка ОТПУЩЕНА (mouse_up), drag
        # не продолжен — не оставляем мышь зажатой.
        clock = FakeClock()
        hid = ScriptedHid(clock)
        FakeCursor().track(hid)
        state = {"armed": False}

        def cancelled():
            if state["armed"]:
                return True
            # Вооружаемся после press: следующий запрос (после hold) — True.
            if any(a == "mouse_down" for _, a, _ in hid.log):
                state["armed"] = True
            return False

        result = run_drag(hid, clock, FakeChecker(), cancelled=cancelled)

        assert result.cancelled is True
        assert result.success is False
        actions = [a for _, a, _ in hid.log]
        assert actions == ["move_rel", "mouse_down", "mouse_up"]  # traj нет
        assert hid.log[-1][0] < 0.2  # не ждали полный hold после отмены

    def test_cancel_on_target_pause_releases_button(self):
        # Отмена в паузе на цели: траектория прошла, кнопка отпущена.
        clock = FakeClock()
        hid = ScriptedHid(clock)
        FakeCursor().track(hid)
        state = {"armed": False}

        def cancelled():
            if state["armed"]:
                return True
            if any(a == "traj" for _, a, _ in hid.log):
                state["armed"] = True
            return False

        result = run_drag(hid, clock, FakeChecker(), cancelled=cancelled)

        assert result.cancelled is True
        actions = [a for _, a, _ in hid.log]
        assert actions == ["move_rel", "mouse_down", "traj", "mouse_up"]


class TestModelBoundaries:
    def test_trajectory_exceeds_protocol_fails_fast(self):
        # Число промежуточных точек из профиля не влезает в пакет траектории
        # HID-протокола — ошибка модели/конфига до HID (fail fast).
        clock = FakeClock()
        hid = ScriptedHid(clock)
        FakeCursor().track(hid)
        behavior = BehaviorConfig(
            click_jitter_px=CLICK_JITTER, move_jitter_px=MOVE_JITTER,
            press_delay_min_ms=PRESS_MIN, press_delay_max_ms=PRESS_MAX,
            move_step_delay_min_ms=STEP_MIN, move_step_delay_max_ms=STEP_MAX,
            drag_midpoints_min=100, drag_midpoints_max=100,
        )
        config = CoreConfig(
            timing=TimingConfig(verify_interval=0.05, hover_pause=0.2),
            behavior=behavior,
            resolution=ResolutionConfig(width=1920, height=1080),
        )
        from l2core.primitives.drag import DragElementError

        with pytest.raises(DragElementError):
            run_drag(hid, clock, FakeChecker(), config=config)
        assert hid.log == []


class TestVerifyRetry:
    def test_verify_fail_retried_full_drag(self):
        # Verify не проходит дважды, третья попытка — OK: ровно 3 полных drag
        # (press+traj+release каждый), on_fail не исполнялся.
        clock = FakeClock()
        hid = ScriptedHid(clock)
        FakeCursor().track(hid)
        checker = FakeChecker(verify=[False, False, True])
        result = run_drag(hid, clock, checker, on_fail=ON_FAIL)

        assert result.success is True
        assert result.attempts == 3
        actions = [a for _, a, _ in hid.log]
        assert actions.count("mouse_down") == 3
        assert actions.count("traj") == 3
        assert actions.count("mouse_up") == 3
        assert result.on_fail_executed is False

    def test_retry_count_from_config(self):
        # Число попыток = 1 + timing.click_retry_count.
        clock = FakeClock()
        hid = ScriptedHid(clock)
        FakeCursor().track(hid)
        checker = FakeChecker(verify=[False, False])
        result = run_drag(
            hid, clock, checker, on_fail=(), config=make_config(click_retry_count=1)
        )
        assert result.success is False
        assert result.attempts == 2
        assert [a for _, a, _ in hid.log].count("mouse_down") == 2

    def test_exhausted_verify_runs_on_fail_then_recover_signal(self):
        # Исчерпание retry -> повторный пролог -> шаги on_fail[] -> сигнал
        # recover; лишних HID-действий сверх лимита нет.
        clock = FakeClock()
        hid = ScriptedHid(clock)
        cursor = FakeCursor()
        cursor.track(hid)
        checker = FakeChecker(verify=[False, False, False])
        result = run_drag(
            hid, clock, checker, on_fail=ON_FAIL, free_area_point=(100, 700),
            cursor_pos=cursor.get,
        )

        assert result.success is False
        assert result.attempts == 3
        assert result.on_fail_executed is True
        assert result.needs_recover is True
        assert result.failures

        actions = [a for _, a, _ in hid.log]
        assert actions[-3:] == ["key_press", "move_rel", "click"]
        assert hid.log[-3][2] == (0xB1,)  # KEY_ESC
        assert cursor.pos == (100, 700)  # клик в заданную свободную точку

    def test_on_fail_rechecks_prologue(self):
        # Перед on_fail-шагами пролог повторён (activate + статус found):
        # окно живо — шаги исполняются, activate вызван дважды.
        clock = FakeClock()
        hid = ScriptedHid(clock)
        FakeCursor().track(hid)
        activates = {"count": 0}

        def activate():
            activates["count"] += 1
            return ActivateResult(hwnd=1, already_active=True, attempts=0, cancelled=False)

        checker = FakeChecker(verify=[False, False, False])
        result = run_drag(hid, clock, checker, on_fail=ON_FAIL, activate=activate)

        assert result.on_fail_executed is True
        assert activates["count"] == 2  # пролог drag + пролог on_fail

    def test_on_fail_skipped_when_window_degraded(self):
        # К моменту on_fail статус not_found: шаги НЕ исполняются — слепые
        # HID-шаги при деградировавшем окне запрещены (инвариант 9);
        # needs_recover=True, причина — в failures.
        clock = FakeClock()
        hid = ScriptedHid(clock)
        FakeCursor().track(hid)
        state = {"calls": 0}

        def status_fn(frame):
            state["calls"] += 1
            if state["calls"] == 1:
                return found_status(frame)
            return WindowStatus(
                status=STATUS_NOT_FOUND, reason="window_not_found", anchor=None,
                content_loaded=None, content_detail="",
            )

        checker = FakeChecker(verify=[False, False, False])
        result = run_drag(hid, clock, checker, on_fail=ON_FAIL, status=status_fn)

        assert result.success is False
        assert result.on_fail_executed is False
        assert result.needs_recover is True
        assert [a for _, a, _ in hid.log].count("key_press") == 0
        assert any("on_fail" in failure for failure in result.failures)

    def test_unknown_on_fail_step_fails_fast(self):
        clock = FakeClock()
        hid = ScriptedHid(clock)
        FakeCursor().track(hid)
        checker = FakeChecker(verify=[False])
        from l2core.primitives.drag import DragElementError

        with pytest.raises(DragElementError):
            run_drag(hid, clock, checker, on_fail=("press_esc", "teleport"),
                     config=make_config(click_retry_count=0))

    def test_anchor_lost_fails_attempt_without_press(self):
        # Якорь не найден на свежем кадре попытки: попытка потрачена БЕЗ
        # press (слепого drag нет), дальше обычный retry.
        clock = FakeClock()
        hid = ScriptedHid(clock)
        FakeCursor().track(hid)
        state = {"calls": 0}

        def find_anchor(frame):
            state["calls"] += 1
            return None if state["calls"] == 1 else ANCHOR

        checker = FakeChecker(verify=[True])
        result = run_drag(hid, clock, checker, find_anchor=find_anchor)

        assert result.success is True
        assert result.attempts == 2
        assert [a for _, a, _ in hid.log].count("mouse_down") == 1


class TestPrologue:
    def test_activate_failure_escalates_no_hid(self):
        # Пролог: activate_game не удался -> ActivateGameError проброшена,
        # HID не трогали (no blind drag).
        clock = FakeClock()
        hid = ScriptedHid(clock)

        def failing_activate():
            raise ActivateGameError(reason="focus_not_acquired", detail="нет фокуса")

        with pytest.raises(ActivateGameError):
            run_drag(hid, clock, FakeChecker(), activate=failing_activate)
        assert hid.log == []

    def test_status_not_found_blocks_drag(self):
        # Пролог: статус окна не found -> DragPreconditionError, HID не трогали.
        clock = FakeClock()
        hid = ScriptedHid(clock)

        def occluded(frame):
            return WindowStatus(
                status=STATUS_OCCLUDED, reason="anchor_lost", anchor=None,
                content_loaded=None, content_detail="",
            )

        from l2core.primitives.drag import DragPreconditionError

        with pytest.raises(DragPreconditionError) as exc_info:
            run_drag(hid, clock, FakeChecker(), status=occluded)
        assert exc_info.value.status == STATUS_OCCLUDED
        assert hid.log == []

    def test_status_not_found_reason_in_error(self):
        clock = FakeClock()
        hid = ScriptedHid(clock)

        def not_found(frame):
            return WindowStatus(
                status=STATUS_NOT_FOUND, reason="window_not_found", anchor=None,
                content_loaded=None, content_detail="",
            )

        from l2core.primitives.drag import DragPreconditionError

        with pytest.raises(DragPreconditionError) as exc_info:
            run_drag(hid, clock, FakeChecker(), status=not_found)
        assert "window_not_found" in str(exc_info.value)
        assert hid.log == []


class TestVerifyContract:
    def test_drag_without_verify_is_statically_required(self):
        # Guard: публичный API drag требует verify-спецификацию — параметр
        # `verify` обязателен (keyword-only, без дефолта). Drag без verify
        # запрещён (spec action-primitives); guard ловит регрессию сигнатуры.
        source = (Path(__file__).resolve().parent.parent
                  / "l2core" / "primitives" / "drag.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        func = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "drag_item"
        )
        kwonly = func.args.kwonlyargs
        defaults = func.args.kw_defaults  # выровнены с kwonlyargs (None = без дефолта)
        verify = next((a for a in kwonly if a.arg == "verify"), None)
        assert verify is not None, "drag_item обязан принимать verify"
        assert defaults[kwonly.index(verify)] is None, \
            "verify не должен иметь дефолта — drag без verify запрещён"
