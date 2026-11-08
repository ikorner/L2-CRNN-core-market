"""Тесты click_element (задача 5.3, spec action-primitives: «Каждое действие
верифицируется, при неудаче — retry → on_fail → recover», «Устойчивость к
hover и тултипам»; safety: клики без предусловий запрещены — инвариант 9).

Детерминизм (инвариант 7): ScriptedHid фиксирует действия с метками фейкового
времени, скриптованные run_checks/find_anchor/status, rng — random.Random с
фиксированным seed, часы — FakeClock (sleep продвигает время). Никаких
обращений к живому экрану/окнам/HID.
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
from l2core.model import Check, ElementSpec, Rect, UiModel, VariantSpec, WindowSpec
from l2core.primitives.activate import ActivateGameError, ActivateResult
from l2core.primitives.click import (
    TOOLTIP_SHIFT_PX,
    ClickElementError,
    ClickPreconditionError,
    ClickResult,
    VerifySpec,
    click_element,
)

ELEMENT_ID = "tab"
ANCHOR = Anchor(window_id="market", x=202, y=103, score=0.99, method="template")
RECT = (10, 20, 100, 30)  # center -> (50, 15) от якоря
ON_FAIL = ("press_esc", "click_free_area")


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


class FakeCursor:
    """Позиция курсора: move_rel двигает."""

    def __init__(self, pos=(500, 300)):
        self.pos = pos

    def get(self):
        return self.pos

    def track(self, hid):
        original = hid.move_rel

        def move_rel(dx, dy):
            self.pos = (self.pos[0] + dx, self.pos[1] + dy)
            original(dx, dy)

        hid.move_rel = move_rel


class FakeChecker:
    """Скриптованная оценка проверок.

    Различение «проверка присутствия после hover» (одна template-проверка
    элемента с вариантом normal) и verify-спецификации (вариант active и
    прочие) — по имени варианта первой проверки.
    """

    def __init__(self, presence=None, verify=None):
        self._presence = list(presence or [])
        self._verify = list(verify or [])
        self.calls = []

    def __call__(self, checks, frame, anchor):
        kind = "presence" if checks[0].variant == "normal" else "verify"
        self.calls.append((kind, tuple(checks)))
        queue = self._presence if kind == "presence" else self._verify
        if queue:
            passed = queue.pop(0)
        else:
            passed = True
        return _report(passed, checks)


def _report(passed, checks):
    from l2core.primitives.click import CheckReport

    failures = () if passed else tuple(f"fail:{c.element}/{c.variant}" for c in checks)
    return CheckReport(passed=passed, failures=failures)


def make_model(variants=("normal", "active"), click_point=None, hud=False):
    specs = {
        name: VariantSpec(
            name=name, template=f"test/{ELEMENT_ID}_{name}.png",
            click_point=click_point if name == "active" else None,
        )
        for name in variants
    }
    element = ElementSpec(
        id=ELEMENT_ID,
        role="click_target",
        rect=None if hud else Rect(*RECT),
        hud=hud,
        search_area=Rect(1000, 420, 200, 160) if hud else None,
        variants=specs,
    )
    return UiModel(
        path="test",
        window=WindowSpec(id="market", title="Обзор лавок", size=(1008, 674)),
        elements={ELEMENT_ID: element},
        states={},
        transitions=(),
        ocr_regions={},
        table_geometry=None,
    )


def make_config(**timing_kwargs):
    timing = TimingConfig(verify_interval=0.05, hover_pause=0.2, **timing_kwargs)
    return CoreConfig(
        timing=timing,
        behavior=BehaviorConfig(click_jitter_px=3, press_delay_min_ms=40, press_delay_max_ms=120),
        resolution=ResolutionConfig(width=1920, height=1080),
    )


def found_status(frame):
    return WindowStatus(
        status=STATUS_FOUND, reason=None, anchor=ANCHOR,
        content_loaded=None, content_detail="",
    )


def run_click(hid, clock, checker, element_id=ELEMENT_ID, model=None, verify=None,
              activate=None, status=None, find_anchor=None, on_fail=(),
              rng_seed=7, **kwargs):
    model = model or make_model()
    verify = verify or VerifySpec(
        checks=(Check(kind="template", element=ELEMENT_ID, variant="active"),),
        expected="target_state",
    )
    activate = activate or (lambda: ActivateResult(hwnd=1, already_active=True, attempts=0, cancelled=False))
    return click_element(
        element_id,
        model=model,
        verify=verify,
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


class TestSuccessfulClick:
    def test_click_tab_verify_ok(self):
        # Успешный клик по вкладке: hover -> пауза -> click -> verify OK.
        clock = FakeClock()
        hid = ScriptedHid(clock)
        cursor = FakeCursor()
        cursor.track(hid)
        result = run_click(hid, clock, FakeChecker())

        assert isinstance(result, ClickResult)
        assert result.success is True
        assert result.attempts == 1
        assert result.cancelled is False
        assert result.on_fail_executed is False
        assert result.needs_recover is False

        actions = [a for _, a, _ in hid.log]
        assert actions == ["move_rel", "click"]
        # Центр rect (50,15) от якоря (202,103) с джиттером seed=7:
        rng = random.Random(7)
        jx, jy = rng.randint(-3, 3), rng.randint(-3, 3)
        hold = rng.randint(40, 120)
        target = (202 + RECT[0] + 50 + jx, 103 + RECT[1] + 15 + jy)
        start = (500, 300)
        assert hid.log[0][2] == (target[0] - start[0], target[1] - start[1])
        # hold_ms из профиля (config.behavior), не из вендорного random.
        assert hid.log[1][2] == (1, 1, hold)

    def test_hover_pause_precedes_click(self):
        # Hover-пауза (timing.hover_pause) — между move и click, без sleep:
        # клик зафиксирован не раньше, чем часы дошли до hover_pause.
        clock = FakeClock()
        hid = ScriptedHid(clock)
        FakeCursor().track(hid)
        run_click(hid, clock, FakeChecker())

        move_t = next(t for t, a, _ in hid.log if a == "move_rel")
        click_t = next(t for t, a, _ in hid.log if a == "click")
        assert move_t == 0.0  # наводимся сразу
        assert click_t >= 0.2  # hover_pause прошёл до клика

    def test_click_point_from_variant(self):
        # click_point варианта active используется как точка клика (от rect).
        clock = FakeClock()
        hid = ScriptedHid(clock)
        cursor = FakeCursor()
        cursor.track(hid)
        model = make_model(click_point=(20, 7))
        run_click(hid, clock, FakeChecker(), model=model)

        rng = random.Random(7)
        jx, jy = rng.randint(-3, 3), rng.randint(-3, 3)
        target = (202 + RECT[0] + 20 + jx, 103 + RECT[1] + 7 + jy)
        assert hid.log[0][2] == (target[0] - 500, target[1] - 300)

    def test_rng_seed_deterministic(self):
        # Джиттер/задержки детерминированы при фиксированном seed: два прогона
        # с одним seed дают одинаковую последовательность действий.
        logs = []
        for _ in range(2):
            clock = FakeClock()
            hid = ScriptedHid(clock)
            FakeCursor().track(hid)
            run_click(hid, clock, FakeChecker())
            logs.append(hid.log)
        assert logs[0] == logs[1]

    def test_cancelled_flag_result(self):
        # Отмена (kill switch 8.1) — флаг в результате, не исключение; клика нет.
        clock = FakeClock()
        hid = ScriptedHid(clock)
        result = run_click(hid, clock, FakeChecker(), cancelled=lambda: True)
        assert result.cancelled is True
        assert result.success is False
        assert result.attempts == 0
        assert result.needs_recover is False
        assert hid.log == []


class TestVerifyRetry:
    def test_verify_fail_retried_until_success(self):
        # Verify не проходит дважды, третья попытка — OK: ровно 3 клика,
        # on_fail не исполнялся.
        clock = FakeClock()
        hid = ScriptedHid(clock)
        FakeCursor().track(hid)
        checker = FakeChecker(verify=[False, False, True])
        result = run_click(hid, clock, checker, on_fail=ON_FAIL)

        assert result.success is True
        assert result.attempts == 3
        clicks = [a for _, a, _ in hid.log if a == "click"]
        assert len(clicks) == 3
        assert result.on_fail_executed is False

    def test_retry_count_from_config(self):
        # Число попыток = 1 + timing.click_retry_count.
        clock = FakeClock()
        hid = ScriptedHid(clock)
        FakeCursor().track(hid)
        checker = FakeChecker(verify=[False, False])
        result = run_click(
            hid, clock, checker, on_fail=(), config=make_config(click_retry_count=1)
        )
        assert result.success is False
        assert result.attempts == 2
        assert len([a for _, a, _ in hid.log if a == "click"]) == 2

    def test_exhausted_verify_runs_on_fail_then_recover_signal(self):
        # Исчерпание retry -> шаги on_fail[] исполнены -> сигнал recover
        # (needs_recover), повторных кликов сверх лимита нет.
        clock = FakeClock()
        hid = ScriptedHid(clock)
        cursor = FakeCursor()
        cursor.track(hid)
        checker = FakeChecker(verify=[False, False, False])
        result = run_click(
            hid, clock, checker, on_fail=ON_FAIL, free_area_point=(100, 700),
            cursor_pos=cursor.get,
        )

        assert result.success is False
        assert result.attempts == 3  # 1 + click_retry_count(2)
        assert result.on_fail_executed is True
        assert result.needs_recover is True
        assert result.failures  # диагностика провалов

        actions = [a for _, a, _ in hid.log]
        # 3 попытки (move+click), затем press_esc и клик в свободную область.
        assert actions.count("click") == 4  # 3 попытки + free_area
        assert actions[-3:] == ["key_press", "move_rel", "click"]
        assert hid.log[-3][2] == (0xB1,)  # KEY_ESC
        # Клик в свободную область — в заданную точку (курсор доведён туда).
        assert cursor.pos == (100, 700)

    def test_no_on_fail_steps_empty(self):
        # on_fail не задан: результат — сигнал recover без esc/кликов.
        clock = FakeClock()
        hid = ScriptedHid(clock)
        FakeCursor().track(hid)
        checker = FakeChecker(verify=[False, False, False])
        result = run_click(hid, clock, checker, on_fail=())
        assert result.on_fail_executed is False
        assert result.needs_recover is True
        assert [a for _, a, _ in hid.log].count("click") == 3

    def test_unknown_on_fail_step_fails_fast(self):
        clock = FakeClock()
        hid = ScriptedHid(clock)
        FakeCursor().track(hid)
        checker = FakeChecker(verify=[False])
        with pytest.raises(ClickElementError):
            run_click(
                hid, clock, checker, on_fail=("press_esc", "teleport"),
                config=make_config(click_retry_count=0),
            )


class TestPrologue:
    def test_activate_failure_escalates_no_click(self):
        # Пролог: activate_game не удался -> ActivateGameError проброшена,
        # HID не трогали (no blind clicks).
        clock = FakeClock()
        hid = ScriptedHid(clock)

        def failing_activate():
            raise ActivateGameError(reason="focus_not_acquired", detail="нет фокуса")

        with pytest.raises(ActivateGameError):
            run_click(hid, clock, FakeChecker(), activate=failing_activate)
        assert hid.log == []

    def test_status_not_found_blocks_click(self):
        # Пролог: статус окна не found -> ClickPreconditionError, клика нет.
        clock = FakeClock()
        hid = ScriptedHid(clock)

        def occluded(frame):
            return WindowStatus(
                status=STATUS_OCCLUDED, reason="anchor_lost", anchor=None,
                content_loaded=None, content_detail="",
            )

        with pytest.raises(ClickPreconditionError) as exc_info:
            run_click(hid, clock, FakeChecker(), status=occluded)
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

        with pytest.raises(ClickPreconditionError) as exc_info:
            run_click(hid, clock, FakeChecker(), status=not_found)
        assert "window_not_found" in str(exc_info.value)
        assert hid.log == []


class TestHoverAndTooltip:
    def test_hover_changed_element_blocks_click_that_attempt(self):
        # Hover изменил вид элемента (проверка присутствия не прошла): в этой
        # попытке КЛИКА НЕТ, попытка потрачена, дальше — обычный retry.
        clock = FakeClock()
        hid = ScriptedHid(clock)
        FakeCursor().track(hid)
        checker = FakeChecker(presence=[False, True], verify=[True])
        result = run_click(hid, clock, checker)

        assert result.success is True
        assert result.attempts == 2
        # Первый клик только со второй попытки.
        clicks = [t for t, a, _ in hid.log if a == "click"]
        assert len(clicks) == 1
        # Первая попытка: move (hover) и остановка — без click.
        first_click = clicks[0]
        moves_before = [t for t, a, _ in hid.log if a == "move_rel" and t < first_click]
        # hover (попытка 1) + тултип-сдвиг (попытка 2) + hover (попытка 2).
        assert len(moves_before) == 3

    def test_tooltip_shift_before_retry(self):
        # Тултип-гипотеза: после verify-fail курсор сдвигается в сторону на
        # TOOLTIP_SHIFT_PX и ждёт паузу, затем действие повторяется.
        clock = FakeClock()
        hid = ScriptedHid(clock)
        cursor = FakeCursor()
        cursor.track(hid)
        checker = FakeChecker(verify=[False, True])
        run_click(hid, clock, checker)

        moves = [(t, args) for t, a, args in hid.log if a == "move_rel"]
        # Первый move — hover к цели; второй — тултип-сдвиг ровно на константу.
        assert moves[1][1] == (TOOLTIP_SHIFT_PX, TOOLTIP_SHIFT_PX)
        # Между сдвигом и следующим hover — пауза (ожидание исчезновения).
        shift_t = moves[1][0]
        next_move_t = moves[2][0]
        assert next_move_t - shift_t >= 0.2 - 1e-9  # float-дрейф фейковых часов

    def test_anchor_lost_on_attempt_fails_attempt(self):
        # Якорь не найден на свежем кадре попытки: попытка потрачена без клика.
        clock = FakeClock()
        hid = ScriptedHid(clock)
        FakeCursor().track(hid)
        anchors = [ANCHOR, None, ANCHOR]  # пролог status, попытка 2, verify 2
        state = {"calls": 0}

        def find_anchor(frame):
            state["calls"] += 1
            return ANCHOR if state["calls"] != 2 else None

        checker = FakeChecker(verify=[True])
        result = run_click(hid, clock, checker, find_anchor=find_anchor)
        assert result.success is True
        assert result.attempts == 2


class TestModelBoundaries:
    def test_unknown_element_fails_fast(self):
        clock = FakeClock()
        hid = ScriptedHid(clock)
        with pytest.raises(ClickElementError):
            run_click(hid, clock, FakeChecker(), element_id="nope")
        assert hid.log == []

    def test_hud_element_requires_locator(self):
        # HUD-элемент без hud_locator кликнуть нельзя (позиция неизвестна —
        # слепой клик запрещён).
        clock = FakeClock()
        hid = ScriptedHid(clock)
        with pytest.raises(ClickElementError):
            run_click(hid, clock, FakeChecker(), model=make_model(hud=True))
        assert hid.log == []

    def test_hud_element_with_locator(self):
        # С hud_locator точка берётся из локатора (+wiring с matcher — 6.x).
        clock = FakeClock()
        hid = ScriptedHid(clock)
        cursor = FakeCursor()
        cursor.track(hid)
        result = run_click(
            hid, clock, FakeChecker(), model=make_model(hud=True),
            hud_locator=lambda element, variant, frame: (1086, 484),
        )
        assert result.success is True
        rng = random.Random(7)
        jx, jy = rng.randint(-3, 3), rng.randint(-3, 3)
        assert hid.log[0][2] == (1086 - 500 + jx, 484 - 300 + jy)


class TestVerifyContract:
    def test_click_without_verify_is_statically_required(self):
        # Guard: публичный API клика требует verify-спецификацию — параметр
        # `verify` обязателен (keyword-only, без дефолта). Клики без verify
        # запрещены (spec action-primitives); guard ловит регрессию сигнатуры.
        source = (Path(__file__).resolve().parent.parent
                  / "l2core" / "primitives" / "click.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        func = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "click_element"
        )
        kwonly = func.args.kwonlyargs
        defaults = func.args.kw_defaults  # выровнены с kwonlyargs (None = без дефолта)
        verify = next((a for a in kwonly if a.arg == "verify"), None)
        assert verify is not None, "click_element обязан принимать verify"
        assert defaults[kwonly.index(verify)] is None, \
            "verify не должен иметь дефолта — клики без verify запрещены"
