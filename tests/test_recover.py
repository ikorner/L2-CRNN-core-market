"""Тесты общей recover-цепочки (задача 5.8, spec action-primitives:
«Recover-цепочка с пассивными шагами и эскалацией», «Штатный FAILED при
невыполненных предусловиях открытия», «Recover переоткрывает окно рецептом
навигации» (menu-quickslot-navigation), «Исчерпание цепочки»,
«Лавку нельзя создать в текущей зоне»).

Детерминизм (инвариант 7): ScriptedHid фиксирует действия и мутирует
скриптованный «мир» состояния, goal/current_window/find_anchor — чистые
функции мира, rng — random.Random с фиксированным seed, часы — FakeClock
(sleep продвигает время). Никаких обращений к живому экрану/окнам/HID.
"""
import random

import numpy as np
import pytest

from l2core.adapters.hid import KEY_ESC
from l2core.config import (
    BehaviorConfig,
    CoreConfig,
    ResolutionConfig,
    TimingConfig,
)
from l2core.detector.anchor import Anchor
from l2core.detector.status import STATUS_FOUND, WindowStatus
from l2core.model import ElementSpec, Rect, UiModel, VariantSpec, WindowSpec
from l2core.primitives.activate import ActivateGameError, ActivateResult
from l2core.primitives.navigation import RecipeResult
from l2core.primitives.outcome import ActionOutcome, to_outcome
from l2core.primitives.recover import (
    STEP_CLOSE_WINDOW,
    STEP_ESCALATION,
    STEP_FREE_CLICK,
    STEP_RECALIBRATE,
    STEP_REOPEN,
    STEP_ESC,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_GOAL_REACHED,
    STATUS_SKIPPED,
    RecoverError,
    RecoverResult,
    recover,
)

TARGET = "shop_sell"
OPEN_RECIPE = "open_shop_sell"
CLOSE_POINT = (700, 200)
FREE_POINT = (960, 540)  # центр 1920x1080 — свободная область по дефолту
ANCHOR = Anchor(window_id=TARGET, x=100, y=100, score=0.99, method="template")


class FakeClock:
    """Фейковые часы: sleep продвигает время."""

    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeScreen:
    """Экран: один и тот же кадр на все grab'ы."""

    def __init__(self):
        self.frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    def grab(self):
        return self.frame


class World:
    """Скриптованный мир состояния: что открыто, цель, якорь."""

    def __init__(self):
        self.current = None      # какое окно торговли открыто
        self.goal = False        # целевое состояние достигнуто
        self.anchor_ok = False   # якорь целевого окна на кадре
        self.esc_opens_goal = False    # ESC снимает блокировку -> цель
        self.esc_closes = False        # ESC закрывает верхнее окно
        self.free_click_opens_goal = False
        self.hud_click_opens = False   # клик по HUD открывает целевое окно
        self.close_click_closes = False  # клик по close_point закрывает окно


class ScriptedHid:
    """HID с записью действий и позицией курсора; клики мутируют мир."""

    def __init__(self, clock, world):
        self.clock = clock
        self.world = world
        self.log = []
        self.pos = (0, 0)
        self.release_all_fail = False

    def _append(self, action, args):
        self.log.append((self.clock.now, action, args))

    def move_rel(self, dx, dy):
        self._append("move_rel", (dx, dy))
        self.pos = (self.pos[0] + dx, self.pos[1] + dy)

    def click(self, button=1, count=1, hold_ms=None):
        self._append("click", (button, count, hold_ms, self.pos))
        if self.pos == CLOSE_POINT and self.world.close_click_closes:
            self.world.current = None
        elif self.pos == FREE_POINT and self.world.free_click_opens_goal:
            self.world.goal = True

    def key_press(self, code):
        self._append("key_press", (code,))
        if code == KEY_ESC:
            self.esc_count = getattr(self, "esc_count", 0) + 1
            if self.world.esc_opens_goal:
                self.world.goal = True
            # Первый ESC (пассивный шаг цепочки) мог уйти в тултип/модалку;
            # закрывает верхнее окно второй ESC — шаг «закрыть окно».
            if self.world.esc_closes and self.esc_count >= 2:
                self.world.current = None

    def release_all(self):
        self._append("release_all", ())
        if self.release_all_fail:
            from l2core.adapters.hid import HidError
            raise HidError("транспортный сбой")


class Cause:
    """Подделка результата примитива 5.x с needs_recover."""

    def __init__(self, success=False, cancelled=False, unconfirmed=False):
        self.success = success
        self.cancelled = cancelled
        self.unconfirmed = unconfirmed


def make_config():
    return CoreConfig(
        timing=TimingConfig(wait_default_timeout=1.0, verify_interval=0.1,
                            hover_pause=0.01, click_retry_count=1),
        behavior=BehaviorConfig(click_jitter_px=0),
        resolution=ResolutionConfig(width=1920, height=1080),
    )


def make_deps(world, hid, *, clock, goal=None, current_window=None,
              close_point=None, recipe_runner="default", on_escalation=None,
              resume=None, pause_timeout=5.0, capture_state=None,
              cause=None, cancelled=None, find_anchor=None):
    """Стандартный набор инъекций recover поверх мира."""
    config = make_config()

    def activate():
        return ActivateResult(hwnd=1, already_active=True, attempts=0,
                              cancelled=False)

    def window_status(frame):
        return WindowStatus(status=STATUS_FOUND, reason=None, anchor=ANCHOR,
                            content_loaded=True, content_detail="ok")

    def _find_anchor(frame):
        return ANCHOR if world.anchor_ok else None

    calls = {"capture_state": 0, "escalation": [], "recipe": []}

    def _capture_state():
        calls["capture_state"] += 1

    def _on_escalation(result):
        calls["escalation"].append(result)

    if recipe_runner == "default":
        def recipe_runner(recipe_id):
            # Исполнитель рецепта — фейк: открытие мира по флагу
            # (как раньше клик по HUD-иконке).
            calls["recipe"].append(recipe_id)
            if world.hud_click_opens:
                world.current = TARGET
                world.goal = True
            ok = world.goal
            return RecipeResult(
                recipe_id, success=ok, attempts=1,
                failures=() if ok else ("verify открытия не сошёлся",))

    deps = dict(
        recipe_runner=recipe_runner,
        screen=FakeScreen(),
        hid=hid,
        cursor_pos=lambda: hid.pos,
        activate=activate,
        window_status=window_status,
        find_anchor=find_anchor or _find_anchor,
        goal_reached=(goal or (lambda f: world.goal)),
        config=config,
        current_window=current_window
        or (lambda f: world.current),
        close_point=close_point,
        on_escalation=on_escalation or _on_escalation,
        resume=resume,
        pause_timeout=pause_timeout,
        capture_state=capture_state or _capture_state,
        cause=cause,
        cancelled=cancelled,
        rng=random.Random(42),
        clock=clock.monotonic,
        sleep=clock.sleep,
    )
    return deps, calls


def click_positions(hid):
    return [args[-1] for _, a, args in hid.log if a == "click"]


def actions(hid):
    return [a for _, a, _ in hid.log]


class TestGoalReached:
    def test_goal_already_reached_short_circuits(self):
        # Цель достигнута на входе: цепочка не исполняется, HID не трогается.
        clock = FakeClock()
        world = World()
        world.goal = True
        hid = ScriptedHid(clock, world)
        deps, calls = make_deps(world, hid, clock=clock)

        result = recover(TARGET, **deps)

        assert result.success
        assert not result.escalated
        assert result.steps == ()
        assert hid.log == []
        assert calls["escalation"] == []

    def test_esc_step_reaches_goal_chain_ends(self):
        # ESC снимает блокировку (модалка) — цепочка завершается досрочно
        # после первого пассивного шага: свободный клик/закрытие/HUD нет.
        clock = FakeClock()
        world = World()
        world.esc_opens_goal = True
        hid = ScriptedHid(clock, world)
        deps, calls = make_deps(world, hid, clock=clock)

        result = recover(TARGET, **deps)

        assert result.success
        assert [s.name for s in result.steps] == [STEP_ESC]
        assert result.steps[0].status == STATUS_GOAL_REACHED
        assert "click" not in actions(hid)
        assert calls["escalation"] == []

    def test_free_click_step_reaches_goal(self):
        clock = FakeClock()
        world = World()
        world.free_click_opens_goal = True
        hid = ScriptedHid(clock, world)
        deps, calls = make_deps(world, hid, clock=clock)

        result = recover(TARGET, **deps)

        assert result.success
        assert [s.name for s in result.steps] == [STEP_ESC, STEP_FREE_CLICK]
        assert result.steps[-1].status == STATUS_GOAL_REACHED
        assert calls["escalation"] == []


class TestFullChain:
    def test_full_pass_reopens_via_hud(self):
        # Ничего не помогает, пока не переоткрыли через HUD: полный проход
        # esc -> free_click -> close_window -> reopen_hud; перекалибровка
        # после достижения цели не выполняется.
        clock = FakeClock()
        world = World()
        world.current = "market"          # открыто НЕ то окно
        world.hud_click_opens = True
        world.close_click_closes = True
        hid = ScriptedHid(clock, world)

        def close_point(window_id, frame):
            assert window_id == "market"
            return CLOSE_POINT if world.close_click_closes else None

        anchor_polls = []

        def find_anchor(frame):
            anchor_polls.append(1)
            return ANCHOR if world.anchor_ok else None

        deps, calls = make_deps(world, hid, clock=clock,
                                close_point=close_point,
                                find_anchor=find_anchor)

        result = recover(TARGET, **deps)

        assert result.success
        names = [s.name for s in result.steps]
        assert names == [STEP_ESC, STEP_FREE_CLICK, STEP_CLOSE_WINDOW, STEP_REOPEN]
        assert result.steps[-1].status == STATUS_GOAL_REACHED
        # Перекалибровка (поиск якоря) не выполнялась — цель достигнута раньше.
        assert anchor_polls == []
        assert calls["escalation"] == []
        # Клики: свободная область, кнопка закрытия чужого окна;
        # переоткрытие — рецепт (фейк без HID-клика).
        assert click_positions(hid) == [FREE_POINT, CLOSE_POINT]


class TestCloseWindow:
    def test_close_skipped_when_nothing_open(self):
        clock = FakeClock()
        world = World()  # current=None — окна нет
        world.hud_click_opens = True
        hid = ScriptedHid(clock, world)
        deps, _ = make_deps(world, hid, clock=clock,
                            close_point=lambda w, f: CLOSE_POINT)

        result = recover(TARGET, **deps)

        close_step = [s for s in result.steps if s.name == STEP_CLOSE_WINDOW][0]
        assert close_step.status == STATUS_SKIPPED
        assert "нечего" in close_step.detail or "нет" in close_step.detail
        # Кнопка закрытия не кликалась: только свободная область;
        # переоткрытие — рецепт (фейк без HID-клика).
        assert click_positions(hid) == [FREE_POINT]

    def test_close_skipped_when_target_already_open(self):
        clock = FakeClock()
        world = World()
        world.current = TARGET  # открыто ЦЕЛЕВОЕ — закрывать нечего
        world.hud_click_opens = True
        hid = ScriptedHid(clock, world)
        deps, _ = make_deps(world, hid, clock=clock)

        result = recover(TARGET, **deps)

        close_step = [s for s in result.steps if s.name == STEP_CLOSE_WINDOW][0]
        assert close_step.status == STATUS_SKIPPED

    def test_close_falls_back_to_esc_without_close_point(self):
        # Кнопки закрытия в модели нет (close_point -> None): закрытие —
        # ESC (документированный fallback).
        clock = FakeClock()
        world = World()
        world.current = "shop_buy"  # не то окно, без кнопки закрытия
        world.esc_closes = True
        world.hud_click_opens = True
        hid = ScriptedHid(clock, world)
        deps, _ = make_deps(world, hid, clock=clock,
                            close_point=lambda w, f: None)

        result = recover(TARGET, **deps)

        close_step = [s for s in result.steps if s.name == STEP_CLOSE_WINDOW][0]
        assert close_step.status == STATUS_DONE
        assert "ESC" in close_step.detail
        assert ("key_press", (KEY_ESC,)) in [
            (a, args) for _, a, args in hid.log]

    def test_close_step_diagnoses_failure_and_continues(self):
        # Клик по кнопке закрытия не закрыл окно: шаг failed с диагностикой,
        # цепочка продолжается (recover — путь к цели, не ритуал).
        clock = FakeClock()
        world = World()
        world.current = "market"  # и после close_click_closes=False остаётся
        world.hud_click_opens = True
        hid = ScriptedHid(clock, world)
        deps, _ = make_deps(world, hid, clock=clock,
                            close_point=lambda w, f: CLOSE_POINT)

        result = recover(TARGET, **deps)

        close_step = [s for s in result.steps if s.name == STEP_CLOSE_WINDOW][0]
        assert close_step.status == STATUS_FAILED
        assert result.success  # цель достигнута на шаге reopen
        assert [s.name for s in result.steps][-1] == STEP_REOPEN


class TestReopenTimeout:
    def test_opening_timeout_failed_plus_escalation_no_retry(self):
        # «Лавку нельзя создать в текущей зоне»: рецепт open_shop_sell не
        # открыл окно -> FAILED, фиксация состояния, событие escalation,
        # повторных попыток открытия нет.
        clock = FakeClock()
        world = World()
        hid = ScriptedHid(clock, world)
        deps, calls = make_deps(world, hid, clock=clock)

        result = recover(TARGET, **deps)

        assert not result.success
        assert result.escalated
        assert to_outcome(result) is ActionOutcome.FAILED
        # Рецепт вызван ровно один раз.
        assert calls["recipe"] == [OPEN_RECIPE]
        # Фиксация состояния и событие escalation — ровно по разу.
        assert calls["capture_state"] == 1
        assert len(calls["escalation"]) == 1
        assert calls["escalation"][0] is result
        # Перекалибровка не запускалась: после FAILED открытия — сразу эскалация.
        assert STEP_RECALIBRATE not in [s.name for s in result.steps]
        reopen = [s for s in result.steps if s.name == STEP_REOPEN][0]
        assert reopen.status == STATUS_FAILED
        assert any(OPEN_RECIPE in f for f in result.failures)

    def test_recipe_runner_missing_skips_reopen(self):
        # Цель достигается после перекалибровки (якорь стабилен) — reopen
        # пропущен (recipe_runner не задан), эскалации нет.
        clock = FakeClock()
        world = World()
        world.anchor_ok = True
        hid = ScriptedHid(clock, world)
        seen = [0]

        def find_anchor(frame):
            seen[0] = seen[0] + 1 if world.anchor_ok else 0
            return ANCHOR if world.anchor_ok else None

        def goal(frame):
            return seen[0] >= 2

        deps, calls = make_deps(
            world, hid, clock=clock,
            recipe_runner=None,
            find_anchor=find_anchor, goal=goal,
        )

        result = recover(TARGET, **deps)

        reopen = [s for s in result.steps if s.name == STEP_REOPEN][0]
        assert reopen.status == STATUS_SKIPPED
        assert "recipe_runner" in reopen.detail
        assert calls["recipe"] == []
        assert result.success
        assert calls["escalation"] == []


class TestRecalibrateAndEscalation:
    def test_recalibrate_reaches_goal(self):
        # Переоткрытие невозможно (HUD не найден), но якорь стабилен и
        # цель достигнута после перекалибровки — эскалации нет.
        clock = FakeClock()
        world = World()
        world.anchor_ok = True
        hid = ScriptedHid(clock, world)
        seen = [0]

        def find_anchor(frame):
            seen[0] = seen[0] + 1 if world.anchor_ok else 0
            return ANCHOR if world.anchor_ok else None

        def goal(frame):
            return seen[0] >= 2

        deps, calls = make_deps(
            world, hid, clock=clock,
            recipe_runner=None,
            find_anchor=find_anchor, goal=goal,
        )

        result = recover(TARGET, **deps)

        assert result.success
        step = [s for s in result.steps if s.name == STEP_RECALIBRATE][0]
        assert step.status == STATUS_GOAL_REACHED
        assert calls["escalation"] == []

    def test_recalibrate_unstable_then_escalation(self):
        # Цепочка исчерпана: эскалация = событие стратегии + пауза до resume;
        # переоткрытие пропущено (HUD не найден), якорь нестабилен.
        clock = FakeClock()
        world = World()  # goal False, anchor нестабилен всегда
        hid = ScriptedHid(clock, world)
        resume_calls = []

        def resume():
            resume_calls.append(1)
            return len(resume_calls) >= 3

        deps, calls = make_deps(world, hid, clock=clock, resume=resume,
                                pause_timeout=5.0,
                                recipe_runner=None)

        result = recover(TARGET, **deps)

        assert not result.success
        assert result.escalated
        assert to_outcome(result) is ActionOutcome.FAILED
        # Эскалация — последний шаг, событие отправлено до паузы.
        assert result.steps[-1].name == STEP_ESCALATION
        recalibrate = [s for s in result.steps
                       if s.name == STEP_RECALIBRATE][0]
        assert recalibrate.status == STATUS_FAILED
        assert len(calls["escalation"]) == 1
        assert len(resume_calls) == 3  # пауза дождалась resume
        assert not result.cancelled

    def test_escalation_pause_waits_until_timeout_without_resume(self):
        clock = FakeClock()
        world = World()
        hid = ScriptedHid(clock, world)
        deps, calls = make_deps(world, hid, clock=clock, resume=None,
                                pause_timeout=2.0)

        result = recover(TARGET, **deps)

        assert result.escalated
        # Без resume пауза дождалась таймаута (фейковые часы, мгновенно).
        assert clock.now >= 2.0


class TestPrologueAndCancel:
    def test_activate_failure_escalates_without_hid(self):
        # Окно игры не активируется: HID не трогаем, сразу эскалация
        # (осмысленная ошибка вместо слепых действий, инвариант 9).
        clock = FakeClock()
        world = World()
        hid = ScriptedHid(clock, world)
        deps, calls = make_deps(world, hid, clock=clock)

        def activate():
            raise ActivateGameError(reason="window_not_found",
                                    detail="окно недоступно")

        deps["activate"] = activate
        result = recover(TARGET, **deps)

        assert not result.success
        assert result.escalated
        assert hid.log == []
        assert len(calls["escalation"]) == 1
        assert any("window_not_found" in f for f in result.failures)

    def test_cancel_stops_chain_without_escalation(self):
        clock = FakeClock()
        world = World()
        hid = ScriptedHid(clock, world)
        deps, calls = make_deps(world, hid, clock=clock,
                                cancelled=lambda: True)

        result = recover(TARGET, **deps)

        assert result.cancelled
        assert not result.success
        assert not result.escalated
        assert calls["escalation"] == []
        assert hid.log == []

    def test_cancel_during_recalibrate_no_escalation(self):
        # Отмена в ожидании стабильности якоря: результат cancelled, событие
        # escalation НЕ отправляется (отмена — домен kill switch, 8.1).
        clock = FakeClock()
        world = World()  # anchor нестабилен: recalibrate ждёт до отмены
        hid = ScriptedHid(clock, world)
        deps, calls = make_deps(world, hid, clock=clock,
                                cancelled=lambda: clock.now >= 2.5,
                                recipe_runner=None)

        result = recover(TARGET, **deps)

        assert result.cancelled
        assert not result.escalated
        assert calls["escalation"] == []

    def test_cancel_during_escalation_pause(self):
        clock = FakeClock()
        world = World()
        hid = ScriptedHid(clock, world)
        fired = []

        def cancelled():
            # Отмена срабатывает уже в паузе эскалации (цепочка до неё
            # потребляет 3 таймаута wait_default_timeout = 3.0 с).
            return clock.now >= 4.0

        deps, calls = make_deps(world, hid, clock=clock, cancelled=cancelled,
                                resume=None, pause_timeout=10.0)
        deps["on_escalation"] = lambda r: fired.append(r)

        result = recover(TARGET, **deps)

        assert result.cancelled
        assert result.escalated  # эскалация уже определена
        # Событие ушло до паузы: оно о неотменённом результате, возвращённый
        # — с cancelled=True (отмена во время паузы).
        assert len(fired) == 1
        assert fired[0].escalated and not fired[0].cancelled


class TestReleaseAll:
    def test_release_all_on_cause_needs_recover(self):
        # Залипшая кнопка из 5.6 (needs_recover от примитива): release_all —
        # первое HID-действие цепочки, до любых шагов.
        clock = FakeClock()
        world = World()
        world.esc_opens_goal = True
        hid = ScriptedHid(clock, world)
        deps, _ = make_deps(world, hid, clock=clock,
                            cause=Cause(success=False, unconfirmed=True))

        result = recover(TARGET, **deps)

        assert result.success
        assert actions(hid)[0] == "release_all"
        assert actions(hid).count("release_all") == 1

    def test_release_all_without_cause_not_called(self):
        clock = FakeClock()
        world = World()
        world.esc_opens_goal = True
        hid = ScriptedHid(clock, world)
        deps, _ = make_deps(world, hid, clock=clock)

        recover(TARGET, **deps)

        assert "release_all" not in actions(hid)

    def test_release_all_hid_error_swallowed(self):
        # Сбой release_all не роняет recover: фиксируется в failures,
        # цепочка продолжается (best-effort).
        clock = FakeClock()
        world = World()
        world.esc_opens_goal = True
        hid = ScriptedHid(clock, world)
        hid.release_all_fail = True
        deps, _ = make_deps(world, hid, clock=clock, cause=Cause())

        result = recover(TARGET, **deps)

        assert result.success
        assert any("release_all" in f for f in result.failures)
