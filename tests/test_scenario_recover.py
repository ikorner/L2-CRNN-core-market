"""Тесты сценарной точки входа recover (задача 6.4, spec market-scenarios:
«Сценарий recover как композиция примитивов без внутреннего состояния»).

Сценарий — тонкая обвязка поверх цепочки 5.8 (l2core/primitives/recover.py):
принимает cause (результат примитива/сценария с needs_recover) и target
(какое окно восстанавливаем), протягивает провайдеры от вызывающего слоя,
возвращает RecoverResult. Тесты — контракт вызова: правильный target,
протянутые провайдеры (ScriptedHid: последовательность шагов), прозрачная
эскалация, отмена, fail fast на cause без needs_recover.

Детерминизм (инвариант 7): ScriptedHid фиксирует действия и мутирует
скриптованный «мир», rng — random.Random с фиксированным seed, часы —
FakeClock (sleep продвигает время). Никаких обращений к живому экрану/окнам.
"""
import random

import numpy as np
import pytest

from l2core.config import (
    BehaviorConfig,
    CoreConfig,
    ResolutionConfig,
    TimingConfig,
)
from l2core.detector.anchor import Anchor
from l2core.detector.status import STATUS_FOUND, WindowStatus
from l2core.model import ElementSpec, Rect, UiModel, VariantSpec, WindowSpec
from l2core.primitives.navigation import RecipeResult
from l2core.primitives.activate import ActivateResult
from l2core.primitives.outcome import ActionOutcome, to_outcome
from l2core.primitives.recover import STEP_REOPEN, RecoverError
from l2core.scenarios.recover import ScenarioRecoverError, scenario_recover

TARGET = "shop_sell"
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
        self.hud_click_opens = False     # клик по HUD открывает целевое окно
        self.close_click_closes = False  # клик по close_point закрывает окно


class ScriptedHid:
    """HID с записью действий и позицией курсора; клики мутируют мир."""

    def __init__(self, clock, world):
        self.clock = clock
        self.world = world
        self.log = []
        self.pos = (0, 0)

    def _append(self, action, args):
        self.log.append((self.clock.now, action, args))

    def move_rel(self, dx, dy):
        self._append("move_rel", (dx, dy))
        self.pos = (self.pos[0] + dx, self.pos[1] + dy)

    def click(self, button=1, count=1, hold_ms=None):
        self._append("click", (button, count, hold_ms, self.pos))
        if self.pos == CLOSE_POINT and self.world.close_click_closes:
            self.world.current = None

    def key_press(self, code):
        self._append("key_press", (code,))

    def release_all(self):
        self._append("release_all", ())


class Cause:
    """Подделка результата примитива 5.x / сценария 6.x с needs_recover."""

    def __init__(self, success=False, cancelled=False, unconfirmed=False):
        self.success = success
        self.cancelled = cancelled
        self.unconfirmed = unconfirmed

    @property
    def needs_recover(self):
        return not self.success and not self.cancelled


def make_config():
    return CoreConfig(
        timing=TimingConfig(wait_default_timeout=1.0, verify_interval=0.1,
                            hover_pause=0.01, click_retry_count=1),
        behavior=BehaviorConfig(click_jitter_px=0),
        resolution=ResolutionConfig(width=1920, height=1080),
    )


def make_deps(world, hid, *, clock, goal=None, cancelled=None,
              on_escalation=None, resume=None, pause_timeout=5.0,
              recipe_runner="default"):
    """Стандартный набор провайдеров сценария поверх мира."""
    calls = {"escalation": [], "close_point": [], "recipe": []}

    def _on_escalation(result):
        calls["escalation"].append(result)

    def _close_point(window_id, frame):
        calls["close_point"].append(window_id)
        return CLOSE_POINT

    def window_status(frame):
        return WindowStatus(status=STATUS_FOUND, reason=None, anchor=ANCHOR,
                            content_loaded=True, content_detail="ok")

    if recipe_runner == "default":
        def recipe_runner(recipe_id):
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
        activate=lambda: ActivateResult(hwnd=1, already_active=True,
                                        attempts=0, cancelled=False),
        window_status=window_status,
        find_anchor=lambda f: ANCHOR if world.anchor_ok else None,
        goal_reached=goal or (lambda f: world.goal),
        config=make_config(),
        current_window=lambda f: world.current,
        close_point=_close_point,
        on_escalation=on_escalation or _on_escalation,
        resume=resume,
        pause_timeout=pause_timeout,
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


class TestCallContract:
    def test_cause_with_needs_recover_invokes_chain_with_target(self):
        # cause с needs_recover → цепочка 5.8 вызвана, target протянут,
        # цель достигнута (мир: клик по HUD открывает целевое окно).
        clock = FakeClock()
        world = World()
        world.hud_click_opens = True
        hid = ScriptedHid(clock, world)
        deps, calls = make_deps(world, hid, clock=clock)

        result = scenario_recover(Cause(), TARGET, **deps)

        assert result.success
        assert not result.escalated
        assert result.target_window == TARGET
        assert calls["escalation"] == []
        assert STEP_REOPEN in [s.name for s in result.steps]

    def test_providers_passed_through_scripted_sequence(self):
        # Провайдеры вызывающего слоя протянуты в цепочку: полный проход
        # esc -> свободный клик -> закрытие чужого окна -> переоткрытие HUD;
        # close_point получил id открытого окна, hud_locator — элемент HUD.
        clock = FakeClock()
        world = World()
        world.current = "market"          # открыто НЕ то окно
        world.hud_click_opens = True
        world.close_click_closes = True
        hid = ScriptedHid(clock, world)
        deps, calls = make_deps(world, hid, clock=clock)

        result = scenario_recover(Cause(), TARGET, **deps)

        assert result.success
        # Переоткрытие — рецепт (фейк без HID-клика).
        assert click_positions(hid) == [FREE_POINT, CLOSE_POINT]
        assert calls["close_point"] == ["market"]
        assert calls["recipe"] == ["open_shop_sell"]

    def test_cause_forwarded_release_all_first(self):
        # cause протянут в цепочку 5.8: при needs_recover release_all —
        # первое HID-действие (залипшая кнопка из 5.6 не исказит шаги).
        clock = FakeClock()
        world = World()
        world.hud_click_opens = True
        hid = ScriptedHid(clock, world)
        deps, _ = make_deps(world, hid, clock=clock)

        result = scenario_recover(Cause(unconfirmed=True), TARGET, **deps)

        assert result.success
        assert actions(hid)[0] == "release_all"

    def test_escalation_transparently_returned(self):
        # Эскалация не перехватывается сценарием: результат с escalated=True
        # возвращён как есть, событие ушло вызывающему слою (push-callback).
        clock = FakeClock()
        world = World()  # ничего не помогает: окно не открылось за таймаут
        hid = ScriptedHid(clock, world)
        deps, calls = make_deps(world, hid, clock=clock)

        result = scenario_recover(Cause(), TARGET, **deps)

        assert not result.success
        assert result.escalated
        assert to_outcome(result) is ActionOutcome.FAILED
        assert calls["escalation"] == [result]

    def test_cancel_transparently_returned(self):
        # Отмена (kill switch 8.1) проверяется цепочкой: cancelled=True,
        # событие escalation НЕ отправлено, HID не трогается.
        clock = FakeClock()
        world = World()
        hid = ScriptedHid(clock, world)
        deps, calls = make_deps(world, hid, clock=clock,
                                cancelled=lambda: True)

        result = scenario_recover(Cause(), TARGET, **deps)

        assert result.cancelled
        assert not result.success
        assert not result.escalated
        assert calls["escalation"] == []
        assert hid.log == []


class TestFailFast:
    def test_successful_cause_rejected(self):
        # cause без needs_recover (успех) — fail fast: recover не вызывается.
        clock = FakeClock()
        world = World()
        world.hud_click_opens = True
        hid = ScriptedHid(clock, world)
        deps, calls = make_deps(world, hid, clock=clock)

        with pytest.raises(ScenarioRecoverError):
            scenario_recover(Cause(success=True), TARGET, **deps)

        assert hid.log == []
        assert calls["escalation"] == []

    def test_cancelled_cause_rejected(self):
        # Отменённый cause — домен kill switch, recover не вызывается.
        clock = FakeClock()
        world = World()
        world.hud_click_opens = True
        hid = ScriptedHid(clock, world)
        deps, calls = make_deps(world, hid, clock=clock)

        with pytest.raises(ScenarioRecoverError):
            scenario_recover(Cause(cancelled=True), TARGET, **deps)

        assert hid.log == []
        assert calls["escalation"] == []

    def test_model_errors_propagate(self):
        # Ошибки модели (исполнителя рецептов — RecipeError) — контракт
        # 5.8: пробрасываются как есть, не оборачиваются сценарием.
        clock = FakeClock()
        world = World()
        hid = ScriptedHid(clock, world)
        deps, _ = make_deps(world, hid, clock=clock)
        from l2core.primitives.navigation import RecipeError

        def bad_runner(recipe_id):
            raise RecipeError(f"рецепт {recipe_id!r} отсутствует")

        deps["recipe_runner"] = bad_runner

        with pytest.raises(RecipeError):
            scenario_recover(Cause(), TARGET, **deps)


class TestThinWrapperGuard:
    """Тонкая обвязка: механика цепочки 5.8 здесь не дублируется —
    сценарий делегирует primitives/recover.py и не трогает HID сам."""

    def test_module_delegates_without_hid_mechanics(self):
        import ast
        import inspect

        import l2core.scenarios.recover as module

        tree = ast.parse(inspect.getsource(module))
        hid_imports = [
            node for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            and "adapters.hid" in ast.dump(node)
        ]
        assert hid_imports == [], (
            "сценарий recover не должен импортировать l2core.adapters.hid: "
            "HID-механика — внутри цепочки 5.8")
        # Делегирование: единственный вызов recover из primitives.
        recover_calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "recover"
        ]
        assert len(recover_calls) == 1
