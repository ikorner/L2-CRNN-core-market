"""Исполнитель рецептов навигации (Task 8, l2core.primitives.navigation).

Детерминизм (инвариант 7): ScriptedHid/FakeClock/FakeScreen из test_click,
скриптованный evaluate (оценщик проверок по кадру), rng — seed 7,
скриптованный hud_locator. Никаких обращений к живому экрану/HID.
"""

import random

import pytest

from l2core.adapters.hid import BTN_LEFT, HidError
from l2core.model import Check, ElementSpec, Rect, VariantSpec
from l2core.model.navigation import NavigationModel, Recipe, RecipeStep
from l2core.primitives.activate import ActivateResult
from l2core.primitives.click import CheckReport
from l2core.primitives.navigation import RecipeError, run_recipe
from tests.test_click import FakeClock, FakeScreen, ScriptedHid, make_config

MENU_CHECKS = (Check(kind="template", element="menu_anchor", variant="normal"),)
ICON_CHECKS = (Check(kind="template", element="menu_market_icon", variant="normal"),)
MENU_GONE_CHECKS = (Check(kind="template", element="menu_anchor",
                          variant="normal", expect="absent"),)


def _hud(element_id):
    return ElementSpec(
        id=element_id, role="icon_sample", hud=True,
        search_area=Rect(100, 100, 400, 200),
        variants={"normal": VariantSpec("normal", f"navigation/{element_id}.png",
                                        click_point=(20, 20))})


def make_nav(*, with_alternate=True):
    """Навигационная модель: open_market (key X + click иконки меню,
    alternate open_market_via_hamburger) и open_shop_buy (when/skip_if +
    click quick-slot)."""
    elements = {
        "menu_anchor": _hud("menu_anchor"),
        "hamburger_button": _hud("hamburger_button"),
        "quickslot_buy_icon": _hud("quickslot_buy_icon"),
        "menu_market_icon": ElementSpec(
            id="menu_market_icon", role="click_target", hud=False,
            rect=Rect(40, 60, 48, 48),
            variants={"normal": VariantSpec(
                "normal", "navigation/menu_market_icon.png",
                click_point=(24, 24))}),
    }
    key_step = RecipeStep(
        action="key", element=None, key="X",
        verify=MENU_CHECKS, on_fail=("press_esc",),
        skip_if=MENU_CHECKS)
    icon_step = RecipeStep(
        action="click", element="menu_market_icon", key=None,
        verify=ICON_CHECKS, on_fail=("press_esc", "click_free_area"))
    hamburger_step = RecipeStep(
        action="click", element="hamburger_button", key=None,
        verify=MENU_CHECKS, on_fail=("press_esc", "click_free_area"))
    quickslot_step = RecipeStep(
        action="click", element="quickslot_buy_icon", key=None,
        verify=(Check(kind="template", element="quickslot_buy_icon",
                      variant="normal"),),
        on_fail=("press_esc", "click_free_area"))
    close_menu_step = RecipeStep(
        action="key", element=None, key="X",
        verify=MENU_GONE_CHECKS, on_fail=("press_esc",),
        when=MENU_CHECKS)
    recipes = {
        "open_market": Recipe(
            id="open_market", opens="market",
            steps=(key_step, icon_step),
            on_fail=("press_esc", "click_free_area"),
            alternate="open_market_via_hamburger" if with_alternate else None),
        "open_market_via_hamburger": Recipe(
            id="open_market_via_hamburger", opens="market",
            steps=(hamburger_step, icon_step),
            on_fail=("press_esc", "click_free_area")),
        "open_shop_buy": Recipe(
            id="open_shop_buy", opens="shop_buy",
            steps=(close_menu_step, quickslot_step),
            on_fail=("press_esc", "click_free_area")),
    }
    return NavigationModel(path="test", elements=elements,
                           signatures={"menu_open": MENU_CHECKS},
                           recipes=recipes)


class ScriptedEvaluate:
    """Скриптованный ответ evaluate(checks, frame): по индексу вызова
    (устойчиво к числу poll'ов wait_until)."""

    def __init__(self, script):
        if callable(script):
            self._fn = script
        else:
            outcomes = list(script)
            self._fn = lambda checks, i: (
                outcomes[i] if i < len(outcomes) else True)
        self.calls = []

    def __call__(self, checks, frame):
        index = len(self.calls)
        self.calls.append(checks)
        passed = self._fn(checks, index)
        return CheckReport(passed=passed,
                           failures=() if passed else ("fail",))


def _activate():
    return ActivateResult(hwnd=1, already_active=True, attempts=0,
                          cancelled=False)


def _run(recipe_id, nav, evaluate, hud_points=None, hid=None, clock=None,
         cancelled=None, config=None):
    clock = clock or FakeClock()
    hid = hid or ScriptedHid(clock)
    hud_points = hud_points or {}

    def hud_locator(element, variant, frame):
        return hud_points.get(element.id, (200, 200))

    result = run_recipe(
        recipe_id,
        nav_model=nav,
        evaluate=evaluate,
        hud_locator=hud_locator,
        screen=FakeScreen(),
        hid=hid,
        cursor_pos=lambda: (500, 300),
        activate=_activate,
        config=config or make_config(click_retry_count=1,
                                     wait_default_timeout=0.2),
        cancelled=cancelled,
        rng=random.Random(7),
        clock=clock.monotonic,
        sleep=clock.sleep,
    )
    return result, hid


def _presses(hid):
    return [a for a in hid.log if a[1] == "key_press"]


def _clicks(hid):
    return [a for a in hid.log if a[1] == "click"]


class TestRunRecipe:
    def test_open_market_happy_path(self):
        nav = make_nav()
        # skip_if(False) -> key X, verify сходится с первого poll ->
        # click иконки, verify сходится.
        evaluate = ScriptedEvaluate([False, True, True])
        result, hid = _run("open_market", nav, evaluate)
        assert result.success
        assert [a[2] for a in _presses(hid)] == [(0x1B,)]
        assert len(_clicks(hid)) == 1

    def test_skip_if_menu_already_open(self):
        nav = make_nav()
        # skip_if проверяется до шага key: первый вызов evaluate — skip_if
        # (True, меню открыто) -> key_press не выполняется, клик — да.
        evaluate = ScriptedEvaluate([True, True])
        result, hid = _run("open_market", nav, evaluate)
        assert result.success
        assert not _presses(hid)
        assert len(_clicks(hid)) == 1

    def test_when_closes_menu_before_quickslot(self):
        nav = make_nav()
        # when=True -> key X; verify absent=True; verify quickslot=True.
        evaluate = ScriptedEvaluate([True, True, True])
        result, hid = _run("open_shop_buy", nav, evaluate)
        assert result.success
        assert [a[2] for a in _presses(hid)] == [(0x1B,)]
        assert len(_clicks(hid)) == 1

    def test_when_skipped_menu_closed(self):
        nav = make_nav()
        # when=False -> key-шаг пропущен (key_press нет), клик выполнен.
        evaluate = ScriptedEvaluate([False, True])
        result, hid = _run("open_shop_buy", nav, evaluate)
        assert result.success
        assert not _presses(hid)
        assert len(_clicks(hid)) == 1

    def test_menu_element_click_uses_anchor_point(self):
        nav = make_nav()
        # hud_locator даёт якорь (1000, 500); точка иконки =
        # якорь + rect + click_point = (1000+40+24, 500+60+24).
        evaluate = ScriptedEvaluate([True, True])
        result, hid = _run("open_market", nav, evaluate,
                           hud_points={"menu_anchor": (1000, 500)})
        assert result.success
        moves = [a for a in hid.log if a[1] == "move_rel"]
        assert moves
        # Курсор из (500, 300) -> точка (1064, 584): дельта (564, 284).
        assert (moves[0][2]) == (564, 284)

    def test_hud_icon_not_localized_fails_without_click(self):
        nav = make_nav(with_alternate=False)
        # skip_if(False) -> key X OK -> якорь меню не локализован: обе
        # попытки клика потрачены БЕЗ клика (слепых кликов нет).
        evaluate = ScriptedEvaluate([False, True])
        result, hid = _run("open_market", nav, evaluate,
                           hud_points={"menu_anchor": None})
        assert not result.success and not result.unconfirmed
        assert any("не локализован" in f for f in result.failures)
        # Клик по иконке меню шёл бы в точку (1064, 584) — дельта курсора
        # (564, 284) от (500, 300); таких move_rel нет (on_fail'овые
        # click_free_area кликают в центр экрана).
        moves = [a[2] for a in hid.log if a[1] == "move_rel"]
        assert (564, 284) not in moves

    def test_hid_error_on_click_is_unknown(self):
        clock = FakeClock()

        class FailingHid(ScriptedHid):
            def click(self, button=BTN_LEFT, count=1, hold_ms=None):
                raise HidError("транспортный сбой")

        nav = make_nav()
        evaluate = ScriptedEvaluate([True, True])
        result, _ = _run("open_market", nav, evaluate, hid=FailingHid(clock))
        assert not result.success and result.unconfirmed

    def test_failure_falls_back_to_alternate(self):
        nav = make_nav(with_alternate=True)
        # Первые ~25 вызовов evaluate (skip_if, retry key X, retry клика
        # иконки) — False, дальше (alternate) — True.
        evaluate = ScriptedEvaluate(lambda checks, i: i >= 25)
        result, hid = _run("open_market", nav, evaluate)
        assert result.success
        # Иконка меню (основной, 2 попытки) + on_fail шага + on_fail
        # рецепта (оба click_free_area) + гамбургер + иконка (alternate).
        assert len(_clicks(hid)) == 6

    def test_cancelled_before_step(self):
        nav = make_nav()
        evaluate = ScriptedEvaluate([True])
        result, hid = _run("open_market", nav, evaluate,
                           cancelled=lambda: True)
        assert result.cancelled and not result.success
        assert not _presses(hid) and not _clicks(hid)

    def test_unknown_recipe_fails_fast(self):
        nav = make_nav()
        with pytest.raises(RecipeError):
            _run("nope", nav, ScriptedEvaluate([True]))
