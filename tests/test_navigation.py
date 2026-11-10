"""Навигационная модель ui_model/navigation.yaml (menu-quickslot-navigation,
Task 3): парсер элементов/сигнатур/рецептов; Tasks 4–5 — валидатор и
трассировка дополняют этот файл.
"""

import pytest

from l2core.model.navigation import (
    EXPECT_ABSENT,
    load_navigation,
)
from l2core.model.loader import ModelFormatError

NAV_YAML = """\
elements:
  menu_anchor:
    role: icon_sample
    hud: true
    search_area: [700, 300, 520, 400]
    variants:
      normal: {template: navigation/menu_anchor.png}
  menu_market_icon:
    role: click_target
    rect: [40, 60, 48, 48]
    variants:
      normal: {template: navigation/menu_market_icon.png, click_point: [24, 24]}
  hamburger_button:
    role: icon_sample
    hud: true
    search_area: [1800, 980, 100, 80]
    variants:
      normal: {template: navigation/hamburger.png, click_point: [50, 40]}
  quickslot_buy_icon:
    role: icon_sample
    hud: true
    search_area: [100, 900, 1700, 120]
    variants:
      normal: {template: navigation/quickslot_buy.png, click_point: [20, 20]}
signatures:
  menu_open:
    - {kind: template, element: menu_anchor, variant: normal}
    - {kind: template, element: menu_market_icon, variant: normal}
recipes:
  open_market:
    opens: market
    steps:
      - action: key
        key: X
        skip_if: [menu_open]
        verify: [menu_open]
        on_fail: [press_esc]
      - action: click
        element: menu_market_icon
        verify:
          - {kind: template, element: menu_anchor, expect: absent}
        on_fail: [press_esc, click_free_area]
    on_fail: [press_esc, click_free_area]
    alternate: open_market_via_hamburger
  open_market_via_hamburger:
    opens: market
    steps:
      - action: click
        element: hamburger_button
        verify: [menu_open]
        on_fail: [press_esc, click_free_area]
      - action: click
        element: menu_market_icon
        verify: [menu_open]
        on_fail: [press_esc, click_free_area]
    on_fail: [press_esc, click_free_area]
  open_shop_buy:
    opens: shop_buy
    steps:
      - action: key
        key: X
        when: [menu_open]
        verify:
          - {kind: template, element: menu_anchor, expect: absent}
        on_fail: [press_esc]
      - action: click
        element: quickslot_buy_icon
        verify:
          - {kind: template, element: quickslot_buy_icon, variant: normal}
        on_fail: [press_esc, click_free_area]
    on_fail: [press_esc, click_free_area]
"""


def _write_nav(tmp_path, text=NAV_YAML):
    path = tmp_path / "navigation.yaml"
    path.write_text(text, encoding="utf-8")
    return path


class TestLoadNavigation:
    def test_recipes_and_signatures(self, tmp_path):
        nav = load_navigation(_write_nav(tmp_path))
        assert set(nav.elements) == {
            "menu_anchor", "menu_market_icon", "hamburger_button",
            "quickslot_buy_icon",
        }
        assert len(nav.signatures["menu_open"]) == 2
        open_market = nav.recipes["open_market"]
        assert open_market.opens == "market"
        assert open_market.alternate == "open_market_via_hamburger"
        assert [s.action for s in open_market.steps] == ["key", "click"]
        # Строка-ссылка на сигнатуру развёрнута в её проверки.
        assert open_market.steps[0].verify == nav.signatures["menu_open"]
        assert open_market.steps[0].skip_if == nav.signatures["menu_open"]
        # Отрицание template-проверки.
        absent = open_market.steps[1].verify[0]
        assert absent.kind == "template" and absent.expect == EXPECT_ABSENT

    def test_when_expands_signature(self, tmp_path):
        nav = load_navigation(_write_nav(tmp_path))
        step = nav.recipes["open_shop_buy"].steps[0]
        assert step.when == nav.signatures["menu_open"]

    def test_unknown_recipe_key_fails(self, tmp_path):
        bad = NAV_YAML.replace(
            "    opens: market\n", "    opens: market\n    typo: 1\n", 1)
        with pytest.raises(ModelFormatError):
            load_navigation(_write_nav(tmp_path, bad))

    def test_key_step_with_element_fails(self, tmp_path):
        bad = NAV_YAML.replace(
            "        key: X\n        skip_if", "        key: X\n        element: menu_anchor\n        skip_if", 1)
        with pytest.raises(ModelFormatError):
            load_navigation(_write_nav(tmp_path, bad))

    def test_ocr_check_fails(self, tmp_path):
        bad = NAV_YAML.replace(
            "verify:\n          - {kind: template, element: quickslot_buy_icon, variant: normal}",
            "verify:\n          - {kind: ocr, region: foo, expect: bar}",
            1)
        with pytest.raises(ModelFormatError):
            load_navigation(_write_nav(tmp_path, bad))

    def test_unknown_signature_ref_fails(self, tmp_path):
        bad = NAV_YAML.replace("skip_if: [menu_open]", "skip_if: [no_such]", 1)
        with pytest.raises(ModelFormatError) as exc_info:
            load_navigation(_write_nav(tmp_path, bad))
        assert "no_such" in str(exc_info.value)

    def test_template_expect_only_absent(self, tmp_path):
        bad = NAV_YAML.replace(
            "expect: absent}", "expect: present}", 1)
        with pytest.raises(ModelFormatError):
            load_navigation(_write_nav(tmp_path, bad))


# ---------------------------------------------------------------------------
# Валидатор навигационной модели (Task 4).
# ---------------------------------------------------------------------------

import dataclasses

from l2core.model.validator import validate_navigation


def _nav_with_assets(tmp_path):
    """Валидная модель NAV_YAML + файлы шаблонов в tmp assets."""
    import cv2
    import numpy as np

    assets = tmp_path / "assets"
    (assets / "navigation").mkdir(parents=True)
    for name in ("menu_anchor", "menu_market_icon", "hamburger",
                 "quickslot_buy"):
        cv2.imwrite(str(assets / "navigation" / f"{name}.png"),
                    np.zeros((8, 8), dtype=np.uint8))
    nav = load_navigation(_write_nav(tmp_path))
    return nav, assets


def _rules(violations):
    return {v.rule for v in violations}


class TestValidateNavigation:
    def test_valid_model_passes(self, tmp_path):
        nav, assets = _nav_with_assets(tmp_path)
        assert validate_navigation(nav, assets) == []

    def test_signature_with_single_check_fails(self, tmp_path):
        nav, assets = _nav_with_assets(tmp_path)
        nav = dataclasses.replace(
            nav, signatures={"menu_open": nav.signatures["menu_open"][:1]})
        assert "nav_signatures" in _rules(
            validate_navigation(nav, assets))

    def test_step_without_on_fail_fails(self, tmp_path):
        nav, assets = _nav_with_assets(tmp_path)
        recipe = nav.recipes["open_shop_buy"]
        step0 = dataclasses.replace(recipe.steps[0], on_fail=())
        nav = dataclasses.replace(nav, recipes={
            **nav.recipes,
            "open_shop_buy": dataclasses.replace(recipe, steps=(step0,) + recipe.steps[1:]),
        })
        assert "nav_step_verify" in _rules(
            validate_navigation(nav, assets))

    def test_recipe_without_on_fail_and_alternate_fails(self, tmp_path):
        nav, assets = _nav_with_assets(tmp_path)
        recipe = nav.recipes["open_shop_buy"]
        nav = dataclasses.replace(nav, recipes={
            **nav.recipes,
            "open_shop_buy": dataclasses.replace(recipe, on_fail=()),
        })
        rules = _rules(validate_navigation(nav, assets))
        assert "nav_recipe" in rules and "nav_recipe_no_exit" not in rules

    def test_unknown_alternate_fails(self, tmp_path):
        nav, assets = _nav_with_assets(tmp_path)
        recipe = nav.recipes["open_market"]
        nav = dataclasses.replace(nav, recipes={
            **nav.recipes,
            "open_market": dataclasses.replace(recipe, alternate="nope"),
        })
        assert "nav_recipe" in _rules(validate_navigation(nav, assets))

    def test_step_with_unknown_element_fails(self, tmp_path):
        nav, assets = _nav_with_assets(tmp_path)
        recipe = nav.recipes["open_shop_buy"]
        step1 = dataclasses.replace(recipe.steps[1], element="ghost")
        nav = dataclasses.replace(nav, recipes={
            **nav.recipes,
            "open_shop_buy": dataclasses.replace(recipe, steps=recipe.steps[:1] + (step1,)),
        })
        assert "nav_refs" in _rules(validate_navigation(nav, assets))

    def test_unknown_on_fail_step_fails(self, tmp_path):
        nav, assets = _nav_with_assets(tmp_path)
        recipe = nav.recipes["open_shop_buy"]
        step1 = dataclasses.replace(
            recipe.steps[1], on_fail=("press_esc", "click_at_0_0"))
        nav = dataclasses.replace(nav, recipes={
            **nav.recipes,
            "open_shop_buy": dataclasses.replace(recipe, steps=recipe.steps[:1] + (step1,)),
        })
        assert "nav_on_fail_steps" in _rules(
            validate_navigation(nav, assets))

    def test_unknown_key_fails(self, tmp_path):
        nav, assets = _nav_with_assets(tmp_path)
        recipe = nav.recipes["open_shop_buy"]
        step0 = dataclasses.replace(recipe.steps[0], key="Q")
        nav = dataclasses.replace(nav, recipes={
            **nav.recipes,
            "open_shop_buy": dataclasses.replace(recipe, steps=(step0,) + recipe.steps[1:]),
        })
        assert "nav_key_known" in _rules(validate_navigation(nav, assets))

    def test_missing_template_fails(self, tmp_path):
        nav, assets = _nav_with_assets(tmp_path)
        empty_assets = tmp_path / "empty_assets"
        empty_assets.mkdir()
        assert "nav_templates" in _rules(
            validate_navigation(nav, empty_assets))

    def test_menu_element_without_anchor_fails(self, tmp_path):
        # rect-элемент есть, а hud-якоря меню в модели нет.
        nav, assets = _nav_with_assets(tmp_path)
        nav = dataclasses.replace(
            nav,
            elements={
                "stray": nav.elements["menu_market_icon"],
            })
        assert "nav_menu_anchor" in _rules(
            validate_navigation(nav, assets))


# ---------------------------------------------------------------------------
# Трассировка навигационных рецептов (Task 5): окно достижимо из своего
# recover_closed через рецепт open_<window> (соглашение имён + opens);
# неуспех рецепта имеет выход в recover (on_fail/alternate).
# ---------------------------------------------------------------------------

import dataclasses as _dc

from l2core.model import (
    Check,
    StateSpec,
    UiModel,
    WindowSpec,
)
from l2core.model.trace import trace_navigation


def _window_model(tmp_path, window_id: str) -> UiModel:
    """Минимальная модель окна: одно состояние recover_closed."""
    return UiModel(
        path=str(tmp_path / f"{window_id}.yaml"),
        window=WindowSpec(id=window_id, title=window_id, size=(100, 100)),
        elements={},
        states={"recover_closed": StateSpec(
            id="recover_closed", params={},
            signatures=(Check(kind="pixel_probe", points=()),))},
        transitions=(),
        ocr_regions={},
        table_geometry=None)


class TestTraceNavigation:
    def test_all_windows_reachable_via_recipes(self, tmp_path):
        nav, _ = _nav_with_assets(tmp_path)
        nav = _dc.replace(nav, recipes={
            **nav.recipes,
            "open_shop_sell": _dc.replace(
                nav.recipes["open_shop_buy"],
                id="open_shop_sell", opens="shop_sell"),
        })
        models = {
            "market": _window_model(tmp_path, "market"),
            "shop_buy": _window_model(tmp_path, "shop_buy"),
            "shop_sell": _window_model(tmp_path, "shop_sell"),
        }
        assert trace_navigation(models, nav) == []

    def test_missing_recipe_fails(self, tmp_path):
        nav, _ = _nav_with_assets(tmp_path)
        models = {
            "market": _window_model(tmp_path, "market"),
            "shop_buy": _window_model(tmp_path, "shop_buy"),
            "shop_sell": _window_model(tmp_path, "shop_sell"),
        }
        rules = {v.rule for v in trace_navigation(models, nav)}
        assert "nav_window_unreachable" in rules
        subjects = {v.subject for v in trace_navigation(models, nav)
                    if v.rule == "nav_window_unreachable"}
        assert subjects == {"shop_sell"}

    def test_recipe_without_exit_fails(self, tmp_path):
        nav, _ = _nav_with_assets(tmp_path)
        # open_shop_buy без on_fail и без alternate — неуспех не имеет выхода.
        recipe = nav.recipes["open_shop_buy"]
        nav = _dc.replace(nav, recipes={
            **nav.recipes,
            "open_shop_sell": _dc.replace(
                nav.recipes["open_shop_buy"],
                id="open_shop_sell", opens="shop_sell"),
            "open_shop_buy": _dc.replace(
                recipe, on_fail=(), alternate=None),
        })
        models = {
            "market": _window_model(tmp_path, "market"),
            "shop_buy": _window_model(tmp_path, "shop_buy"),
            "shop_sell": _window_model(tmp_path, "shop_sell"),
        }
        violations = trace_navigation(models, nav)
        rules = {v.rule for v in violations}
        assert "nav_recipe_no_exit" in rules
        subjects = {v.subject for v in violations
                    if v.rule == "nav_recipe_no_exit"}
        assert subjects == {"recipes.open_shop_buy"}

    def test_recover_closed_missing_fails(self, tmp_path):
        nav, _ = _nav_with_assets(tmp_path)
        nav = _dc.replace(nav, recipes={
            **nav.recipes,
            "open_shop_sell": _dc.replace(
                nav.recipes["open_shop_buy"],
                id="open_shop_sell", opens="shop_sell"),
        })
        model = _window_model(tmp_path, "shop_sell")
        model = _dc.replace(model, states={
            "some_state": model.states["recover_closed"],
        })
        models = {
            "market": _window_model(tmp_path, "market"),
            "shop_buy": _window_model(tmp_path, "shop_buy"),
            "shop_sell": model,
        }
        rules = {v.rule for v in trace_navigation(models, nav)}
        assert "nav_recover_state_missing" in rules
