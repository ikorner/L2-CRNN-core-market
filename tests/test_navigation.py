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
