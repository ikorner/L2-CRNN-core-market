"""Подтверждённая семантика переходов отгруженных моделей (задача 2.5).

Механизм переходов полностью покрыт парсером (2.1) и валидатором (2.3) —
нового кода он не потребовал; эти тесты фиксируют ДАННЫЕ: переходы,
подтверждённые метой/скриншотами (R4: в мете `click_leads_to: null`,
разметка строго по наблюдаемому). Неподтверждённая семантика не выдумывается
— она в ui_model/KNOWN_GAPS.yaml (текстом) и docs/OWNER_QUESTIONS.md.

Подтверждённые переходы (обоснование — в отчёте 2.5):
- HUD-иконка market_pic -> окно market открыто на tab=home
  (scr_153511: иконка на HUD, окна нет -> scr_153551: tab home активен);
- таб «Снаряжение» <-> «Главная» в market
  (scr_153551 <-> scr_153600, активные варианты кропов);
- кнопка закрытия окна market -> все окна торговли закрыты
  (scr_153551: кнопка есть -> scr_all_close: окна нет).
"""

from pathlib import Path

import pytest

from l2core.model import load_model, load_navigation

REPO_ROOT = Path(__file__).resolve().parent.parent
UI_MODEL = REPO_ROOT / "ui_model"

# Локальные пассивные шаги on_fail (design, решение 6: ESC -> клик в
# свободную область -> ...). Точную цепочку recover определяет рантайм
# (5.8), в модели её не дублируем.
LOCAL_ON_FAIL_STEPS = frozenset({"press_esc", "click_free_area"})

MARKET_HOME = "list__tab_home"
MARKET_EQUIPMENT = "list__filter_all__global_filter_all__tab_equipment"


def _shipped(window):
    return load_model(UI_MODEL / f"{window}.yaml")


def _find_click(model, source, element):
    for transition in model.transitions:
        if (
            transition.source == source
            and transition.action.type == "click"
            and transition.action.element == element
        ):
            return transition
    return None


def _assert_confirmed_shape(transition):
    """Форма подтверждённого перехода: verify непуст, шаги on_fail локальны,
    timeout не задан (берётся из конфига при исполнении)."""
    assert transition.verify, "у подтверждённого перехода есть verify"
    assert transition.on_fail, "у подтверждённого перехода есть on_fail"
    assert set(transition.on_fail) <= LOCAL_ON_FAIL_STEPS, (
        "on_fail модели — только локальные пассивные шаги; "
        f"получено {sorted(transition.on_fail)}"
    )
    assert transition.timeout is None


# --- market: открытие обзора — рецепт навигации, не переход модели ---


def test_opening_is_recipe_not_a_model_transition():
    # menu-quickslot-navigation: открытие обзора (X -> иконка в «Меню»,
    # alternate — гамбургер) исполняет рецепт open_market навигационной
    # модели; межоконные переходы форматом окон не выражены (KNOWN_GAPS
    # п. 3, развязка рецептами). В модели market перехода открытия нет.
    market = _shipped("market")
    assert _find_click(market, "recover_closed", "market_pic") is None
    nav = load_navigation(UI_MODEL / "navigation.yaml")
    recipe = nav.recipes["open_market"]
    assert recipe.opens == "market"
    assert recipe.alternate == "open_market_via_menu"
    checks = {(c.element, c.variant)
              for c in recipe.steps[-1].verify if c.kind == "template"}
    assert checks == {("market_open_anchor", "normal")}


def test_recover_closed_state_present_with_observed_signatures():
    """Состояние «все окна торговли закрыты»: сигнатуры — видимость
    HUD-элементов навигации (гамбургер + иконка лавки продажи на панели
    quick-slot, scr_nav_qs_panel, досъёмка menu-quickslot-navigation).
    Префикс recover — соглашение валидатора 2.3 (ратификация владельцем —
    вопрос Q1, docs/OWNER_QUESTIONS.md)."""
    market = _shipped("market")
    state = market.states["recover_closed"]
    assert state.id.startswith("recover")
    checks = {(c.element, c.variant) for c in state.signatures if c.kind == "template"}
    assert ("hamburger_button", "normal") in checks
    assert ("quickslot_sell_icon", "normal") in checks


# --- market: табы «Главная» <-> «Снаряжение» ---


def test_tab_home_to_equipment():
    market = _shipped("market")
    transition = _find_click(market, MARKET_HOME, "market_snar_passive")
    assert transition is not None, "переход home --Снаряжение--> equipment нет"
    assert transition.expected == MARKET_EQUIPMENT
    _assert_confirmed_shape(transition)
    checks = {(c.element, c.variant) for c in transition.verify if c.kind == "template"}
    assert ("snaryg_active", "active") in checks
    assert ("filter_button_all_active", "active") in checks


def test_tab_equipment_to_home():
    market = _shipped("market")
    transition = _find_click(market, MARKET_EQUIPMENT, "main_not_active")
    assert transition is not None, "переход equipment --Главная--> home нет"
    assert transition.expected == MARKET_HOME
    _assert_confirmed_shape(transition)


# --- market: закрытие окна ---


def test_close_button_leads_to_recover_closed():
    market = _shipped("market")
    transition = _find_click(market, MARKET_HOME, "main_market_close_button")
    assert transition is not None, "переход home --close--> recover_closed нет"
    assert transition.expected == "recover_closed"
    _assert_confirmed_shape(transition)


# --- shop_buy / shop_sell: семантика НЕ подтверждена сырьём ---


@pytest.mark.parametrize("window", ["shop_buy", "shop_sell"])
def test_shop_models_have_no_transitions_yet(window):
    """Внутренние переходы лавок (opening -> open и обратно) мета не
    подтверждает (`click_leads_to: null`) — до ответов владельца
    (docs/OWNER_QUESTIONS.md) секции пусты (R3: не ошибка)."""
    assert _shipped(window).transitions == ()
