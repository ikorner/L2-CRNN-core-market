"""Синхронный блокирующий API для стратегий (задача 7.1, spec
core-api-events: навигация / управление обзором / чтение / лавка продажи /
сервисные вызовы; стратегия MUST вызывать API только из своего потока).

L2CoreAPI — ТОНКИЙ ФАСАД: каждый метод — композиция существующих сценариев
6.1–6.4 и примитивов 5.x поверх провайдеров сборки ApiWiring
(l2core/api/wiring.py). Механика кликов/ввода/скролла/drag/recover здесь НЕ
дублируется — тесты проверяют делегирование на скриптованном «мире»:
ScriptedHid фиксирует HID-трафик и мутирует мир, run_checks скриптован по
элементам, FakeOcr отдаёт готовые тексты.

Контракт, проверяемый здесь:
- каждый метод API: делегирование, типизированный результат, отмена;
- навигация: успех -> OK; verify не сошёлся после retry -> FAILED +
  on_fail; якорь потерян после клика -> UNKNOWN (unconfirmed);
- thread-owner: вызов из чужого потока -> RuntimeError;
- read_shop_state — разовое чтение показателей (паттерн монитора 6.3 без
  baseline): деградация — флагами, не исключением;
- get_state/health — агрегаты поверх провайдеров wiring.

Детерминизм (инвариант 7): FakeClock (sleep продвигает время),
rng = random.Random(42), скриптованные кадры/HID/OCR — живого экрана,
окон и железа нет.
"""
import random
import threading

import numpy as np
import pytest

from l2core.adapters.win32 import GameWindow
from l2core.api.core import (
    ApiError,
    CoreState,
    HealthResult,
    L2CoreAPI,
    NavigationResult,
    ShopStateResult,
)
from l2core.api.wiring import ApiWiring
from l2core.config import (
    BehaviorConfig,
    CoreConfig,
    OcrConfig,
    ResolutionConfig,
    TimingConfig,
)
from l2core.detector.anchor import Anchor
from l2core.detector.states import StateReport
from l2core.detector.status import STATUS_FOUND, STATUS_NOT_FOUND, WindowStatus
from l2core.model import (
    ActionSpec,
    Check,
    ColumnZone,
    ElementSpec,
    GrammarSpec,
    GridGeometry,
    OcrRegionSpec,
    Rect,
    TableGeometry,
    TransitionSpec,
    UiModel,
    VariantSpec,
    WindowSpec,
)
from l2core.primitives.activate import ActivateResult
from l2core.primitives.click import CheckReport, VerifySpec
from l2core.primitives.outcome import ActionOutcome, IrreversibleRepeatError
from l2core.primitives.scroll import DIR_DOWN, DIR_UP
from l2core.scenarios.recover import ScenarioRecoverError
from l2core.scenarios.setup_sell import SellItem

HUD_POINT = (1086, 484)
OPEN_BUY_POINT = (760, 930)
OPEN_SELL_POINT = (760, 880)
TAB_EQUIPMENT_POINT = (170, 120)      # якорь (100,100) + rect (50,10) + центр (20,10)
CLOSE_MARKET_POINT = (125, 125)       # якорь (100,100) + rect (10,10) + центр (15,15)
FREE_POINT = (960, 540)               # центр 1920x1080 — свободная область по дефолту
RECOVER_CLOSE_POINT = (700, 200)
SELL_START_POINT = (270, 185)         # якорь лавки (200,150) + rect (20,20) + центр (50,15)
SELL_CANCEL_POINT = (270, 225)   # якорь (200,150) + rect (20,60) + центр (50,15)

MARKET_ANCHOR = Anchor(window_id="market", x=100, y=100, score=0.99,
                       method="template")
SELL_ANCHOR = Anchor(window_id="shop_sell", x=200, y=150, score=0.99,
                     method="template")
BUY_ANCHOR = Anchor(window_id="shop_buy", x=300, y=120, score=0.99,
                    method="template")

TEXT_GRAMMAR = GrammarSpec(name="text", kind="text")
PRICE_GRAMMAR = GrammarSpec(name="price", kind="price")
DIGITS_GRAMMAR = GrammarSpec(name="digits", kind="digits")


# --- Скриптованный мир ---


class FakeClock:
    """Фейковые часы: sleep продвигает время."""

    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeScreen:
    """Экран с фиксированным кадром (пиксель — признак текста для нарезки)."""

    def __init__(self):
        self.frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.frame[0, 0] = 255

    def grab(self):
        return self.frame


class RotatingScreen:
    """Каждый grab — новый кадр (хэш нижней строки меняется между кадрами)."""

    def __init__(self):
        self.n = 0

    def grab(self):
        self.n += 1
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame[0, 0] = self.n % 255 + 1
        return frame


class FakeCell:
    def __init__(self, image, meta):
        self.image = image
        self.meta = meta


class FakeSegmentation:
    """Нарезка-заглушка: одна строка, одна ячейка «name» (= колонка item)."""

    def rows_and_columns(self, gray, origin=(0, 0)):
        return {"rows": [(50, 70)], "cols": {"name": (0, 0)}}

    def extract_cells(self, frame, structure):
        return [FakeCell(
            image=np.ascontiguousarray(frame[0:20, 0:40]),
            meta={"row": 0, "col": "name", "rect": (0, 0, 40, 20)})]


class FakeOcr:
    """OCR-заглушка: очередь ответов (text, confidence), иначе default."""

    def __init__(self, answers=(), default=("", 0.0)):
        self.answers = list(answers)
        self.default = default
        self.calls = 0

    def read(self, region_img):
        self.calls += 1
        return self.answers.pop(0) if self.answers else self.default


class World:
    """Скриптованное состояние: флаги, которые переключает клик по точке."""

    def __init__(self):
        self.market_opened = False
        self.open_click_works = True   # FAILED-тест: клик не открывает обзор
        self.market_closed = False
        self.tab_equipment = False
        self.shop_buy_opened = False
        self.shop_sell_opened = False
        self.sell_started = False
        self.sell_cancelled = False
        self.drag_ok = False
        # --- мир recover (6.4):
        self.current = None      # какое окно торговли открыто
        self.goal = False        # целевое состояние достигнуто
        self.hud_click_opens = False
        self.close_click_closes = False

    def react_click(self, pos):
        if pos == HUD_POINT and self.open_click_works:
            self.market_opened = True
        if pos == OPEN_BUY_POINT:
            self.shop_buy_opened = True
        if pos == OPEN_SELL_POINT:
            self.shop_sell_opened = True
        if pos == TAB_EQUIPMENT_POINT:
            self.tab_equipment = True
        if pos == CLOSE_MARKET_POINT:
            self.market_closed = True
        if pos == SELL_START_POINT:
            self.sell_started = True
        if pos == SELL_CANCEL_POINT:
            self.sell_cancelled = True
        if pos == OPEN_SELL_POINT and self.hud_click_opens:
            self.current = "shop_sell"
            self.goal = True
        if pos == RECOVER_CLOSE_POINT and self.close_click_closes:
            self.current = None


class ScriptedHid:
    """HID с записью действий и позицией курсора; клики мутируют мир."""

    def __init__(self, clock, world):
        self.clock = clock
        self.world = world
        self.log = []
        self.pos = (0, 0)
        self.on_click = None   # хук теста (постановка потери якоря)

    def _append(self, action, args):
        self.log.append((self.clock.now, action, args))

    def move_rel(self, dx, dy):
        self._append("move_rel", (dx, dy))
        self.pos = (self.pos[0] + dx, self.pos[1] + dy)

    def click(self, button=1, count=1, hold_ms=None):
        self._append("click", (button, count, hold_ms, self.pos))
        if self.on_click is not None:
            self.on_click()
        self.world.react_click(self.pos)

    def key_press(self, code):
        self._append("key_press", (code,))

    def key_down(self, code):
        self._append("key_down", (code,))

    def key_up(self, code):
        self._append("key_up", (code,))

    def scroll(self, count, step, delay_ms=0):
        self._append("scroll", (count, step, delay_ms, self.pos))

    def mouse_down(self, button=1):
        self._append("mouse_down", (button, self.pos))

    def mouse_up(self, button=1):
        self._append("mouse_up", (button, self.pos))

    def traj(self, points):
        self._append("traj", (tuple(points),))

    def release_all(self):
        self._append("release_all", ())


class AnchorScript:
    """find_anchor-заглушка: якорь по window_id; lose_next — следующий
    вызов вернёт None (эмуляция «якорь потерян после действия»)."""

    def __init__(self, anchors=None):
        self.anchors = dict(anchors or {})
        self.calls = 0
        self.lose_next = False
        self.lose_all = False

    def __call__(self, frame, window_id):
        self.calls += 1
        if self.lose_all:
            return None
        if self.lose_next:
            self.lose_next = False
            return None
        return self.anchors.get(window_id)


class Cause:
    """Подделка результата примитива/сценария с needs_recover."""

    def __init__(self, success=False, cancelled=False, unconfirmed=False):
        self.success = success
        self.cancelled = cancelled
        self.unconfirmed = unconfirmed

    @property
    def needs_recover(self):
        return not self.success and not self.cancelled


def make_config():
    return CoreConfig(
        timing=TimingConfig(wait_default_timeout=1.0, verify_interval=0.05,
                            hover_pause=0.01, click_retry_count=1),
        ocr=OcrConfig(confidence_threshold=0.85, reread_count=1),
        behavior=BehaviorConfig(
            numpad_fraction=0.0, enter_fraction=0.0, click_jitter_px=0,
            move_jitter_px=0, press_delay_min_ms=50, press_delay_max_ms=50,
            key_delay_min_ms=10, key_delay_max_ms=10,
            move_step_delay_min_ms=5, move_step_delay_max_ms=5,
            drag_midpoints_min=1, drag_midpoints_max=1),
        resolution=ResolutionConfig(width=1920, height=1080),
    )


def make_market_model():
    def elem(eid, rect, click_point=None):
        variants = {"normal": VariantSpec(name="normal", template=f"t/{eid}.png",
                                          click_point=click_point)}
        return ElementSpec(id=eid, role="click_target", rect=rect,
                           variants=variants)

    elements = {
        "market_pic": ElementSpec(
            id="market_pic", role="icon_sample", hud=True,
            search_area=Rect(1000, 420, 200, 160),
            variants={"normal": VariantSpec(
                name="normal", template="t/market_pic.png",
                click_point=(86, 64))}),
        "open_buy_trade": ElementSpec(
            id="open_buy_trade", role="icon_sample", hud=True,
            search_area=Rect(751, 920, 34, 24),
            variants={"normal": VariantSpec(name="normal",
                                            template="t/open_buy.png")}),
        "open_sale_trade": ElementSpec(
            id="open_sale_trade", role="icon_sample", hud=True,
            search_area=Rect(751, 871, 34, 24),
            variants={"normal": VariantSpec(name="normal",
                                            template="t/open_sale.png")}),
        "main_market_close_button": elem("main_market_close_button",
                                         Rect(10, 10, 30, 30)),
        "tab_equipment": elem("tab_equipment", Rect(50, 10, 40, 20)),
        "filter_all": elem("filter_all", Rect(200, 10, 40, 20)),
    }
    transitions = (
        TransitionSpec(
            source="recover_closed",
            action=ActionSpec(type="click", element="market_pic"),
            expected="list__tab_home",
            verify=(Check(kind="template", element="market_main_button_active",
                          variant="active"),),
            on_fail=("press_esc", "click_free_area"), timeout=None),
        TransitionSpec(
            source="list__tab_home",
            action=ActionSpec(type="click", element="tab_equipment"),
            expected="list__tab_equipment",
            verify=(Check(kind="template", element="snaryg_active",
                          variant="active"),),
            on_fail=(), timeout=None),
        TransitionSpec(
            source="list__tab_home",
            action=ActionSpec(type="click", element="main_market_close_button"),
            expected="recover_closed",
            verify=(Check(kind="template", element="market_pic",
                          variant="normal"),),
            on_fail=(), timeout=None),
    )
    geometry = TableGeometry(
        col_seps=(10, 100), row_y0=50, row_pitch=20, max_rows=3,
        columns={
            "item": ColumnZone(id="item", zone=Rect(5, 2, 60, 16),
                               grammar=TEXT_GRAMMAR),
            "price": ColumnZone(id="price", zone=Rect(70, 2, 25, 16),
                                grammar=PRICE_GRAMMAR),
        })
    return UiModel(
        path="market", window=WindowSpec(id="market", title="m",
                                         size=(1000, 700)),
        elements=elements, states={}, transitions=transitions,
        ocr_regions={}, table_geometry=geometry)


def make_shop_sell_model():
    def elem(eid, rect):
        return ElementSpec(
            id=eid, role="click_target", rect=rect,
            variants={"normal": VariantSpec(name="normal",
                                            template=f"t/{eid}.png")})

    return UiModel(
        path="shop_sell", window=WindowSpec(id="shop_sell", title="s",
                                            size=(400, 300)),
        elements={
            "sell_start_button": elem("sell_start_button", Rect(20, 20, 100, 30)),
            "cansel_sell_button": elem("cansel_sell_button", Rect(20, 60, 100, 30)),
        },
        states={}, transitions=(), ocr_regions={}, table_geometry=None,
        grid_geometry=GridGeometry(origin=(10, 10), pitch=(40, 40),
                                   cell=(32, 32), cols=4, rows=3))


def make_shop_buy_model():
    return UiModel(
        path="shop_buy", window=WindowSpec(id="shop_buy", title="b",
                                           size=(400, 300)),
        elements={}, states={}, transitions=(), ocr_regions={},
        table_geometry=None)


def make_run_checks(world, calls):
    """run_checks-заглушка: провал по скриптованным признакам мира."""

    def run_checks(checks, frame, anchor):
        calls.append(checks)
        for check in checks:
            if check.kind != "template":
                continue
            if (check.element == "market_main_button_active"
                    and not world.market_opened):
                return CheckReport(False, ("active-таб «Главная» не найден",))
            if check.element == "snaryg_active" and not world.tab_equipment:
                return CheckReport(False, ("active-таб «Снаряжение» не найден",))
            if check.element == "market_pic" and not world.market_closed:
                return CheckReport(False, ("иконка HUD не найдена",))
            if check.element == "drag_marker" and not world.drag_ok:
                return CheckReport(False, ("предмет в слоте не подтверждён",))
        return CheckReport(True, ())

    return run_checks


def make_hud_locator(world, calls):
    points = {"market_pic": HUD_POINT, "open_buy_trade": OPEN_BUY_POINT,
              "open_sale_trade": OPEN_SELL_POINT}

    def hud_locator(element, variant, frame):
        calls.append(element.id)
        return points.get(element.id)

    return hud_locator


# --- Навигация через рецепты (menu-quickslot-navigation, Task 9) ---

MENU_ANCHOR_POINT = (820, 420)
MENU_ICON_POINT = (884, 504)   # якорь (820,420) + rect (40,60) + click (24,24)
HAMBURGER_POINT = (1850, 1020)
QS_BUY_POINT = (300, 950)
QS_SELL_POINT = (360, 950)


class NavWorld:
    """Мир навигации: X toggles меню; клики по точкам открывают окна."""

    def __init__(self):
        self.menu_open = False
        self.market_opened = False
        self.shop_buy_opened = False
        self.shop_sell_opened = False
        self.open_click_works = True

    def react_click(self, pos):
        if pos == HAMBURGER_POINT:
            self.menu_open = True
        elif (pos == MENU_ICON_POINT and self.menu_open
                and self.open_click_works):
            self.market_opened = True
        elif pos == QS_BUY_POINT and self.open_click_works:
            self.shop_buy_opened = True
        elif pos == QS_SELL_POINT and self.open_click_works:
            self.shop_sell_opened = True


class NavHid(ScriptedHid):
    """HID навигации: X (0x1B) toggles меню в мире."""

    def key_press(self, code):
        super().key_press(code)
        if code == 0x1B:
            self.world.menu_open = not self.world.menu_open


def _nav_elem_hud(eid, point):
    return ElementSpec(
        id=eid, role="icon_sample", hud=True,
        search_area=Rect(0, 0, 1920, 1080),
        variants={"normal": VariantSpec(name="normal",
                                        template=f"t/{eid}.png",
                                        click_point=point)})


def make_nav_model(*, with_alternate=True):
    """Навигационная модель: market через меню (X/гамбургер), лавки через
    quick-slot; verify финальных шагов — условные маркеры окон."""
    from l2core.model.navigation import NavigationModel, Recipe, RecipeStep

    menu_open_checks = (
        Check(kind="template", element="menu_anchor", variant="normal"),
        Check(kind="template", element="menu_market_icon", variant="normal"))
    menu_gone_checks = (
        Check(kind="template", element="menu_anchor", variant="normal",
              expect="absent"),)
    market_checks = (Check(kind="template", element="window_title_market"),)
    buy_checks = (Check(kind="template", element="window_title_buy"),)
    sell_checks = (Check(kind="template", element="window_title_sell"),)

    elements = {
        "menu_anchor": _nav_elem_hud("menu_anchor", (20, 20)),
        "hamburger_button": _nav_elem_hud("hamburger_button", (10, 10)),
        "quickslot_buy_icon": _nav_elem_hud("quickslot_buy_icon", (20, 20)),
        "quickslot_sell_icon": _nav_elem_hud("quickslot_sell_icon", (20, 20)),
        "menu_market_icon": ElementSpec(
            id="menu_market_icon", role="click_target", hud=False,
            rect=Rect(40, 60, 48, 48),
            variants={"normal": VariantSpec(
                name="normal", template="t/menu_market_icon.png",
                click_point=(24, 24))}),
    }

    def click_step(element, verify):
        return RecipeStep(action="click", element=element, key=None,
                          verify=verify,
                          on_fail=("press_esc", "click_free_area"))

    def key_step(verify, *, when=(), skip_if=()):
        return RecipeStep(action="key", element=None, key="X", verify=verify,
                          on_fail=("press_esc",), when=when, skip_if=skip_if)

    recipes = {
        "open_market": Recipe(
            id="open_market", opens="market",
            steps=(key_step(menu_open_checks, skip_if=menu_open_checks),
                   click_step("menu_market_icon", market_checks)),
            on_fail=("press_esc", "click_free_area"),
            alternate="open_market_via_hamburger" if with_alternate else None),
        "open_market_via_hamburger": Recipe(
            id="open_market_via_hamburger", opens="market",
            steps=(click_step("hamburger_button", menu_open_checks),
                   click_step("menu_market_icon", market_checks)),
            on_fail=("press_esc", "click_free_area")),
        "open_shop_buy": Recipe(
            id="open_shop_buy", opens="shop_buy",
            steps=(key_step(menu_gone_checks, when=menu_open_checks),
                   click_step("quickslot_buy_icon", buy_checks)),
            on_fail=("press_esc", "click_free_area")),
        "open_shop_sell": Recipe(
            id="open_shop_sell", opens="shop_sell",
            steps=(key_step(menu_gone_checks, when=menu_open_checks),
                   click_step("quickslot_sell_icon", sell_checks)),
            on_fail=("press_esc", "click_free_area")),
    }
    return NavigationModel(path="navigation", elements=elements,
                           signatures={"menu_open": menu_open_checks},
                           recipes=recipes)


def make_nav_evaluate(world, calls):
    """evaluate-заглушка навигации: признаки из мира, expect absent —
    отрицание."""

    def found(check):
        element = check.element
        if element in ("menu_anchor", "menu_market_icon"):
            return world.menu_open
        if element == "hamburger_button":
            return True
        if element in ("quickslot_buy_icon", "quickslot_sell_icon"):
            return True
        if element == "window_title_market":
            return world.market_opened
        if element == "window_title_buy":
            return world.shop_buy_opened
        if element == "window_title_sell":
            return world.shop_sell_opened
        return True

    def evaluate(checks, frame):
        calls.append(checks)
        failures = []
        for check in checks:
            ok = found(check)
            if check.expect == "absent":
                ok = not ok
            if not ok:
                failures.append(f"fail:{check.element}")
        return CheckReport(passed=not failures, failures=tuple(failures))

    return evaluate


def make_nav_hud_locator(world, calls):
    """hud_locator навигации: якорь меню — только при открытом меню."""

    points = {"menu_anchor": MENU_ANCHOR_POINT,
              "hamburger_button": HAMBURGER_POINT,
              "quickslot_buy_icon": QS_BUY_POINT,
              "quickslot_sell_icon": QS_SELL_POINT}

    def hud_locator(element, variant, frame):
        calls.append(element.id)
        if element.id == "menu_anchor" and not world.menu_open:
            return None
        return points.get(element.id)

    return hud_locator


def make_wiring(world, hid, *, clock, run_checks_calls=None,
                hud_calls=None, anchor_script=None, screen=None, ocr=None,
                segmentation=None, search_region=None, price_region=None,
                shop_regions=None, detect_state=None, current_window=None,
                game_window=None, ocr_health=None, hid_health=None,
                cancelled=None, run_checks=None, hud_locator=None,
                navigation=None, nav_evaluate=None):
    market = make_market_model()
    anchors = anchor_script or AnchorScript({
        "market": MARKET_ANCHOR, "shop_sell": SELL_ANCHOR,
        "shop_buy": BUY_ANCHOR})
    rc_calls = run_checks_calls if run_checks_calls is not None else []
    hud = hud_locator or make_hud_locator(
        world, hud_calls if hud_calls is not None else [])
    return ApiWiring(
        config=make_config(),
        screen=screen or FakeScreen(),
        hid=hid,
        ocr=ocr or FakeOcr(default=("100", 1.0)),
        market=market,
        shop_buy=make_shop_buy_model(),
        shop_sell=make_shop_sell_model(),
        hud_model=market,
        cursor_pos=lambda: hid.pos,
        activate=lambda: ActivateResult(hwnd=1, already_active=True,
                                        attempts=0, cancelled=False),
        find_anchor=anchors,
        window_status=lambda frame, window_id: WindowStatus(
            status=STATUS_FOUND, reason=None,
            anchor=MARKET_ANCHOR, content_loaded=True, content_detail="ok"),
        run_checks=run_checks or make_run_checks(world, rc_calls),
        detect_state=detect_state,
        current_window=current_window or (lambda frame: world.current),
        close_point=lambda window_id, frame: RECOVER_CLOSE_POINT,
        hud_locator=hud,
        goal_reached=lambda target: (lambda frame: world.goal),
        game_window=game_window or (lambda: GameWindow.found(1, (0, 0, 1920, 1080))),
        ocr_health=ocr_health or (lambda: True),
        hid_health=hid_health or (lambda: True),
        search_region=search_region,
        price_region=price_region,
        shop_regions=shop_regions,
        segmentation=segmentation,
        cancelled=cancelled,
        rng=random.Random(42),
        clock=clock.monotonic,
        sleep=clock.sleep,
        navigation=navigation,
        nav_evaluate=nav_evaluate,
    )


def make_nav_api(nav_world, clock=None, nav_calls=None, hud_calls=None,
                 **wiring_kwargs):
    """API на навигационном мире: X toggles меню, точки — из NavWorld."""
    clock = clock or FakeClock()
    nav_calls = nav_calls if nav_calls is not None else []
    hud_calls = hud_calls if hud_calls is not None else []
    hid = NavHid(clock, nav_world)
    api = make_api(nav_world, hid, clock=clock,
                   navigation=make_nav_model(),
                   nav_evaluate=make_nav_evaluate(nav_world, nav_calls),
                   hud_locator=make_nav_hud_locator(nav_world, hud_calls),
                   **wiring_kwargs)
    return api, hid


def make_api(world=None, hid=None, clock=None, events=None, **wiring_kwargs):
    clock = clock or FakeClock()
    world = world or World()
    hid = hid or ScriptedHid(clock, world)
    return L2CoreAPI(make_wiring(world, hid, clock=clock, **wiring_kwargs),
                     events=events)


def actions(hid):
    return [a for _, a, _ in hid.log]


def click_positions(hid):
    return [args[-1] for _, a, args in hid.log if a == "click"]


# --- Тесты ---


class TestThreadOwner:
    """Стратегия MUST вызывать блокирующий API только из своего потока:
    владелец фиксируется при создании, чужой поток -> RuntimeError."""

    def test_wrong_thread_raises_runtime_error(self):
        api = make_api()
        outcome = {}

        def worker():
            try:
                api.health()
            except RuntimeError as exc:
                outcome["error"] = exc

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

        assert isinstance(outcome.get("error"), RuntimeError)
        assert "поток" in str(outcome["error"])

    def test_owner_thread_param_accepted(self):
        api = make_api()
        explicit = L2CoreAPI(
            make_wiring(World(), ScriptedHid(FakeClock(), World()),
                        clock=FakeClock()),
            owner_thread=threading.current_thread())
        assert explicit.health().ok
        assert api.health().ok


class TestOpenMarket:
    """Открытие обзора рецептом навигации: X (skip_if, если меню открыто)
    -> клик по иконке обзора в меню; alternate — через гамбургер."""

    def test_open_success_outcome_ok(self):
        nav_world, clock = NavWorld(), FakeClock()
        api, hid = make_nav_api(nav_world, clock=clock)

        result = api.open_market()

        assert isinstance(result, NavigationResult)
        assert result.success
        assert result.outcome is ActionOutcome.OK
        assert not result.needs_recover
        assert nav_world.market_opened
        # Один X (меню было закрыто) и один клик по иконке обзора.
        assert [a[2] for a in hid.log if a[1] == "key_press"] == [(0x1B,)]
        assert MENU_ICON_POINT in click_positions(hid)

    def test_skip_if_menu_already_open(self):
        nav_world, clock = NavWorld(), FakeClock()
        nav_world.menu_open = True
        api, hid = make_nav_api(nav_world, clock=clock)

        result = api.open_market()

        assert result.success
        assert not [a for a in hid.log if a[1] == "key_press"]
        assert nav_world.market_opened

    def test_verify_timeout_is_failed(self):
        # Окно не открылось (клик не работает): рецепт и alternate
        # исчерпаны -> FAILED (эффект опровергнут), сигнал recover.
        nav_world, clock = NavWorld(), FakeClock()
        nav_world.open_click_works = False
        api, hid = make_nav_api(nav_world, clock=clock)

        result = api.open_market()

        assert not result.success
        assert result.outcome is ActionOutcome.FAILED
        assert result.needs_recover

    def test_hid_error_is_unknown(self):
        # Транспортный сбой HID при клике: клик мог уйти, мог нет —
        # UNKNOWN (инвариант 4), сигнал recover.
        from l2core.adapters.hid import HidError
        nav_world, clock = NavWorld(), FakeClock()
        api, hid = make_nav_api(nav_world, clock=clock)

        def _boom(button=1, count=1, hold_ms=None):
            raise HidError("транспортный сбой")

        hid.click = _boom

        result = api.open_market()

        assert not result.success
        assert result.outcome is ActionOutcome.UNKNOWN
        assert result.unconfirmed
        assert result.needs_recover

    def test_cancelled_returns_cancelled_without_hid(self):
        nav_world, clock = NavWorld(), FakeClock()
        api, hid = make_nav_api(nav_world, clock=clock,
                                cancelled=lambda: True)

        result = api.open_market()

        assert result.cancelled
        assert result.outcome is None
        assert hid.log == []

    def test_menu_anchor_not_located_is_failed_without_click(self):
        # Меню не открылось (X не сработал): якорь меню не локализован —
        # клик по иконке НЕ выполнялся (слепых кликов нет), штатный FAILED.
        nav_world, clock = NavWorld(), FakeClock()

        class NoToggleHid(NavHid):
            def key_press(self, code):
                # X не трогает меню (лаг/перехват окном).
                from tests.test_click import ScriptedHid
                ScriptedHid.key_press(self, code)

        api = make_api(nav_world, NoToggleHid(clock, nav_world), clock=clock,
                       navigation=make_nav_model(with_alternate=False),
                       nav_evaluate=make_nav_evaluate(nav_world, []),
                       hud_locator=make_nav_hud_locator(nav_world, []))

        result = api.open_market()

        assert not result.success
        assert result.outcome is ActionOutcome.FAILED
        assert not result.unconfirmed
        assert not nav_world.market_opened

    def test_navigation_not_configured_fails_fast(self):
        api = make_api()
        with pytest.raises(ApiError):
            api.open_market()


class TestCloseMarket:
    def test_close_success(self):
        world, clock = World(), FakeClock()
        hid = ScriptedHid(clock, world)
        api = make_api(world, hid, clock=clock)

        result = api.close_market()

        assert result.success
        assert result.outcome is ActionOutcome.OK
        assert world.market_closed
        assert CLOSE_MARKET_POINT in click_positions(hid)


class TestOpenShop:
    """open_shop_buy/open_shop_sell: рецепты quick-slot; verify из рецепта
    (параметр verify — override последнего шага)."""

    def test_open_shop_buy_success(self):
        nav_world, clock = NavWorld(), FakeClock()
        api, hid = make_nav_api(nav_world, clock=clock)

        result = api.open_shop_buy()

        assert result.success
        assert result.outcome is ActionOutcome.OK
        assert nav_world.shop_buy_opened
        assert QS_BUY_POINT in click_positions(hid)
        # Меню было закрыто — шаг «закрыть меню» пропущен (X не нажат).
        assert not [a for a in hid.log if a[1] == "key_press"]

    def test_open_shop_sell_success(self):
        nav_world, clock = NavWorld(), FakeClock()
        api, hid = make_nav_api(nav_world, clock=clock)

        result = api.open_shop_sell()

        assert result.success
        assert nav_world.shop_sell_opened
        assert QS_SELL_POINT in click_positions(hid)

    def test_open_shop_closes_open_menu_first(self):
        # Меню открыто: сначала X (закрыть меню), затем клик по quick-slot.
        nav_world, clock = NavWorld(), FakeClock()
        nav_world.menu_open = True
        api, hid = make_nav_api(nav_world, clock=clock)

        result = api.open_shop_buy()

        assert result.success
        assert [a[2] for a in hid.log if a[1] == "key_press"] == [(0x1B,)]
        assert not nav_world.menu_open
        assert nav_world.shop_buy_opened

    def test_verify_override_replaces_final_step_checks(self):
        # verify параметром заменяет verify последнего шага рецепта.
        nav_world, clock = NavWorld(), FakeClock()
        nav_calls = []
        api, hid = make_nav_api(nav_world, clock=clock, nav_calls=nav_calls)
        override = VerifySpec(checks=(
            Check(kind="template", element="custom_marker"),))

        result = api.open_shop_buy(verify=override)

        assert result.success
        seen = {check.element for checks in nav_calls for check in checks}
        assert "custom_marker" in seen
        assert "window_title_buy" not in seen

    @pytest.mark.parametrize("method", ["open_shop_buy", "open_shop_sell"])
    def test_open_shop_verify_timeout_is_failed(self, method):
        # Окно лавки не открылось: рецепт исчерпан -> FAILED, не UNKNOWN.
        nav_world, clock = NavWorld(), FakeClock()
        nav_world.open_click_works = False
        api, hid = make_nav_api(nav_world, clock=clock)

        result = getattr(api, method)()

        assert not result.success
        assert result.outcome is ActionOutcome.FAILED
        assert not result.unconfirmed
        assert result.needs_recover


class TestSetTabSetFilter:
    def test_set_tab_resolves_verify_from_transition(self):
        world, clock = World(), FakeClock()
        hid = ScriptedHid(clock, world)
        calls = []
        api = make_api(world, hid, clock=clock, run_checks_calls=calls)

        result = api.set_tab("tab_equipment")

        assert result.success
        assert result.outcome is ActionOutcome.OK
        assert world.tab_equipment
        assert TAB_EQUIPMENT_POINT in click_positions(hid)
        assert any(Check(kind="template", element="snaryg_active",
                         variant="active") in c for c in calls)

    def test_unknown_element_without_verify_fail_fast(self):
        api = make_api()
        with pytest.raises(ApiError):
            api.set_tab("no_such_tab")

    def test_unknown_element_with_explicit_verify_delegates(self):
        world, clock = World(), FakeClock()
        api = make_api(world, ScriptedHid(clock, world), clock=clock)

        with pytest.raises(Exception):  # элемент не в модели — ошибка модели
            api.set_tab("no_such_tab", verify=VerifySpec(checks=()))

    def test_set_filter_with_explicit_verify(self):
        world, clock = World(), FakeClock()
        hid = ScriptedHid(clock, world)
        api = make_api(world, hid, clock=clock)

        result = api.set_filter("filter_all", verify=VerifySpec(checks=()))

        assert result.success
        assert result.outcome is ActionOutcome.OK


class TestSearch:
    def test_search_success_delegates_to_enter_text(self):
        world, clock = World(), FakeClock()
        hid = ScriptedHid(clock, world)
        region = OcrRegionSpec(id="search_field", rect=Rect(0, 0, 10, 10),
                               grammar=TEXT_GRAMMAR)
        ocr = FakeOcr(default=("sword", 1.0))
        api = make_api(world, hid, clock=clock, ocr=ocr,
                       search_region=region)

        result = api.search("sword")

        assert result.success
        assert result.region_id == "search_field"
        assert not result.cancelled
        assert "key_press" in actions(hid) or "key_down" in actions(hid)

    def test_search_without_region_fail_fast(self):
        api = make_api()
        with pytest.raises(ApiError):
            api.search("sword")

    def test_search_cancelled(self):
        region = OcrRegionSpec(id="search_field", rect=Rect(0, 0, 10, 10),
                               grammar=TEXT_GRAMMAR)
        world, clock = World(), FakeClock()
        hid = ScriptedHid(clock, world)
        api = make_api(world, hid, clock=clock, search_region=region,
                       cancelled=lambda: True)

        result = api.search("sword")

        assert result.cancelled
        assert not result.success
        assert hid.log == []


class TestScrollList:
    def test_scroll_delegates_to_primitive(self):
        world, clock = World(), FakeClock()
        hid = ScriptedHid(clock, world)
        api = make_api(world, hid, clock=clock,
                       screen=RotatingScreen(),
                       segmentation=FakeSegmentation())

        result = api.scroll_list(DIR_DOWN, 2)

        assert result.success
        assert result.scrolled == 2
        assert actions(hid).count("scroll") == 2

    def test_scroll_up_direction(self):
        world, clock = World(), FakeClock()
        hid = ScriptedHid(clock, world)
        api = make_api(world, hid, clock=clock,
                       screen=RotatingScreen(),
                       segmentation=FakeSegmentation())

        result = api.scroll_list(DIR_UP, 1)

        assert result.success
        scrolls = [args for _, a, args in hid.log if a == "scroll"]
        assert scrolls[0][1] == -1   # вверх — отрицательный шаг

    def test_scroll_cancelled(self):
        world, clock = World(), FakeClock()
        hid = ScriptedHid(clock, world)
        api = make_api(world, hid, clock=clock,
                       screen=RotatingScreen(),
                       segmentation=FakeSegmentation(),
                       cancelled=lambda: True)

        result = api.scroll_list(DIR_DOWN, 1)

        assert result.cancelled
        assert "scroll" not in actions(hid)


class TestReadRows:
    def test_read_rows_returns_rows_and_snapshot(self):
        world, clock = World(), FakeClock()
        ocr = FakeOcr(default=("Меч", 1.0))
        api = make_api(world, ScriptedHid(clock, world), clock=clock,
                       ocr=ocr, segmentation=FakeSegmentation())

        result = api.read_rows()

        assert len(result.rows) == 1
        assert result.rows[0].cells["item"].text == "Меч"
        assert result.snapshot_id

    def test_anchor_missing_fail_fast(self):
        world, clock = World(), FakeClock()
        api = make_api(world, ScriptedHid(clock, world), clock=clock,
                       anchor_script=AnchorScript({}),
                       segmentation=FakeSegmentation())

        with pytest.raises(ApiError):
            api.read_rows()

    def test_cancelled_fail_fast(self):
        world, clock = World(), FakeClock()
        api = make_api(world, ScriptedHid(clock, world), clock=clock,
                       segmentation=FakeSegmentation(),
                       cancelled=lambda: True)

        with pytest.raises(ApiError):
            api.read_rows()


class TestSetupSell:
    def test_setup_sell_success(self):
        world, clock = World(), FakeClock()
        world.drag_ok = True
        world.shop_sell_opened = True   # лавка открыта вызывающим слоем
        hid = ScriptedHid(clock, world)
        price_region = OcrRegionSpec(id="price_field", rect=Rect(0, 0, 10, 10),
                                     grammar=PRICE_GRAMMAR)
        api = make_api(world, hid, clock=clock, price_region=price_region,
                       ocr=FakeOcr(default=("100", 1.0)))

        result = api.setup_sell(
            [SellItem(slot=0, price=100)],
            source_locator=lambda item: (600, 900),
            open_recipe=None,
            drag_verify=VerifySpec(checks=(Check(kind="template",
                                                 element="drag_marker",
                                                 variant="normal"),)),
            final_verify=VerifySpec(checks=()),
        )

        assert not result.needs_recover
        assert not result.cancelled
        assert len(result.items) == 1
        assert world.shop_sell_opened
        assert "mouse_down" in actions(hid)
        assert "mouse_up" in actions(hid)

    def test_setup_sell_needs_recover_on_drag_verify_fail(self):
        world, clock = World(), FakeClock()
        world.drag_ok = False   # предмет в слоте не подтверждён
        world.shop_sell_opened = True   # лавка открыта вызывающим слоем
        hid = ScriptedHid(clock, world)
        price_region = OcrRegionSpec(id="price_field", rect=Rect(0, 0, 10, 10),
                                     grammar=PRICE_GRAMMAR)
        api = make_api(world, hid, clock=clock, price_region=price_region)

        result = api.setup_sell(
            [SellItem(slot=0, price=100)],
            source_locator=lambda item: (600, 900),
            open_recipe=None,
            drag_verify=VerifySpec(checks=(Check(kind="template",
                                                 element="drag_marker",
                                                 variant="normal"),)),
        )

        assert result.failures
        assert result.needs_recover

    def test_setup_sell_without_price_region_fail_fast(self):
        api = make_api()
        with pytest.raises(ApiError):
            api.setup_sell([SellItem(slot=0, price=100)],
                           source_locator=lambda item: (600, 900),
                           drag_verify=VerifySpec(checks=()))


class TestStartSell:
    def test_start_sell_success_outcome_ok(self):
        world, clock = World(), FakeClock()
        hid = ScriptedHid(clock, world)
        calls = []
        api = make_api(world, hid, clock=clock, run_checks_calls=calls)

        result = api.start_sell(confirm_checks=VerifySpec(checks=()),
                                verify=VerifySpec(checks=()))

        assert result.success
        assert result.reverified
        assert result.outcome is ActionOutcome.OK
        assert SELL_START_POINT in click_positions(hid)

    def test_unknown_blocks_repeat_until_confirmed(self):
        # UNKNOWN: клик ушёл, подтвердить нельзя — страж необратимых (5.7)
        # блокирует повтор до явного confirm_after_unknown стратегией.
        world, clock = World(), FakeClock()
        hid = ScriptedHid(clock, world)
        script = AnchorScript({"shop_sell": SELL_ANCHOR})
        hid.on_click = lambda: setattr(script, "lose_next", True)
        api = make_api(world, hid, clock=clock, anchor_script=script)

        result = api.start_sell(confirm_checks=VerifySpec(checks=()),
                                verify=VerifySpec(checks=()))
        assert result.outcome is ActionOutcome.UNKNOWN
        assert not result.success

        with pytest.raises(IrreversibleRepeatError):
            api.start_sell(confirm_checks=VerifySpec(checks=()),
                           verify=VerifySpec(checks=()))

        api.guard.confirm_after_unknown("start_sell", "тест: снятие блока")
        hid.on_click = None   # якорь на месте: повтор подтверждается
        retry = api.start_sell(confirm_checks=VerifySpec(checks=()),
                               verify=VerifySpec(checks=()))
        assert retry.success
        assert retry.outcome is ActionOutcome.OK

    def test_start_sell_cancelled(self):
        world, clock = World(), FakeClock()
        hid = ScriptedHid(clock, world)
        api = make_api(world, hid, clock=clock, cancelled=lambda: True)

        result = api.start_sell(confirm_checks=VerifySpec(checks=()),
                                verify=VerifySpec(checks=()))

        assert result.cancelled
        assert result.outcome is None
        assert hid.log == []


class TestCancelSell:
    def test_cancel_sell_success(self):
        world, clock = World(), FakeClock()
        hid = ScriptedHid(clock, world)
        api = make_api(world, hid, clock=clock)

        result = api.cancel_sell(verify=VerifySpec(checks=()))

        assert result.success
        assert world.sell_cancelled
        assert SELL_CANCEL_POINT in click_positions(hid)


class TestReadShopState:
    """Разовое чтение показателей лавки (паттерн монитора 6.3 без
    baseline): окно закрыто/якорь потерян — деградация флагами."""

    def _regions(self):
        return {
            "sold_count": OcrRegionSpec(id="sold_count",
                                        rect=Rect(0, 0, 10, 10),
                                        grammar=DIGITS_GRAMMAR),
            "revenue": OcrRegionSpec(id="revenue", rect=Rect(0, 20, 10, 10),
                                     grammar=PRICE_GRAMMAR),
        }

    def test_read_shop_state_success(self):
        world, clock = World(), FakeClock()
        ocr = FakeOcr(answers=[("7", 1.0), ("1500", 1.0)])
        api = make_api(world, ScriptedHid(clock, world), clock=clock,
                       ocr=ocr, shop_regions=self._regions())

        result = api.read_shop_state()

        assert isinstance(result, ShopStateResult)
        assert result.sold_count == 7
        assert result.revenue == 1500
        assert result.snapshot_id
        assert not result.degraded
        assert not result.failures

    def test_missing_counter_region_fail_fast(self):
        api = make_api()
        with pytest.raises(ApiError):
            api.read_shop_state()

    def test_anchor_missing_is_degradation_not_exception(self):
        world, clock = World(), FakeClock()
        api = make_api(world, ScriptedHid(clock, world), clock=clock,
                       anchor_script=AnchorScript({}),
                       shop_regions=self._regions())

        result = api.read_shop_state()

        assert result.degraded
        assert result.sold_count is None
        assert result.failures
        # Монитор-паттерн: чтение показателей HID не трогает вообще.
        assert api._w.hid.log == []


class TestGetState:
    def _detect_state(self, window_id="market", state="list__tab_home",
                      params=None):
        def detect_state(frame, anchor):
            return StateReport(
                window_id=anchor.window_id, state=state,
                params=dict(params or {"tab": "home"}), matched=(),
                failed=(), conflicts=(), ocr_degraded=False)
        return detect_state

    def test_get_state_returns_window_state_params(self):
        world, clock = World(), FakeClock()
        api = make_api(world, ScriptedHid(clock, world), clock=clock,
                       detect_state=self._detect_state(),
                       current_window=lambda frame: "market")

        result = api.get_state()

        assert isinstance(result, CoreState)
        assert result.window == "market"
        assert result.state == "list__tab_home"
        assert result.params == {"tab": "home"}
        assert result.status == STATUS_FOUND
        assert result.content_loaded is True

    def test_no_trading_window(self):
        world, clock = World(), FakeClock()
        api = make_api(world, ScriptedHid(clock, world), clock=clock,
                       detect_state=self._detect_state(),
                       current_window=lambda frame: None)

        result = api.get_state()

        assert result.window is None
        assert result.status == STATUS_NOT_FOUND
        assert result.state is None

    def test_get_state_cancelled(self):
        world, clock = World(), FakeClock()
        api = make_api(world, ScriptedHid(clock, world), clock=clock,
                       cancelled=lambda: True)

        with pytest.raises(ApiError):
            api.get_state()


class TestRecover:
    def test_recover_success_at_entry_goal(self):
        world, clock = World(), FakeClock()
        world.goal = True
        hid = ScriptedHid(clock, world)
        api = make_api(world, hid, clock=clock)

        result = api.recover(Cause(), "shop_sell", hud_element="open_sale_trade")

        assert result.success
        assert result.target_window == "shop_sell"
        assert not result.escalated

    def test_recover_runs_chain_with_providers(self):
        # Цель не достигнута на входе: цепочка 5.8 идёт по шагам —
        # ESC, свободный клик, закрытие чужого окна, переоткрытие через HUD.
        world, clock = World(), FakeClock()
        world.current = "market"
        world.hud_click_opens = True
        world.close_click_closes = True
        hid = ScriptedHid(clock, world)
        api = make_api(world, hid, clock=clock)

        result = api.recover(Cause(), "shop_sell", hud_element="open_sale_trade")

        assert result.success
        assert actions(hid)[0] == "release_all"   # cause протянут
        assert click_positions(hid) == [FREE_POINT, RECOVER_CLOSE_POINT,
                                        OPEN_SELL_POINT]

    def test_recover_fail_fast_on_successful_cause(self):
        api = make_api()
        with pytest.raises(ScenarioRecoverError):
            api.recover(Cause(success=True), "shop_sell",
                        hud_element="open_sale_trade")

    def test_recover_cancelled(self):
        world, clock = World(), FakeClock()
        hid = ScriptedHid(clock, world)
        api = make_api(world, hid, clock=clock, cancelled=lambda: True)

        result = api.recover(Cause(), "shop_sell", hud_element="open_sale_trade")

        assert result.cancelled
        assert not result.escalated
        assert hid.log == []


class TestHealth:
    def test_health_all_ok(self):
        api = make_api(current_window=lambda frame: "market")

        result = api.health()

        assert isinstance(result, HealthResult)
        assert result.ok
        assert result.game_window
        assert result.trading_window == "market"
        assert result.ocr and result.hid

    def test_game_window_missing(self):
        api = make_api(
            game_window=lambda: GameWindow.failed("window_not_found"))

        result = api.health()

        assert not result.ok
        assert not result.game_window
        assert "window_not_found" in result.game_detail

    def test_hid_dead(self):
        api = make_api(hid_health=lambda: False)

        result = api.health()

        assert not result.ok
        assert not result.hid

    def test_no_trading_window_is_not_failure(self):
        # Окно торговли закрыто — не нездоровье: игра жива, а обзор
        # стратегия открывает сама.
        api = make_api(current_window=lambda frame: None)

        result = api.health()

        assert result.ok
        assert result.trading_window is None
