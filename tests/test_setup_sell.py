"""Тесты сценария выставления лавки setup_sell (задача 6.2, spec
market-scenarios: «Сценарий выставления лавки продажи» — drag предметов,
цены по slot из геометрии сетки, start_sell()/cancel_sell() с re-verify
перед необратимым подтверждением; сценарии без внутреннего состояния).

Детерминизм (инвариант 7): ScriptedScreen (кадр — функция состояния:
текст цены в поле), ScriptedHid (key_press меняет «содержимое» поля цены
экрана — эмуляция игры), FakeClock, rng с seed, скриптованный find_anchor.

Guard-семантика: re-verify перед необратимым НЕ прошёл — клик по кнопке
подтверждения не выполняется (no blind click на необратимое); UNKNOWN
стражем необратимых (5.7) блокирует повтор start_sell до явного снятия.
"""
import random

import numpy as np
import pytest

from l2core.config import (
    BehaviorConfig,
    CoreConfig,
    OcrConfig,
    ResolutionConfig,
    TimingConfig,
)
from l2core.detector.anchor import Anchor
from l2core.detector.status import STATUS_FOUND, WindowStatus
from l2core.model import (
    Check,
    ElementSpec,
    GrammarSpec,
    GridGeometry,
    OcrRegionSpec,
    Rect,
    UiModel,
    VariantSpec,
    WindowSpec,
)
from l2core.primitives.activate import ActivateResult
from l2core.primitives.click import CheckReport, VerifySpec
from l2core.primitives.outcome import (
    ActionOutcome,
    IrreversibleGuard,
    IrreversibleRepeatError,
)
from l2core.scenarios.setup_sell import (
    SellItem,
    SetupSellError,
    SetupSellResult,
    cancel_sell,
    setup_sell,
    slot_point,
    start_sell,
)
from tests.test_rows import ColorOcr  # noqa: F401  (не используем напрямую)

ANCHOR = Anchor(window_id="shop_sell", x=100, y=50, score=0.99,
                method="template")
WINDOW_SIZE = (200, 160)
FRAME_SIZE = (400, 300)  # (w, h) — вмещает окно со смещением якоря

# Геометрия сетки лавки: 3 колонки × 2 строки, шаг (30, 10), ячейка 28x8.
GRID = GridGeometry(origin=(10, 20), pitch=(30, 10), cell=(28, 8),
                    cols=3, rows=2)
# Центры ячеек (координаты окна от якоря): row-major.
SLOT_CENTERS = [(24, 24), (54, 24), (84, 24), (24, 34), (54, 34), (84, 34)]

# OCR-регион поля цены: rect от якоря; цвет маркера -> (текст, conf).
PRICE_REGION = OcrRegionSpec(
    id="price_field",
    rect=Rect(140, 100, 40, 12),
    grammar=GrammarSpec(name="price", kind="price"),
)
COLOR_PRICE_1 = (250, 120, 40)
COLOR_PRICE_2 = (40, 220, 60)
OCR_TEXTS = {COLOR_PRICE_1: ("1500000", 0.99),
             COLOR_PRICE_2: ("25000", 0.99)}

OPEN_ELEMENT = "open_sale_trade"
SELL_START_ELEMENT = "sell_start_button"
CANCEL_ELEMENT = "cansel_sell_button"
DRAG_MARKER = "drag_marker"
FINAL_MARKER = "final_marker"
CONFIRM_MARKER = "confirm_marker"

GRID_MAX_SLOT = GRID.cols * GRID.rows - 1

# Цифры по HID keycode (usage table, как в enter._ascii_to_hid).
_DIGIT_BY_CODE = {0x1E + (d - 1) % 10: str(d) for d in range(10)}
KEY_ENTER_CODE = 0xB0
KEY_ESC_CODE = 0xB1


def make_shop_model(grid=None):
    elements = {
        SELL_START_ELEMENT: ElementSpec(
            id=SELL_START_ELEMENT, role="click_target",
            rect=Rect(10, 140, 40, 10),
            variants={"normal": VariantSpec(name="normal",
                                            template="test/start.png")}),
        CANCEL_ELEMENT: ElementSpec(
            id=CANCEL_ELEMENT, role="click_target",
            rect=Rect(60, 140, 40, 10),
            variants={"normal": VariantSpec(name="normal",
                                            template="test/cancel.png")}),
        PRICE_FIELD_ELEMENT: ElementSpec(
            id="price_field_element", role="click_target",
            rect=Rect(140, 100, 40, 12),
            variants={"normal": VariantSpec(name="normal",
                                            template="test/field.png")}),
    }
    return UiModel(
        path="test",
        window=WindowSpec(id="shop_sell", title="Торговая лавка (Продажа)",
                          size=WINDOW_SIZE),
        elements=elements,
        states={},
        transitions=(),
        ocr_regions={},
        table_geometry=None,
        grid_geometry=grid,
    )


PRICE_FIELD_ELEMENT = "price_field_element"


def make_hud_model():
    return UiModel(
        path="test-hud",
        window=WindowSpec(id="market", title="Обзор лавок", size=WINDOW_SIZE),
        elements={
            OPEN_ELEMENT: ElementSpec(
                id=OPEN_ELEMENT, role="icon_sample", hud=True,
                search_area=(180, 0, 30, 20),
                variants={"normal": VariantSpec(name="normal",
                                                template="test/open.png")}),
        },
        states={},
        transitions=(),
        ocr_regions={},
        table_geometry=None,
    )


OPEN_VERIFY = VerifySpec(checks=(
    Check(kind="template", element="open_marker", variant="normal"),))
DRAG_VERIFY = VerifySpec(checks=(
    Check(kind="template", element=DRAG_MARKER, variant="normal"),))
FINAL_VERIFY = VerifySpec(checks=(
    Check(kind="template", element=FINAL_MARKER, variant="normal"),))
CONFIRM_VERIFY = VerifySpec(checks=(
    Check(kind="template", element=CONFIRM_MARKER, variant="normal"),),
    expected="selling")
CANCEL_VERIFY = VerifySpec(checks=(
    Check(kind="template", element="cancel_marker", variant="normal"),))


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class ShopScreen:
    """Экран лавки: поле цены показывает «набранный» текст цветом-маркером.

    Состояние — буфер цены; ScriptedHid меняет его key_press'ами (цифры,
    backspace, Enter очищает поле после submit). Кадр строится заново при
    каждом grab — свежесть кадров не эмулируем, детекторы инъецированы.
    """

    def __init__(self):
        self.price_buffer = ""

    def grab(self):
        frame = np.full((FRAME_SIZE[1], FRAME_SIZE[0], 3), 30,
                        dtype=np.uint8)
        color = {v[0]: k for k, v in OCR_TEXTS.items()}.get(self.price_buffer)
        if color is not None:
            # Маркер — в координатах кадра: rect региона от якоря окна.
            r = PRICE_REGION.rect
            x0, y0 = ANCHOR.x + r.x, ANCHOR.y + r.y
            frame[y0:y0 + r.height, x0:x0 + r.width] = color
        return frame


class DictOcr:
    """OcrClient: текст по доминантному нефоновому цвету (как ColorOcr)."""

    def __init__(self, texts_by_color):
        self.texts_by_color = dict(texts_by_color)

    def read(self, region_img):
        img = np.asarray(region_img)
        px = img.reshape(-1, 3).astype(np.int16)
        bg = np.median(px, axis=0)
        mask = np.abs(px - bg).max(axis=1) > 25
        if not mask.any():
            return ("", 0.5)
        colors, counts = np.unique(px[mask], axis=0, return_counts=True)
        color = tuple(int(c) for c in colors[counts.argmax()])
        return self.texts_by_color.get(color, ("???", 0.1))


class ScriptedHid:
    """HID с журналом; key_press эмулирует набор в поле цены экрана."""

    def __init__(self, clock, screen):
        self.clock = clock
        self.screen = screen
        self.log = []

    def _add(self, action, args):
        self.log.append((self.clock.now, action, args))

    def actions(self):
        return [action for _, action, _ in self.log]

    def move_rel(self, dx, dy):
        self._add("move_rel", (dx, dy))

    def click(self, button=1, count=1, hold_ms=None):
        self._add("click", (button, count, hold_ms))

    def key_press(self, code):
        self._add("key_press", (code,))
        if code == KEY_ENTER_CODE:
            self.screen.price_buffer = ""
        elif code == KEY_ESC_CODE:
            self.screen.price_buffer = ""
        elif code == 0xB2:  # KEY_BACKSPACE
            self.screen.price_buffer = self.screen.price_buffer[:-1]
        elif code in _DIGIT_BY_CODE:
            self.screen.price_buffer += _DIGIT_BY_CODE[code]

    def key_down(self, code):
        self._add("key_down", (code,))

    def key_up(self, code):
        self._add("key_up", (code,))

    def mouse_down(self, button=1):
        self._add("mouse_down", (button,))

    def mouse_up(self, button=1):
        self._add("mouse_up", (button,))

    def traj(self, points):
        self._add("traj", tuple(points))


class FakeCursor:
    def __init__(self):
        self.pos = (500, 300)
        self.after_traj = []

    def get(self):
        return self.pos

    def track(self, hid):
        original = hid.move_rel

        def move_rel(dx, dy):
            self.pos = (self.pos[0] + dx, self.pos[1] + dy)
            original(dx, dy)

        hid.move_rel = move_rel
        original_traj = hid.traj
        cursor = self

        def traj(points):
            cursor.pos = (cursor.pos[0] + sum(p[0] for p in points),
                          cursor.pos[1] + sum(p[1] for p in points))
            cursor.after_traj.append(cursor.pos)
            original_traj(points)

        hid.traj = traj


class FakeRunner:
    """Оценка проверок: элементы из fail_elements не проходят, остальное ок."""

    def __init__(self, fail_elements=()):
        self.fail_elements = set(fail_elements)

    def __call__(self, checks, frame, anchor):
        failures = tuple(
            c.element or "?" for c in checks
            if c.element in self.fail_elements
        )
        return CheckReport(passed=not failures, failures=failures)


class ScriptedAnchor:
    """find_anchor по скрипту ответов; исчерпание — последний ответ."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def __call__(self, frame):
        index = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        return self.responses[index]


def make_config():
    return CoreConfig(
        timing=TimingConfig(wait_default_timeout=0.5, verify_interval=0.05,
                            hover_pause=0.01, click_retry_count=0),
        ocr=OcrConfig(confidence_threshold=0.85, reread_count=1),
        behavior=BehaviorConfig(
            numpad_fraction=0.0, enter_fraction=0.0,
            click_jitter_px=0, press_delay_min_ms=1, press_delay_max_ms=1,
            key_delay_min_ms=1, key_delay_max_ms=1,
            drag_midpoints_min=0, drag_midpoints_max=1,
            move_jitter_px=0, move_step_delay_min_ms=1,
            move_step_delay_max_ms=1),
        resolution=ResolutionConfig(width=1920, height=1080),
    )


def found_status(frame):
    return WindowStatus(status=STATUS_FOUND, reason=None, anchor=ANCHOR,
                        content_loaded=None, content_detail="")


def hud_locator(element, variant, frame):
    return (210, 10)


def run_setup(items, *, grid=GRID, model_grid=GRID, price_region=PRICE_REGION,
              source_points=None, fail_elements=(), find_anchor=None,
              cancelled=None, rng_seed=7, ocr=None):
    clock = FakeClock()
    screen = ShopScreen()
    hid = ScriptedHid(clock, screen)
    cursor = FakeCursor()
    cursor.track(hid)
    source_iter = iter(source_points or {})
    result = setup_sell(
        items,
        model=make_shop_model(grid=model_grid),
        hud_model=make_hud_model(),
        grid=grid,
        ocr=ocr or DictOcr(OCR_TEXTS),
        screen=screen,
        hid=hid,
        cursor_pos=cursor.get,
        activate=lambda: ActivateResult(hwnd=1, already_active=True,
                                        attempts=0, cancelled=False),
        window_status=found_status,
        find_anchor=find_anchor or (lambda frame: ANCHOR),
        run_checks=FakeRunner(fail_elements),
        config=make_config(),
        hud_locator=hud_locator,
        open_element=OPEN_ELEMENT,
        open_verify=OPEN_VERIFY,
        source_locator=lambda item: next(source_iter, None),
        price_region=price_region,
        price_field_element=None,
        drag_verify=DRAG_VERIFY,
        final_verify=FINAL_VERIFY,
        cancelled=cancelled,
        rng=random.Random(rng_seed),
        clock=clock.monotonic,
        sleep=clock.sleep,
    )
    return result, hid, cursor, screen


class TestSlotPoint:
    def test_centers_computed_row_major(self):
        for index, expected in enumerate(SLOT_CENTERS):
            assert slot_point(GRID, index) == expected

    def test_out_of_range_rejected(self):
        with pytest.raises(SetupSellError):
            slot_point(GRID, GRID_MAX_SLOT + 1)
        with pytest.raises(SetupSellError):
            slot_point(GRID, -1)

    def test_non_default_grid(self):
        grid = GridGeometry(origin=(0, 0), pitch=(40, 40), cell=(32, 32),
                            cols=4, rows=3)
        assert slot_point(grid, 0) == (16, 16)
        assert slot_point(grid, 5) == (56, 56)  # row 1, col 1
        assert slot_point(grid, 11) == (136, 96)  # row 2, col 3


class TestSetupSell:
    def test_happy_path_places_items_and_prices(self):
        items = [SellItem(slot=1, price=1500000), SellItem(slot=4, price=25000)]
        result, hid, cursor, screen = run_setup(
            items, source_points=[(40, 120), (50, 130)])
        assert isinstance(result, SetupSellResult)
        assert not result.cancelled
        assert not result.needs_recover
        assert [o.item for o in result.items] == items
        actions = hid.actions()
        # Открытие лавки через HUD — первый клик сценария.
        assert actions[0] == "move_rel"
        assert "click" in actions
        # Drag каждого предмета: press -> traj -> release.
        assert actions.count("mouse_down") == 2
        assert actions.count("traj") == 2
        assert actions.count("mouse_up") == 2
        # Цена каждого предмета: 7 цифр + Enter, затем 5 цифр + Enter.
        key_args = [args for _, a, args in hid.log if a == "key_press"]
        typed = [args[0] for args in key_args]
        expected_digits = ([0x1E + (int(d) - 1) % 10 for d in "1500000"]
                           + [KEY_ENTER_CODE]
                           + [0x1E + (int(d) - 1) % 10 for d in "25000"]
                           + [KEY_ENTER_CODE])
        assert typed == expected_digits
        # Курсор в конце каждого drag — ровно в центре целевого слота
        # (координаты кадра: якорь + центр ячейки окна).
        assert cursor.after_traj == [
            (ANCHOR.x + SLOT_CENTERS[item.slot][0],
             ANCHOR.y + SLOT_CENTERS[item.slot][1])
            for item in items
        ]

    def test_drag_targets_are_slot_centers_plus_anchor(self):
        items = [SellItem(slot=0, price=1500000),
                 SellItem(slot=5, price=25000)]
        _, hid, cursor, _ = run_setup(items, source_points=[(40, 120),
                                                            (50, 130)])
        # Два drag'а, каждый завершён release; цели — центры слотов 0 и 5.
        assert hid.actions().count("mouse_down") == 2
        assert cursor.after_traj == [
            (ANCHOR.x + SLOT_CENTERS[0][0], ANCHOR.y + SLOT_CENTERS[0][1]),
            (ANCHOR.x + SLOT_CENTERS[5][0], ANCHOR.y + SLOT_CENTERS[5][1]),
        ]

    def test_grid_taken_from_model_when_parameter_absent(self):
        items = [SellItem(slot=2, price=1500000)]
        result, _, cursor, _ = run_setup(items, grid=None, model_grid=GRID,
                                         source_points=[(40, 120)])
        assert not result.needs_recover
        assert cursor.after_traj == [
            (ANCHOR.x + SLOT_CENTERS[2][0], ANCHOR.y + SLOT_CENTERS[2][1]),
        ]

    def test_missing_grid_is_setup_error(self):
        with pytest.raises(SetupSellError, match="геометрия сетки"):
            run_setup([SellItem(slot=0, price=100)], grid=None,
                      model_grid=None, source_points=[(40, 120)])

    def test_missing_price_region_is_setup_error(self):
        with pytest.raises(SetupSellError, match="price"):
            run_setup([SellItem(slot=0, price=100)],
                      price_region=None, source_points=[(40, 120)])

    def test_duplicate_slot_is_setup_error(self):
        with pytest.raises(SetupSellError, match="повтор"):
            run_setup([SellItem(slot=0, price=100),
                       SellItem(slot=0, price=200)],
                      source_points=[(40, 120), (50, 130)])

    def test_zero_price_is_setup_error(self):
        with pytest.raises(SetupSellError, match="цена"):
            run_setup([SellItem(slot=0, price=0)], source_points=[(40, 120)])

    def test_open_failure_aborts_without_drag(self):
        result, hid, _, _ = run_setup(
            [SellItem(slot=0, price=1500000)],
            source_points=[(40, 120)],
            fail_elements=("open_marker",))
        assert result.needs_recover
        assert result.items == ()
        assert "mouse_down" not in hid.actions()

    def test_anchor_lost_before_item_aborts_without_drag(self):
        anchor = ScriptedAnchor([ANCHOR, None])
        result, hid, _, _ = run_setup(
            [SellItem(slot=0, price=1500000)],
            source_points=[(40, 120)],
            find_anchor=anchor)
        assert result.needs_recover
        assert "mouse_down" not in hid.actions()
        assert any("якорь" in f for f in result.failures)

    def test_source_not_located_aborts_without_drag(self):
        result, hid, _, _ = run_setup(
            [SellItem(slot=0, price=1500000)],
            source_points=iter(()))  # локатор ничего не даёт
        assert result.needs_recover
        assert "mouse_down" not in hid.actions()
        assert any("источник" in f for f in result.failures)

    def test_drag_failure_aborts_before_price(self):
        result, hid, _, _ = run_setup(
            [SellItem(slot=0, price=1500000)],
            source_points=[(40, 120)],
            fail_elements=(DRAG_MARKER,))
        assert result.needs_recover
        # Цена не вводилась: цифровых key_press нет.
        assert not any(a == "key_press" for a in hid.actions())
        assert any("drag" in f.lower() or "drag" in f for f in result.failures)

    def test_price_failure_aborts_with_recover(self):
        # OCR, не знающий ни одного цвета, возвращает «???» — грамматика
        # price не сходится, восстановление раскладки не спасает.
        result, hid, _, _ = run_setup(
            [SellItem(slot=0, price=1500000)],
            source_points=[(40, 120)],
            ocr=DictOcr({}))
        assert result.needs_recover
        assert any("цена" in f for f in result.failures)

    def test_final_verify_failure_is_recover(self):
        result, _, _, _ = run_setup(
            [SellItem(slot=0, price=1500000)],
            source_points=[(40, 120)],
            fail_elements=(FINAL_MARKER,))
        assert result.needs_recover
        assert any(FINAL_MARKER in f for f in result.failures)
        # Предмет выставлен, verify итога — нет: сценарий не подтверждён.
        assert len(result.items) == 1

    def test_cancelled_midway_stops_without_escalation(self):
        calls = {"n": 0}

        def cancelled():
            calls["n"] += 1
            return calls["n"] > 3  # после открытия, в середине первого drag

        result, _, _, _ = run_setup(
            [SellItem(slot=0, price=1500000), SellItem(slot=1, price=200)],
            source_points=[(40, 120), (50, 130)],
            cancelled=cancelled)
        assert result.cancelled
        assert not result.needs_recover


class TestStartSell:
    def _run(self, *, fail_elements=(), anchor=None, guard=None,
             cancelled=None):
        clock = FakeClock()
        screen = ShopScreen()
        hid = ScriptedHid(clock, screen)
        cursor = FakeCursor()
        cursor.track(hid)
        result = start_sell(
            model=make_shop_model(),
            screen=screen,
            hid=hid,
            cursor_pos=cursor.get,
            activate=lambda: ActivateResult(hwnd=1, already_active=True,
                                            attempts=0, cancelled=False),
            window_status=found_status,
            find_anchor=anchor or (lambda frame: ANCHOR),
            run_checks=FakeRunner(fail_elements),
            config=make_config(),
            confirm_checks=CONFIRM_VERIFY,
            verify=CONFIRM_VERIFY,
            guard=guard,
            cancelled=cancelled,
            rng=random.Random(3),
            clock=clock.monotonic,
            sleep=clock.sleep,
        )
        return result, hid

    def test_reverify_blocks_confirmation_click(self):
        # confirm-проверки не проходят -> re-verify не сошёлся, клик
        # по необратимой кнопке НЕ выполняется (no blind click).
        result, hid = self._run(fail_elements=(CONFIRM_MARKER,))
        assert not result.success
        assert not result.reverified
        assert result.outcome is None
        assert "click" not in hid.actions()
        assert result.failures

    def test_success_path_clicks_and_returns_ok(self):
        result, hid = self._run()
        assert result.success
        assert result.reverified
        assert result.outcome is ActionOutcome.OK
        assert "click" in hid.actions()

    def test_unknown_outcome_blocks_repeat_until_confirmed(self):
        # re-verify ок; при verify самого клика якорь потерян -> эффект
        # ушёл, подтвердить нельзя -> UNKNOWN.
        anchor = ScriptedAnchor([ANCHOR, ANCHOR, ANCHOR, None])
        guard = IrreversibleGuard()
        result, _ = self._run(anchor=anchor, guard=guard)
        assert not result.success
        assert result.outcome is ActionOutcome.UNKNOWN
        assert guard.is_blocked("start_sell")
        # Повтор заблокирован стражем (спека: ядро MUST NOT повторять
        # необратимое при UNKNOWN).
        with pytest.raises(IrreversibleRepeatError):
            self._run(guard=guard)
        # Явное снятие стратегией после разборок — повтор снова возможен.
        guard.confirm_after_unknown("start_sell", "лавка не выставлена: "
                                    "окно закрыто, проверено вручную")
        assert not guard.is_blocked("start_sell")
        assert len(guard.resolutions) == 1

    def test_cancelled_before_click(self):
        result, hid = self._run(cancelled=lambda: True)
        assert result.cancelled
        assert not result.success
        assert "click" not in hid.actions()


class TestCancelSell:
    def _run(self, *, element_id=CANCEL_ELEMENT, fail_elements=(),
             cancelled=None):
        clock = FakeClock()
        screen = ShopScreen()
        hid = ScriptedHid(clock, screen)
        cursor = FakeCursor()
        cursor.track(hid)
        result = cancel_sell(
            model=make_shop_model(),
            screen=screen,
            hid=hid,
            cursor_pos=cursor.get,
            activate=lambda: ActivateResult(hwnd=1, already_active=True,
                                            attempts=0, cancelled=False),
            window_status=found_status,
            find_anchor=lambda frame: ANCHOR,
            run_checks=FakeRunner(fail_elements),
            config=make_config(),
            verify=CANCEL_VERIFY,
            element_id=element_id,
            cancelled=cancelled,
            rng=random.Random(3),
            clock=clock.monotonic,
            sleep=clock.sleep,
        )
        return result, hid

    def test_cancel_clicks_model_button(self):
        result, hid = self._run()
        assert result.success
        assert "click" in hid.actions()

    def test_cancel_falls_back_to_esc_without_element(self):
        result, hid = self._run(element_id=None)
        assert result.success
        assert (0xB1,) in [args for _, a, args in hid.log if a == "key_press"]

    def test_cancel_verify_failure_is_recover(self):
        result, _ = self._run(fail_elements=("cancel_marker",))
        assert not result.success
        assert result.needs_recover

    def test_cancelled(self):
        result, hid = self._run(cancelled=lambda: True)
        assert result.cancelled
        assert "click" not in hid.actions()
        assert (0xB1,) not in [args for _, a, args in hid.log
                               if a == "key_press"]
