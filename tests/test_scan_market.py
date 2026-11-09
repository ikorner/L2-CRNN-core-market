"""Тесты сценария scan_market (задача 6.1, spec market-scenarios: обход
табов/фильтров/поиск/скролл, чтение строк, обзор ТОЛЬКО для разведки цен —
открытие чужих лавок отсутствует; сценарии без внутреннего состояния).

Детерминизм (инвариант 7): ScriptedScreen/ScrollScreen (кадр — функция
позиции скролла), фейковый OCR по цвету, FakeClock, rng с seed. Guard:
код сценария не открывает чужие лавки и не кликает по строкам таблицы.
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
    ActionSpec,
    Check,
    ElementSpec,
    GrammarSpec,
    OcrRegionSpec,
    Rect,
    TransitionSpec,
    UiModel,
    VariantSpec,
    WindowSpec,
)
from l2core.primitives.activate import ActivateResult
from l2core.primitives.click import CheckReport, VerifySpec
from l2core.primitives.scroll import (
    DOWN_ARROW_DISABLED_ELEMENT,
    UP_ARROW_DISABLED_ELEMENT,
)
from l2core.scenarios.scan import (
    ScanError,
    ScanPoint,
    ScanResult,
    scan_market,
)
from tests.test_rows import (
    COLOR_ITEM_A,
    COLOR_ITEM_B,
    COLOR_ITEM_C,
    COLOR_PRICE_A,
    COLOR_PRICE_B,
    COLOR_SELLER,
    GEOMETRY,
    ICON_A,
    ICON_B,
    ICON_C,
    ColorOcr,
    make_frame,
    vendor_state,  # noqa: F401  (autouse: снапшот констант вендора)
)

ANCHOR = Anchor(window_id="market", x=0, y=0, score=0.99, method="template")
WINDOW_SIZE = (160, 100)

COLOR_SEARCH = (33, 66, 99)
SEARCH_TEXT = "Mech"

OCR_TEXTS = dict(
    list(ColorOcr().texts_by_color.items()) + [(COLOR_SEARCH, (SEARCH_TEXT, 0.99))]
)

# Клики сценария живут ниже таблицы (строки занимают y 20..50): открытие
# HUD (150, 90), таб (20, 65), фильтр (50, 65), поле поиска (80, 65),
# кнопка «Поиск» (110, 65).
TAB_ELEMENT = "tab_snar"
FILTER_ELEMENT = "filter_weapon"
FIELD_ELEMENT = "search_pole"
BUTTON_ELEMENT = "search_pole_search_button"
OPEN_ELEMENT = "market_pic"

TABLE_ROWS_Y = (GEOMETRY.row_y0,
                GEOMETRY.row_y0 + GEOMETRY.max_rows * GEOMETRY.row_pitch)


def make_model():
    elements = {
        OPEN_ELEMENT: ElementSpec(
            id=OPEN_ELEMENT, role="icon_sample", hud=True,
            search_area=(130, 80, 40, 20),
            variants={"normal": VariantSpec(
                name="normal", template="test/market_pic.png",
                click_point=(20, 10))}),
        TAB_ELEMENT: ElementSpec(
            id=TAB_ELEMENT, role="click_target", rect=Rect(10, 60, 20, 10),
            variants={"normal": VariantSpec(name="normal",
                                            template="test/tab.png")}),
        FILTER_ELEMENT: ElementSpec(
            id=FILTER_ELEMENT, role="click_target", rect=Rect(40, 60, 20, 10),
            variants={"normal": VariantSpec(name="normal",
                                            template="test/filter.png")}),
        FIELD_ELEMENT: ElementSpec(
            id=FIELD_ELEMENT, role="click_target", rect=Rect(70, 60, 20, 10),
            variants={"normal": VariantSpec(name="normal",
                                            template="test/field.png")}),
        BUTTON_ELEMENT: ElementSpec(
            id=BUTTON_ELEMENT, role="click_target", rect=Rect(100, 60, 20, 10),
            variants={"normal": VariantSpec(name="normal",
                                            template="test/button.png")}),
    }
    for eid in (UP_ARROW_DISABLED_ELEMENT, DOWN_ARROW_DISABLED_ELEMENT):
        elements[eid] = ElementSpec(
            id=eid, role="click_target", rect=Rect(150, 10, 8, 8),
            variants={"active": VariantSpec(name="active",
                                            template=f"test/{eid}.png")})
    transitions = (
        TransitionSpec(
            source="recover_closed",
            action=ActionSpec(type="click", element=OPEN_ELEMENT),
            expected="list__tab_home",
            verify=(Check(kind="template", element="tab_home_marker",
                          variant="active"),),
            timeout=None,
            on_fail=("press_esc", "click_free_area"),
        ),
    )
    return UiModel(
        path="test",
        window=WindowSpec(id="market", title="Обзор лавок", size=WINDOW_SIZE),
        elements=elements,
        states={},
        transitions=transitions,
        ocr_regions={},
        table_geometry=GEOMETRY,
    )


POINT_VERIFY = VerifySpec(
    checks=(Check(kind="template", element="list_marker", variant="active"),),
    expected="list_ready",
)
SEARCH_REGION = OcrRegionSpec(
    id="search_field",
    rect=Rect(70, 60, 20, 10),
    grammar=GrammarSpec(name="text", kind="text"),
)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class ScrollScreen:
    """Экран, кадр которого — функция позиции скролла (pos = число wheel'ов)."""

    def __init__(self, frames):
        self._frames = list(frames)
        self.position = 0

    def grab(self):
        return self._frames[min(self.position, len(self._frames) - 1)]


class ScriptedHid:
    """HID с записью действий; wheel продвигает позицию скролла экрана."""

    def __init__(self, clock, screen):
        self.clock = clock
        self.screen = screen
        self.log = []

    def _add(self, action, args):
        self.log.append((self.clock.now, action, args))

    def move_rel(self, dx, dy):
        self._add("move_rel", (dx, dy))

    def scroll(self, count, step, delay_ms):
        self._add("scroll", (count, step, delay_ms))
        if step > 0:
            self.screen.position += count

    def key_press(self, code):
        self._add("key_press", (code,))

    def key_down(self, code):
        self._add("key_down", (code,))

    def key_up(self, code):
        self._add("key_up", (code,))

    def click(self, button=1, count=1, hold_ms=None):
        self._add("click", (button, count, hold_ms))


class FakeCursor:
    def __init__(self):
        self.pos = (500, 300)

    def get(self):
        return self.pos

    def track(self, hid):
        original = hid.move_rel

        def move_rel(dx, dy):
            self.pos = (self.pos[0] + dx, self.pos[1] + dy)
            original(dx, dy)

        hid.move_rel = move_rel


class FakeRunner:
    """Оценка проверок: стрелки — по позиции скролла, fail_always —
    принудительный провал (ветки отказов обхода), остальное — проходит."""

    def __init__(self, screen, disabled_by_pos=None, fail_always=()):
        self.screen = screen
        self.disabled_by_pos = disabled_by_pos or {}
        self.fail_always = set(fail_always)

    def __call__(self, checks, frame, anchor):
        disabled = self.disabled_by_pos.get(self.screen.position, set())
        failures = []
        for c in checks:
            if c.element in self.fail_always:
                ok = False
            elif c.element in (UP_ARROW_DISABLED_ELEMENT,
                               DOWN_ARROW_DISABLED_ELEMENT):
                ok = c.element in disabled
            else:
                ok = True
            if not ok:
                failures.append(c.element or "?")
        return CheckReport(passed=not failures, failures=tuple(failures))


def make_config():
    return CoreConfig(
        timing=TimingConfig(wait_default_timeout=0.5, verify_interval=0.05,
                            hover_pause=0.01, click_retry_count=0),
        ocr=OcrConfig(confidence_threshold=0.85, reread_count=1),
        behavior=BehaviorConfig(
            numpad_fraction=0.0, enter_fraction=0.0,
            click_jitter_px=0, press_delay_min_ms=1, press_delay_max_ms=1,
            key_delay_min_ms=1, key_delay_max_ms=1),
        resolution=ResolutionConfig(width=1920, height=1080),
    )


def found_status(frame):
    return WindowStatus(status=STATUS_FOUND, reason=None, anchor=ANCHOR,
                        content_loaded=None, content_detail="")


def hud_locator(element, variant, frame):
    return (150, 90)


def frame_pos0():
    return make_frame(rows={
        0: {"texts": {"item": COLOR_ITEM_A, "price": COLOR_PRICE_A,
                      "seller": COLOR_SELLER}, "icon": ICON_A},
        1: {"texts": {"item": COLOR_ITEM_B, "price": COLOR_PRICE_B,
                      "seller": COLOR_SELLER}, "icon": ICON_B},
    })


def frame_pos1():
    return make_frame(rows={
        0: {"texts": {"item": COLOR_ITEM_C, "price": COLOR_PRICE_A,
                      "seller": COLOR_SELLER}, "icon": ICON_C},
    })


def frame_with_search():
    frame = frame_pos0()
    frame[60:70, 70:90] = COLOR_SEARCH
    return frame


def run_scan(points, *, frames, disabled_by_pos=None, search_region=None,
             field_element=FIELD_ELEMENT, button_element=BUTTON_ELEMENT,
             cancelled=None, rng_seed=7, ocr=None, fail_always=(),
             max_scrolls=200, find_anchor=None):
    clock = FakeClock()
    screen = ScrollScreen(frames)
    hid = ScriptedHid(clock, screen)
    cursor = FakeCursor()
    cursor.track(hid)
    checker = FakeRunner(screen, disabled_by_pos, fail_always)
    result = scan_market(
        points,
        model=make_model(),
        ocr=ocr or ColorOcr(OCR_TEXTS),
        screen=screen,
        hid=hid,
        cursor_pos=cursor.get,
        activate=lambda: ActivateResult(hwnd=1, already_active=True,
                                        attempts=0, cancelled=False),
        window_status=found_status,
        find_anchor=find_anchor or (lambda frame: ANCHOR),
        run_checks=checker,
        config=make_config(),
        hud_locator=hud_locator,
        search_region=search_region,
        search_field_element=field_element,
        search_button_element=button_element,
        scroll_chunk=1,
        max_scrolls=max_scrolls,
        cancelled=cancelled,
        rng=random.Random(rng_seed),
        clock=clock.monotonic,
        sleep=clock.sleep,
    )
    return result, hid


class TestScanMarket:
    def test_scrolls_to_end_and_aggregates_rows(self):
        result, hid = run_scan(
            [ScanPoint(verify=POINT_VERIFY, tab=TAB_ELEMENT,
                       filters=(FILTER_ELEMENT,))],
            frames=[frame_pos0(), frame_pos1()],
            disabled_by_pos={1: {DOWN_ARROW_DISABLED_ELEMENT}},
        )
        assert isinstance(result, ScanResult)
        assert not result.cancelled
        assert result.failures == ()
        (point,) = result.points
        # Агрегат по кадрам: A, B (кадр 1) + C (кадр 2), дублей нет.
        assert [(r.cells["item"].value, r.screen_y) for r in point.rows] == [
            ("Меч", 20), ("Щит", 30), ("Зелье", 20),
        ]
        # Два снимка (по чтению на кадре) с разными snapshot_id.
        assert len(point.snapshots) == 2
        assert point.snapshots[0].snapshot_id != point.snapshots[1].snapshot_id

    def test_end_marker_stops_scroll_early(self):
        # Конец по признаку модели (disabled-стрелка на позиции 1): wheel
        # уходит ровно один раз — скролл не крутится «вхолостую» дальше.
        result, hid = run_scan(
            [ScanPoint(verify=POINT_VERIFY)],
            frames=[frame_pos0(), frame_pos1()],
            disabled_by_pos={1: {DOWN_ARROW_DISABLED_ELEMENT}},
        )
        wheels = [a for _, a, _ in hid.log if a == "scroll"]
        assert wheels == ["scroll"]
        assert len(result.points[0].rows) == 3  # A, B + C после скролла
    def test_no_clicks_inside_table_rows(self):
        _, hid = run_scan(
            [ScanPoint(verify=POINT_VERIFY, tab=TAB_ELEMENT,
                       filters=(FILTER_ELEMENT,))],
            frames=[frame_pos0(), frame_pos1()],
            disabled_by_pos={1: {DOWN_ARROW_DISABLED_ELEMENT}},
        )
        clicks = [entry for entry in hid.log if entry[1] == "click"]
        assert clicks, "клики открытия/таба/фильтра должны быть"
        # Координаты кликов: позиция курсора на момент click — после move_rel.
        cursor = FakeCursor()
        for _, action, args in hid.log:
            if action == "move_rel":
                cursor.pos = (cursor.pos[0] + args[0], cursor.pos[1] + args[1])
            elif action == "click":
                x, y = cursor.pos
                in_table_y = TABLE_ROWS_Y[0] <= y <= TABLE_ROWS_Y[1]
                in_table_x = GEOMETRY.col_seps[0] <= x <= GEOMETRY.col_seps[-1]
                assert not (in_table_y and in_table_x), \
                    f"клик по области строк таблицы: {(x, y)}"

    def test_search_requires_region_from_caller(self):
        with pytest.raises(ScanError, match="search_region"):
            run_scan(
                [ScanPoint(verify=POINT_VERIFY, search=SEARCH_TEXT)],
                frames=[frame_pos0()],
            )

    def test_search_point_completes(self):
        result, hid = run_scan(
            [ScanPoint(verify=POINT_VERIFY, search=SEARCH_TEXT)],
            frames=[frame_with_search()],
            disabled_by_pos={0: {DOWN_ARROW_DISABLED_ELEMENT}},
            search_region=SEARCH_REGION,
        )
        assert not result.cancelled
        assert result.failures == ()
        key_presses = [a for _, a, _ in hid.log if a == "key_press"]
        assert key_presses, "ввод текста поиска клавиатурой ожидался"
        # Таблица прочитана (замкнутый список: стрелка disabled сразу).
        assert len(result.points[0].rows) == 2

    def test_cancelled_before_any_action(self):
        result, hid = run_scan(
            [ScanPoint(verify=POINT_VERIFY)],
            frames=[frame_pos0()],
            cancelled=lambda: True,
        )
        assert result.cancelled
        assert hid.log == []

    def test_open_transition_missing_is_model_error(self):
        model = make_model()
        bare = UiModel(
            path=model.path, window=model.window, elements=model.elements,
            states=model.states, transitions=(), ocr_regions=model.ocr_regions,
            table_geometry=model.table_geometry,
        )
        clock = FakeClock()
        screen = ScrollScreen([frame_pos0()])
        hid = ScriptedHid(clock, screen)
        with pytest.raises(ScanError, match="market_pic"):
            scan_market(
                [ScanPoint(verify=POINT_VERIFY)],
                model=bare,
                ocr=ColorOcr(OCR_TEXTS),
                screen=screen,
                hid=hid,
                cursor_pos=FakeCursor().get,
                activate=lambda: ActivateResult(hwnd=1, already_active=True,
                                                attempts=0, cancelled=False),
                window_status=found_status,
                find_anchor=lambda frame: ANCHOR,
                run_checks=FakeRunner(screen),
                config=make_config(),
                hud_locator=hud_locator,
                clock=clock.monotonic,
                sleep=clock.sleep,
            )


class TestScanFailures:
    """Трёхзначный результат в композиции сценария (task 9.3): отказ
    примитива фиксируется в ScanResult.failures и обрывает обход — recover
    исполняет вызывающий слой, слепых действий/чтения нет (инвариант 9)."""

    def test_opening_not_confirmed_aborts_scan(self):
        # Verify открытия (tab_home_marker) не сходится: обход не начался,
        # точек нет, wheel не уходил.
        result, hid = run_scan(
            [ScanPoint(verify=POINT_VERIFY)],
            frames=[frame_pos0()],
            fail_always={"tab_home_marker"},
        )
        assert result.points == ()
        assert result.needs_recover
        assert not result.cancelled
        assert "открытие обзора" in result.failures[0]
        assert "tab_home_marker" in result.failures[0]
        assert "scroll" not in [a for _, a, _ in hid.log]

    def test_tab_not_confirmed_aborts_point(self):
        # Клик по табу прошёл, verify точки (list_marker) не сошёлся:
        # точка зафиксирована пустой, обход оборван, строк не прочитано.
        result, hid = run_scan(
            [ScanPoint(verify=POINT_VERIFY, tab=TAB_ELEMENT)],
            frames=[frame_pos0()],
            fail_always={"list_marker"},
        )
        assert result.needs_recover
        assert len(result.points) == 1
        assert result.points[0].rows == ()
        assert result.points[0].snapshots == ()
        assert f"элемент {TAB_ELEMENT!r} не подтверждён" in result.failures[0]
        assert "scroll" not in [a for _, a, _ in hid.log]

    def test_anchor_lost_on_read_frame_aborts_point(self):
        # Якорь есть на кадрах открытия (3 вызова find_anchor у успешного
        # клика: hover-кадр, re-verify после hover, verify после клика —
        # l2core/primitives/click.py), теряется на первом же кадре чтения
        # строк — чтение обрывается честной фиксацией, слепого чтения нет.
        seen = {"n": 0}

        def find_anchor(frame):
            seen["n"] += 1
            return ANCHOR if seen["n"] <= 3 else None

        result, _ = run_scan(
            [ScanPoint(verify=POINT_VERIFY)],
            frames=[frame_pos0()],
            find_anchor=find_anchor,
        )
        assert result.needs_recover
        assert "якорь не найден на кадре чтения строк" in result.failures[0]

    def test_max_scrolls_guard_aborts_when_end_marker_never_comes(self):
        # Скролл подтверждается (кадры различаются), но признак конца из
        # модели (disabled-стрелка) не получен: предохранитель max_scrolls
        # обрывает обход с честной фиксацией.
        frame_pos2 = make_frame(rows={
            0: {"texts": {"item": COLOR_ITEM_B, "price": COLOR_PRICE_A,
                          "seller": COLOR_SELLER}, "icon": ICON_B},
            1: {"texts": {"item": COLOR_ITEM_A, "price": COLOR_PRICE_B,
                          "seller": COLOR_SELLER}, "icon": ICON_A},
        })
        result, hid = run_scan(
            [ScanPoint(verify=POINT_VERIFY)],
            frames=[frame_pos0(), frame_pos1(), frame_pos2],
            disabled_by_pos={},
            max_scrolls=2,
        )
        wheels = [a for _, a, _ in hid.log if a == "scroll"]
        assert len(wheels) == 2  # дальше предохранителя не крутим
        assert result.needs_recover
        assert "max_scrolls=2" in result.failures[0]

    def test_scroll_not_confirmed_aborts_point(self):
        # Wheel ушёл, но нижняя строка не менялась и край не достигнут
        # (замороженный кадр, стрелка активна): скролл needs_recover —
        # точка зафиксирована прочитанным, обход оборван.
        result, hid = run_scan(
            [ScanPoint(verify=POINT_VERIFY)],
            frames=[frame_pos0()],
            disabled_by_pos={},
        )
        assert result.needs_recover
        assert len(result.points) == 1
        assert len(result.points[0].snapshots) == 1  # чтение до скролла
        assert "скролл не подтверждён" in result.failures[0]


class TestNoLavkaOpeningGuard:
    """Guard: сценарий scan_market не открывает чужие лавки и не кликает
    напрямую через HID (spec market-scenarios «Чужая лавка не открывается»)."""

    def _source(self):
        import pathlib

        return pathlib.Path(__file__).parents[1] / "l2core" / "scenarios" / "scan.py"

    def test_no_lavka_opening_elements_in_source(self):
        src = self._source().read_text(encoding="utf-8")
        assert "open_buy_trade" not in src
        assert "open_sale_trade" not in src

    def test_no_direct_hid_click_in_source(self):
        import ast

        tree = ast.parse(self._source().read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                target = node.func.value
                if (node.func.attr == "click"
                        and isinstance(target, ast.Name)
                        and target.id == "hid"):
                    pytest.fail(
                        f"прямой вызов hid.click в сценарии (строка "
                        f"{node.lineno}): клики — только через click_element")
