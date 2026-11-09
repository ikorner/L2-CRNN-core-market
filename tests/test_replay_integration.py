"""Интеграционные replay-тесты сценариев (задача 9.2, spec replay-logging:
«Интеграционные replay-тесты сценариев» — обход/выставление/мониторинг
прогоняются на записанных кадрах с assert на последовательности состояний и
OCR-результаты; binding R7: живых записей нет — сессии синтетические).

УСТРОЙСТВО. Сессия собирается двухпроходно, без живого экрана и железа:

1. Проход ЗАПИСИ: сценарий исполняется против скриптованного «мира»
   (MarketWorld/ShopWorld — состояние -> кадр) с RecordingHid: каждая
   эффектная HID-команда (click/key_press/key_down/key_up/scroll/
   mouse_down/mouse_up/traj) исполняется в мире (реакция по координате
   клика/keycode) и фиксируется с кадром-после; move_rel (позиционирование
   курсора) действием не считается. События пишутся SessionWriter
   (начальный кадр -> action+кадр-после -> ...), детерминировано (tmp_path).
2. Проход ВОСПРОИЗВЕДЕНИЯ: тот же сценарий (тот же closure, те же seed rng
   и часы) исполняется против ReplayScreen(сессия) + ReplayHid: каждое
   эффектное действие уходит в ReplayScreen.notify_action — строгое
   сопоставление с записью (ActionMismatchError при расхождении — это и есть
   assert последовательности состояний/действий: сценарий обязан генерировать
   ровно те действия, что записаны). Кадры-после декодируются из PNG
   (потерь нет — мир воспроизводится побайтово).

Проверки сценариев сходятся с юнит-мирьями test_scan_market/test_setup_sell
(те же модели, OCR по цвету, геометрия, конфиг) — отличие только в слое
экрана/HID: здесь он реальный l2core (ReplayScreen) поверх записанной сессии.

Monitor (observer, HID не трогает вообще): курсор ReplayScreen двигают
только действия (инвариант формата 3.1), frame-события grab не продвигает,
поэтому изменения мира между poll'ами подаёт тестовый SessionFrameFeed
(каждый grab — следующий кадр сессии, последний повторяется) — честная
модель «мир изменился без действий ядра» для наблюдателя.

«Без обращений к mss, HID и игре» фиксируется по построению: единственный
Screen — ReplayScreen/SessionFrameFeed поверх файлов сессии в tmp_path,
единственный HID — RecordingHid/ReplayHid (скрипт); assert isinstance
закрепляет границу. Детерминизм (инвариант 7): синтетические кадры,
скриптованные OCR/HID/часы, rng с фиксированными seed, меток времени нет.
"""
from __future__ import annotations

import json
import random

import numpy as np
import pytest

from l2core.adapters.screen import (
    ActionMismatchError,
    ReplayScreen,
    Screen,
    SessionExhaustedError,
)
from l2core.adapters.session import SessionWriter, load_session
from l2core.model import GrammarSpec, OcrRegionSpec, Rect
from l2core.primitives.activate import ActivateResult
from l2core.primitives.click import CheckReport
from l2core.primitives.outcome import ActionOutcome, IrreversibleGuard
from l2core.scenarios.monitor import ShopMonitor
from l2core.scenarios.scan import ScanPoint, scan_market
from l2core.scenarios.setup_sell import SellItem, setup_sell, start_sell
from tests.test_rows import vendor_state  # noqa: F401  (autouse: снапшот констант вендора)
from tests.test_scan_market import (
    ANCHOR as MARKET_ANCHOR,
    DOWN_ARROW_DISABLED_ELEMENT,
    FILTER_ELEMENT,
    OPEN_ELEMENT,
    POINT_VERIFY,
    TAB_ELEMENT,
    UP_ARROW_DISABLED_ELEMENT,
    found_status as market_found_status,
    frame_pos0,
    frame_pos1,
    hud_locator as market_hud_locator,
    make_config as make_market_config,
    make_model as make_market_model,
)
from tests.test_scan_market import OCR_TEXTS as MARKET_OCR_TEXTS
from tests.test_scan_market import ColorOcr as MarketColorOcr
from tests.test_setup_sell import (
    ANCHOR as SHOP_ANCHOR,
    CONFIRM_VERIFY,
    DRAG_VERIFY,
    FINAL_VERIFY,
    GRID,
    KEY_ENTER_CODE,
    KEY_ESC_CODE,
    OCR_TEXTS as SHOP_OCR_TEXTS,
    OPEN_ELEMENT as SHOP_OPEN_ELEMENT,
    OPEN_VERIFY,
    PRICE_REGION,
    SELL_START_ELEMENT,
    _DIGIT_BY_CODE,
    found_status as shop_found_status,
    hud_locator as shop_hud_locator,
    make_config,
    make_hud_model,
    make_shop_model,
)
from tests.test_setup_sell import DictOcr as ShopDictOcr
from tests.test_setup_sell import FakeClock as ShopFakeClock

# --- Маркеры состояний на кадре ------------------------------------------------

# Маркер — цветной пиксель во второй строке от нижнего края кадра, колонка —
# регистрационный индекс элемента; цвет уникален. Мир рисует активные маркеры,
# MarkerRunner оценивает проверки строго по содержимому кадра (в обоих проходах
# — одна и та же оценка, к скрытому состоянию мира runner не обращается:
# replay честен).
MARKET_REGISTRY = {
    OPEN_ELEMENT: (0, (255, 40, 40)),          # HUD-иконка «Обзор лавок»
    "tab_home_marker": (1, (40, 255, 40)),     # verify открытия (переход модели)
    "list_marker": (2, (40, 40, 255)),         # verify готовности списка (точка)
    TAB_ELEMENT: (3, (255, 255, 40)),          # присутствие таба
    FILTER_ELEMENT: (4, (40, 255, 255)),       # присутствие фильтра
    UP_ARROW_DISABLED_ELEMENT: (5, (255, 40, 255)),
    DOWN_ARROW_DISABLED_ELEMENT: (6, (200, 40, 255)),
}

SHOP_REGISTRY = {
    SHOP_OPEN_ELEMENT: (0, (255, 40, 40)),     # HUD-иконка открытия лавки
    "open_marker": (1, (40, 255, 40)),         # verify открытия лавки
    "drag_marker": (2, (40, 40, 255)),         # verify drag'а
    "final_marker": (3, (255, 255, 40)),       # итоговый verify лавки
    SELL_START_ELEMENT: (4, (40, 255, 255)),   # кнопка «Начать продажу»
    "confirm_marker": (5, (255, 40, 255)),     # re-verify/verify подтверждения
}


def _paint_markers(frame: np.ndarray, registry: dict, active: set) -> None:
    row = frame.shape[0] - 2
    for name in active:
        col, color = registry[name]
        frame[row, col] = color


class MarkerRunner:
    """Оценка проверок по маркерам кадра (содержимое кадра — единый источник
    истины для прохода записи и воспроизведения)."""

    def __init__(self, registry: dict):
        self._registry = registry

    def __call__(self, checks, frame, anchor) -> CheckReport:
        row = frame.shape[0] - 2
        failures = []
        for check in checks:
            entry = self._registry.get(check.element)
            if entry is None:
                failures.append(check.element or "?")
                continue
            col, color = entry
            if tuple(int(v) for v in frame[row, col]) != color:
                failures.append(check.element or "?")
        return CheckReport(passed=not failures, failures=tuple(failures))


# --- Скриптованные HID-проходы -------------------------------------------------


class RecordingHid:
    """HID прохода ЗАПИСИ: команда исполняется в «мире», эффектное действие
    фиксируется с кадром-после (нормализация — JSON-совместимая, как в сессиях)."""

    def __init__(self, screen, clock, react):
        self.screen = screen
        self.clock = clock
        self._react = react
        self._cursor = [500, 300]
        self.events: list = []  # [(action_dict, frame_after)]

    def cursor_get(self):
        return (self._cursor[0], self._cursor[1])

    def _execute(self, action: dict) -> None:
        self._react(action)
        self.events.append((action, np.asarray(self.screen.grab()).copy()))

    def move_rel(self, dx, dy):
        self._cursor[0] += dx
        self._cursor[1] += dy

    def click(self, button=1, count=1, hold_ms=None):
        self._execute({"type": "click", "pos": [self._cursor[0], self._cursor[1]],
                       "button": button, "count": count, "hold_ms": hold_ms})

    def key_press(self, code):
        self._execute({"type": "key_press", "code": code})

    def key_down(self, code):
        self._execute({"type": "key_down", "code": code})

    def key_up(self, code):
        self._execute({"type": "key_up", "code": code})

    def scroll(self, count, step, delay_ms):
        self._execute({"type": "scroll", "count": count, "step": step,
                       "delay_ms": delay_ms})

    def mouse_down(self, button=1):
        self._execute({"type": "mouse_down", "button": button})

    def mouse_up(self, button=1):
        self._execute({"type": "mouse_up", "button": button})

    def traj(self, points):
        self._execute({"type": "traj",
                       "points": [[int(dx), int(dy), int(d)] for dx, dy, d in points]})


class ReplayHid:
    """HID прохода ВОСПРОИЗВЕДЕНИЯ: та же нормализация действий, что и при
    записи; каждое эффектное действие уходит в ReplayScreen.notify_action —
    строгое сопоставление с записью (несовпадение — ActionMismatchError)."""

    def __init__(self, screen, clock):
        self.screen = screen
        self.clock = clock
        self._cursor = [500, 300]
        self.log: list = []

    def cursor_get(self):
        return (self._cursor[0], self._cursor[1])

    def _execute(self, action: dict) -> None:
        self.screen.notify_action(action)
        self.log.append(action)

    def move_rel(self, dx, dy):
        self._cursor[0] += dx
        self._cursor[1] += dy

    def click(self, button=1, count=1, hold_ms=None):
        self._execute({"type": "click", "pos": [self._cursor[0], self._cursor[1]],
                       "button": button, "count": count, "hold_ms": hold_ms})

    def key_press(self, code):
        self._execute({"type": "key_press", "code": code})

    def key_down(self, code):
        self._execute({"type": "key_down", "code": code})

    def key_up(self, code):
        self._execute({"type": "key_up", "code": code})

    def scroll(self, count, step, delay_ms):
        self._execute({"type": "scroll", "count": count, "step": step,
                       "delay_ms": delay_ms})

    def mouse_down(self, button=1):
        self._execute({"type": "mouse_down", "button": button})

    def mouse_up(self, button=1):
        self._execute({"type": "mouse_up", "button": button})

    def traj(self, points):
        self._execute({"type": "traj",
                       "points": [[int(dx), int(dy), int(d)] for dx, dy, d in points]})


class WorldScreen:
    """Экран «мира» прохода записи: кадр — чистая функция состояния."""

    def __init__(self, world):
        self.world = world

    def grab(self):
        return self.world.render()


class SessionFrameFeed(Screen):
    """Подача кадров сессии observer-сценарию (monitor): каждый grab —
    следующий кадр записи (последний повторяется). Мир меняется между poll'ами
    без действий ядра; курсор ReplayScreen двигают только действия (формат 3.1),
    поэтому для наблюдателя — тестовый экран-податчик."""

    def __init__(self, session_path):
        self._session = load_session(session_path)
        self._refs = [e["image"] for e in self._session.events
                      if e["type"] == "frame" and e["image"] is not None]
        assert self._refs, "в сессии монитора должны быть кадры"
        self._pos = 0

    def grab(self):
        ref = self._refs[min(self._pos, len(self._refs) - 1)]
        if self._pos < len(self._refs) - 1:
            self._pos += 1
        return self._session.read_image(ref)


# --- Проход записи / сборка сессии ---------------------------------------------


def capture_session(world, run):
    """Проход записи: исполнить сценарий(и) против мира; вернуть (result,
    initial_frame, events) — result последнего вызова run."""
    screen = WorldScreen(world)
    clock = ShopFakeClock()
    hid = RecordingHid(screen, clock, world.react)
    initial = np.asarray(screen.grab()).copy()
    result = run(hid)
    return result, initial, hid.events


def write_session(path, initial, events, meta) -> None:
    with SessionWriter(path, meta=meta) as writer:
        writer.record_frame(initial)
        for action, frame_after in events:
            writer.record_action(action, frame_after=frame_after)


def recorded_actions(events) -> list:
    return [action for action, _ in events]


# --- Мир «Обзор лавок» ----------------------------------------------------------


class MarketWorld:
    """Мир обзора: скролл-позиция pos, открытие/таб/фильтр; кадр — функция
    состояния (таблица из test_scan_market + маркеры). pos>=1 — признак
    «вниз disabled» и кадр с пересекающейся строкой A (дедуп)."""

    REGISTRY = MARKET_REGISTRY

    def __init__(self):
        self.pos = 0
        self.opened = False
        self.active = {OPEN_ELEMENT}

    def react(self, action: dict) -> None:
        kind = action["type"]
        if kind == "click":
            x, y = action["pos"]
            if not self.opened:
                # До открытия клик — только по HUD-иконке (сценарий такой).
                self.opened = True
                self.active |= {"tab_home_marker", "list_marker",
                                TAB_ELEMENT, FILTER_ELEMENT}
            elif 10 <= x <= 30 and 60 <= y <= 70:
                self.active.add(TAB_ELEMENT)
            elif 40 <= x <= 60 and 60 <= y <= 70:
                self.active.add(FILTER_ELEMENT)
        elif kind == "scroll":
            self.pos += action["count"] if action["step"] > 0 else -action["count"]

    def render(self):
        frame = frame_pos0() if self.pos == 0 else frame_pos1()
        active = set(self.active)
        if self.pos >= 1:
            active.add(DOWN_ARROW_DISABLED_ELEMENT)
        _paint_markers(frame, self.REGISTRY, active)
        return frame


def run_scan_scenario(hid):
    """Один и тот же closure для прохода записи и воспроизведения
    (одинаковые seed/часы/инъекции — детерминированная пара проходов)."""
    return scan_market(
        [ScanPoint(verify=POINT_VERIFY, tab=TAB_ELEMENT,
                   filters=(FILTER_ELEMENT,))],
        model=make_market_model(),
        ocr=MarketColorOcr(MARKET_OCR_TEXTS),
        screen=hid.screen,
        hid=hid,
        cursor_pos=hid.cursor_get,
        activate=lambda: ActivateResult(hwnd=1, already_active=True,
                                        attempts=0, cancelled=False),
        window_status=market_found_status,
        find_anchor=lambda frame: MARKET_ANCHOR,
        run_checks=MarkerRunner(MARKET_REGISTRY),
        config=make_market_config(),
        hud_locator=market_hud_locator,
        scroll_chunk=1,
        rng=random.Random(7),
        clock=hid.clock.monotonic,
        sleep=hid.clock.sleep,
    )


SCAN_META = {"source": "task-9.2-synthetic", "scenario": "scan_market"}


# --- Мир «Торговая лавка (Продажа)» ---------------------------------------------


class ShopWorld:
    """Мир лавки продажи: открытие, буфер цены (набор цифр key_press'ами);
    маркеры open/drag/final/start/confirm активны после открытия."""

    REGISTRY = SHOP_REGISTRY

    def __init__(self):
        self.opened = False
        self.price_buffer = ""
        self.active = {SHOP_OPEN_ELEMENT}

    def react(self, action: dict) -> None:
        kind = action["type"]
        if kind == "click":
            if not self.opened:
                self.opened = True
                self.active |= {"open_marker", "drag_marker", "final_marker",
                                SELL_START_ELEMENT, "confirm_marker"}
            # Клик по «Начать продажу»: мир уже в состоянии продажи,
            # маркеры те же — изменений нет.
        elif kind == "key_press":
            code = action["code"]
            if code in (KEY_ENTER_CODE, KEY_ESC_CODE):
                self.price_buffer = ""
            elif code == 0xB2:  # KEY_BACKSPACE
                self.price_buffer = self.price_buffer[:-1]
            elif code in _DIGIT_BY_CODE:
                self.price_buffer += _DIGIT_BY_CODE[code]
        # mouse_down/traj/mouse_up: drag в этом мире не меняет картинку.

    def render(self):
        frame = np.full((300, 400, 3), 30, dtype=np.uint8)
        color = {v[0]: k for k, v in SHOP_OCR_TEXTS.items()}.get(self.price_buffer)
        if color is not None:
            r = PRICE_REGION.rect
            y0, x0 = SHOP_ANCHOR.y + r.y, SHOP_ANCHOR.x + r.x
            frame[y0:y0 + r.height, x0:x0 + r.width] = color
        _paint_markers(frame, self.REGISTRY, self.active)
        return frame


def run_setup_and_start_sell(hid, items=(SellItem(slot=1, price=1500000),)):
    """setup_sell() + start_sell() над одним экраном/сессией (closure общий
    для обоих проходов; items фиксирует обход — смена списка ломает
    воспроизведение, см. негативный тест)."""
    setup = setup_sell(
        list(items),
        model=make_shop_model(),
        hud_model=make_hud_model(),
        grid=GRID,
        ocr=ShopDictOcr(SHOP_OCR_TEXTS),
        screen=hid.screen,
        hid=hid,
        cursor_pos=hid.cursor_get,
        activate=lambda: ActivateResult(hwnd=1, already_active=True,
                                        attempts=0, cancelled=False),
        window_status=shop_found_status,
        find_anchor=lambda frame: SHOP_ANCHOR,
        run_checks=MarkerRunner(SHOP_REGISTRY),
        config=make_config(),
        hud_locator=shop_hud_locator,
        open_element=SHOP_OPEN_ELEMENT,
        open_verify=OPEN_VERIFY,
        source_locator=lambda item: (40, 120),
        price_region=PRICE_REGION,
        price_field_element=None,
        drag_verify=DRAG_VERIFY,
        final_verify=FINAL_VERIFY,
        rng=random.Random(7),
        clock=hid.clock.monotonic,
        sleep=hid.clock.sleep,
    )
    guard = IrreversibleGuard()
    started = start_sell(
        model=make_shop_model(),
        screen=hid.screen,
        hid=hid,
        cursor_pos=hid.cursor_get,
        activate=lambda: ActivateResult(hwnd=1, already_active=True,
                                        attempts=0, cancelled=False),
        window_status=shop_found_status,
        find_anchor=lambda frame: SHOP_ANCHOR,
        run_checks=MarkerRunner(SHOP_REGISTRY),
        config=make_config(),
        confirm_checks=CONFIRM_VERIFY,
        verify=CONFIRM_VERIFY,
        element_id=SELL_START_ELEMENT,
        guard=guard,
        rng=random.Random(3),
        clock=hid.clock.monotonic,
        sleep=hid.clock.sleep,
    )
    return setup, started, guard


SHOP_META = {"source": "task-9.2-synthetic", "scenario": "setup_sell_shop"}


# --- Прогоны --------------------------------------------------------------------


class TestScanMarketReplay:
    def test_scan_replays_recorded_session(self, tmp_path):
        # Проход записи: сценарий исполняется против мира, действия+кадры фиксируются.
        captured, initial, events = capture_session(MarketWorld(), run_scan_scenario)
        assert captured.failures == ()
        assert len(captured.points[0].rows) == 3  # мир сходен: дедуп дал 3 строки
        path = tmp_path / "scan.session"
        write_session(path, initial, events, meta=SCAN_META)

        session = load_session(path)
        assert len(session.events) == 1 + 4  # начальный кадр + 4 действия
        # Открытие -> таб -> фильтр -> wheel (порядок зафиксирован записью).
        assert [e["action"]["type"] for e in session.events
                if e["type"] == "action"] == ["click", "click", "click", "scroll"]

        # Проход воспроизведения: тот же сценарий против записи, без mss/HID/игры.
        screen = ReplayScreen(path)
        hid = ReplayHid(screen, ShopFakeClock())
        assert isinstance(screen, ReplayScreen)
        result = run_scan_scenario(hid)

        assert result.cancelled is False
        assert result.failures == ()
        (point,) = result.points
        # OCR-результаты с записанных кадров: агрегат со дедупом пересекающихся
        # строк (A на pos0 и pos1 — одна строка).
        assert [(r.cells["item"].value, r.cells["price"].value, r.screen_y)
                for r in point.rows] == [
            ("Меч", 1500000, 20), ("Щит", 25000, 30), ("Зелье", 1500000, 20),
        ]
        assert len(point.snapshots) == 2
        assert point.snapshots[0].snapshot_id != point.snapshots[1].snapshot_id
        # Последовательность действий сценария сошлась с записью ровно.
        assert hid.log == recorded_actions(events)
        # Все записанные действия исчерпаны: сценарий дошёл до конца сессии.
        with pytest.raises(SessionExhaustedError):
            screen.notify_action({"type": "click", "pos": [0, 0], "button": 1,
                                  "count": 1, "hold_ms": 1})
        # Текущий кадр replay — последний записанный (мир дошёл до финала).
        assert np.array_equal(screen.grab(), events[-1][1])

    def test_tampered_session_breaks_with_action_mismatch(self, tmp_path):
        # Негатив: подменённое действие в записи — сценарий реально ломается
        # (строгое сопоставление — assert последовательности состояний).
        _, initial, events = capture_session(MarketWorld(), run_scan_scenario)
        path = tmp_path / "scan-tampered.session"
        write_session(path, initial, events, meta=SCAN_META)
        manifest = json.loads((path / "session.json").read_text(encoding="utf-8"))
        actions = [e for e in manifest["events"] if e["type"] == "action"]
        actions[1]["action"]["pos"] = [999, 999]  # клик по табу «испорчен»
        (path / "session.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        screen = ReplayScreen(path)
        with pytest.raises(ActionMismatchError):
            run_scan_scenario(ReplayHid(screen, ShopFakeClock()))


class TestSetupSellReplay:
    def test_setup_and_start_sell_replay(self, tmp_path):
        # Проход записи: setup_sell + start_sell против мира лавки.
        (captured_setup, captured_start, _), initial, events = capture_session(
            ShopWorld(), run_setup_and_start_sell)
        assert not captured_setup.needs_recover
        assert captured_start.success and captured_start.outcome is ActionOutcome.OK

        path = tmp_path / "shop.session"
        write_session(path, initial, events, meta=SHOP_META)

        session = load_session(path)
        kinds = [e["action"]["type"] for e in session.events if e["type"] == "action"]
        # Открытие -> (down, traj, up) -> 7 цифр -> Enter -> клик «Начать продажу».
        assert kinds[0] == "click"
        assert kinds[1:4] == ["mouse_down", "traj", "mouse_up"]
        assert kinds[-1] == "click"
        digits = [a["code"] for a in (e["action"] for e in session.events
                                      if e["type"] == "action")
                  if a["type"] == "key_press"]
        assert digits == [0x1E + (int(d) - 1) % 10 for d in "1500000"] + [0xB0]

        screen = ReplayScreen(path)
        hid = ReplayHid(screen, ShopFakeClock())
        setup, started, guard = run_setup_and_start_sell(hid)

        # Выставление прошло целиком по записи: drag и ввод цены подтверждены.
        assert not setup.needs_recover and not setup.cancelled
        (item,) = setup.items
        assert item.drag.success
        assert item.price.success and item.price.submit_method == "enter"
        # start_sell: re-verify перед необратимым прошёл, клик ушёл и
        # подтверждён — страж необратимых НЕ блокирует повтор (OK, не UNKNOWN).
        assert started.reverified
        assert started.success and started.outcome is ActionOutcome.OK
        assert not guard.is_blocked("start_sell")
        # Последовательность действий сошлась с записью ровно.
        assert hid.log == recorded_actions(events)
        with pytest.raises(SessionExhaustedError):
            screen.notify_action({"type": "click", "pos": [0, 0], "button": 1,
                                  "count": 1, "hold_ms": 1})

    def test_diverged_scenario_breaks_with_action_mismatch(self, tmp_path):
        # Негатив: сценарий отклонился от записи (в обходе на предмет больше,
        # чем записано) — первое «лишнее» действие не сходится с записью.
        # Проверки присутствия при этом проходят: сессия — своя, испорчена
        # именно последовательность действий.
        _, initial, events = capture_session(ShopWorld(), run_setup_and_start_sell)
        path = tmp_path / "shop.session"
        write_session(path, initial, events, meta=SHOP_META)

        screen = ReplayScreen(path)
        extra = (SellItem(slot=1, price=1500000), SellItem(slot=4, price=25000))
        with pytest.raises(ActionMismatchError):
            run_setup_and_start_sell(ReplayHid(screen, ShopFakeClock()),
                                     items=extra)


# --- Мониторинг -----------------------------------------------------------------

COUNTER_REGION = OcrRegionSpec(
    id="sold_count",
    rect=Rect(20, 10, 30, 10),
    grammar=GrammarSpec(name="digits", kind="digits"),
)
REVENUE_REGION = OcrRegionSpec(
    id="revenue",
    rect=Rect(20, 30, 50, 10),
    grammar=GrammarSpec(name="digits", kind="digits"),
)

COLOR_CNT0 = (11, 22, 33)
COLOR_CNT1 = (44, 55, 66)
COLOR_REV0 = (77, 88, 99)
COLOR_REV1 = (111, 122, 133)

MONITOR_OCR_TEXTS = {
    COLOR_CNT0: ("0", 0.99),
    COLOR_CNT1: ("1", 0.99),
    COLOR_REV0: ("0", 0.99),
    COLOR_REV1: ("1500000", 0.99),
}


def render_monitor_frame(counter_color, revenue_color) -> np.ndarray:
    """Кадр лавки с показателями: маркер — заливка центральной части региона
    (контраст с фоном региона есть при любом чтении)."""
    frame = np.full((300, 400, 3), 30, dtype=np.uint8)
    for region, color in ((COUNTER_REGION, counter_color),
                          (REVENUE_REGION, revenue_color)):
        r = region.rect
        y0, x0 = SHOP_ANCHOR.y + r.y, SHOP_ANCHOR.x + r.x
        # Маркер меньше региона: перечитывание с расширением (4.4/4.5) тоже
        # видит контраст, первое чтение — тоже.
        frame[y0 + 2:y0 + r.height - 2, x0 + 4:x0 + r.width - 4] = color
    return frame


class TestMonitorShopReplay:
    def test_monitor_detects_sold_on_recorded_frames(self, tmp_path):
        # Сессия монитора — только кадры (мир меняется без действий ядра).
        path = tmp_path / "monitor.session"
        with SessionWriter(path, meta={"source": "task-9.2-synthetic",
                                       "scenario": "monitor_shop"}) as writer:
            writer.record_frame(render_monitor_frame(COLOR_CNT0, COLOR_REV0))
            writer.record_frame(render_monitor_frame(COLOR_CNT1, COLOR_REV1))

        sold_events = []
        monitor = ShopMonitor(
            model=make_shop_model(),
            ocr=ShopDictOcr(MONITOR_OCR_TEXTS),
            screen=SessionFrameFeed(path),
            find_anchor=lambda frame: SHOP_ANCHOR,
            config=make_config(),
            regions={"sold_count": COUNTER_REGION, "revenue": REVENUE_REGION},
            on_sold=sold_events.append,
        )
        baseline = monitor.capture_baseline()
        assert not baseline.failures
        assert monitor.baseline is not None

        poll = monitor.poll()
        assert not poll.degraded and poll.failures == ()
        (payload,) = sold_events
        # OCR-результаты с записанных кадров: показатели выросли к линии.
        assert payload.sold_count == 1 and payload.sold_delta == 1
        assert payload.revenue == 1500000 and payload.revenue_delta == 1500000
        assert poll.sold == payload
        assert payload.snapshot_id == poll.snapshot_id
        # Повторный poll на том же кадре: линия уже сдвинута — события нет.
        again = monitor.poll()
        assert again.sold is None
        assert not again.degraded
