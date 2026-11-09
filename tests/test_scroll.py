"""Тесты scroll_list/scroll_position (задача 5.5, spec market-scenarios:
«Сканирование списка до конца», action-primitives: verify/retry/on_fail;
safety: слепых действий нет — инвариант 9).

Детерминизм (инвариант 7): ScriptedHid фиксирует действия с метками фейкового
времени, скриптованные экран/оценка стрелок, часы — FakeClock (sleep
продвигает время). Кадры синтетические: окно со списком собирается из
параметров (нижняя строка — яркий блок, значение меняет хэш). Никаких
обращений к живому экрану/HID.
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
from l2core.detector.status import (
    STATUS_FOUND,
    STATUS_OCCLUDED,
    WindowStatus,
)
from l2core.model import (
    Check,
    ColumnZone,
    ElementSpec,
    GrammarSpec,
    Rect,
    TableGeometry,
    UiModel,
    VariantSpec,
    WindowSpec,
)
from l2core.primitives.activate import ActivateResult
from l2core.primitives.click import CheckReport
from l2core.primitives.scroll import (
    DIR_DOWN,
    DIR_UP,
    DOWN_ARROW_DISABLED_ELEMENT,
    UP_ARROW_DISABLED_ELEMENT,
    ScrollError,
    ScrollPosition,
    ScrollPreconditionError,
    ScrollResult,
    scroll_list,
    scroll_position,
)

ANCHOR = Anchor(window_id="market", x=0, y=0, score=0.99, method="template")
WINDOW_SIZE = (100, 60)


@pytest.fixture(autouse=True)
def vendor_state():
    """Снапшот/восстановление module-level констант вендора сегментации.

    TableSegmentation подменяет их процесс-wide (документировано): без
    восстановления синтетическая геометрия теста протухла бы гвард-эталоны
    вендора (tests/test_table_segmentation_vendor.py).
    """
    from l2core.adapters.table_segmentation import build_dataset as vendor

    keys = (
        "CONTENT_X0", "COL_SEPS", "CONTENT_X1", "COLUMNS",
        "HEADER_Y0", "HEADER_Y1", "ROW_Y0", "ROW_PITCH", "ROW_H",
        "TABLE_BOTTOM", "MAX_ROWS", "NAME_TEXT_X", "PRICE_TEXT_X",
    )
    snapshot = {k: getattr(vendor, k) for k in keys}
    yield
    for k, v in snapshot.items():
        setattr(vendor, k, v)

# Геометрия таблицы синтетического окна: 2 строки (y 20..30, 30..40),
# колонки item x 10..50, price x 50..90; зоны текста без иконок.
GEOMETRY = TableGeometry(
    col_seps=(10, 50, 90),
    row_y0=20,
    row_pitch=10,
    max_rows=2,
    columns={
        "item": ColumnZone(id="item", zone=Rect(12, 2, 30, 6),
                           grammar=GrammarSpec(name="text", kind="text")),
        "price": ColumnZone(id="price", zone=Rect(52, 2, 30, 6),
                            grammar=GrammarSpec(name="price", kind="price")),
    },
)


class FakeClock:
    """Фейковые часы: sleep продвигает время."""

    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class ScriptedScreen:
    """Экран со скриптованной последовательностью кадров (последний — по кругу)."""

    def __init__(self, frames):
        self._frames = list(frames)
        self._index = 0

    def grab(self):
        frame = self._frames[min(self._index, len(self._frames) - 1)]
        if self._index < len(self._frames) - 1:
            self._index += 1
        return frame


class MovingScreen:
    """Экран, на котором каждый grab даёт кадр с новой нижней строкой."""

    def __init__(self, start=100):
        self._value = start

    def grab(self):
        self._value += 17
        return make_frame(self._value)


class ScriptedHid:
    """HID с записью действий: (время, действие, args)."""

    def __init__(self, clock):
        self.clock = clock
        self.log = []

    def move_rel(self, dx, dy):
        self.log.append((self.clock.now, "move_rel", (dx, dy)))

    def scroll(self, count, step, delay_ms):
        self.log.append((self.clock.now, "scroll", (count, step, delay_ms)))

    def key_press(self, code):
        self.log.append((self.clock.now, "key_press", (code,)))

    def click(self, button=1, count=1, hold_ms=None):
        self.log.append((self.clock.now, "click", (button, count, hold_ms)))


class FakeCursor:
    """Позиция курсора: move_rel двигает."""

    def __init__(self, pos=(500, 300)):
        self.pos = pos

    def get(self):
        return self.pos

    def track(self, hid):
        original = hid.move_rel

        def move_rel(dx, dy):
            self.pos = (self.pos[0] + dx, self.pos[1] + dy)
            original(dx, dy)

        hid.move_rel = move_rel


class ArrowChecker:
    """Скриптованная оценка проверок стрелок: очередь множеств disabled-элементов.

    Элемент в текущем множестве — шаблон его disabled-состояния совпал на
    кадре (проверка проходит, стрелка disabled). Каждый вызов оценивает
    ВСЕ checks одним passed (последнее множество — по кругу).
    """

    def __init__(self, disabled_sets=None):
        self._sets = list(disabled_sets or [set()])
        self.calls = []

    def __call__(self, checks, frame, anchor):
        self.calls.append(tuple(checks))
        # Пара вызовов (up+down) — один «кадр»: множества по номеру кадра.
        step = (len(self.calls) - 1) // 2
        disabled = self._sets[min(step, len(self._sets) - 1)]
        passed = all(c.element in disabled for c in checks)
        failures = () if passed else tuple(
            f"not-disabled:{c.element}" for c in checks
            if c.element not in disabled)
        return CheckReport(passed=passed, failures=failures)


def make_frame(bottom_value=200, size=(160, 100)):
    """Синтетический кадр: окно market в (0,0), нижняя строка — яркий блок
    значения bottom_value в зоне item второй строки (y 30..40)."""
    frame = np.full((size[1], size[0], 3), 30, dtype=np.uint8)
    # Блок меньше половины зоны item строки (30x10): медиана фона
    # вендорной эвристики _row_is_empty остаётся фоном.
    frame[30:34, 12:42] = bottom_value
    return frame


def make_model(with_arrows=True, with_geometry=True):
    elements = {}
    if with_arrows:
        for eid in (UP_ARROW_DISABLED_ELEMENT, DOWN_ARROW_DISABLED_ELEMENT):
            elements[eid] = ElementSpec(
                id=eid,
                role="click_target",
                rect=Rect(90, 10, 8, 8),
                variants={"active": VariantSpec(
                    name="active", template=f"test/{eid}.png")},
            )
    return UiModel(
        path="test",
        window=WindowSpec(id="market", title="Обзор лавок", size=WINDOW_SIZE),
        elements=elements,
        states={},
        transitions=(),
        ocr_regions={},
        table_geometry=GEOMETRY if with_geometry else None,
    )


def make_config(**timing_kwargs):
    timing = TimingConfig(wait_default_timeout=0.5, verify_interval=0.05,
                          **timing_kwargs)
    return CoreConfig(
        timing=timing,
        behavior=BehaviorConfig(click_jitter_px=0, press_delay_min_ms=40,
                                press_delay_max_ms=120),
        resolution=ResolutionConfig(width=1920, height=1080),
    )


def found_status(frame):
    return WindowStatus(
        status=STATUS_FOUND, reason=None, anchor=ANCHOR,
        content_loaded=None, content_detail="",
    )


def run_scroll(direction, n, hid, clock, checker, *, model=None, screen=None,
               activate=None, status=None, on_fail=(), rng_seed=7,
               wheel_delay_ms=0, cancelled=None, find_anchor=None):
    model = model or make_model()
    activate = activate or (lambda: ActivateResult(
        hwnd=1, already_active=True, attempts=0, cancelled=False))
    return scroll_list(
        direction, n,
        model=model,
        screen=screen or MovingScreen(),
        hid=hid,
        cursor_pos=FakeCursor().get,
        activate=activate,
        window_status=status or found_status,
        find_anchor=find_anchor or (lambda frame: ANCHOR),
        run_checks=checker,
        config=make_config(),
        on_fail=on_fail,
        rng=random.Random(rng_seed),
        cancelled=cancelled,
        wheel_delay_ms=wheel_delay_ms,
        clock=clock.monotonic,
        sleep=clock.sleep,
    )


class TestScrollDown:
    def test_scroll_down_n_wheels_verify_each(self):
        # Скролл вниз на n=3: ровно 3 wheel-команды (+1 вверх нет),
        # каждая итерация подтверждена сменой нижней строки.
        clock = FakeClock()
        hid = ScriptedHid(clock)
        checker = ArrowChecker()
        result = run_scroll(DIR_DOWN, 3, hid, clock, checker)

        assert isinstance(result, ScrollResult)
        assert result.success is True
        assert result.scrolled == 3
        assert result.requested == 3
        assert result.reached_end is False
        assert result.needs_recover is False
        assert result.cancelled is False
        wheels = [a for _, a, _ in hid.log if a == "scroll"]
        assert len(wheels) == 3
        # Единица n = щелчок колеса: count=1, шаг +1 (вниз).
        steps = [args for _, a, args in hid.log if a == "scroll"]
        assert steps == [(1, 1, 0), (1, 1, 0), (1, 1, 0)]
        # Позиция прочитана: нижняя строка менялась между кадрами.
        assert result.position is not None
        assert result.position.bottom_row_hash is not None

    def test_wheel_step_sign_by_direction(self):
        # Вверх — отрицательный шаг колеса, вниз — положительный.
        clock = FakeClock()
        hid = ScriptedHid(clock)
        run_scroll(DIR_UP, 1, hid, clock, ArrowChecker())
        steps = [args for _, a, args in hid.log if a == "scroll"]
        assert steps == [(1, -1, 0)]

    def test_wheel_delay_ms_passed(self):
        clock = FakeClock()
        hid = ScriptedHid(clock)
        run_scroll(DIR_DOWN, 1, hid, clock, ArrowChecker(), wheel_delay_ms=25)
        steps = [args for _, a, args in hid.log if a == "scroll"]
        assert steps == [(1, 1, 25)]

    def test_cursor_hovered_over_table_center(self):
        # Позиционирование: центр области таблицы (col_seps 10..90, строки
        # 20..40 от якоря) -> (50, 30); движение относительными дельтами.
        clock = FakeClock()
        hid = ScriptedHid(clock)
        cursor = FakeCursor()
        cursor.track(hid)
        run_scroll(DIR_DOWN, 1, hid, clock, ArrowChecker())
        moves = [args for _, a, args in hid.log if a == "move_rel"]
        assert moves == [(50 - 500, 30 - 300)]

    def test_arrow_checks_evaluated_from_model(self):
        # Позиция читает признаки стрелок проверками template по disabled-
        # элементам модели (вариант «active» в записи элемента).
        checker = ArrowChecker()
        pos = scroll_position(make_frame(), model=make_model(),
                              anchor=ANCHOR, run_checks=checker)
        checked = {(c.element, c.variant) for call in checker.calls for c in call}
        assert checked == {
            (UP_ARROW_DISABLED_ELEMENT, "active"),
            (DOWN_ARROW_DISABLED_ELEMENT, "active"),
        }
        assert isinstance(pos, ScrollPosition)
        assert pos.at_start is False
        assert pos.at_end is False


class TestEndOfList:
    def test_end_marker_stops_scrolling(self):
        # Стрелка «вниз» disabled на кадре после первого щелчка: verify
        # итерации сходится по признаку модели, дальше крутить не надо.
        clock = FakeClock()
        hid = ScriptedHid(clock)
        checker = ArrowChecker(disabled_sets=[set(), {DOWN_ARROW_DISABLED_ELEMENT}])
        result = run_scroll(DIR_DOWN, 5, hid, clock, checker)
        assert result.scrolled == 1
        assert result.reached_end is True
        assert result.success is True
        wheels = [a for _, a, _ in hid.log if a == "scroll"]
        assert len(wheels) == 1
        assert result.position.at_end is True

    def test_already_at_end_no_wheels(self):
        # Стрелка «вниз» disabled с самого начала: ни одного щелчка.
        clock = FakeClock()
        hid = ScriptedHid(clock)
        checker = ArrowChecker(disabled_sets=[{DOWN_ARROW_DISABLED_ELEMENT}])
        result = run_scroll(DIR_DOWN, 3, hid, clock, checker)
        assert result.scrolled == 0
        assert result.reached_end is True
        assert result.success is True
        assert [a for _, a, _ in hid.log if a == "scroll"] == []

    def test_at_start_blocks_scroll_up(self):
        # Стрелка «вверх» disabled: скролл вверх не выполняется.
        clock = FakeClock()
        hid = ScriptedHid(clock)
        checker = ArrowChecker(disabled_sets=[{UP_ARROW_DISABLED_ELEMENT}])
        result = run_scroll(DIR_UP, 3, hid, clock, checker)
        assert result.scrolled == 0
        assert result.reached_end is True  # конец по направлению (начало)
        assert [a for _, a, _ in hid.log if a == "scroll"] == []

    def test_fallback_hash_unchanged(self):
        # Модель без стрелок (at_end=None): конец честно не детектируется,
        # verify опирается на смену нижней строки; кадр статичен — итерация
        # не подтверждается, retry исчерпаны, on_fail исполнен.
        clock = FakeClock()
        hid = ScriptedHid(clock)
        screen = ScriptedScreen([make_frame(200)] * 40)
        result = run_scroll(DIR_DOWN, 1, hid, clock, ArrowChecker(),
                            model=make_model(with_arrows=False), screen=screen,
                            on_fail=("press_esc",))
        assert result.success is False
        assert result.scrolled == 0
        assert result.reached_end is False
        assert result.needs_recover is True
        assert result.on_fail_executed is True
        # retry: 1 + click_retry_count(2) = 3 wheel-команды на итерацию.
        wheels = [a for _, a, _ in hid.log if a == "scroll"]
        assert len(wheels) == 3
        esc = [args for _, a, args in hid.log if a == "key_press"]
        assert esc  # press_esc из on_fail

    def test_unknown_on_fail_step_is_model_error(self):
        from l2core.primitives.click import ClickElementError

        clock = FakeClock()
        hid = ScriptedHid(clock)
        screen = ScriptedScreen([make_frame(200)] * 40)
        with pytest.raises(ClickElementError):
            run_scroll(DIR_DOWN, 1, hid, clock, ArrowChecker(),
                       model=make_model(with_arrows=False), screen=screen,
                       on_fail=("teleport",))


class TestRetry:
    def test_retry_then_success(self):
        # Первая попытка итерации не подтверждается (статичный кадр), вторая
        # — сменой строки: итерация засчитана, провал зафиксирован.
        clock = FakeClock()
        hid = ScriptedHid(clock)
        static = [make_frame(200)] * 20
        moving = [make_frame(v) for v in range(100, 200, 17)]
        screen = ScriptedScreen(static + moving)
        result = run_scroll(DIR_DOWN, 1, hid, clock, ArrowChecker(), screen=screen)
        assert result.success is True
        assert result.scrolled == 1
        wheels = [a for _, a, _ in hid.log if a == "scroll"]
        assert len(wheels) == 2  # провал + успешный повтор
        assert result.failures  # диагностика провала первой попытки


class TestNoBlindScroll:
    def test_no_initial_anchor_returns_recover_without_wheels(self):
        # Стартовый кадр без якоря: позиционировать курсор на область списка
        # нельзя — wheel в непроверенную позицию запрещён (инвариант 9):
        # ноль wheel-команд, сразу сигнал recover.
        clock = FakeClock()
        hid = ScriptedHid(clock)
        result = run_scroll(DIR_DOWN, 3, hid, clock, ArrowChecker(),
                            find_anchor=lambda frame: None)
        assert result.success is False
        assert result.needs_recover is True
        assert result.scrolled == 0
        assert result.wheels == 0
        assert [a for _, a, _ in hid.log if a == "scroll"] == []

    def test_anchor_lost_before_wheel_skips_attempt(self):
        # Якорь есть на старте (hover выполнен), потерян на свежем кадре
        # перед wheel попытки 1: команда НЕ уходит, попытка потрачена;
        # попытка 2 с подтверждённым якорем — wheel уходит, verify OK.
        clock = FakeClock()
        hid = ScriptedHid(clock)
        anchors = [ANCHOR, None, ANCHOR, ANCHOR]
        state = {"i": 0}

        def find_anchor(frame):
            i = min(state["i"], len(anchors) - 1)
            state["i"] += 1
            return anchors[i]

        result = run_scroll(DIR_DOWN, 1, hid, clock, ArrowChecker(),
                            find_anchor=find_anchor)
        assert result.success is True
        assert result.scrolled == 1
        wheels = [args for _, a, args in hid.log if a == "scroll"]
        assert wheels == [(1, 1, 0)]  # wheel только подтверждённой попытки
        assert result.failures  # потеря якоря зафиксирована в диагностике

    def test_anchor_lost_all_attempts_recover_without_wheels(self):
        # Якорь потерян на всех retry-попытках итерации: ни одного wheel,
        # on_fail исполнен, сигнал recover.
        clock = FakeClock()
        hid = ScriptedHid(clock)
        anchors = [ANCHOR, None, None, None]
        state = {"i": 0}

        def find_anchor(frame):
            i = min(state["i"], len(anchors) - 1)
            state["i"] += 1
            return anchors[i]

        result = run_scroll(DIR_DOWN, 1, hid, clock, ArrowChecker(),
                            find_anchor=find_anchor, on_fail=("press_esc",))
        assert result.success is False
        assert result.needs_recover is True
        assert result.wheels == 0
        assert [a for _, a, _ in hid.log if a == "scroll"] == []
        assert result.on_fail_executed is True


class TestScrollPosition:
    def test_bottom_row_hash_changes_between_frames(self):
        checker = ArrowChecker()
        model = make_model()
        h1 = scroll_position(make_frame(200), model=model, anchor=ANCHOR,
                             run_checks=checker).bottom_row_hash
        h2 = scroll_position(make_frame(180), model=model, anchor=ANCHOR,
                             run_checks=checker).bottom_row_hash
        h1_repeat = scroll_position(make_frame(200), model=model, anchor=ANCHOR,
                                    run_checks=checker).bottom_row_hash
        assert h1 is not None and h2 is not None
        assert h1 != h2
        assert h1 == h1_repeat

    def test_arrows_absent_in_model(self):
        # Элементов-стрелок нет: края не выражены (None), хэш живёт.
        checker = ArrowChecker()
        pos = scroll_position(make_frame(), model=make_model(with_arrows=False),
                              anchor=ANCHOR, run_checks=checker)
        assert pos.at_start is None
        assert pos.at_end is None
        assert pos.bottom_row_hash is not None
        assert checker.calls == []  # проверок стрелок не задаём

    def test_empty_table_hash_none(self):
        # Ни одной непустой строки: bottom_row_hash None (сравнивать нечего).
        frame = np.full((100, 160, 3), 30, dtype=np.uint8)
        pos = scroll_position(frame, model=make_model(), anchor=ANCHOR,
                              run_checks=ArrowChecker())
        assert pos.bottom_row_hash is None

    def test_no_geometry_hash_none(self):
        pos = scroll_position(make_frame(), model=make_model(with_geometry=False),
                              anchor=ANCHOR, run_checks=ArrowChecker())
        assert pos.bottom_row_hash is None
        assert pos.at_start is False  # стрелки читаются и без геометрии


class TestProlog:
    def test_status_not_found_raises_no_wheels(self):
        clock = FakeClock()
        hid = ScriptedHid(clock)

        def occluded(frame):
            return WindowStatus(
                status=STATUS_OCCLUDED, reason="anchor_moved", anchor=ANCHOR,
                content_loaded=None, content_detail="")

        with pytest.raises(ScrollPreconditionError):
            run_scroll(DIR_DOWN, 1, hid, clock, ArrowChecker(), status=occluded)
        assert hid.log == []  # HID не трогается (инвариант 9)

    def test_activate_cancelled(self):
        clock = FakeClock()
        hid = ScriptedHid(clock)
        activate = lambda: ActivateResult(hwnd=0, already_active=False,
                                          attempts=1, cancelled=True)
        result = run_scroll(DIR_DOWN, 1, hid, clock, ArrowChecker(),
                            activate=activate)
        assert result.cancelled is True
        assert result.success is False
        assert result.needs_recover is False
        assert hid.log == []

    def test_cancelled_mid_scroll(self):
        clock = FakeClock()
        hid = ScriptedHid(clock)
        calls = {"n": 0}

        def cancelled():
            calls["n"] += 1
            return calls["n"] > 1  # до первого wheel не отменяем

        result = run_scroll(DIR_DOWN, 3, hid, clock, ArrowChecker(),
                            cancelled=cancelled)
        assert result.cancelled is True
        assert result.scrolled == 0

    def test_missing_geometry_is_model_error(self):
        clock = FakeClock()
        hid = ScriptedHid(clock)
        with pytest.raises(ScrollError):
            run_scroll(DIR_DOWN, 1, hid, clock, ArrowChecker(),
                       model=make_model(with_geometry=False))

    def test_invalid_args(self):
        clock = FakeClock()
        hid = ScriptedHid(clock)
        with pytest.raises(ScrollError):
            run_scroll("sideways", 1, hid, clock, ArrowChecker())
        with pytest.raises(ScrollError):
            run_scroll(DIR_DOWN, 0, hid, clock, ArrowChecker())
