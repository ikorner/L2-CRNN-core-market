"""Тесты enter_text/enter_price (задача 5.4, spec action-primitives: «Ввод
цены и текста с выбором способа по профилю», «Восстановление раскладки при
вводе текста»; инвариант 9: слепых действий нет).

Детерминизм (инвариант 7): ScriptedHid фиксирует действия с метками фейкового
времени, FakeOcr — скриптованные ответы поля ввода, rng — random.Random с
фиксированным seed, часы — FakeClock (sleep продвигает время). Никаких
обращений к живому экрану/HID/OCR.
"""
import random

import numpy as np
import pytest

from l2core.adapters.hid import (
    KEY_BACKSPACE,
    KEY_ENTER,
    KEY_LEFT_SHIFT,
)
from l2core.config import (
    BehaviorConfig,
    CoreConfig,
    ResolutionConfig,
    TimingConfig,
)
from l2core.detector.anchor import Anchor
from l2core.detector.status import (
    STATUS_FOUND,
    STATUS_NOT_FOUND,
    WindowStatus,
)
from l2core.model import (
    ElementSpec,
    GrammarSpec,
    OcrRegionSpec,
    Rect,
    UiModel,
    VariantSpec,
    WindowSpec,
)
from l2core.primitives.activate import ActivateGameError, ActivateResult
from l2core.primitives.enter import (
    KEY_LEFT_CTRL,
    EnterPreconditionError,
    EnterResult,
    EnterTextError,
    enter_price,
    enter_text,
)

ANCHOR = Anchor(window_id="market", x=202, y=103, score=0.99, method="template")
FIELD_RECT = (10, 20, 100, 30)     # центр -> (60, 15) от якоря
SEARCH_RECT = (200, 20, 60, 20)    # центр -> (230, 10) от якоря
KEY_RECTS = {                      # центры клавиш экранного нумпада
    "1": (0, 60, 20, 20),          # -> (212, 173) от абсолюта
    "2": (30, 60, 20, 20),         # -> (242, 173)
    "3": (60, 60, 20, 20),         # -> (272, 173)
}
REGION_RECT = (5, 5, 90, 20)
ON_FAIL = ("press_esc", "click_free_area")

# HID keycodes (usage table): цифры '1'..'3' -> 0x1E..0x20, буквы a-z -> 0x04..
HID_DIGITS = {"1": 0x1E, "2": 0x1F, "3": 0x20}


def hid_code(ch: str) -> int:
    """Базовый HID keycode символа (для заглавных печать оборачивается в
    SHIFT: key_down/key_press/key_up — см. expected_key_events)."""
    if ch.isdigit():
        return 0x1E + ((ord(ch) - ord("0") - 1) % 10)
    return 0x04 + (ord(ch.lower()) - ord("a"))


def expected_key_events(text, final=()):
    """Эталонная последовательность клавиатурных HID-событий для текста:
    заглавные буквы — с SHIFT-модификатором (key_down/press/up)."""
    events = []
    for ch in text:
        if ch.isalpha() and ch.isupper():
            events += [("key_down", KEY_LEFT_SHIFT),
                       ("key_press", hid_code(ch)),
                       ("key_up", KEY_LEFT_SHIFT)]
        else:
            events.append(("key_press", hid_code(ch)))
    return events + list(final)


class FakeClock:
    """Фейковые часы: sleep продвигает время."""

    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeScreen:
    """Экран со скриптованной последовательностью кадров (последний — по кругу)."""

    def __init__(self, frames=None):
        self._frames = list(frames or [np.zeros((1080, 1920, 3), dtype=np.uint8)])
        self._index = 0

    def grab(self):
        frame = self._frames[min(self._index, len(self._frames) - 1)]
        if self._index < len(self._frames) - 1:
            self._index += 1
        return frame


class ScriptedHid:
    """HID с записью действий: (время, действие, args)."""

    def __init__(self, clock):
        self.clock = clock
        self.log = []

    def move_rel(self, dx, dy):
        self.log.append((self.clock.now, "move_rel", (dx, dy)))

    def click(self, button=1, count=1, hold_ms=None):
        self.log.append((self.clock.now, "click", (button, count, hold_ms)))

    def key_press(self, code):
        self.log.append((self.clock.now, "key_press", (code,)))

    def key_down(self, code):
        self.log.append((self.clock.now, "key_down", (code,)))

    def key_up(self, code):
        self.log.append((self.clock.now, "key_up", (code,)))


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


class FakeChecker:
    """Скриптованная оценка проверок присутствия (template, вариант normal)."""

    def __init__(self, presence=None):
        self._presence = list(presence or [])
        self.calls = []

    def __call__(self, checks, frame, anchor):
        self.calls.append(tuple(checks))
        from l2core.primitives.click import CheckReport

        if self._presence:
            passed = self._presence.pop(0)
        else:
            passed = True
        failures = () if passed else tuple(f"fail:{c.element}" for c in checks)
        return CheckReport(passed=passed, failures=failures)


class FakeOcr:
    """Фейк OcrClient: очередь ответов (text, confidence)."""

    def __init__(self, answers):
        self._answers = list(answers)

    def read(self, region_img):
        if not self._answers:
            raise AssertionError("FakeOcr: ответы исчерпаны")
        return self._answers.pop(0)


def _element(eid, rect):
    return ElementSpec(
        id=eid,
        role="click_target",
        rect=Rect(*rect),
        variants={"normal": VariantSpec(name="normal", template=f"test/{eid}.png")},
    )


def make_model(grammar_kind="text"):
    elements = {
        "field": _element("field", FIELD_RECT),
        "search_button": _element("search_button", SEARCH_RECT),
    }
    for digit, rect in KEY_RECTS.items():
        elements[f"key_{digit}"] = _element(f"key_{digit}", rect)
    grammar_kwargs = {}
    if grammar_kind == "price":
        grammar_kwargs = dict(separators=frozenset({",", "."}),
                              suffixes=(("к", 1000), ("кк", 1000000)))
    region = OcrRegionSpec(
        id="field",
        rect=Rect(*REGION_RECT),
        grammar=GrammarSpec(name=grammar_kind, kind=grammar_kind, **grammar_kwargs),
    )
    return UiModel(
        path="test",
        window=WindowSpec(id="market", title="Обзор лавок", size=(1008, 674)),
        elements=elements,
        states={},
        transitions=(),
        ocr_regions={"field": region},
        table_geometry=None,
    )


def make_config(**timing_kwargs):
    timing = TimingConfig(verify_interval=0.05, hover_pause=0.2, **timing_kwargs)
    behavior = BehaviorConfig(
        numpad_fraction=0.0,
        enter_fraction=1.0,
        click_jitter_px=0,          # джиттер выключен — точки клика строго от rect
        press_delay_min_ms=40,
        press_delay_max_ms=40,      # фикс — hold не расходует rng
        key_delay_min_ms=50,
        key_delay_max_ms=50,        # фикс — пауза между клавишами строго 50 мс
    )
    return CoreConfig(
        timing=timing,
        behavior=behavior,
        resolution=ResolutionConfig(width=1920, height=1080),
    )


def found_status(frame):
    return WindowStatus(
        status=STATUS_FOUND, reason=None, anchor=ANCHOR,
        content_loaded=None, content_detail="",
    )


def ok_activate():
    return ActivateResult(hwnd=1, already_active=True, attempts=0, cancelled=False)


def run_enter(hid, clock, checker, text="Sword", model=None, ocr=None, activate=None,
              status=None, find_anchor=None, on_fail=(), rng_seed=1, config=None,
              **kwargs):
    model = model or make_model()
    ocr = ocr or FakeOcr([(text, 0.9), (text, 0.9)])
    region_id = kwargs.pop("region_id", "field")
    return enter_text(
        text,
        model=model,
        region_id=region_id,
        ocr=ocr,
        screen=kwargs.pop("screen", FakeScreen()),
        hid=hid,
        cursor_pos=kwargs.pop("cursor_pos", FakeCursor().get),
        activate=activate or ok_activate,
        window_status=status or found_status,
        find_anchor=find_anchor or (lambda frame: ANCHOR),
        run_checks=checker,
        config=config or make_config(),
        rng=random.Random(rng_seed),
        clock=clock.monotonic,
        sleep=clock.sleep,
        on_fail=on_fail,
        **kwargs,
    )


def actions(hid):
    return [a for _, a, _ in hid.log]


class TestMethodChoice:
    def test_keyboard_input_verify_ok(self):
        # Клавиатурный ввод: фокус-клик -> печать с паузами -> Enter;
        # OCR поля подтвердил текст — успех с первой попытки.
        clock = FakeClock()
        hid = ScriptedHid(clock)
        FakeCursor().track(hid)
        result = run_enter(hid, clock, FakeChecker(), text="Sword")

        assert isinstance(result, EnterResult)
        assert result.success is True
        assert result.attempts == 1
        assert result.input_method == "keyboard"
        assert result.submit_method == "enter"
        assert result.layout_recovered is False
        assert result.needs_recover is False

        key_events = [(a, args[0]) for _, a, args in hid.log
                      if a in ("key_press", "key_down", "key_up")]
        # 'S' печатается с SHIFT (key_down/press/up), далее строчные;
        # последний key_press — Enter.
        assert key_events == expected_key_events("Sword", final=[("key_press", KEY_ENTER)])
        # Пауза между клавишами (key_delay из профиля) — по фейковым часам:
        # время каждой следующей клавиши не раньше key_delay после предыдущей
        # (события SHIFT-модификатора 'S' — в один момент).
        times = [t for t, a, _ in hid.log if a == "key_press"]
        for prev, nxt in zip(times, times[1:]):
            assert nxt - prev >= 0.05 - 1e-9  # float-дрейф фейковых часов

    def test_uppercase_uses_shift_modifier(self):
        # Ревью-фикс: заглавная 'S' печатается как key_down(SHIFT) +
        # key_press('s') + key_up(SHIFT) — иначе поле получило бы «sword»
        # и verify гарантированно не сошёлся бы.
        clock = FakeClock()
        hid = ScriptedHid(clock)
        FakeCursor().track(hid)
        run_enter(hid, clock, FakeChecker(), text="Sword")
        mods = [(a, args[0]) for _, a, args in hid.log if a in ("key_down", "key_up")]
        assert mods == [("key_down", KEY_LEFT_SHIFT), ("key_up", KEY_LEFT_SHIFT)]

    def test_numpad_input_clicks_key_elements(self):
        # Экранный нумпад: каждый символ — клик по элементу клавиши модели.
        clock = FakeClock()
        hid = ScriptedHid(clock)
        cursor = FakeCursor()
        cursor.track(hid)
        config = make_config()
        config = CoreConfig(
            timing=config.timing,
            behavior=BehaviorConfig(
                numpad_fraction=1.0,  # всегда экранный нумпад
                enter_fraction=1.0,
                click_jitter_px=0,
                press_delay_min_ms=40,
                press_delay_max_ms=40,
                key_delay_min_ms=50,
                key_delay_max_ms=50,
            ),
            resolution=config.resolution,
        )
        result = run_enter(
            hid, clock, FakeChecker(), text="123",
            numpad_map={"1": "key_1", "2": "key_2", "3": "key_3"},
            field_element_id="field", config=config, cursor_pos=cursor.get,
        )

        assert result.success is True
        assert result.input_method == "numpad"
        # Клики: фокус (field) + 3 клавиши; submit — Enter (enter_fraction=1.0).
        clicks = [t for t, a, _ in hid.log if a == "click"]
        assert len(clicks) == 4
        # Курсор побывал в центре каждой клавиши (rect'ы от якоря, джиттер 0).
        for digit in "123":
            x, y, w, h = KEY_RECTS[digit]
            point = (ANCHOR.x + x + w // 2, ANCHOR.y + y + h // 2)
            assert cursor_visited(hid, point)

    def test_choice_by_profile_seed_deterministic(self):
        # Способ ввода — по numpad_fraction профиля; seed фиксирует ветку:
        # random.Random(1).random()=0.134 < 0.5 -> нумпад, seed 0 -> клавиатура.
        config = make_config()
        config = CoreConfig(
            timing=config.timing,
            behavior=BehaviorConfig(
                numpad_fraction=0.5,
                enter_fraction=1.0,
                click_jitter_px=0,
                press_delay_min_ms=40,
                press_delay_max_ms=40,
                key_delay_min_ms=50,
                key_delay_max_ms=50,
            ),
            resolution=config.resolution,
        )
        clock = FakeClock()
        hid_np = ScriptedHid(clock)
        FakeCursor().track(hid_np)
        res_np = run_enter(
            hid_np, clock, FakeChecker(), text="1",
            numpad_map={"1": "key_1"}, rng_seed=1, config=config,
        )
        assert res_np.input_method == "numpad"

        clock = FakeClock()
        hid_kb = ScriptedHid(clock)
        FakeCursor().track(hid_kb)
        res_kb = run_enter(
            hid_kb, clock, FakeChecker(), text="1",
            numpad_map={"1": "key_1"}, rng_seed=0, config=config,
        )
        assert res_kb.input_method == "keyboard"
        # Фокус не задан: клавиатурный путь — без кликов, нумпад — клик по
        # единственной клавише.
        assert [a for _, a, _ in hid_kb.log].count("click") == 0
        assert [a for _, a, _ in hid_np.log].count("click") == 1


def cursor_visited(hid, point):
    """Был ли курсор наведён на абсолютную точку (move_rel к ней из позиции)."""
    pos = [500, 300]
    for _, action, args in hid.log:
        if action == "move_rel":
            pos[0] += args[0]
            pos[1] += args[1]
        elif action == "click":
            if tuple(pos) == point:
                return True
    return False


class TestLayoutRecovery:
    def test_translit_backspace_switch_retype(self):
        # OCR после набора показывает транслит: backspace x N, switch_layout,
        # повтор ввода, повторная OCR-проверка — успех, layout_recovered.
        clock = FakeClock()
        hid = ScriptedHid(clock)
        FakeCursor().track(hid)
        ocr = FakeOcr([("Ckjdt", 0.9), ("Ckjdt", 0.9),
                       ("Sword", 0.9), ("Sword", 0.9)])
        result = run_enter(hid, clock, FakeChecker(), text="Sword", ocr=ocr)

        assert result.success is True
        assert result.layout_recovered is True
        assert result.attempts == 1

        presses = [(t, args[0]) for t, a, args in hid.log if a == "key_press"]
        codes = [code for _, code in presses]
        # Транслит набран, затем 5 backspace, switch (down SHIFT up вокруг
        # key_press? нет: key_down(CTRL) -> key_press(SHIFT) -> key_up(CTRL)),
        # затем повторный набор и Enter.
        assert codes.count(KEY_BACKSPACE) == 5
        assert codes.count(hid_code("S")) == 2  # набор выполнен дважды
        assert codes[-1] == KEY_ENTER
        # Механика switch_layout зафиксирована в HID-логе: key_down(CTRL)
        # -> key_press(SHIFT) -> key_up(CTRL); SHIFT также держался при
        # печати заглавной 'S' в каждом из двух наборов.
        downs = [args[0] for _, a, args in hid.log if a == "key_down"]
        ups = [args[0] for _, a, args in hid.log if a == "key_up"]
        assert downs == [KEY_LEFT_SHIFT, KEY_LEFT_CTRL, KEY_LEFT_SHIFT]
        assert ups == [KEY_LEFT_SHIFT, KEY_LEFT_CTRL, KEY_LEFT_SHIFT]

    def test_recovery_reverify_fails_attempt(self):
        # Повторная проверка после восстановления тоже не сошлась — попытка
        # потрачена; retry из конфига; дальше — on_fail/сигнал recover.
        clock = FakeClock()
        hid = ScriptedHid(clock)
        FakeCursor().track(hid)
        answers = [("Ckjdt", 0.9), ("Ckjdt", 0.9), ("Ckjdt", 0.9), ("Ckjdt", 0.9)]
        ocr = FakeOcr(answers)
        result = run_enter(
            hid, clock, FakeChecker(), text="Sword", ocr=ocr,
            config=make_config(click_retry_count=0), on_fail=ON_FAIL,
        )
        assert result.success is False
        assert result.attempts == 1
        assert result.layout_recovered is False
        assert result.on_fail_executed is True
        assert result.needs_recover is True


class TestEraseObserved:
    def test_recovery_erases_observed_not_typed(self):
        # Ревью-фикс: стирание — по НАБЛЮДАЕМОМУ OCR-содержимому, а не по
        # длине введённого текста: OCR увидел 4 символа из 5 набранных ->
        # ровно 4 backspace перед повторным вводом.
        clock = FakeClock()
        hid = ScriptedHid(clock)
        FakeCursor().track(hid)
        ocr = FakeOcr([("Ckjd", 0.9), ("Ckjd", 0.9),
                       ("Sword", 0.9), ("Sword", 0.9)])
        result = run_enter(hid, clock, FakeChecker(), text="Sword", ocr=ocr)

        assert result.success is True
        assert result.layout_recovered is True
        presses = [args[0] for _, a, args in hid.log if a == "key_press"]
        assert presses.count(KEY_BACKSPACE) == 4  # len("Ckjd"), не len(text)

    def test_retry_erases_observed_content(self):
        # Ревью-фикс: повторная попытка стирает наблюдаемое содержимое
        # поля (попытка 1 оставила «Swor») перед повторным набором —
        # иначе попытка 2 печатала бы поверх и verify не сошёлся бы.
        clock = FakeClock()
        hid = ScriptedHid(clock)
        FakeCursor().track(hid)
        ocr = FakeOcr([("Swor", 0.9), ("Swor", 0.9),
                       ("Sword", 0.9), ("Sword", 0.9)])
        result = run_enter(
            hid, clock, FakeChecker(), text="Sword", ocr=ocr,
            recover_layout=False,
            config=make_config(click_retry_count=1),
        )

        assert result.success is True
        assert result.attempts == 2
        key_events = [(a, args[0]) for _, a, args in hid.log
                      if a in ("key_press", "key_down", "key_up")]
        # Первый набор «Sword» (без backspace), затем блок из 4 backspace
        # (стёрто наблюдаемое «Swor»), затем повторный набор «Sword».
        first = expected_key_events("Sword")
        assert key_events[:len(first)] == first
        erase = [("key_press", KEY_BACKSPACE)] * 4
        assert key_events[len(first):len(first) + 4] == erase
        second = expected_key_events("Sword", final=[("key_press", KEY_ENTER)])
        assert key_events[len(first) + 4:] == second


class TestSubmitChoice:
    def test_enter_key_by_profile(self):
        clock = FakeClock()
        hid = ScriptedHid(clock)
        FakeCursor().track(hid)
        result = run_enter(hid, clock, FakeChecker(), search_button_id="search_button")
        assert result.submit_method == "enter"
        presses = [args[0] for _, a, args in hid.log if a == "key_press"]
        assert presses[-1] == KEY_ENTER

    def test_search_button_by_profile(self):
        clock = FakeClock()
        hid = ScriptedHid(clock)
        cursor = FakeCursor()
        cursor.track(hid)
        config = make_config()
        config = CoreConfig(
            timing=config.timing,
            behavior=BehaviorConfig(
                numpad_fraction=0.0,
                enter_fraction=0.0,  # всегда кнопка «Поиск»
                click_jitter_px=0,
                press_delay_min_ms=40,
                press_delay_max_ms=40,
                key_delay_min_ms=50,
                key_delay_max_ms=50,
            ),
            resolution=config.resolution,
        )
        result = run_enter(
            hid, clock, FakeChecker(), search_button_id="search_button",
            config=config,
        )
        assert result.submit_method == "button"
        assert result.success is True
        # Клик по кнопке «Поиск» (фокус + кнопка), Enter не нажимался.
        x, y, w, h = SEARCH_RECT
        point = (ANCHOR.x + x + w // 2, ANCHOR.y + y + h // 2)
        assert cursor_visited(hid, point)
        assert KEY_ENTER not in [args[0] for _, a, args in hid.log if a == "key_press"]

    def test_submit_disabled(self):
        clock = FakeClock()
        hid = ScriptedHid(clock)
        FakeCursor().track(hid)
        result = run_enter(hid, clock, FakeChecker(), submit=False)
        assert result.success is True
        assert result.submit_method is None
        assert KEY_ENTER not in [args[0] for _, a, args in hid.log if a == "key_press"]


class TestEnterPrice:
    def test_price_verified_by_parsed_value(self):
        # enter_price: OCR может вернуть отформатированную цену («1,500,000»);
        # verify сравнивает разобранное по грамматике значение, не сырой текст.
        clock = FakeClock()
        hid = ScriptedHid(clock)
        FakeCursor().track(hid)
        model = make_model(grammar_kind="price")
        ocr = FakeOcr([("1,500,000", 0.9), ("1,500,000", 0.9)])
        result = enter_price(
            1500000,
            model=model,
            region_id="field",
            ocr=ocr,
            screen=FakeScreen(),
            hid=hid,
            cursor_pos=FakeCursor().get,
            activate=ok_activate,
            window_status=found_status,
            find_anchor=lambda frame: ANCHOR,
            run_checks=FakeChecker(),
            config=make_config(),
            rng=random.Random(1),
            clock=clock.monotonic,
            sleep=clock.sleep,
        )
        assert result.success is True
        # В поле введены голые цифры цены: '1'->0x1E, '5'->0x22, '0'->0x27.
        presses = [args[0] for _, a, args in hid.log if a == "key_press"]
        assert presses == [0x1E, 0x22, 0x27, 0x27, 0x27, 0x27, 0x27, KEY_ENTER]


class TestRetryAndEscalation:
    def test_retry_exhausted_runs_on_fail_then_recover_signal(self):
        # OCR поля всегда чужой текст, восстановление выключено: попытки
        # исчерпаны -> on_fail[] исполнен -> сигнал recover.
        clock = FakeClock()
        hid = ScriptedHid(clock)
        cursor = FakeCursor()
        cursor.track(hid)
        answers = [("Wrong", 0.9)] * 4  # 2 попытки x 2 чтения стабилизатора
        result = run_enter(
            hid, clock, FakeChecker(), text="Sword",
            ocr=FakeOcr(answers), recover_layout=False,
            config=make_config(click_retry_count=1), on_fail=ON_FAIL,
            free_area_point=(100, 700), cursor_pos=cursor.get,
        )
        assert result.success is False
        assert result.attempts == 2
        assert result.on_fail_executed is True
        assert result.needs_recover is True
        assert result.failures
        acts = actions(hid)
        assert acts[-3:] == ["key_press", "move_rel", "click"]
        assert hid.log[-3][2] == (0xB1,)  # KEY_ESC
        assert cursor.pos == (100, 700)

    def test_cancelled_flag_result(self):
        clock = FakeClock()
        hid = ScriptedHid(clock)
        result = run_enter(hid, clock, FakeChecker(), cancelled=lambda: True)
        assert result.cancelled is True
        assert result.success is False
        assert result.attempts == 0
        assert result.needs_recover is False
        assert hid.log == []


class TestPrologue:
    def test_status_not_found_blocks_input(self):
        clock = FakeClock()
        hid = ScriptedHid(clock)

        def not_found(frame):
            return WindowStatus(
                status=STATUS_NOT_FOUND, reason="window_not_found", anchor=None,
                content_loaded=None, content_detail="",
            )

        with pytest.raises(EnterPreconditionError) as exc_info:
            run_enter(hid, clock, FakeChecker(), status=not_found)
        assert "window_not_found" in str(exc_info.value)
        assert hid.log == []

    def test_activate_failure_escalates_no_hid(self):
        clock = FakeClock()
        hid = ScriptedHid(clock)

        def failing_activate():
            raise ActivateGameError(reason="focus_not_acquired", detail="нет фокуса")

        with pytest.raises(ActivateGameError):
            run_enter(hid, clock, FakeChecker(), activate=failing_activate)
        assert hid.log == []


class TestModelBoundaries:
    def test_unknown_region_fails_fast(self):
        clock = FakeClock()
        hid = ScriptedHid(clock)
        with pytest.raises(EnterTextError):
            run_enter(hid, clock, FakeChecker(), region_id="nope")
        assert hid.log == []

    def test_numpad_unmapped_char_fails_fast(self):
        clock = FakeClock()
        hid = ScriptedHid(clock)
        with pytest.raises(EnterTextError):
            run_enter(hid, clock, FakeChecker(), text="12,3",
                      numpad_map={"1": "key_1", "2": "key_2", "3": "key_3"})
        assert hid.log == []

    def test_non_ascii_char_fails_fast(self):
        # Кириллица вне 1-байтного протокола HID (type_char — latin-1):
        # честная граница, вопрос владельцу (OWNER_QUESTIONS Q25).
        clock = FakeClock()
        hid = ScriptedHid(clock)
        with pytest.raises(EnterTextError):
            run_enter(hid, clock, FakeChecker(), text="Меч")
        assert hid.log == []

    def test_unknown_field_element_fails_fast(self):
        clock = FakeClock()
        hid = ScriptedHid(clock)
        with pytest.raises(EnterTextError):
            run_enter(hid, clock, FakeChecker(), field_element_id="nope")
        assert hid.log == []

    def test_rng_seed_deterministic(self):
        logs = []
        for _ in range(2):
            clock = FakeClock()
            hid = ScriptedHid(clock)
            FakeCursor().track(hid)
            run_enter(
                hid, clock, FakeChecker(), text="123",
                numpad_map={"1": "key_1", "2": "key_2", "3": "key_3"},
                rng_seed=1,
            )
            logs.append(hid.log)
        assert logs[0] == logs[1]
