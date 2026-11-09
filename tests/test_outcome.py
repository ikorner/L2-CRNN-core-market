"""Тесты трёхзначного результата OK/FAILED/UNKNOWN (задача 5.7, spec
action-primitives: «Критичные действия SHALL возвращать OK/FAILED/UNKNOWN,
где UNKNOWN = эффект не подтверждён (лаг, перекрытие)»; «ядро MUST NOT
повторять потенциально необратимые действия при UNKNOWN — решение передаётся
стратегии»; safety: kill switch — отдельный домен, CANCELLED ≠ результат
действия).

Детерминизм (инвариант 7): harness'ы примитивов переиспользуются из
tests/test_click.py, test_drag.py, test_enter.py, test_scroll.py (скриптованные
экран/HID/часы, rng с фиксированным seed). Никаких обращений к живому
экрану/HID/OCR.

Семантика UNKNOWN (пины этого файла): HID-эффект действия, скорее всего,
уже применён (клик ушёл / текст набран / wheel послан / drag отпущен), но
подтвердить его невозможно — якорь потерян на свежем кадре ПОСЛЕ действия
или verify иначе не выполнялся. FAILED — verify выполнялся и провалился
(либо действие до HID не дошло): эффект отсутствует либо опровергнут.
"""
import random

import pytest

from tests import test_click, test_drag, test_enter, test_scroll
from tests.test_click import ClickResult
from tests.test_enter import EnterResult
from tests.test_scroll import ScrollResult

from l2core.model import Check
from l2core.primitives.drag import DragResult
from l2core.primitives.click import VerifySpec
from l2core.primitives.outcome import (
    ActionOutcome,
    GuardResolution,
    IrreversibleGuard,
    IrreversibleRepeatError,
    PrimitiveResult,
    to_outcome,
)

CLICK_VERIFY = VerifySpec(
    checks=(Check(kind="template", element=test_click.ELEMENT_ID,
                  variant="active"),),
    expected="target_state",
)


@pytest.fixture(autouse=True)
def vendor_state():
    """Снапшот/восстановление module-level констант вендора сегментации.

    TableSegmentation подменяет их процесс-wide (документировано): без
    восстановления синтетическая геометрия тестов протухла бы гвард-эталоны
    вендора (как в tests/test_scroll.py).
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


def retry0_config(mod):
    """Конфиг примитивного harness'а с click_retry_count=0 (одна попытка)."""
    return mod.make_config(click_retry_count=0)


def scripted_anchor(anchors):
    """find_anchor по скрипту (последний элемент — по кругу)."""
    state = {"i": 0}

    def find_anchor(frame):
        anchor = anchors[min(state["i"], len(anchors) - 1)]
        state["i"] += 1
        return anchor

    return find_anchor


class TestOutcomeMapping:
    """Маппинг результатов примитивов в ActionOutcome (адаптер to_outcome)."""

    def test_click_ok(self):
        clock = test_click.FakeClock()
        hid = test_click.ScriptedHid(clock)
        test_click.FakeCursor().track(hid)
        result = test_click.run_click(hid, clock, test_click.FakeChecker())
        assert result.success is True
        assert to_outcome(result) is ActionOutcome.OK

    def test_click_unknown_anchor_lost_after_click(self):
        # Клик ушёл в HID, но якорь потерян на кадре verify: эффект мог
        # свершиться, подтверждения нет — UNKNOWN (не FAILED).
        clock = test_click.FakeClock()
        hid = test_click.ScriptedHid(clock)
        test_click.FakeCursor().track(hid)
        find_anchor = scripted_anchor(
            [test_click.ANCHOR, test_click.ANCHOR, None])
        result = test_click.run_click(
            hid, clock, test_click.FakeChecker(), find_anchor=find_anchor,
            config=retry0_config(test_click))
        assert result.success is False
        assert result.unconfirmed is True
        assert to_outcome(result) is ActionOutcome.UNKNOWN
        assert any("якорь потерян перед verify" in f
                   for f in result.failures)

    def test_click_failed_verify_exhausted(self):
        # Verify выполнялся и провалился: эффект опровергнут — FAILED,
        # страж UNKNOWN не задействован.
        clock = test_click.FakeClock()
        hid = test_click.ScriptedHid(clock)
        test_click.FakeCursor().track(hid)
        checker = test_click.FakeChecker(verify=[False])
        result = test_click.run_click(
            hid, clock, checker,
            config=retry0_config(test_click))
        assert result.success is False
        assert result.unconfirmed is False
        assert to_outcome(result) is ActionOutcome.FAILED

    def test_click_failed_anchor_lost_before_click(self):
        # Якорь потерян ДО клика: HID-эффекта не было — FAILED, не UNKNOWN.
        clock = test_click.FakeClock()
        hid = test_click.ScriptedHid(clock)
        test_click.FakeCursor().track(hid)
        result = test_click.run_click(
            hid, clock, test_click.FakeChecker(), find_anchor=lambda f: None,
            config=retry0_config(test_click))
        assert result.success is False
        assert result.unconfirmed is False
        assert to_outcome(result) is ActionOutcome.FAILED
        assert not any(a == "click" for _, a, _ in hid.log)

    def test_drag_ok(self):
        clock = test_drag.FakeClock()
        hid = test_drag.ScriptedHid(clock)
        test_drag.FakeCursor().track(hid)
        result = test_drag.run_drag(hid, clock, test_drag.FakeChecker())
        assert result.success is True
        assert to_outcome(result) is ActionOutcome.OK

    def test_drag_unknown_anchor_lost_after_release(self):
        # Drag отпущен (release ушёл), якорь потерян перед verify — UNKNOWN.
        clock = test_drag.FakeClock()
        hid = test_drag.ScriptedHid(clock)
        test_drag.FakeCursor().track(hid)
        find_anchor = scripted_anchor([test_drag.ANCHOR, None])
        result = test_drag.run_drag(
            hid, clock, test_drag.FakeChecker(), find_anchor=find_anchor,
            config=retry0_config(test_drag))
        assert result.success is False
        assert result.unconfirmed is True
        assert to_outcome(result) is ActionOutcome.UNKNOWN
        assert any(a == "mouse_up" for _, a, _ in hid.log)

    def test_drag_failed_verify_exhausted(self):
        clock = test_drag.FakeClock()
        hid = test_drag.ScriptedHid(clock)
        test_drag.FakeCursor().track(hid)
        result = test_drag.run_drag(
            hid, clock, test_drag.FakeChecker(verify=[False]),
            config=retry0_config(test_drag))
        assert result.success is False
        assert result.unconfirmed is False
        assert to_outcome(result) is ActionOutcome.FAILED

    def test_enter_ok(self):
        clock = test_enter.FakeClock()
        hid = test_enter.ScriptedHid(clock)
        test_enter.FakeCursor().track(hid)
        result = test_enter.run_enter(hid, clock, test_enter.FakeChecker())
        assert result.success is True
        assert to_outcome(result) is ActionOutcome.OK

    def test_enter_unknown_ocr_impossible_after_typing(self):
        # Текст набран клавиатурой, но поле прочитать невозможно (якорь
        # потерян на кадре чтения): содержимое могло уйти — UNKNOWN.
        clock = test_enter.FakeClock()
        hid = test_enter.ScriptedHid(clock)
        test_enter.FakeCursor().track(hid)
        result = test_enter.run_enter(
            hid, clock, test_enter.FakeChecker(), find_anchor=lambda f: None,
            config=retry0_config(test_enter))
        assert result.success is False
        assert result.unconfirmed is True
        assert to_outcome(result) is ActionOutcome.UNKNOWN
        # Набор реально уходил в HID — эффект мог примениться.
        assert any(a == "key_press" for _, a, _ in hid.log)

    def test_enter_failed_ocr_mismatch(self):
        # OCR прочитал поле, содержимое не сошлось (recover_layout=False):
        # verify выполнялся и провалился — FAILED.
        clock = test_enter.FakeClock()
        hid = test_enter.ScriptedHid(clock)
        test_enter.FakeCursor().track(hid)
        ocr = test_enter.FakeOcr([("Xyzzy", 0.9), ("Xyzzy", 0.9)])
        result = test_enter.run_enter(
            hid, clock, test_enter.FakeChecker(), text="Sword", ocr=ocr,
            recover_layout=False, config=retry0_config(test_enter))
        assert result.success is False
        assert result.unconfirmed is False
        assert to_outcome(result) is ActionOutcome.FAILED

    def test_scroll_ok(self):
        clock = test_scroll.FakeClock()
        hid = test_scroll.ScriptedHid(clock)
        result = test_scroll.run_scroll(
            test_scroll.DIR_DOWN, 2, hid, clock, test_scroll.ArrowChecker())
        assert result.success is True
        assert to_outcome(result) is ActionOutcome.OK

    def test_scroll_unknown_anchor_lost_after_wheel(self):
        # Wheel ушёл, но на всех poll'ах verify якоря не было: смена строки
        # не подтверждена и не опровергнута — UNKNOWN.
        clock = test_scroll.FakeClock()
        hid = test_scroll.ScriptedHid(clock)
        config = retry0_config(test_scroll)
        result = run_scroll_local(
            test_scroll.DIR_DOWN, 1, hid, clock, test_scroll.ArrowChecker(),
            screen=AlternatingScreen(), config=config,
            find_anchor=scripted_anchor([test_scroll.ANCHOR,
                                         test_scroll.ANCHOR, None]))
        assert result.success is False
        assert result.unconfirmed is True
        assert to_outcome(result) is ActionOutcome.UNKNOWN
        assert any(a == "scroll" for _, a, _ in hid.log)

    def test_scroll_failed_row_not_changing(self):
        # Якорь на месте, verify выполнялся (строка не меняется, край не
        # достигнут): эффект опровергнут — FAILED, не UNKNOWN.
        clock = test_scroll.FakeClock()
        hid = test_scroll.ScriptedHid(clock)
        config = retry0_config(test_scroll)
        screen = test_scroll.ScriptedScreen([test_scroll.make_frame(200)])
        result = run_scroll_local(
            test_scroll.DIR_DOWN, 1, hid, clock, test_scroll.ArrowChecker(),
            screen=screen, config=config)
        assert result.success is False
        assert result.unconfirmed is False
        assert to_outcome(result) is ActionOutcome.FAILED


class AlternatingScreen:
    """Экран, чередующий два кадра (разные хэши нижней строки)."""

    def __init__(self):
        self._frames = [test_scroll.make_frame(200),
                        test_scroll.make_frame(220)]
        self._i = 0

    def grab(self):
        frame = self._frames[self._i % 2]
        self._i += 1
        return frame


def run_scroll_local(direction, n, hid, clock, checker, *, model=None,
                     screen=None, config=None, find_anchor=None):
    """scroll_list напрямую (harness test_scroll.run_scroll config не
    принимает; здесь нужен click_retry_count=0)."""
    from l2core.primitives.activate import ActivateResult
    from l2core.primitives.scroll import scroll_list

    return scroll_list(
        direction, n,
        model=model or test_scroll.make_model(),
        screen=screen or test_scroll.MovingScreen(),
        hid=hid,
        cursor_pos=test_scroll.FakeCursor().get,
        activate=lambda: ActivateResult(hwnd=1, already_active=True,
                                        attempts=0, cancelled=False),
        window_status=test_scroll.found_status,
        find_anchor=find_anchor or (lambda frame: test_scroll.ANCHOR),
        run_checks=checker,
        config=config or test_scroll.make_config(),
        rng=random.Random(7),
        wheel_delay_ms=0,
        clock=clock.monotonic,
        sleep=clock.sleep,
    )


class TestCancelledSeparation:
    """CANCELLED — домен kill switch (8.1), не результат действия: to_outcome
    отменённый результат не пропускает (fail fast до адаптации)."""

    @pytest.mark.parametrize("make", [
        lambda: ClickResult("tab", success=False, attempts=0, cancelled=True),
        lambda: DragResult((1, 2), (3, 4), success=False, attempts=0,
                           cancelled=True),
        lambda: EnterResult("field", success=False, attempts=0, cancelled=True),
        lambda: ScrollResult(requested=1, scrolled=0, reached_end=False,
                             position=None, cancelled=True),
    ])
    def test_cancelled_result_not_mappable(self, make):
        result = make()
        assert result.cancelled is True
        with pytest.raises(ValueError):
            to_outcome(result)


class TestPrimitiveResultProtocol:
    """Все результаты примитивов структурно удовлетворяют PrimitiveResult
    (success/cancelled/unconfirmed — контракт адаптера)."""

    def test_all_results_satisfy_protocol(self):
        assert isinstance(ClickResult("tab", success=True, attempts=1),
                          PrimitiveResult)
        assert isinstance(DragResult((1, 2), (3, 4), success=True, attempts=1),
                          PrimitiveResult)
        assert isinstance(EnterResult("field", success=True, attempts=1),
                          PrimitiveResult)
        assert isinstance(ScrollResult(requested=1, scrolled=1,
                                       reached_end=False, position=None),
                          PrimitiveResult)


class TestIrreversibleGuard:
    """Страж необратимых действий: UNKNOWN блокирует повтор до явного снятия
    стратегией; OK/FAILED страж не задействуют (спека: «решение передаётся
    стратегии» — повтор обратимых действий при FAILED допустим на уровне ядра)."""

    def test_unknown_blocks_repeat_with_diagnostics(self):
        guard = IrreversibleGuard()
        guard.after("start_sell", ActionOutcome.UNKNOWN,
                    detail="якорь потерян перед verify")
        assert guard.is_blocked("start_sell") is True
        with pytest.raises(IrreversibleRepeatError) as excinfo:
            guard.before("start_sell")
        message = str(excinfo.value)
        assert "start_sell" in message
        assert "якорь потерян перед verify" in message
        # Другие действия страж не трогает.
        guard.before("setup_sell")

    def test_confirm_after_unknown_unblocks(self):
        guard = IrreversibleGuard()
        guard.after("start_sell", ActionOutcome.UNKNOWN, detail="лаг окна")
        resolution = guard.confirm_after_unknown(
            "start_sell", detail="вручную подтверждено: лавка создана")
        assert isinstance(resolution, GuardResolution)
        assert resolution.action_id == "start_sell"
        assert guard.is_blocked("start_sell") is False
        guard.before("start_sell")  # повтор возможен после разборок
        assert guard.resolutions == (resolution,)

    def test_confirm_without_block_fails_fast(self):
        guard = IrreversibleGuard()
        with pytest.raises(ValueError):
            guard.confirm_after_unknown("start_sell")

    def test_ok_does_not_block(self):
        guard = IrreversibleGuard()
        guard.after("start_sell", ActionOutcome.OK)
        guard.before("start_sell")
        assert guard.is_blocked("start_sell") is False
        assert guard.resolutions == ()

    def test_failed_does_not_block(self):
        guard = IrreversibleGuard()
        guard.after("start_sell", ActionOutcome.FAILED)
        guard.before("start_sell")
        assert guard.is_blocked("start_sell") is False

    def test_after_rejects_non_outcome(self):
        # Результат примитива (в т.ч. отменённый — домен kill switch) сюда
        # недопустим: стратегия обязана отделить отмену и адаптировать через
        # to_outcome. Fail fast вместо молчаливого проглатывания.
        guard = IrreversibleGuard()
        with pytest.raises(TypeError):
            guard.after("start_sell",
                        ClickResult("tab", success=False, attempts=0,
                                    cancelled=True))
        with pytest.raises(TypeError):
            guard.after("start_sell", "UNKNOWN")
        assert guard.is_blocked("start_sell") is False

    def test_unknown_after_failed_reblocks(self):
        guard = IrreversibleGuard()
        guard.after("start_sell", ActionOutcome.FAILED)
        guard.after("start_sell", ActionOutcome.UNKNOWN, detail="повторный лаг")
        with pytest.raises(IrreversibleRepeatError):
            guard.before("start_sell")

    def test_wrap_blocks_second_call(self):
        guard = IrreversibleGuard()
        outcomes = iter([ActionOutcome.UNKNOWN, ActionOutcome.OK])
        calls = []

        def action():
            calls.append(1)
            return next(outcomes)

        wrapped = guard.wrap("start_sell", action)
        assert wrapped() is ActionOutcome.UNKNOWN
        with pytest.raises(IrreversibleRepeatError):
            wrapped()
        assert len(calls) == 1  # повторный вызов дошёл до стража, не до fn
        guard.confirm_after_unknown("start_sell")
        assert wrapped() is ActionOutcome.OK
        assert len(calls) == 2

    def test_wrap_ok_never_blocks(self):
        guard = IrreversibleGuard()
        wrapped = guard.wrap("scroll", lambda: ActionOutcome.OK)
        assert wrapped() is ActionOutcome.OK
        assert wrapped() is ActionOutcome.OK
        assert guard.is_blocked("scroll") is False
