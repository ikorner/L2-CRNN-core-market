"""Тесты press_key (примитив 5.9, menu-quickslot-navigation): нажатие
клавиши с verify и трёхзначным исходом.

Детерминизм (инвариант 7): ScriptedHid/FakeClock/FakeScreen из test_click,
rng — с фиксированным seed, evaluate — скриптованный. Никаких обращений
к живому экрану/HID.
"""

import random

import numpy as np
import pytest

from l2core.adapters.hid import HidError
from l2core.model import Check
from l2core.primitives.activate import ActivateResult
from l2core.primitives.click import CheckReport, VerifySpec
from l2core.primitives.key import PressKeyError, press_key
from tests.test_click import FakeClock, FakeScreen, ScriptedHid, make_config

KEY_ID = "X"
KEY_CODE = 0x1B
VERIFY = VerifySpec(
    checks=(Check(kind="template", element="menu_anchor", variant="normal"),),
    expected="menu_open",
)


class FakeEvaluate:
    """Скриптованная оценка проверок по кадру (без якоря окна)."""

    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls = 0

    def __call__(self, checks, frame):
        self.calls += 1
        passed = self._outcomes.pop(0) if self._outcomes else True
        failures = () if passed else ("fail:menu_anchor",)
        return CheckReport(passed=passed, failures=failures)


def _activate():
    return ActivateResult(hwnd=1, already_active=True, attempts=0,
                          cancelled=False)


def _run(evaluate, hid=None, clock=None, on_fail=(), cancelled=None,
         config=None, screen=None):
    clock = clock or FakeClock()
    hid = hid or ScriptedHid(clock)
    config = config or make_config(click_retry_count=1,
                                   wait_default_timeout=0.2)
    result = press_key(
        KEY_ID,
        verify=VERIFY,
        evaluate=evaluate,
        screen=screen or FakeScreen(),
        hid=hid,
        cursor_pos=lambda: (500, 300),
        activate=_activate,
        config=config,
        on_fail=on_fail,
        cancelled=cancelled,
        rng=random.Random(7),
        clock=clock.monotonic,
        sleep=clock.sleep,
    )
    return result, hid, clock


class TestPressKey:
    def test_ok_presses_x_and_verifies(self):
        evaluate = FakeEvaluate([True])
        result, hid, _ = _run(evaluate)
        presses = [a for a in hid.log if a[1] == "key_press"]
        assert [a[2] for a in presses] == [(KEY_CODE,)]
        assert result.success and result.attempts == 1
        assert not result.unconfirmed and not result.cancelled

    def test_verify_timeout_fails_with_on_fail(self):
        evaluate = FakeEvaluate([False] * 20)
        result, hid, _ = _run(evaluate, on_fail=("press_esc",))
        assert not result.success and not result.unconfirmed
        assert result.on_fail_executed
        presses = [a for a in hid.log if a[1] == "key_press"]
        codes = [a[2] for a in presses]
        # 1 + click_retry_count попыток нажатия X...
        assert codes.count((KEY_CODE,)) == 1 + 1
        # ...и on_fail press_esc ушёл в HID как ESC (0xB1).
        assert (0xB1,) in codes

    def test_hid_error_is_unknown_unconfirmed(self):
        clock = FakeClock()

        class FailingHid(ScriptedHid):
            def key_press(self, code):
                raise HidError("транспортный сбой")

        result, _, _ = _run(FakeEvaluate([True]), hid=FailingHid(clock),
                            on_fail=("press_esc",))
        assert not result.success and result.unconfirmed
        assert not result.on_fail_executed

    def test_cancelled_before_press(self):
        evaluate = FakeEvaluate([True])
        result, hid, _ = _run(evaluate, cancelled=lambda: True)
        assert result.cancelled and not result.success
        assert not [a for a in hid.log if a[1] == "key_press"]

    def test_unknown_key_fails_fast(self):
        clock = FakeClock()
        with pytest.raises(PressKeyError):
            press_key(
                "Q",
                verify=VERIFY,
                evaluate=FakeEvaluate([True]),
                screen=FakeScreen(),
                hid=ScriptedHid(clock),
                cursor_pos=lambda: (500, 300),
                activate=_activate,
                config=make_config(),
                clock=clock.monotonic,
                sleep=clock.sleep,
            )
