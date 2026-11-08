"""Тесты wait_until (задача 5.1, spec action-primitives: «Ожидание только
по предикату, без sleep»).

Детерминизм (инвариант 7): часы и сон — инъекции (clock/sleep), тесты на
фейковом времени без реального ожидания. Конфиг — обычный CoreConfig()
(дефолты timing: wait_default_timeout=10.0, verify_interval=0.2).
"""
import pytest

from l2core.config import CoreConfig
from l2core.primitives.wait import WaitResult, wait_until


class FakeClock:
    """Фейковые часы: sleep продвигает время, записывая длительности сна."""

    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def run(predicate, *, timeout=1.0, interval=0.2, cancelled=None, config=None):
    clock = FakeClock()
    result = wait_until(
        predicate,
        timeout=timeout,
        interval=interval,
        config=config,
        cancelled=cancelled,
        clock=clock.monotonic,
        sleep=clock.sleep,
    )
    return result, clock


def test_success_first_poll_no_sleep():
    result, clock = run(lambda: True)
    assert isinstance(result, WaitResult)
    assert result.success is True
    assert result.cancelled is False
    assert result.polls == 1
    assert result.elapsed == 0.0
    assert result.last_value is True
    assert clock.sleeps == []


def test_success_after_n_polls():
    values = iter([False, False, "готово"])
    result, clock = run(lambda: next(values), timeout=5.0, interval=0.5)
    assert result.success is True
    assert result.polls == 3
    assert result.last_value == "готово"  # truthy не-bool тоже успех
    assert result.elapsed == 1.0
    assert clock.sleeps == [0.5, 0.5]


def test_timeout_polls_until_deadline():
    # timeout=1.0, interval=0.2: poll в 0, 0.2, ..., 1.0 — 6 poll'ов,
    # не дольше дедлайна, суммарный сон == timeout.
    result, clock = run(lambda: False, timeout=1.0, interval=0.2)
    assert result.success is False
    assert result.cancelled is False
    assert result.polls == 6
    assert result.elapsed == 1.0
    assert result.last_value is False
    assert clock.sleeps == [0.2] * 5
    assert sum(clock.sleeps) == 1.0


def test_last_sleep_capped_by_deadline():
    # interval=0.3, timeout=1.0: poll в 0, 0.3, 0.6, 0.9; остаток до
    # дедлайна 0.1 — финальный poll ровно на дедлайне (не позже, не раньше).
    result, clock = run(lambda: False, timeout=1.0, interval=0.3)
    assert result.success is False
    assert result.polls == 5
    assert result.elapsed == 1.0
    assert clock.sleeps == [0.3, 0.3, 0.3, 0.1]
    assert max(clock.sleeps) <= 0.3


def test_cancel_stops_polling():
    calls = []

    def predicate():
        calls.append(1)
        return False

    fired = {"n": 0}

    def cancelled():
        fired["n"] += 1
        return len(calls) >= 2  # отмена срабатывает перед третьим poll'ом

    result, clock = run(predicate, timeout=10.0, interval=0.2, cancelled=cancelled)
    assert result.success is False
    assert result.cancelled is True
    assert result.polls == 2
    assert len(calls) == 2  # третий poll не выполнялся
    assert clock.sleeps == [0.2]  # уснуть после отмены не успели


def test_cancel_before_first_poll():
    result, clock = run(lambda: False, timeout=10.0, cancelled=lambda: True)
    assert result.cancelled is True
    assert result.success is False
    assert result.polls == 0
    assert clock.sleeps == []


def test_defaults_from_config():
    # Без явных timeout/interval — значения берутся из config.timing.
    config = CoreConfig()  # wait_default_timeout=10.0, verify_interval=0.2
    result, clock = run(lambda: False, timeout=None, interval=None, config=config)
    assert result.success is False
    assert result.elapsed == 10.0
    assert result.polls == 51  # 0, 0.2, ..., 10.0
    assert clock.sleeps == [0.2] * 50


def test_explicit_values_override_config():
    config = CoreConfig()
    result, clock = run(
        lambda: False, timeout=1.0, interval=None, config=config
    )
    assert result.elapsed == 1.0
    assert clock.sleeps == [0.2] * 5  # interval из конфига, timeout явный


def test_missing_timeout_and_config_is_error():
    clock = FakeClock()
    with pytest.raises(ValueError, match="timeout"):
        wait_until(
            lambda: False, interval=0.2,
            clock=clock.monotonic, sleep=clock.sleep,
        )


def test_invalid_time_args_rejected():
    clock = FakeClock()
    for bad in (0.0, -1.0, True, "1"):
        with pytest.raises(ValueError, match="timeout"):
            wait_until(
                lambda: False, timeout=bad, interval=0.2,
                clock=clock.monotonic, sleep=clock.sleep,
            )
        with pytest.raises(ValueError, match="interval"):
            wait_until(
                lambda: False, timeout=1.0, interval=bad,
                clock=clock.monotonic, sleep=clock.sleep,
            )


def test_no_real_time_injected_defaults_are_callables():
    # Дефолтные clock/sleep — time.monotonic/time.sleep (процессные),
    # здесь только проверка типа подписи, без реального ожидания.
    import inspect

    import time

    sig = inspect.signature(wait_until)
    assert sig.parameters["clock"].default is time.monotonic
    assert sig.parameters["sleep"].default is time.sleep
