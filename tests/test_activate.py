"""Тесты activate_game (задача 5.2, spec core-api-events: «Фокус и активность
окна перед каждым действием», сценарий «Окно не активно»; safety: осмысленная
ошибка/эскалация вместо слепых кликов).

Детерминизм (инвариант 7): fake-шлюз Win32 (dict'ы + сценарий фокуса),
фейковые часы (sleep продвигает время, реального ожидания нет). Никаких
обращений к реальным окнам.
"""

import pytest

from l2core.adapters.win32 import REASON_MINIMIZED, REASON_NOT_FOUND
from l2core.config import CoreConfig, TimingConfig, WindowConfig
from l2core.primitives.activate import (
    REASON_FOCUS_NOT_ACQUIRED,
    ActivateGameError,
    activate_game,
)


class FakeClock:
    """Фейковые часы: sleep продвигает время."""

    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeWin32Gateway:
    """Полный протокол Win32Gateway: окна (dict) + сценарий фокуса.

    windows[hwnd] = {title, class, iconic, client}; foreground — hwnd окна
    в фокусе; focus_results — сценарий срабатывания SetForeground по вызовам
    (None — срабатывает всегда).
    """

    def __init__(self, windows=None, foreground=None, focus_results=None):
        self.windows = dict(windows or {})
        self.foreground = foreground
        self.focus_results = focus_results
        self.set_calls = []
        self._set_index = 0

    def find_windows(self, title=None, class_name=None):
        result = []
        for hwnd, w in self.windows.items():
            if title is not None and w["title"] != title:
                continue
            if class_name is not None and w["class"] != class_name:
                continue
            result.append(hwnd)
        return result

    def get_client_rect(self, hwnd):
        return self.windows[hwnd]["client"]

    def is_iconic(self, hwnd):
        return self.windows[hwnd]["iconic"]

    def set_process_dpi_aware(self):
        return True

    def get_foreground_window(self):
        return self.foreground

    def set_foreground(self, hwnd):
        self.set_calls.append(hwnd)
        if self.focus_results is None:
            self.foreground = hwnd
        else:
            if self._set_index < len(self.focus_results):
                if self.focus_results[self._set_index]:
                    self.foreground = hwnd
            self._set_index += 1
        return True


def make_window(title="Lineage II", cls="L2Client", iconic=False, client=(10, 20, 1920, 1080)):
    return {"title": title, "class": cls, "iconic": iconic, "client": client}


def make_config(**timing_kwargs):
    timing = TimingConfig(**timing_kwargs)
    return CoreConfig(timing=timing, window=WindowConfig(title="Lineage II"))


class TestActivateGame:
    def test_already_active_window_no_set_foreground(self):
        # Фокус уже у окна игры: SetForeground не вызывается, слепой
        # активации нет.
        fake = FakeWin32Gateway({100: make_window()}, foreground=100)
        clock = FakeClock()
        result = activate_game(fake, config=make_config(), clock=clock.monotonic, sleep=clock.sleep)
        assert result.already_active is True
        assert result.attempts == 0
        assert result.cancelled is False
        assert result.hwnd == 100
        assert fake.set_calls == []

    def test_inactive_window_set_foreground_and_verify(self):
        # Неактивное окно: SetForeground + verify (фокус получен) → OK.
        fake = FakeWin32Gateway({100: make_window()}, foreground=999)
        clock = FakeClock()
        result = activate_game(fake, config=make_config(), clock=clock.monotonic, sleep=clock.sleep)
        assert result.already_active is False
        assert result.attempts == 1
        assert result.cancelled is False
        assert fake.set_calls == [100]
        assert fake.foreground == 100

    def test_retry_on_verify_fail_then_success(self):
        # Первый SetForeground не дал фокус (verify таймаутит), повтор —
        # сработал. Verify после каждой попытки (инвариант 2).
        fake = FakeWin32Gateway(
            {100: make_window()}, foreground=999, focus_results=[False, True]
        )
        clock = FakeClock()
        result = activate_game(fake, config=make_config(), clock=clock.monotonic, sleep=clock.sleep)
        assert result.attempts == 2
        assert result.cancelled is False
        assert fake.set_calls == [100, 100]

    def test_failed_verify_consumes_config_timeout(self):
        # Таймаут verify одной попытки — timing.wait_default_timeout из
        # конфига: часы продвинулись на него до успешного повтора.
        fake = FakeWin32Gateway(
            {100: make_window()}, foreground=999, focus_results=[False, True]
        )
        clock = FakeClock()
        activate_game(
            fake,
            config=make_config(wait_default_timeout=3.0),
            clock=clock.monotonic,
            sleep=clock.sleep,
        )
        assert clock.now >= 3.0

    def test_exhausted_retries_raise_escalation(self):
        # Исчерпание retry: сигнал эскалации (ActivateGameError), не слепые
        # попытки дальше. attempts = 1 + timing.click_retry_count.
        fake = FakeWin32Gateway(
            {100: make_window()}, foreground=999, focus_results=[False, False, False]
        )
        clock = FakeClock()
        with pytest.raises(ActivateGameError) as exc_info:
            activate_game(fake, config=make_config(), clock=clock.monotonic, sleep=clock.sleep)
        error = exc_info.value
        assert error.reason == REASON_FOCUS_NOT_ACQUIRED
        assert "3" in error.detail  # число попыток в диагностике
        assert len(fake.set_calls) == 3  # 1 + click_retry_count(2), не больше

    def test_exhausted_retries_respect_config_retry_count(self):
        fake = FakeWin32Gateway(
            {100: make_window()}, foreground=999, focus_results=[False, False]
        )
        clock = FakeClock()
        with pytest.raises(ActivateGameError):
            activate_game(
                fake,
                config=make_config(click_retry_count=1),
                clock=clock.monotonic,
                sleep=clock.sleep,
            )
        assert len(fake.set_calls) == 2  # 1 + click_retry_count(1)

    def test_window_not_found_raises_with_reason(self):
        # Окно не найдено: диагностика находилки (3.2), SetForeground не
        # вызывается.
        fake = FakeWin32Gateway({}, foreground=999)
        clock = FakeClock()
        with pytest.raises(ActivateGameError) as exc_info:
            activate_game(fake, config=make_config(), clock=clock.monotonic, sleep=clock.sleep)
        assert exc_info.value.reason == REASON_NOT_FOUND
        assert fake.set_calls == []

    def test_minimized_window_raises_with_reason(self):
        fake = FakeWin32Gateway({100: make_window(iconic=True)}, foreground=999)
        clock = FakeClock()
        with pytest.raises(ActivateGameError) as exc_info:
            activate_game(fake, config=make_config(), clock=clock.monotonic, sleep=clock.sleep)
        assert exc_info.value.reason == REASON_MINIMIZED
        assert fake.set_calls == []

    def test_window_criteria_come_from_config(self):
        # Критерии поиска — config.window: под другим заголовком окно не
        # находится.
        fake = FakeWin32Gateway({100: make_window(title="Другое окно")}, foreground=100)
        clock = FakeClock()
        with pytest.raises(ActivateGameError) as exc_info:
            activate_game(fake, config=make_config(), clock=clock.monotonic, sleep=clock.sleep)
        assert exc_info.value.reason == REASON_NOT_FOUND

    def test_cancelled_interrupts_waiting(self):
        # cancelled (kill switch 8.1) прерывает ожидание verify: определённый
        # результат с флагом, не исключение и не бесконечное ожидание.
        fake = FakeWin32Gateway({100: make_window()}, foreground=999)
        clock = FakeClock()
        result = activate_game(
            fake,
            config=make_config(),
            cancelled=lambda: True,
            clock=clock.monotonic,
            sleep=clock.sleep,
        )
        assert result.cancelled is True
        assert result.already_active is False
        assert clock.now == 0.0  # не ждали ни секунды

    def test_cancelled_before_activation_checks_first(self):
        # Отмена до первого poll'а wait_until: SetForeground уже вызван
        # (попытка началась), но ожидание не стартовало.
        fake = FakeWin32Gateway({100: make_window()}, foreground=999)
        clock = FakeClock()
        result = activate_game(
            fake,
            config=make_config(),
            cancelled=lambda: True,
            clock=clock.monotonic,
            sleep=clock.sleep,
        )
        assert result.attempts == 1
        assert fake.set_calls == [100]
