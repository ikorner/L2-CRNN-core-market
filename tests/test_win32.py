"""Юнит-тесты находилки окна игры и DPI-aware объявления на fake-шлюзе Win32.

Никаких обращений к реальным окнам: FakeWin32Gateway — словарь «окон»,
реализующий тот же протокол, что и PyWin32Gateway (l2core.adapters.win32).
"""

import pytest

from l2core.adapters.win32 import (
    REASON_MINIMIZED,
    REASON_NO_CLIENT_AREA,
    REASON_NOT_FOUND,
    GameWindowFinder,
    ensure_dpi_aware,
)
from l2core.config import ConfigError, load_config


class FakeWin32Gateway:
    """Шлюз Win32 поверх dict'ов: windows[hwnd] = {title, class, iconic, client}."""

    def __init__(self, windows=None, dpi_ok=True):
        self.windows = dict(windows or {})
        self.dpi_ok = dpi_ok
        self.dpi_aware = False
        self.dpi_calls = 0

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
        # Прямоугольник клиентской области в экранных физических пикселях
        # (x, y, width, height); None — окно без клиентской области.
        return self.windows[hwnd]["client"]

    def is_iconic(self, hwnd):
        return self.windows[hwnd]["iconic"]

    def set_process_dpi_aware(self):
        self.dpi_calls += 1
        if self.dpi_ok:
            self.dpi_aware = True
            return True
        return False


def make_window(title="Lineage II", cls="L2Client", iconic=False, client=(10, 20, 1920, 1080)):
    return {"title": title, "class": cls, "iconic": iconic, "client": client}


class TestFindGameWindow:
    def test_found_returns_hwnd_and_client_rect(self):
        fake = FakeWin32Gateway({100: make_window()})
        result = GameWindowFinder(fake, title="Lineage II").find()
        assert result.ok
        assert result.hwnd == 100
        assert result.rect == (10, 20, 1920, 1080)
        assert result.reason is None

    def test_finder_requires_some_criterion(self):
        fake = FakeWin32Gateway()
        with pytest.raises(ValueError):
            GameWindowFinder(fake, title=None, class_name=None)

    def test_not_found_by_title(self):
        fake = FakeWin32Gateway({100: make_window(title="Другое окно")})
        result = GameWindowFinder(fake, title="Lineage II").find()
        assert not result.ok
        assert result.hwnd is None
        assert result.rect is None
        assert result.reason == REASON_NOT_FOUND

    def test_found_by_class_name(self):
        fake = FakeWin32Gateway({100: make_window(title="Что угодно", cls="L2Client")})
        result = GameWindowFinder(fake, title=None, class_name="L2Client").find()
        assert result.ok
        assert result.hwnd == 100

    def test_minimized_window_is_rejected_with_reason(self):
        fake = FakeWin32Gateway({100: make_window(iconic=True)})
        result = GameWindowFinder(fake, title="Lineage II").find()
        assert not result.ok
        assert result.reason == REASON_MINIMIZED

    def test_window_without_client_area_is_rejected(self):
        fake = FakeWin32Gateway({100: make_window(client=None)})
        result = GameWindowFinder(fake, title="Lineage II").find()
        assert not result.ok
        assert result.reason == REASON_NO_CLIENT_AREA

    def test_zero_sized_client_area_is_rejected(self):
        fake = FakeWin32Gateway({100: make_window(client=(0, 0, 0, 0))})
        result = GameWindowFinder(fake, title="Lineage II").find()
        assert not result.ok
        assert result.reason == REASON_NO_CLIENT_AREA

    def test_multiple_candidates_prefers_largest_visible(self):
        # Политика выбора: из не свёрнутых окон с непустой клиентской
        # областью выбирается окно с максимальной площадью клиентской области.
        fake = FakeWin32Gateway(
            {
                100: make_window(client=(0, 0, 800, 600)),
                101: make_window(iconic=True, client=(0, 0, 1920, 1080)),
                102: make_window(client=(10, 20, 1920, 1080)),
            }
        )
        result = GameWindowFinder(fake, title="Lineage II").find()
        assert result.ok
        assert result.hwnd == 102

    def test_all_candidates_minimized(self):
        fake = FakeWin32Gateway(
            {100: make_window(iconic=True), 101: make_window(iconic=True)}
        )
        result = GameWindowFinder(fake, title="Lineage II").find()
        assert not result.ok
        assert result.reason == REASON_MINIMIZED

    def test_minimized_takes_precedence_over_zero_area(self):
        fake = FakeWin32Gateway(
            {100: make_window(iconic=False, client=(0, 0, 0, 0)),
             101: make_window(iconic=True)}
        )
        result = GameWindowFinder(fake, title="Lineage II").find()
        assert not result.ok
        assert result.reason == REASON_MINIMIZED


class TestDpiAware:
    def test_process_marked_dpi_aware(self):
        fake = FakeWin32Gateway()
        assert ensure_dpi_aware(fake) is True
        assert fake.dpi_aware is True

    def test_dpi_aware_failure_is_reported(self):
        fake = FakeWin32Gateway(dpi_ok=False)
        assert ensure_dpi_aware(fake) is False
        assert fake.dpi_aware is False


class TestWindowConfig:
    def test_defaults(self):
        cfg = load_config(None)
        assert cfg.window.title == "Lineage II"
        assert cfg.window.class_name is None

    def test_window_section_from_yaml(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("window:\n  title: Моя игра\n  class_name: L2Client\n", encoding="utf-8")
        cfg = load_config(path)
        assert cfg.window.title == "Моя игра"
        assert cfg.window.class_name == "L2Client"

    def test_empty_title_rejected(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("window:\n  title: ''\n", encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(path)


class TestCursorPos:
    """get_cursor_pos — позиция курсора для wiring HID (задача 7.1).

    Провайдер cursor_y/cursor_x протокола HID-адаптера: ядру нужна текущая
    абсолютная позиция курсора для расчёта относительных дельт move_rel.
    """

    def test_real_gateway_returns_tuple_of_ints(self):
        pytest.importorskip("win32gui")
        from l2core.adapters.win32 import PyWin32Gateway

        pos = PyWin32Gateway().get_cursor_pos()
        assert isinstance(pos, tuple)
        assert len(pos) == 2
        assert all(isinstance(v, int) for v in pos)

    def test_protocol_declares_method(self):
        import inspect

        from l2core.adapters.win32 import Win32Gateway

        assert hasattr(Win32Gateway, "get_cursor_pos")
        sig = inspect.signature(Win32Gateway.get_cursor_pos)
        assert list(sig.parameters) == ["self"]
