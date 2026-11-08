"""Win32-шлюз и находилка окна игры.

Тонкий шлюз изолирует обращения к Win32 (поиск окон, клиентская область,
свёрнутость, DPI-awareness) за протоколом Win32Gateway — в тестах он
подменяется fake-реализацией поверх dict'ов, живой запуск использует
PyWin32Gateway. Никаких обращений к реальным окнам в тестах.

GameWindowFinder — логика выбора окна игры (тестируется без Windows):
критерии поиска (заголовок/класс) передаются в конструктор, результат —
GameWindow: hwnd, прямоугольник клиентской области в экранных
физических пикселях (x, y, width, height), признак валидности и причина
отказа. Политика выбора при нескольких кандидатах: из не свёрнутых окон
с непустой клиентской областью выбирается окно с максимальной площадью
клиентской области; если валидных нет, причина отказа — по приоритету:
«все свёрнуты» важнее «нет клиентской области» важнее «не найдено».

ensure_dpi_aware объявляет процесс ядра DPI-aware (требование safety:
без этого координаты Win32 и mss расходятся при системном масштабе != 100%).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# Причины отказа находилки (диагностика для статуса not_found, задача 3.5)
REASON_NOT_FOUND = "window_not_found"  # окно не найдено по заголовку/классу
REASON_MINIMIZED = "window_minimized"  # найденные окна свёрнуты (IsIconic)
REASON_NO_CLIENT_AREA = "no_client_area"  # у окон нет видимой клиентской области


@runtime_checkable
class Win32Gateway(Protocol):
    """Тонкий шлюз к Win32; реальный — PyWin32Gateway, в тестах — fake."""

    def find_windows(self, title: str | None, class_name: str | None) -> list[int]:
        """Список hwnd окон, точно совпадающих по заголовку и/или классу."""

    def get_client_rect(self, hwnd: int) -> tuple[int, int, int, int] | None:
        """Клиентская область окна в экранных физических пикселях
        (x, y, width, height); None — клиентской области нет."""

    def is_iconic(self, hwnd: int) -> bool:
        """True, если окно свёрнуто в значок."""

    def set_process_dpi_aware(self) -> bool:
        """Объявить процесс DPI-aware; False — объявить не удалось."""


@dataclass(frozen=True)
class GameWindow:
    """Результат поиска окна игры."""

    hwnd: int | None
    rect: tuple[int, int, int, int] | None  # (x, y, width, height), физические пиксели
    ok: bool
    reason: str | None  # None при ok; иначе одна из REASON_*

    @staticmethod
    def found(hwnd: int, rect: tuple[int, int, int, int]) -> "GameWindow":
        return GameWindow(hwnd=hwnd, rect=rect, ok=True, reason=None)

    @staticmethod
    def failed(reason: str) -> "GameWindow":
        return GameWindow(hwnd=None, rect=None, ok=False, reason=reason)


class GameWindowFinder:
    """Поиск окна игры по критериям (заголовок/класс) поверх шлюза."""

    def __init__(
        self,
        gateway: Win32Gateway,
        title: str | None = "Lineage II",
        class_name: str | None = None,
    ):
        if title is None and class_name is None:
            raise ValueError("нужен хотя бы один критерий поиска: title или class_name")
        self._gateway = gateway
        self._title = title
        self._class_name = class_name

    def find(self) -> GameWindow:
        candidates = self._gateway.find_windows(self._title, self._class_name)
        if not candidates:
            return GameWindow.failed(REASON_NOT_FOUND)

        # Политика выбора: из не свёрнутых окон с непустой клиентской областью
        # — окно с максимальной площадью клиентской области.
        valid: list[tuple[int, tuple[int, int, int, int]]] = []
        saw_iconic = False
        for hwnd in candidates:
            if self._gateway.is_iconic(hwnd):
                saw_iconic = True
                continue
            rect = self._gateway.get_client_rect(hwnd)
            if rect is None or rect[2] <= 0 or rect[3] <= 0:
                continue
            valid.append((hwnd, rect))

        if valid:
            hwnd, rect = max(valid, key=lambda item: item[1][2] * item[1][3])
            return GameWindow.found(hwnd, rect)
        if saw_iconic:
            return GameWindow.failed(REASON_MINIMIZED)
        return GameWindow.failed(REASON_NO_CLIENT_AREA)


def ensure_dpi_aware(gateway: Win32Gateway) -> bool:
    """Объявить процесс ядра DPI-aware; True — объявлен, False — не удалось."""
    return gateway.set_process_dpi_aware()


class Win32GatewayError(Exception):
    """Win32-вызов недоступен (pywin32 не установлен, ошибка API)."""


class PyWin32Gateway:
    """Реальный шлюз через pywin32/ctypes (ленивый импорт — тесты без Windows-окон)."""

    def find_windows(self, title: str | None, class_name: str | None) -> list[int]:
        try:
            import win32gui
        except ImportError as exc:  # pragma: no cover - зависит от окружения
            raise Win32GatewayError("pywin32 не установлен") from exc

        result: list[int] = []

        def callback(hwnd, _extra):
            if not win32gui.IsWindowVisible(hwnd):
                return True
            if title is not None and win32gui.GetWindowText(hwnd) != title:
                return True
            if class_name is not None and win32gui.GetClassName(hwnd) != class_name:
                return True
            result.append(hwnd)
            return True

        win32gui.EnumWindows(callback, None)
        return result

    def get_client_rect(self, hwnd: int) -> tuple[int, int, int, int] | None:
        import win32gui

        left, top, right, bottom = win32gui.GetClientRect(hwnd)
        if right <= 0 or bottom <= 0:
            return None
        # GetClientRect — в координатах клиентской области; переводим в экранные
        # физические пиксели (процесс объявлен DPI-aware, маппинг 1:1).
        x, y = win32gui.ClientToScreen(hwnd, (0, 0))
        return (x, y, right - left, bottom - top)

    def is_iconic(self, hwnd: int) -> bool:
        import win32gui

        return bool(win32gui.IsIconic(hwnd))

    def set_process_dpi_aware(self) -> bool:
        # Сначала per-monitor DPI awareness (Vista+), при недоступности —
        # системная SetProcessDPIAware. Обе идемпотентны на уровне ОС.
        try:
            import ctypes

            shcore = ctypes.windll.shcore
            # PROCESS_PER_MONITOR_DPI_AWARE = 2
            shcore.SetProcessDpiAwareness(2)
            return True
        except Exception:  # pragma: no cover - зависит от ОС
            try:
                import ctypes

                return bool(ctypes.windll.user32.SetProcessDPIAware())
            except Exception:
                return False
