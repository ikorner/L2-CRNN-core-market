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

import logging
import threading
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

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

    def get_foreground_window(self) -> int | None:
        """hwnd окна в фокусе ввода; None — фокус не установлен."""

    def set_foreground(self, hwnd: int) -> bool:
        """SetForegroundWindow; True — вызов выполнен, False — ошибка API."""

    def set_process_dpi_aware(self) -> bool:
        """Объявить процесс DPI-aware; False — объявить не удалось."""

    def get_cursor_pos(self) -> tuple[int, int] | None:
        """Позиция курсора в экранных пикселях (x, y); None — недоступна.
        Нужна ядру для расчёта относительных дельт HID (wiring 7.1)."""

    def set_cursor_pos(self, x: int, y: int) -> bool:
        """Установить позицию курсора; True — выполнено. Нужен kill switch
        (8.1) для увода курсора в безопасную точку при аварийной остановке."""


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

    def get_foreground_window(self) -> int | None:
        import win32gui

        hwnd = win32gui.GetForegroundWindow()
        return hwnd if hwnd else None

    def set_foreground(self, hwnd: int) -> bool:
        import win32gui

        try:
            win32gui.SetForegroundWindow(hwnd)
            return True
        except Exception:  # pragma: no cover - зависит от ОС/прав
            return False

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

    def get_cursor_pos(self) -> tuple[int, int] | None:
        import win32gui

        try:
            x, y = win32gui.GetCursorPos()
            return (int(x), int(y))
        except Exception:  # pragma: no cover - зависит от ОС
            return None

    def set_cursor_pos(self, x: int, y: int) -> bool:
        import win32gui

        try:
            win32gui.SetCursorPos((int(x), int(y)))
            return True
        except Exception:  # pragma: no cover - зависит от ОС/прав
            return False


# --- Источник локального хоткея для kill switch (задача 8.1) ---

# Виртуальные коды клавиш (Win32): буквы A-Z, цифры 0-9, F1-F12.
_VK_LETTER_BASE = 0x41
_VK_DIGIT_BASE = 0x30
_VK_F1 = 0x70
_MODIFIERS = ("ctrl", "shift", "alt")
_WH_KEYBOARD_LL = 13
_WM_KEYDOWN = 0x100
_WM_SYSKEYDOWN = 0x104
_WM_QUIT = 0x12
_VK_CONTROL = 0x11
_VK_SHIFT = 0x10
_VK_MENU = 0x12
# Левая/правая клавиши модификаторов: для LL-хука GetKeyState на них
# не реагирует — состояние смотрим и по обычному, и по расширенному VK.
_VK_LCONTROL = 0xA2
_VK_RCONTROL = 0xA3
_VK_LSHIFT = 0xA0
_VK_RSHIFT = 0xA1
_VK_LMENU = 0xA4
_VK_RMENU = 0xA5


def _parse_hotkey(hotkey: str) -> tuple[int, frozenset]:
    """Строка вида «ctrl+shift+k» -> (vk, модификаторы). Fail fast:
    неизвестный формат — ValueError (опечатка в конфиге не должна
    привести к молчаливому хоткею, который не сработает)."""
    if not isinstance(hotkey, str) or not hotkey.strip():
        raise ValueError(f"хоткей: ожидается непустая строка (получено {hotkey!r})")
    parts = [part.strip().lower() for part in hotkey.split("+")]
    if any(not part for part in parts):
        raise ValueError(f"хоткей {hotkey!r}: пустой компонент ('ctrl++')")

    modifiers: set = set()
    key: str | None = None
    for part in parts:
        if part in _MODIFIERS:
            if part in modifiers:
                raise ValueError(f"хоткей {hotkey!r}: дубль модификатора {part!r}")
            modifiers.add(part)
        elif key is None:
            key = part
        else:
            raise ValueError(f"хоткей {hotkey!r}: больше одной клавиши ({key!r}, {part!r})")

    if key is None:
        raise ValueError(f"хоткей {hotkey!r}: нет клавиши (только модификаторы)")
    if len(key) == 1 and "a" <= key <= "z":
        vk = _VK_LETTER_BASE + (ord(key) - ord("a"))
    elif len(key) == 1 and "0" <= key <= "9":
        vk = _VK_DIGIT_BASE + (ord(key) - ord("0"))
    elif len(key) == 2 and key[0] == "f" and key[1].isdigit():
        number = int(key[1])
        if not 1 <= number <= 9:
            raise ValueError(f"хоткей {hotkey!r}: неизвестная клавиша {key!r}")
        vk = _VK_F1 + number - 1
    elif key in ("f10", "f11", "f12"):
        vk = _VK_F1 + int(key[1:]) - 1
    else:
        raise ValueError(f"хоткей {hotkey!r}: неизвестная клавиша {key!r} "
                         f"(поддерживаются a-z, 0-9, f1-f12)")
    return vk, frozenset(modifiers)


class Win32HotkeySource:
    """Прод-источник локального хоткея kill switch: низкоуровневый
    клавиатурный хук WH_KEYBOARD_LL в собственном потоке (spec safety:
    слушаем клавиатуру локально, НЕ через arduino).

    Хук, а не опрос GetAsyncKeyState: нажатие фиксируется самим хуком в
    момент события — короткое нажатие не пропадёт между poll'ами слушателя,
    реакция мгновенная (миллисекунды, а не интервал опроса). Импорт Win32 —
    ленивый, как в 3.2: парсинг хоткея в __init__ работает без pywin32,
    hook ставится только в start() (живой код, на машине разработчика).

    Контракт (потребляет l2core/api/killswitch.py): start() — запустить
    поток хука; stop() — снять хук и завершить поток; pressed() — факт
    нажатия хоткея (после срабатывания флаг держится до конца процесса —
    kill switch идемпотентен, повторные нажатия не нужны).
    """

    def __init__(self, hotkey: str):
        self._vk, self._modifiers = _parse_hotkey(hotkey)
        self._hotkey = hotkey
        self._pressed = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def vk(self) -> int:
        """Виртуальный код клавиши хоткея (Win32 VK)."""
        return self._vk

    @property
    def modifiers(self) -> frozenset:
        """Модификаторы хоткея ('ctrl'/'shift'/'alt')."""
        return self._modifiers

    def start(self) -> None:
        """Запустить поток хука (повторный вызов — no-op)."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._hook_loop, name="l2core-hotkey-hook", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Снять хук и завершить поток (join с таймаутом; idempotent)."""
        thread = self._thread
        if thread is None:
            return
        import ctypes

        user32 = ctypes.windll.user32
        # WM_QUIT потоку хука: GetMessage-цикл выйдет и снимет хук.
        user32.PostThreadMessageW(thread.ident, _WM_QUIT, 0, 0)
        thread.join(timeout=2.0)
        self._thread = None

    def pressed(self) -> bool:
        """Хоткей нажат (флаг выставляет хук при совпадении комбинации)."""
        return self._pressed.is_set()

    def _hook_loop(self) -> None:  # pragma: no cover - живой Win32-канал
        """Тело потока хука: SetWindowsHookEx + message loop.

        pywin32 не экспортирует удобной обёртки LL-хуков — делаем через
        ctypes (ленивый импорт, как весь шлюз). Ошибки инициализации
        фиксируются в лог: без хука остаётся таймаут неактивности.
        """
        import ctypes
        import ctypes.wintypes

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        hook_handle = None
        try:
            callback_type = ctypes.WINFUNCTYPE(
                ctypes.c_long, ctypes.c_int, ctypes.c_size_t, ctypes.c_size_t)

            vk_target = self._vk
            modifiers = self._modifiers
            pressed = self._pressed

            class _KbdLlHookStruct(ctypes.Structure):
                _fields_ = [("vkCode", ctypes.c_ulong),
                            ("scanCode", ctypes.c_ulong),
                            ("flags", ctypes.c_ulong),
                            ("time", ctypes.c_ulong),
                            ("dwExtraInfo", ctypes.c_size_t)]

            def _modifier_down(*vks) -> bool:
                # GetKeyState: старший бит — клавиша сейчас нажата.
                return any(user32.GetKeyState(vk) & 0x8000 for vk in vks)

            def _hook_proc(code, w_param, l_param):
                if code == 0 and w_param in (_WM_KEYDOWN, _WM_SYSKEYDOWN):
                    info = ctypes.cast(
                        l_param, ctypes.POINTER(_KbdLlHookStruct)).contents
                    if info.vkCode == vk_target:
                        mods_ok = all((
                            "ctrl" not in modifiers
                            or _modifier_down(_VK_CONTROL, _VK_LCONTROL,
                                              _VK_RCONTROL),
                            "shift" not in modifiers
                            or _modifier_down(_VK_SHIFT, _VK_LSHIFT, _VK_RSHIFT),
                            "alt" not in modifiers
                            or _modifier_down(_VK_MENU, _VK_LMENU, _VK_RMENU),
                        ))
                        if mods_ok:
                            pressed.set()
                return user32.CallNextHookEx(None, code, w_param, l_param)

            callback = callback_type(_hook_proc)
            # HINSTANCE NULL (0) — хук на уровне потока-сообщений,
            # dwThreadId 0 — все потоки (глобальный LL-хук).
            hook_handle = user32.SetWindowsHookExW(
                _WH_KEYBOARD_LL, callback, 0, 0)
            if not hook_handle:
                logger.error(
                    "hotkey: SetWindowsHookEx не удалось (код %s): "
                    "хоткей %s не активен, остаётся таймаут неактивности",
                    kernel32.GetLastError(), self._hotkey)
                return
            logger.info("hotkey: хук WH_KEYBOARD_LL активен (%s)",
                        self._hotkey)
            message = ctypes.wintypes.MSG()
            # Message loop: хук жив, пока поток крутит GetMessage.
            while user32.GetMessageW(ctypes.byref(message), 0, 0, 0) > 0:
                pass
        except Exception:
            logger.exception("hotkey: поток хука завершился ошибкой")
        finally:
            if hook_handle:
                user32.UnhookWindowsHookEx(hook_handle)
            logger.info("hotkey: хук снят")
