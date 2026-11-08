"""
Вендоринг: бинарный протокол HID-контроллера (Arduino Leonardo).

Источник: L2-CRNN-arduino/hid.py, коммит 6be55bc0af73b2b848487f987f36de851a61d21b.
Копия без редизайна; правки вендоренного кода запрещены (менять в источнике).
"""

"""
Бинарный протокол HID-контроллера (Arduino Leonardo).

Формат пакета: [CMD][PAYLOAD][XOR-чексумма], чексумма = XOR всех байт
от CMD до конца payload. Прошивка: hid_controller.ino (v2.0).

Статусы ответов:
  0x00 = OK (у GET_VERSION далее [major][minor])
  0xFF = PONG
  0xE0 = CHECKSUM_ERROR
  0xE1 = UNKNOWN_CMD
  0xE2 = INVALID_ARG
"""

import random
import struct
import threading
import time
import warnings

# Команды
CMD_MOVE_REL = 0x01
CMD_CLICK = 0x02
CMD_MOUSE_DOWN = 0x03
CMD_MOUSE_UP = 0x04
CMD_TRAJ = 0x05
CMD_SCROLL = 0x06
CMD_RELEASE_ALL = 0x07
CMD_KEY_PRESS = 0x10
CMD_KEY_DOWN = 0x11
CMD_KEY_UP = 0x12
CMD_KEY_COMBO = 0x13
CMD_TYPE_CHAR = 0x20
CMD_GET_VERSION = 0x30
CMD_PING = 0xFF

# Статусы ответов
RESP_OK = 0x00
RESP_PONG = 0xFF
RESP_CHECKSUM_ERR = 0xE0
RESP_UNKNOWN_CMD = 0xE1
RESP_INVALID_ARG = 0xE2

# Кнопки мыши (совпадают с MOUSE_* битами прошивки)
BTN_LEFT = 0x01
BTN_RIGHT = 0x02
BTN_MIDDLE = 0x04

# Клавиши (HID keycodes, как в Keyboard.h)
KEY_LEFT_SHIFT = 0xE1
KEY_LEFT_ALT = 0xE2
KEY_BACKSPACE = 0xB2
KEY_ENTER = 0xB0
KEY_TAB = 0xB3
KEY_ESC = 0xB1

# Требуемая версия прошивки: major обязан совпадать, minor >= указанного
REQUIRED_VERSION = (2, 0)

# VID:PID Arduino Leonardo (bootloader / running sketch)
LEONARDO_VID_PID = [
    (0x2341, 0x8036),
    (0x2341, 0x0036),
]


def xor_checksum(data: bytes) -> int:
    """XOR всех байт пакета (CMD + PAYLOAD, без самой чексуммы)."""
    cs = 0
    for b in data:
        cs ^= b
    return cs


def encode_packet(cmd: int, payload: bytes = b"") -> bytes:
    """Собрать пакет [CMD][PAYLOAD][CHECKSUM]."""
    body = bytes([cmd]) + payload
    return body + bytes([xor_checksum(body)])


# --- Исключения ---

class HidError(Exception):
    """Базовое исключение модуля HID."""


class HidNotFoundError(HidError):
    """Устройство не найдено или порт не открывается."""


class HidTimeoutError(HidError):
    """Устройство не ответило вовремя (команда НЕ повторяется автоматически)."""


class HidChecksumError(HidError):
    """Устройство отклонило пакет по чексумме."""


class IncompatibleFirmwareError(HidError):
    """Прошивка устарела или несовместима по major-версии."""


class HidProtocolError(HidError):
    """Неожиданный ответ устройства или неверные аргументы команды."""


# --- Транспорт ---



class HidDevice:
    """
    Транспорт бинарного протокола HID-контроллера.

    Единственный владелец serial-порта. Все команды сериализуются локом.
    Автоматический повтор — только у идемпотентных PING/GET_VERSION;
    неидемпотентные при таймауте поднимают HidTimeoutError без повтора.
    """

    TRAJ_MAX_POINTS = 60

    def __init__(self, port: str | None = None, baudrate: int = 115200,
                 timeout: float = 2.0, serial_obj=None):
        """
        Args:
            port: COM-порт (None — авто-поиск Leonardo при connect)
            baudrate: скорость (прошивка v2: 115200)
            timeout: базовый таймаут ожидания ответа, сек
            serial_obj: объект с интерфейсом pyserial для тестов (FakeHidDevice)
        """
        self.port_name = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._serial = serial_obj
        self._lock = threading.Lock()
        self.version: tuple[int, int] | None = None

    # --- Жизненный цикл ---

    def connect(self, port: str | None = None) -> None:
        """Подключиться: открыть порт, проверить PING и версию прошивки."""
        if self._serial is None:
            port_to_use = port or self.port_name or self._find_port()
            if port_to_use is None:
                raise HidNotFoundError(
                    "Arduino Leonardo не найдена. Подключите устройство.")
            try:
                import serial
                self._serial = serial.Serial(port_to_use, self.baudrate, timeout=0)
            except Exception as e:
                raise HidNotFoundError(f"Не удалось открыть {port_to_use}: {e}")
            self.port_name = port_to_use
            # Leonardo перезагружается при открытии порта
            time.sleep(2.0)

        self._set_dpi_aware()
        self._check_pointer_settings()

        if not self.ping():
            raise HidTimeoutError("Устройство не отвечает на PING.")
        try:
            version = self.get_version()
        except HidProtocolError as e:
            raise IncompatibleFirmwareError(
                f"{e} Вероятно прошивка устарела — запустите: python flasher.py")
        major, minor = version
        if major != REQUIRED_VERSION[0] or minor < REQUIRED_VERSION[1]:
            raise IncompatibleFirmwareError(
                f"Прошивка {major}.{minor}, требуется >= "
                f"{REQUIRED_VERSION[0]}.{REQUIRED_VERSION[1]} — "
                f"запустите: python flasher.py")
        self.version = version

    def close(self) -> None:
        if self._serial is not None:
            try:
                self._serial.close()
            finally:
                self._serial = None

    def is_connected(self) -> bool:
        return self._serial is not None and bool(getattr(self._serial, "is_open", True))

    def __enter__(self):
        if self._serial is None or not self.is_connected():
            self.connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.release_all()
        except HidError:
            pass
        self.close()

    # --- Низкоуровневый обмен ---

    def _transact(self, cmd: int, payload: bytes = b"", *,
                  idempotent: bool = False, exec_budget_ms: int = 0,
                  extra_bytes: int = 0) -> bytes:
        """
        Отправить пакет и дождаться ответа.

        Returns:
            extra_bytes байт payload ответа (после статуса OK)

        Raises:
            HidTimeoutError: таймаут (без повтора, если не idempotent)
            HidChecksumError: устройство отклонило чексумму
            HidProtocolError: неизвестная команда / неверные аргументы
        """
        packet = encode_packet(cmd, payload)
        attempts = 2 if idempotent else 1
        budget = self.timeout + exec_budget_ms / 1000.0

        with self._lock:
            for attempt in range(attempts):
                t0 = time.monotonic()
                self._serial.write(packet)
                flush = getattr(self._serial, "flush", None)
                if flush:
                    flush()
                deadline = t0 + budget
                try:
                    status = self._read_byte(deadline)
                except HidTimeoutError:
                    if attempt < attempts - 1:
                        continue
                    raise HidTimeoutError(
                        f"Таймаут ожидания ответа на команду 0x{cmd:02X} "
                        f"({self.port_name or 'fake'}). Команда могла выполниться "
                        f"— повтор не выполняется, синхронизируйте состояние.")

                if status == RESP_OK:
                    return bytes(self._read_byte(deadline) for _ in range(extra_bytes))
                if status == RESP_PONG and cmd == CMD_PING:
                    return b""
                if status == RESP_CHECKSUM_ERR:
                    raise HidChecksumError(
                        f"Устройство отклонило пакет команды 0x{cmd:02X} по чексумме.")
                if status == RESP_UNKNOWN_CMD:
                    raise HidProtocolError(
                        f"Устройство не знает команду 0x{cmd:02X}.")
                if status == RESP_INVALID_ARG:
                    raise HidProtocolError(
                        f"Неверные аргументы команды 0x{cmd:02X}.")
                raise HidProtocolError(
                    f"Неожиданный ответ 0x{status:02X} на команду 0x{cmd:02X}.")

    def _read_byte(self, deadline: float) -> int:
        while True:
            data = self._serial.read(1)
            if data:
                return data[0]
            if time.monotonic() >= deadline:
                raise HidTimeoutError("таймаут чтения")
            time.sleep(0.001)

    # --- Сервисные команды ---

    def ping(self) -> bool:
        try:
            self._transact(CMD_PING, idempotent=True)
            return True
        except HidTimeoutError:
            return False

    def get_version(self) -> tuple[int, int]:
        extra = self._transact(CMD_GET_VERSION, idempotent=True, extra_bytes=2)
        return extra[0], extra[1]

    def release_all(self) -> None:
        """Отпустить все кнопки мыши и клавиши (восстановление после ошибок)."""
        self._transact(CMD_RELEASE_ALL)

    # --- Команды устройства ---

    def move_rel(self, dx: int, dy: int) -> None:
        payload = struct.pack("<hh", dx, dy)
        self._transact(CMD_MOVE_REL, payload)

    def click(self, button: int = BTN_LEFT, count: int = 1,
              hold_ms: int | None = None) -> None:
        if hold_ms is None:
            hold_ms = random.randint(30, 120)
        payload = bytes([button, count, hold_ms & 0xFF])
        self._transact(CMD_CLICK, payload)

    def mouse_down(self, button: int = BTN_LEFT) -> None:
        self._transact(CMD_MOUSE_DOWN, bytes([button]))

    def mouse_up(self, button: int = BTN_LEFT) -> None:
        self._transact(CMD_MOUSE_UP, bytes([button]))

    def traj(self, points: list[tuple[int, int, int]]) -> None:
        """
        Пакетная траектория: (dx, dy, delay_ms) за точку, исполнение на плате.

        Одна команда — до TRAJ_MAX_POINTS точек (одна траектория = один пакет).
        """
        if not 1 <= len(points) <= self.TRAJ_MAX_POINTS:
            raise ValueError(f"точек должно быть 1..{self.TRAJ_MAX_POINTS}")
        payload = bytes([len(points)])
        for dx, dy, delay in points:
            payload += struct.pack("<hhH", dx, dy, delay)
        budget = sum(p[2] for p in points) + 500
        self._transact(CMD_TRAJ, payload, exec_budget_ms=budget)

    def scroll(self, count: int, step: int, delay_ms: int) -> None:
        if not 1 <= count <= 255 or not -128 <= step <= 127 or step == 0:
            raise ValueError("count 1..255, step -128..127 и != 0")
        payload = bytes([count, step & 0xFF, delay_ms & 0xFF])
        self._transact(CMD_SCROLL, payload, exec_budget_ms=count * delay_ms + 500)

    def key_press(self, code: int) -> None:
        self._transact(CMD_KEY_PRESS, bytes([code]))

    def key_down(self, code: int) -> None:
        self._transact(CMD_KEY_DOWN, bytes([code]))

    def key_up(self, code: int) -> None:
        self._transact(CMD_KEY_UP, bytes([code]))

    def key_combo(self, mod: int, code: int) -> None:
        self._transact(CMD_KEY_COMBO, bytes([mod, code]))

    def type_char(self, ch: str) -> None:
        self._transact(CMD_TYPE_CHAR, ch.encode("latin-1"))

    # --- Окружение Windows ---

    @staticmethod
    def _find_port() -> str | None:
        import serial.tools.list_ports
        for port in serial.tools.list_ports.comports():
            for vid, pid in LEONARDO_VID_PID:
                if port.vid == vid and port.pid == pid:
                    return port.device
        return None

    @staticmethod
    def _set_dpi_aware() -> None:
        """Пометить процесс DPI-aware, иначе GetCursorPos врёт при масштабировании."""
        import sys
        if sys.platform != "win32":
            return
        try:
            import ctypes
            # PROCESS_PER_MONITOR_DPI_AWARE = 2
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            pass  # уже помечен или нет shcore — не критично

    @staticmethod
    def _check_pointer_settings() -> None:
        """Предупредить, если настройки указателя искажают дельты мыши."""
        import sys
        if sys.platform != "win32":
            return
        try:
            import ctypes
            user32 = ctypes.windll.user32

            speed = ctypes.c_int(0)
            # SPI_GETMOUSESPEED = 0x0070
            user32.SystemParametersInfoW(0x0070, 0, ctypes.byref(speed), 0)

            mouse = (ctypes.c_int * 3)()
            # SPI_GETMOUSE = 0x0003; mouse[2] != 0 — включено ускорение указателя
            user32.SystemParametersInfoW(0x0003, 0, ctypes.byref(mouse), 0)

            if speed.value != 10 or mouse[2] != 0:
                warnings.warn(
                    f"Настройки указателя Windows искажают дельты мыши "
                    f"(скорость={speed.value}/10, ускорение={'вкл' if mouse[2] else 'выкл'}). "
                    f"Для точности задайте скорость 10/11 и отключите "
                    f"'Повышенную точность установки указателя'.")
        except Exception:
            pass
