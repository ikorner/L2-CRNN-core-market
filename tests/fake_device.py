"""
FakeHidDevice — эмулятор прошивки hid_controller.ino для тестов без железа.

Интерфейс совместим с pyserial (write/read/is_open/close), read неблокирующий:
возвращает b'' когда ответов нет (как serial.Serial с timeout=0).

Вендоринг из L2-CRNN-arduino (tests/fake_device.py), коммит-источник
6be55bc0af73b2b848487f987f36de851a61d21b; адаптированы только импорты
(константы протокола теперь из l2core.adapters.hid).

Режимы (firmware):
  "2.0"    — полная прошивка v2.0 (валидация чексумм, ответы по протоколу)
  "1.0"    — старая прошивка: на CMD_GET_VERSION отвечает UNKNOWN_CMD
  "drop"   — на неидемпотентные команды ack не отправляет (эмуляция потери)
  "corrupt"— на каждый пакет отвечает CHECKSUM_ERROR
"""

from l2core.adapters.hid import (
    CMD_CLICK,
    CMD_GET_VERSION,
    CMD_KEY_COMBO,
    CMD_KEY_DOWN,
    CMD_KEY_PRESS,
    CMD_KEY_UP,
    CMD_MOUSE_DOWN,
    CMD_MOUSE_UP,
    CMD_MOVE_REL,
    CMD_PING,
    CMD_RELEASE_ALL,
    CMD_SCROLL,
    CMD_TRAJ,
    CMD_TYPE_CHAR,
    RESP_CHECKSUM_ERR,
    RESP_OK,
    RESP_PONG,
    RESP_UNKNOWN_CMD,
    xor_checksum,
)

# Длина payload по команде (кроме CMD_TRAJ — переменная)
_PAYLOAD_LEN = {
    CMD_MOVE_REL: 4,
    CMD_CLICK: 3,
    CMD_MOUSE_DOWN: 1,
    CMD_MOUSE_UP: 1,
    CMD_SCROLL: 3,
    CMD_RELEASE_ALL: 0,
    CMD_KEY_PRESS: 1,
    CMD_KEY_DOWN: 1,
    CMD_KEY_UP: 1,
    CMD_KEY_COMBO: 2,
    CMD_TYPE_CHAR: 1,
    CMD_GET_VERSION: 0,
    CMD_PING: 0,
}

_EFFECTFUL = {
    CMD_MOVE_REL, CMD_CLICK, CMD_MOUSE_DOWN, CMD_MOUSE_UP, CMD_TRAJ,
    CMD_SCROLL, CMD_KEY_PRESS, CMD_KEY_DOWN, CMD_KEY_UP, CMD_KEY_COMBO,
    CMD_TYPE_CHAR,
}


class FakeHidDevice:
    """Эмуляция Arduino с прошивкой HID-контроллера."""

    def __init__(self, firmware: str = "2.0"):
        if firmware not in ("2.0", "1.0", "drop", "corrupt"):
            raise ValueError(f"неизвестный режим firmware: {firmware}")
        self.firmware = firmware
        self.is_open = True
        self.rx = bytearray()      # неразобранный входящий поток
        self.tx = bytearray()      # очередь ответов
        self.received: list[bytes] = []   # принятые пакеты целиком
        # (пакет, сколько ответов уже отправлено на момент приёма) —
        # для проверки, что следующая команда ушла только после ответа
        self.write_log: list[tuple[bytes, int]] = []
        self.responses_sent = 0

    # --- интерфейс pyserial ---

    def write(self, data: bytes) -> int:
        assert self.is_open
        self.rx.extend(data)
        self._drain()
        return len(data)

    def read(self, n: int) -> bytes:
        if not self.tx:
            return b""
        take = min(n, len(self.tx))
        out = bytes(self.tx[:take])
        del self.tx[:take]
        return out

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.is_open = False

    # --- эмуляция прошивки ---

    def _packet_total_len(self) -> int | None:
        """Длина первого пакета в буфере rx, или None если ещё неполный."""
        if not self.rx:
            return None
        cmd = self.rx[0]
        if cmd == CMD_TRAJ:
            if len(self.rx) < 2:
                return None
            count = self.rx[1]
            return 1 + 1 + count * 6 + 1
        plen = _PAYLOAD_LEN.get(cmd, 0)
        return 1 + plen + 1

    def _drain(self) -> None:
        while True:
            total = self._packet_total_len()
            if total is None or len(self.rx) < total:
                return
            packet = bytes(self.rx[:total])
            del self.rx[:total]
            self.received.append(packet)
            self.write_log.append((packet, self.responses_sent))
            self._process(packet)

    def _respond(self, data: bytes) -> None:
        self.tx.extend(data)
        self.responses_sent += 1

    def _process(self, packet: bytes) -> None:
        # Прошивка валидирует чексумму до исполнения
        if xor_checksum(packet[:-1]) != packet[-1]:
            self._respond(bytes([RESP_CHECKSUM_ERR]))
            return

        cmd = packet[0]

        if self.firmware == "corrupt":
            self._respond(bytes([RESP_CHECKSUM_ERR]))
            return

        if cmd == CMD_PING:
            self._respond(bytes([RESP_PONG]))
            return

        if cmd == CMD_GET_VERSION:
            if self.firmware == "1.0":
                self._respond(bytes([RESP_UNKNOWN_CMD]))  # v1 не знает команды
            else:
                self._respond(bytes([RESP_OK, 2, 0]))
            return

        if self.firmware == "1.0":
            # v1 не знает новых команд и старые исполняет
            if cmd in (CMD_TRAJ, CMD_SCROLL, CMD_RELEASE_ALL) or cmd == CMD_CLICK:
                self._respond(bytes([RESP_UNKNOWN_CMD]))
            else:
                self._respond(bytes([RESP_OK]))
            return

        if self.firmware == "drop" and cmd in _EFFECTFUL:
            return  # ack потерян

        if cmd in _PAYLOAD_LEN or cmd == CMD_TRAJ:
            self._respond(bytes([RESP_OK]))
        else:
            self._respond(bytes([RESP_UNKNOWN_CMD]))
