"""Юнит-тесты вендоренного HID-транспорта на эмуляторе (без платы)."""

import struct

import pytest

from l2core.adapters.hid import (
    BTN_LEFT,
    CMD_CLICK,
    CMD_GET_VERSION,
    CMD_KEY_DOWN,
    CMD_KEY_PRESS,
    CMD_KEY_UP,
    CMD_MOVE_REL,
    CMD_PING,
    CMD_RELEASE_ALL,
    HidChecksumError,
    HidDevice,
    HidTimeoutError,
    IncompatibleFirmwareError,
    KEY_ENTER,
    encode_packet,
)
from tests.fake_device import FakeHidDevice


def make_connected(firmware: str = "2.0") -> tuple[HidDevice, FakeHidDevice]:
    """HidDevice поверх эмулятора с выполненным рукопожатием connect()."""
    fake = FakeHidDevice(firmware)
    hid = HidDevice(serial_obj=fake)
    hid.connect()
    return hid, fake


class TestConnect:
    def test_connect_ping_and_version(self):
        fake = FakeHidDevice()
        hid = HidDevice(serial_obj=fake)
        hid.connect()
        assert hid.version == (2, 0)
        assert hid.is_connected()
        cmds = [p[0] for p in fake.received]
        assert cmds[:2] == [CMD_PING, CMD_GET_VERSION]

    def test_old_firmware_rejected(self):
        hid = HidDevice(serial_obj=FakeHidDevice("1.0"))
        with pytest.raises(IncompatibleFirmwareError):
            hid.connect()

    def test_drop_mode_ping_ok_but_click_times_out(self):
        # PING идемпотентен и ретраится, click — нет (UNKNOWN эффект)
        hid = HidDevice(serial_obj=FakeHidDevice("drop"), timeout=0.05)
        hid.connect()  # рукопожатие проходит (PING/GET_VERSION не effectful)
        with pytest.raises(HidTimeoutError):
            hid.click()

    def test_corrupt_mode_rejected_by_checksum(self):
        hid = HidDevice(serial_obj=FakeHidDevice("corrupt"))
        with pytest.raises(HidChecksumError):
            hid.connect()


class TestActions:
    def test_click_at_coordinates(self):
        """Клик по координатам: перемещение + клик, пакеты зафиксированы."""
        hid, fake = make_connected()
        hid.move_rel(500, 300)
        hid.click()
        cmds = [p[0] for p in fake.received]
        move_idx = cmds.index(CMD_MOVE_REL, 2)
        click_idx = cmds.index(CMD_CLICK, 2)
        assert move_idx < click_idx
        dx, dy = struct.unpack("<hh", fake.received[move_idx][1:-1])
        assert (dx, dy) == (500, 300)
        button, count, _hold = fake.received[click_idx][1:-1]
        assert button == BTN_LEFT and count == 1

    def test_key_press_and_release(self):
        hid, fake = make_connected()
        hid.key_press(KEY_ENTER)
        hid.key_down(KEY_ENTER)
        hid.key_up(KEY_ENTER)
        cmds = [p[0] for p in fake.received[2:]]
        assert cmds == [CMD_KEY_PRESS, CMD_KEY_DOWN, CMD_KEY_UP]
        assert all(p[1] == KEY_ENTER for p in fake.received[2:])

    def test_release_all_releases(self):
        hid, fake = make_connected()
        hid.release_all()
        assert fake.received[-1] == encode_packet(CMD_RELEASE_ALL)

    def test_each_command_waits_for_response(self):
        """Следующий пакет уходит только после ответа на предыдущий."""
        hid, fake = make_connected()
        hid.key_press(KEY_ENTER)
        hid.key_press(KEY_ENTER)
        # responses_sent на момент приёма каждого пакета: каждая следующая
        # команда ушла только после ответа на предыдущую (0, 1, 2, 3, ...)
        sent = [n for _, n in fake.write_log]
        assert sent == list(range(len(sent)))

    def test_context_manager_releases_all_on_exit(self):
        fake = FakeHidDevice()
        with HidDevice(serial_obj=fake) as hid:
            assert hid.is_connected()
        assert fake.received[-1][0] == CMD_RELEASE_ALL
        assert not fake.is_open
