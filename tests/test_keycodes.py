"""Кейкоды клавиш (menu-quickslot-navigation, Task 2)."""

from l2core.adapters.keycodes import KEY_X, KEYCODES


def test_key_x_hid_usage_id():
    # HID usage id клавиши X, как KEY_ESC = 0xB1 в Arduino Keyboard.h.
    assert KEY_X == 0x1B
    assert KEYCODES["X"] == KEY_X
