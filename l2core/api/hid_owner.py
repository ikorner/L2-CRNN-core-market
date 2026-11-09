"""HID-owner guard: единственный владелец HID — action executor (задача 7.2,
spec core-api-events: «action executor — единственный владелец HID» и
сценарий «Observer не отправляет действия»).

OwnerGuardedHid — guard-обёртка поверх HidDevice (ВЕНДОРЕННЫЙ
adapters/hid.py не трогаем — обёртка снаружи): поток-владелец фиксируется
при создании, каждая HID-команда проверяет threading.current_thread().
Вызов из чужого потока (state observer, event dispatcher, чужая стратегия)
— HidOwnershipError ДО обращения к железу: слепых/чужих эффектов нет
(инвариант 9 — enforcement, а не дисциплина).

ВЫБОР АРХИТЕКТУРЫ (вариант Б из design-развилки 7.2, обоснование — в
отчёте task-7.2-report.md): action executor — это поток стратегии (API
синхронен и блокирующий, вызывается только из него — thread-owner
enforcement 7.1). Отдельный executor-поток с очередью команд (вариант А)
при однопоточной стратегии не имеет потребителя: добавил бы async-слой
без выгоды. Guard делает владение исполняемым правилом и одновременно
закрывает сценарий спеки «Observer не отправляет действия». Готовность
к многопоточным стратегиям — не-цель v1 (design 8 не требует).

Провайдеры/примитивы получают guard-обёртку через wiring (L2CoreBot
подменяет wiring.hid обёрнутой копией, см. engine.py); прямые вызовы
исходного HidDevice обходят guard — граница зафиксирована в докстринге
engine.py (сборка — единственная точка, где живой HidDevice доступен).
"""
from __future__ import annotations

import threading
from typing import Optional

__all__ = ["HidOwnershipError", "OwnerGuardedHid"]


class HidOwnershipError(RuntimeError):
    """HID-команда вызвана не из потока-владельца (action executor).

    Команда НЕ дошла до железа: слепых эффектов из чужого потока нет
    (сценарий спеки «Observer не отправляет действия» — исполняемый).
    """


class OwnerGuardedHid:
    """Guard-обёртка HidDevice: команды только из потока-владельца.

    Делегирует весь командный интерфейс HidDevice (плюс connect/close/
    is_connected/ping/get_version — жизненный цикл тоже только у владельца)
    с проверкой потока перед каждым вызовом. Атрибуты (version, port_name,
    TRAJ_MAX_POINTS и пр.) пробрасываются без проверки — чтение метаданных
    безопасно из любого потока.
    """

    def __init__(self, hid, owner_thread: Optional[threading.Thread] = None):
        self._hid = hid
        self._owner = owner_thread or threading.current_thread()

    @property
    def owner_thread(self) -> threading.Thread:
        """Поток-владелец HID (action executor)."""
        return self._owner

    @property
    def inner(self):
        """Обернутый адаптер (диагностика/сборка; команды — только guard'ом)."""
        return self._hid

    def _check(self) -> None:
        """Проверка владения: чужой поток — HidOwnershipError до железа."""
        current = threading.current_thread()
        if current is not self._owner:
            raise HidOwnershipError(
                f"HID-команда из чужого потока ({current.name!r}; владелец — "
                f"{self._owner.name!r}): action executor — единственный "
                f"владелец HID (spec core-api-events, сценарий «Observer не "
                f"отправляет действия»)")

    def __getattr__(self, name: str):
        # Атрибуты (не команды) — прозрачно; HidDevice не имеет других
        # публичных методов вне явного списка ниже — явный список и есть
        # контракт команд.
        return getattr(self._hid, name)

    # --- Жизненный цикл (только владелец) ---

    def connect(self, *args, **kwargs):
        self._check()
        return self._hid.connect(*args, **kwargs)

    def close(self):
        self._check()
        return self._hid.close()

    def is_connected(self):
        self._check()
        return self._hid.is_connected()

    def ping(self):
        self._check()
        return self._hid.ping()

    def get_version(self):
        self._check()
        return self._hid.get_version()

    # --- Команды устройства (только владелец) ---

    def release_all(self):
        self._check()
        return self._hid.release_all()

    def move_rel(self, *args, **kwargs):
        self._check()
        return self._hid.move_rel(*args, **kwargs)

    def click(self, *args, **kwargs):
        self._check()
        return self._hid.click(*args, **kwargs)

    def mouse_down(self, *args, **kwargs):
        self._check()
        return self._hid.mouse_down(*args, **kwargs)

    def mouse_up(self, *args, **kwargs):
        self._check()
        return self._hid.mouse_up(*args, **kwargs)

    def traj(self, *args, **kwargs):
        self._check()
        return self._hid.traj(*args, **kwargs)

    def scroll(self, *args, **kwargs):
        self._check()
        return self._hid.scroll(*args, **kwargs)

    def key_press(self, *args, **kwargs):
        self._check()
        return self._hid.key_press(*args, **kwargs)

    def key_down(self, *args, **kwargs):
        self._check()
        return self._hid.key_down(*args, **kwargs)

    def key_up(self, *args, **kwargs):
        self._check()
        return self._hid.key_up(*args, **kwargs)

    def key_combo(self, *args, **kwargs):
        self._check()
        return self._hid.key_combo(*args, **kwargs)

    def type_char(self, *args, **kwargs):
        self._check()
        return self._hid.type_char(*args, **kwargs)
