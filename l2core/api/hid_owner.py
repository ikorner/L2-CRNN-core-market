"""HID-owner guard: единственный владелец HID — action executor (задача 7.2,
spec core-api-events: «action executor — единственный владелец HID» и
сценарий «Observer не отправляет действия»).

OwnerGuardedHid — guard-обёртка поверх HidDevice (ВЕНДОРЕННЫЙ
adapters/hid.py не трогаем — обёртка снаружи): поток-владелец фиксируется
при создании, каждая HID-команда проверяет threading.current_thread().
Вызов из чужого потока (state observer, event dispatcher, чужая стратегия)
— HidOwnershipError ДО обращения к железу: слепых/чужих эффектов нет
(инвариант 9 — enforcement, а не дисциплина).

ГРАНИЦА ОБХОДА (fail closed): обход проверки владельца невозможен через
публичную поверхность обёртки. (1) Публичный аксессор к сырому устройству
отсутствует — сырой адаптер хранится только под приватным именем `_hid`
(конвенция Python: доступ к нему снаружи класса — вне поддерживаемой
границы guard'а, как `obj._private` в любом модуле проекта). (2) Все
команды — собственные методы обёртки с проверкой `_check()`; `__getattr__`
пробрасывает ТОЛЬКО публичные не-вызываемые атрибуты (метаданные: version,
port_name, TRAJ_MAX_POINTS) — появившаяся у оборачиваемого объекта
команда вне явного списка получает AttributeError, а не обход. Паритет
списка команд с публичными методами вендорного HidDevice следит
guard-тест tests/test_engine.py::TestHidOwnerContract: новая команда
вендора не проскочит молча — расхождение падает с именем метода.

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
from typing import Callable, Optional

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

    def __init__(self, hid, owner_thread: Optional[threading.Thread] = None,
                 kill_switch: Optional[Callable[[], bool]] = None):
        self._hid = hid
        self._owner = owner_thread or threading.current_thread()
        # Санкционированный обход владельца для release_all при kill switch
        # (8.1): kill-слушатель живёт в своём потоке, а освобождать HID
        # при аварийной остановке обязан именно он. См. _check/release_all.
        self._kill_switch = kill_switch

    @property
    def owner_thread(self) -> threading.Thread:
        """Поток-владелец HID (action executor)."""
        return self._owner

    def _check(self, *, allow_kill: bool = False) -> None:
        """Проверка владения: чужой поток — HidOwnershipError до железа.

        allow_kill (только release_all, 8.1): kill switch сработал —
        владельца не проверяем. Обоснование (killswitch.py): аварийная
        остановка важнее владения; release_all идемпотентен и только
        отпускает кнопки — безопасен из любого потока; любая ДРУГАЯ
        команда из чужого потока при активном kill по-прежнему
        отклоняется (guard fail closed)."""
        if allow_kill and self._kill_switch is not None \
                and self._kill_switch():
            return
        current = threading.current_thread()
        if current is not self._owner:
            raise HidOwnershipError(
                f"HID-команда из чужого потока ({current.name!r}; владелец — "
                f"{self._owner.name!r}): action executor — единственный "
                f"владелец HID (spec core-api-events, сценарий «Observer не "
                f"отправляет действия»)")

    def __getattr__(self, name: str):
        # Fail closed: сюда попадают только атрибуты, отсутствующие в
        # явном списке обёрнутых команд выше. Команды (callable) и приватные
        # имена НЕ пробрасываются — иначе появившаяся у оборачиваемого
        # объекта команда обошла бы проверку владельца (__getattr__-
        # проскок), а `_hid` дал бы публичный люк к сырому устройству.
        # Доступны только публичные данные (version, port_name,
        # TRAJ_MAX_POINTS, ...) — чтение метаданных безопасно из любого
        # потока. Проверка startswith идёт до обращения к self._hid:
        # без неё при отсутствии атрибута была бы бесконечная рекурсия.
        if name.startswith("_"):
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r} "
                f"(приватные имена оборачиваемого устройства вне "
                f"поддерживаемой границы guard'а)")
        attribute = getattr(self._hid, name)
        if callable(attribute):
            raise AttributeError(
                f"{name}: команда не входит в явный список guard'а — "
                f"обход проверки владельца запрещён (guard-тест паритета "
                f"tests/test_engine.py::TestHidOwnerContract)")
        return attribute

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
        # Единственная команда с санкционированным обходом владельца:
        # при сработавшем kill switch (8.1) её выполняет слушатель из
        # своего потока (обоснование — в _check и killswitch.py).
        self._check(allow_kill=True)
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
