"""Dry-run HID-адаптер (задача 8.2, spec safety «Dry-run режим без HID»).

DryRunHid — самостоятельная реализация командного интерфейса HidDevice
БЕЗ железа и без инъекции реального устройства: каждое действие
структурированно фиксируется как намерение (logging + список intentions
в памяти), к порту arduino обращений нет. Экран при этом читается и
детекция работает по-настоящему — verify проверяет реальный кадр, эффекта
«клика» нет, поэтому результаты примитивов честно уходят в FAILED/UNKNOWN.
Семантика для первых запусков на живой игре: детекция работает, действия
бы не сработали — намерения смотрим в логе.

Thread-owner дисциплина здесь НЕ дублируется: DryRunHid — устройство, а
guard владения — OwnerGuardedHid (7.2), который L2CoreBot применяет к
wiring.hid независимо от режима (проверка владельца — одна точка;
санкционированный release_all при kill switch 8.1 сохраняется).

Запросы состояния (is_connected/ping/get_version) — не действия: в журнал
намерений не пишутся, адаптер отчитывается «живым» (True / (0, 0)) — в
dry-run HID-путь доступен по построению, health() работает без изменений;
факт режима — wiring.dry_run и лог намерений, а не ложная диагностика
сбоя железа. connect/close — логируемые no-op (устройство виртуально,
сессия — жизнь объекта).

Место в слоях: адаптер живёт в l2core/api (рядом с OwnerGuardedHid), а не
в l2core/adapters — это часть сборки API (как guard 7.2), а не драйвер
устройства; живой HID заменяется на этапе create_api.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

__all__ = ["DryRunHid", "DryRunIntention"]

logger = logging.getLogger("l2core.dry_run")


@dataclass(frozen=True)
class DryRunIntention:
    """Намерение действия в dry-run: что было бы отправлено на железо."""

    action: str            # имя команды (click/move_rel/key_press/...)
    args: tuple            # позиционные аргументы вызова
    kwargs: dict           # именованные аргументы вызова
    seq: int = 0           # порядковый номер в журнале адаптера


class DryRunHid:
    """No-op реализация HID командного интерфейса с журналом намерений.

    Все действия (движение/клики/клавиши/скролл/траектории/release_all/
    connect/close) записываются в intentions и в лог l2core.dry_run,
    к реальному HID нет обращений и инъекция его не требуется. Возвраты —
    как у успешного no-op исполнения (None), чтобы примитивы проходили
    свой штатный путь до verify. Метаданные: TRAJ_MAX_POINTS — паритет
    с вендором (drag 5.6 проверяет ёмкость пакета); port_name/version —
    читаемые метки виртуального устройства.
    """

    TRAJ_MAX_POINTS = 60   # ёмкость пакета CMD_TRAJ (паритет вендора)

    def __init__(self, *, intentions: Optional[list] = None):
        self.intentions: list[DryRunIntention] = (
            intentions if intentions is not None else [])
        self._seq = 0
        # Метки виртуального устройства (читаются guard'ом как метаданные).
        self.port_name = "DRY-RUN"
        self.version = (0, 0)

    def _record(self, action: str, args: tuple, kwargs: dict) -> None:
        """Зафиксировать намерение: журнал в памяти + структурированный лог."""
        self._seq += 1
        intention = DryRunIntention(action=action, args=tuple(args),
                                    kwargs=dict(kwargs), seq=self._seq)
        self.intentions.append(intention)
        logger.info("dry-run намерение #%s: %s args=%s kwargs=%s",
                    intention.seq, action, intention.args, intention.kwargs)

    # --- Жизненный цикл (логируемые no-op) ---

    def connect(self, *args, **kwargs) -> None:
        self._record("connect", args, kwargs)

    def close(self, *args, **kwargs) -> None:
        self._record("close", args, kwargs)

    # --- Запросы состояния (не действия, в журнал не пишутся) ---

    def is_connected(self) -> bool:
        return True

    def ping(self) -> bool:
        return True

    def get_version(self) -> tuple[int, int]:
        return self.version

    # --- Команды устройства (намерения, без железа) ---
    # Сигнатуры — *args/**kwargs: намерение фиксирует аргументы ровно так,
    # как их передал примитив (паритет вызова, а не нормализация).

    def release_all(self, *args, **kwargs) -> None:
        self._record("release_all", args, kwargs)

    def move_rel(self, *args, **kwargs) -> None:
        self._record("move_rel", args, kwargs)

    def click(self, *args, **kwargs) -> None:
        self._record("click", args, kwargs)

    def mouse_down(self, *args, **kwargs) -> None:
        self._record("mouse_down", args, kwargs)

    def mouse_up(self, *args, **kwargs) -> None:
        self._record("mouse_up", args, kwargs)

    def traj(self, points, *args, **kwargs) -> None:
        # Паритет валидации вендора: примитивы полагаются на тот же лимит
        # ёмкости пакета (drag 5.6 fail-fast до HID).
        if not 1 <= len(points) <= self.TRAJ_MAX_POINTS:
            raise ValueError(f"точек должно быть 1..{self.TRAJ_MAX_POINTS}")
        self._record("traj", (tuple(points),) + args, kwargs)

    def scroll(self, *args, **kwargs) -> None:
        self._record("scroll", args, kwargs)

    def key_press(self, *args, **kwargs) -> None:
        self._record("key_press", args, kwargs)

    def key_down(self, *args, **kwargs) -> None:
        self._record("key_down", args, kwargs)

    def key_up(self, *args, **kwargs) -> None:
        self._record("key_up", args, kwargs)

    def key_combo(self, *args, **kwargs) -> None:
        self._record("key_combo", args, kwargs)

    def type_char(self, *args, **kwargs) -> None:
        self._record("type_char", args, kwargs)
