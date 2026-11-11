"""Абстракция Screen и адаптеры: MssScreen, ReplayScreen, RecordingScreen, FakeScreen.

Screen — минимальный протокол слоя, через который ядро работает с экраном:

- grab() -> numpy.ndarray (BGR uint8) — текущий кадр;
- notify_action(action) — уведомление о выполненном действии ядра
  (примитивы, 5.x); ReplayScreen по нему продвигает запись, декоратор
  RecordingScreen — пишет действие в сессию;
- record_result(status, detail) — verify-результат последнего действия;
- close() — освобождение ресурсов (контекстный менеджер поддержан).

Реализации:
- MssScreen — реальная игра через mss (ленивый импорт/захват; источник
  кадров можно инъектировать grabber'ом — тесты и CI без монитора);
- ReplayScreen — воспроизведение записанной сессии (формат — session.py),
  без обращений к mss, HID и игре; «эффект действия» = переход на кадр,
  записанный после действия;
- RecordingScreen — декоратор: оборачивает любой Screen и пишет каждый
  кадр, действие и результат в файл сессии (живые прогоны → replay-тесты);
- FakeScreen — скриптованная последовательность кадров для юнит-тестов
  примитивов (каждое действие — следующий кадр скрипта).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from l2core.adapters.session import (
    SessionFormatError,
    SessionWriter,
    actions_equal,
    load_session,
    normalize_action,
)


class ScreenError(Exception):
    """Базовое исключение слоя Screen."""


class MssError(ScreenError):
    """Захват экрана недоступен (mss не установлен, нет мониторов и т.п.)."""


class SessionExhaustedError(ScreenError):
    """Replay: сценарий требует действия/кадра за пределами записи."""


class ActionMismatchError(ScreenError):
    """Replay: действие сценария не совпало с записанным в сессии."""

    def __init__(self, expected, actual):
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"действие не совпало с записью: ожидалось {expected!r}, получено {actual!r}"
        )


class ScreenClosedError(ScreenError):
    """Обращение к закрытому экрану."""


class Screen(ABC):
    """Минимальный протокол экрана (см. докстринг модуля)."""

    @abstractmethod
    def grab(self) -> np.ndarray:
        """Текущий кадр (numpy BGR uint8)."""

    def notify_action(self, action: dict) -> None:
        """Уведомить экран о выполненном действии ядра (по умолчанию no-op)."""

    def record_result(self, status: str, detail: dict | None = None) -> None:
        """Привязать verify-результат к последнему действию (по умолчанию no-op)."""

    def close(self) -> None:
        """Освободить ресурсы (по умолчанию no-op)."""

    def __enter__(self) -> "Screen":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _bgra_to_bgr(frame: np.ndarray) -> np.ndarray:
    """Кадр mss (BGRA) -> BGR, непрерывный массив."""
    return np.ascontiguousarray(frame[:, :, :3])


class MssScreen(Screen):
    """Реальный экран через mss.

    Создание объекта не требует mss и монитора: захват ленивый, происходит
    при первом grab() без инъектированного grabber'а. grabber — callable,
    возвращающий кадр BGR uint8; инъекция изолирует недетерминированное
    (живой экран) в тестах и CI.

    mss.MSS() не потокобезопасен: конкурентные grab() движка (поток
    стратегии + observer, 7.2) ломали захват (битый кадр/BitBlt) — экземпляр
    mss thread-local, каждый поток получает свой (2026-11-12).
    """

    def __init__(self, monitor: int = 1, grabber=None):
        """
        Args:
            monitor: индекс монитора в mss (1 — первый, 0 — все вместе).
            grabber: источник кадров для тестов (None — реальный mss-захват).
        """
        self._monitor = monitor
        self._grabber = grabber
        import threading
        self._local = threading.local()

    def grab(self) -> np.ndarray:
        if self._grabber is not None:
            return self._grabber()
        return _bgra_to_bgr(np.asarray(self._capture()))

    def _capture(self):
        sct = getattr(self._local, "sct", None)
        if sct is None:
            try:
                import mss
            except ImportError as e:
                raise MssError(
                    "mss недоступен: pip install mss (или передайте grabber)"
                ) from e
            try:
                sct = self._local.sct = mss.MSS()
            except Exception as e:
                raise MssError(f"не удалось инициализировать захват экрана: {e}") from e
        try:
            return sct.grab(sct.monitors[self._monitor])
        except Exception as e:
            raise MssError(f"не удалось захватить монитор {self._monitor}: {e}") from e

    def close(self) -> None:
        sct = getattr(self._local, "sct", None)
        if sct is not None:
            sct.close()
            self._local.sct = None


class FakeScreen(Screen):
    """Скриптованная последовательность кадров для юнит-тестов примитивов.

    grab возвращает текущий кадр скрипта; notify_action переводит на
    следующий (модель «действие → следующий кадр»). Исчерпание скрипта —
    SessionExhaustedError.
    """

    def __init__(self, frames: list[np.ndarray]):
        if not frames:
            raise ValueError("FakeScreen требует хотя бы один кадр")
        self._frames = list(frames)
        self._pos = 0

    def grab(self) -> np.ndarray:
        return self._frames[self._pos]

    def notify_action(self, action: dict) -> None:
        if self._pos + 1 >= len(self._frames):
            raise SessionExhaustedError(
                f"FakeScreen: кадров больше нет (позиция {self._pos}, "
                f"всего {len(self._frames)})"
            )
        self._pos += 1


class ReplayScreen(Screen):
    """Воспроизведение записанной сессии без mss, HID и игры.

    Курсор стартует на первом frame-событии. grab отдаёт кадр текущей
    позиции (повторные grab не продвигают запись). notify_action ищет
    следующее action-событие, проверяет совпадение действия
    (ActionMismatchError при расхождении сценария с записью) и переносит
    курсор на него: текущим становится записанный после действия кадр.
    Действий больше нет — SessionExhaustedError; grab после исчерпания
    возвращает последний кадр.
    """

    def __init__(self, session_path: str | Path):
        self._session = load_session(session_path)
        self._cursor = 0
        self._current = self._session.read_image(self._session.events[0]["image"])

    def grab(self) -> np.ndarray:
        if self._session is None:
            raise ScreenClosedError("ReplayScreen закрыт")
        return self._current

    def notify_action(self, action: dict) -> None:
        events = self._session.events
        for i in range(self._cursor + 1, len(events)):
            if events[i]["type"] != "action":
                continue
            expected = events[i]["action"]
            if not actions_equal(expected, action):
                raise ActionMismatchError(expected, normalize_action(action))
            self._cursor = i
            ref = events[i]["image"]
            if ref is not None:
                self._current = self._session.read_image(ref)
            return
        raise SessionExhaustedError(
            "сессия исчерпана: действий больше нет "
            f"(всего событий {len(events)}, курсор {self._cursor})"
        )

    def close(self) -> None:
        self._session = None


class RecordingScreen(Screen):
    """Декоратор: пишет кадры, действия и результаты живого прогона в сессию.

    Оборачивает любой Screen (обычно MssScreen). Семантика записи:

    - grab: кадр от внутреннего экрана; если с прошлого grab было действие —
      этот кадр фиксируется как «кадр после действия» (action-событие);
      если кадр изменился без действия — frame-событие; одинаковые кадры
      не дублируются;
    - notify_action: запоминает действие до следующего grab;
    - record_result: привязывает verify-результат к последнему действию;
    - close: сбрасывает висящее действие (image=null) и дозаписывает
      манифест; повторный close — no-op. close закрывает внутренний экран.
    """

    def __init__(self, inner: Screen, session_path: str | Path, meta: dict | None = None):
        self._inner = inner
        default_meta = {"source": "recording"}
        if meta:
            default_meta.update(meta)
        self._writer = SessionWriter(session_path, meta=default_meta)
        self._pending_action: dict | None = None
        self._last_recorded: np.ndarray | None = None
        self._closed = False

    def grab(self) -> np.ndarray:
        if self._closed:
            raise ScreenClosedError("RecordingScreen закрыт")
        frame = self._inner.grab()
        if self._pending_action is not None:
            self._writer.record_action(self._pending_action, frame_after=frame)
            self._pending_action = None
            self._last_recorded = frame
        elif self._last_recorded is None:
            self._writer.record_frame(frame)
            self._last_recorded = frame
        elif not np.array_equal(frame, self._last_recorded):
            self._writer.record_frame(frame)
            self._last_recorded = frame
        return frame

    def notify_action(self, action: dict) -> None:
        if self._closed:
            raise ScreenClosedError("RecordingScreen закрыт")
        # Сначала — внутреннему экрану (Replay/Fake продвигаются по действию);
        # если он отказал (например SessionExhaustedError), запись не начинается.
        self._inner.notify_action(action)
        if self._pending_action is not None:
            # Действие без следующего grab: эффект не зафиксирован.
            self._writer.record_action(self._pending_action, frame_after=None)
        self._pending_action = normalize_action(action)

    def record_result(self, status: str, detail: dict | None = None) -> None:
        if self._closed:
            raise ScreenClosedError("RecordingScreen закрыт")
        self._writer.set_result(status, detail)
        self._inner.record_result(status, detail)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._pending_action is not None:
            self._writer.record_action(self._pending_action, frame_after=None)
            self._pending_action = None
        self._writer.save()
        self._inner.close()
