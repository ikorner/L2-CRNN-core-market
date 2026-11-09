"""Транзакция кадра (задача 3.6, design.md решение 18).

Спека action-primitives («Кэш кадра в рамках транзакции»): детектор SHALL
анализировать один и тот же кадр не более одного раза внутри шага
(grab → analyze once → решения по свежему кадру); перед необратимым
действием SHALL выполняться свежий grab и re-verify цели.

FrameTransaction — явный контекстный менеджер поверх Screen (адаптеры 3.1):

- внутри контекста frame возвращает один и тот же кадр: первый grab
  кэшируется, повторные обращения анализируют тот же массив (analyze once);
- refresh() — инвалидация кэша и свежий grab: решения после него идут по
  новому кадру;
- reverify(predicate) — свежий grab + проверка predicate по новому кадру:
  точка входа «re-verify перед необратимым» (решение за стратегией при
  False — инвариант 4; исполнитель — примитивы 5.x);
- кэш привязан к идентичности кадра (сам массив), не ко времени: живости
  кадра между grab'ами транзакция не предполагает, свежесть — только через
  refresh()/reverify();
- вложенный вход — RuntimeError (общий кэш без явного договора запрещён);
  повторный вход после выхода — свежая транзакция; обращение к frame/
  refresh/reverify вне контекста — RuntimeError.

Граница: это инфраструктура для примитивов 5.x — здесь нет ни примитивов,
ни выбора решений по результату reverify.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from l2core.adapters.screen import Screen


class FrameTransaction:
    """Транзакция кадра: один grab анализируется один раз (см. модуль)."""

    def __init__(self, screen: Screen):
        self._screen = screen
        self._frame: Optional[np.ndarray] = None
        self._active = False

    def __enter__(self) -> "FrameTransaction":
        # Защита от вложенности — на стороне экрана: общий кэш между
        # двумя объектами-транзакциями без явного договора запрещён.
        if getattr(self._screen, "_l2core_frame_tx_active", False):
            raise RuntimeError(
                "вложенная транзакция кадра запрещена: завершите текущую")
        self._screen._l2core_frame_tx_active = True
        self._active = True
        self._frame = None
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._active = False
        self._frame = None
        self._screen._l2core_frame_tx_active = False

    def _require_active(self) -> None:
        if not self._active:
            raise RuntimeError(
                "транзакция кадра не активна: используйте контекстный менеджер")

    @property
    def frame(self) -> np.ndarray:
        """Кадр транзакции: grab выполняется не более одного раза."""
        self._require_active()
        if self._frame is None:
            self._frame = self._screen.grab()
        return self._frame

    def refresh(self) -> np.ndarray:
        """Инвалидация кэша + свежий grab (возвращает новый кадр)."""
        self._require_active()
        self._frame = self._screen.grab()
        return self._frame

    def reverify(self, predicate: Callable[[np.ndarray], bool]) -> bool:
        """Re-verify перед необратимым: свежий grab, проверка predicate.

        Кэш заменяется свежим кадром: дальнейшие анализы в транзакции идут
        уже по нему.
        """
        self._require_active()
        return bool(predicate(self.refresh()))
