"""Статусы окна торговли и признак прогрузки контента (задача 3.5).

Спека window-state-detection: статусы `not_found` (закрыто/нет на экране),
`occluded_or_unstable` (якорь мерцает / перекрыто игровым окном или тултипом)
и `found` (шаблон совпал), плюс признак «контент прогрузился» по контентным
сигнатурам, а не только по заголовку. design.md, решение 10.

Модель статуса (check() — постановщик решения по кадру):

- окно игры недоступно (находилка 3.2, причина в диагностике) -> not_found;
- якоря нет на кадре, истории о нём нет или он отсутствовал и на прошлом
  кадре -> not_found (окно торговли закрыто/не открыто);
- результаты детекции якоря РАЗЛИЧАЮТСЯ между двумя кадрами подряд ->
  occluded_or_unstable: `anchor_lost` (был — пропал), `anchor_appeared`
  (не было — появился), `anchor_moved` (позиция прыгнула сверх допуска).
  Это честный минимум для перекрытия тултипом/окном: из одного кадра
  перекрытие не отличить от закрытия, нестабильность между кадрами —
  отличимо (сценарий «Мерцающий якорь»);
- якорь найден и совпадает с предыдущим кадром в пределах допуска -> found;
  первый кадр без истории — found (нестабильность недоказуема).

Граница инварианта «кэша позиции между кадрами нет» (anchor.py): поиск якоря
выполняется на каждом кадре свежо, это состояние НЕ используется для пропуска
поиска. Внутреннее состояние компонента — факт результата предыдущего кадра
(позиция для сравнения стабильности): монитору история разрешена
(design.md, решение 8), статус-детектор — монитор, а не исполнитель действий.

Признак «контент прогрузился»: делегируется StateDetector.check_content()
(сигнатуры состояний, кроме опор якоря — заголовок/целое окно не контент).
Проверяется только при статусе found: без устойчивого якоря оценивать
контентные сигнатуры бессмысленно. Детектор состояний опционален: без него
content_loaded=None (деградация с диагностикой, не исключение).

Перекрытие из одного кадра (тултип поверх заголовка и т.п.) надежного
признака не имеет — не детектируем, честный минимум — нестабильность между
кадрами (см. отчёт 3.5).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

from l2core.adapters.win32 import GameWindow
from l2core.detector.anchor import DEFAULT_ANCHOR_TEMPLATES, Anchor
from l2core.detector.states import StateDetector

logger = logging.getLogger(__name__)

# Статусы окна (спека window-state-detection).
STATUS_NOT_FOUND = "not_found"
STATUS_OCCLUDED = "occluded_or_unstable"
STATUS_FOUND = "found"

# Причины not_found, порождённые этим компонентом (окно игры живо).
REASON_ANCHOR_NOT_FOUND = "anchor_not_found"


@runtime_checkable
class GameWindowSource(Protocol):
    """Источник состояния окна игры: GameWindowFinder ему удовлетворяет."""

    def find(self) -> GameWindow: ...


@runtime_checkable
class AnchorSource(Protocol):
    """Источник якоря: AnchorDetector ему удовлетворяет."""

    def find_anchor(self, frame: np.ndarray, window_id: str) -> Optional[Anchor]: ...


@dataclass(frozen=True)
class WindowStatus:
    """Статус окна торговли на кадре.

    status — one of STATUS_*; reason — диагностическая причина (None при
    found); anchor — найденный якорь (None при not_found/occluded без якоря);
    content_loaded — признак прогрузки контента (None — не проверялся/нечем);
    content_detail — какие контентные сигнатуры совпали/отсутствуют.
    """

    status: str
    reason: Optional[str]
    anchor: Optional[Anchor]
    content_loaded: Optional[bool]
    content_detail: str


class WindowStatusDetector:
    """Постановщик статуса окна: находилка окна игры + детектор якоря
    + (опционально) детектор состояний для признака прогрузки контента.

    Args:
        game_finder: источник состояния окна игры (GameWindowFinder 3.2).
        anchor_detector: детектор якоря (AnchorDetector 3.3).
        state_detector: опциональный StateDetector (3.4) для проверки
            контентных сигнатур; без него content_loaded=None.
        anchor_templates: окно -> (element_id, variant) опоры якоря —
            элементы-заголовки исключаются из контентных сигнатур. Должен
            совпадать с конфигурацией anchor_detector (по умолчанию —
            DEFAULT_ANCHOR_TEMPLATES обоих компонентов).
        position_tolerance: допуск скачка позиции якоря между кадрами, px
            по каждой оси (допущение: точность template matching — целые
            пиксели, наблюдаемый джиттер 0; запас на рендер-шум живой игры).
    """

    def __init__(
        self,
        game_finder: GameWindowSource,
        anchor_detector: AnchorSource,
        *,
        state_detector: Optional[StateDetector] = None,
        anchor_templates: Optional[Mapping[str, Tuple[str, str]]] = None,
        position_tolerance: int = 8,
    ):
        self._game_finder = game_finder
        self._anchor_detector = anchor_detector
        self._state_detector = state_detector
        self._anchor_templates = dict(anchor_templates or DEFAULT_ANCHOR_TEMPLATES)
        self._position_tolerance = position_tolerance
        self._prev_anchor: Optional[Anchor] = None
        self._has_prev = False
        self._prev_window_id: Optional[str] = None

    def reset(self) -> None:
        """Сброс истории стабильности (смена контекста/сессии)."""
        self._prev_anchor = None
        self._has_prev = False

    def check(self, frame: np.ndarray, window_id: str) -> WindowStatus:
        """Статус окна торговли на кадре (поиск якоря — свежий)."""
        if window_id != self._prev_window_id:
            self.reset()
            self._prev_window_id = window_id

        game = self._game_finder.find()
        if not game.ok:
            # Окна игры нет — позиционная история бессмысленна.
            self.reset()
            return WindowStatus(
                status=STATUS_NOT_FOUND,
                reason=game.reason,
                anchor=None,
                content_loaded=None,
                content_detail="окно игры недоступно",
            )

        anchor = self._anchor_detector.find_anchor(frame, window_id)
        status, reason = self._decide(anchor)
        self._prev_anchor = anchor
        self._has_prev = True

        content_loaded: Optional[bool] = None
        content_detail = "статус не found — прогрузка контента не проверялась"
        if status == STATUS_FOUND:
            content_loaded, content_detail = self._check_content(frame, anchor)

        if status == STATUS_OCCLUDED:
            logger.warning(
                "окно %s: %s (%s)", window_id, STATUS_OCCLUDED, reason)
        return WindowStatus(
            status=status,
            reason=reason,
            anchor=anchor,
            content_loaded=content_loaded,
            content_detail=content_detail,
        )

    def _decide(self, anchor: Optional[Anchor]) -> Tuple[str, Optional[str]]:
        """Статус по результату детекции якоря и истории предыдущего кадра."""
        if anchor is None:
            if not self._has_prev or self._prev_anchor is None:
                return STATUS_NOT_FOUND, (
                    f"{REASON_ANCHOR_NOT_FOUND}: окно торговли не найдено "
                    f"на кадре (устойчивое отсутствие)")
            prev = self._prev_anchor
            return STATUS_OCCLUDED, (
                f"anchor_lost: якорь был на предыдущем кадре "
                f"({prev.x},{prev.y}), на текущем не найден")

        if not self._has_prev:
            return STATUS_FOUND, None
        if self._prev_anchor is None:
            return STATUS_OCCLUDED, (
                "anchor_appeared: на предыдущем кадре якоря не было, "
                "на текущем появился")

        prev = self._prev_anchor
        dx, dy = anchor.x - prev.x, anchor.y - prev.y
        if abs(dx) <= self._position_tolerance and abs(dy) <= self._position_tolerance:
            return STATUS_FOUND, None
        return STATUS_OCCLUDED, (
            f"anchor_moved: скачок позиции якоря на ({dx},{dy}) "
            f"при допуске {self._position_tolerance}")

    def _check_content(self, frame: np.ndarray,
                       anchor: Anchor) -> Tuple[Optional[bool], str]:
        if self._state_detector is None:
            return None, "нет детектора состояний: прогрузку проверить нечем"
        exclude = set()
        spec = self._anchor_templates.get(anchor.window_id)
        if spec is not None:
            exclude.add(spec[0])
        report = self._state_detector.check_content(
            frame, anchor, exclude_elements=tuple(sorted(exclude)))
        if report.loaded is None:
            return None, (
                "контентных сигнатур в модели нет: прогрузку проверить нечем")
        if report.loaded:
            names = ", ".join(r.name for r in report.matched)
            return True, f"контентные сигнатуры совпали: {names}"
        names = ", ".join(r.name for r in report.failed)
        return False, f"контентные сигнатуры отсутствуют: {names}"
