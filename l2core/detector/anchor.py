"""Детектор якоря окна торговли (задача 3.3).

Якорь — начало координат окна: все rect'ы элементов модели отсчитываются
от него 1:1 (ui_model/FORMAT.md, design.md решение 4). Детектор двухэтапный:

- «Обзор лавок» (market): numpy-only detect_window() — яркие линии рамки
  (первый дешёвый этап), кандидат подтверждается template matching'ом
  целого окна в окрестности кандидата; при отсутствии кандидата —
  полный поиск того же шаблона по кадру;
- лавки (shop_buy/shop_sell): template matching заголовка окна
  («Торговая лавка (...)», элементы jak/jakor_sell модели) по всему кадру.

Кэша позиции между кадрами нет (design.md, решение 4): объект детектора
не хранит состояния кадров, каждый find_anchor — свежий поиск.

Ограничение (сырьё): в состоянии «open» лавки заголовок другой
(«<Покупка>»/«<Продажа>»), шаблонов таких заголовков в модели нет —
детектор честно возвращает None (отчёт 3.3).

Пороги — замер на 39 скриншотах base_screenshot/scr/, числа и выбор
реализации matcher'а по умолчанию — в отчёте task-3.3-report.md и AGENTS.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from l2core.adapters.template import Match, TemplateMatcher
from l2core.model import UiModel

# Смещение ярких линий рамки относительно якоря модели: замер на обеих
# наблюдаемых позициях окна обзора — рамка (209, 79) при якоре (207, 77)
# и рамка (204, 105) при якоре (202, 103).
_FRAME_LINE_OFFSET = (2, 2)

# Элементы-опоры якоря по умолчанию: окно -> кортеж опор (element_id,
# variant); find_anchor перебирает опоры template matching'ом по порогу окна,
# первая сошедшаяся даёт Anchor. market подтверждается целым окном
# (market_main_menu), лавки — заголовком; у лавок две опоры: «opening»
# (jak / jakor_sell) и «open» (заголовок открытого окна — досъёмка
# menu-quickslot-navigation, развязка KNOWN_GAPS п. 6).
DEFAULT_ANCHOR_TEMPLATES = {
    "market": (("market_main_menu", "normal"),),
    "shop_buy": (("jak", "normal"), ("jak", "open")),
    "shop_sell": (("jakor_sell", "normal"), ("jakor_sell", "open")),
}

# Допустимые форматы anchor_templates: (element, variant) — одна опора
# (легаси-формат, нормализуется) или кортеж из нескольких опор.


def _normalize_supports(supports) -> Tuple[Tuple[str, str], ...]:
    """Нормализация опор окна: легаси (element, variant) -> 1-опорный
    кортеж; кортеж опор — как есть."""
    if (
        isinstance(supports, tuple)
        and len(supports) == 2
        and all(isinstance(part, str) for part in supports)
    ):
        return (supports,)
    return tuple(supports)

# Окна, для которых работает numpy-only первый этап (яркие линии рамки):
# рамка «Обзора лавок» даёт устойчивый признак, рамки лавок сырьё не покрывает.
DEFAULT_FRAME_STAGE = ("market",)


@dataclass(frozen=True)
class Anchor:
    """Якорь окна торговли на кадре.

    x, y — начало координат модели окна (anchor + rect элемента = позиция
    элемента на кадре); score — NCC подтверждающего совпадения; method —
    этап, давший якорь: «frame» (рамка + подтверждение) или «template»
    (полный поиск шаблона).
    """

    window_id: str
    x: int
    y: int
    score: float
    method: str


def _to_gray(frame: np.ndarray) -> np.ndarray:
    """BGR-кадр -> серый (веса BT.601, как в OpenCV), float64."""
    return frame[..., 0] * 0.114 + frame[..., 1] * 0.587 + frame[..., 2] * 0.299


def detect_window(
    frame: np.ndarray,
    window_size: Tuple[int, int],
    *,
    line_brightness: int = 140,
    vertical_brightness: int = 120,
    length_tolerance: int = 4,
    min_vertical_fraction: float = 0.7,
) -> Optional[Tuple[int, int]]:
    """numpy-only поиск рамки окна по ярким линиям (первый этап обзора).

    Ищет горизонтальную яркую линию длины «ширина окна − 4» (замеренная
    длина рамки обзора) и пару вертикальных линий по её концам; возвращает
    якорь окна (начало координат модели) или None.

    Пороги (происхождение — замер 3.3 на 39 скриншотах):
    - line_brightness=140: верхняя линия рамки mean 161, фон вокруг <= 120;
    - vertical_brightness=120: боковые линии рамки mean ~130;
    - length_tolerance=4: рамка ровно 1004 px на обеих позициях окна;
      ложные длинные линии (995 — панель задач, 973 — шапка таблицы,
      1900+ — рамки игрового клиента) вне допуска;
    - min_vertical_fraction=0.7: доля ярких пикселей вертикали (замер
      0.99, запас на перекрытия).

    Args:
        frame: кадр BGR uint8.
        window_size: (width, height) окна в референсе (модель window.size).
    """
    width, height = window_size
    gray = _to_gray(frame)
    frame_h, frame_w = gray.shape
    if frame_h < height or frame_w < width:
        return None
    bright = gray > line_brightness
    # Замер: длина верхней линии рамки = width - 4 (1004 при 1008).
    target = width - 4
    low = target - length_tolerance
    high = target + length_tolerance

    rows = np.where(bright.sum(axis=1) >= low)[0]
    vertical_check = height - 8  # интервал вертикали без углов
    for y in rows:
        if y + height > frame_h:
            continue
        row = bright[y]
        edges = np.diff(row.astype(np.int8), prepend=0, append=0)
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0]
        for start, end in zip(starts, ends):
            line_len = end - start
            if not (low <= line_len <= high):
                continue
            left, right = start, end - 1
            left_frac = (
                gray[y + 4 : y + height - 4, left] > vertical_brightness
            ).mean()
            if left_frac < min_vertical_fraction:
                continue
            right_frac = (
                gray[y + 4 : y + height - 4, right] > vertical_brightness
            ).mean()
            if right_frac < min_vertical_fraction:
                continue
            return (int(left - _FRAME_LINE_OFFSET[0]), int(y - _FRAME_LINE_OFFSET[1]))
    return None


class AnchorDetector:
    """Двухэтапный детектор якоря окон торговли (см. докстринг модуля).

    Пороги template matching (происхождение — замер 3.3):
    - market_threshold=0.18: min позитив 0.234 (окно со сдвинутым контентом),
      max негатив 0.083 (окна нет); подтверждение рамки и полный поиск;
    - shop_threshold=0.9: позитивы 1.000; максимум чужого окна 0.743
      (заголовок другой лавки отличается одним словом), обычные негативы
      <= 0.416 (окно обзора, пустой экран).
    """

    def __init__(
        self,
        models: Mapping[str, UiModel],
        matcher: TemplateMatcher,
        assets_dir: Path,
        *,
        anchor_templates: Optional[Mapping[str, Tuple[str, str]]] = None,
        frame_stage: Sequence[str] = DEFAULT_FRAME_STAGE,
        market_threshold: float = 0.18,
        shop_threshold: float = 0.9,
        search_radius: int = 8,
    ):
        """
        Args:
            models: модели окон (id -> UiModel).
            matcher: реализация template matching (см. l2core.adapters.template).
            assets_dir: каталог assets/ с шаблонами модели.
            anchor_templates: окно -> опоры якоря: (element_id, variant) —
                одна опора или кортеж опор; по умолчанию
                DEFAULT_ANCHOR_TEMPLATES (перебор по порогу окна, первая
                сошедшаяся опора даёт Anchor).
            frame_stage: окна, для которых выполняется первый этап
                (numpy-only рамка) перед template matching.
            market_threshold / shop_threshold: пороги NCC.
            search_radius: окрестность кандидата рамки для подтверждения.
        """
        self._models = dict(models)
        self._matcher = matcher
        self._assets_dir = Path(assets_dir)
        self._anchor_templates = {
            window_id: _normalize_supports(supports)
            for window_id, supports in dict(
                anchor_templates or DEFAULT_ANCHOR_TEMPLATES).items()
        }
        self._frame_stage = tuple(frame_stage)
        self._market_threshold = market_threshold
        self._shop_threshold = shop_threshold
        self._search_radius = search_radius
        self._templates = self._load_templates()

    @classmethod
    def for_windows(cls, models, matcher, assets_dir) -> "AnchorDetector":
        """Детектор с опорами якоря по умолчанию (DEFAULT_ANCHOR_TEMPLATES)."""
        return cls(models, matcher, assets_dir)

    def _load_templates(self) -> dict:
        import cv2

        templates = {}
        for window_id, supports in self._anchor_templates.items():
            model = self._models[window_id]
            loaded = []
            for element_id, variant in supports:
                element = model.elements[element_id]
                spec = element.variants[variant]
                path = self._assets_dir / spec.template
                image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise FileNotFoundError(f"шаблон якоря не найден: {path}")
                loaded.append(
                    (np.asarray(image, dtype=np.float64), element))
            templates[window_id] = loaded
        return templates

    def find_anchor(self, frame: np.ndarray, window_id: str) -> Optional[Anchor]:
        """Свежий поиск якоря окна на кадре (без кэша между кадрами).

        Возвращает Anchor или None (окно не найдено/не подтверждено).
        Опоры перебираются по порогу окна; первая сошедшаяся даёт Anchor.
        """
        if window_id not in self._templates:
            raise KeyError(f"неизвестное окно торговли: {window_id!r}")
        supports = self._templates[window_id]
        gray = _to_gray(frame)
        if window_id in self._frame_stage:
            # Первый этап market: кандидат рамки подтверждается первой опорой.
            template, _element = supports[0]
            candidate = detect_window(frame, self._models[window_id].window.size)
            if candidate is not None:
                match = self._match_near(gray, template, candidate)
                if match.score >= self._market_threshold:
                    return Anchor(window_id, candidate[0], candidate[1],
                                  match.score, "frame")
        threshold = (self._market_threshold
                     if window_id in self._frame_stage
                     else self._shop_threshold)
        for template, element in supports:
            match = self._matcher.match(gray, template)
            if match.score < threshold:
                continue
            rect = element.rect
            offset_x, offset_y = (rect.x, rect.y) if rect is not None else (0, 0)
            return Anchor(window_id, match.x - offset_x, match.y - offset_y,
                          match.score, "template")
        return None

    def _match_near(self, gray: np.ndarray, template: np.ndarray,
                    anchor: Tuple[int, int]):
        """Подтверждение кандидата рамки: поиск шаблона в окрестности."""
        radius = self._search_radius
        h, w = template.shape
        frame_h, frame_w = gray.shape
        x, y = anchor
        x0, y0 = max(0, x - radius), max(0, y - radius)
        x1, y1 = min(frame_w, x + w + radius), min(frame_h, y + h + radius)
        region = gray[y0:y1, x0:x1]
        if region.shape[0] < h or region.shape[1] < w:
            return Match(x=0, y=0, score=0.0)
        match = self._matcher.match(region, template)
        # score без изменений; позиция в локальной системе не нужна
        return match
