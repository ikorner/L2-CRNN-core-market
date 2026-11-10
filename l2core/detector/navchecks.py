"""Оценщик проверок навигационной модели (menu-quickslot-navigation).

build_nav_evaluator() возвращает EvaluateChecks (l2core.primitives.key):
оценку набора проверок (сигнатур menu_open, verify/when/skip_if шагов
рецептов) по кадру БЕЗ якоря окна торговли — навигационные признаки живут
в клиентской области игры, а не в окне (меню закрыто = якоря окна нет).

Семантика (как у сигнатур состояний 3.4): passed — совпали ВСЕ проверки.

- template, HUD-элемент (hud: true, search_area): template matching
  шаблона варианта (первого, если variant не задан) по search_area,
  порог nav_threshold;
- template, элемент меню (rect): rect задан 1:1 от якоря меню — якорь
  (HUD-элемент menu_anchor) локализуется тем же matching'ом по его
  search_area, затем шаблон ищется в области anchor + rect;
- expect: absent (EXPECT_ABSENT, types.py) инвертирует итог проверки:
  элемент обязан ОТСУТСТВОВАТЬ (нормализация «Меню»: X — toggle);
- pixel_probe: точки от якоря меню (как точки сигнатур от якоря окна,
  инвариант 3); цвет модели в RGB, кадр BGR, допуск на канал
  probe_tolerance (допущение, как color_tolerance детектора состояний).

Провал локации якоря меню проваливает ВСЕ проверки элементов меню
(причина в failures); для expect: absent отсутствие якоря — штатный
«элемент не найден» (проверка отрицания сошлась для элементов меню и
провалена для самого якоря — оценивается честно, без якоря позиции
rect-элементов на кадре неизвестны).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from l2core.adapters.template import TemplateMatcher
from l2core.model.navigation import NavigationModel
from l2core.model.types import EXPECT_ABSENT, Check
from l2core.primitives.click import CheckReport
from l2core.primitives.key import EvaluateChecks

__all__ = [
    "MENU_ANCHOR_ID",
    "build_nav_evaluator",
]

# Соглашение id: HUD-якорь открытого «Меню» (элементы меню — 1:1 от него).
MENU_ANCHOR_ID = "menu_anchor"


def _to_gray(frame: np.ndarray) -> np.ndarray:
    """BGR-кадр -> серый (веса BT.601, как в OpenCV и l2core.detector.anchor)."""
    return frame[..., 0] * 0.114 + frame[..., 1] * 0.587 + frame[..., 2] * 0.299


def _load_templates(nav: NavigationModel, assets_dir) -> Dict[Tuple[str, str], np.ndarray]:
    import cv2

    assets = Path(assets_dir)
    templates = {}
    for element_id, element in nav.elements.items():
        for variant_name, spec in element.variants.items():
            path = assets / spec.template
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(f"шаблон навигации не найден: {path}")
            templates[(element_id, variant_name)] = np.asarray(image,
                                                               dtype=np.float64)
    return templates


def build_nav_evaluator(nav: NavigationModel, matcher: TemplateMatcher,
                        assets_dir, *, nav_threshold: float = 0.9,
                        probe_tolerance: int = 16) -> EvaluateChecks:
    """Собрать оценщик проверок навигационной модели (см. docstring модуля).

    nav_threshold — порог NCC template matching'а (по умолчанию 0.9, как
    shop_threshold детектора якоря 3.3); probe_tolerance — допуск
    пиксель-пробы на канал (допущение, как color_tolerance состояний 3.4).
    """
    templates = _load_templates(nav, assets_dir)

    def locate(element_id: str, variant: Optional[str],
               gray: np.ndarray) -> Optional[Tuple[int, int, float]]:
        """Абсолютная позиция элемента (левый верхний угол шаблона) или None."""
        element = nav.elements.get(element_id)
        if element is None or element.search_area is None:
            return None
        key = (element_id, variant or next(iter(element.variants), None))
        template = templates.get(key)
        if template is None:
            return None
        area = element.search_area
        frame_h, frame_w = gray.shape
        if (area.y < 0 or area.x < 0
                or area.y + area.height > frame_h
                or area.x + area.width > frame_w):
            return None
        crop = gray[area.y:area.y + area.height, area.x:area.x + area.width]
        h, w = template.shape
        if crop.shape[0] < h or crop.shape[1] < w:
            return None
        match = matcher.match(crop, template)
        if match.score < nav_threshold:
            return None
        return (area.x + match.x, area.y + match.y, match.score)

    def evaluate(checks: Tuple[Check, ...],
                 frame: np.ndarray) -> CheckReport:
        gray = _to_gray(np.asarray(frame))
        # Якорь меню локализуется лениво и один раз на набор проверок.
        anchor_pos: Optional[Tuple[int, int]] = None
        anchor_tried = False

        def menu_anchor() -> Optional[Tuple[int, int]]:
            nonlocal anchor_pos, anchor_tried
            if not anchor_tried:
                anchor_tried = True
                found = locate(MENU_ANCHOR_ID, None, gray)
                anchor_pos = (found[0], found[1]) if found else None
            return anchor_pos

        failures = []
        for check in checks:
            negate = check.kind == "template" and check.expect == EXPECT_ABSENT
            if check.kind == "template":
                element = nav.elements.get(check.element)
                if element is None:
                    failures.append(f"неизвестный элемент {check.element!r}")
                    continue
                found = False
                if element.hud:
                    found = locate(check.element, check.variant, gray) is not None
                else:
                    anchor = menu_anchor()
                    if anchor is None:
                        failures.append(
                            f"элемент {check.element!r}: якорь меню "
                            f"({MENU_ANCHOR_ID!r}) не локализован — позиции "
                            "элементов меню на кадре неизвестны")
                        continue
                    rect = element.rect
                    template = templates.get(
                        (check.element,
                         check.variant or next(iter(element.variants), None)))
                    if rect is None or template is None:
                        failures.append(
                            f"элемент {check.element!r}: rect или шаблон "
                            "не заданы")
                        continue
                    x0, y0 = anchor[0] + rect.x, anchor[1] + rect.y
                    h, w = template.shape
                    crop = gray[y0:y0 + rect.height, x0:x0 + rect.width]
                    if crop.shape[0] < h or crop.shape[1] < w:
                        failures.append(
                            f"элемент {check.element!r}: область rect за "
                            "пределами кадра")
                        continue
                    found = matcher.match(crop, template).score >= nav_threshold
                if negate:
                    found = not found
                if not found:
                    failures.append(
                        f"template {check.element!r}: элемент "
                        f"{'найден' if negate else 'не найден'} "
                        f"(порог {nav_threshold})")
            elif check.kind == "pixel_probe":
                anchor = menu_anchor()
                if anchor is None:
                    failures.append(
                        "pixel_probe: якорь меню не локализован")
                    continue
                frame_h, frame_w = np.asarray(frame).shape[:2]
                ok = True
                for point in check.points:
                    x, y = anchor[0] + point.at[0], anchor[1] + point.at[1]
                    if not (0 <= x < frame_w and 0 <= y < frame_h):
                        failures.append(
                            f"точка ({x},{y}) вне кадра {frame_w}x{frame_h}")
                        ok = False
                        break
                    actual = np.asarray(frame)[y, x].astype(int)
                    expected = np.array(point.color, dtype=int)[::-1]  # BGR
                    if np.abs(actual - expected).max() > probe_tolerance:
                        failures.append(
                            f"точка {point.at}: цвет вне допуска "
                            f"{probe_tolerance} от {point.color}")
                        ok = False
                if not ok:
                    continue
            else:
                failures.append(
                    f"проверка вида {check.kind!r} в навигации "
                    "не поддержана")
        return CheckReport(passed=not failures, failures=tuple(failures))

    return evaluate
