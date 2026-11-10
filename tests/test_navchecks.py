"""Оценщик проверок навигационной модели (Task 7, l2core.detector.navchecks).

Детерминизм: фейковый matcher различает шаблоны по содержимому (нулевой —
не находится, единичный — находится с score 1.0), кадры — синтетические
numpy-массивы.
"""

from pathlib import Path

import cv2
import numpy as np

from l2core.adapters.template import Match
from l2core.detector.navchecks import MENU_ANCHOR_ID, build_nav_evaluator
from l2core.model import Check, ProbePoint
from l2core.model.navigation import load_navigation

NAV_YAML = """\
elements:
  menu_anchor:
    role: icon_sample
    hud: true
    search_area: [10, 10, 200, 200]
    variants:
      normal: {template: navigation/anchor.png}
  menu_market_icon:
    role: click_target
    rect: [40, 60, 48, 48]
    variants:
      normal: {template: navigation/icon.png}
signatures:
  menu_open:
    - {kind: template, element: menu_anchor, variant: normal}
    - {kind: template, element: menu_market_icon, variant: normal}
recipes:
  open_market:
    opens: market
    steps:
      - action: key
        key: X
        verify: [menu_open]
        on_fail: [press_esc]
    on_fail: [press_esc]
"""


class ContentMatcher:
    """«Нулевой» шаблон не находится; «единичный» — Match(2, 3, 1.0)."""

    def __init__(self, found_score=1.0):
        self.found_score = found_score

    def match(self, image, template):
        if float(template.sum()) == 0.0:
            return Match(x=0, y=0, score=0.0)
        return Match(x=2, y=3, score=self.found_score)


def _assets(tmp_path, *, anchor_white=True, icon_white=True):
    assets = tmp_path / "assets"
    (assets / "navigation").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(assets / "navigation" / "anchor.png"),
                np.full((8, 8), 255 if anchor_white else 0, dtype=np.uint8))
    cv2.imwrite(str(assets / "navigation" / "icon.png"),
                np.full((8, 8), 255 if icon_white else 0, dtype=np.uint8))
    return assets


def _nav(tmp_path):
    path = tmp_path / "navigation.yaml"
    path.write_text(NAV_YAML, encoding="utf-8")
    return load_navigation(path)


FRAME = np.zeros((300, 300, 3), dtype=np.uint8)


class TestNavEvaluator:
    def test_hud_template_found(self, tmp_path):
        nav = _nav(tmp_path)
        evaluate = build_nav_evaluator(
            nav, ContentMatcher(), _assets(tmp_path))
        report = evaluate(
            (Check(kind="template", element="menu_anchor", variant="normal"),),
            FRAME)
        assert report.passed

    def test_hud_template_not_found(self, tmp_path):
        nav = _nav(tmp_path)
        evaluate = build_nav_evaluator(
            nav, ContentMatcher(), _assets(tmp_path, anchor_white=False))
        report = evaluate(
            (Check(kind="template", element="menu_anchor", variant="normal"),),
            FRAME)
        assert not report.passed and report.failures

    def test_absent_negation(self, tmp_path):
        nav = _nav(tmp_path)
        absent = Check(kind="template", element="menu_anchor",
                       variant="normal", expect="absent")
        # Якорь найден — отрицание провалено.
        found_eval = build_nav_evaluator(
            nav, ContentMatcher(), _assets(tmp_path))
        assert not found_eval((absent,), FRAME).passed
        # Якоря нет — отрицание сошлось.
        gone_eval = build_nav_evaluator(
            nav, ContentMatcher(), _assets(tmp_path, anchor_white=False))
        assert gone_eval((absent,), FRAME).passed

    def test_menu_element_fails_without_anchor(self, tmp_path):
        nav = _nav(tmp_path)
        # Якорь «нулевой» (не находится) — проверка иконки меню честно падает.
        evaluate = build_nav_evaluator(
            nav, ContentMatcher(), _assets(tmp_path, anchor_white=False))
        report = evaluate(
            (Check(kind="template", element="menu_market_icon",
                   variant="normal"),), FRAME)
        assert not report.passed
        assert any("menu_anchor" in f for f in report.failures)

    def test_menu_element_found_with_anchor(self, tmp_path):
        nav = _nav(tmp_path)
        evaluate = build_nav_evaluator(
            nav, ContentMatcher(), _assets(tmp_path))
        report = evaluate(nav.signatures["menu_open"], FRAME)
        assert report.passed

    def test_pixel_probe_against_menu_anchor(self, tmp_path):
        nav = _nav(tmp_path)
        evaluate = build_nav_evaluator(
            nav, ContentMatcher(), _assets(tmp_path))
        # ContentMatcher даёт якорю позицию (2, 3) внутри search_area:
        # абсолютный якорь = (10 + 2, 10 + 3). Точка (5, 7) от якоря —
        # пиксель (17, 20) кадра. Модель в RGB, кадр BGR.
        frame = FRAME.copy()
        frame[20, 17] = (30, 200, 100)  # BGR = RGB(100, 200, 30)
        probe = Check(kind="pixel_probe", points=(
            ProbePoint(at=(5, 7), color=(100, 200, 30)),))
        assert evaluate((probe,), frame).passed
        frame[20, 17] = (0, 0, 0)
        report = evaluate((probe,), frame)
        assert not report.passed

    def test_probe_out_of_frame_fails(self, tmp_path):
        nav = _nav(tmp_path)
        evaluate = build_nav_evaluator(
            nav, ContentMatcher(), _assets(tmp_path))
        probe = Check(kind="pixel_probe", points=(
            ProbePoint(at=(5000, 5000), color=(0, 0, 0)),))
        assert not evaluate((probe,), FRAME).passed
