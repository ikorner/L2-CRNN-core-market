"""Детектор якоря окна торговли (задача 3.3).

Ground truth — meta.json (base_screenshot/scr/) плюс сверка позиции окна
рамкой (numpy-only detect_window) и template matching'ом (числа — в отчёте
3.3):

- market, батч 1 (scr_153551..scr_153840): окно открыто, якорь (207, 77);
- market, батч 2 (scr_153901..scr_154134): окно открыто, якорь (202, 103)
  (окно сдвинуто; rect'ы элементов модели из батча 2 конвертер считал
  от (207, 77) — исправлены вручную в задаче 3.3b, guard-тест ниже
  (test_market_batch2_rects_match_meta) следит за регрессом);
- shop_buy «opening» (scr_154330/154340/154355): якорь (155, 224);
- shop_sell «opening» (scr_154213/154222/154244): якорь (154, 223);
- негативы (окон торговли нет): scr_153511, scr_153527, scr_all_close;
- честно не находит: scr_154318/154424 — лавки в состоянии «open»,
  заголовок «<Продажа>»/«<Покупка>»; шаблонов таких заголовков в модели
  нет (ограничение зафиксировано в отчёте 3.3).
"""

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pytest

from l2core.adapters.template import NumpyTemplateMatcher, OpenCvTemplateMatcher
from l2core.detector.anchor import AnchorDetector, detect_window
from l2core.model import load_model

REPO_ROOT = Path(__file__).resolve().parent.parent
SCR_DIR = REPO_ROOT / "base_screenshot" / "scr"
ASSETS_DIR = REPO_ROOT / "assets"
UI_MODEL_DIR = REPO_ROOT / "ui_model"

MARKET_BATCH_1 = [
    "scr_153551", "scr_153600", "scr_153611", "scr_153619", "scr_153631",
    "scr_153646", "scr_153656", "scr_153712", "scr_153722", "scr_153731",
    "scr_153740", "scr_153750", "scr_153802", "scr_153810", "scr_153819",
    "scr_153830", "scr_153840",
]
MARKET_BATCH_2 = [
    "scr_153901", "scr_153929", "scr_153941", "scr_153949", "scr_154002",
    "scr_154012", "scr_154021", "scr_154033", "scr_154049", "scr_154103",
    "scr_154134",
]
MARKET_ANCHORS = {name: (207, 77) for name in MARKET_BATCH_1}
MARKET_ANCHORS.update({name: (202, 103) for name in MARKET_BATCH_2})

# Скриншоты без окна «Обзор лавок» (market якорь обязан не находиться).
MARKET_NEGATIVES = ["scr_153511", "scr_153527", "scr_all_close"]
# Скриншоты окон лавок (рамки обзора там нет).
SHOP_SCREENSHOTS = [
    "scr_154213", "scr_154222", "scr_154244", "scr_154318",
    "scr_154330", "scr_154340", "scr_154355", "scr_154424",
]

SHOP_BUY_ANCHORS = {
    "scr_154330": (155, 224),
    "scr_154340": (155, 224),
    "scr_154355": (155, 224),
}
SHOP_SELL_ANCHORS = {
    "scr_154213": (154, 223),
    "scr_154222": (154, 223),
    "scr_154244": (154, 223),
}
# Состояние «open»: заголовок «<Покупка>»/«<Продажа>», шаблона в модели нет.
SHOP_OPEN_STATE = ["scr_154318", "scr_154424"]

MARKET_SIZE = (1008, 674)  # window.size модели market


@lru_cache(maxsize=None)
def frame_bgr(name: str):
    return cv2.imread(str(SCR_DIR / f"{name}.png"))


@pytest.fixture(scope="module")
def models():
    return {
        window_id: load_model(UI_MODEL_DIR / f"{window_id}.yaml")
        for window_id in ("market", "shop_buy", "shop_sell")
    }


@pytest.fixture(
    scope="module",
    params=[NumpyTemplateMatcher(), OpenCvTemplateMatcher()],
    ids=["numpy", "opencv"],
)
def matcher(request):
    return request.param


@pytest.fixture(scope="module")
def detector(models, matcher):
    return AnchorDetector.for_windows(models, matcher, ASSETS_DIR)


class TestDetectWindow:
    """numpy-only первый этап: яркие линии рамки окна «Обзор лавок»."""

    @pytest.mark.parametrize("name", sorted(MARKET_ANCHORS))
    def test_finds_frame_on_market_positives(self, name):
        got = detect_window(frame_bgr(name), MARKET_SIZE)
        assert got == MARKET_ANCHORS[name]

    @pytest.mark.parametrize("name", MARKET_NEGATIVES + SHOP_SCREENSHOTS)
    def test_not_found_without_market_window(self, name):
        assert detect_window(frame_bgr(name), MARKET_SIZE) is None

    def test_blank_frame_returns_none(self):
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        assert detect_window(frame, MARKET_SIZE) is None


class TestAnchorDetectorMarket:
    @pytest.mark.parametrize("name", sorted(MARKET_ANCHORS))
    def test_market_anchor_found(self, detector, name):
        anchor = detector.find_anchor(frame_bgr(name), "market")
        assert anchor is not None
        assert anchor.window_id == "market"
        assert (anchor.x, anchor.y) == MARKET_ANCHORS[name]

    @pytest.mark.parametrize("name", MARKET_NEGATIVES)
    def test_market_not_found_on_negatives(self, detector, name):
        assert detector.find_anchor(frame_bgr(name), "market") is None


class TestAnchorDetectorShops:
    @pytest.mark.parametrize("name", sorted(SHOP_BUY_ANCHORS))
    def test_shop_buy_anchor_found(self, detector, name):
        anchor = detector.find_anchor(frame_bgr(name), "shop_buy")
        assert anchor is not None
        assert (anchor.x, anchor.y) == SHOP_BUY_ANCHORS[name]

    @pytest.mark.parametrize("name", sorted(SHOP_SELL_ANCHORS))
    def test_shop_sell_anchor_found(self, detector, name):
        anchor = detector.find_anchor(frame_bgr(name), "shop_sell")
        assert anchor is not None
        assert (anchor.x, anchor.y) == SHOP_SELL_ANCHORS[name]

    @pytest.mark.parametrize("name", SHOP_OPEN_STATE)
    def test_open_state_honestly_not_found(self, detector, name):
        """Заголовок «open»-состояния другой — якорь честно не находится."""
        assert detector.find_anchor(frame_bgr(name), "shop_buy") is None
        assert detector.find_anchor(frame_bgr(name), "shop_sell") is None

    def test_shop_not_confused_by_market_window(self, detector):
        """На скриншоте обзора лавок заголовок лавки не должен находиться."""
        assert detector.find_anchor(frame_bgr("scr_153551"), "shop_buy") is None
        assert detector.find_anchor(frame_bgr("scr_153551"), "shop_sell") is None


class TestNoCacheBetweenFrames:
    """Кэша позиции между кадрами нет: каждый grab — свежий поиск."""

    def test_alternating_frames_give_fresh_anchors(self, detector):
        frames = ["scr_153551", "scr_153901", "scr_153551", "scr_153901"]
        expected = [(207, 77), (202, 103), (207, 77), (202, 103)]
        for name, want in zip(frames, expected):
            anchor = detector.find_anchor(frame_bgr(name), "market")
            assert anchor is not None
            assert (anchor.x, anchor.y) == want


class TestAnchorModelConsistency:
    """Якорь + rect элемента модели == наблюдаемая позиция кропа в мете."""

    @pytest.mark.parametrize("name", MARKET_BATCH_1)
    def test_market_batch1_rects_match_meta(self, models, name):
        meta = _load_meta()
        model = models["market"]
        anchor = MARKET_ANCHORS[name]
        for crop_name in meta["screenshots"][name]["crops"]:
            crop = meta["crops"].get(crop_name)
            if crop is None:
                continue
            element = _element_by_template(model, f"market/{crop_name}.png")
            if element is None or element.rect is None:
                continue
            expect = crop["rect"]
            got = [anchor[0] + element.rect[0], anchor[1] + element.rect[1],
                   element.rect[2], element.rect[3]]
            assert got == expect, (
                f"{name}/{crop_name}: якорь+rect {got} != мета {expect}"
            )

    @pytest.mark.parametrize("name", MARKET_BATCH_2)
    def test_market_batch2_rects_match_meta(self, models, name):
        # rect'ы элементов батча 2 конвертер 2.2 вычел из (207, 77) вместо
        # истинного якоря (202, 103) — исправлено вручную (задача 3.3b).
        # Регресс должен ловиться здесь, а не пиксельным ревью.
        meta = _load_meta()
        model = models["market"]
        anchor = MARKET_ANCHORS[name]
        for crop_name in meta["screenshots"][name]["crops"]:
            crop = meta["crops"].get(crop_name)
            if crop is None:
                continue
            element = _element_by_template(model, f"market/{crop_name}.png")
            if element is None or element.rect is None:
                continue
            expect = crop["rect"]
            got = [anchor[0] + element.rect[0], anchor[1] + element.rect[1],
                   element.rect[2], element.rect[3]]
            assert got == expect, (
                f"{name}/{crop_name}: якорь+rect {got} != мета {expect}"
            )

    def test_shop_buy_jak_rect_matches_meta(self, models):
        jak = models["shop_buy"].elements["jak"]
        assert jak.rect == (160, 8, 198, 29)
        anchor = SHOP_BUY_ANCHORS["scr_154330"]
        assert (anchor[0] + jak.rect[0], anchor[1] + jak.rect[1]) == (315, 232)

    def test_shop_sell_jakor_rect_matches_meta(self, models):
        jakor = models["shop_sell"].elements["jakor_sell"]
        assert jakor.rect == (156, 9, 208, 26)
        anchor = SHOP_SELL_ANCHORS["scr_154213"]
        assert (anchor[0] + jakor.rect[0], anchor[1] + jakor.rect[1]) == (310, 232)


@lru_cache(maxsize=None)
def _load_meta():
    import json

    return json.loads((SCR_DIR / "meta.json").read_text(encoding="utf-8"))


def _element_by_template(model, template_path: str):
    for element in model.elements.values():
        for variant in element.variants.values():
            if variant.template == template_path:
                return element
    return None


# ---------------------------------------------------------------------------
# Многоопорный якорь (menu-quickslot-navigation, Task 1): на окно — кортеж
# опор (element_id, variant); find_anchor перебирает опоры по порогу окна,
# первая сошедшаяся даёт Anchor. Поведение market (frame_stage, одна опора)
# не меняется.
# ---------------------------------------------------------------------------

from l2core.adapters.template import Match
from l2core.detector.anchor import (
    DEFAULT_ANCHOR_TEMPLATES,
    AnchorDetector,
)
from l2core.model import ElementSpec, Rect, UiModel, VariantSpec, WindowSpec


class _FakeMatcher:
    """Matcher по содержимому шаблона: «нулевой» шаблон не находится,
    «единичный» находится в (5, 7) с score 1.0."""

    def match(self, gray, template):
        if float(template.sum()) == 0.0:
            return Match(x=0, y=0, score=0.0)
        return Match(x=5, y=7, score=1.0)


def _model_with_supports(tmp_path, assets_dir):
    """Синтетическая модель окна с двумя элементами-опорами: op_a (нулевой
    шаблон — никогда не находится) и op_b (единичный — находится)."""
    blank = np.zeros((10, 10), dtype=np.uint8)
    white = np.full((10, 10), 255, dtype=np.uint8)
    cv2.imwrite(str(assets_dir / "t_a.png"), blank)
    cv2.imwrite(str(assets_dir / "t_b.png"), white)
    elements = {
        "op_a": ElementSpec(
            id="op_a", role="state_signature",
            rect=Rect(0, 0, 10, 10),
            variants={"normal": VariantSpec("normal", "t_a.png")}),
        "op_b": ElementSpec(
            id="op_b", role="state_signature",
            rect=Rect(2, 3, 10, 10),
            variants={"normal": VariantSpec("normal", "t_b.png")}),
    }
    model = UiModel(
        path=str(tmp_path / "win.yaml"),
        window=WindowSpec(id="win", title="тест", size=(100, 100)),
        elements=elements, states={}, transitions=(), ocr_regions={},
        table_geometry=None)
    return {"win": model}


class TestMultiSupportAnchor:
    def test_second_support_matches_when_first_absent(self, tmp_path):
        assets = tmp_path / "assets"
        assets.mkdir()
        models = _model_with_supports(tmp_path, assets)
        detector = AnchorDetector(
            models, _FakeMatcher(), assets,
            anchor_templates={"win": (("op_a", "normal"), ("op_b", "normal"))})
        anchor = detector.find_anchor(np.zeros((50, 50, 3), np.uint8), "win")
        assert anchor is not None
        assert anchor.method == "template"
        # Anchor = позиция совпадения минус rect офсета элемента-опоры op_b.
        assert (anchor.x, anchor.y) == (5 - 2, 7 - 3)

    def test_none_support_matches_returns_none(self, tmp_path):
        assets = tmp_path / "assets"
        assets.mkdir()
        models = _model_with_supports(tmp_path, assets)
        # Обе опоры ссылаются на «нулевой» шаблон — matcher никогда не находит.
        detector = AnchorDetector(
            models, _FakeMatcher(), assets,
            anchor_templates={"win": (("op_a", "normal"), ("op_a", "normal"))})
        assert detector.find_anchor(
            np.zeros((50, 50, 3), np.uint8), "win") is None

    def test_legacy_single_support_normalized(self, tmp_path):
        """Старый формат anchor_templates (element, variant) принимается:
        нормализуется в одноопорный кортеж."""
        assets = tmp_path / "assets"
        assets.mkdir()
        models = _model_with_supports(tmp_path, assets)
        detector = AnchorDetector(
            models, _FakeMatcher(), assets,
            anchor_templates={"win": ("op_b", "normal")})
        anchor = detector.find_anchor(np.zeros((50, 50, 3), np.uint8), "win")
        assert (anchor.x, anchor.y) == (5 - 2, 7 - 3)

    def test_default_anchor_templates_is_tuple_of_tuples(self):
        for window_id, supports in DEFAULT_ANCHOR_TEMPLATES.items():
            assert isinstance(supports, tuple)
            for support in supports:
                assert isinstance(support, tuple) and len(support) == 2
