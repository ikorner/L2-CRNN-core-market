"""Детектор состояний окна торговли (задача 3.4).

Модель принятия решений (spec window-state-detection):
- состояние подтверждается, когда совпали ВСЕ его сигнатуры (валидатор 2.3
  гарантирует минимум две; состояние с одной сигнатурой не подтверждается
  никогда);
- сигнатуры оцениваются в иерархии «пиксель-пробы -> template -> OCR»
  (дешёвые первыми) с остановкой состояния на первой несовпавшей;
- OCR-сигнатуры оцениваются только при наличии OCR-клиента; без него
  состояние с OCR-сигнатурами не подтверждается (деградация с диагностикой,
  а не исключение);
- ни одно состояние не подтверждено -> unknown; несколько подтверждены ->
  unknown + журналирование конфликтующих состояний и их признаков;
- состояния с префиксом recover_ (глобальный recover, рантайм 5.8)
  классификацией не занимаются.

Ground truth (R4): мета знает window для всех 39 скриншотов, params у неё
частичные/пустые — на живых кадрах проверяется связка «active-вариант
элемента -> params» там, где мета подтверждает активный таб описанием кропа
(scr_153551 tab=home, scr_153600 tab=equipment + filter/global_filter=all,
лавки «opening»: scr_154330/154340/154213). Скрины с неразмеченными табами
(scr_153802 tab=scroll и др.) -> unknown (состояний этих табов в модели нет).

Синтетика: модели собираются прямо в объектных типах (l2core.model.types),
шаблоны — константные PNG во временном каталоге assets; поддельный matcher
возвращает score, закодированный яркостью шаблона (детерминизм, без cv2).
"""

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pytest

from l2core.adapters.template import Match, OpenCvTemplateMatcher, TemplateMatcher
from l2core.detector.anchor import Anchor, AnchorDetector
from l2core.detector.states import StateDetector
from l2core.model import (
    Check,
    ElementSpec,
    OcrRegionSpec,
    ProbePoint,
    Rect,
    StateSpec,
    UiModel,
    VariantSpec,
    WindowSpec,
    GrammarSpec,
    load_model,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SCR_DIR = REPO_ROOT / "base_screenshot" / "scr"
ASSETS_DIR = REPO_ROOT / "assets"
UI_MODEL_DIR = REPO_ROOT / "ui_model"


def make_model(*, states, elements=None, ocr_regions=None,
               window_id="market", size=(400, 300)):
    """Синтетическая модель окна из объектных типов."""
    return UiModel(
        path="synthetic",
        window=WindowSpec(id=window_id, title="Синтетика", size=size),
        elements=elements or {},
        states=states,
        transitions=(),
        ocr_regions=ocr_regions or {},
        table_geometry=None,
    )


def tpl_check(element, variant="normal"):
    return Check(kind="template", element=element, variant=variant)


def state(sid, signatures, params=None):
    return StateSpec(id=sid, params=params or {}, signatures=tuple(signatures))


def element(eid, rect, template, variant="normal", hud=False, search_area=None):
    spec = VariantSpec(name=variant, template=template)
    return ElementSpec(
        id=eid, role="click_target", rect=rect, hud=hud,
        search_area=search_area, variants={variant: spec},
    )


def write_template(assets, rel_path, brightness):
    """Константный серый PNG; яркость кодирует желаемый NCC-скор."""
    path = assets / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((8, 8), brightness, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)
    return f"{rel_path.as_posix()}"


class RecordingMatcher(TemplateMatcher):
    """Matcher-заглушка: score = яркость шаблона / 255, вызовы подсчитываются."""

    def __init__(self):
        self.calls = []

    def match(self, image, template):
        self.calls.append(np.asarray(template).copy())
        return Match(x=0, y=0, score=float(template[0, 0]) / 255.0)


class FakeOcr:
    """OCR-заглушка: очередь ответов (text, confidence), вызовы подсчитываются."""

    def __init__(self, answers=()):
        self.answers = list(answers)
        self.calls = []

    def read(self, region_img):
        self.calls.append(np.asarray(region_img).copy())
        return self.answers.pop(0) if self.answers else ("", 0.0)


@pytest.fixture()
def synth(tmp_path):
    """Каталог assets для синтетических шаблонов."""
    assets = tmp_path / "assets"
    assets.mkdir()
    return assets


def detector_for(models, matcher, assets, **kwargs):
    return StateDetector(models, matcher, assets, **kwargs)


class TestStrictAllSignatures:
    """Состояние подтверждается только совпадением ВСЕХ сигнатур (>= 2)."""

    def test_two_signatures_confirmed(self, synth):
        t1 = write_template(synth, Path("t/a.png"), 250)  # ~0.98
        t2 = write_template(synth, Path("t/b.png"), 240)  # ~0.94
        model = make_model(
            states={"list__tab_home": state(
                "list__tab_home",
                [tpl_check("el_a"), tpl_check("el_b")],
                params={"tab": "home"})},
            elements={"el_a": element("el_a", Rect(0, 0, 8, 8), t1),
                      "el_b": element("el_b", Rect(20, 20, 8, 8), t2)},
        )
        matcher = RecordingMatcher()
        report = detector_for({"market": model}, matcher, synth).detect(
            np.zeros((300, 400, 3), dtype=np.uint8), Anchor("market", 0, 0, 1.0, "template"))
        assert report.state == "list__tab_home"
        assert report.params == {"tab": "home"}
        assert len(report.matched) == 2
        assert report.conflicts == ()
        assert not report.ocr_degraded

    def test_one_of_two_failed_is_unknown(self, synth):
        t1 = write_template(synth, Path("t/a.png"), 250)
        t2 = write_template(synth, Path("t/b.png"), 100)  # ~0.39 < порога
        model = make_model(
            states={"s": state("s", [tpl_check("el_a"), tpl_check("el_b")])},
            elements={"el_a": element("el_a", Rect(0, 0, 8, 8), t1),
                      "el_b": element("el_b", Rect(20, 20, 8, 8), t2)},
        )
        report = detector_for({"market": model}, RecordingMatcher(), synth).detect(
            np.zeros((300, 400, 3), dtype=np.uint8), Anchor("market", 0, 0, 1.0, "template"))
        assert report.state is None
        assert report.params == {}
        assert [r.name for r in report.failed] == ["template:el_b/normal"]

    def test_single_signature_state_never_confirmed(self, synth):
        t1 = write_template(synth, Path("t/a.png"), 250)
        model = make_model(
            states={"s": state("s", [tpl_check("el_a")])},
            elements={"el_a": element("el_a", Rect(0, 0, 8, 8), t1)},
        )
        report = detector_for({"market": model}, RecordingMatcher(), synth).detect(
            np.zeros((300, 400, 3), dtype=np.uint8), Anchor("market", 0, 0, 1.0, "template"))
        assert report.state is None
        assert any("одна сигнатура" in r.detail for r in report.failed)


class TestConflict:
    """Два подтверждённых состояния -> unknown + журнал конфликта."""

    def test_two_confirmed_states_conflict(self, synth, caplog):
        t1 = write_template(synth, Path("t/a.png"), 250)
        model = make_model(
            states={
                "list__tab_home": state(
                    "list__tab_home", [tpl_check("el_a"), tpl_check("el_b")],
                    params={"tab": "home"}),
                "list__tab_weapon": state(
                    "list__tab_weapon", [tpl_check("el_c"), tpl_check("el_d")],
                    params={"tab": "weapon"}),
            },
            elements={e: element(e, Rect(0, 0, 8, 8), t1)
                      for e in ("el_a", "el_b", "el_c", "el_d")},
        )
        with caplog.at_level("WARNING", logger="l2core.detector.states"):
            report = detector_for({"market": model}, RecordingMatcher(), synth).detect(
                np.zeros((300, 400, 3), dtype=np.uint8),
                Anchor("market", 0, 0, 1.0, "template"))
        assert report.state is None
        assert sorted(report.conflicts) == ["list__tab_home", "list__tab_weapon"]
        messages = "\n".join(r.getMessage() for r in caplog.records)
        assert "list__tab_home" in messages and "list__tab_weapon" in messages
        assert "template:el_a/normal" in messages  # имена конфликтующих признаков


class TestParamsBinding:
    """params — из размеченного состояния (связка active-вариант -> params)."""

    def test_params_follow_confirmed_state(self, synth):
        t_ok = write_template(synth, Path("t/ok.png"), 250)
        t_ok2 = write_template(synth, Path("t/ok2.png"), 248)
        t_bad = write_template(synth, Path("t/bad.png"), 50)
        model = make_model(
            states={
                "list__tab_home": state(
                    "list__tab_home", [tpl_check("el_home"), tpl_check("el_home2")],
                    params={"tab": "home"}),
                "list__tab_equipment": state(
                    "list__tab_equipment", [tpl_check("el_eq", "active"), tpl_check("el_eq2")],
                    params={"tab": "equipment", "filter": "all"}),
            },
            elements={
                "el_home": element("el_home", Rect(0, 0, 8, 8), t_ok),
                "el_home2": element("el_home2", Rect(10, 0, 8, 8), t_ok2),
                "el_eq": element("el_eq", Rect(0, 0, 8, 8), t_bad, variant="active"),
                "el_eq2": element("el_eq2", Rect(10, 0, 8, 8), t_bad),
            },
        )
        report = detector_for({"market": model}, RecordingMatcher(), synth).detect(
            np.zeros((300, 400, 3), dtype=np.uint8), Anchor("market", 0, 0, 1.0, "template"))
        assert report.state == "list__tab_home"
        assert report.params == {"tab": "home"}


class TestHierarchy:
    """Иерархия: пиксель-пробы -> template -> OCR; остановка на первом провале."""

    def test_pixel_probe_failure_skips_template(self, synth):
        t1 = write_template(synth, Path("t/a.png"), 250)
        model = make_model(
            states={"s": state("s", [
                Check(kind="pixel_probe",
                      points=(ProbePoint((5, 5), (255, 255, 255)),)),
                tpl_check("el_a"),
            ])},
            elements={"el_a": element("el_a", Rect(0, 0, 8, 8), t1)},
        )
        frame = np.zeros((300, 400, 3), dtype=np.uint8)  # (5,5) чёрный != белый
        matcher = RecordingMatcher()
        report = detector_for({"market": model}, matcher, synth).detect(
            frame, Anchor("market", 0, 0, 1.0, "template"))
        assert report.state is None
        assert matcher.calls == []  # template не вызывался
        assert report.failed[0].kind == "pixel_probe"

    def test_template_failure_skips_ocr(self, synth):
        t1 = write_template(synth, Path("t/a.png"), 50)  # < порога
        model = make_model(
            states={"s": state("s", [
                tpl_check("el_a"),
                Check(kind="ocr", region="title", expect="Обзор лавок"),
            ])},
            elements={"el_a": element("el_a", Rect(0, 0, 8, 8), t1)},
            ocr_regions={"title": OcrRegionSpec(
                id="title", rect=Rect(0, 0, 8, 8),
                grammar=GrammarSpec(name="text", kind="text"))},
        )
        ocr = FakeOcr([("Обзор лавок", 0.99)])
        report = detector_for({"market": model}, RecordingMatcher(), synth,
                              ocr_client=ocr).detect(
            np.zeros((300, 400, 3), dtype=np.uint8),
            Anchor("market", 0, 0, 1.0, "template"))
        assert report.state is None
        assert ocr.calls == []  # OCR не вызывался

    def test_evaluation_order_pixel_template_ocr(self, synth):
        t1 = write_template(synth, Path("t/a.png"), 250)
        model = make_model(
            states={"s": state("s", [
                Check(kind="ocr", region="title", expect="x"),
                tpl_check("el_a"),
                Check(kind="pixel_probe",
                      points=(ProbePoint((5, 5), (0, 0, 0)),)),
            ])},
            elements={"el_a": element("el_a", Rect(0, 0, 8, 8), t1)},
            ocr_regions={"title": OcrRegionSpec(
                id="title", rect=Rect(0, 0, 8, 8),
                grammar=GrammarSpec(name="text", kind="text"))},
        )
        ocr = FakeOcr([("x", 1.0)])
        matcher = RecordingMatcher()
        report = detector_for({"market": model}, matcher, synth,
                              ocr_client=ocr).detect(
            np.zeros((300, 400, 3), dtype=np.uint8),
            Anchor("market", 0, 0, 1.0, "template"))
        # Сигнатуры в модели перечислены ocr-first, оценка — по иерархии:
        # пиксель-проба, template, OCR.
        assert [r.kind for r in report.matched] == [
            "pixel_probe", "template", "ocr"]
        assert matcher.calls and ocr.calls  # оба дорогих вида дождались очереди


class TestOcrDegradation:
    """OCR-сигнатуры без клиента: деградация с диагностикой, не исключение."""

    def test_no_client_state_with_ocr_signatures_unknown(self, synth):
        t1 = write_template(synth, Path("t/a.png"), 250)
        model = make_model(
            states={"s": state("s", [
                tpl_check("el_a"),
                Check(kind="ocr", region="title", expect="Обзор лавок"),
            ])},
            elements={"el_a": element("el_a", Rect(0, 0, 8, 8), t1)},
            ocr_regions={"title": OcrRegionSpec(
                id="title", rect=Rect(0, 0, 8, 8),
                grammar=GrammarSpec(name="text", kind="text"))},
        )
        report = detector_for({"market": model}, RecordingMatcher(), synth).detect(
            np.zeros((300, 400, 3), dtype=np.uint8), Anchor("market", 0, 0, 1.0, "template"))
        assert report.state is None
        assert report.ocr_degraded
        assert report.failed[-1].matched is None  # unavailable, не False
        assert "OCR" in report.failed[-1].detail

    def test_no_client_state_without_ocr_works_fully(self, synth):
        t1 = write_template(synth, Path("t/a.png"), 250)
        t2 = write_template(synth, Path("t/b.png"), 245)
        model = make_model(
            states={"s": state("s", [tpl_check("el_a"), tpl_check("el_b")],
                               params={"tab": "home"})},
            elements={"el_a": element("el_a", Rect(0, 0, 8, 8), t1),
                      "el_b": element("el_b", Rect(20, 20, 8, 8), t2)},
        )
        report = detector_for({"market": model}, RecordingMatcher(), synth).detect(
            np.zeros((300, 400, 3), dtype=np.uint8), Anchor("market", 0, 0, 1.0, "template"))
        assert report.state == "s"
        assert not report.ocr_degraded

    def test_ocr_text_mismatch_is_unknown(self, synth):
        t1 = write_template(synth, Path("t/a.png"), 250)
        model = make_model(
            states={"s": state("s", [
                tpl_check("el_a"),
                Check(kind="ocr", region="title", expect="Обзор лавок"),
            ])},
            elements={"el_a": element("el_a", Rect(0, 0, 8, 8), t1)},
            ocr_regions={"title": OcrRegionSpec(
                id="title", rect=Rect(0, 0, 8, 8),
                grammar=GrammarSpec(name="text", kind="text"))},
        )
        ocr = FakeOcr([("Торговая лавка", 0.9)])
        report = detector_for({"market": model}, RecordingMatcher(), synth,
                              ocr_client=ocr).detect(
            np.zeros((300, 400, 3), dtype=np.uint8),
            Anchor("market", 0, 0, 1.0, "template"))
        assert report.state is None
        assert not report.ocr_degraded
        assert report.failed[-1].kind == "ocr"


class TestPixelProbe:
    def test_color_within_tolerance_matches(self, synth):
        model = make_model(
            states={"s": state("s", [
                Check(kind="pixel_probe",
                      points=(ProbePoint((5, 5), (250, 240, 230)),
                              ProbePoint((6, 6), (10, 20, 30)))),
                Check(kind="pixel_probe",
                      points=(ProbePoint((7, 7), (255, 255, 255)),)),
            ])},
        )
        frame = np.zeros((300, 400, 3), dtype=np.uint8)
        # Якорь (100, 50): точки сигнатуры отсчитываются от него.
        frame[55, 105] = (230, 245, 251)  # BGR; отклонение <= 16 по каналам
        frame[56, 106] = (30, 20, 10)
        frame[57, 107] = (255, 255, 254)  # отклонение 1
        report = detector_for({"market": model}, RecordingMatcher(), synth).detect(
            frame, Anchor("market", 100, 50, 1.0, "template"))
        assert report.state == "s"

    def test_color_outside_tolerance_fails(self, synth):
        model = make_model(
            states={"s": state("s", [
                Check(kind="pixel_probe",
                      points=(ProbePoint((5, 5), (255, 255, 255)),)),
                Check(kind="pixel_probe",
                      points=(ProbePoint((7, 7), (255, 255, 255)),)),
            ])},
        )
        frame = np.zeros((300, 400, 3), dtype=np.uint8)
        frame[55, 105] = (200, 200, 200)
        frame[57, 107] = (255, 255, 255)
        report = detector_for({"market": model}, RecordingMatcher(), synth).detect(
            frame, Anchor("market", 100, 50, 1.0, "template"))
        assert report.state is None
        assert report.failed[0].kind == "pixel_probe"


class TestRecoverStatesSkipped:
    """Состояния recover_* — не классификация, а рантайм 5.8: пропускаются."""

    def test_recover_state_does_not_conflict(self, synth):
        t1 = write_template(synth, Path("t/a.png"), 250)
        model = make_model(
            states={
                "recover_closed": state(
                    "recover_closed", [tpl_check("el_a"), tpl_check("el_b")]),
                "list__tab_home": state(
                    "list__tab_home", [tpl_check("el_c"), tpl_check("el_d")],
                    params={"tab": "home"}),
            },
            elements={e: element(e, Rect(0, 0, 8, 8), t1)
                      for e in ("el_a", "el_b", "el_c", "el_d")},
        )
        report = detector_for({"market": model}, RecordingMatcher(), synth).detect(
            np.zeros((300, 400, 3), dtype=np.uint8), Anchor("market", 0, 0, 1.0, "template"))
        assert report.state == "list__tab_home"
        assert report.conflicts == ()

    def test_only_recover_state_is_unknown(self, synth):
        t1 = write_template(synth, Path("t/a.png"), 250)
        model = make_model(
            states={"recover_closed": state(
                "recover_closed", [tpl_check("el_a"), tpl_check("el_b")])},
            elements={e: element(e, Rect(0, 0, 8, 8), t1)
                      for e in ("el_a", "el_b")},
        )
        report = detector_for({"market": model}, RecordingMatcher(), synth).detect(
            np.zeros((300, 400, 3), dtype=np.uint8), Anchor("market", 0, 0, 1.0, "template"))
        assert report.state is None


class TestRegionBounds:
    def test_region_outside_frame_fails_gracefully(self, synth):
        t1 = write_template(synth, Path("t/a.png"), 250)
        model = make_model(
            states={"s": state("s", [
                tpl_check("el_a"),
                tpl_check("el_b"),
            ])},
            elements={"el_a": element("el_a", Rect(0, 0, 8, 8), t1),
                      "el_b": element("el_b", Rect(390, 290, 8, 8),
                                      write_template(synth, Path("t/b.png"), 250))},
        )
        frame = np.zeros((300, 400, 3), dtype=np.uint8)
        # Якорь таков, что el_b вылетает за кадр.
        report = detector_for({"market": model}, RecordingMatcher(), synth).detect(
            frame, Anchor("market", 20, 20, 1.0, "template"))
        assert report.state is None
        assert "кадра" in report.failed[-1].detail


class TestModelIntegrity:
    """Ошибки модели (несуществующие элементы/регионы) падают при создании."""

    def test_unknown_element_signature_fails_at_init(self, synth):
        model = make_model(
            states={"s": state("s", [tpl_check("nope"), tpl_check("nope")])})
        with pytest.raises(KeyError, match="nope"):
            detector_for({"market": model}, RecordingMatcher(), synth)

    def test_unknown_ocr_region_fails_at_init(self, synth):
        t1 = write_template(synth, Path("t/a.png"), 250)
        model = make_model(
            states={"s": state("s", [
                tpl_check("el_a"),
                Check(kind="ocr", region="nope", expect="x"),
            ])},
            elements={"el_a": element("el_a", Rect(0, 0, 8, 8), t1)})
        with pytest.raises(KeyError, match="nope"):
            detector_for({"market": model}, RecordingMatcher(), synth,
                          ocr_client=FakeOcr())


@lru_cache(maxsize=None)
def frame_bgr(name):
    return cv2.imread(str(SCR_DIR / f"{name}.png"))


@pytest.fixture(scope="module")
def shipped_models():
    return {
        window_id: load_model(UI_MODEL_DIR / f"{window_id}.yaml")
        for window_id in ("market", "shop_buy", "shop_sell")
    }


@pytest.fixture(scope="module")
def real_detector(shipped_models):
    anchor = AnchorDetector.for_windows(shipped_models, OpenCvTemplateMatcher(), ASSETS_DIR)
    states = StateDetector(shipped_models, OpenCvTemplateMatcher(), ASSETS_DIR)
    return anchor, states


def detect_on(real_detector, name, window_id):
    anchor_detector, state_detector = real_detector
    frame = frame_bgr(name)
    anchor = anchor_detector.find_anchor(frame, window_id)
    assert anchor is not None, f"якорь {window_id} не найден на {name}"
    return state_detector.detect(frame, anchor)


class TestRealScreenshots:
    """Интеграция на живых скринах: окно по мете, params где мета
    подтверждает активный таб (description кропов)."""

    def test_market_home(self, real_detector):
        report = detect_on(real_detector, "scr_153551", "market")
        assert report.state == "list__tab_home"
        assert report.params == {"tab": "home"}

    def test_market_equipment(self, real_detector):
        report = detect_on(real_detector, "scr_153600", "market")
        assert report.state == "list__filter_all__global_filter_all__tab_equipment"
        assert report.params == {"filter": "all", "global_filter": "all",
                                 "tab": "equipment"}

    @pytest.mark.parametrize("name", ["scr_154330", "scr_154340"])
    def test_shop_buy_opening(self, real_detector, name):
        report = detect_on(real_detector, name, "shop_buy")
        assert report.state == "opening"

    def test_shop_sell_opening(self, real_detector):
        report = detect_on(real_detector, "scr_154213", "shop_sell")
        assert report.state == "opening"

    def test_market_untabbed_state_is_unknown(self, real_detector):
        """scr_153802: мета params tab=scroll, состояния таба scroll в
        модели нет -> честный unknown (а не уверенное чужое состояние)."""
        report = detect_on(real_detector, "scr_153802", "market")
        assert report.state is None


class TestEvaluateChecks:
    """Публичная оценка произвольного набора проверок (wiring 7.1).

    run_checks-провайдер сборки API (l2core/api/wiring.py) обязан оценивать
    verify-наборы переходов той же семантикой, что и сигнатуры состояний:
    тот же matcher/порог/допуск/OCR-клиент. Точка переиспользования —
    StateDetector.evaluate: без него wiring дублировал бы _evaluate.
    """

    def test_template_checks_use_signature_semantics(self, synth):
        t_ok = write_template(synth, Path("t/a.png"), 250)   # ~0.98 >= 0.9
        t_bad = write_template(synth, Path("t/b.png"), 100)  # ~0.39 < 0.9
        model = make_model(
            states={},
            elements={"el_a": element("el_a", Rect(0, 0, 8, 8), t_ok),
                      "el_b": element("el_b", Rect(20, 20, 8, 8), t_bad)},
        )
        detector = detector_for({"market": model}, RecordingMatcher(), synth)
        results = detector.evaluate(
            (tpl_check("el_a"), tpl_check("el_b")),
            np.zeros((300, 400, 3), dtype=np.uint8),
            Anchor("market", 0, 0, 1.0, "template"))
        assert [r.matched for r in results] == [True, False]
        assert results[0].name == "template:el_a/normal"
        assert "ncc=" in results[1].detail

    def test_ocr_check_without_client_is_unavailable(self, synth):
        model = make_model(
            states={},
            elements={"el_a": element("el_a", Rect(0, 0, 8, 8),
                                      write_template(synth, Path("t/a.png"), 250))},
            ocr_regions={"reg": OcrRegionSpec(
                id="reg", rect=Rect(0, 0, 8, 8),
                grammar=GrammarSpec(name="g", kind="text"))},
        )
        detector = detector_for({"market": model}, RecordingMatcher(), synth)
        results = detector.evaluate(
            (Check(kind="ocr", region="reg", expect="abc"),),
            np.zeros((300, 400, 3), dtype=np.uint8),
            Anchor("market", 0, 0, 1.0, "template"))
        assert [r.matched for r in results] == [None]
        assert "нет OCR-клиента" in results[0].detail

    def test_ocr_check_with_client_compares_text(self, synth):
        model = make_model(
            states={},
            elements={"el_a": element("el_a", Rect(0, 0, 8, 8),
                                      write_template(synth, Path("t/a.png"), 250))},
            ocr_regions={"reg": OcrRegionSpec(
                id="reg", rect=Rect(0, 0, 8, 8),
                grammar=GrammarSpec(name="g", kind="text"))},
        )
        detector = detector_for(
            {"market": model}, RecordingMatcher(), synth,
            ocr_client=FakeOcr(answers=[("abc", 0.99)]))
        results = detector.evaluate(
            (Check(kind="ocr", region="reg", expect="abc"),),
            np.zeros((300, 400, 3), dtype=np.uint8),
            Anchor("market", 0, 0, 1.0, "template"))
        assert [r.matched for r in results] == [True]

    def test_empty_checks_are_vacuously_true(self, synth):
        model = make_model(states={})
        detector = detector_for({"market": model}, RecordingMatcher(), synth)
        results = detector.evaluate(
            (), np.zeros((300, 400, 3), dtype=np.uint8),
            Anchor("market", 0, 0, 1.0, "template"))
        assert results == ()
