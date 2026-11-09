"""Сборка провайдеров API из детекторов и адаптеров (задача 7.1,
l2core/api/wiring.py): первая точка, где провайдеры сценариев 6.x
(find_anchor/window_status/run_checks/goal_reached/current_window/
close_point/hud_locator/...) собираются из реальных детекторов 3.4/3.5.

Тесты — на скриптованных кадрах и фейковых детекторах/матчере: проверяем
математику провайдеров (close_point от якоря + rect, hud_locator от
search_area + match + click_point), делегирование детекторам и дефолты
здоровья; create_api — с подменёнными живыми адаптерами (MssScreen/
HidDevice/TorchOcr/PyWin32Gateway), реальные модели из ui_model/.

Детерминизм (инвариант 7): фейковый matcher возвращает заданный Match,
константные кадры, никаких живых окон/экранов/портов.
"""
from pathlib import Path

import cv2
import numpy as np
import pytest

from l2core.adapters.template import Match, TemplateMatcher
from l2core.adapters.win32 import GameWindow
from l2core.api.core import L2CoreAPI
from l2core.api.wiring import CLOSE_ELEMENTS, build_wiring, create_api
from l2core.config import load_config
from l2core.detector.anchor import Anchor
from l2core.detector.states import ContentReport, SignatureResult, StateReport
from l2core.detector.status import STATUS_FOUND, STATUS_NOT_FOUND
from l2core.model import ElementSpec, Rect, UiModel, VariantSpec, WindowSpec

FRAME = np.zeros((1080, 1920, 3), dtype=np.uint8)
MARKET_ANCHOR = Anchor(window_id="market", x=100, y=100, score=0.99,
                       method="template")
SELL_ANCHOR = Anchor(window_id="shop_sell", x=200, y=150, score=0.99,
                     method="template")


def make_model(window_id, elements=None, size=(1000, 700)):
    return UiModel(
        path=window_id, window=WindowSpec(id=window_id, title="t",
                                          size=size),
        elements=elements or {}, states={}, transitions=(),
        ocr_regions={}, table_geometry=None)


def make_models():
    market = make_model("market", elements={
        "main_market_close_button": ElementSpec(
            id="main_market_close_button", role="click_target",
            rect=Rect(10, 10, 20, 20),
            variants={"normal": VariantSpec(name="normal", template="t/a.png")}),
    })
    shop_buy = make_model("shop_buy", size=(400, 300))
    shop_sell = make_model("shop_sell", elements={
        "close_sell_button": ElementSpec(
            id="close_sell_button", role="click_target",
            rect=Rect(5, 5, 10, 10),
            variants={"normal": VariantSpec(name="normal", template="t/b.png")}),
    }, size=(400, 300))
    return {"market": market, "shop_buy": shop_buy, "shop_sell": shop_sell}


class FakeMatcher(TemplateMatcher):
    """Matcher-заглушка: заданный Match на каждый вызов, вызовы считаются."""

    def __init__(self, result):
        self.result = result
        self.calls = 0

    def match(self, image, template):
        self.calls += 1
        return self.result


class FakeAnchorDetector:
    def __init__(self, anchors):
        self._anchors = anchors
        self.calls = []

    def find_anchor(self, frame, window_id):
        self.calls.append(window_id)
        return self._anchors.get(window_id)


class FakeStateDetector:
    def __init__(self, report=None, matched=True):
        self._report = report
        self._matched = matched
        self.detect_calls = []
        self.evaluate_calls = []

    def detect(self, frame, anchor):
        self.detect_calls.append(anchor.window_id)
        return self._report

    def evaluate(self, checks, frame, anchor):
        self.evaluate_calls.append((checks, anchor.window_id))
        return tuple(SignatureResult(
            state_id="", kind=c.kind, name=f"check:{c.kind}",
            matched=self._matched if c.kind else True, detail="ok")
            for c in checks)

    def check_content(self, frame, anchor, exclude_elements=()):
        return ContentReport(loaded=True, matched=(), failed=(), skipped=(),
                             ocr_degraded=False)


class FakeGameFinder:
    def __init__(self, window):
        self._window = window
        self.calls = 0

    def find(self):
        self.calls += 1
        return self._window


class FakeHid:
    def __init__(self, connected=True):
        self.log = []
        self._connected = connected

    def is_connected(self):
        return self._connected


class FakeOcr:
    def read(self, region_img):
        return ("", 0.0)


STATE_REPORT = StateReport(
    window_id="market", state="list__tab_home", params={"tab": "home"},
    matched=(), failed=(), conflicts=(), ocr_degraded=False)


def make_wiring(anchors=None, matcher=None, game=None, hid=None,
                ocr_health=None, hid_health=None, hud_threshold=0.9,
                tmp_assets=None):
    return build_wiring(
        config=load_config(None),
        models=make_models(),
        screen=None, hid=hid or FakeHid(), ocr=FakeOcr(),
        matcher=matcher or FakeMatcher(Match(0, 0, 1.0)),
        assets_dir=tmp_assets or Path("assets"),
        cursor_pos=lambda: (0, 0),
        activate=lambda: None,
        game_finder=game or FakeGameFinder(
            GameWindow.found(1, (0, 0, 1920, 1080))),
        anchor_detector=FakeAnchorDetector(anchors or {}),
        state_detector=FakeStateDetector(report=STATE_REPORT),
        ocr_health=ocr_health,
        hid_health=hid_health,
        hud_threshold=hud_threshold,
    )


class TestProviderDelegation:
    def test_find_anchor_delegates_with_window(self):
        detector = FakeAnchorDetector({"market": MARKET_ANCHOR})
        wiring = build_wiring(
            config=load_config(None), models=make_models(), screen=None,
            hid=FakeHid(), ocr=FakeOcr(),
            matcher=FakeMatcher(Match(0, 0, 1.0)), assets_dir=Path("assets"),
            cursor_pos=lambda: (0, 0), activate=lambda: None,
            game_finder=FakeGameFinder(GameWindow.found(1, (0, 0, 1920, 1080))),
            anchor_detector=detector,
            state_detector=FakeStateDetector(report=STATE_REPORT))

        assert wiring.find_anchor(FRAME, "market") is MARKET_ANCHOR
        assert detector.calls == ["market"]

    def test_window_status_built_on_status_detector(self):
        wiring = make_wiring(anchors={"market": MARKET_ANCHOR})

        status = wiring.window_status(FRAME, "market")

        assert status.status == STATUS_FOUND
        assert status.anchor is MARKET_ANCHOR
        assert status.content_loaded is True   # контент проверил state detector

    def test_window_status_not_found_without_game(self):
        wiring = make_wiring(
            anchors={"market": MARKET_ANCHOR},
            game=FakeGameFinder(GameWindow.failed("window_not_found")))

        status = wiring.window_status(FRAME, "market")

        assert status.status == STATUS_NOT_FOUND
        assert status.reason == "window_not_found"

    def test_run_checks_uses_state_detector_evaluate(self):
        state = FakeStateDetector(report=STATE_REPORT)
        wiring = build_wiring(
            config=load_config(None), models=make_models(), screen=None,
            hid=FakeHid(), ocr=FakeOcr(),
            matcher=FakeMatcher(Match(0, 0, 1.0)), assets_dir=Path("assets"),
            cursor_pos=lambda: (0, 0), activate=lambda: None,
            game_finder=FakeGameFinder(GameWindow.found(1, (0, 0, 1920, 1080))),
            anchor_detector=FakeAnchorDetector({}),
            state_detector=state)

        from l2core.model import Check
        report = wiring.run_checks(
            (Check(kind="template", element="main_market_close_button",
                   variant="normal"),), FRAME, MARKET_ANCHOR)

        assert report.passed
        assert state.evaluate_calls[0][1] == "market"

    def test_run_checks_fails_when_signature_unmatched(self):
        wiring = build_wiring(
            config=load_config(None), models=make_models(), screen=None,
            hid=FakeHid(), ocr=FakeOcr(),
            matcher=FakeMatcher(Match(0, 0, 1.0)), assets_dir=Path("assets"),
            cursor_pos=lambda: (0, 0), activate=lambda: None,
            game_finder=FakeGameFinder(GameWindow.found(1, (0, 0, 1920, 1080))),
            anchor_detector=FakeAnchorDetector({}),
            state_detector=FakeStateDetector(report=STATE_REPORT, matched=False))

        from l2core.model import Check
        report = wiring.run_checks(
            (Check(kind="template", element="x", variant="normal"),),
            FRAME, MARKET_ANCHOR)

        assert not report.passed
        assert report.failures

    def test_detect_state_delegates(self):
        state = FakeStateDetector(report=STATE_REPORT)
        wiring = build_wiring(
            config=load_config(None), models=make_models(), screen=None,
            hid=FakeHid(), ocr=FakeOcr(),
            matcher=FakeMatcher(Match(0, 0, 1.0)), assets_dir=Path("assets"),
            cursor_pos=lambda: (0, 0), activate=lambda: None,
            game_finder=FakeGameFinder(GameWindow.found(1, (0, 0, 1920, 1080))),
            anchor_detector=FakeAnchorDetector({}),
            state_detector=state)

        report = wiring.detect_state(FRAME, MARKET_ANCHOR)

        assert report is STATE_REPORT
        assert state.detect_calls == ["market"]

    def test_current_window_picks_anchored_window(self):
        wiring = make_wiring(anchors={"shop_sell": SELL_ANCHOR})
        assert wiring.current_window(FRAME) == "shop_sell"

        wiring = make_wiring(anchors={})
        assert wiring.current_window(FRAME) is None

    def test_goal_reached_is_anchor_presence(self):
        wiring = make_wiring(anchors={"shop_sell": SELL_ANCHOR})
        assert wiring.goal_reached("shop_sell")(FRAME) is True
        assert wiring.goal_reached("market")(FRAME) is False


class TestClosePoint:
    def test_close_point_computed_from_anchor_and_rect(self):
        wiring = make_wiring(anchors={"market": MARKET_ANCHOR,
                                      "shop_sell": SELL_ANCHOR})

        assert wiring.close_point("market", FRAME) == (120, 120)
        assert wiring.close_point("shop_sell", FRAME) == (210, 160)

    def test_close_point_none_for_shop_buy_and_missing_anchor(self):
        wiring = make_wiring(anchors={"market": MARKET_ANCHOR})
        # shop_buy: кнопки закрытия в модели нет — fallback 5.8 (ESC).
        assert wiring.close_point("shop_buy", FRAME) is None
        # Якоря нет — точку закрытия не вычислить (слепых кликов нет).
        assert wiring.close_point("shop_sell", FRAME) is None

    def test_close_elements_map_matches_models(self):
        # Конвенция: market -> main_market_close_button,
        # shop_sell -> close_sell_button (обе есть в отгруженных моделях).
        assert CLOSE_ELEMENTS["market"] == "main_market_close_button"
        assert CLOSE_ELEMENTS["shop_sell"] == "close_sell_button"
        assert "shop_buy" not in CLOSE_ELEMENTS


class TestHealthProviders:
    def test_health_defaults(self):
        wiring = make_wiring(hid=FakeHid(connected=True))
        assert wiring.game_window().ok
        assert wiring.ocr_health()
        assert wiring.hid_health()

    def test_hid_health_uses_is_connected(self):
        wiring = make_wiring(hid=FakeHid(connected=False))
        assert not wiring.hid_health()


class TestHudLocator:
    def _hud_models(self, assets, click_point=None):
        spec = VariantSpec(name="normal", template="hud/mp.png",
                           click_point=click_point)
        market = make_model("market", elements={
            "market_pic": ElementSpec(
                id="market_pic", role="icon_sample", hud=True,
                search_area=Rect(1000, 420, 200, 160),
                variants={"normal": spec}),
        })
        models = make_models()
        models["market"] = market
        return models

    def test_hud_point_from_match_and_click_point(self, tmp_path):
        assets = tmp_path / "assets"
        (assets / "hud").mkdir(parents=True)
        assert cv2.imwrite(str(assets / "hud" / "mp.png"),
                           np.full((8, 8), 200, dtype=np.uint8))
        models = self._hud_models(assets, click_point=(86, 64))
        matcher = FakeMatcher(Match(5, 7, 0.99))
        wiring = build_wiring(
            config=load_config(None), models=models, screen=None,
            hid=FakeHid(), ocr=FakeOcr(), matcher=matcher,
            assets_dir=assets, cursor_pos=lambda: (0, 0),
            activate=lambda: None,
            game_finder=FakeGameFinder(GameWindow.found(1, (0, 0, 1920, 1080))),
            anchor_detector=FakeAnchorDetector({}),
            state_detector=FakeStateDetector(report=STATE_REPORT))
        element = models["market"].elements["market_pic"]

        point = wiring.hud_locator(element, "normal", FRAME)

        # search_area (1000,420) + match (5,7) + click_point (86,64).
        assert point == (1091, 491)
        assert matcher.calls == 1

    def test_hud_point_defaults_to_template_center(self, tmp_path):
        assets = tmp_path / "assets"
        (assets / "hud").mkdir(parents=True)
        assert cv2.imwrite(str(assets / "hud" / "mp.png"),
                           np.full((8, 8), 200, dtype=np.uint8))
        models = self._hud_models(assets)
        wiring = build_wiring(
            config=load_config(None), models=models, screen=None,
            hid=FakeHid(), ocr=FakeOcr(),
            matcher=FakeMatcher(Match(0, 0, 1.0)), assets_dir=assets,
            cursor_pos=lambda: (0, 0), activate=lambda: None,
            game_finder=FakeGameFinder(GameWindow.found(1, (0, 0, 1920, 1080))),
            anchor_detector=FakeAnchorDetector({}),
            state_detector=FakeStateDetector(report=STATE_REPORT))
        element = models["market"].elements["market_pic"]

        point = wiring.hud_locator(element, "normal", FRAME)

        assert point == (1000 + 4, 420 + 4)   # центр шаблона 8x8

    def test_hud_low_score_returns_none(self, tmp_path):
        assets = tmp_path / "assets"
        (assets / "hud").mkdir(parents=True)
        assert cv2.imwrite(str(assets / "hud" / "mp.png"),
                           np.full((8, 8), 200, dtype=np.uint8))
        models = self._hud_models(assets)
        wiring = build_wiring(
            config=load_config(None), models=models, screen=None,
            hid=FakeHid(), ocr=FakeOcr(),
            matcher=FakeMatcher(Match(0, 0, 0.1)), assets_dir=assets,
            cursor_pos=lambda: (0, 0), activate=lambda: None,
            game_finder=FakeGameFinder(GameWindow.found(1, (0, 0, 1920, 1080))),
            anchor_detector=FakeAnchorDetector({}),
            state_detector=FakeStateDetector(report=STATE_REPORT))
        element = models["market"].elements["market_pic"]

        assert wiring.hud_locator(element, "normal", FRAME) is None


class TestCreateApi:
    """Фабрика create_api собирает живые адаптеры (патчим их фейками);
    реальные UI-модели из ui_model/, wiring собран полностью."""

    def test_create_api_builds_facade_with_real_models(self, tmp_path,
                                                       monkeypatch):
        import l2core.api.wiring as wiring_module

        class FakeScreen:
            def grab(self):
                return FRAME

            def close(self):
                pass

        class FakeHidDevice:
            def __init__(self, port=None, *args, **kwargs):
                self.port = port
                self.connected = False

            def connect(self):
                self.connected = True

            def is_connected(self):
                return self.connected

        class FakeGateway:
            def find_windows(self, title, class_name):
                return [1]

            def get_client_rect(self, hwnd):
                return (0, 0, 1920, 1080)

            def is_iconic(self, hwnd):
                return False

            def get_foreground_window(self):
                return 1

            def set_foreground(self, hwnd):
                return True

            def set_process_dpi_aware(self):
                return True

            def get_cursor_pos(self):
                return (100, 200)

        monkeypatch.setattr(wiring_module, "MssScreen", FakeScreen)
        monkeypatch.setattr(wiring_module, "HidDevice", FakeHidDevice)
        monkeypatch.setattr(wiring_module, "create_ocr_client",
                            lambda *a, **kw: FakeOcr())
        monkeypatch.setattr(wiring_module, "PyWin32Gateway", FakeGateway)

        repo = Path(__file__).resolve().parent.parent
        api = create_api(
            load_config(None),
            models_dir=repo / "ui_model",
            assets_dir=repo / "assets",
        )

        assert isinstance(api, L2CoreAPI)
        assert api._w.market.window.id == "market"
        assert api._w.shop_sell.window.id == "shop_sell"
        assert api._w.shop_buy.window.id == "shop_buy"
        # health() агрегирует провайдеров собранного wiring.
        health = api.health()
        assert health.game_window
