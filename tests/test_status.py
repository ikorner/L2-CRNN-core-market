"""Статусы окна торговли и признак прогрузки контента (задача 3.5).

Спека window-state-detection: статусы `not_found` / `occluded_or_unstable` /
`found` + признак «контент прогрузился» по контентным сигнатурам (не только
по заголовку). Модель статусов (status.py):

- not_found: окно игры недоступно (причина из находилки 3.2 — Win32) или
  якорь окна торговли не найден на кадре устойчиво (нет истории о нём);
- occluded_or_unstable: результаты детекции якоря РАЗЛИЧАЮТСЯ между двумя
  кадрами подряд — якорь был и пропал (anchor_lost), появился после пропуска
  (anchor_appeared) или позиция прыгнула сверх допуска (anchor_moved);
- found: якорь найден и стабилен относительно предыдущего кадра. Первый кадр
  без истории — found (нестабильность недоказуема, документировано).

Граница инварианта «кэша позиции нет» (anchor.py): детектор статусов не
пропускает поиск — каждый кадр AnchorDetector ищет якорь заново; внутреннее
состояние компонента — только факт результата предыдущего кадра для
сравнения стабильности (монитору история разрешена, design.md решение 8).

Контентные сигнатуры: template/pixel_probe-сигнатуры не-recover состояний,
КРОМЕ элемента-опоры якоря окна (заголовок/целое окно — не контент).
Флаг «контентности» в формат модели не добавлен: опора якоря уже задана
AnchorDetector'ом (DEFAULT_ANCHOR_TEMPLATES), повторная маркировка —
дублирование; обоснование — в отчёте 3.5.

Синтетика: scripted-заглушки находилки окна игры и детектора якоря
(детерминизм без cv2); контентные проверки — на реальном StateDetector с
моделями из объектных типов (яркость шаблона кодирует NCC-скор).
Интеграция: живые скрины base_screenshot/scr/ против меты.
"""

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pytest

from l2core.adapters.template import Match, OpenCvTemplateMatcher, TemplateMatcher
from l2core.adapters.win32 import (
    REASON_MINIMIZED,
    REASON_NO_CLIENT_AREA,
    REASON_NOT_FOUND,
    GameWindow,
)
from l2core.detector.anchor import Anchor, AnchorDetector
from l2core.detector.states import StateDetector
from l2core.detector.status import (
    STATUS_FOUND,
    STATUS_NOT_FOUND,
    STATUS_OCCLUDED,
    WindowStatusDetector,
)
from l2core.model import (
    Check,
    ElementSpec,
    Rect,
    StateSpec,
    UiModel,
    VariantSpec,
    WindowSpec,
    load_model,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SCR_DIR = REPO_ROOT / "base_screenshot" / "scr"
ASSETS_DIR = REPO_ROOT / "assets"
UI_MODEL_DIR = REPO_ROOT / "ui_model"


class ScriptedGameFinder:
    """Находилка окна игры со скриптованной последовательностью результатов.

    После исчерпания списка повторяет последний результат (окно обычно
    присутствует на всех кадрах сценария).
    """

    def __init__(self, games):
        assert games, "нужен хотя бы один результат окна игры"
        self._games = list(games)

    def find(self):
        if len(self._games) > 1:
            return self._games.pop(0)
        return self._games[0]


class ScriptedAnchorDetector:
    """Детектор якоря со скриптованной последовательностью результатов."""

    def __init__(self, anchors):
        self._anchors = list(anchors)
        self.calls = 0

    def find_anchor(self, frame, window_id):
        self.calls += 1
        if not self._anchors:
            raise AssertionError("скрипт якорей исчерпан")
        return self._anchors.pop(0)


def status_detector(games, anchors, **kwargs):
    """WindowStatusDetector поверх scripted-заглушек."""
    return WindowStatusDetector(
        ScriptedGameFinder(games), ScriptedAnchorDetector(anchors), **kwargs
    )


FRAME = np.zeros((1080, 1920, 3), dtype=np.uint8)


def anchor_at(x, y, window_id="market"):
    return Anchor(window_id=window_id, x=x, y=y, score=0.99, method="template")


class TestNotFound:
    """not_found: окно игры недоступно (причина из 3.2) или якоря устойчиво нет."""

    @pytest.mark.parametrize(
        "reason",
        [REASON_NOT_FOUND, REASON_MINIMIZED, REASON_NO_CLIENT_AREA],
    )
    def test_game_window_reasons_propagate(self, reason):
        """Причина отказа находилки 3.2 — в диагностике статуса."""
        game = GameWindow.failed(reason)
        detector = status_detector([game], [])
        report = detector.check(FRAME, "market")
        assert report.status == STATUS_NOT_FOUND
        assert report.reason == reason
        assert report.anchor is None
        assert report.content_loaded is None

    def test_game_window_missing_resets_history(self):
        """После пропадания окна игры история стирается: возврат не «мерцание»."""
        game = GameWindow.found(100, (0, 0, 1920, 1080))
        detector = status_detector(
            [game, GameWindow.failed(REASON_MINIMIZED), game],
            [anchor_at(207, 77), anchor_at(207, 77)],
        )
        assert detector.check(FRAME, "market").status == STATUS_FOUND
        lost = detector.check(FRAME, "market")
        assert lost.status == STATUS_NOT_FOUND
        assert lost.reason == REASON_MINIMIZED
        # Возврат окна: истории нет -> found, а не occluded.
        assert detector.check(FRAME, "market").status == STATUS_FOUND

    def test_anchor_absent_first_frame(self):
        game = GameWindow.found(100, (0, 0, 1920, 1080))
        detector = status_detector([game], [None])
        report = detector.check(FRAME, "market")
        assert report.status == STATUS_NOT_FOUND
        assert "anchor_not_found" in report.reason

    def test_anchor_absent_steady_is_not_occluded(self):
        """Устойчивое отсутствие якоря (два кадра подряд) — not_found."""
        game = GameWindow.found(100, (0, 0, 1920, 1080))
        detector = status_detector([game], [None, None])
        assert detector.check(FRAME, "market").status == STATUS_NOT_FOUND
        assert detector.check(FRAME, "market").status == STATUS_NOT_FOUND


class TestFound:
    """found: якорь найден и стабилен между кадрами."""

    def test_first_frame_without_history_is_found(self):
        game = GameWindow.found(100, (0, 0, 1920, 1080))
        detector = status_detector([game], [anchor_at(207, 77)])
        report = detector.check(FRAME, "market")
        assert report.status == STATUS_FOUND
        assert report.reason is None
        assert report.anchor == anchor_at(207, 77)

    def test_stable_anchor_between_frames(self):
        game = GameWindow.found(100, (0, 0, 1920, 1080))
        detector = status_detector([game], [anchor_at(207, 77), anchor_at(207, 77)])
        assert detector.check(FRAME, "market").status == STATUS_FOUND
        assert detector.check(FRAME, "market").status == STATUS_FOUND

    def test_small_jitter_within_tolerance_is_found(self):
        """Дрожание позиции в пределах допуска — всё ещё found."""
        game = GameWindow.found(100, (0, 0, 1920, 1080))
        detector = status_detector(
            [game], [anchor_at(207, 77), anchor_at(209, 79)],
            position_tolerance=8,
        )
        assert detector.check(FRAME, "market").status == STATUS_FOUND
        assert detector.check(FRAME, "market").status == STATUS_FOUND

    def test_tolerance_is_parameterized(self):
        game = GameWindow.found(100, (0, 0, 1920, 1080))
        detector = status_detector(
            [game], [anchor_at(207, 77), anchor_at(210, 80)],
            position_tolerance=2,
        )
        assert detector.check(FRAME, "market").status == STATUS_FOUND
        assert detector.check(FRAME, "market").status == STATUS_OCCLUDED


class TestOccludedOrUnstable:
    """occluded_or_unstable: результаты детекции якоря различаются между кадрами."""

    def test_anchor_lost_after_found(self):
        game = GameWindow.found(100, (0, 0, 1920, 1080))
        detector = status_detector([game], [anchor_at(207, 77), None])
        assert detector.check(FRAME, "market").status == STATUS_FOUND
        report = detector.check(FRAME, "market")
        assert report.status == STATUS_OCCLUDED
        assert "anchor_lost" in report.reason

    def test_anchor_appeared_after_miss(self):
        game = GameWindow.found(100, (0, 0, 1920, 1080))
        detector = status_detector([game], [None, anchor_at(207, 77)])
        assert detector.check(FRAME, "market").status == STATUS_NOT_FOUND
        report = detector.check(FRAME, "market")
        assert report.status == STATUS_OCCLUDED
        assert "anchor_appeared" in report.reason

    def test_position_jump_beyond_tolerance(self):
        game = GameWindow.found(100, (0, 0, 1920, 1080))
        detector = status_detector([game], [anchor_at(207, 77), anchor_at(400, 300)])
        assert detector.check(FRAME, "market").status == STATUS_FOUND
        report = detector.check(FRAME, "market")
        assert report.status == STATUS_OCCLUDED
        assert "anchor_moved" in report.reason

    def test_flickering_anchor_is_unstable(self, caplog):
        """Сценарий «Мерцающий якорь»: found/None через кадр -> occluded."""
        import logging

        game = GameWindow.found(100, (0, 0, 1920, 1080))
        detector = status_detector(
            [game], [anchor_at(207, 77), None, anchor_at(207, 77), None],
        )
        assert detector.check(FRAME, "market").status == STATUS_FOUND
        with caplog.at_level(logging.WARNING, logger="l2core.detector.status"):
            assert detector.check(FRAME, "market").status == STATUS_OCCLUDED
            assert detector.check(FRAME, "market").status == STATUS_OCCLUDED
            assert detector.check(FRAME, "market").status == STATUS_OCCLUDED
        messages = "\n".join(r.getMessage() for r in caplog.records)
        assert "anchor_lost" in messages and "anchor_appeared" in messages

    def test_stable_again_after_flicker(self):
        """После мерцания стабильная пара кадров снова даёт found."""
        game = GameWindow.found(100, (0, 0, 1920, 1080))
        detector = status_detector(
            [game],
            [anchor_at(207, 77), None, anchor_at(207, 77), anchor_at(207, 77)],
        )
        assert detector.check(FRAME, "market").status == STATUS_FOUND
        assert detector.check(FRAME, "market").status == STATUS_OCCLUDED
        assert detector.check(FRAME, "market").status == STATUS_OCCLUDED
        assert detector.check(FRAME, "market").status == STATUS_FOUND

    def test_window_id_change_resets_history(self):
        """Смена окна торговли — новая история: без «скачка позиции»."""
        game = GameWindow.found(100, (0, 0, 1920, 1080))
        detector = status_detector(
            [game], [anchor_at(207, 77, "market"), anchor_at(155, 224, "shop_buy")],
        )
        assert detector.check(FRAME, "market").status == STATUS_FOUND
        report = detector.check(FRAME, "shop_buy")
        assert report.status == STATUS_FOUND
        assert report.anchor.x == 155


# --- Контентные сигнатуры (StateDetector.check_content) ---------------------


def make_model(*, states, elements, window_id="market", size=(400, 300)):
    return UiModel(
        path="synthetic",
        window=WindowSpec(id=window_id, title="Синтетика", size=size),
        elements=elements,
        states=states,
        transitions=(),
        ocr_regions={},
        table_geometry=None,
    )


def tpl_check(element, variant="normal"):
    return Check(kind="template", element=element, variant=variant)


@pytest.fixture()
def synth(tmp_path):
    """Каталог assets для синтетических шаблонов."""
    assets = tmp_path / "assets"
    assets.mkdir()
    return assets


def state(sid, signatures):
    return StateSpec(id=sid, params={}, signatures=tuple(signatures))


def element(eid, rect, template, variant="normal"):
    return ElementSpec(
        id=eid, role="click_target", rect=rect,
        variants={variant: VariantSpec(name=variant, template=template)},
    )


def write_template(assets, rel_path, brightness):
    """Константный серый PNG; яркость кодирует желаемый NCC-скор."""
    path = assets / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((8, 8), brightness, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)
    return rel_path.as_posix()


class RecordingMatcher(TemplateMatcher):
    """Matcher-заглушка: score = яркость шаблона / 255."""

    def match(self, image, template):
        return Match(x=0, y=0, score=float(np.asarray(template)[0, 0]) / 255.0)


def content_model(synth, *, title_brightness=250, content_brightness=250):
    """Модель: опора якоря title_el + контент tab_el (состояние из двух)."""
    t_title = write_template(synth, Path("t/title.png"), title_brightness)
    t_tab = write_template(synth, Path("t/tab.png"), content_brightness)
    return make_model(
        states={"list__tab_home": state(
            "list__tab_home", [tpl_check("title_el"), tpl_check("tab_el")])},
        elements={
            "title_el": element("title_el", Rect(0, 0, 8, 8), t_title),
            "tab_el": element("tab_el", Rect(20, 20, 8, 8), t_tab),
        },
    )


class TestContentSignatures:
    """«Контент прогрузился»: контентные сигнатуры, а не заголовок."""

    def test_content_matched_means_loaded(self, synth):
        model = content_model(synth, content_brightness=250)
        detector = StateDetector({"market": model}, RecordingMatcher(), synth)
        report = detector.check_content(
            np.zeros((300, 400, 3), dtype=np.uint8),
            Anchor("market", 0, 0, 1.0, "template"),
            exclude_elements={"title_el"},
        )
        assert report.loaded is True
        assert [r.name for r in report.matched] == ["template:tab_el/normal"]
        assert not report.ocr_degraded

    def test_only_title_present_means_not_loaded(self, synth):
        """Заголовок есть, контентных признаков нет — «контент не прогружен»."""
        model = content_model(synth, content_brightness=40)
        detector = StateDetector({"market": model}, RecordingMatcher(), synth)
        report = detector.check_content(
            np.zeros((300, 400, 3), dtype=np.uint8),
            Anchor("market", 0, 0, 1.0, "template"),
            exclude_elements={"title_el"},
        )
        assert report.loaded is False
        assert [r.name for r in report.failed] == ["template:tab_el/normal"]

    def test_without_exclusion_title_counts(self, synth):
        """Ответственность за исключение заголовка — у вызывающей стороны."""
        model = content_model(synth, content_brightness=40)
        detector = StateDetector({"market": model}, RecordingMatcher(), synth)
        report = detector.check_content(
            np.zeros((300, 400, 3), dtype=np.uint8),
            Anchor("market", 0, 0, 1.0, "template"),
        )
        assert report.loaded is True

    def test_no_content_signatures_is_unknown(self, synth):
        """Все сигнатуры — опоры якоря: прогрузку проверить нечем (None)."""
        t_title = write_template(synth, Path("t/title.png"), 250)
        t_title2 = write_template(synth, Path("t/title2.png"), 250)
        model = make_model(
            states={"list__tab_home": state(
                "list__tab_home", [tpl_check("title_el"), tpl_check("title2_el")])},
            elements={
                "title_el": element("title_el", Rect(0, 0, 8, 8), t_title),
                "title2_el": element("title2_el", Rect(0, 20, 8, 8), t_title2),
            },
        )
        detector = StateDetector({"market": model}, RecordingMatcher(), synth)
        report = detector.check_content(
            np.zeros((300, 400, 3), dtype=np.uint8),
            Anchor("market", 0, 0, 1.0, "template"),
            exclude_elements={"title_el", "title2_el"},
        )
        assert report.loaded is None

    def test_recover_state_signatures_not_content(self, synth):
        """recover_* — глобальные признаки (HUD), не контент окна."""
        t_title = write_template(synth, Path("t/title.png"), 40)
        t_hud = write_template(synth, Path("t/hud.png"), 250)
        model = make_model(
            states={
                "list__tab_home": state(
                    "list__tab_home", [tpl_check("title_el"), tpl_check("tab_el")]),
                "recover_closed": state(
                    "recover_closed", [tpl_check("hud_el"), tpl_check("tab_el")]),
            },
            elements={
                "title_el": element("title_el", Rect(0, 0, 8, 8), t_title),
                "tab_el": element("tab_el", Rect(20, 20, 8, 8),
                                  write_template(synth, Path("t/tab.png"), 40)),
                "hud_el": element("hud_el", Rect(0, 40, 8, 8), t_hud),
            },
        )
        detector = StateDetector({"market": model}, RecordingMatcher(), synth)
        report = detector.check_content(
            np.zeros((300, 400, 3), dtype=np.uint8),
            Anchor("market", 0, 0, 1.0, "template"),
            exclude_elements={"title_el"},
        )
        # Совпал только HUD-признак recover-состояния — контент не прогружен.
        assert report.loaded is False


# --- Статус + контент --------------------------------------------------------


class TestStatusContentFlow:
    """Признак «контент прогрузился» в статусе окна."""

    def _detector(self, anchors, *, assets=None, states=None, **kwargs):
        game = GameWindow.found(100, (0, 0, 1920, 1080))
        state_detector = None
        if states is not None:
            state_detector = StateDetector(
                {"market": states}, RecordingMatcher(), assets)
        return status_detector(
            [game], anchors, state_detector=state_detector,
            anchor_templates={"market": ("title_el", "normal")}, **kwargs)

    def test_found_carries_content_flag(self, synth):
        model = content_model(synth, content_brightness=250)
        detector = self._detector(
            [anchor_at(0, 0)], assets=synth, states=model)
        report = detector.check(np.zeros((300, 400, 3), dtype=np.uint8), "market")
        assert report.status == STATUS_FOUND
        assert report.content_loaded is True
        assert "tab_el" in report.content_detail

    def test_found_with_unloaded_content(self, synth):
        """Заголовок есть, контент не прогрузился — found + content_loaded=False."""
        model = content_model(synth, content_brightness=40)
        detector = self._detector(
            [anchor_at(0, 0)], assets=synth, states=model)
        report = detector.check(np.zeros((300, 400, 3), dtype=np.uint8), "market")
        assert report.status == STATUS_FOUND
        assert report.content_loaded is False

    def test_occluded_has_no_content_flag(self, synth):
        model = content_model(synth)
        detector = self._detector(
            [anchor_at(0, 0), None], assets=synth, states=model)
        detector.check(np.zeros((300, 400, 3), dtype=np.uint8), "market")
        report = detector.check(np.zeros((300, 400, 3), dtype=np.uint8), "market")
        assert report.status == STATUS_OCCLUDED
        assert report.content_loaded is None

    def test_without_state_detector_content_unknown(self):
        """Деградация: нет детектора состояний — прогрузка не проверяется."""
        detector = self._detector([anchor_at(0, 0)])
        report = detector.check(FRAME, "market")
        assert report.status == STATUS_FOUND
        assert report.content_loaded is None
        assert "детектора состояний" in report.content_detail


# --- Интеграция на живых скринах ---------------------------------------------


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
def real_status_detector(shipped_models):
    matcher = OpenCvTemplateMatcher()
    anchor_detector = AnchorDetector.for_windows(shipped_models, matcher, ASSETS_DIR)
    state_detector = StateDetector(shipped_models, matcher, ASSETS_DIR)
    return WindowStatusDetector(
        _ConstantGameFinder(GameWindow.found(100, (0, 0, 1920, 1080))),
        anchor_detector,
        state_detector=state_detector,
    )


class _ConstantGameFinder:
    """Окно игры присутствует: интеграционные скрины — сам клиентский кадр."""

    def __init__(self, game):
        self._game = game

    def find(self):
        return self._game


class TestRealScreenshots:
    """Ground truth — meta.json: окна market открыты на батчах 1/2 и лавках,
    скрины scr_153511/scr_153527/scr_all_close — без окна торговли."""

    @pytest.fixture(autouse=True)
    def _reset_history(self, real_status_detector):
        """Детектор модульный: история стабильности не перетекает между тестами."""
        real_status_detector.reset()

    @pytest.mark.parametrize("name,window_id", [
        ("scr_153551", "market"), ("scr_153600", "market"),  # якорь (207, 77)
        ("scr_153901", "market"),  # якорь (202, 103)
        ("scr_154330", "shop_buy"),
        ("scr_154213", "shop_sell"),
    ])
    def test_open_window_is_found_with_content(self, real_status_detector,
                                               name, window_id):
        frame = frame_bgr(name)
        first = real_status_detector.check(frame, window_id)
        assert first.status == STATUS_FOUND, (
            f"{name}: {first.status} ({first.reason})"
        )
        assert first.content_loaded is True, first.content_detail
        second = real_status_detector.check(frame, window_id)
        assert second.status == STATUS_FOUND
        assert second.content_loaded is True

    @pytest.mark.parametrize("name", ["scr_153511", "scr_153527", "scr_all_close"])
    def test_closed_window_is_not_found(self, real_status_detector, name):
        report = real_status_detector.check(frame_bgr(name), "market")
        assert report.status == STATUS_NOT_FOUND
        assert "anchor_not_found" in report.reason

    def test_window_move_between_frames_is_unstable(self, real_status_detector):
        """Два размеченных положения окна обзора: скачок (5, 26) > допуска."""
        first = real_status_detector.check(frame_bgr("scr_153551"), "market")
        assert first.status == STATUS_FOUND
        second = real_status_detector.check(frame_bgr("scr_153901"), "market")
        assert second.status == STATUS_OCCLUDED
        assert "anchor_moved" in second.reason

