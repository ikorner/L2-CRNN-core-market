"""Перехватчик неизвестных модалок `modal.unknown` (задача 3.6).

Спека window-state-detection («Перехватчик неизвестных модалок»): контур/
текст вне известных окон → снимок в лог, состояние modal.unknown, точка
входа recover (исполнитель — 5.8, здесь только событие/флаг).

R4 (честность): из сырья не выводится надёжный детектор «контур/текст вне
окон». Реализован честный минимум — консервативная эвристика (l2core/
detector/modal.py): окно игры есть, ни один якорь известных окон не найден,
при этом в центральной области окна игры — крупный яркий прямоугольный
регион (связный компонент ярких пикселей выше порога площади/заполнения).
Ограничения документированы в модуле и отчёте 3.6.

Контракт:
- check(frame, game, anchors) -> ModalReport: detected/state/region/reason/
  snapshot_path; срабатывание → PNG-снимок в log_dir + JSONL-строка
  (координаты, причина, имя снимка) + запись в лог (координаты/причина);
- якорь любого известного окна найден → эвристика не применяется;
- окно игры недоступно → эвристика не применяется (модалка неотличима);
- on_unknown — опциональный callback (точка входа recover 5.8), YAGNI:
  исполнителя recover здесь нет.

Синтетика — numpy-кадры без cv2 (детерминизм); интеграция — живые скрины
base_screenshot/scr/ против реального AnchorDetector (нет ложных
срабатываний на скринах с известным окном).
"""

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import pytest

from l2core.adapters.template import OpenCvTemplateMatcher
from l2core.adapters.win32 import GameWindow
from l2core.detector.anchor import AnchorDetector
from l2core.detector.modal import MODAL_UNKNOWN, ModalInterceptor
from l2core.model import load_model

REPO_ROOT = Path(__file__).resolve().parent.parent
SCR_DIR = REPO_ROOT / "base_screenshot" / "scr"
ASSETS_DIR = REPO_ROOT / "assets"
UI_MODEL_DIR = REPO_ROOT / "ui_model"

GAME_OK = GameWindow.found(1, (0, 0, 1920, 1080))
NO_ANCHORS = {"market": None, "shop_buy": None, "shop_sell": None}

# Скрины, на которых по мете открыто известное окно торговли: эвристика
# обязана молчать (якорь найден — проверка не применяется).
KNOWN_WINDOW = [
    "scr_153551", "scr_153600", "scr_153611", "scr_153619", "scr_153631",
    "scr_153646", "scr_153656", "scr_153712", "scr_153722", "scr_153731",
    "scr_153740", "scr_153750", "scr_153802", "scr_153810", "scr_153819",
    "scr_153830", "scr_153840",
    "scr_153901", "scr_153929", "scr_153941", "scr_153949", "scr_154002",
    "scr_154012", "scr_154021", "scr_154033", "scr_154049", "scr_154103",
    "scr_154134",
    "scr_154213", "scr_154222", "scr_154244",
    "scr_154330", "scr_154340", "scr_154355",
]

# Скрины без известного якоря: поведение эвристики зафиксировано замером
# (characterization test, см. отчёт 3.6 — на части кадров игровой клиент
# даёт крупный яркий регион, эвристика честно срабатывает; это не ложный
# позитив «известного окна», а documented behavior эвристики).
NO_KNOWN_ANCHOR = [
    "scr_153511", "scr_153527", "scr_all_close",
    "scr_154318", "scr_154424",
]
# Ожидаемое поведение на NO_KNOWN_ANCHOR (замер task-3.6): эвристика НЕ
# стреляет ни на одном из 39 реальных кадров — ни шума на известных окнах,
# ни кандидатов на играбельных кадрах без окна торговли (лавки «open»,
# закрытый обзор). Эвристика видит «крупное яркое пятно», реальный клиент
# его в центральной области не даёт.
EXPECTED_ON_NO_ANCHOR = {
    "scr_153511": False,
    "scr_153527": False,
    "scr_all_close": False,
    "scr_154318": False,
    "scr_154424": False,
}


def dark_frame(w=1920, h=1080, value=30):
    return np.full((h, w, 3), value, dtype=np.uint8)


def bright_rect(frame, x, y, w, h, value=220):
    frame[y:y + h, x:x + w] = value
    return frame


@pytest.fixture()
def interceptor(tmp_path):
    return ModalInterceptor(tmp_path)


def frame_with_modal():
    """Тёмный клиент + крупный яркий прямоугольник в центре."""
    return bright_rect(dark_frame(), 660, 340, 600, 400)


class TestHeuristic:
    def test_unknown_modal_detected(self, interceptor):
        report = interceptor.check(frame_with_modal(), GAME_OK, NO_ANCHORS)
        assert report.detected
        assert report.state == MODAL_UNKNOWN
        assert report.region == (660, 340, 600, 400)
        assert report.reason

    def test_clean_frame_no_detection(self, interceptor, tmp_path):
        report = interceptor.check(dark_frame(), GAME_OK, NO_ANCHORS)
        assert not report.detected
        assert report.state is None
        assert report.region is None
        assert list(tmp_path.iterdir()) == []

    def test_small_bright_region_ignored(self, interceptor):
        """Мелкий яркий регион (кнопка/иконка) — не модалка."""
        frame = bright_rect(dark_frame(), 900, 500, 120, 80)
        assert not interceptor.check(frame, GAME_OK, NO_ANCHORS).detected

    def test_offcenter_region_ignored(self, interceptor):
        """Яркий регион вне центральной области — не кандидат."""
        frame = bright_rect(dark_frame(), 60, 60, 600, 400)
        assert not interceptor.check(frame, GAME_OK, NO_ANCHORS).detected

    def test_anchor_found_suppresses_heuristic(self, interceptor):
        from l2core.detector.anchor import Anchor
        anchors = dict(NO_ANCHORS)
        anchors["market"] = Anchor("market", 207, 77, 0.99, "template")
        report = interceptor.check(frame_with_modal(), GAME_OK, anchors)
        assert not report.detected

    def test_game_window_missing_suppresses_heuristic(self, interceptor):
        from l2core.adapters.win32 import REASON_MINIMIZED
        game = GameWindow.failed(REASON_MINIMIZED)
        report = interceptor.check(frame_with_modal(), game, NO_ANCHORS)
        assert not report.detected


class TestLogging:
    def test_snapshot_png_written(self, interceptor, tmp_path):
        report = interceptor.check(frame_with_modal(), GAME_OK, NO_ANCHORS)
        assert report.snapshot_path is not None
        assert report.snapshot_path.exists()
        saved = cv2.imread(str(report.snapshot_path))
        assert saved is not None
        assert saved.shape == (1080, 1920, 3)

    def test_jsonl_contains_coords_and_reason(self, interceptor, tmp_path):
        report = interceptor.check(frame_with_modal(), GAME_OK, NO_ANCHORS)
        lines = (tmp_path / "modal.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["event"] == MODAL_UNKNOWN
        assert record["region"] == [660, 340, 600, 400]
        assert record["reason"] == report.reason
        assert record["snapshot"] == report.snapshot_path.name

    def test_log_record_has_coords_and_reason(self, interceptor, caplog):
        with caplog.at_level(logging.WARNING, logger="l2core.detector.modal"):
            interceptor.check(frame_with_modal(), GAME_OK, NO_ANCHORS)
        messages = [r.getMessage() for r in caplog.records]
        assert any("modal.unknown" in m and "660" in m and "340" in m for m in messages)
        assert any("причина" in m or "region" in m for m in messages)

    def test_no_files_on_clean_frame(self, interceptor, tmp_path):
        interceptor.check(dark_frame(), GAME_OK, NO_ANCHORS)
        assert list(tmp_path.iterdir()) == []

    def test_repeated_detections_append_and_count(self, interceptor, tmp_path):
        first = interceptor.check(frame_with_modal(), GAME_OK, NO_ANCHORS)
        second = interceptor.check(frame_with_modal(), GAME_OK, NO_ANCHORS)
        assert first.snapshot_path != second.snapshot_path
        lines = (tmp_path / "modal.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2

    def test_on_unknown_callback_is_entry_point(self, tmp_path):
        """Точка входа recover 5.8: callback получает отчёт о срабатывании."""
        calls = []
        interceptor = ModalInterceptor(tmp_path, on_unknown=calls.append)
        report = interceptor.check(frame_with_modal(), GAME_OK, NO_ANCHORS)
        assert calls == [report]


class TestRealScreenshots:
    """Интеграция: реальные скрины, реальный AnchorDetector (39 кадров)."""

    @pytest.fixture(scope="class")
    def anchors_by_frame(self):
        models = {
            window_id: load_model(UI_MODEL_DIR / f"{window_id}.yaml")
            for window_id in ("market", "shop_buy", "shop_sell")
        }
        detector = AnchorDetector.for_windows(
            models, OpenCvTemplateMatcher(), ASSETS_DIR)
        cache = {}
        for name in KNOWN_WINDOW + NO_KNOWN_ANCHOR:
            frame = cv2.imread(str(SCR_DIR / f"{name}.png"))
            assert frame is not None, f"нет скрина {name}"
            cache[name] = {
                window_id: detector.find_anchor(frame, window_id)
                for window_id in models
            }
        return cache

    @pytest.mark.parametrize("name", KNOWN_WINDOW)
    def test_known_window_never_triggers(self, tmp_path, anchors_by_frame, name):
        """На кадрах с известным окном эвристика не стреляет (якорь найден)."""
        frame = cv2.imread(str(SCR_DIR / f"{name}.png"))
        interceptor = ModalInterceptor(tmp_path)
        report = interceptor.check(frame, GAME_OK, anchors_by_frame[name])
        assert not report.detected, (
            f"{name}: ложное modal.unknown при живом якоре")

    @pytest.mark.parametrize("name", NO_KNOWN_ANCHOR)
    def test_no_anchor_behavior_is_pinned(self, tmp_path, anchors_by_frame, name):
        """Поведение на кадрах без известного якоря — зафиксировано замером."""
        frame = cv2.imread(str(SCR_DIR / f"{name}.png"))
        interceptor = ModalInterceptor(tmp_path)
        report = interceptor.check(frame, GAME_OK, anchors_by_frame[name])
        assert report.detected == EXPECTED_ON_NO_ANCHOR[name]
