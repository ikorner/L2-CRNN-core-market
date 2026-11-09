"""Юнит-тесты template matching: OpenCV и numpy-only реализации (задача 3.3).

Обе реализации возвращают лучшую позицию и score нормализованной
кросс-корреляции (эквивалент OpenCV TM_CCOEFF_NORMED). Согласованность
реализаций проверяется на синтетическом детерминированном кадре и на
реальном скриншоте; выбор реализации по умолчанию — по замеру 3.3
(отчёт task-3.3-report.md).
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from l2core.adapters.template import (
    Match,
    NumpyTemplateMatcher,
    OpenCvTemplateMatcher,
    default_matcher,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SCR_DIR = REPO_ROOT / "base_screenshot" / "scr"


def synthetic_frame(h: int = 60, w: int = 80) -> np.ndarray:
    """Детерминированный серый кадр без случайных чисел."""
    yy, xx = np.mgrid[0:h, 0:w]
    return ((xx * 7 + yy * 13 + (xx * yy) % 17) % 251).astype(np.float64)


@pytest.fixture(
    scope="module",
    params=[NumpyTemplateMatcher(), OpenCvTemplateMatcher()],
    ids=["numpy", "opencv"],
)
def matcher(request):
    return request.param


class TestMatchContract:
    def test_exact_patch_gives_unit_score_and_position(self, matcher):
        image = synthetic_frame()
        x0, y0 = 41, 17
        template = image[y0 : y0 + 20, x0 : x0 + 30]
        got = matcher.match(image, template)
        assert isinstance(got, Match)
        assert (got.x, got.y) == (x0, y0)
        assert got.score == pytest.approx(1.0, abs=1e-6)

    def test_match_finds_displaced_copy(self, matcher):
        """Патч, перенесённый в другую область однородного кадра, находится."""
        image = np.full((50, 60), 100.0)
        template = np.arange(12 * 8, dtype=np.float64).reshape(12, 8) * 3 % 200
        image[30:42, 45:53] = template
        got = matcher.match(image, template)
        assert (got.x, got.y) == (45, 30)
        assert got.score == pytest.approx(1.0, abs=1e-6)

    def test_no_match_below_expectation(self, matcher):
        """Плоский кадр не даёт совпадения с неплоским шаблоном."""
        image = np.full((50, 60), 100.0)
        template = np.arange(12 * 8, dtype=np.float64).reshape(12, 8) % 37
        got = matcher.match(image, template)
        assert got.score == pytest.approx(0.0, abs=1e-6)

    def test_template_larger_than_image_raises(self, matcher):
        image = synthetic_frame(20, 30)
        template = synthetic_frame(25, 35)
        with pytest.raises(ValueError):
            matcher.match(image, template)

    def test_best_position_wins_over_noise_copy(self, matcher):
        image = synthetic_frame(70, 90)
        template = image[10:26, 12:40].copy()
        # зашумлённая копия патча ближе к началу кадра
        image[50:66, 60:88] = template * 0.8 + 20
        got = matcher.match(image, template)
        assert (got.x, got.y) == (12, 10)


class TestDefaultMatcher:
    def test_default_is_opencv_by_measurement(self):
        """Выбор по умолчанию — OpenCV (замер 3.3, решение 13): numpy-only
        альтернатива идентична по точности, но в 2.3–4 раза медленнее."""
        assert isinstance(default_matcher(), OpenCvTemplateMatcher)


class TestImplementationsAgree:
    def test_same_result_on_real_screenshot(self):
        """Обе реализации дают ту же позицию и близкий score на реальном кадре."""
        image = cv2.imread(
            str(SCR_DIR / "scr_154330.png"), cv2.IMREAD_GRAYSCALE
        ).astype(np.float64)
        template = cv2.imread(
            str(REPO_ROOT / "assets" / "shop_buy" / "scr_154330_jak.png"),
            cv2.IMREAD_GRAYSCALE,
        ).astype(np.float64)
        by_numpy = NumpyTemplateMatcher().match(image, template)
        by_opencv = OpenCvTemplateMatcher().match(image, template)
        assert (by_numpy.x, by_numpy.y) == (by_opencv.x, by_opencv.y) == (315, 232)
        assert by_numpy.score == pytest.approx(by_opencv.score, abs=1e-2)
        assert by_numpy.score == pytest.approx(1.0, abs=1e-2)

    def test_opencv_backend_uses_tm_ccoeff_normed(self):
        """OpenCV-реализация совпадает с прямым вызовом cv2.matchTemplate."""
        image = synthetic_frame(50, 70)
        template = image[20:35, 40:60]
        got = OpenCvTemplateMatcher().match(image, template)
        expected = cv2.matchTemplate(
            image.astype(np.float32), template.astype(np.float32),
            cv2.TM_CCOEFF_NORMED,
        )
        _, max_val, _, max_loc = cv2.minMaxLoc(expected)
        assert (got.x, got.y) == max_loc
        assert got.score == pytest.approx(max_val, abs=1e-5)
