"""Template matching: OpenCV-реализация и numpy-only альтернатива (задача 3.3).

Основной вид поиска элементов UI (design.md, решение 13): нормализованная
кросс-корреляция, эквивалент OpenCV TM_CCOEFF_NORMED. Обе реализации
возвращают лучшую позицию (верхний левый угол шаблона) и score в [-1, 1].

OpenCV matchTemplate допустим только если numpy-only альтернатива ощутимо
хуже — выбор реализации по умолчанию сделан по замеру на 39 скриншотах
(отчёт task-3.3-report.md, AGENTS.md).

Numpy-only реализация: корреляция через FFT (rfft2) + локальные суммы и
суммы квадратов через интегральные изображения (cumsum). Детерминизм:
одинаковый вход даёт одинаковый выход, внешних сервисов и случайных чисел
нет.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple

import numpy as np


class Match(NamedTuple):
    """Лучшее совпадение шаблона: позиция верхнего левого угла и score."""

    x: int
    y: int
    score: float  # NCC в [-1, 1]


def default_matcher() -> "TemplateMatcher":
    """Реализация template matching по умолчанию — OpenCV matchTemplate.

    Выбор по замеру 3.3 (решение 13): numpy-only альтернатива
    (NumpyTemplateMatcher) даёт идентичные решения на всех 39 скриншотах
    (0 расхождений), но в 2.3–4 раза медленнее (jak 198x29: ~237 мс против
    ~58 мс; целое окно 1008x674: ~183 мс против ~80 мс на кадре 1920x1080;
    CPU, один поток) — ощутимо хуже на горячем пути. Числа — в отчёте
    task-3.3-report.md и AGENTS.md.
    """
    return OpenCvTemplateMatcher()


class TemplateMatcher(ABC):
    """Поиск шаблона на сером кадре: лучшая позиция + score."""

    @abstractmethod
    def match(self, image: np.ndarray, template: np.ndarray) -> Match:
        """image, template — 2D-массивы (H, W) одного dtype.

        Шаблон обязан быть не больше кадра по обеим осям, иначе ValueError.
        """


class OpenCvTemplateMatcher(TemplateMatcher):
    """cv2.matchTemplate с методом TM_CCOEFF_NORMED."""

    def match(self, image: np.ndarray, template: np.ndarray) -> Match:
        h, w = template.shape[:2]
        image_h, image_w = image.shape[:2]
        if image_h < h or image_w < w:
            raise ValueError(
                f"шаблон ({w}x{h}) больше кадра ({image_w}x{image_h})"
            )
        import cv2

        result = cv2.matchTemplate(
            image.astype(np.float32), template.astype(np.float32),
            cv2.TM_CCOEFF_NORMED,
        )
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        score = float(max_val)
        if score != score:  # NaN: вырожденный шаблон/окно
            score = 0.0
        return Match(x=max_loc[0], y=max_loc[1], score=score)


class NumpyTemplateMatcher(TemplateMatcher):
    """numpy-only NCC: FFT-корреляция + интегральные изображения."""

    def match(self, image: np.ndarray, template: np.ndarray) -> Match:
        img = np.ascontiguousarray(image, dtype=np.float64)
        tpl = np.ascontiguousarray(template, dtype=np.float64)
        image_h, image_w = img.shape
        h, w = tpl.shape
        if image_h < h or image_w < w:
            raise ValueError(
                f"шаблон ({w}x{h}) больше кадра ({image_w}x{image_h})"
            )
        out_h, out_w = image_h - h + 1, image_w - w + 1

        # corr[y, x] = sum(img[y+i, x+j] * tpl[i, j]) по окну (h, w).
        spectrum = np.fft.rfft2(img) * np.conj(np.fft.rfft2(tpl, s=img.shape))
        corr = np.fft.irfft2(spectrum, s=img.shape)[:out_h, :out_w]

        # Локальные суммы img и img**2 по окну (h, w) через интегральные
        # изображения с нулевой рамкой: s[y, x] — сумма над img[y:y+h, x:x+w].
        def box_sums(values: np.ndarray) -> np.ndarray:
            integral = np.zeros((image_h + 1, image_w + 1), dtype=np.float64)
            integral[1:, 1:] = values.cumsum(axis=0).cumsum(axis=1)
            return (
                integral[h : image_h + 1, w : image_w + 1]
                - integral[0 : out_h, w : image_w + 1]
                - integral[h : image_h + 1, 0 : out_w]
                + integral[0:out_h, 0:out_w]
            )

        n = float(h * w)
        sum_i = box_sums(img)
        sum_i2 = box_sums(img * img)
        sum_t = float(tpl.sum())
        sum_t2 = float((tpl * tpl).sum())

        # sum((I - mI)(T - mT)) = corr - sumT*sumI/n
        numerator = corr - sum_t * sum_i / n
        var_i = sum_i2 - sum_i * sum_i / n
        var_t = sum_t2 - sum_t * sum_t / n
        if var_t <= 0.0:
            # Плоский шаблон: NCC не определён. OpenCV в этом вырожденном
            # случае возвращает 1.0; ядро такие шаблоны не использует —
            # считаем «нет совпадения», позиция не осмысленна.
            return Match(x=0, y=0, score=0.0)
        denominator = np.sqrt(np.maximum(var_i, 0.0) * var_t)
        with np.errstate(invalid="ignore", divide="ignore"):
            scores = np.where(denominator > 0.0, numerator / denominator, 0.0)
        scores = np.nan_to_num(scores, nan=-1.0, posinf=-1.0, neginf=-1.0)
        flat = int(scores.argmax())
        y, x = divmod(flat, out_w)
        return Match(x=x, y=y, score=float(scores[y, x]))
