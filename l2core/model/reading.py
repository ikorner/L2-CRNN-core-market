"""Слой чтения OCR-регионов: цикл перечитываний при низком confidence
и стабилизация кадра для мигающей каретки (задача 4.5, spec ocr-reading).

Два независимых компонента поверх `read_region` (задача 4.4):

- `read_confident` — разовое чтение региона с гарантией confidence:
  цикл до `reread_count` перечитываний с увеличением региона; исчерпание
  → `OcrReadError` с диагностикой (spec: «Обработка низкого confidence»).
- `read_stable` — чтение поля ввода между кадрами: два подряд одинаковых
  чтения = стабильный результат; плейсхолдер отличается от введённого
  текста (spec: «Стабилизация кадра для мигающей каретки»).

Семантика попыток (внешняя/внутренняя, без двойного расширения):

- ВНЕШНЯЯ попытка k (цикл здесь) стартует с региона, увеличенного на
  `k * pad` от исходного (k=0 — исходный rect). Причина перехода к
  следующей попытке — низкий confidence (ниже порога), в т.ч. status
  "invalid" (confidence=0.0, задача 4.4).
- ВНУТРЕННЯЯ перечитка (внутри read_region) — одно перечитывание с
  увеличением на `pad` поверх стартового ректа попытки; причина —
  несовпадение грамматики/диапазона. Расширение не дублируется:
  внешнее — между попытками, внутреннее — внутри попытки.

Порог и число перечитываний — параметры; единый путь подстановки —
значения `ocr.confidence_threshold` / `ocr.reread_count` конфигурации
(l2core/config.py) передаёт вызывающий слой (примитивы 5.x, сценарии).

Инварианты: без sleep/wait (инвариант 1 — здесь только чтение переданных
кадров/вызывающего источника), детерминизм (OCR и источник кадров —
инъекции, инвариант 7). Код и комментарии — на русском (инвариант 8).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import numpy as np

from .grammars import (
    STATUS_INVALID,
    STATUS_OK,
    GrammarError,
    RegionReadResult,
    expand_rect,
    parse_value,
    read_region,
)
from .types import OcrRegionSpec, Rect

__all__ = [
    "OcrReadError",
    "ReadAttempt",
    "STATUS_PLACEHOLDER",
    "StableReadResult",
    "read_confident",
    "read_stable",
]

# Статусы результата стабилизатора (совпадают с grammars.STATUS_* где есть смысл).
STATUS_PLACEHOLDER = "placeholder"  # поле пусто: плейсхолдер или пустой текст


@dataclass(frozen=True)
class ReadAttempt:
    """Одна попытка чтения региона (внешняя попытка цикла / кадр стабилизатора).

    rect — стартовый rect попытки; text/confidence/status — итоговые
    значения read_region (после внутренней перечитки, если была).
    """

    rect: Rect
    text: str
    confidence: float
    status: str


class OcrReadError(Exception):
    """Исчерпание перечитываний/стабилизации: исключение стратегии.

    Поля — диагностика для решения стратегии (spec: исключение с
    диагностикой региона и значениями): region_id, threshold (None у
    стабилизатора — там нет порога), attempts — история попыток
    (rect, text, confidence, status).
    """

    def __init__(
        self,
        message: str,
        *,
        region_id: str,
        attempts: Tuple[ReadAttempt, ...],
        threshold: Optional[float] = None,
    ):
        self.region_id = region_id
        self.threshold = threshold
        self.attempts = tuple(attempts)
        super().__init__(message)

    @property
    def last_text(self) -> str:
        return self.attempts[-1].text if self.attempts else ""

    @property
    def last_confidence(self) -> float:
        return self.attempts[-1].confidence if self.attempts else 0.0


def _check_confidence_args(threshold: float, reread_count: int) -> None:
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not 0.0 <= threshold <= 1.0
    ):
        raise ValueError(f"threshold: ожидается число в [0, 1] (получено {threshold!r})")
    if isinstance(reread_count, bool) or not isinstance(reread_count, int) or reread_count < 0:
        raise ValueError(f"reread_count: ожидается целое >= 0 (получено {reread_count!r})")


def read_confident(
    ocr,
    frame: np.ndarray,
    region: OcrRegionSpec,
    *,
    threshold: float,
    reread_count: int,
    pad: int = 8,
) -> RegionReadResult:
    """Прочитать регион с гарантией confidence (цикл перечитываний 4.5).

    Внешние попытки: 1 + reread_count. Попытка k стартует с региона,
    увеличенного на k*pad от исходного (внешнее расширение — причина:
    низкий confidence). Внутри попытки read_region при несовпадении
    грамматики перечитывает с увеличением на pad один раз (внутреннее
    расширение — расширения не дублируются).

    Успех: status "ok" и confidence >= threshold. Исчерпание →
    OcrReadError с историей попыток (диагностика региона и значениями).

    ocr — контракт OcrClient; frame — кадр, region — OCR-регион модели
    (rect уже в координатах кадра, как у read_region); threshold и
    reread_count — из конфига (ocr.confidence_threshold, ocr.reread_count).
    """
    _check_confidence_args(threshold, reread_count)
    frame = np.asarray(frame)
    attempts = []
    for k in range(reread_count + 1):
        start = expand_rect(region.rect, frame.shape, k * pad) if k else region.rect
        result = read_region(ocr, frame, start, region.grammar, pad=pad)
        attempts.append(
            ReadAttempt(
                rect=start,
                text=result.text,
                confidence=result.confidence,
                status=result.status,
            )
        )
        if result.status == STATUS_OK and result.confidence >= threshold:
            return result
    values = ", ".join(f"{a.text!r}@{a.confidence:.4f}" for a in attempts)
    raise OcrReadError(
        f"OCR региона {region.id!r}: confidence ниже порога {threshold} "
        f"после {len(attempts)} попыток (reread_count={reread_count}); "
        f"значения: {values}",
        region_id=region.id,
        threshold=threshold,
        attempts=tuple(attempts),
    )


@dataclass(frozen=True)
class StableReadResult:
    """Стабильное чтение поля ввода (два подряд одинаковых результата).

    status: "ok" — стабильный введённый текст (value разобрано по
    грамматике); "placeholder" — поле пусто (плейсхолдер или пустой
    текст), value=None: плейсхолдер ≠ введённый текст.
    """

    status: str
    value: Optional[Union[int, str]]
    text: str  # стабильное сырое значение
    confidence: float  # confidence последнего чтения
    frames: int  # число прочитанных кадров


def read_stable(
    ocr,
    frame_source: Callable[[], np.ndarray],
    region: OcrRegionSpec,
    *,
    max_frames: int = 10,
    pad: int = 8,
) -> StableReadResult:
    """Прочитать поле ввода стабильно: два подряд одинаковых чтения.

    Мигающая каретка даёт разный OCR-текст между кадрами («ab|»/«ab»);
    чтение повторяется (свежий кадр от frame_source) до двух подряд
    одинаковых текстов (spec: «Стабилизация кадра для мигающей каретки»).

    Стабильный текст:
    - пустой или совпавший с regex-паттерном placeholder региона
      (опциональное поле ocr_regions) → status "placeholder", value=None;
    - иначе разбор по грамматике региона → status "ok", value.

    Исчерпание max_frames кадров без двух одинаковых → OcrReadError
    с историей текстов; стабильный текст вне грамматики → OcrReadError
    (стратегия решает по диагностике).

    frame_source — callable () -> кадр (совместимо с Screen.grab:
    стабилизатор сам берёт следующий кадр; в тестах — скрипт).
    Порог confidence здесь не проверяется — цикл по confidence это
    read_confident для разовых чтений (YAGNI).
    """
    if isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < 1:
        raise ValueError(f"max_frames: ожидается целое >= 1 (получено {max_frames!r})")

    attempts = []
    previous_text: Optional[str] = None
    for _ in range(max_frames):
        frame = np.asarray(frame_source())
        result = read_region(ocr, frame, region.rect, region.grammar, pad=pad)
        attempts.append(
            ReadAttempt(
                rect=region.rect,
                text=result.text,
                confidence=result.confidence,
                status=result.status,
            )
        )
        if previous_text is not None and result.text == previous_text:
            return _stable_result(region, result, len(attempts))
        previous_text = result.text
    values = ", ".join(repr(a.text) for a in attempts)
    raise OcrReadError(
        f"OCR региона {region.id!r}: нет двух одинаковых чтений подряд "
        f"за {len(attempts)} кадров (max_frames={max_frames}); "
        f"значения: {values}",
        region_id=region.id,
        attempts=tuple(attempts),
    )


def _stable_result(
    region: OcrRegionSpec, result: RegionReadResult, frames: int
) -> StableReadResult:
    """Сформировать результат из стабильного чтения (placeholder vs текст)."""
    text = result.text
    if not text.strip():
        return StableReadResult(
            status=STATUS_PLACEHOLDER, value=None, text=text,
            confidence=result.confidence, frames=frames,
        )
    if region.placeholder is not None and re.search(region.placeholder, text):
        return StableReadResult(
            status=STATUS_PLACEHOLDER, value=None, text=text,
            confidence=result.confidence, frames=frames,
        )
    try:
        value = parse_value(region.grammar, text)
    except GrammarError as exc:
        raise OcrReadError(
            f"OCR региона {region.id!r}: стабильный текст {text!r} не разобран "
            f"по грамматике: {exc}",
            region_id=region.id,
            attempts=(
                ReadAttempt(
                    rect=region.rect, text=text, confidence=result.confidence,
                    status=STATUS_INVALID,
                ),
            ),
        ) from exc
    return StableReadResult(
        status=STATUS_OK, value=value, text=text,
        confidence=result.confidence, frames=frames,
    )
