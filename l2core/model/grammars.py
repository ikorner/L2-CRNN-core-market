"""Грамматики OCR-регионов: разбор значений по грамматике модели, валидация
по диапазону и механизм перечитывания при несовпадении (задача 4.4,
spec ocr-reading).

Разбор и валидация — чистые функции без I/O (parse_value, expand_rect);
единственный внешний вызов — чтение региона через контракт OcrClient
(инъекция, фейк в тестах). OCR-вызовов вне OcrClient нет
(spec ocr-reading, требование 2).

Виды грамматик (common.yaml, FORMAT.md):
- digits — строго неотрицательное целое;
- price — разделители разрядов (`,` `.`) и суффиксы (`к`=1000, `кк`=1000000),
  опциональные единицы измерения (units, отсекаются перед разбором),
  валидация по диапазону;
- text — свободный непустой текст.

Дефолтные диапазоны (границы не заданы в модели): price — 1..2^63-1
(адена не бывает нулевой/отрицательной), digits — 0..2^63-1 («0 из N»).
Конкретные игровые границы в сырье не зафиксированы — опциональные
min/max в грамматике региона (вопрос владельцу, docs/OWNER_QUESTIONS.md).

Неоднозначность разделителя: `1,500` — разрядная группа (1500),
`1,5к` — дробь (1500). Правило: если все группы после первой ровно
по 3 цифры — разрядная запись, иначе (одна группа-«дробь») — десятичная
дробь, допустимая только с суффиксом, иначе — несовпадение.

Цикл перечитываний (до N, порог confidence) и стабилизация кадра —
задача 4.5 (YAGNI); здесь — одиночное перечитывание с увеличением
региона и сигнал «не сошлось»: status="invalid", confidence=0.0.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Optional, Tuple, Union

import numpy as np

from .types import GrammarSpec, Rect

__all__ = [
    "DEFAULT_DIGITS_MAX",
    "DEFAULT_DIGITS_MIN",
    "DEFAULT_PRICE_MAX",
    "DEFAULT_PRICE_MIN",
    "GrammarError",
    "GrammarMismatch",
    "RegionReadResult",
    "ValueOutOfRange",
    "expand_rect",
    "parse_value",
    "read_region",
]

# Дефолтные диапазоны (min/max в грамматике не заданы).
DEFAULT_PRICE_MIN = 1
DEFAULT_PRICE_MAX = 2**63 - 1
DEFAULT_DIGITS_MIN = 0
DEFAULT_DIGITS_MAX = 2**63 - 1

# Статусы результата чтения региона.
STATUS_OK = "ok"
STATUS_INVALID = "invalid"  # грамматика/диапазон не сошлись после перечитывания


class GrammarError(Exception):
    """Базовая ошибка разбора: прочитанный текст не даёт валидного значения."""

    def __init__(self, grammar: GrammarSpec, text: str, detail: str):
        self.grammar = grammar
        self.text = text
        super().__init__(
            f"грамматика {grammar.name!r} ({grammar.kind}): "
            f"текст {text!r}: {detail}"
        )


class GrammarMismatch(GrammarError):
    """Текст не соответствует грамматике — сигнал к перечитыванию."""


class ValueOutOfRange(GrammarError):
    """Значение соответствует грамматике, но вне допустимого диапазона."""


def parse_value(grammar: GrammarSpec, text: str) -> Union[int, str]:
    """Разобрать прочитанный текст по грамматике -> значение (int | str).

    Чистая функция. GrammarMismatch — несовпадение грамматики,
    ValueOutOfRange — вне диапазона.
    """
    if grammar.kind == "text":
        value = text.strip()
        if not value:
            raise GrammarMismatch(grammar, text, "пустой текст")
        return value
    if grammar.kind == "digits":
        core = _strip_spaces(text)
        if not core or not core.isdigit():
            raise GrammarMismatch(grammar, text, "ожидаются только цифры")
        value = int(core)
        _check_range(grammar, value, DEFAULT_DIGITS_MIN, DEFAULT_DIGITS_MAX)
        return value
    if grammar.kind == "price":
        return _parse_price(grammar, text)
    raise GrammarMismatch(grammar, text, f"неизвестный вид {grammar.kind!r}")


def _strip_spaces(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _strip_unit(grammar: GrammarSpec, text: str) -> str:
    """Отсечь единицу измерения (units) с границей слова, регистронезависимо."""
    stripped = text.strip()
    for unit in sorted(grammar.units, key=len, reverse=True):
        pattern = r"(?:^|\s)" + re.escape(unit) + r"$"
        match = re.search(pattern, stripped, flags=re.IGNORECASE)
        if match:
            return stripped[: match.start()].strip()
    return stripped


def _parse_price(grammar: GrammarSpec, text: str) -> int:
    core = _strip_spaces(_strip_unit(grammar, text))
    if not core:
        raise GrammarMismatch(grammar, text, "пусто после отсечения единиц")

    # Суффикс: самый длинный совпавший из грамматики («кк» раньше «к»).
    multiplier = 1
    suffix_len = 0
    for suffix, mult in sorted(grammar.suffixes, key=lambda p: len(p[0]), reverse=True):
        if core.endswith(suffix):
            multiplier = mult
            suffix_len = len(suffix)
            break
    number = core[: len(core) - suffix_len] if suffix_len else core

    if not grammar.separators and not number.isdigit():
        raise GrammarMismatch(grammar, text, "ожидаются цифры")

    groups = _split_groups(grammar, number)
    value = _groups_to_value(grammar, text, groups, suffix_len > 0) * multiplier
    if value.denominator != 1:
        raise GrammarMismatch(
            grammar, text, "дробная цена не кратна множителю суффикса"
        )
    result = value.numerator
    _check_range(grammar, result, DEFAULT_PRICE_MIN, DEFAULT_PRICE_MAX)
    return result


def _split_groups(grammar: GrammarSpec, number: str) -> Tuple[str, ...]:
    """Разбить числовую часть на группы по разделителям грамматики."""
    if not grammar.separators:
        if not number.isdigit():
            raise GrammarMismatch(
                grammar, number, "ожидаются только цифры без разделителей"
            )
        return (number,)
    pattern = "|".join(re.escape(s) for s in sorted(grammar.separators, key=len, reverse=True))
    groups = re.split(pattern, number)
    if any(not g.isdigit() for g in groups):
        raise GrammarMismatch(
            grammar, number, "ожидаются цифры и разделители разрядов"
        )
    return tuple(groups)


def _groups_to_value(
    grammar: GrammarSpec, raw_text: str, groups: Tuple[str, ...], has_suffix: bool
) -> Fraction:
    """Интерпретировать группы: разрядная запись или десятичная дробь.

    Дробь допустима только с суффиксом (игровая цена целая); итог после
    умножения на множитель суффикса обязан быть целым.
    """
    if len(groups) == 1:
        return Fraction(int(groups[0]))
    tail = groups[1:]
    if all(len(g) == 3 for g in tail):
        # разрядная запись: 1,500 -> 1500, 1,500,000 -> 1500000
        return Fraction(int("".join(groups)))
    if len(groups) == 2:
        # десятичная дробь: 1.5 -> 3/2 (только с суффиксом)
        if not has_suffix:
            raise GrammarMismatch(
                grammar, raw_text, "дробная цена без суффикса не допускается"
            )
        return Fraction(int(groups[0] + groups[1]), 10 ** len(groups[1]))
    raise GrammarMismatch(grammar, raw_text, "неразличимые разрядные группы")


def _check_range(grammar: GrammarSpec, value: int, default_min: int, default_max: int):
    low = grammar.min_value if grammar.min_value is not None else default_min
    high = grammar.max_value if grammar.max_value is not None else default_max
    if not low <= value <= high:
        raise ValueOutOfRange(grammar, str(value), f"вне диапазона {low}..{high}")


def expand_rect(rect: Rect, frame_shape: tuple, pad: int) -> Rect:
    """Увеличить rect на pad px с каждой стороны, прижать к границам кадра.

    Чистая функция; frame_shape — shape numpy-кадра (h, w[, c]).
    """
    height, width = frame_shape[0], frame_shape[1]
    x0 = max(0, rect.x - pad)
    y0 = max(0, rect.y - pad)
    x1 = min(width, rect.x + rect.width + pad)
    y1 = min(height, rect.y + rect.height + pad)
    return Rect(x0, y0, max(0, x1 - x0), max(0, y1 - y0))


@dataclass(frozen=True)
class RegionReadResult:
    """Результат чтения OCR-региона по грамматике.

    status: "ok" — значение разобрано и валидно; "invalid" — не сошлось
    после перечитывания с увеличением региона, confidence=0.0 — сигнал
    «низкий confidence» для слоя 4.5.
    """

    status: str
    value: Optional[Union[int, str]]
    text: str  # последнее прочитанное сырое значение
    confidence: float
    attempts: int  # число чтений: 1 — с первого, 2 — после перечитывания
    rect: Rect  # rect последнего чтения


def read_region(
    ocr,
    frame: np.ndarray,
    rect: Rect,
    grammar: GrammarSpec,
    *,
    pad: int = 8,
) -> RegionReadResult:
    """Прочитать регион кадра по грамматике с одним перечитыванием.

    Первое чтение; при GrammarError (несовпадение грамматики или выход
    за диапазон) — перечитывание с увеличением региона на pad px
    (spec ocr-reading: несовпадение -> перечитывание с увеличением
    региона; при повторе — низкий confidence). Цикл из N перечитываний
    и порог confidence — задача 4.5 (YAGNI).

    ocr — контракт OcrClient (read(region_img) -> (text, confidence));
    frame — BGR/gray uint8 кадр; rect — от якоря окна (перевод в
    координаты кадра — у вызывающего слоя, здесь rect уже в кадре).
    """
    current = rect
    last_text = ""
    last_conf = 0.0
    for attempt in (1, 2):
        crop = np.asarray(frame)[
            current.y : current.y + current.height,
            current.x : current.x + current.width,
        ]
        text, confidence = ocr.read(crop)
        last_text, last_conf = text, confidence
        try:
            value = parse_value(grammar, text)
        except GrammarError:
            if attempt == 1:
                current = expand_rect(rect, np.asarray(frame).shape, pad)
                continue
            return RegionReadResult(
                status=STATUS_INVALID,
                value=None,
                text=last_text,
                confidence=0.0,  # не сошлось: фиксируем низкий confidence
                attempts=2,
                rect=current,
            )
        return RegionReadResult(
            status=STATUS_OK,
            value=value,
            text=text,
            confidence=confidence,
            attempts=attempt,
            rect=current,
        )
    raise AssertionError("недостижимо")  # pragma: no cover
