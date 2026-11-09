"""Сценарий мониторинга своей лавки monitor_shop (задача 6.3, spec
market-scenarios «Сценарий мониторинга своей лавки»).

ДЕТЕКЦИЯ ПРОДАЖИ — перечитыванием окна лавки: OCR показателей (проданное
количество/прибыль) со сравнением снимков относительно базовой линии, НЕ
через окно сообщений игры. Очередное чтение показывает увеличение проданного
количества относительно линии → событие sold (spec: «Детекция продажи»).

ShopMonitor — ЯВНОЕ исключение из правила «сценарии — композиции без
внутреннего состояния» (spec market-scenarios: «внутреннее состояние
допускается только мониторам/наблюдателям — базовая линия для детекции
"продано"»): монитор хранит baseline (стартовые показатели лавки после
выставления) и сравнивает с ней каждый poll().

ПОКАЗАТЕЛИ И ИХ ИСТОЧНИК: OCR-регионы проданного количества/прибыли в
отгруженной модели shop_sell НЕ размечены (ocr_regions пуст — KNOWN_GAPS,
досъёмка/калибровка rect'ов у владельца): R4 не выдумывает константы, поэтому
регионы передаёт вызывающий слой (параметр regions, rect'ы в координатах
окна от якоря), при отсутствии — подшиваются из секции ocr_regions модели.
Регион-количество (counter_region) обязателен и должен иметь числовую
грамматику (digits/price); регион-прибыль (revenue_region) опционален — без
него события sold идут с revenue=None.

СЕМАНТИКА baseline/ДЕЛЬТ:

- capture_baseline() — чтение текущих показателей лавки после
  выставления/переразметки: стратегия вызывает после start_sell() (6.2)
  или переоткрытия лавки. Чтение не удалось — линия не устанавливается
  (poll() до линии отклоняется с диагностикой).
- Увеличение количества относительно линии = продажа: событие sold
  (callback-инъекция on_sold), baseline := текущие показатели — повторная
  продажа даст собственную свежую дельту (накопительный счётчик не нужен:
  показатели окна и так суммарные).
- Уменьшение = переразметка лавки (снятие/перевыставление): события НЕТ
  (YAGNI: фиксируется только sold), baseline := текущие показатели — иначе
  продажи после переразметки не детектятся от устаревшей линии; факт
  переразметки — в диагностике результата.
- OCR-деградация: счётчик не прочитан — продажу подтвердить нельзя: события
  нет, результат degraded с диагностикой региона (линия не сдвигается
  ложным чтением). Прибыль не прочитана при читаемом счётчике — событие
  идёт с revenue=None (честная деградация, флаг degraded).

СОБЫТИЕ sold: push-форма через callback-инъекцию on_sold(payload);
payload (типизированный SoldPayload) — суммарные показатели снимка и дельты
к линии: sold_count (продано), sold_delta, revenue (прибыль), revenue_delta,
snapshot_id (детерминированный id кадра, sha1 байтов — тот же источник, что
у RowsResult 6.1). Полная система событий (dispatcher/pull/on-шина) — 7.3,
здесь только генерация события.

ГРАНИЦЫ: монитор — компонент для observer-потока (7.2): poll() синхронен,
HID не трогает ВООБЩЕ (ни кликов, ни скролла — только grab/find_anchor/OCR,
guard-тесты следят); внутреннее состояние — допустимое исключение spec.
Потокобезопасность: poll()/capture_baseline() не защищены замками — монитором
владеет ОДИН поток (observer, 7.2); внешняя синхронизация границы — решение
7.2 (YAGNI: замков здесь нет). YAGNI: observer-поток (7.2), dispatcher (7.3),
публичный API (7.1) — не здесь.

Отмена (kill switch 8.1, инъекция cancelled): проверяется до чтения кадра —
результат с cancelled=True, экран не трогается.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Tuple

import numpy as np

from l2core.adapters.screen import Screen
from l2core.config import CoreConfig
from l2core.detector.anchor import Anchor
from l2core.model import OcrRegionSpec, Rect, UiModel
from l2core.model.reading import OcrReadError, read_confident

__all__ = [
    "Baseline",
    "MonitorError",
    "MonitorPollResult",
    "ShopMonitor",
    "SoldPayload",
]

# Числовые грамматики показателей: сравнение снимков — только по int-значениям
# (text-грамматика показателем продажи не является — ошибка конфигурации).
_NUMERIC_GRAMMARS = frozenset({"digits", "price"})


class MonitorError(Exception):
    """Ошибка монитора (конфигурация/модель), не эскалация рантайма."""

    def __init__(self, detail: str):
        super().__init__(f"monitor_shop: {detail}")
        self.detail = detail


@dataclass(frozen=True)
class SoldPayload:
    """Полезная нагрузка события sold (spec: продано, прибыль).

    sold_count/revenue — суммарные показатели снимка; sold_delta/
    revenue_delta — приращение относительно линии, давшей событие.
    revenue/revenue_delta — None, когда регион прибыли не настроен или его
    не удалось прочитать в этом poll'е (честная деградация, не ноль).
    snapshot_id — детерминированный id кадра (sha1 байтов кадра).
    """

    sold_count: int
    sold_delta: int
    revenue: Optional[int]
    revenue_delta: Optional[int]
    snapshot_id: str


@dataclass(frozen=True)
class Baseline:
    """Базовая линия монитора: показатели окна лавки на момент захвата.

    Внутреннее состояние монитора — допустимое исключение spec (мониторы/
    наблюдатели); значение None у revenue — регион прибыли не настроен или
    не прочитан при захвате.
    """

    values: Mapping[str, Optional[int]]
    snapshot_id: str


@dataclass(frozen=True)
class MonitorPollResult:
    """Итог одного чтения монитора.

    sold — событие продажи этого чтения (None — продажи нет/подтвердить
    нельзя); degraded — чтение было, но частично/с ошибками (OCR не сошёлся,
    якоря не было): событие при degraded выставляется только если счётчик
    прочитан и вырос; failures — диагностика деградации/отказа; diagnostics —
    неэскалационные наблюдения (переразметка лавки); snapshot_id — id кадра
    чтения (None — кадр не читался).
    """

    sold: Optional[SoldPayload] = None
    degraded: bool = False
    cancelled: bool = False
    failures: Tuple[str, ...] = ()
    diagnostics: Tuple[str, ...] = ()
    snapshot_id: Optional[str] = None


def _snapshot_id(frame: np.ndarray) -> str:
    """Детерминированный id кадра: sha1 по байтам (как RowsResult 6.1)."""
    return hashlib.sha1(np.ascontiguousarray(frame).tobytes()).hexdigest()


class ShopMonitor:
    """Монитор своей лавки продажи: baseline + poll() с детекцией продажи.

    Конструктор валидирует конфигурацию регионов (fail fast): регион-
    количество обязателен, грамматики показателей — числовые (digits/price).
    Чтение — только grab/find_anchor/OCR-инъекции; HID и действия не
    используются (monitor — наблюдатель, не исполнитель). Контракт модуля —
    в докстринге модуля; потоковая модель — один владеющий поток (7.2).
    """

    def __init__(
        self,
        *,
        model: UiModel,
        ocr,
        screen: Screen,
        find_anchor: Callable[[np.ndarray], Optional[Anchor]],
        config: CoreConfig,
        regions: Optional[Mapping[str, OcrRegionSpec]] = None,
        counter_region: str = "sold_count",
        revenue_region: Optional[str] = "revenue",
        on_sold: Optional[Callable[[SoldPayload], None]] = None,
        cancelled: Optional[Callable[[], bool]] = None,
    ):
        # Регионы: параметр побеждает, секция ocr_regions модели — fallback
        # (регионы в отгруженной модели не размечены — KNOWN_GAPS; R4).
        resolved = {**model.ocr_regions, **dict(regions or {})}
        counter = resolved.get(counter_region)
        if counter is None:
            raise MonitorError(
                f"OCR-регион количества проданного не задан: ни параметром "
                f"regions, ни секцией ocr_regions модели окна "
                f"{model.window.id!r} (id {counter_region!r}; в отгруженной "
                f"модели пусто — KNOWN_GAPS): детекция продажи без "
                f"перечитывания показателей невозможна")
        self._validate_grammar(counter)
        revenue = None
        if revenue_region is not None:
            revenue = resolved.get(revenue_region)
            if revenue is None:
                raise MonitorError(
                    f"OCR-регион прибыли ({revenue_region!r}) задан ключом, "
                    f"но не найден ни в параметре regions, ни в ocr_regions "
                    f"модели {model.window.id!r}: конфигурация монитора "
                    f"некорректна (fail fast)")
            self._validate_grammar(revenue)
        self._model = model
        self._ocr = ocr
        self._screen = screen
        self.find_anchor = find_anchor
        self._config = config
        self._counter = counter
        self._revenue = revenue
        self._on_sold = on_sold
        self._cancelled = cancelled
        self._baseline: Optional[Baseline] = None

    @staticmethod
    def _validate_grammar(region: OcrRegionSpec) -> None:
        """Показатель — число (digits/price): сравнение снимков по значениям."""
        if region.grammar.kind not in _NUMERIC_GRAMMARS:
            raise MonitorError(
                f"OCR-регион {region.id!r}: грамматика "
                f"{region.grammar.kind!r} не числовая "
                f"({sorted(_NUMERIC_GRAMMARS)}): показателем продажи может "
                f"быть только число")

    @property
    def baseline(self) -> Optional[Baseline]:
        """Текущая базовая линия (None — ещё не захвачена)."""
        return self._baseline

    def capture_baseline(self) -> MonitorPollResult:
        """Захватить базовую линию: показатели лавки становятся точкой
        отсчёта продаж (вызывается после start_sell()/переразметки).
        Чтение не удалось — линия прежняя/пустая, диагностика в failures."""
        reading = self._read()
        if reading.cancelled:
            return MonitorPollResult(cancelled=True)
        if reading.failures:
            return MonitorPollResult(
                degraded=reading.degraded, failures=reading.failures,
                diagnostics=reading.diagnostics,
                snapshot_id=reading.snapshot_id)
        self._baseline = Baseline(values=reading.values,
                                  snapshot_id=reading.snapshot_id)
        return MonitorPollResult(
            degraded=reading.degraded, failures=reading.failures,
            diagnostics=reading.diagnostics,
            snapshot_id=reading.snapshot_id)

    def poll(self) -> MonitorPollResult:
        """Очередное чтение окна лавки: сравнение с линией, детекция продажи.

        Увеличение количества относительно линии → событие sold (on_sold)
        и линия := текущие показатели. Уменьшение → переразметка (без события,
        линия := текущие показатели, диагностика). Без линии — отказ."""
        if self._cancelled is not None and self._cancelled():
            return MonitorPollResult(cancelled=True)
        if self._baseline is None:
            return MonitorPollResult(failures=(
                "baseline не задан: сначала capture_baseline() "
                "(после выставления/переразметки лавки)",))
        baseline = self._baseline
        reading = self._read()
        if reading.cancelled:
            return MonitorPollResult(cancelled=True)
        if reading.counter is None:
            # Счётчик не прочитан: продажу подтвердить нельзя — только
            # честная деградация, линия не сдвигается ложным чтением.
            return MonitorPollResult(
                degraded=True, failures=reading.failures,
                diagnostics=reading.diagnostics,
                snapshot_id=reading.snapshot_id)
        sold_base = baseline.values.get(self._counter.id, 0)
        revenue_base = baseline.values.get(
            self._revenue.id) if self._revenue is not None else None
        diagnostics = list(reading.diagnostics)
        sold = None
        if reading.counter > sold_base:
            if reading.revenue is not None and revenue_base is not None:
                revenue_delta = reading.revenue - revenue_base
            else:
                revenue_delta = None
            sold = SoldPayload(
                sold_count=reading.counter,
                sold_delta=reading.counter - sold_base,
                revenue=reading.revenue,
                revenue_delta=revenue_delta,
                snapshot_id=reading.snapshot_id)
        elif reading.counter < sold_base:
            # Переразметка лавки: события нет (YAGNI: только sold), линия
            # := текущие показатели — продажи после переразметки иначе не
            # детектятся от устаревшей линии.
            diagnostics.append(
                f"показатель {self._counter.id!r} уменьшился "
                f"({reading.counter} < {sold_base}): переразметка лавки, "
                f"принята новая базовая линия; событие sold не генерируется")
        if sold is not None:
            if self._on_sold is not None:
                self._on_sold(sold)
        if sold is not None or reading.counter < sold_base:
            # Продажа: линия := показатели снимка (свежая дельта следующей
            # продаже). Переразметка: линия := текущие показатели — продажи
            # после неё иначе не детектятся от устаревшей линии.
            self._baseline = Baseline(values=reading.values,
                                      snapshot_id=reading.snapshot_id)
        return MonitorPollResult(
            sold=sold,
            degraded=reading.degraded,
            failures=reading.failures,
            diagnostics=tuple(diagnostics),
            snapshot_id=reading.snapshot_id)

    def _read(self) -> "_Reading":
        """Одно чтение показателей окна: кадр -> якорь -> OCR регионов.

        Только чтение (grab/find_anchor/OCR); якоря нет или счётчик не
        прочитан — деградация с диагностикой, не исключение (монитор —
        наблюдатель: наблюдать нельзя — честно докладывает о деградации).
        """
        if self._cancelled is not None and self._cancelled():
            return _Reading(cancelled=True)
        frame = np.asarray(self._screen.grab())
        anchor = self.find_anchor(frame)
        if anchor is None:
            return _Reading(
                degraded=True, snapshot_id=_snapshot_id(frame),
                failures=("якорь окна лавки не найден на кадре: чтение "
                          "показателей невозможно (окно закрыто/сдвинуто)"))
        failures: list = []
        counter = self._read_region(frame, anchor, self._counter, failures)
        revenue = None
        degraded = False
        if self._revenue is not None:
            revenue = self._read_region(frame, anchor, self._revenue, failures)
        if failures:
            degraded = True
        values = {self._counter.id: counter}
        if self._revenue is not None:
            values[self._revenue.id] = revenue
        return _Reading(
            counter=counter, revenue=revenue, values=values,
            degraded=degraded, failures=tuple(failures),
            snapshot_id=_snapshot_id(frame))

    def _read_region(self, frame: np.ndarray, anchor: Anchor,
                     region: OcrRegionSpec, failures: list) -> Optional[int]:
        """Прочитать один показатель (rect региона -> координаты кадра ->
        read_confident 4.5). Не прочитан — None + диагностика в failures."""
        rect: Rect = region.rect
        frame_region = OcrRegionSpec(
            id=region.id,
            rect=Rect(anchor.x + rect.x, anchor.y + rect.y,
                      rect.width, rect.height),
            grammar=region.grammar,
            placeholder=region.placeholder,
        )
        try:
            result = read_confident(
                self._ocr, frame, frame_region,
                threshold=self._config.ocr.confidence_threshold,
                reread_count=self._config.ocr.reread_count)
        except OcrReadError as exc:
            failures.append(
                f"OCR региона {region.id!r} не сошёлся: {exc}")
            return None
        return int(result.value)


@dataclass(frozen=True)
class _Reading:
    """Внутренний результат одного чтения показателей (не публичный API)."""

    counter: Optional[int] = None
    revenue: Optional[int] = None
    values: Optional[Mapping[str, Optional[int]]] = None
    degraded: bool = False
    cancelled: bool = False
    failures: Tuple[str, ...] = ()
    diagnostics: Tuple[str, ...] = ()
    snapshot_id: Optional[str] = None
