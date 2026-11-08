"""Сценарий обхода «Обзора лавок» scan_market (задача 6.1, spec
market-scenarios: обход табов/фильтров/поиск/скролл, чтение строк; обзор
ТОЛЬКО для разведки цен — открытие чужих лавок отсутствует (non-goal),
guard-тесты следят за этим).

Сценарий — КОМПОЗИЦИЯ примитивов 5.x БЕЗ внутреннего состояния (spec:
«Сценарии — композиции примитивов без внутреннего состояния»): текущее
состояние всегда спрашивается у детектора через инъекции (find_anchor /
window_status / run_checks), ничего не кэшируется «в голове». Единственное
накопленное — результат обхода (строки/снимки), это вывод, не состояние
решений.

ОБХОД: список точек ScanPoint передаётся ИЗВНЕ (параметризация обхода):
params табов/фильтров частично не размечены моделью (KNOWN_GAPS), поэтому
стратегия сама описывает, что кликать. Точка: tab (element id таба),
filters (element id фильтров, по порядку), search (текст поиска,
опционально), verify (VerifySpec состояния готовности списка — обязателен,
клики без verify запрещены). Для каждой точки:

    открыть обзор (первый раз: клик по market_pic, verify из перехода
    модели recover_closed -> list__tab_home) -> таб -> фильтры -> поиск ->
    цикл: чтение строк read_rows (6.1, l2core.model.rows) -> скролл
    scroll_list вниз -> до признака конца из модели (disabled-стрелка
    «вниз», scroll_position 5.5) с предохранителем max_scrolls

ДЕДУП: строки каждого кадра сливаются сквозным дедупом dedup_rows
(icon_hash+цена+y, 6.1); по точке — агрегат + снимки (RowsResult со
snapshot_id: рынок меняется во время сканирования, стратегия сверяет).

ПОИСК: OCR-регион поля ввода в модели market.yaml не размечен
(ocr_regions пуст, KNOWN_GAPS) — регион, элемент поля и кнопка «Поиск»
передаются снаружи (search_region/search_field_element/
search_button_element); точка с search без search_region — ScanError
(fail fast, честная граница, не молчаливый пропуск).

ОТКАЗЫ: needs_recover примитивов фиксируется в failures и обрывает обход
(исполнение recover — 6.4, вызывающий слой); отмена (kill switch 8.1,
инъекция cancelled) — результат с cancelled=True без эскалаций.
Модельные ошибки (нет перехода открытия, нет search_region) — ScanError.

ГРАНИЦЫ (R4): семантика params фильтров/табов и переходов между ними в
модели не подтверждены метой — verify на точку обхода даёт стратегия;
экранный нумпад не используется (текст поиска ASCII; кириллица — Q25).
"""
from __future__ import annotations

import dataclasses
import random
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Tuple

import numpy as np

from l2core.adapters.screen import Screen
from l2core.adapters.table_segmentation import TableSegmentation
from l2core.config import CoreConfig
from l2core.detector.anchor import Anchor
from l2core.detector.status import WindowStatus
from l2core.model import OcrRegionSpec, UiModel
from l2core.model.rows import Row, RowsResult, dedup_rows, read_rows
from l2core.primitives.activate import ActivateResult
from l2core.primitives.click import (
    CheckRunner,
    JitterRng,
    VerifySpec,
    click_element,
)
from l2core.primitives.enter import enter_text
from l2core.primitives.scroll import (
    DIR_DOWN,
    ScrollResult,
    scroll_list,
    scroll_position,
)

__all__ = [
    "PointScan",
    "ScanError",
    "ScanPoint",
    "ScanResult",
    "scan_market",
]

class ScanError(Exception):
    """Ошибка сценария (модель/аргументы обхода), не эскалация рантайма."""

    def __init__(self, detail: str):
        super().__init__(f"scan_market: {detail}")
        self.detail = detail


@dataclass(frozen=True)
class ScanPoint:
    """Точка обхода обзора: таб + фильтры + поиск + verify готовности.

    tab/filters — element id в модели окна (клики через click_element,
    verify каждого — verify точки); search — текст ввода (опционально,
    требует search_region у сценария); verify — наблюдаемые признаки
    состояния «список готов» после настройки точки (инвариант 2).
    """

    verify: VerifySpec
    tab: Optional[str] = None
    filters: Tuple[str, ...] = ()
    search: Optional[str] = None


@dataclass(frozen=True)
class PointScan:
    """Результат одной точки обхода: агрегат строк + снимки кадров."""

    point: ScanPoint
    rows: Tuple[Row, ...]
    snapshots: Tuple[RowsResult, ...]


@dataclass(frozen=True)
class ScanResult:
    """Итог обхода: результаты точек по порядку; отказ/отмена — флагами.

    failures — диагностика отказа (needs_recover примитива: исполнение
    recover — вызывающий слой, 6.4); cancelled — прервано через cancelled
    (kill switch 8.1), HID-эффекты прекращены.
    """

    points: Tuple[PointScan, ...]
    cancelled: bool = False
    failures: Tuple[str, ...] = ()

    @property
    def needs_recover(self) -> bool:
        """Сигнал общего recover: отказ без отмены."""
        return bool(self.failures) and not self.cancelled


def _resolve_open_verify(model: UiModel, open_element: str,
                         open_from: str) -> VerifySpec:
    """Verify открытия обзора из перехода модели (recover_closed -> tab).

    Переход подтверждён метой (market.yaml); его отсутствие — ошибка
    модели (ScanError, fail fast): открывать обзор «вслепую» нельзя.
    """
    for transition in model.transitions:
        if (transition.source == open_from
                and transition.action.type == "click"
                and transition.action.element == open_element):
            return VerifySpec.from_transition(transition)
    raise ScanError(
        f"в модели {model.path!r} нет перехода открытия обзора "
        f"({open_from!r} --click {open_element!r}--> ...): verify "
        f"открытия неизвестен, открытие отменено")


def _click(element_id: str, *, verify: VerifySpec, model: UiModel,
           run_checks: CheckRunner, screen: Screen, hid,
           cursor_pos: Callable[[], Tuple[int, int]],
           activate: Callable[[], ActivateResult],
           window_status: Callable[[np.ndarray], WindowStatus],
           find_anchor: Callable[[np.ndarray], Optional[Anchor]],
           config: CoreConfig, hud_locator, on_fail: Tuple[str, ...],
           cancelled: Optional[Callable[[], bool]], rng: JitterRng,
           clock: Callable[[], float],
           sleep: Callable[[float], None]):
    """Клик по элементу обхода с единым набором инъекций сценария."""
    return click_element(
        element_id,
        model=model,
        verify=verify,
        run_checks=run_checks,
        screen=screen,
        hid=hid,
        cursor_pos=cursor_pos,
        activate=activate,
        window_status=window_status,
        find_anchor=find_anchor,
        config=config,
        hud_locator=hud_locator,
        on_fail=on_fail,
        cancelled=cancelled,
        rng=rng,
        clock=clock,
        sleep=sleep,
    )


def scan_market(
    points: Iterable[ScanPoint],
    *,
    model: UiModel,
    ocr,
    screen: Screen,
    hid,
    cursor_pos: Callable[[], Tuple[int, int]],
    activate: Callable[[], ActivateResult],
    window_status: Callable[[np.ndarray], WindowStatus],
    find_anchor: Callable[[np.ndarray], Optional[Anchor]],
    run_checks: CheckRunner,
    config: CoreConfig,
    hud_locator=None,
    search_region: Optional[OcrRegionSpec] = None,
    search_region_id: str = "search_field",
    search_field_element: Optional[str] = None,
    search_button_element: Optional[str] = None,
    open_element: str = "market_pic",
    open_from: str = "recover_closed",
    segmentation: Optional[TableSegmentation] = None,
    scroll_chunk: int = 3,
    max_scrolls: int = 200,
    on_fail: Tuple[str, ...] = (),
    free_area_point: Optional[Tuple[int, int]] = None,
    cancelled: Optional[Callable[[], bool]] = None,
    rng: Optional[JitterRng] = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> ScanResult:
    """Обход «Обзора лавок»: открытие -> для каждой точки (таб/фильтры/
    поиск) -> чтение строк со скроллом до признака конца из модели.
    Контракт — в докстринге модуля. Обзор — только разведка цен."""
    if scroll_chunk < 1:
        raise ScanError(f"scroll_chunk должен быть >= 1, получено {scroll_chunk}")
    if max_scrolls < 1:
        raise ScanError(f"max_scrolls должен быть >= 1, получено {max_scrolls}")
    points = tuple(points)
    if not points:
        return ScanResult(points=())
    for point in points:
        if point.search is not None and search_region is None:
            raise ScanError(
                f"точка обхода требует поиск {point.search!r}, но OCR-регион "
                f"поля ввода не задан (search_region; в модели окна "
                f"{model.window.id!r} ocr_regions не размечены — KNOWN_GAPS): "
                f"поиск без verify введённого текста запрещён")
    rng = rng or random.Random()
    seg = segmentation or (
        TableSegmentation(model.table_geometry)
        if model.table_geometry is not None else None)

    if cancelled is not None and cancelled():
        return ScanResult(points=(), cancelled=True)

    open_verify = _resolve_open_verify(model, open_element, open_from)
    opened = _click(
        open_element, verify=open_verify, model=model, run_checks=run_checks,
        screen=screen, hid=hid, cursor_pos=cursor_pos, activate=activate,
        window_status=window_status, find_anchor=find_anchor, config=config,
        hud_locator=hud_locator, on_fail=on_fail, cancelled=cancelled,
        rng=rng, clock=clock, sleep=sleep)
    if opened.cancelled:
        return ScanResult(points=(), cancelled=True)
    if opened.needs_recover:
        return ScanResult(points=(), failures=(
            f"открытие обзора ({open_element!r}) не подтверждено: "
            f"{'; '.join(opened.failures) or 'verify не сошёлся'}",))

    # Поиск: регион подшивается в копию модели (модель frozen, ручные
    # правки не трогаем) — enter_text читает ocr_regions по id.
    search_model = model
    if search_region is not None:
        search_model = dataclasses.replace(
            model, ocr_regions={**model.ocr_regions,
                                search_region_id: search_region})

    results = []
    for point in points:
        if cancelled is not None and cancelled():
            return ScanResult(points=tuple(results), cancelled=True)
        # --- Настройка точки обхода: таб -> фильтры -> поиск.
        setup_error = None
        for element_id in ((point.tab,) if point.tab else ()) + point.filters:
            clicked = _click(
                element_id, verify=point.verify, model=model,
                run_checks=run_checks, screen=screen, hid=hid,
                cursor_pos=cursor_pos, activate=activate,
                window_status=window_status, find_anchor=find_anchor,
                config=config, hud_locator=hud_locator, on_fail=on_fail,
                cancelled=cancelled, rng=rng, clock=clock, sleep=sleep)
            if clicked.cancelled:
                return ScanResult(points=tuple(results), cancelled=True)
            if clicked.needs_recover:
                setup_error = (
                    f"элемент {element_id!r} не подтверждён: "
                    f"{'; '.join(clicked.failures) or 'verify не сошёлся'}")
                break
        if setup_error is None and point.search is not None:
            entered = enter_text(
                point.search,
                model=search_model,
                region_id=search_region_id,
                ocr=ocr,
                screen=screen,
                hid=hid,
                cursor_pos=cursor_pos,
                activate=activate,
                window_status=window_status,
                find_anchor=find_anchor,
                run_checks=run_checks,
                config=config,
                field_element_id=search_field_element,
                search_button_id=search_button_element,
                expected_value=point.search,
                on_fail=on_fail,
                free_area_point=free_area_point,
                cancelled=cancelled,
                rng=rng,
                clock=clock,
                sleep=sleep,
            )
            if entered.cancelled:
                return ScanResult(points=tuple(results), cancelled=True)
            if not entered.success:
                setup_error = (
                    f"поиск {point.search!r} не подтверждён: "
                    f"{'; '.join(entered.failures) or 'OCR/verify не сошёлся'}")
        if setup_error is not None:
            results.append(PointScan(point=point, rows=(), snapshots=()))
            return ScanResult(points=tuple(results), failures=(setup_error,))

        # --- Чтение списка: кадр -> строки -> скролл -> до признака конца.
        collected: list = []
        snapshots: list = []
        scrolls = 0
        failure = None
        while True:
            if cancelled is not None and cancelled():
                return ScanResult(points=tuple(results), cancelled=True)
            frame = np.asarray(screen.grab())
            anchor = find_anchor(frame)
            if anchor is None:
                failure = "якорь не найден на кадре чтения строк — слепое чтение невозможно"
                break
            read = read_rows(
                frame, model=model, ocr=ocr, anchor=anchor,
                threshold=config.ocr.confidence_threshold,
                reread_count=config.ocr.reread_count,
                segmentation=seg,
            )
            snapshots.append(read)
            collected = list(dedup_rows([*collected, *read.rows]))
            position = scroll_position(
                frame, model=model, anchor=anchor, run_checks=run_checks,
                segmentation=seg)
            if position.at_end:
                break
            if scrolls >= max_scrolls:
                failure = (f"признак конца списка не получен за "
                           f"max_scrolls={max_scrolls} wheel-групп")
                break
            scrolled: ScrollResult = scroll_list(
                DIR_DOWN, scroll_chunk,
                model=model,
                screen=screen,
                hid=hid,
                cursor_pos=cursor_pos,
                activate=activate,
                window_status=window_status,
                find_anchor=find_anchor,
                run_checks=run_checks,
                config=config,
                segmentation=seg,
                on_fail=on_fail,
                free_area_point=free_area_point,
                cancelled=cancelled,
                rng=rng,
                clock=clock,
                sleep=sleep,
            )
            if scrolled.cancelled:
                return ScanResult(points=tuple(results), cancelled=True)
            if scrolled.needs_recover:
                failure = (f"скролл не подтверждён: "
                           f"{'; '.join(scrolled.failures) or 'verify не сошёлся'}")
                break
            scrolls += 1

        results.append(PointScan(point=point, rows=tuple(collected),
                                 snapshots=tuple(snapshots)))
        if failure is not None:
            return ScanResult(points=tuple(results), failures=(failure,))
    return ScanResult(points=tuple(results))
