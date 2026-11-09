"""Сценарий выставления лавки продажи setup_sell (задача 6.2, spec
market-scenarios: «Сценарий выставления лавки продажи» — drag предметов,
цены по slot из геометрии сетки, подтверждение через start_sell(); перед
необратимым подтверждением SHALL выполняться re-verify по кэшу кадра в
транзакции).

Сценарий — КОМПОЗИЦИЯ примитивов 5.x БЕЗ внутреннего состояния (spec:
«Сценарии — композиции примитивов без внутреннего состояния»): каждое
решение — от свежего кадра/детектора через инъекции (find_anchor /
window_status / run_checks / source_locator), ничего не кэшируется «в голове».

СЛОТЫ И ГЕОМЕТРИЯ СЕТКИ (design.md контракт: «slot: координаты ячейки
генерируются из геометрии сетки лавки, не размечаются вручную»):

- Сетка параметрична: GridGeometry (секция grid_geometry модели окна,
  формат — ui_model/FORMAT.md): origin/pitch/cell/cols/rows от якоря окна.
  Генератор slot_point(grid, index) считает центр ячейки (координаты окна),
  нумерация слотов row-major. Руками точки не размечаются.
- Источник констант: параметр grid сценария; если не задан — секция
  grid_geometry модели окна. Ни того ни другого нет — SetupSellError
  (fail fast, R4: не выдумывать константы). В отгруженной модели shop_sell
  секция не размечена (сырьё поле slot не описывает — KNOWN_GAPS,
  OWNER_QUESTIONS Q26): до калибровки константы передаёт стратегия.
- Точки drag'а: target = якорь(свежий) + центр слота; source — точка
  предмета в ИНВЕНТАРЕ, который в окна модели не входит (это HUD игры):
  источник даёт инъекция source_locator(item), вызываемая заново перед
  каждым drag'ом (инвентарь может двигаться — слепых drag нет).

ПОТОК setup_sell(items):

    открытие лавки (open_shop_sell через HUD: клик по open_element —
    HUD-иконка open_sale_trade из модели market, verify снаружи: переходы
    межоконные формат модели не выражает, KNOWN_GAPS п.3) -> для каждого
    SellItem(slot, price): свежий якорь + source_locator -> drag_item ->
    enter_price -> итоговый verify состояния (final_verify).

ОТКАЗЫ: needs_recover примитивов фиксирует отказ выставления (исполнение
recover — вызывающий слой, 6.4); отмена (kill switch 8.1, инъекция
cancelled) — результат с cancelled=True. Цена: OCR-регион поля цены и
элемент фокуса в shop_sell не размечены (ocr_regions пуст) — регион
подшивается в копию модели из параметра price_region (как search_region
в 6.1), без него — SetupSellError. Дубликат слота и цена < 1 — SetupSellError
(модель обхода некорректна, fail fast).

start_sell() — ПОТЕНЦИАЛЬНО НЕОБРАТИМО (спека action-primitives: при UNKNOWN
не повторяется — решение за стратегией):

    re-verify ПЕРЕД кликом (FrameTransaction.reverify, 3.6: свежий grab +
    проверка confirm_checks по свежему якорю) -> НЕ сошёлся — клик по
    кнопке «Начать продажу» НЕ выполняется (no blind click на необратимое),
    результат с reverified=False -> страж необратимых (5.7): guard.before/
    after вокруг клика — UNKNOWN блокирует повтор до явного
    confirm_after_unknown стратегией.

cancel_sell() — клик по кнопке отмены (элемент модели cansel_sell_button);
element_id=None — честный fallback на ESC (после пролога activate+found,
verify после действия — инвариант 2).

ГРАНИЦЫ (R4): семантика «поле одно на выбранный предмет»/«поле на слот»
сырьём не подтверждена — регион цены передаёт стратегия; verify открытия
лавки, drag'а, итоговый, confirm/cancel — параметры (модель transitions
пуста, KNOWN_GAPS п.3); сетка не откалибрована (Q26).
"""
from __future__ import annotations

import dataclasses
import random
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Tuple

import numpy as np

from l2core.adapters.hid import KEY_ESC
from l2core.adapters.screen import Screen
from l2core.config import CoreConfig
from l2core.detector.anchor import Anchor
from l2core.detector.status import STATUS_FOUND, WindowStatus
from l2core.detector.transaction import FrameTransaction
from l2core.model import GridGeometry, OcrRegionSpec, UiModel
from l2core.primitives.activate import ActivateResult
from l2core.primitives.click import (
    CheckRunner,
    ClickElementError,
    JitterRng,
    VerifySpec,
    _run_on_fail,
    click_element,
)
from l2core.primitives.drag import DragResult, drag_item
from l2core.primitives.enter import EnterResult, enter_price
from l2core.primitives.outcome import ActionOutcome, IrreversibleGuard, to_outcome

__all__ = [
    "CancelSellResult",
    "SellItem",
    "SetupSellError",
    "SetupSellResult",
    "StartSellResult",
    "cancel_sell",
    "setup_sell",
    "slot_point",
    "start_sell",
]


class SetupSellError(Exception):
    """Ошибка сценария (модель/аргументы выставления), не эскалация рантайма."""

    def __init__(self, detail: str):
        super().__init__(f"setup_sell: {detail}")
        self.detail = detail


@dataclass(frozen=True)
class SellItem:
    """Предмет к выставлению: слот сетки лавки (row-major) и цена (адена).

    slot — индекс ячейки grid_geometry (0..cols*rows-1); координаты центра
    ячейки генерирует slot_point, не размечаются вручную. Источник предмета
    (инвентарь) в item не входит: его точку даёт source_locator сценария.
    """

    slot: int
    price: int


@dataclass(frozen=True)
class ItemSetup:
    """Исход выставления одного предмета: drag + ввод цены."""

    item: SellItem
    drag: DragResult
    price: EnterResult


@dataclass(frozen=True)
class SetupSellResult:
    """Итог выставления: исходы по предметам; отказ/отмена — флагами.

    failures — диагностика отказа (needs_recover примитива: исполнение
    recover — вызывающий слой, 6.4); cancelled — прервано через cancelled
    (kill switch 8.1), HID-эффекты прекращены.
    """

    items: Tuple[ItemSetup, ...]
    cancelled: bool = False
    failures: Tuple[str, ...] = ()

    @property
    def needs_recover(self) -> bool:
        """Сигнал общего recover: отказ без отмены."""
        return bool(self.failures) and not self.cancelled


@dataclass(frozen=True)
class StartSellResult:
    """Итог подтверждения лавки (потенциально необратимое действие).

    reverified — прошёл ли re-verify перед кликом (False — клик НЕ
    выполнялся); outcome — трёхзначный итог клика (None при отмене/блоке
    re-verify): UNKNOWN означает «клик, скорее всего, ушёл, подтвердить
    нельзя» — повтор запрещён стражем до явного confirm_after_unknown.
    """

    success: bool
    outcome: Optional[ActionOutcome]
    reverified: bool
    cancelled: bool = False
    failures: Tuple[str, ...] = ()

    @property
    def needs_recover(self) -> bool:
        """Сигнал общего recover: неуспех без отмены."""
        return not self.success and not self.cancelled


@dataclass(frozen=True)
class CancelSellResult:
    """Итог отмены выставления (клик по отмене или ESC)."""

    success: bool
    attempts: int
    cancelled: bool = False
    unconfirmed: bool = False
    failures: Tuple[str, ...] = ()

    @property
    def needs_recover(self) -> bool:
        """Сигнал общего recover: неуспех без отмены."""
        return not self.success and not self.cancelled


def slot_point(grid: GridGeometry, index: int) -> Tuple[int, int]:
    """Центр ячейки сетки в координатах ОКНА (от якоря), row-major.

    index = row * cols + col. Вне диапазона 0..cols*rows-1 — SetupSellError
    (fail fast: слота нет, drag в никуда запрещён).
    """
    if not 0 <= index < grid.capacity:
        raise SetupSellError(
            f"слот {index} вне сетки {grid.cols}x{grid.rows} "
            f"(ёмкость {grid.capacity})")
    col, row = index % grid.cols, index // grid.cols
    return (grid.origin[0] + col * grid.pitch[0] + grid.cell[0] // 2,
            grid.origin[1] + row * grid.pitch[1] + grid.cell[1] // 2)


def _resolve_grid(grid: Optional[GridGeometry], model: UiModel
                  ) -> GridGeometry:
    """Геометрия сетки: параметр сценария, иначе секция модели (fail fast)."""
    resolved = grid if grid is not None else model.grid_geometry
    if resolved is None:
        raise SetupSellError(
            f"геометрия сетки слотов не задана: ни параметром grid, ни "
            f"секцией grid_geometry модели окна {model.window.id!r} "
            f"(в отгруженной модели не размечена — KNOWN_GAPS, Q26): "
            f"координаты слотов не выдумываются (R4)")
    return resolved


def _validate_items(items: Tuple[SellItem, ...], grid: GridGeometry) -> None:
    """Модель обхода корректна: слоты в сетке, без повторов, цена >= 1."""
    seen = set()
    for item in items:
        if not 0 <= item.slot < grid.capacity:
            raise SetupSellError(
                f"слот {item.slot} вне сетки {grid.cols}x{grid.rows} "
                f"(ёмкость {grid.capacity})")
        if item.slot in seen:
            raise SetupSellError(
                f"повтор слота {item.slot}: два предмета в одну ячейку "
                f"сырьём не подтверждены — модель обхода некорректна")
        seen.add(item.slot)
        if item.price < 1:
            raise SetupSellError(
                f"цена {item.price} предмета в слоте {item.slot} < 1 — "
                f"адена не бывает нулевой (дефолт диапазона price)")


def setup_sell(
    items: Iterable[SellItem],
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
    grid: Optional[GridGeometry] = None,
    hud_model: Optional[UiModel] = None,
    hud_locator=None,
    open_element: Optional[str] = "open_sale_trade",
    open_verify: Optional[VerifySpec] = None,
    source_locator: Callable[[SellItem], Optional[Tuple[int, int]]],
    price_region: Optional[OcrRegionSpec] = None,
    price_region_id: str = "price_field",
    price_field_element: Optional[str] = None,
    price_submit: bool = True,
    drag_verify: VerifySpec,
    final_verify: Optional[VerifySpec] = None,
    on_fail: Tuple[str, ...] = (),
    free_area_point: Optional[Tuple[int, int]] = None,
    cancelled: Optional[Callable[[], bool]] = None,
    rng: Optional[JitterRng] = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> SetupSellResult:
    """Выставление лавки продажи: открытие -> drag+цена на предмет ->
    итоговый verify. Контракт — в докстринге модуля.

    model — модель окна shop_sell; hud_model — модель, содержащая
    HUD-элемент открытия (market, open_sale_trade); без неё открытие
    невозможно (open_element есть, а элемент не найден — SetupSellError).
    open_element=None — открытие пропускается (лавка открыта вызывающим
    слоем); тогда open_verify не нужен.
    """
    items = tuple(items)
    grid = _resolve_grid(grid, model)
    _validate_items(items, grid)
    if not items:
        return SetupSellResult(items=())
    rng = rng or random.Random()

    # Цена: регион поля в shop_sell не размечен (ocr_regions пуст) —
    # подшивается в копию модели из параметра (ручные правки не трогаем),
    # как search_region в 6.1; без региона ввод цены без OCR-verify
    # запрещён (инвариант 2).
    region = price_region if price_region is not None else \
        model.ocr_regions.get(price_region_id)
    if region is None:
        raise SetupSellError(
            f"OCR-регион поля цены не задан: ни параметром price_region, ни "
            f"секцией ocr_regions модели окна {model.window.id!r} "
            f"(id {price_region_id!r}; в отгруженной модели пусто — "
            f"KNOWN_GAPS): ввод цены без OCR-verify запрещён")
    price_model = dataclasses.replace(
        model, ocr_regions={**model.ocr_regions, price_region_id: region})

    if cancelled is not None and cancelled():
        return SetupSellResult(items=(), cancelled=True)

    # --- Открытие лавки через HUD (как open в 6.1, межоконный переход
    # формат модели не выражает — verify открытия передаёт стратегия).
    if open_element is not None:
        open_model = hud_model if hud_model is not None else model
        if open_verify is None:
            raise SetupSellError(
                f"открытие лавки ({open_element!r}) требует open_verify: "
                f"клики без verify запрещены (инвариант 2), перехода "
                f"модель не выражает (KNOWN_GAPS п.3)")
        opened = click_element(
            open_element,
            model=open_model,
            verify=open_verify,
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
            free_area_point=free_area_point,
            cancelled=cancelled,
            rng=rng,
            clock=clock,
            sleep=sleep,
        )
        if opened.cancelled:
            return SetupSellResult(items=(), cancelled=True)
        if opened.needs_recover:
            return SetupSellResult(items=(), failures=(
                f"открытие лавки ({open_element!r}) не подтверждено: "
                f"{'; '.join(opened.failures) or 'verify не сошёлся'}",))

    outcomes: list = []
    for item in items:
        if cancelled is not None and cancelled():
            return SetupSellResult(items=tuple(outcomes), cancelled=True)
        # Точки drag'а: якорь свежий + источник из инъекции (инвентарь вне
        # модели, локатор решает сам — заново на каждый предмет).
        frame = np.asarray(screen.grab())
        anchor = find_anchor(frame)
        if anchor is None:
            return SetupSellResult(items=tuple(outcomes), failures=(
                f"предмет слота {item.slot}: якорь не найден на кадре — "
                f"слепой drag запрещён",))
        target = slot_point(grid, item.slot)
        target_point = (anchor.x + target[0], anchor.y + target[1])
        source_point = source_locator(item)
        if source_point is None:
            return SetupSellResult(items=tuple(outcomes), failures=(
                f"предмет слота {item.slot}: источник в инвентаре не "
                f"локализован (source_locator) — слепой drag запрещён",))
        dragged = drag_item(
            source_point,
            target_point,
            verify=drag_verify,
            run_checks=run_checks,
            screen=screen,
            hid=hid,
            cursor_pos=cursor_pos,
            activate=activate,
            window_status=window_status,
            find_anchor=find_anchor,
            config=config,
            on_fail=on_fail,
            free_area_point=free_area_point,
            cancelled=cancelled,
            rng=rng,
            clock=clock,
            sleep=sleep,
        )
        if dragged.cancelled:
            return SetupSellResult(items=tuple(outcomes), cancelled=True)
        if not dragged.success:
            return SetupSellResult(items=tuple(outcomes), failures=(
                f"предмет слота {item.slot}: drag не подтверждён: "
                f"{'; '.join(dragged.failures) or 'verify не сошёлся'}",))
        priced = enter_price(
            item.price,
            model=price_model,
            region_id=price_region_id,
            ocr=ocr,
            screen=screen,
            hid=hid,
            cursor_pos=cursor_pos,
            activate=activate,
            window_status=window_status,
            find_anchor=find_anchor,
            run_checks=run_checks,
            config=config,
            field_element_id=price_field_element,
            submit=price_submit,
            on_fail=on_fail,
            free_area_point=free_area_point,
            cancelled=cancelled,
            rng=rng,
            clock=clock,
            sleep=sleep,
        )
        if priced.cancelled:
            return SetupSellResult(items=tuple(outcomes), cancelled=True)
        if not priced.success:
            return SetupSellResult(items=tuple(outcomes), failures=(
                f"предмет слота {item.slot}: цена {item.price} не подтверждена "
                f"(OCR/verify не сошёлся): "
                f"{'; '.join(priced.failures) or 'ввод не сошёлся'}",))
        outcomes.append(ItemSetup(item=item, drag=dragged, price=priced))

    # --- Итоговый verify состояния лавки (свежий кадр + якорь).
    if final_verify is not None:
        frame = np.asarray(screen.grab())
        anchor = find_anchor(frame)
        report = None
        if anchor is not None:
            report = run_checks(final_verify.checks, frame, anchor)
        if anchor is None or not report.passed:
            detail = ("якорь не найден на кадре итогового verify"
                      if anchor is None else
                      f"признаки не сошлись: "
                      f"{'; '.join(report.failures) or 'verify не пройден'}")
            return SetupSellResult(items=tuple(outcomes), failures=(
                f"итоговый verify лавки не пройден ({detail})",))
    return SetupSellResult(items=tuple(outcomes))


def start_sell(
    *,
    model: UiModel,
    screen: Screen,
    hid,
    cursor_pos: Callable[[], Tuple[int, int]],
    activate: Callable[[], ActivateResult],
    window_status: Callable[[np.ndarray], WindowStatus],
    find_anchor: Callable[[np.ndarray], Optional[Anchor]],
    run_checks: CheckRunner,
    config: CoreConfig,
    confirm_checks: VerifySpec,
    verify: VerifySpec,
    element_id: str = "sell_start_button",
    guard: Optional[IrreversibleGuard] = None,
    action_id: str = "start_sell",
    hud_locator=None,
    on_fail: Tuple[str, ...] = (),
    free_area_point: Optional[Tuple[int, int]] = None,
    cancelled: Optional[Callable[[], bool]] = None,
    rng: Optional[JitterRng] = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> StartSellResult:
    """Подтверждение лавки («Начать продажу»): ПОТЕНЦИАЛЬНО НЕОБРАТИМО.

    Re-verify перед кликом (3.6: свежий grab + confirm_checks по свежему
    якорю); не сошёлся — клик НЕ выполняется (no blind click на необратимое).
    Страж необратимых (5.7): guard.before/after вокруг клика — UNKNOWN
    блокирует повтор до явного confirm_after_unknown стратегией
    (IrreversibleRepeatError проброшена, fail fast).
    """
    if cancelled is not None and cancelled():
        return StartSellResult(success=False, outcome=None, reverified=False,
                               cancelled=True)
    rng = rng or random.Random()

    # --- Re-verify перед необратимым (свежий кадр, не кэш).
    with FrameTransaction(screen) as tx:
        def _confirm(frame: np.ndarray) -> bool:
            anchor = find_anchor(frame)
            if anchor is None:
                return False
            return run_checks(confirm_checks.checks, frame, anchor).passed

        reverified = tx.reverify(_confirm)
    if not reverified:
        return StartSellResult(
            success=False, outcome=None, reverified=False,
            failures=("re-verify перед подтверждением не прошёл (свежий "
                      "кадр не подтвердил цель) — клик по «Начать продажу» "
                      "не выполняется (необратимое действие без verify "
                      "запрещено)",))

    if guard is not None:
        guard.before(action_id)  # UNKNOWN не снят — повтор запрещён
    clicked = click_element(
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
        free_area_point=free_area_point,
        cancelled=cancelled,
        rng=rng,
        clock=clock,
        sleep=sleep,
    )
    if clicked.cancelled:
        return StartSellResult(success=False, outcome=None, reverified=True,
                               cancelled=True)
    outcome = to_outcome(clicked)
    if guard is not None:
        guard.after(action_id, outcome,
                    detail="; ".join(clicked.failures) or "verify не сошёлся")
    return StartSellResult(success=clicked.success, outcome=outcome,
                           reverified=True, failures=clicked.failures)


def cancel_sell(
    *,
    model: UiModel,
    screen: Screen,
    hid,
    cursor_pos: Callable[[], Tuple[int, int]],
    activate: Callable[[], ActivateResult],
    window_status: Callable[[np.ndarray], WindowStatus],
    find_anchor: Callable[[np.ndarray], Optional[Anchor]],
    run_checks: CheckRunner,
    config: CoreConfig,
    verify: VerifySpec,
    element_id: Optional[str] = "cansel_sell_button",
    on_fail: Tuple[str, ...] = (),
    free_area_point: Optional[Tuple[int, int]] = None,
    cancelled: Optional[Callable[[], bool]] = None,
    rng: Optional[JitterRng] = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> CancelSellResult:
    """Отмена выставления лавки: клик по кнопке отмены (элемент модели) или
    ESC (element_id=None — честный fallback, если кнопки в модели нет).

    ESC-путь: пролог (activate OK + статус found, иначе CancelSellError,
    HID не трогается — инвариант 9) -> ESC -> verify на свежем кадре ->
    retry (1 + click_retry_count) -> on_fail[].
    """
    rng = rng or random.Random()
    if element_id is not None:
        clicked = click_element(
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
            on_fail=on_fail,
            free_area_point=free_area_point,
            cancelled=cancelled,
            rng=rng,
            clock=clock,
            sleep=sleep,
        )
        return CancelSellResult(success=clicked.success,
                                attempts=clicked.attempts,
                                cancelled=clicked.cancelled,
                                unconfirmed=clicked.unconfirmed,
                                failures=clicked.failures)

    # --- Fallback на ESC (кнопки отмены в модели нет).
    max_attempts = 1 + config.timing.click_retry_count
    failures: list = []
    effect_unconfirmed = False
    for attempt in range(1, max_attempts + 1):
        if cancelled is not None and cancelled():
            return CancelSellResult(success=False, attempts=attempt - 1,
                                    cancelled=True)
        with FrameTransaction(screen) as tx:
            activated = activate()
            if activated.cancelled:
                return CancelSellResult(success=False, attempts=attempt - 1,
                                        cancelled=True)
            status = window_status(tx.frame)
            if status.status != STATUS_FOUND:
                raise _CancelPreconditionError(status)
            hid.key_press(KEY_ESC)
            verify_frame = tx.refresh()
            verify_anchor = find_anchor(verify_frame)
            if verify_anchor is None:
                failures.append(f"попытка {attempt}: якорь потерян перед verify")
                effect_unconfirmed = True  # ESC мог уйти, подтвердить нельзя
                continue
            report = run_checks(verify.checks, verify_frame, verify_anchor)
            if report.passed:
                return CancelSellResult(success=True, attempts=attempt)
            failures.append(
                f"попытка {attempt}: verify не подтверждён "
                f"({'; '.join(report.failures) or 'признаки не совпали'})")
    # Исчерпание retry: повторный пролог перед on_fail (инвариант 9).
    reactivated = activate()
    if reactivated.cancelled:
        return CancelSellResult(success=False, attempts=max_attempts,
                                cancelled=True)
    status = window_status(np.asarray(screen.grab()))
    if status.status != STATUS_FOUND:
        failures.append(
            f"on_fail пропущен: статус окна {status.status!r} "
            f"({status.reason or status.content_detail}) — слепые HID-шаги "
            f"запрещены")
        return CancelSellResult(success=False, attempts=max_attempts,
                                unconfirmed=effect_unconfirmed,
                                failures=tuple(failures))
    try:
        _run_on_fail(on_fail, hid=hid, cursor_pos=cursor_pos,
                     free_area_point=free_area_point, config=config, rng=rng)
    except ClickElementError as exc:
        raise SetupSellError(exc.detail) from exc
    return CancelSellResult(success=False, attempts=max_attempts,
                            unconfirmed=effect_unconfirmed,
                            failures=tuple(failures))


class _CancelPreconditionError(SetupSellError):
    """Пролог не пройден: статус окна не found — ESC запрещён (инвариант 9)."""

    def __init__(self, status: WindowStatus):
        self.status = status.status
        super().__init__(
            f"отмена через ESC: статус окна {status.status!r} "
            f"({status.reason or status.content_detail}) — ESC не отправляется"
        )
