"""Примитив клика по элементу модели (задача 5.3, spec action-primitives:
«Каждое действие верифицируется, при неудаче — retry → on_fail → recover»,
«Устойчивость к hover и тултипам»; инвариант 9: слепых кликов нет).

click_element — оркестратор, весь внешний мир — инъекции (HID, экран/
транзакция кадра, activate, статус окна, поиск якоря, оценка проверок,
rng, часы):

    пролог → цикл попыток (retry из конфига) → on_fail[] → сигнал recover

ПРОЛОГ (binding ruling из ревью 5.2, зафиксирован в AGENTS.md): клик
допустим только при activate_game OK (5.2, иначе ActivateGameError проброшена
— эскалация) И статусе окна found (3.5: не occluded_or_unstable/not_found).
Иначе — исключение, HID не трогается (no blind clicks).

ЦИКЛ ПОПЫТОК (попыток = 1 + timing.click_retry_count):
1. Тултип-гипотеза (attempt > 1): после verify-fail курсор уводится в сторону
   на TOOLTIP_SHIFT_PX и паузой timing.hover_pause ждётся исчезновение.
   Честная граница: модель не содержит элемента-тултипа (детекции из кадра
   нет — KNOWN_GAPS), это механика «сдвинуть и повторить», не детекция;
2. Свежий кадр (FrameTransaction.refresh, 3.6), свежий якорь (инъекция,
   кэша позиции нет — инвариант 3). Якоря нет → попытка потрачена;
3. Точка клика: click_point варианта (от rect элемента, иначе центр rect;
   HUD — только через инъекцию hud_locator, без него клик невозможен)
   + джиттер behavior.click_jitter_px через rng (инъекция с seed — инвариант
   7); движение — относительными дельтами (HID-протокол), текущую позицию
   даёт инъекция cursor_pos;
4. Hover: курсор наведён на точку, затем hover-пауза (timing.hover_pause) и
   re-verify присутствия элемента (проверка шаблона варианта на свежем
   кадре): hover мог изменить вид/перекрыть элемент — клика по чужому
   состоянию нет. Точка клика пересчитывается от свежего (post-hover)
   якоря: при сдвиге окна курсор корректируется дельтой (джиттер считается
   раз на попытку, пересчёт rng не расходует). Провал re-verify → попытка
   потрачена;
5. HID click (hold_ms из behavior.press_delay_min/max_ms через rng —
   детерминизм вместо вендорного random модуля);
6. Verify (инвариант 2): verify.checks (VerifySpec — спецификация перехода
   модели) оцениваются на свежем кадре через инъекцию run_checks. Успех →
   ClickResult(success=True). Иначе → следуюшая попытка.

HOVER-ПАУЗА БЕЗ SLEEP (инвариант 1): фиксированная задержка — не ожидание
условия, поэтому механика — wait_until по инъецированным часам с предикатом
«прошло >= seconds» (интервал — timing.verify_interval, единая точка
дефолтов). WaitResult игнорируется: задержка гарантирована самим механизмом
дедлайна wait_until (к возврату прошло ровно timeout, дольше не ждёт), а
EPS-дрейф float на финальном poll значения не имеет — это задержка, а не
граница условия. Прямых вызовов sleep здесь нет.

ON_FAIL[]: шаги перехода модели — «press_esc» (HID ESC) и «click_free_area»
(клик в свободную область клиента игры; точка — free_area_point или центр
config.resolution — честное допущение: свободная область не размечена
моделью, параметризуемо; точная цепочка — 5.8). Перед on_fail пролог
повторяется (activate + статус found): без него click_free_area был бы
слепым кликом при деградировавшем окне (инвариант 9) — при провале on_fail
пропускается, результат сразу needs_recover=True с причиной в failures.
Неизвестный шаг — ошибка модели (ClickElementError, fail fast).

РЕЗУЛЬТАТ: ClickResult (success/attempts/cancelled/unconfirmed/
on_fail_executed/failures). Отмена (kill switch 8.1, инъекция cancelled) —
флаг в результате, не исключение (консистентно с wait.py/activate.py);
трёхзначная обёртка OK/FAILED/UNKNOWN — адаптер outcome.py (5.7) поверх
полей success/unconfirmed. Эскалации предусловий — исключениями
(инвариант 9).

ГРАНИЦЫ: детекции тултипа из кадра нет (см. выше); перекрытие окном игры
статус-детектор видит между кадрами (3.5) — пролог ловит occluded; verify
оценивает шаблонные/прочие проверки через run_checks — сама оценка (matcher,
assets) остаётся слоем детекции (wiring — сценарии 6.x).
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

from l2core.adapters.hid import BTN_LEFT, KEY_ESC
from l2core.adapters.screen import Screen
from l2core.config import CoreConfig
from l2core.detector.anchor import Anchor
from l2core.detector.status import STATUS_FOUND, WindowStatus
from l2core.detector.transaction import FrameTransaction
from l2core.model import Check, ElementSpec, UiModel
from l2core.primitives.activate import ActivateResult
from l2core.primitives.wait import wait_until

__all__ = [
    "TOOLTIP_SHIFT_PX",
    "CheckReport",
    "CheckRunner",
    "ClickElementError",
    "ClickPreconditionError",
    "ClickResult",
    "VerifySpec",
    "click_element",
]

# Сдвиг курсора при тултип-гипотезе: уводим вправо-вниз на фиксированное
# число пикселей (тултипы встают над элементом/справа от курсора; точное
# направление не принципиально — следующая попытка возвращает курсор к цели).
TOOLTIP_SHIFT_PX = 24
# Дедлайн settle-подтверждения verify после потери якоря (смена вида
# окна, анимация): дольше — UNKNOWN. Не конфиг: компромисс скорости
# ретрая и шанса дождаться анимации.
_VERIFY_SETTLE_TIMEOUT = 2.0

# Шаги on_fail[], известные примитиву (остальные — ошибка модели).
_ON_FAIL_PRESS_ESC = "press_esc"
_ON_FAIL_CLICK_FREE_AREA = "click_free_area"


@dataclass(frozen=True)
class VerifySpec:
    """Verify-спецификация клика: наблюдаемые признаки целевого состояния.

    checks — проверки перехода модели (template/pixel_probe/ocr); expected —
    id ожидаемого состояния (диагностика/логирование вызывающим слоем).
    """

    checks: Tuple[Check, ...]
    expected: Optional[str] = None

    @classmethod
    def from_transition(cls, transition) -> "VerifySpec":
        """Спецификация из перехода модели (задача 2.5)."""
        return cls(checks=tuple(transition.verify), expected=transition.expected)


@dataclass(frozen=True)
class CheckReport:
    """Итог оценки набора проверок: passed + причины проваленных."""

    passed: bool
    failures: Tuple[str, ...] = ()


@runtime_checkable
class CheckRunner(Protocol):
    """Оценка проверок по кадру и якорю (wiring — слой детекции, 6.x).

    Семантика: passed — совпали ВСЕ checks (как сигнатуры состояний, 3.4).
    """

    def __call__(self, checks: Tuple[Check, ...], frame: np.ndarray,
                 anchor: Optional[Anchor]) -> CheckReport: ...


@runtime_checkable
class JitterRng(Protocol):
    """Источник «случайности» профиля: randint как у random.Random."""

    def randint(self, a: int, b: int) -> int: ...


@dataclass(frozen=True)
class ClickResult:
    """Итог клика (успех/отмена/сигнал recover); трёхзначная обёртка
    OK/FAILED/UNKNOWN — адаптер outcome.py (5.7) поверх полей
    success/unconfirmed.

    success — verify подтверждён; attempts — число потраченных попыток;
    cancelled — прервано через cancelled (kill switch 8.1), verify не
    подтверждён; unconfirmed — HID-эффект, скорее всего, уже применён (клик
    ушёл), но подтвердить его невозможно (якорь потерян после действия) —
    итог UNKNOWN в адаптере outcome.py (5.7); on_fail_executed — шаги
    on_fail[] исполнены; failures — диагностика провалов по попыткам.
    """

    element_id: str
    success: bool
    attempts: int
    cancelled: bool = False
    unconfirmed: bool = False
    on_fail_executed: bool = False
    failures: Tuple[str, ...] = ()

    @property
    def needs_recover(self) -> bool:
        """Сигнал общего recover (5.8): неуспех без отмены."""
        return not self.success and not self.cancelled


class ClickElementError(Exception):
    """Ошибка исполнения примитива (модель/аргументы), не эскалация рантайма."""

    def __init__(self, detail: str):
        super().__init__(f"click_element: {detail}")
        self.detail = detail


class ClickPreconditionError(ClickElementError):
    """Пролог не пройден: статус окна не found — клик запрещён (инвариант 9).

    activate_game не удался — отдельно: ActivateGameError (5.2) проброшена.
    """

    def __init__(self, element_id: str, status: WindowStatus):
        self.status = status.status
        super().__init__(
            f"элемент {element_id!r}: статус окна {status.status!r} "
            f"({status.reason or status.content_detail}) — клик не выполняется"
        )


def _fixed_pause(seconds: float, *, config: CoreConfig,
                 cancelled: Optional[Callable[[], bool]],
                 clock: Callable[[], float],
                 sleep: Callable[[float], None]) -> None:
    """Фиксированная задержка без прямого sleep (инвариант 1, см. модуль).

    Предикат — «прошло >= seconds» по инъецированным часам; результат
    wait_until игнорируется: к моменту возврата задержка выдержана по
    построению механизма дедлайна.
    """
    start = clock()
    wait_until(
        lambda: clock() - start >= seconds,
        timeout=seconds,
        interval=config.timing.verify_interval,
        cancelled=cancelled,
        clock=clock,
        sleep=sleep,
    )


def _move_to(hid, cursor_pos: Callable[[], Tuple[int, int]],
             point: Tuple[int, int]) -> None:
    """Навести курсор на абсолютную точку относительными дельтами HID."""
    cx, cy = cursor_pos()
    hid.move_rel(point[0] - cx, point[1] - cy)


def _jitter(rng: JitterRng, px: int) -> Tuple[int, int]:
    if px <= 0:
        return 0, 0
    return rng.randint(-px, px), rng.randint(-px, px)


def _click_point(element: ElementSpec, variant: Optional[str],
                 anchor: Anchor, frame: np.ndarray, hud_locator,
                 config: CoreConfig, jitter: Tuple[int, int]
                 ) -> Optional[Tuple[int, int]]:
    """Абсолютная точка клика или None (HUD не локализован).

    Обычный элемент: anchor + rect + click_point варианта (иначе центр rect).
    HUD: точку даёт только hud_locator(element, variant, frame) — без него
    позиция в search_area неизвестна и слепой клик запрещён (ClickElementError).
    Джиттер профиля передаётся готовым (считается раз на попытку): пересчёт
    точки от другого якоря не расходует rng и детерминирован (инвариант 7).
    """
    jx, jy = jitter
    if element.hud:
        if hud_locator is None:
            raise ClickElementError(
                f"HUD-элемент {element.id!r}: hud_locator не задан — позицию "
                f"клика определить нельзя (слепой клик запрещён)")
        point = hud_locator(element, variant, frame)
        if point is None:
            return None
        return point[0] + jx, point[1] + jy
    rect = element.rect
    click_point: Optional[Tuple[int, int]] = None
    spec = element.variants.get(variant) if variant else None
    if spec is not None and spec.click_point is not None:
        click_point = spec.click_point
    if click_point is None:
        click_point = next(
            (v.click_point for v in element.variants.values()
             if v.click_point is not None),
            (rect.width // 2, rect.height // 2),
        )
    return anchor.x + rect.x + click_point[0] + jx, anchor.y + rect.y + click_point[1] + jy


def _hold_ms(config: CoreConfig, rng: JitterRng) -> int:
    return rng.randint(config.behavior.press_delay_min_ms,
                       config.behavior.press_delay_max_ms)


def _run_on_fail(on_fail: Tuple[str, ...], *, hid,
                 cursor_pos: Callable[[], Tuple[int, int]],
                 free_area_point: Optional[Tuple[int, int]],
                 config: CoreConfig, rng: JitterRng) -> None:
    """Шаги on_fail[] перехода: простые пассивные HID-действия.

    «press_esc» — ESC; «click_free_area» — клик в свободную область клиента
    игры (точка — free_area_point или центр config.resolution; честное
    допущение — область не размечена моделью, уточнение — в 5.8).
    Неизвестный шаг — ошибка модели (fail fast).
    """
    for step in on_fail:
        if step == _ON_FAIL_PRESS_ESC:
            hid.key_press(KEY_ESC)
        elif step == _ON_FAIL_CLICK_FREE_AREA:
            point = free_area_point or (
                config.resolution.width // 2, config.resolution.height // 2)
            _move_to(hid, cursor_pos, point)
            hid.click(BTN_LEFT, 1, _hold_ms(config, rng))
        else:
            raise ClickElementError(f"неизвестный шаг on_fail: {step!r}")


def click_element(
    element_id: str,
    *,
    model: UiModel,
    verify: VerifySpec,
    run_checks: CheckRunner,
    screen: Screen,
    hid,
    cursor_pos: Callable[[], Tuple[int, int]],
    activate: Callable[[], ActivateResult],
    window_status: Callable[[np.ndarray], WindowStatus],
    find_anchor: Callable[[np.ndarray], Optional[Anchor]],
    config: CoreConfig,
    variant: Optional[str] = None,
    on_fail: Tuple[str, ...] = (),
    hud_locator: Optional[Callable[[ElementSpec, Optional[str], np.ndarray],
                                   Optional[Tuple[int, int]]]] = None,
    free_area_point: Optional[Tuple[int, int]] = None,
    cancelled: Optional[Callable[[], bool]] = None,
    rng: Optional[JitterRng] = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    allow_anchorless_verify: bool = True,
) -> ClickResult:
    """Клик по элементу модели: пролог → hover → click → verify → retry →
    on_fail → сигнал recover. Контракт — в докстринге модуля.

    verify — обязателен: клики без verify-спецификации запрещены (spec
    action-primitives; guard — tests/test_click.py::TestVerifyContract).

    allow_anchorless_verify — якорь исчез с verify-кадра (клик закрыл/сменил
    окно, анимация): verify-набор оценивается без якоря (HUD-признаки
    глобального состояния, окно hud_window модели). Для ПОТЕНЦИАЛЬНО
    НЕОБРАТИМЫХ действий (start_sell, 5.7) вызывающая сторона передаёт
    False: окно обязано остаться, потеря якоря — UNKNOWN со стражом.
    """
    element = model.elements.get(element_id)
    if element is None:
        raise ClickElementError(
            f"элемент {element_id!r} отсутствует в модели окна {model.window.id!r}")
    resolved_variant = variant or next(iter(element.variants), None)
    rng = rng or random.Random()
    failures: list = []

    with FrameTransaction(screen) as tx:
        # --- ПРОЛОГ: activate OK (5.2) + статус found (3.5), иначе клика нет.
        activated = activate()
        if activated.cancelled:
            return ClickResult(element_id, success=False, attempts=0, cancelled=True)
        status = window_status(tx.frame)
        if status.status != STATUS_FOUND:
            raise ClickPreconditionError(element_id, status)

        max_attempts = 1 + config.timing.click_retry_count
        # Эффект неподтверждён: клик ушёл в HID, но verify оказался
        # невозможен (якорь потерян на свежем кадре после клика) — итог
        # UNKNOWN в адаптере outcome.py (5.7), не FAILED.
        effect_unconfirmed = False
        for attempt in range(1, max_attempts + 1):
            if cancelled is not None and cancelled():
                return ClickResult(element_id, success=False,
                                   attempts=attempt - 1, cancelled=True)
            if attempt > 1:
                # Тултип-гипотеза: уводим курсор в сторону, ждём исчезновения
                # (детекции тултипа из кадра модель не даёт — честная граница).
                hid.move_rel(TOOLTIP_SHIFT_PX, TOOLTIP_SHIFT_PX)
                _fixed_pause(config.timing.hover_pause, config=config,
                             cancelled=cancelled, clock=clock, sleep=sleep)

            frame = tx.refresh()
            anchor = find_anchor(frame)
            if anchor is None:
                failures.append(f"попытка {attempt}: якорь не найден на кадре")
                continue
            jitter = _jitter(rng, config.behavior.click_jitter_px)
            point = _click_point(element, resolved_variant, anchor, frame,
                                 hud_locator, config, jitter)
            if point is None:
                failures.append(
                    f"попытка {attempt}: HUD-элемент {element_id!r} не "
                    f"локализован в search_area")
                continue

            _move_to(hid, cursor_pos, point)
            _fixed_pause(config.timing.hover_pause, config=config,
                         cancelled=cancelled, clock=clock, sleep=sleep)
            if cancelled is not None and cancelled():
                return ClickResult(element_id, success=False,
                                   attempts=attempt - 1, cancelled=True)

            # Re-verify после hover-паузы: элемент на месте и не перекрыт
            # (hover мог изменить вид/накрыть тултипом — кликать вслепую нет).
            hover_frame = tx.refresh()
            hover_anchor = find_anchor(hover_frame)
            if hover_anchor is None:
                failures.append(f"попытка {attempt}: якорь потерян после hover")
                continue
            if resolved_variant is not None:
                presence = run_checks(
                    (Check(kind="template", element=element_id,
                           variant=resolved_variant),),
                    hover_frame, hover_anchor)
                if not presence.passed:
                    failures.append(
                        f"попытка {attempt}: элемент не подтверждён после "
                        f"hover ({'; '.join(presence.failures) or 'шаблон не совпал'})")
                    continue
            # Точка клика пересчитывается от свежего (post-hover) якоря:
            # за паузу окно могло сдвинуться — корректируем курсор дельтой
            # (джиттер тот же, rng не расходуется; при неподвижном якоре
            # дельта нулевая и HID-трафика лишнего нет).
            final_point = _click_point(element, resolved_variant, hover_anchor,
                                       hover_frame, hud_locator, config, jitter)
            if final_point is None:
                failures.append(
                    f"попытка {attempt}: HUD-элемент {element_id!r} не "
                    f"локализован после hover")
                continue
            if final_point != point:
                _move_to(hid, cursor_pos, final_point)

            hid.click(BTN_LEFT, 1, _hold_ms(config, rng))

            verify_frame = tx.refresh()
            verify_anchor = find_anchor(verify_frame)
            if verify_anchor is None:
                # Клик мог закрыть/сменить вид окна: якорь исчез с
                # verify-кадра раньше анимации. Два пути подтверждения:
                # 1) якорно-независимый verify — HUD-признаки целевого
                #    глобального состояния (recover_closed; пустой verify
                #    без якоря ничего не подтверждает — UNKNOWN, страж 5.7:
                #    с якорем пустой набор истинен вакуумно, без якоря
                #    доверия к кадру нет);
                # 2) settle-цикл — клик переключил ВИД окна (старт/отмена
                #    лавки): якорь нового вида появится через анимацию,
                #    ждём свежий кадр+якорь коротким опросом (живой кейс
                #    2026-11-12: cancel_sell останавливал лавку, но уходил
                #    в unconfirmed, не дождавшись формы).
                if allow_anchorless_verify and verify.checks:
                    report = run_checks(verify.checks, verify_frame, None)
                    if report.passed:
                        return ClickResult(element_id, success=True,
                                           attempts=attempt)

                if not allow_anchorless_verify:
                    # Необратимое: без якоря на verify-кадре подтверждений
                    # нет — ни якорно-независимых, ни settle (пустой verify
                    # со свежим якорем истинен вакуумно — settle на него
                    # не должен опираться, страж 5.7).
                    failures.append(f"попытка {attempt}: якорь потерян перед verify")
                    effect_unconfirmed = True  # клик ушёл, подтвердить нельзя
                    continue

                def _settled() -> bool:
                    fresh = tx.refresh()
                    anchor = find_anchor(fresh)
                    if anchor is None:
                        return False
                    return run_checks(verify.checks, fresh, anchor).passed

                settled = wait_until(
                    _settled, timeout=_VERIFY_SETTLE_TIMEOUT, config=config,
                    cancelled=cancelled, clock=clock, sleep=sleep)
                if settled.cancelled:
                    return ClickResult(element_id, success=False,
                                       attempts=attempt, cancelled=True)
                if settled.success:
                    return ClickResult(element_id, success=True,
                                       attempts=attempt)
                failures.append(f"попытка {attempt}: якорь потерян перед verify")
                effect_unconfirmed = True  # клик ушёл, подтвердить нельзя
                continue
            report = run_checks(verify.checks, verify_frame, verify_anchor)
            if report.passed:
                return ClickResult(element_id, success=True, attempts=attempt)
            failures.append(
                f"попытка {attempt}: verify не подтверждён "
                f"({'; '.join(report.failures) or 'признаки не совпали'})")

        # Исчерпание retry: ПОВТОРНЫЙ ПРОЛОГ перед on_fail — за цикл окно
        # могло деградировать (якорь потерян на всех попытках = окно
        # перекрыто/исчезло), а click_free_area кликает в область экрана:
        # без activate OK + статуса found это слепой клик (инвариант 9),
        # on_fail пропускаем и сразу отдаём сигнал recover с причиной.
        reactivated = activate()
        if reactivated.cancelled:
            return ClickResult(element_id, success=False,
                               attempts=max_attempts, cancelled=True)
        status = window_status(tx.refresh())
        if status.status != STATUS_FOUND:
            failures.append(
                f"on_fail пропущен: статус окна {status.status!r} "
                f"({status.reason or status.content_detail}) — слепые "
                f"HID-шаги запрещены")
            return ClickResult(element_id, success=False, attempts=max_attempts,
                               unconfirmed=effect_unconfirmed,
                               on_fail_executed=False, failures=tuple(failures))
        _run_on_fail(on_fail, hid=hid, cursor_pos=cursor_pos,
                     free_area_point=free_area_point, config=config, rng=rng)
        return ClickResult(element_id, success=False, attempts=max_attempts,
                           unconfirmed=effect_unconfirmed,
                           on_fail_executed=bool(on_fail),
                           failures=tuple(failures))
