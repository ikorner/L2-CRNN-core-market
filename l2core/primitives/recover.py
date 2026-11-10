"""Общая recover-цепочка (задача 5.8, spec action-primitives:
«Recover-цепочка с пассивными шагами и эскалацией», «Штатный FAILED при
невыполненных предусловиях открытия»; design.md решение 6/10; recover —
исполнитель сигнала needs_recover примитивов 5.1–5.6).

recover — оркестратор ПОВЕРХ существующих примитивов (5.3 не переписывается;
пассивные шаги исполняются общей механикой on_fail click.py). Цепочка:

    ESC → клик в свободную область → закрыть окно (если открыто не то) →
    переоткрыть рецептом навигации (recipe_runner) →
    перекалибровка якоря → эскалация (пауза + событие стратегии)

УСТРОЙСТВО: recover — не ритуал, а путь к цели. Цель — инъекция
`goal_reached(frame) -> bool` (wiring-слой решает по детекторам 3.4/3.5,
что «состояние восстановлено»). Цель проверяется на входе и ПОСЛЕ КАЖДОГО
шага (wait_until по свежим кадрам, таймаут timing.wait_default_timeout):
достигнута — цепочка завершается досрочно (например, ESC снял модалку —
дальше ничего не делаем). Шаги, неприменимые к ситуации (окон нет, открыто
уже целевое, HUD не локализован), — пропуск с диагностикой в шагах.
Каждый шаг однократен: повторы — это сама цепочка (recover — эскалация
retry → on_fail[]; внутри шага retry не разворачивается, иначе это тот же
примитив с другой обёрткой).

ПАССИВНЫЕ ШАГИ ПЕРВЫМИ (design.md решение 10): ESC и клик в свободную
область — пассивные HID-действия (тот же набор, что on_fail[] переходов
модели). Автоматическое перетаскивание окон ЗАПРЕЩЕНО (спека: MUST);
клик по заголовку торгового окна для подъёма допустим — в модели есть
элемент-заголовок (jakor_sell), но подъём окна — ответственность модели/
сценария, recover его не исполняет (окно не перекрывает само себя: если
окно недоступно из-за перекрытия чужим окном, это ловит статус-детектор
3.5 → эскалация).

ПРОЛОГ (инвариант 9): вся цепочка HID-действий гейтится одной активацией
окна игры (5.2) на входе и повторной перед каждым активным шагом (кнопка
закрытия, HUD): активация невозможна — любой HID-шаг был бы слепым
(ESC ушло бы в чужое окно) → осмысленная фиксация + эскалация, HID не
трогается. Пролог 5.3–5.5 (activate + статус found ЦЕЛЕВОГО окна) здесь
НЕ применяется осознанно: recover вызывается при деградированном состоянии,
целевое окно законно отсутствует; «found окна игры» — единственный честный
предусловный барьер. Статус целевого окна (3.5) используется для ФИКСАЦИИ
состояния при эскалации, не как барьер.

ШАГ «ЗАКРЫТЬ ОКНО» (если открыто не то): открытое окно определяет инъекция
`current_window(frame) -> Optional[str]` (wiring, на детекторах 3.4).
Открыто целевое или ничего — пропуск. Открыто другое: точку кнопки закрытия
даёт инъекция `close_point(window_id, frame)` (wiring считает её от якоря
того окна по модели: market — main_market_close_button, shop_sell —
close_sell_button); точки нет (кнопки в модели нет, как у shop_buy) —
документированный fallback: ESC как закрытие верхнего окна. Verify шага:
окно закрылось устойчиво — wait_until по свежим кадрам: current_window !=
закрываемое подряд stability_frames кадров.

ШАГ «ПЕРЕОТКРЫТЬ РЕЦЕПТОМ НАВИГАЦИИ» (menu-quickslot-navigation): окно
открывает инъекция recipe_runner(recipe_id) — исполнитель рецептов
(l2core/primitives/navigation.py; id рецепта — соглашение open_<window>).
Рецепт сам верифицирует открытие (verify финального шага — якорь окна),
после успеха — повторная проверка цели (goal_reached по свежим кадрам,
таймаут из конфига). Спека «Лавку нельзя создать в текущей зоне»:
неоткрытие за таймаут → штатный FAILED (успех=False, unconfirmed=False
→ ActionOutcome.FAILED): фиксация состояния (инъекция capture_state —
снимок/перехватчик модалок 3.6; статус окна — в failures) + событие
escalation; ПОВТОРНЫХ попыток открытия нет (MUST) — предусловия (зона
торговли, живой персонаж) за это время не меняются, повтор был бы
ритуалом. recipe_runner не задан — шаг пропускается с диагностикой
(честная деградация, как раньше с hud_locator=None).

ШАГ «ПЕРЕКАЛИБРОВКА ЯКОРЯ»: кэша позиции в архитектуре нет — якорь ищется
на каждом кадре свежий (инвариант 3), поэтому «перекалибровка» — это
подтверждение СТАБИЛЬНОСТИ: wait_until по свежим кадрам, find_anchor
целевого окна подряд stability_frames раз (подъём целевого окна курсором
за заголовок — допустим, но только по решению стратегии, не ядром).
Стабилен — финальная проверка цели; нестабилен — эскалация.

ЭСКАЛАЦИЯ (пауза + событие): фиксация состояния (статус 3.5 + capture_state)
→ событие escalation (callback-инъекция on_escalation, push-форма; полный
диспетчер событий — 7.3, здесь точка входа) → пауза: wait_until по инъекции
`resume() -> bool` — блокировка до явного продолжения стратегией
(pause_timeout=None — бесконечно) или таймаута. Событие до паузы: стратегия
узнаёт об эскалации до того, как ей же предстоит продолжить работу.

RELEASE_ALL (залипшая кнопка из 5.6): вызывающий слой передаёт результат
примитива-причины (`cause`); при needs_recover (не success, не cancelled)
recover делает best-effort hid.release_all() ПЕРВЫМ HID-действием — до
любого шага: залипшая кнопка исказила бы любой последующий клик (drag 5.6
оставляет её залипшей при сбое release). Место — до пассивных шагов, а не
в эскалации: ESC/свободный клик сами страдают от залипшей кнопки.
HidError release_all глушится с фиксацией в failures (HID и так в сбое —
recover продолжает пассивные шаги; kill switch 8.1 делает свой release_all).

ОТМЕНА: cancelled (kill switch 8.1, инъекция) проверяется на входе, перед
шагами и прокидывается во все wait_until: отмена — результат с
cancelled=True, событие escalation НЕ отправляется (эскалация уже
определённого состояния, не отмены); release_all при отмене — ответственность
владельца kill switch, не recover.

ДЕТЕРМИНИЗМ (инвариант 7): вся недетерминированность — за инъекциями
(hid, экран, активация, статус, якорь, goal/current_window/close_point/
recipe_runner, resume, часы); rng — для hold_ms кликов (профиль 5.3).
Прямых вызовов sleep нет (инвариант 1): паузы и ожидания — только
wait_until по инъецированным часам.

ГРАНИЦЫ: никакого event dispatcher (7.3), API (7.1), kill switch (8.1) —
YAGNI; исполнение recover сценариями — 6.4. Поле unconfirmed результата
всегда False: recover сам не производит необратимый эффект, FAILED здесь
означает «восстановить не удалось», а не «эффект не подтверждён».
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, replace
from typing import Callable, Optional, Tuple

import numpy as np

from l2core.adapters.hid import BTN_LEFT, HidError, KEY_ESC
from l2core.adapters.screen import Screen
from l2core.config import CoreConfig
from l2core.detector.anchor import Anchor
from l2core.detector.status import WindowStatus
from l2core.detector.transaction import FrameTransaction
from l2core.primitives.navigation import RecipeResult
from l2core.primitives.activate import ActivateGameError, ActivateResult
from l2core.primitives.click import (
    JitterRng,
    _hold_ms,
    _move_to,
    _run_on_fail,
)
from l2core.primitives.outcome import PrimitiveResult
from l2core.primitives.wait import wait_until

__all__ = [
    "STEP_CLOSE_WINDOW",
    "STEP_ESCALATION",
    "STEP_FREE_CLICK",
    "STEP_RECALIBRATE",
    "STEP_REOPEN",
    "STEP_ESC",
    "STATUS_DONE",
    "STATUS_FAILED",
    "STATUS_GOAL_REACHED",
    "STATUS_SKIPPED",
    "RecoverError",
    "RecoverResult",
    "RecoverStep",
    "recover",
]

# Шаги цепочки (порядок — контракт спеки, не переупорядочивать).
STEP_ESC = "press_esc"
STEP_FREE_CLICK = "click_free_area"
STEP_CLOSE_WINDOW = "close_window"
STEP_REOPEN = "reopen_via_hud"
STEP_RECALIBRATE = "recalibrate_anchor"
STEP_ESCALATION = "escalation"

# Итоги шагов: done — исполнен (цель пока не достигнута); skipped — не
# применим к ситуации (причина в detail); failed — исполнен, verify не
# прошёл; goal_reached — после шага цель достигнута, цепочка завершена.
STATUS_DONE = "done"
STATUS_SKIPPED = "skipped"
STATUS_FAILED = "failed"
STATUS_GOAL_REACHED = "goal_reached"


@dataclass(frozen=True)
class RecoverStep:
    """Запись одного шага цепочки: какой шаг чем закончился (диагностика)."""

    name: str
    status: str  # STATUS_*
    detail: str = ""


@dataclass(frozen=True)
class RecoverResult:
    """Итог recover; трёхзначная обёртка OK/FAILED — адаптер outcome.py (5.7)
    поверх success/unconfirmed (cancelled — домен kill switch, 8.1).

    success — цель достигнута (цепочка завершена досрочно или полным
    проходом); escalated — цепочка исчерпана/активация невозможна/окно не
    открылось: событие escalation отправлено, пауза отработана; steps —
    диагностика по шагам (порядок и итоги); failures — причины провалов;
    escalation_detail — почему ушли в эскалацию.
    """

    target_window: str
    success: bool
    cancelled: bool = False
    escalated: bool = False
    unconfirmed: bool = False  # всегда False: см. докстринг модуля
    steps: Tuple[RecoverStep, ...] = ()
    failures: Tuple[str, ...] = ()
    escalation_detail: str = ""

    @property
    def needs_recover(self) -> bool:
        """Recover сам — исполнитель recover: неуспех без отмены."""
        return not self.success and not self.cancelled


class RecoverError(Exception):
    """Ошибка исполнения recover (модель/аргументы), не эскалация рантайма."""

    def __init__(self, detail: str):
        super().__init__(f"recover: {detail}")
        self.detail = detail


def recover(
    target_window: str,
    *,
    recipe_runner: Optional[Callable[[str], "RecipeResult"]] = None,
    screen: Screen,
    hid,
    cursor_pos: Callable[[], Tuple[int, int]],
    activate: Callable[[], ActivateResult],
    window_status: Callable[[np.ndarray], WindowStatus],
    find_anchor: Callable[[np.ndarray], Optional[Anchor]],
    goal_reached: Callable[[np.ndarray], bool],
    config: CoreConfig,
    current_window: Optional[Callable[[np.ndarray], Optional[str]]] = None,
    close_point: Optional[Callable[[str, np.ndarray],
                                   Optional[Tuple[int, int]]]] = None,
    on_escalation: Optional[Callable[[RecoverResult], None]] = None,
    resume: Optional[Callable[[], bool]] = None,
    pause_timeout: Optional[float] = None,
    capture_state: Optional[Callable[[], None]] = None,
    stability_frames: int = 2,
    cause: Optional[PrimitiveResult] = None,
    free_area_point: Optional[Tuple[int, int]] = None,
    cancelled: Optional[Callable[[], bool]] = None,
    rng: Optional[JitterRng] = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> RecoverResult:
    """Общий recover: цепочка пассивных шагов → переоткрытие рецептом
    навигации → перекалибровка якоря → эскалация. Контракт — в
    докстринге модуля.

    recipe_runner — исполнитель рецептов открытия (id по соглашению
    open_<target_window>); None — шаг переоткрытия пропускается с
    диагностикой (сборка без навигационной модели).
    goal_reached — «состояние восстановлено» (wiring на детекторах 3.4/3.5).
    current_window/close_point — какое окно открыто и где его кнопка закрытия.
    on_escalation — push-callback события escalation (7.3 не здесь).
    resume/pause_timeout — пауза эскалации до явного продолжения или таймаута.
    cause — результат примитива-причины (needs_recover → release_all).
    """
    if stability_frames < 1:
        raise RecoverError(
            f"stability_frames должно быть >= 1, получено {stability_frames}")
    rng = rng or random.Random()
    steps: list[RecoverStep] = []
    failures: list = []

    def _cancelled() -> bool:
        return cancelled is not None and cancelled()

    def _cancelled_result() -> RecoverResult:
        return RecoverResult(target_window, success=False, cancelled=True,
                             steps=tuple(steps), failures=tuple(failures))

    def _success_result() -> RecoverResult:
        return RecoverResult(target_window, success=True, steps=tuple(steps),
                             failures=tuple(failures))

    def _activate_guard() -> str:
        """Пролог активного HID-шага: 'ok' | 'cancelled' | 'escalate'.

        activate_game невозможен — любой HID-шаг был бы слепым (инвариант 9):
        фиксация причины, вызывающий слой получает эскалацию.
        """
        if _cancelled():
            return "cancelled"
        try:
            activated = activate()
        except ActivateGameError as e:
            failures.append(f"activate_game: {e.reason}: {e.detail}")
            return "escalate"
        if activated.cancelled:
            return "cancelled"
        return "ok"

    with FrameTransaction(screen) as tx:
        def _goal_wait():
            # Verify шага: цель по СВЕЖИМ кадрам до таймаута из конфига.
            return wait_until(
                lambda: goal_reached(tx.refresh()),
                timeout=config.timing.wait_default_timeout,
                interval=config.timing.verify_interval,
                cancelled=cancelled, clock=clock, sleep=sleep)

        def _wait_stable(probe):
            # Устойчивость: probe(fresh frame) truthy подряд stability_frames
            # раз (закрытие окна, стабильность якоря). Свежий кадр на каждый
            # poll — кэша позиции нет (инвариант 3).
            counter = [0]

            def _pred() -> bool:
                if probe(tx.refresh()):
                    counter[0] += 1
                else:
                    counter[0] = 0
                return counter[0] >= stability_frames

            return wait_until(
                _pred,
                timeout=config.timing.wait_default_timeout,
                interval=config.timing.verify_interval,
                cancelled=cancelled, clock=clock, sleep=sleep)

        def _escalate(detail: str) -> RecoverResult:
            # Фиксация состояния (включая возможную модалку с причиной):
            # статус окна — в failures, снимок/перехватчик — capture_state.
            status = window_status(tx.refresh())
            failures.append(
                f"фиксация при эскалации: статус окна {status.status!r} "
                f"({status.reason or status.content_detail})")
            if capture_state is not None:
                try:
                    capture_state()
                except Exception as e:  # фиксация — best-effort
                    failures.append(f"capture_state: {e}")
            steps.append(RecoverStep(STEP_ESCALATION, STATUS_DONE, detail))
            result = RecoverResult(
                target_window, success=False, escalated=True,
                steps=tuple(steps), failures=tuple(failures),
                escalation_detail=detail)
            if on_escalation is not None:
                on_escalation(result)
            # Пауза: блокировка до явного продолжения стратегией
            # (pause_timeout=None — бесконечно) или таймаута. Прямого sleep
            # нет (инвариант 1) — только wait_until по инъецированным часам.
            paused = wait_until(
                resume or (lambda: False),
                timeout=pause_timeout if pause_timeout is not None else math.inf,
                interval=config.timing.verify_interval,
                cancelled=cancelled, clock=clock, sleep=sleep)
            if paused.cancelled:
                return replace(result, cancelled=True)
            return result

        def _passive_step(name: str, on_fail_steps: Tuple[str, ...]) -> str:
            """Пассивный HID-шаг (ESC / свободный клик): действие → verify цели.
            Возврат: 'goal' | 'continue' | 'cancelled'."""
            if _cancelled():
                return "cancelled"
            try:
                _run_on_fail(on_fail_steps, hid=hid, cursor_pos=cursor_pos,
                             free_area_point=free_area_point, config=config,
                             rng=rng)
            except HidError as e:
                failures.append(f"{name}: HID-сбой ({e})")
                steps.append(RecoverStep(name, STATUS_FAILED, f"HID-сбой: {e}"))
                return "continue"
            waited = _goal_wait()
            if waited.cancelled:
                return "cancelled"
            if waited.success:
                steps.append(RecoverStep(name, STATUS_GOAL_REACHED,
                                         "цель достигнута"))
                return "goal"
            steps.append(RecoverStep(name, STATUS_DONE, "цель не достигнута"))
            return "continue"

        # --- Вход: recover — не ритуал; цель уже достигнута — работать нечего.
        if _cancelled():
            return _cancelled_result()
        if goal_reached(tx.frame):
            return _success_result()

        # --- RELEASE_ALL (залипшая кнопка из 5.6): первое HID-действие,
        # до любого шага; best-effort — HID и так может быть в сбое.
        if (cause is not None and not cause.cancelled
                and not cause.success):
            try:
                hid.release_all()
            except HidError as e:
                failures.append(f"release_all: сбой ({e}) — продолжаем")

        # --- ПРОЛОГ: активация окна игры гейтит ВСЮ цепочку HID (инвариант 9).
        gate = _activate_guard()
        if gate == "cancelled":
            return _cancelled_result()
        if gate == "escalate":
            return _escalate("активация окна игры невозможна — слепые "
                             "HID-шаги запрещены")

        # --- ШАГ 1: ESC (пассивный).
        outcome = _passive_step(STEP_ESC, ("press_esc",))
        if outcome == "cancelled":
            return _cancelled_result()
        if outcome == "goal":
            return _success_result()

        # --- ШАГ 2: клик в свободную область клиента (пассивный).
        outcome = _passive_step(STEP_FREE_CLICK, ("click_free_area",))
        if outcome == "cancelled":
            return _cancelled_result()
        if outcome == "goal":
            return _success_result()

        # --- ШАГ 3: закрыть окно (если открыто не то).
        if current_window is None:
            steps.append(RecoverStep(STEP_CLOSE_WINDOW, STATUS_SKIPPED,
                                     "current_window не задан — открытое окно "
                                     "не определяется"))
        else:
            opened = current_window(tx.refresh())
            if opened is None:
                steps.append(RecoverStep(STEP_CLOSE_WINDOW, STATUS_SKIPPED,
                                         "окна торговли не открыты — закрывать "
                                         "нечего"))
            elif opened == target_window:
                steps.append(RecoverStep(
                    STEP_CLOSE_WINDOW, STATUS_SKIPPED,
                    f"открыто целевое окно {target_window!r} — закрывать нечего"))
            else:
                gate = _activate_guard()
                if gate == "cancelled":
                    return _cancelled_result()
                if gate == "escalate":
                    return _escalate("активация окна игры невозможна "
                                     "(шаг закрытия окна)")
                point = (close_point(opened, tx.refresh())
                         if close_point is not None else None)
                closed_detail = ""
                try:
                    if point is None:
                        # Fallback: кнопки закрытия в модели нет (например,
                        # shop_buy) — закрытие верхнего окна через ESC.
                        hid.key_press(KEY_ESC)
                        closed_detail = "ESC как закрытие (кнопки закрытия в " \
                                        "модели нет)"
                    else:
                        _move_to(hid, cursor_pos, point)
                        hid.click(BTN_LEFT, 1, _hold_ms(config, rng))
                        closed_detail = f"клик по кнопке закрытия окна {opened!r}"
                except HidError as e:
                    failures.append(f"{STEP_CLOSE_WINDOW}: HID-сбой ({e})")
                    steps.append(RecoverStep(STEP_CLOSE_WINDOW, STATUS_FAILED,
                                             f"HID-сбой: {e}"))
                else:
                    closed = _wait_stable(
                        lambda f: current_window(f) != opened)
                    if closed.cancelled:
                        return _cancelled_result()
                    if closed.success:
                        waited = _goal_wait()
                        if waited.cancelled:
                            return _cancelled_result()
                        if waited.success:
                            steps.append(RecoverStep(
                                STEP_CLOSE_WINDOW, STATUS_GOAL_REACHED,
                                "цель достигнута"))
                            return _success_result()
                        steps.append(RecoverStep(STEP_CLOSE_WINDOW, STATUS_DONE,
                                                 closed_detail))
                    else:
                        steps.append(RecoverStep(
                            STEP_CLOSE_WINDOW, STATUS_FAILED,
                            f"окно {opened!r} не закрылось за таймаут"))

        # --- ШАГ 4: переоткрыть рецептом навигации (menu-quickslot-
        # navigation; рецепт сам верифицирует открытие, дальше — проверка
        # цели по свежим кадрам).
        if recipe_runner is None:
            steps.append(RecoverStep(STEP_REOPEN, STATUS_SKIPPED,
                                     "recipe_runner не задан — переоткрытие "
                                     "невозможно (сборка без навигационной "
                                     "модели)"))
        else:
            gate = _activate_guard()
            if gate == "cancelled":
                return _cancelled_result()
            if gate == "escalate":
                return _escalate("активация окна игры невозможна "
                                 "(шаг переоткрытия)")
            recipe_id = f"open_{target_window}"
            opened_result = recipe_runner(recipe_id)
            if opened_result.cancelled:
                return _cancelled_result()
            if not opened_result.success:
                # «Лавку нельзя создать в текущей зоне»: неоткрытие →
                # штатный FAILED + фиксация + escalation; повторных
                # попыток открытия нет (MUST, spec).
                reason = "; ".join(opened_result.failures) or (
                    "verify открытия не сошёлся")
                failures.append(
                    f"{STEP_REOPEN}: рецепт {recipe_id!r} не открыл окно "
                    f"{target_window!r}: {reason} — предусловия открытия "
                    f"не выполнены (зона торговли, живой персонаж?)")
                steps.append(RecoverStep(
                    STEP_REOPEN, STATUS_FAILED,
                    f"окно не открыто рецептом {recipe_id!r}"))
                return _escalate(
                    f"неоткрытие окна {target_window!r} рецептом "
                    f"{recipe_id!r}")
            opened = _goal_wait()
            if opened.cancelled:
                return _cancelled_result()
            if opened.success:
                steps.append(RecoverStep(STEP_REOPEN,
                                         STATUS_GOAL_REACHED,
                                         "окно открыто (verify)"))
                return _success_result()
            failures.append(
                f"{STEP_REOPEN}: окно {target_window!r} открыто рецептом "
                f"{recipe_id!r}, но цель не достигнута за таймаут "
                f"{config.timing.wait_default_timeout} с")
            steps.append(RecoverStep(
                STEP_REOPEN, STATUS_FAILED,
                f"цель не достигнута за таймаут "
                f"{config.timing.wait_default_timeout} с"))
            return _escalate(
                f"цель не достигнута после открытия окна "
                f"{target_window!r} рецептом {recipe_id!r}")

        # --- ШАГ 5: перекалибровка якоря (стабильность found подряд K кадров).
        stable = _wait_stable(lambda f: find_anchor(f) is not None)
        if stable.cancelled:
            return _cancelled_result()
        if stable.success:
            waited = _goal_wait()
            if waited.cancelled:
                return _cancelled_result()
            if waited.success:
                steps.append(RecoverStep(STEP_RECALIBRATE,
                                         STATUS_GOAL_REACHED,
                                         "цель достигнута"))
                return _success_result()
            steps.append(RecoverStep(STEP_RECALIBRATE, STATUS_DONE,
                                     f"якорь стабилен подряд "
                                     f"{stability_frames} кадров, цель не "
                                     f"достигнута"))
        else:
            steps.append(RecoverStep(STEP_RECALIBRATE, STATUS_FAILED,
                                     "якорь нестабилен на свежих кадрах"))

        # --- ШАГ 6: эскалация (пауза + событие стратегии).
        return _escalate("цепочка recover исчерпана — цель не достигнута")
