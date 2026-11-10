"""Синхронный блокирующий API для стратегий (задача 7.1, spec
core-api-events: «Синхронный блокирующий API для стратегий»; события/потоки
— 7.2/7.3: фасад подключает EventAPI через параметр events (bot.event_api)).

L2CoreAPI — ТОНКИЙ ФАСАД поверх существующих сценариев 6.1–6.4 и примитивов
5.x: каждый метод — композиция с провайдерами сборки ApiWiring
(l2core/api/wiring.py), механика кликов/ввода/скролла/drag/recover здесь не
дублируется. Карта методов:

- open_market/close_market, open_shop_buy/open_shop_sell — навигация;
- set_tab/set_filter, search, scroll_list — управление обзором;
- read_rows — чтение видимых строк (RowsResult со snapshot_id, 6.1);
- setup_sell/start_sell/cancel_sell — лавка продажи (6.2, страж 5.7);
- read_shop_state — разовое чтение показателей лавки (паттерн монитора 6.3
  БЕЗ baseline: чтение, не детекция продажи);
- get_state — текущее (window, state, params, status);
- recover — сценарная точка входа 6.4 по сигналу needs_recover (публикует
  события recovered/escalation по итогу цепочки, журнал 7.3);
- health — агрегат диагностики (окно игры/окно торговли/OCR/HID);
- on/wait_event/get_events — события двух форм и журнал (7.3, R14:
  push — dispatcher-поток, pull — строго из потока стратегии).

ОТКРЫТИЕ ОКОН (open_*, menu-quickslot-navigation): исполнители рецептов
навигационной модели (l2core.model.navigation, l2core.primitives.
navigation): обзор открывается через игровое «Меню» (клавиша X / alternate
гамбургер -> иконка обзора в меню), лавки — кликом по иконкам quick-slot;
«Меню» перед кликом нормализуется (when/skip_if: X — toggle, слепое
нажатие опасно). click_element 5.3 сюда НЕ подходит — его пролог требует
якорь окна на кадре, а при закрытом окне якоря нет. Verify рецепта —
навигационные признаки (меню, якоря open-окон через многоопорный якорь
3.3); open_shop_* принимают verify — override последнего шага. Исходы
(инвариант 4): OK / FAILED (эффект отсутствует или опровергнут) / UNKNOWN
(HID-сбой или эффект не подтверждён — unconfirmed). Сборка без
навигационной модели — ApiError (fail fast).

ПОТОК-ВЛАДЕЛЕЦ (spec: «стратегия MUST вызывать блокирующий API только из
своего потока»): владелец фиксируется при создании (поток-конструктор или
явный owner_thread); каждый публичный метод проверяет
threading.current_thread() — чужой поток → RuntimeError (дискиплина
контракта, enforcement в точке вызова). Executor/observer/dispatcher —
модель потоков 7.2, здесь нет.

ОТМЕНА: методы принимают cancelled (kill switch 8.1, инъекция; по умолчанию
— wiring.cancelled): отмена до действия — результат с cancelled=True и
чистым HID; внутри ожиданий — через wait_until примитивов. Отменённый
результат не адаптируется в ActionOutcome (outcome=None) — to_outcome на
отмене fail fast (5.7).

ГРАНИЦЫ (R4, KNOWN_GAPS): search требует
search_region (ocr_regions обзора не размечены — 6.1); setup_sell требует
price_region и drag_verify (6.2); read_shop_state требует регион показателей
(6.3). Всё это — ApiError (fail fast: клики/ввод без verify запрещены,
инвариант 2), не молчаливый пропуск.

JSONL-ЛОГ ДЕЙСТВИЙ (8.3, spec replay-logging): каждый действие-метод
(навигация, табы/фильтры, поиск, скролл, setup_sell/start_sell/cancel_sell,
recover — НЕ чтение: read_rows/read_shop_state/get_state/health) обёрнут
_log_action: при wiring.action_logger запись с действием, verify-результатом
и кадрами до/после (граница вызова API, не отдельной HID-команды — отчёт
8.3). Кадры сохраняются только для ключевых записей: не-OK исходов и
необратимых (start_sell) — политика log.policy; ротация и лимиты — в
ActionLogger (l2core/api/actionlog.py).
"""
from __future__ import annotations

import dataclasses
import hashlib
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Mapping, Optional, Tuple

import numpy as np

from l2core.api.events import EVENT_ESCALATION, EVENT_RECOVERED
from l2core.detector.anchor import Anchor
from l2core.detector.status import STATUS_NOT_FOUND
from l2core.model import OcrRegionSpec, Rect, UiModel
from l2core.model.reading import OcrReadError, read_confident
from l2core.model.rows import RowsResult, read_rows
from l2core.primitives.click import (
    ClickResult,
    VerifySpec,
    click_element,
)
from l2core.primitives.enter import EnterResult, enter_text
from l2core.primitives.navigation import RecipeResult, run_recipe
from l2core.primitives.outcome import ActionOutcome, IrreversibleGuard, to_outcome
from l2core.primitives.scroll import DIR_DOWN, ScrollResult, scroll_list
from l2core.scenarios.recover import RecoverCause, RecoverResult, scenario_recover
from l2core.scenarios.setup_sell import (
    CancelSellResult,
    SellItem,
    SetupSellResult,
    StartSellResult,
    cancel_sell,
    setup_sell,
    start_sell,
)

if TYPE_CHECKING:  # циклическая связь: wiring.py собирает ApiWiring поверх core
    from l2core.api.events import Event, EventAPI, EventSubscription
    from l2core.api.wiring import ApiWiring

__all__ = [
    "ApiError",
    "CoreState",
    "HealthResult",
    "L2CoreAPI",
    "NavigationResult",
    "ShopStateResult",
]

# Грамматики числовых показателей лавки (как в мониторе 6.3): сравнивать
# по тексту нельзя — только по разобранным значениям.
_NUMERIC_GRAMMARS = frozenset({"digits", "price"})

# OCR-регион поля поиска обзора: в модели market.yaml не размечен
# (KNOWN_GAPS) — подшивается в копию модели из wiring.search_region (как
# search_region в 6.1). Id фиксирован контрактом API.
SEARCH_REGION_ID = "search_field"


class ApiError(Exception):
    """Ошибка контракта вызова API (нет verify-источника/региона, отмена
    мгновенного вызова) — не эскалация рантайма."""

    def __init__(self, detail: str):
        super().__init__(f"L2CoreAPI: {detail}")
        self.detail = detail


@dataclass(frozen=True)
class NavigationResult:
    """Итог навигационного вызова (open/close/tab/filter).

    outcome — трёхзначный итог адаптера 5.7: OK — сигнатуры целевого
    состояния подтверждены; FAILED — verify выполнялся и провалился (эффект
    опровергнут/не применён); UNKNOWN — HID-эффект ушёл, подтвердить
    невозможно (потенциально необратимое повторять нельзя до явного
    confirm_after_unknown — страж guard, 5.7); None — отмена (kill switch
    8.1), смотри cancelled. needs_recover — сигнал общего recover (6.4);
    исполнение recover — решение вызывающего слоя, метод recover().
    """

    success: bool
    outcome: Optional[ActionOutcome]
    cancelled: bool = False
    unconfirmed: bool = False
    on_fail_executed: bool = False
    failures: Tuple[str, ...] = ()

    @property
    def needs_recover(self) -> bool:
        """Сигнал общего recover: неуспех без отмены."""
        return not self.success and not self.cancelled


@dataclass(frozen=True)
class ShopStateResult:
    """Разовое чтение показателей лавки продажи (без baseline монитора).

    sold_count/revenue — прочитанные значения (None — не прочитаны);
    degraded — чтение было с деградацией (якоря не было/регион не сошёлся);
    cancelled — вызов отменён до чтения кадра. Чтение HID не трогает вообще
    (паттерн монитора 6.3: наблюдатель, не исполнитель).
    """

    sold_count: Optional[int]
    revenue: Optional[int]
    snapshot_id: Optional[str]
    degraded: bool = False
    cancelled: bool = False
    failures: Tuple[str, ...] = ()


@dataclass(frozen=True)
class CoreState:
    """Текущее состояние ядра: открытое окно торговли + статус + состояние.

    window — открытое окно торговли (None — ни одного); status — статус окна
    3.5 (not_found, когда окна нет); state/params — подтверждённое
    параметризованное состояние детектора 3.4 (None — unknown); окно
    открыто, но detect_state не предоставлен сборкой — state=None с
    пояснением в detail.
    """

    window: Optional[str]
    status: str
    state: Optional[str]
    params: Mapping[str, object] = dataclasses.field(default_factory=dict)
    content_loaded: Optional[bool] = None
    detail: str = ""


@dataclass(frozen=True)
class HealthResult:
    """Агрегат живости компонентов (диагностика стратегии перед циклом).

    ok — окно игры доступно, OCR и HID живы; отсутствие открытого окна
    торговли НЕ нездоровье (обзор стратегия открывает сама) —
    trading_window=None с ok=True.
    """

    ok: bool
    game_window: bool
    game_detail: str
    trading_window: Optional[str]
    ocr: bool
    hid: bool


def _navigation(clicked: ClickResult) -> NavigationResult:
    """Адаптация ClickResult в NavigationResult (отмена — outcome=None:
    to_outcome на отменённом результате fail fast, 5.7)."""
    if clicked.cancelled:
        return NavigationResult(
            success=False, outcome=None, cancelled=True,
            unconfirmed=clicked.unconfirmed,
            on_fail_executed=clicked.on_fail_executed,
            failures=clicked.failures)
    return NavigationResult(
        success=clicked.success, outcome=to_outcome(clicked),
        unconfirmed=clicked.unconfirmed,
        on_fail_executed=clicked.on_fail_executed,
        failures=clicked.failures)


def _resolve_transition(model: UiModel, element_id: str):
    """Переход клика по элементу из модели (verify + on_fail перехода).

    verify/on_fail — части контракта перехода модели (задача 2.5); клик без
    verify запрещён (инвариант 2). Перехода нет — ApiError (fail fast):
    verify передаёт стратегия явно (set_tab/set_filter/open_shop_*).
    """
    for transition in model.transitions:
        if (transition.action.type == "click"
                and transition.action.element == element_id):
            return transition
    raise ApiError(
        f"в модели {model.path!r} нет перехода по элементу {element_id!r}: "
        f"verify/on_fail неизвестны — передайте verify явно (клики без "
        f"verify запрещены, инвариант 2)")


class L2CoreAPI:
    """Синхронный блокирующий фасад для модулей стратегий.

    Конструктор принимает готовую сборку провайдеров ApiWiring (живую —
    через create_api, тестовую — на фейках). Владелец — поток конструктора
    (или owner_thread); публичные методы вызываются только из потока
    владельца (spec core-api-events, enforcement проверки потока).
    """

    def __init__(self, wiring: "ApiWiring", *,
                 owner_thread: Optional[threading.Thread] = None,
                 events: Optional["EventAPI"] = None):
        self._w = wiring
        self._owner = owner_thread or threading.current_thread()
        self._events = events

    # ------------------------------------------------------------------
    # Инфраструктура контракта

    @property
    def guard(self) -> IrreversibleGuard:
        """Страж необратимых (5.7): UNKNOWN блокирует повтор до явного
        confirm_after_unknown стратегией (решение и ответственность — за
        стратегией, спека action-primitives)."""
        return self._w.guard

    @property
    def dry_run(self) -> bool:
        """Режим без HID (8.2, spec safety): True — экран читается и
        состояние определяется, а действия только логируются как
        намерения (DryRunHid), обращений к HID нет. Эффективный флаг
        сборки: create_api — параметр dry_run, иначе config.dry_run."""
        return self._w.dry_run

    @property
    def events(self) -> "EventAPI":
        """События двух форм — push (on) и pull (wait_event) — и журнал
        (задача 7.3, spec core-api-events; ruling R14). Сборка без API
        событий — ApiError (fail fast; движок 7.2 подключает всегда)."""
        if self._events is None:
            raise ApiError(
                "API событий не подключён (сборка без EventAPI): "
                "события доступны через L2CoreBot (bot.event_api) — "
                "создайте API через движок")
        return self._events

    def on(self, event_type: str, callback) -> "EventSubscription":
        """Push-подписка: callback доставляется dispatcher-потоком (R14);
        блокирующий API из callback вызывать нельзя (thread-owner)."""
        return self.events.on(event_type, callback)

    def wait_event(self, event_type: str,
                   timeout: Optional[float] = None) -> Optional["Event"]:
        """Pull-ожидание события: блокируется до события типа/таймаута;
        строго из потока стратегии (thread-owner, R14)."""
        self._check_thread()
        return self.events.wait_event(event_type, timeout)

    def get_events(self, since_seq: int = 0):
        """Журнал событий ядра: записи с seq >= since_seq (справочник
        payload'ов — docs/API.md)."""
        return self.events.get_events(since_seq=since_seq)

    def _check_thread(self) -> None:
        """Дисциплина вызова спеки: блокирующий API — только из потока
        стратегии-владельца; чужой поток → RuntimeError. Успешная проверка
        фиксирует активность executor'а (wiring.note_activity — линия
        таймаута неактивности kill switch 8.1)."""
        if threading.current_thread() is not self._owner:
            raise RuntimeError(
                f"L2CoreAPI: блокирующий API вызван из чужого потока "
                f"({threading.current_thread().name!r}; владелец — "
                f"{self._owner.name!r}): стратегия MUST вызывать API "
                f"только из своего потока (spec core-api-events)")
        note = self._w.note_activity
        if note is not None:
            note()

    def _cancelled(self, cancelled: Optional[Callable[[], bool]]
                   ) -> Optional[Callable[[], bool]]:
        """Эффективная инъекция отмены: аргумент метода побеждает,
        иначе — глобальная сборки wiring (kill switch 8.1)."""
        return cancelled if cancelled is not None else self._w.cancelled

    @staticmethod
    def _raise_if_cancelled(cancelled: Optional[Callable[[], bool]]) -> None:
        """Мгновенные вызовы (чтение) без результата-флага: отмена до
        начала работы — осмысленная ошибка, а не пустой результат."""
        if cancelled is not None and cancelled():
            raise ApiError("вызов отменён через cancelled (kill switch 8.1)")

    def _click_bundle(self, model: UiModel, window_id: str,
                      cancelled: Optional[Callable[[], bool]]) -> dict:
        """Единый набор провайдеров примитива клика 5.3, привязанный к окну
        (статус/якорь — этого окна; HUD-точки — через hud_locator)."""
        w = self._w
        return dict(
            model=model, run_checks=w.run_checks, screen=w.screen, hid=w.hid,
            cursor_pos=w.cursor_pos, activate=w.activate,
            window_status=lambda frame: w.window_status(frame, window_id),
            find_anchor=lambda frame: w.find_anchor(frame, window_id),
            config=w.config, hud_locator=w.hud_locator,
            free_area_point=w.free_area_point, cancelled=cancelled,
            rng=w.rng, clock=w.clock, sleep=w.sleep)

    def _log_action(self, kind: str, call: Callable, *,
                    target: Optional[str] = None,
                    args: Optional[dict] = None,
                    irreversible: bool = False):
        """Обёртка записи действия в JSONL-лог (8.3, wiring.action_logger).

        Кадры до/после — на границе ВЫЗОВА API: «до» = кадр на входе,
        «после» = кадр на выходе (метод API = одно действие ядра с одним
        verify-итогом; кадры вокруг отдельной HID-команды — граница, см.
        отчёт 8.3). Logger None — вызов без изменений. Запись идёт после
        исполнения: fail-fast ошибки контракта (ApiError до действия) в
        журнал не пишутся — это не действия. Поля результата собираются
        через getattr: результаты примитивов/сценариев 5.x/6.x разные,
        общие — success/cancelled.
        """
        alog = self._w.action_logger
        if alog is None:
            return call()
        before = np.asarray(self._w.screen.grab())
        result = call()
        after = np.asarray(self._w.screen.grab())
        alog.record(
            {"type": kind, "target": target, "args": args or {}},
            success=result.success,
            outcome=getattr(result, "outcome", None),
            unconfirmed=getattr(result, "unconfirmed", False),
            cancelled=result.cancelled,
            failures=tuple(getattr(result, "failures", ()) or ()),
            attempts=getattr(result, "attempts", None),
            irreversible=irreversible, before=before, after=after)
        return result

    # ------------------------------------------------------------------
    # Навигация (рецепты навигационной модели, menu-quickslot-navigation)

    def _navigation_from_recipe(self, r: RecipeResult) -> NavigationResult:
        """Адаптация RecipeResult в NavigationResult (отмена — outcome=None:
        to_outcome на отменённом результате fail fast, 5.7)."""
        if r.cancelled:
            return NavigationResult(
                success=False, outcome=None, cancelled=True,
                unconfirmed=r.unconfirmed,
                on_fail_executed=r.on_fail_executed, failures=r.failures)
        return NavigationResult(
            success=r.success, outcome=to_outcome(r),
            unconfirmed=r.unconfirmed,
            on_fail_executed=r.on_fail_executed, failures=r.failures)

    def _run_open_recipe(self, recipe_id: str,
                         verify: Optional[VerifySpec],
                         cancelled: Optional[Callable[[], bool]]
                         ) -> NavigationResult:
        """Исполнение рецепта открытия окна. verify — опциональный override
        стратегии: заменяет verify последнего шага рецепта (межоконные
        переходы формат модели окон не выражает — рецепт несёт свой
        verify; стратегия может подменить наблюдаемые признаки)."""
        w = self._w
        if w.navigation is None or w.nav_evaluate is None:
            raise ApiError(
                "навигационная модель не подключена (сборка без "
                "navigation.yaml): открытие окон невозможно")
        recipe = w.navigation.recipes.get(recipe_id)
        if recipe is None:
            raise ApiError(
                f"рецепт {recipe_id!r} отсутствует в навигационной модели "
                f"{w.navigation.path!r}")
        if verify is not None:
            recipe = dataclasses.replace(
                recipe,
                steps=recipe.steps[:-1] + (
                    dataclasses.replace(recipe.steps[-1],
                                        verify=verify.checks),))
        result = run_recipe(
            recipe.id, recipe=recipe,
            nav_model=w.navigation, evaluate=w.nav_evaluate,
            hud_locator=w.hud_locator, screen=w.screen, hid=w.hid,
            cursor_pos=w.cursor_pos, activate=w.activate, config=w.config,
            free_area_point=w.free_area_point, cancelled=cancelled,
            rng=w.rng, clock=w.clock, sleep=w.sleep)
        return self._navigation_from_recipe(result)

    def open_market(self, *, cancelled: Optional[Callable[[], bool]] = None
                    ) -> NavigationResult:
        """Открыть «Обзор лавок» рецептом навигации: X (skip_if, если
        «Меню» уже открыто) -> клик по иконке обзора в меню; alternate —
        гамбургер. По таймауту — FAILED/UNKNOWN (исполнитель рецептов)."""
        self._check_thread()
        c = self._cancelled(cancelled)

        def _call() -> NavigationResult:
            if c is not None and c():
                return NavigationResult(success=False, outcome=None,
                                        cancelled=True)
            return self._run_open_recipe("open_market", None, c)

        return self._log_action(
            "open_market", _call,
            target="market", args={"recipe": "open_market"})

    def close_market(self, *, cancelled: Optional[Callable[[], bool]] = None
                     ) -> NavigationResult:
        """Закрыть «Обзор лавок» (кнопка модели; verify/on_fail — из
        перехода модели)."""
        self._check_thread()
        c = self._cancelled(cancelled)
        transition = _resolve_transition(self._w.market,
                                         "main_market_close_button")
        return self._log_action(
            "close_market",
            lambda: _navigation(click_element(
                "main_market_close_button",
                verify=VerifySpec.from_transition(transition),
                on_fail=transition.on_fail,
                **self._click_bundle(self._w.market, "market", c))),
            target="market", args={"element": "main_market_close_button"})

    def open_shop_buy(self, *, verify: Optional[VerifySpec] = None,
                      cancelled: Optional[Callable[[], bool]] = None
                      ) -> NavigationResult:
        """Открыть лавку выкупа рецептом quick-slot (нормализация «Меню»:
        если открыто — закрыть через X; verify из рецепта, параметр verify
        — override последнего шага)."""
        self._check_thread()
        c = self._cancelled(cancelled)

        def _call() -> NavigationResult:
            if c is not None and c():
                return NavigationResult(success=False, outcome=None,
                                        cancelled=True)
            return self._run_open_recipe("open_shop_buy", verify, c)

        return self._log_action(
            "open_shop_buy", _call,
            target="shop_buy", args={"recipe": "open_shop_buy"})

    def open_shop_sell(self, *, verify: Optional[VerifySpec] = None,
                       cancelled: Optional[Callable[[], bool]] = None
                       ) -> NavigationResult:
        """Открыть лавку продажи рецептом quick-slot (см. open_shop_buy)."""
        self._check_thread()
        c = self._cancelled(cancelled)

        def _call() -> NavigationResult:
            if c is not None and c():
                return NavigationResult(success=False, outcome=None,
                                        cancelled=True)
            return self._run_open_recipe("open_shop_sell", verify, c)

        return self._log_action(
            "open_shop_sell", _call,
            target="shop_sell", args={"recipe": "open_shop_sell"})

    # ------------------------------------------------------------------
    # Управление обзором

    def set_tab(self, element_id: str, *,
                verify: Optional[VerifySpec] = None,
                on_fail: Tuple[str, ...] = (),
                variant: Optional[str] = None,
                cancelled: Optional[Callable[[], bool]] = None
                ) -> NavigationResult:
        """Клик по табу обзора. verify/on_fail — из перехода модели; если
        перехода нет (табы частично не размечены — KNOWN_GAPS) — verify
        обязателен явно."""
        self._check_thread()
        c = self._cancelled(cancelled)
        model = self._w.market
        try:
            transition = _resolve_transition(model, element_id)
        except ApiError:
            transition = None
        if verify is None:
            if transition is None:
                raise ApiError(
                    f"set_tab({element_id!r}): перехода в модели нет — "
                    f"передайте verify явно (табы частично не размечены — "
                    f"KNOWN_GAPS)")
            verify = VerifySpec.from_transition(transition)
        if transition is not None:
            on_fail = on_fail or transition.on_fail
        return self._log_action(
            "set_tab",
            lambda: _navigation(click_element(
                element_id, verify=verify, on_fail=on_fail, variant=variant,
                **self._click_bundle(model, "market", c))),
            target="market", args={"element": element_id, "variant": variant})

    def set_filter(self, element_id: str, *,
                   verify: Optional[VerifySpec] = None,
                   on_fail: Tuple[str, ...] = (),
                   variant: Optional[str] = None,
                   cancelled: Optional[Callable[[], bool]] = None
                   ) -> NavigationResult:
        """Клик по фильтру обзора (контракт — как у set_tab)."""
        self._check_thread()
        c = self._cancelled(cancelled)
        model = self._w.market
        try:
            transition = _resolve_transition(model, element_id)
        except ApiError:
            transition = None
        if verify is None:
            if transition is None:
                raise ApiError(
                    f"set_filter({element_id!r}): перехода в модели нет — "
                    f"передайте verify явно (фильтры частично не размечены — "
                    f"KNOWN_GAPS)")
            verify = VerifySpec.from_transition(transition)
        if transition is not None:
            on_fail = on_fail or transition.on_fail
        return self._log_action(
            "set_filter",
            lambda: _navigation(click_element(
                element_id, verify=verify, on_fail=on_fail, variant=variant,
                **self._click_bundle(model, "market", c))),
            target="market",
            args={"element": element_id, "variant": variant})

    def search(self, text: str, *,
               submit: bool = True,
               field_element_id: Optional[str] = None,
               search_button_id: Optional[str] = None,
               cancelled: Optional[Callable[[], bool]] = None) -> EnterResult:
        """Ввод текста поиска в обзоре (примитив 5.4; OCR-регион поля —
        wiring.search_region, в модели не размечен — 6.1)."""
        self._check_thread()
        c = self._cancelled(cancelled)
        w = self._w
        if w.search_region is None:
            raise ApiError(
                "search: OCR-регион поля поиска не задан (wiring."
                "search_region; в модели обзора не размечен — KNOWN_GAPS): "
                "поиск без OCR-verify введённого текста запрещён")
        model = dataclasses.replace(
            w.market, ocr_regions={**w.market.ocr_regions,
                                   SEARCH_REGION_ID: w.search_region})
        return self._log_action(
            "search",
            lambda: enter_text(
                text, model=model, region_id=SEARCH_REGION_ID, ocr=w.ocr,
                field_element_id=field_element_id,
                search_button_id=search_button_id, submit=submit,
                **self._enter_bundle("market", c)),
            target="market", args={"text": text, "submit": submit})

    def scroll_list(self, direction: str = DIR_DOWN, n: int = 1, *,
                    on_fail: Tuple[str, ...] = (),
                    cancelled: Optional[Callable[[], bool]] = None
                    ) -> ScrollResult:
        """Скролл списка обзора на n щелчков колеса (примитив 5.5)."""
        self._check_thread()
        c = self._cancelled(cancelled)
        w = self._w
        return self._log_action(
            "scroll_list",
            lambda: scroll_list(
                direction, n, model=w.market, screen=w.screen, hid=w.hid,
                cursor_pos=w.cursor_pos, activate=w.activate,
                window_status=lambda frame: w.window_status(frame, "market"),
                find_anchor=lambda frame: w.find_anchor(frame, "market"),
                run_checks=w.run_checks, config=w.config,
                segmentation=w.segmentation, on_fail=on_fail,
                free_area_point=w.free_area_point, cancelled=c,
                rng=w.rng, clock=w.clock, sleep=w.sleep),
            target="market", args={"direction": direction, "n": n})

    # ------------------------------------------------------------------
    # Чтение

    def _enter_bundle(self, window_id: str,
                      cancelled: Optional[Callable[[], bool]]) -> dict:
        w = self._w
        return dict(
            screen=w.screen, hid=w.hid, cursor_pos=w.cursor_pos,
            activate=w.activate,
            window_status=lambda frame: w.window_status(frame, window_id),
            find_anchor=lambda frame: w.find_anchor(frame, window_id),
            run_checks=w.run_checks, config=w.config,
            free_area_point=w.free_area_point, cancelled=cancelled,
            rng=w.rng, clock=w.clock, sleep=w.sleep)

    def read_rows(self, *, cancelled: Optional[Callable[[], bool]] = None
                  ) -> RowsResult:
        """Прочитать видимые строки списка обзора текущего кадра
        (только чтение: скролл — scroll_list, обзор — разведка цен, 6.1).
        Окно обзора обязано быть открыто (якорь на кадре) — иначе ApiError:
        читать закрытый список нельзя, стратегия обязана знать состояние."""
        self._check_thread()
        c = self._cancelled(cancelled)
        self._raise_if_cancelled(c)
        w = self._w
        frame = np.asarray(w.screen.grab())
        anchor = w.find_anchor(frame, "market")
        if anchor is None:
            raise ApiError(
                "read_rows: якорь окна обзора не найден на кадре — "
                "слепое чтение невозможно (обзор открыт? get_state())")
        return read_rows(
            frame, model=w.market, ocr=w.ocr, anchor=anchor,
            threshold=w.config.ocr.confidence_threshold,
            reread_count=w.config.ocr.reread_count,
            segmentation=w.segmentation)

    def read_shop_state(self, *,
                        regions: Optional[Mapping[str, OcrRegionSpec]] = None,
                        counter_region: str = "sold_count",
                        revenue_region: Optional[str] = "revenue",
                        cancelled: Optional[Callable[[], bool]] = None
                        ) -> ShopStateResult:
        """Разовое чтение показателей лавки продажи (паттерн монитора 6.3
        БЕЗ baseline): sold_count обязателен, revenue опционален. Деградация
        (якоря нет/регион не сошёлся) — флагами в результате, не исключением;
        HID не трогается вообще (наблюдатель, не исполнитель)."""
        self._check_thread()
        c = self._cancelled(cancelled)
        self._raise_if_cancelled(c)
        w = self._w
        resolved = {**w.shop_sell.ocr_regions,
                    **dict(w.shop_regions or {}), **dict(regions or {})}
        counter = resolved.get(counter_region)
        if counter is None:
            raise ApiError(
                f"read_shop_state: OCR-регион показателя не задан (id "
                f"{counter_region!r}; ни параметром regions, ни wiring."
                f"shop_regions, ни моделью {w.shop_sell.window.id!r} — "
                f"KNOWN_GAPS): чтение показателей без региона невозможно")
        self._validate_numeric(counter)
        revenue = None
        if revenue_region is not None:
            revenue = resolved.get(revenue_region)
            if revenue is None:
                raise ApiError(
                    f"read_shop_state: OCR-регион прибыли ({revenue_region!r}) "
                    f"задан ключом, но не найден ни в параметре regions, ни в "
                    f"wiring.shop_regions, ни в модели: конфигурация "
                    f"некорректна (fail fast)")
            self._validate_numeric(revenue)

        frame = np.asarray(w.screen.grab())
        snapshot_id = hashlib.sha1(
            np.ascontiguousarray(frame).tobytes()).hexdigest()
        anchor = w.find_anchor(frame, "shop_sell")
        if anchor is None:
            return ShopStateResult(
                sold_count=None, revenue=None, snapshot_id=snapshot_id,
                degraded=True,
                failures=("якорь окна лавки не найден на кадре: чтение "
                          "показателей невозможно (окно закрыто/сдвинуто)",))
        failures: list = []
        sold = self._read_indicator(frame, anchor, counter, failures)
        revenue_value = None
        if revenue is not None:
            revenue_value = self._read_indicator(frame, anchor, revenue,
                                                 failures)
        return ShopStateResult(
            sold_count=sold, revenue=revenue_value, snapshot_id=snapshot_id,
            degraded=bool(failures), failures=tuple(failures))

    @staticmethod
    def _validate_numeric(region: OcrRegionSpec) -> None:
        if region.grammar.kind not in _NUMERIC_GRAMMARS:
            raise ApiError(
                f"OCR-регион {region.id!r}: грамматика "
                f"{region.grammar.kind!r} не числовая "
                f"({sorted(_NUMERIC_GRAMMARS)}): показателем лавки может "
                f"быть только число")

    def _read_indicator(self, frame: np.ndarray, anchor: Anchor,
                        region: OcrRegionSpec,
                        failures: list) -> Optional[int]:
        """Один показатель: rect региона от якоря -> read_confident (4.5).
        Не прочитан — None + диагностика (деградация, не исключение)."""
        rect: Rect = region.rect
        frame_region = OcrRegionSpec(
            id=region.id,
            rect=Rect(anchor.x + rect.x, anchor.y + rect.y,
                      rect.width, rect.height),
            grammar=region.grammar, placeholder=region.placeholder)
        try:
            result = read_confident(
                self._w.ocr, frame, frame_region,
                threshold=self._w.config.ocr.confidence_threshold,
                reread_count=self._w.config.ocr.reread_count)
        except OcrReadError as exc:
            failures.append(f"OCR региона {region.id!r} не сошёлся: {exc}")
            return None
        return None if result.value is None else int(result.value)

    def get_state(self, *, cancelled: Optional[Callable[[], bool]] = None
                  ) -> CoreState:
        """Текущее состояние: открытое окно торговли + статус 3.5 +
        подтверждённое состояние/params 3.4 (только чтение, HID не трогается)."""
        self._check_thread()
        c = self._cancelled(cancelled)
        self._raise_if_cancelled(c)
        w = self._w
        frame = np.asarray(w.screen.grab())
        window = w.current_window(frame)
        if window is None:
            return CoreState(
                window=None, status=STATUS_NOT_FOUND, state=None,
                detail="окно торговли на кадре не найдено")
        status = w.window_status(frame, window)
        state = None
        params: Mapping[str, object] = {}
        if w.detect_state is not None and status.anchor is not None:
            report = w.detect_state(frame, status.anchor)
            state = report.state
            params = report.params
        detail = status.reason or status.content_detail or ""
        return CoreState(
            window=window, status=status.status, state=state, params=params,
            content_loaded=status.content_loaded, detail=detail)

    # ------------------------------------------------------------------
    # Лавка продажи

    def setup_sell(self, items, *,
                   source_locator: Callable[[SellItem], Optional[Tuple[int, int]]],
                   grid=None,
                   price_region: Optional[OcrRegionSpec] = None,
                   open_element: Optional[str] = "open_sale_trade",
                   open_verify: Optional[VerifySpec] = None,
                   price_field_element: Optional[str] = None,
                   drag_verify: VerifySpec,
                   final_verify: Optional[VerifySpec] = None,
                   price_submit: bool = True,
                   on_fail: Tuple[str, ...] = (),
                   cancelled: Optional[Callable[[], bool]] = None
                   ) -> SetupSellResult:
        """Выставление лавки продажи (сценарий 6.2): открытие через HUD
        (open_verify обязателен при open_element — межокнные переходы
        моделью не выражены), drag+цена по слотам сетки. price_region —
        wiring.price_region по умолчанию (региона цены в модели нет — 6.2)."""
        self._check_thread()
        c = self._cancelled(cancelled)
        w = self._w
        region = price_region if price_region is not None else w.price_region
        if region is None:
            raise ApiError(
                "setup_sell: OCR-регион поля цены не задан (параметр "
                "price_region / wiring.price_region; в модели лавки не "
                "размечен — KNOWN_GAPS): ввод цены без OCR-verify запрещён")
        return self._log_action(
            "setup_sell",
            lambda: setup_sell(
                items, model=w.shop_sell, ocr=w.ocr, screen=w.screen,
                hid=w.hid,
                cursor_pos=w.cursor_pos, activate=w.activate,
                window_status=lambda frame: w.window_status(frame, "shop_sell"),
                find_anchor=lambda frame: w.find_anchor(frame, "shop_sell"),
                run_checks=w.run_checks, config=w.config, grid=grid,
                hud_model=w.hud_model, hud_locator=w.hud_locator,
                open_element=open_element, open_verify=open_verify,
                source_locator=source_locator, price_region=region,
                price_field_element=price_field_element,
                price_submit=price_submit, drag_verify=drag_verify,
                final_verify=final_verify, on_fail=on_fail,
                free_area_point=w.free_area_point, cancelled=c,
                rng=w.rng, clock=w.clock, sleep=w.sleep),
            target="shop_sell",
            args={"items": [{"slot": it.slot, "price": it.price}
                            for it in items]})

    def start_sell(self, *,
                   confirm_checks: VerifySpec,
                   verify: VerifySpec,
                   element_id: str = "sell_start_button",
                   action_id: str = "start_sell",
                   on_fail: Tuple[str, ...] = (),
                   cancelled: Optional[Callable[[], bool]] = None
                   ) -> StartSellResult:
        """Подтверждение лавки — ПОТЕНЦИАЛЬНО НЕОБРАТИМО (сценарий 6.2):
        re-verify перед кликом, страж необратимых 5.7 (guard API): UNKNOWN
        блокирует повтор до явного guard.confirm_after_unknown стратегией.
        verify/confirm_checks обязательны (переходы моделью не выражены)."""
        self._check_thread()
        c = self._cancelled(cancelled)
        w = self._w
        return self._log_action(
            "start_sell",
            lambda: start_sell(
                model=w.shop_sell, screen=w.screen, hid=w.hid,
                cursor_pos=w.cursor_pos, activate=w.activate,
                window_status=lambda frame: w.window_status(frame, "shop_sell"),
                find_anchor=lambda frame: w.find_anchor(frame, "shop_sell"),
                run_checks=w.run_checks, config=w.config,
                confirm_checks=confirm_checks, verify=verify,
                element_id=element_id, guard=w.guard, action_id=action_id,
                hud_locator=w.hud_locator, on_fail=on_fail,
                free_area_point=w.free_area_point, cancelled=c,
                rng=w.rng, clock=w.clock, sleep=w.sleep),
            target="shop_sell", irreversible=True,
            args={"element": element_id})

    def cancel_sell(self, *,
                    verify: VerifySpec,
                    element_id: Optional[str] = "cansel_sell_button",
                    on_fail: Tuple[str, ...] = (),
                    cancelled: Optional[Callable[[], bool]] = None
                    ) -> CancelSellResult:
        """Отмена выставления лавки (сценарий 6.2): клик по кнопке отмены
        элемента модели или ESC (element_id=None — честный fallback)."""
        self._check_thread()
        c = self._cancelled(cancelled)
        w = self._w
        return self._log_action(
            "cancel_sell",
            lambda: cancel_sell(
                model=w.shop_sell, screen=w.screen, hid=w.hid,
                cursor_pos=w.cursor_pos, activate=w.activate,
                window_status=lambda frame: w.window_status(frame, "shop_sell"),
                find_anchor=lambda frame: w.find_anchor(frame, "shop_sell"),
                run_checks=w.run_checks, config=w.config, verify=verify,
                element_id=element_id, on_fail=on_fail,
                free_area_point=w.free_area_point, cancelled=c,
                rng=w.rng, clock=w.clock, sleep=w.sleep),
            target="shop_sell",
            args={"element": element_id})

    # ------------------------------------------------------------------
    # Сервисные

    def recover(self, cause: RecoverCause, target: str, *,
                hud_element: str,
                on_escalation: Optional[Callable[[RecoverResult], None]] = None,
                resume: Optional[Callable[[], bool]] = None,
                pause_timeout: Optional[float] = None,
                cancelled: Optional[Callable[[], bool]] = None
                ) -> RecoverResult:
        """Общий recover по сигналу needs_recover (сценарий 6.4 поверх
        цепочки 5.8). cause — результат примитива/сценария/навигации API с
        needs_recover; без него — ScenarioRecoverError (fail fast). Пара
        target/hud_element — решение вызывающего слоя (market+market_pic /
        shop_buy+open_buy_trade / shop_sell+open_sale_trade). По итогу
        публикует события журнально-очередного канала (7.3): успех ->
        recovered, эскалация -> escalation (payload — справочник
        docs/API.md); отмена — без событий."""
        self._check_thread()
        c = self._cancelled(cancelled)
        w = self._w

        def _run() -> RecoverResult:
            result = scenario_recover(
                cause, target,
                hud_model=w.hud_model, hud_element=hud_element,
                screen=w.screen, hid=w.hid, cursor_pos=w.cursor_pos,
                activate=w.activate,
                window_status=lambda frame: w.window_status(frame, target),
                find_anchor=lambda frame: w.find_anchor(frame, target),
                goal_reached=w.goal_reached(target), config=w.config,
                current_window=w.current_window, close_point=w.close_point,
                hud_locator=w.hud_locator,
                on_escalation=on_escalation or w.on_escalation,
                resume=resume, pause_timeout=pause_timeout,
                free_area_point=w.free_area_point, cancelled=c,
                rng=w.rng, clock=w.clock, sleep=w.sleep)
            # События по итогу цепочки (spec core-api-events, журнал 7.3):
            # успех — recovered, эскалация — escalation (payload — справочник
            # docs/API.md); отмена — событий нет. Канал — единый: events.push
            # -> EventQueue -> журнал + dispatcher.
            if self._events is not None:
                steps = tuple(step.name for step in result.steps)
                if result.success:
                    self._events.push(EVENT_RECOVERED,
                                      {"target": target, "steps": steps})
                elif result.escalated:
                    self._events.push(EVENT_ESCALATION,
                                      {"target": target,
                                       "detail": result.escalation_detail,
                                       "steps": steps})
            return result

        return self._log_action("recover", _run, target=target,
                                args={"hud_element": hud_element})

    def health(self) -> HealthResult:
        """Агрегат живости: окно игры (находилка 3.2), открытое окно
        торговли, OCR и HID — через провайдеров сборки. Диагностика перед
        циклом стратегии; самодиагностика не трогает HID-эффекты."""
        self._check_thread()
        w = self._w
        game = w.game_window()
        frame = np.asarray(w.screen.grab())
        trading = w.current_window(frame)
        ocr_ok = w.ocr_health()
        hid_ok = w.hid_health()
        return HealthResult(
            ok=game.ok and ocr_ok and hid_ok,
            game_window=game.ok,
            game_detail=game.reason or "окно игры доступно",
            trading_window=trading, ocr=ocr_ok, hid=hid_ok)
