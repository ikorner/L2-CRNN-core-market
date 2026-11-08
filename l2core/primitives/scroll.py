"""Примитив скролла списка и чтение позиции скролла (задача 5.5,
spec market-scenarios: «Сканирование списка до конца» — scroll_list(dir, n),
единица n = щелчок колеса, конец списка по признаку модели; позиция скролла —
кропы «прокрутка вверх/вниз» + сравнение нижней видимой строки между кадрами;
action-primitives: verify на каждое действие, retry → on_fail → recover;
инвариант 9: слепых действий нет).

scroll_list — оркестратор, весь внешний мир — инъекции (HID, экран/
транзакция кадра 3.6, activate, статус окна, поиск якоря, оценка проверок,
rng, часы):

    пролог → наведение курсора на область списка → n итераций (wheel →
    verify смены нижней строки ИЛИ признака конца) → retry/on_fail → recover

ПРОЛОГ (тот же binding ruling, что в 5.3/5.4): скролл допустим только при
activate_game OK (5.2, иначе ActivateGameError проброшена) И статусе окна
found (3.5). Иначе — исключение, HID не трогается (no blind scroll).

МЕХАНИКА СКРОЛЛА: в протоколе вендора (hid.py) wheel есть — CMD_SCROLL
(HidDevice.scroll(count, step, delay_ms)), поэтому клики по стрелкам как
fallback НЕ нужны (стрелки — только источник признаков позиции). Единица
n = один щелчок колеса = одна HID-команда scroll(1, ±1, delay); знак шага:
вниз +1, вверх −1 (положительная дельта колеса — прокрутка от себя).
wheel_delay_ms — пауза прошивки между шагами внутри пакета (при count=1
не влияет, параметр открыт для профиля).

ОБЛАСТЬ СПИСКА: элемент-область списка в модели отсутствует (импорт 2.2 его
не кропил — KNOWN_GAPS), поэтому курсор наводится на центр таблицы по
table_geometry: x — середина (col_seps[0], col_seps[-1]), y — середина
(row_y0, row_y0 + max_rows·row_pitch), от якоря. table_geometry у окна
обязательна (без неё сравнивать строки нечем — ScrollError, fail fast).

VERIFY ИТЕРАЦИИ (инвариант 2): после каждого щелчка wait_until по свежим
кадрам (refresh транзакции, первый poll немедленно) до смены хэша нижней
видимой строки ИЛИ до признака края по направлению (скролл вниз — стрелка
«вниз» disabled, вверх — «вверх»). Таймаут — timing.wait_default_timeout,
интервал — timing.verify_interval (единая точка дефолтов). Нижняя строка —
последняя непустая строка нарезки TableSegmentation (4.3) на константах
table_geometry модели; хэш — sha1 по обрезанным ячейкам строки (кропы 1:1,
детерминирован, байтовое сравнение между кадрами, не OCR). Прямого sleep
нет (инвариант 1): ожидание — только wait_until по инъецированным часам.

ПРИЗНАК КОНЦА ИЗ МОДЕЛИ (предпочтителен по спеке): disabled-стрелки —
элементы prokrutka_vverh_not_active / prokrutka_vniz_not_active (кропы
scr_153722_*_not_active, вариант записи «active»); оценка — template-проверка
через инъекцию run_checks (matcher/assets остаются слоем детекции, wiring —
6.x). Элементов нет — at_start/at_end = None и verify опирается только на
смену строки; это честный fallback (задокументирован в KNOWN_GAPS): совпадение
строк на границе страницы при живой стрелке выглядит как «не проскроллилось»
→ retry → on_fail. Замкнутый список (строк меньше вьюпорта) закрыт стрелкой:
она disabled с самого начала — цикл не стартует.

RETRY/ON_FAIL: попыток на итерацию = 1 + timing.click_retry_count (единый
параметр повторов действий); исчерпание → повторный пролог (activate +
found — иначе click_free_area был бы слепым, инвариант 9) → on_fail[] тех же
шагов, что в 5.3 (press_esc/click_free_area, общая механика click.py) →
результат needs_recover=True (исполнение recover — 5.8).

РЕЗУЛЬТАТ: ScrollResult (requested/scrolled/reached_end/position/wheels/
cancelled/on_fail_executed/failures) — без трёхзначной обёртки (5.7, YAGNI).
success = всё n проскроллено ИЛИ достигнут край по направлению (дальше
крутить нечего). Отмена (kill switch 8.1, инъекция cancelled) — флаг,
не исключение (консистентно с wait.py/activate.py/click.py).

scroll_position(frame, ...) — отдельное чтение позиции: состояние стрелок
(None — элементов нет в модели) + bottom_row_hash (None — строк нет/окно
вне кадра). Наблюдатель/монитор может дергать её между действиями.
"""
from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from l2core.adapters.screen import Screen
from l2core.adapters.table_segmentation import TableSegmentation
from l2core.config import CoreConfig
from l2core.detector.anchor import Anchor
from l2core.detector.status import STATUS_FOUND, WindowStatus
from l2core.detector.transaction import FrameTransaction
from l2core.model import Check, UiModel
from l2core.primitives.activate import ActivateResult
from l2core.primitives.click import (
    CheckRunner,
    JitterRng,
    _move_to,
    _run_on_fail,
)
from l2core.primitives.wait import wait_until

__all__ = [
    "DIR_DOWN",
    "DIR_UP",
    "DOWN_ARROW_DISABLED_ELEMENT",
    "UP_ARROW_DISABLED_ELEMENT",
    "ScrollError",
    "ScrollPosition",
    "ScrollPreconditionError",
    "ScrollResult",
    "scroll_list",
    "scroll_position",
]

# Направления скролла.
DIR_UP = "up"
DIR_DOWN = "down"

# Disabled-стрелки списка «Обзор лавок»: импорт 2.2 (кропы
# scr_153722_*_not_active). Совпадение шаблона варианта — признак края
# списка (в начале для «вверх», в конце для «вниз»). Ключ варианта в записи
# элемента — «active» (именование импорта, не семантика).
UP_ARROW_DISABLED_ELEMENT = "prokrutka_vverh_not_active"
DOWN_ARROW_DISABLED_ELEMENT = "prokrutka_vniz_not_active"
_DISABLED_VARIANT = "active"


class ScrollError(Exception):
    """Ошибка исполнения примитива (модель/аргументы), не эскалация рантайма."""

    def __init__(self, detail: str):
        super().__init__(f"scroll_list: {detail}")
        self.detail = detail


class ScrollPreconditionError(ScrollError):
    """Пролог не пройден: статус окна не found — скролл запрещён (инвариант 9).

    activate_game не удался — отдельно: ActivateGameError (5.2) проброшена.
    """

    def __init__(self, direction: str, status: WindowStatus):
        self.status = status.status
        super().__init__(
            f"направление {direction!r}: статус окна {status.status!r} "
            f"({status.reason or status.content_detail}) — скролл не выполняется"
        )


@dataclass(frozen=True)
class ScrollPosition:
    """Позиция скролла списка на кадре.

    at_start/at_end — признаки краёв по disabled-стрелкам модели; None —
    элементов-стрелок в модели нет (край не выражен, fallback — только
    сравнение строк). bottom_row_hash — sha1 по обрезанным ячейкам нижней
    видимой строки (нарезка 4.3); None — непустых строк нет или окно вне
    кадра: сравнивать нечего.
    """

    at_start: Optional[bool]
    at_end: Optional[bool]
    bottom_row_hash: Optional[bytes]


@dataclass(frozen=True)
class ScrollResult:
    """Итог scroll_list (обёртка OK/FAILED/UNKNOWN — задача 5.7, YAGNI).

    requested — заказанное число щелчков; scrolled — фактически
    проскроллено (подтверждено verify); reached_end — край по направлению
    достигнут по признаку модели; position — позиция на последнем кадре;
    wheels — всего послано HID-команд wheel (включая неподтверждённые
    попытки); failures — диагностика провалов.
    """

    requested: int
    scrolled: int
    reached_end: bool
    position: Optional[ScrollPosition]
    wheels: int = 0
    cancelled: bool = False
    on_fail_executed: bool = False
    failures: Tuple[str, ...] = ()

    @property
    def success(self) -> bool:
        """Успех: всё n проскроллено ИЛИ достигнут край (дальше нечего)."""
        return not self.cancelled and (
            self.scrolled == self.requested or self.reached_end)

    @property
    def needs_recover(self) -> bool:
        """Сигнал общего recover (5.8): неуспех без отмены."""
        return not self.success and not self.cancelled


def _to_gray(crop: np.ndarray) -> np.ndarray:
    """BGR-кроп -> серый (BT.601, как в OpenCV и detector.anchor)."""
    return crop[..., 0] * 0.114 + crop[..., 1] * 0.587 + crop[..., 2] * 0.299


def _arrow_disabled(frame: np.ndarray, anchor: Anchor, model: UiModel,
                    run_checks: CheckRunner, element_id: str) -> Optional[bool]:
    """Состояние disabled-стрелки: True — disabled (край), None — элемента нет."""
    if element_id not in model.elements:
        return None
    report = run_checks(
        (Check(kind="template", element=element_id,
               variant=_DISABLED_VARIANT),),
        frame, anchor)
    return report.passed


def _bottom_row_hash(frame: np.ndarray, model: UiModel, anchor: Anchor,
                     segmentation: TableSegmentation) -> Optional[bytes]:
    """Хэш нижней видимой строки: sha1 по обрезанным ячейкам (нарезка 4.3).

    Окно вне кадра или непустых строк нет — None (сравнивать нечего).
    """
    w, h = model.window.size
    crop = frame[anchor.y:anchor.y + h, anchor.x:anchor.x + w]
    if crop.shape[0] < h or crop.shape[1] < w:
        return None
    structure = segmentation.rows_and_columns(
        _to_gray(crop), origin=(anchor.x, anchor.y))
    rows = structure["rows"]
    if not rows:
        return None
    last = len(rows) - 1
    cells = segmentation.extract_cells(frame, structure)
    digest = hashlib.sha1()
    for cell in sorted((c for c in cells if c.meta["row"] == last),
                       key=lambda c: c.meta["rect"]):
        digest.update(cell.meta["col"].encode("utf-8"))
        digest.update(np.ascontiguousarray(cell.image).tobytes())
    return digest.digest()


def scroll_position(
    frame: np.ndarray,
    *,
    model: UiModel,
    anchor: Anchor,
    run_checks: CheckRunner,
    segmentation: Optional[TableSegmentation] = None,
) -> ScrollPosition:
    """Позиция скролла на кадре: состояние стрелок + хэш нижней строки.

    Якорь передаётся готовым (поиск — слой детекции, у примитива инъекция
    find_anchor). segmentation — инъекция нарезки (тесты/кастом); по
    умолчанию строится из table_geometry модели (процесс-wide подмена
    констант вендора — экземпляр один на окно, см. segmentation.py).
    """
    if segmentation is None and model.table_geometry is not None:
        segmentation = TableSegmentation(model.table_geometry)
    at_start = _arrow_disabled(frame, anchor, model, run_checks,
                               UP_ARROW_DISABLED_ELEMENT)
    at_end = _arrow_disabled(frame, anchor, model, run_checks,
                             DOWN_ARROW_DISABLED_ELEMENT)
    bottom = None
    if segmentation is not None:
        bottom = _bottom_row_hash(frame, model, anchor, segmentation)
    return ScrollPosition(at_start=at_start, at_end=at_end,
                          bottom_row_hash=bottom)


def _list_area_point(geometry, anchor: Anchor) -> Tuple[int, int]:
    """Центр области таблицы: где должен стоять курсор для wheel.

    Модель элемента-области списка не содержит (KNOWN_GAPS) — честный
    fallback на геометрию таблицы: x — середина контента (col_seps[0]..[-1]),
    y — середина полосы строк (row_y0..row_y0 + max_rows·pitch).
    """
    x = (geometry.col_seps[0] + geometry.col_seps[-1]) // 2
    y = (geometry.row_y0 + geometry.row_y0
         + geometry.max_rows * geometry.row_pitch) // 2
    return anchor.x + x, anchor.y + y


def scroll_list(
    direction: str,
    n: int,
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
    segmentation: Optional[TableSegmentation] = None,
    on_fail: Tuple[str, ...] = (),
    free_area_point: Optional[Tuple[int, int]] = None,
    wheel_delay_ms: int = 0,
    cancelled: Optional[Callable[[], bool]] = None,
    rng: Optional[JitterRng] = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> ScrollResult:
    """Скролл списка на n щелчков колеса: пролог → hover на область списка
    → n итераций (wheel → verify смены строки/края) → retry → on_fail →
    recover. Контракт — в докстринге модуля."""
    if direction not in (DIR_UP, DIR_DOWN):
        raise ScrollError(
            f"неизвестное направление {direction!r} — ожидалось "
            f"{DIR_UP!r}/{DIR_DOWN!r}")
    if n < 1:
        raise ScrollError(f"n должно быть >= 1, получено {n}")
    if model.table_geometry is None:
        raise ScrollError(
            f"окно {model.window.id!r}: table_geometry отсутствует — "
            f"область списка и нижняя строка без неё не выражены")
    rng = rng or random.Random()
    seg = segmentation or TableSegmentation(model.table_geometry)
    step = 1 if direction == DIR_DOWN else -1
    # Признак края по направлению: вниз — конец, вверх — начало.
    end_flag = "at_end" if direction == DIR_DOWN else "at_start"

    failures: list = []
    with FrameTransaction(screen) as tx:
        # --- ПРОЛОГ: activate OK (5.2) + статус found (3.5), иначе скролла нет.
        activated = activate()
        if activated.cancelled:
            return ScrollResult(requested=n, scrolled=0, reached_end=False,
                                position=None, cancelled=True)
        status = window_status(tx.frame)
        if status.status != STATUS_FOUND:
            raise ScrollPreconditionError(direction, status)

        # Начальная позиция + наведение курсора на область списка.
        position: Optional[ScrollPosition] = None
        prev_hash: Optional[bytes] = None
        frame = tx.refresh()
        anchor = find_anchor(frame)
        if anchor is None:
            failures.append("якорь не найден на начальном кадре")
        else:
            position = scroll_position(frame, model=model, anchor=anchor,
                                       run_checks=run_checks, segmentation=seg)
            prev_hash = position.bottom_row_hash
            _move_to(hid, cursor_pos, _list_area_point(model.table_geometry,
                                                       anchor))

        max_attempts = 1 + config.timing.click_retry_count
        wheels = 0
        scrolled = 0
        for i in range(1, n + 1):
            if cancelled is not None and cancelled():
                return ScrollResult(requested=n, scrolled=scrolled,
                                    reached_end=False, position=position,
                                    wheels=wheels, cancelled=True,
                                    failures=tuple(failures))
            # Край по направлению уже на текущем кадре: крутить нечего.
            if position is not None and getattr(position, end_flag):
                break

            done = False
            for attempt in range(1, max_attempts + 1):
                if cancelled is not None and cancelled():
                    return ScrollResult(requested=n, scrolled=scrolled,
                                        reached_end=False, position=position,
                                        wheels=wheels, cancelled=True,
                                        failures=tuple(failures))
                hid.scroll(1, step, wheel_delay_ms)
                wheels += 1

                def _settled() -> Optional[ScrollPosition]:
                    # Verify итерации: свежий кадр, свежий якорь (кэша
                    # позиции нет — инвариант 3), край по направлению ИЛИ
                    # нижняя строка изменилась относительно prev_hash.
                    f = tx.refresh()
                    a = find_anchor(f)
                    if a is None:
                        return None
                    pos = scroll_position(f, model=model, anchor=a,
                                          run_checks=run_checks,
                                          segmentation=seg)
                    if getattr(pos, end_flag):
                        return pos
                    if (pos.bottom_row_hash is not None
                            and pos.bottom_row_hash != prev_hash):
                        return pos
                    return None

                waited = wait_until(
                    _settled,
                    timeout=config.timing.wait_default_timeout,
                    interval=config.timing.verify_interval,
                    cancelled=cancelled,
                    clock=clock,
                    sleep=sleep,
                )
                if waited.cancelled:
                    return ScrollResult(requested=n, scrolled=scrolled,
                                        reached_end=False, position=position,
                                        wheels=wheels, cancelled=True,
                                        failures=tuple(failures))
                if waited.success:
                    position = waited.last_value
                    prev_hash = position.bottom_row_hash
                    done = True
                    break
                failures.append(
                    f"итерация {i}, попытка {attempt}: нижняя строка не "
                    f"изменилась и край по направлению не достигнут за "
                    f"таймаут")
            if not done:
                # Исчерпание retry: ПОВТОРНЫЙ ПРОЛОГ перед on_fail (паттерн
                # 5.3): за цикл окно могло деградировать, а click_free_area
                # кликает в область экрана — без activate OK + found это
                # слепой клик (инвариант 9): on_fail пропускаем и сразу
                # отдаём сигнал recover с причиной.
                reactivated = activate()
                if reactivated.cancelled:
                    return ScrollResult(requested=n, scrolled=scrolled,
                                        reached_end=False, position=position,
                                        wheels=wheels, cancelled=True,
                                        failures=tuple(failures))
                restatus = window_status(tx.refresh())
                if restatus.status != STATUS_FOUND:
                    failures.append(
                        f"on_fail пропущен: статус окна "
                        f"{restatus.status!r} ({restatus.reason or restatus.content_detail})"
                        f" — слепые HID-шаги запрещены")
                    return ScrollResult(requested=n, scrolled=scrolled,
                                        reached_end=False, position=position,
                                        wheels=wheels,
                                        on_fail_executed=False,
                                        failures=tuple(failures))
                _run_on_fail(on_fail, hid=hid, cursor_pos=cursor_pos,
                             free_area_point=free_area_point,
                             config=config, rng=rng)
                return ScrollResult(requested=n, scrolled=scrolled,
                                    reached_end=False, position=position,
                                    wheels=wheels,
                                    on_fail_executed=bool(on_fail),
                                    failures=tuple(failures))
            scrolled = i

        reached_end = position is not None and bool(getattr(position, end_flag))
        return ScrollResult(requested=n, scrolled=scrolled,
                            reached_end=reached_end, position=position,
                            wheels=wheels, failures=tuple(failures))
