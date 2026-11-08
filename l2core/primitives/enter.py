"""Примитивы ввода текста/цены (задача 5.4, spec action-primitives: «Ввод
цены и текста с выбором способа по профилю», «Восстановление раскладки при
вводе текста»; инвариант 9: слепых действий нет).

enter_text/enter_price — оркестраторы поверх готовых механизмов: фокус и
клики по элементам — через click_element (5.3, тот же пролог/hover/verify/
retry), проверка содержимого поля — read_stable (4.5), грамматики (4.4).

ПОТОК (попыток = 1 + timing.click_retry_count):

    пролог (activate OK + статус found) -> выбор способа ввода (rng, профиль)
    -> фокус поля (клик по элементу) -> ввод -> OCR verify (read_stable)
    -> при несовпадении: backspace + switch_layout + повтор + повторная проверка
    -> submit (Enter или кнопка «Поиск» — rng, профиль) -> retry -> on_fail
    -> сигнал recover

СПОСОБ ВВОДА (решение 14 design.md): клавиатура (HID key_press по keycode)
или экранный нумпад (клики по элементам модели через click_element) —
выбор по behavior.numpad_fraction через инъецируемый rng (детерминизм с
seed, инвариант 7). Маппинг символ -> элемент клавиши — параметр
numpad_map: семантика экранной клавиатуры сырьём НЕ подтверждена
(KNOWN_GAPS, OWNER_QUESTIONS Q25), выдуманной геометрии/раскладки клавиш
в ядре нет — только механизм через элементы модели.

КЛАВИАТУРНЫЙ ПУТЬ: символ -> HID keycode (usage table: цифры 0x1E..0x27,
a-z -> 0x04.., пробел 0x2C) через key_press с паузой key_delay_min/max_ms
из профиля (фиксированная задержка через wait_until по инъецированным
часам — инвариант 1, общий механизм — click._fixed_pause). Честная граница:
протокол вендора (hid.py) — 1-байтные коды (type_char кодирует latin-1),
кириллица вне протокола — не-ASCII символы отклоняются fail fast (Q25).

ВОССТАНОВЛЕНИЕ РАСКЛАДКИ (spec: «при транслите SHALL выполняться backspace +
switch_layout + повтор ввода»): OCR поля (read_stable) после набора;
несовпадение -> backspace x len(наблюденного OCR текста) + switch_layout +
повтор ввода + повторная OCR-проверка. Та же семантика «стереть-видимое-
и-набрать» у retry-попытки: перед повторным набором стирается содержимое,
видимое OCR в конце предыдущей попытки (иначе печать поверх делает verify
несходимым принципиально). switch_layout — сочетание клавиш переключения
раскладки: дефолт Ctrl+Shift через key_down/key_press/key_up (константы
LEFT_CTRL в вендорном hid.py нет — 0xE0 по HID usage table задан локально
с комментарием; сочетание подтверждается у владельца, Q25). Точка инъекции
switch_layout — заменяемая callable (тесты/другая раскладка). Чего OCR не
видит (placeholder/invalid), стирание физически не достаёт — честная
граница (очистка поля целиком и select-on-focus сырьём не смоделированы,
Q8/Q25, досъёмка модели).

SUBMIT (поисковое поле): Enter (KEY_ENTER) или кнопка «Поиск» (элемент
модели, клик через click_element) — выбор по behavior.enter_fraction через
rng. ВЕРНОСТЬ SUBMIT: клик по кнопке верифицируется присутствием кнопки
после клика (переход «список обновился» сырьём не подтверждён — KNOWN_GAPS,
задача модели/стратегии); у нажатия Enter наблюдаемого признака в модели
нет — честная граница: submit исполняется последним, после подтверждённого
содержимого поля, verify результата поиска — выше по стеку. submit=False —
ввод без отправки (стратегия сама решает, когда искать).

ОГРАНИЧЕНИЯ (R4, честно паркуется): семантика экранного нумпада (раскладка
клавиш, точка/запятая, backspace-клавиша) не подтверждена сырьём;
сочетание переключения раскладки; влияние submit на список; OCR verify
до submit, а не после (после Enter поле может очиститься); стирание поля —
по наблюдаемому OCR содержимому (см. выше). Детали — в отчёте
task-5.4-report.md и docs/OWNER_QUESTIONS.md (Q25).
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Tuple

import numpy as np

from l2core.adapters.hid import KEY_BACKSPACE, KEY_ENTER, KEY_LEFT_SHIFT
from l2core.adapters.screen import Screen
from l2core.config import CoreConfig
from l2core.detector.anchor import Anchor
from l2core.detector.status import STATUS_FOUND, WindowStatus
from l2core.model import Check, OcrRegionSpec, UiModel
from l2core.model.grammars import STATUS_OK
from l2core.model.reading import OcrReadError, StableReadResult, read_stable
from l2core.primitives.activate import ActivateResult
from l2core.primitives.click import (
    CheckRunner,
    ClickPreconditionError,
    ClickResult,
    JitterRng,
    VerifySpec,
    _fixed_pause,
    _run_on_fail,
    click_element,
)
from l2core.model.types import Rect

__all__ = [
    "INPUT_KEYBOARD",
    "INPUT_NUMPAD",
    "KEY_LEFT_CTRL",
    "SUBMIT_BUTTON",
    "SUBMIT_ENTER",
    "EnterPreconditionError",
    "EnterResult",
    "EnterTextError",
    "enter_price",
    "enter_text",
]

# Способы ввода и отправки (диагностика результата).
INPUT_KEYBOARD = "keyboard"
INPUT_NUMPAD = "numpad"
SUBMIT_ENTER = "enter"
SUBMIT_BUTTON = "button"

# LEFT_CTRL по HID usage table: в вендорном hid.py константы нет (только
# SHIFT/ALT), протокол 1-байтные коды принимает — задано локально,
# сочетание переключения раскладки подтверждается у владельца (Q25).
KEY_LEFT_CTRL = 0xE0


class EnterTextError(Exception):
    """Ошибка исполнения примитива (модель/аргументы), не эскалация рантайма."""

    def __init__(self, detail: str):
        super().__init__(f"enter_text: {detail}")
        self.detail = detail


class EnterPreconditionError(EnterTextError):
    """Пролог не пройден: статус окна не found — ввод запрещён (инвариант 9).

    activate_game не удался — отдельно: ActivateGameError (5.2) проброшена.
    """

    def __init__(self, region_id: str, status: WindowStatus):
        self.status = status.status
        super().__init__(
            f"регион {region_id!r}: статус окна {status.status!r} "
            f"({status.reason or status.content_detail}) — ввод не выполняется"
        )


@dataclass(frozen=True)
class EnterResult:
    """Итог ввода (успех/отмена/сигнал recover; обёртка OK/FAILED/UNKNOWN —
    задача 5.7, YAGNI).

    input_method — выбранный способ (keyboard/numpad); submit_method —
    способ отправки (enter/button/None при submit=False); layout_recovered —
    сработало восстановление раскладки; attempts/failures — как в ClickResult.
    """

    region_id: str
    success: bool
    attempts: int
    input_method: Optional[str] = None
    submit_method: Optional[str] = None
    layout_recovered: bool = False
    cancelled: bool = False
    on_fail_executed: bool = False
    failures: Tuple[str, ...] = ()

    @property
    def needs_recover(self) -> bool:
        """Сигнал общего recover (5.8): неуспех без отмены."""
        return not self.success and not self.cancelled


def _ascii_to_hid(ch: str) -> int:
    """Символ ASCII -> HID keycode (usage table, как в Keyboard.h).

    Цифры 0x1E..0x27 ('1' -> 0x1E, '0' -> 0x27), буквы a-z/A-Z -> 0x04..
    0x1D, пробел 0x2C. Прочие символы (пунктуация) вне таблицы — инъекция
    key_tap (граница протокола, Q25).
    """
    if "0" <= ch <= "9":
        return 0x1E + ((ord(ch) - ord("0") - 1) % 10)
    if "a" <= ch.lower() <= "z":
        return 0x04 + (ord(ch.lower()) - ord("a"))
    if ch == " ":
        return 0x2C
    raise EnterTextError(
        f"символ {ch!r} вне встроенной HID-таблицы (ASCII-цифры/буквы/"
        f"пробел) — задайте key_tap для ввода (протокол вендора 1-байтный, "
        f"Q25)")


def _default_key_tap(hid):
    """Печать символа клавиатурой: заглавные — с SHIFT (key_down/press/up,
    та же схема, что у _default_switch_layout для Ctrl), строчные/цифры/
    пробел — один key_press базового keycode."""
    def tap(ch: str) -> None:
        code = _ascii_to_hid(ch)
        if ch.isalpha() and ch.isupper():
            hid.key_down(KEY_LEFT_SHIFT)
            hid.key_press(code)
            hid.key_up(KEY_LEFT_SHIFT)
        else:
            hid.key_press(code)
    return tap


def _default_switch_layout(hid) -> None:
    """Переключение раскладки: Ctrl+Shift (key_down/up вокруг нажатия).

    Дефолтная механика-заглушка с честной границей: сочетание по умолчанию
    ОС не подтверждено сырьём/владельцем (Q25) — точка инъекции switch_layout.
    """
    hid.key_down(KEY_LEFT_CTRL)
    hid.key_press(KEY_LEFT_SHIFT)
    hid.key_up(KEY_LEFT_CTRL)


def _presence_verify(element_id: str) -> VerifySpec:
    """Verify присутствия элемента: шаблон его варианта normal (как hover
    re-verify в 5.3): факт клика подтверждаем, семантику эффекта модель
    пока не выражает (KNOWN_GAPS)."""
    return VerifySpec(checks=(Check(kind="template", element=element_id,
                                    variant="normal"),))


def _click_element(element_id: str, *, model, run_checks, screen, hid,
                   cursor_pos, activate, window_status, find_anchor,
                   config, rng, clock, sleep) -> ClickResult:
    """Клик по элементу внутри enter: присутствие-верификация, без on_fail
    (эскалацию/on_fail выполняет внешний цикл enter_text)."""
    return click_element(
        element_id,
        model=model,
        verify=_presence_verify(element_id),
        run_checks=run_checks,
        screen=screen,
        hid=hid,
        cursor_pos=cursor_pos,
        activate=activate,
        window_status=window_status,
        find_anchor=find_anchor,
        config=config,
        on_fail=(),
        rng=rng,
        clock=clock,
        sleep=sleep,
    )


class _TypeFailed(Exception):
    """Внутренний сигнал: набор текста прерван (клавиша не подтверждена)."""


def _erase_observed(hid, observed: str) -> None:
    """Стереть наблюдаемое OCR содержимое поля: backspace x len(наблюденного).

    Стирается то, что реально видит OCR, а не len вводимого текста: OCR
    может увидеть меньше набранного. Чего OCR не видит (статус placeholder/
    invalid), физически не стирается — честная граница (очистка поля целиком
    и select-on-focus сырьём не смоделированы, Q8/Q25)."""
    for _ in observed:
        hid.key_press(KEY_BACKSPACE)


def _read_field(ocr, screen, find_anchor, region, region_id) -> StableReadResult:
    """Прочитать поле стабильно (4.5): якорь на свежем кадре -> rect региона
    в координаты кадра -> read_stable. Якоря нет — OcrReadError-совместимый
    сигнал выше по стеку (попытка потрачена)."""
    frame = np.asarray(screen.grab())
    anchor = find_anchor(frame)
    if anchor is None:
        raise OcrReadError(
            f"OCR региона {region_id!r}: якорь не найден на кадре",
            region_id=region_id, attempts=(),
        )
    rect = region.rect
    frame_region = OcrRegionSpec(
        id=region.id,
        rect=Rect(anchor.x + rect.x, anchor.y + rect.y, rect.width, rect.height),
        grammar=region.grammar,
        placeholder=region.placeholder,
    )
    return read_stable(ocr, lambda: np.asarray(screen.grab()), frame_region)


def _matches(stable: StableReadResult, text: str,
             expected_value) -> bool:
    """Содержимое поля сошлось: стабильный статус ok и значение равно
    ожидаемому (разобранному по грамматике — для enter_price, где OCR может
    вернуть отформатированную цену «1,500,000» вместо введённых «1500000»)."""
    if stable.status != STATUS_OK:
        return False
    if expected_value is not None:
        return stable.value == expected_value
    return stable.value == text


def enter_text(
    text: str,
    *,
    model: UiModel,
    region_id: str,
    ocr,
    screen: Screen,
    hid,
    cursor_pos: Callable[[], Tuple[int, int]],
    activate: Callable[[], ActivateResult],
    window_status: Callable[[np.ndarray], WindowStatus],
    find_anchor: Callable[[np.ndarray], Optional[Anchor]],
    run_checks: CheckRunner,
    config: CoreConfig,
    field_element_id: Optional[str] = None,
    search_button_id: Optional[str] = None,
    numpad_map: Optional[Mapping[str, str]] = None,
    submit: bool = True,
    recover_layout: bool = True,
    expected_value=None,
    key_tap: Optional[Callable[[str], None]] = None,
    switch_layout: Optional[Callable[[object], None]] = None,
    on_fail: Tuple[str, ...] = (),
    free_area_point: Optional[Tuple[int, int]] = None,
    cancelled: Optional[Callable[[], bool]] = None,
    rng: Optional[JitterRng] = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> EnterResult:
    """Ввод текста в поле модели: пролог -> способ ввода (профиль) -> фокус
    -> набор -> OCR verify (read_stable) -> восстановление раскладки ->
    submit -> retry -> on_fail -> сигнал recover. Контракт — в докстринге
    модуля.

    text непуст; region_id — OCR-регион поля ввода (rect от якоря, грамматика
    ожидаемого содержимого). field_element_id/search_button_id — элементы
    модели (фокус поля / кнопка «Поиск»); без field_element_id фокус-клика
    нет (поле сфокусировано вызывающим слоем — честное допущение для полей,
    получающих фокус иначе, чем кликом).
    """
    if not text:
        raise EnterTextError("пустой текст ввода")
    region = model.ocr_regions.get(region_id)
    if region is None:
        raise EnterTextError(
            f"OCR-регион {region_id!r} отсутствует в модели окна "
            f"{model.window.id!r}")
    for role, eid in (("поля ввода", field_element_id),
                      ("кнопки «Поиск»", search_button_id)):
        if eid is not None and eid not in model.elements:
            raise EnterTextError(f"элемент {role} {eid!r} отсутствует в модели")

    rng = rng or random.Random()
    default_key_tap = key_tap is None
    key_tap = key_tap or _default_key_tap(hid)
    switch_layout = switch_layout or _default_switch_layout

    # Способ ввода — распределение профиля (решение 14); маппинг нумпада
    # обязан покрывать все символы текста (семантика клавиш — параметр).
    input_method = (INPUT_NUMPAD
                    if rng.random() < config.behavior.numpad_fraction
                    else INPUT_KEYBOARD)
    if input_method == INPUT_NUMPAD:
        if numpad_map is None:
            raise EnterTextError(
                "выбран экранный нумпад, но numpad_map не задан — маппинг "
                "символов на элементы клавиш обязателен (Q25)")
        unmappable = [ch for ch in text if ch not in numpad_map]
        if unmappable:
            raise EnterTextError(
                f"символы вне numpad_map: {unmappable!r} — семантика "
                f"экранной клавиатуры сырьём не подтверждена (Q25)")
    # Проверка кириллицы/прочих не-ASCII до HID (клавиатурный путь):
    # 1-байтный протокол вендора их не несёт — fail fast, не в середине набора.
    if input_method == INPUT_KEYBOARD and default_key_tap:
        for ch in text:
            _ascii_to_hid(ch)  # не-ASCII fail fast до HID (протокол 1-байтный)
    # Способ отправки — enter_fraction профиля (кнопка «Поиск» только если
    # элемент задан; иначе честно Enter — выбора нет).
    if submit:
        if search_button_id is not None and \
                rng.random() >= config.behavior.enter_fraction:
            submit_method = SUBMIT_BUTTON
        else:
            submit_method = SUBMIT_ENTER
    else:
        submit_method = None

    # --- ПРОЛОГ: activate OK (5.2) + статус found (3.5), иначе ввода нет.
    activated = activate()
    if activated.cancelled:
        return EnterResult(region_id, success=False, attempts=0, cancelled=True,
                           input_method=input_method, submit_method=submit_method)
    status = window_status(np.asarray(screen.grab()))
    if status.status != STATUS_FOUND:
        raise EnterPreconditionError(region_id, status)

    max_attempts = 1 + config.timing.click_retry_count
    failures: list = []
    # Наблюдаемое OCR-содержимое поля: перед повторным набором (retry-попытка
    # или восстановление раскладки) стираем ЕГО — backspace x len(наблюденного),
    # не len(text): OCR мог увидеть меньше набранного, а печать поверх
    # нестёртого содержимого делает verify несходимым принципиально.
    observed_text: Optional[str] = None
    for attempt in range(1, max_attempts + 1):
        if cancelled is not None and cancelled():
            return EnterResult(region_id, success=False, attempts=attempt - 1,
                               cancelled=True, input_method=input_method,
                               submit_method=submit_method)
        try:
            if field_element_id is not None:
                focus = _click_element(
                    field_element_id, model=model, run_checks=run_checks,
                    screen=screen, hid=hid, cursor_pos=cursor_pos,
                    activate=activate, window_status=window_status,
                    find_anchor=find_anchor, config=config, rng=rng,
                    clock=clock, sleep=sleep)
                if not focus.success:
                    failures.append(
                        f"попытка {attempt}: фокус {field_element_id!r} не "
                        f"подтверждён ({'; '.join(focus.failures) or 'присутствие не сошлось'})")
                    continue
            if attempt > 1 and observed_text:
                # Стираем содержимое, оставшееся от предыдущей попытки.
                _erase_observed(hid, observed_text)
            _type(text, input_method=input_method, numpad_map=numpad_map,
                  model=model, run_checks=run_checks, screen=screen, hid=hid,
                  cursor_pos=cursor_pos, activate=activate,
                  window_status=window_status, find_anchor=find_anchor,
                  config=config, rng=rng, key_tap=key_tap, clock=clock,
                  sleep=sleep)
            stable = _read_field(ocr, screen, find_anchor, region, region_id)
            observed_text = stable.text
            layout_recovered = False
            if not _matches(stable, text, expected_value):
                if not recover_layout:
                    failures.append(
                        f"попытка {attempt}: OCR поля {stable.text!r} != "
                        f"ожидаемому {text!r}")
                    continue
                # Восстановление раскладки: стираем наблюдаемое содержимое
                # (backspace x len(stable.text) — очистка поля целиком и
                # select-on-focus сырьём не смоделированы, Q8/Q25),
                # переключаем раскладку, повторяем ввод, проверяем.
                _erase_observed(hid, stable.text)
                switch_layout(hid)
                _type(text, input_method=input_method, numpad_map=numpad_map,
                      model=model, run_checks=run_checks, screen=screen, hid=hid,
                      cursor_pos=cursor_pos, activate=activate,
                      window_status=window_status, find_anchor=find_anchor,
                      config=config, rng=rng, key_tap=key_tap, clock=clock,
                      sleep=sleep)
                stable = _read_field(ocr, screen, find_anchor, region, region_id)
                observed_text = stable.text
                layout_recovered = _matches(stable, text, expected_value)
                if not layout_recovered:
                    failures.append(
                        f"попытка {attempt}: после восстановления раскладки "
                        f"OCR поля {stable.text!r} != ожидаемому {text!r}")
                    continue
            if submit_method == SUBMIT_BUTTON:
                submit_result = _click_element(
                    search_button_id, model=model, run_checks=run_checks,
                    screen=screen, hid=hid, cursor_pos=cursor_pos,
                    activate=activate, window_status=window_status,
                    find_anchor=find_anchor, config=config, rng=rng,
                    clock=clock, sleep=sleep)
                if not submit_result.success:
                    failures.append(
                        f"попытка {attempt}: кнопка «Поиск» {search_button_id!r} "
                        f"не подтверждена ({'; '.join(submit_result.failures)})")
                    continue
            elif submit_method == SUBMIT_ENTER:
                hid.key_press(KEY_ENTER)
            return EnterResult(
                region_id, success=True, attempts=attempt,
                input_method=input_method, submit_method=submit_method,
                layout_recovered=layout_recovered,
            )
        except ClickPreconditionError as exc:
            # Вложенный клик поймал деградацию окна — эскалация наверх
            # единым типом ошибки примитива (инвариант 9).
            raise EnterPreconditionError(region_id, WindowStatus(
                status=exc.status, reason=exc.detail, anchor=None,
                content_loaded=None, content_detail=exc.detail)) from exc
        except OcrReadError as exc:
            failures.append(f"попытка {attempt}: OCR поля не удалось: {exc}")
        except _TypeFailed as exc:
            failures.append(str(exc))

    # Исчерпание retry: повторный пролог перед on_fail — за цикл окно могло
    # деградировать, а on_fail кликает в область экрана (инвариант 9 — без
    # activate OK + found слепые HID-шаги запрещены; при провале on_fail
    # пропускается, результат — сразу needs_recover с причиной).
    reactivated = activate()
    if reactivated.cancelled:
        return EnterResult(region_id, success=False, attempts=max_attempts,
                           cancelled=True, input_method=input_method,
                           submit_method=submit_method)
    status = window_status(np.asarray(screen.grab()))
    if status.status != STATUS_FOUND:
        failures.append(
            f"on_fail пропущен: статус окна {status.status!r} "
            f"({status.reason or status.content_detail}) — слепые HID-шаги "
            f"запрещены")
        return EnterResult(region_id, success=False, attempts=max_attempts,
                           input_method=input_method, submit_method=submit_method,
                           failures=tuple(failures))
    _run_on_fail(on_fail, hid=hid, cursor_pos=cursor_pos,
                 free_area_point=free_area_point, config=config, rng=rng)
    return EnterResult(region_id, success=False, attempts=max_attempts,
                       input_method=input_method, submit_method=submit_method,
                       on_fail_executed=bool(on_fail), failures=tuple(failures))


def _type(text, *, input_method, numpad_map, model, run_checks, screen, hid,
          cursor_pos, activate, window_status, find_anchor, config, rng,
          key_tap, clock, sleep) -> None:
    """Набор текста способом ввода; клавиша нумпада не подтверждена —
    _TypeFailed (попытка потрачена)."""
    if input_method == INPUT_KEYBOARD:
        for ch in text:
            key_tap(ch)
            _fixed_pause(rng.randint(config.behavior.key_delay_min_ms,
                                     config.behavior.key_delay_max_ms) / 1000.0,
                         config=config, cancelled=None, clock=clock, sleep=sleep)
        return
    for ch in text:
        key_id = numpad_map[ch]
        result = _click_element(
            key_id, model=model, run_checks=run_checks, screen=screen, hid=hid,
            cursor_pos=cursor_pos, activate=activate, window_status=window_status,
            find_anchor=find_anchor, config=config, rng=rng, clock=clock,
            sleep=sleep)
        if not result.success:
            raise _TypeFailed(
                f"клавиша {key_id!r} (символ {ch!r}) не подтверждена: "
                f"{'; '.join(result.failures) or 'присутствие не сошлось'}")


def enter_price(
    price: int,
    *,
    model: UiModel,
    region_id: str,
    ocr,
    screen: Screen,
    hid,
    cursor_pos: Callable[[], Tuple[int, int]],
    activate: Callable[[], ActivateResult],
    window_status: Callable[[np.ndarray], WindowStatus],
    find_anchor: Callable[[np.ndarray], Optional[Anchor]],
    run_checks: CheckRunner,
    config: CoreConfig,
    field_element_id: Optional[str] = None,
    search_button_id: Optional[str] = None,
    numpad_map: Optional[Mapping[str, str]] = None,
    submit: bool = True,
    recover_layout: bool = True,
    key_tap: Optional[Callable[[str], None]] = None,
    switch_layout: Optional[Callable[[object], None]] = None,
    on_fail: Tuple[str, ...] = (),
    free_area_point: Optional[Tuple[int, int]] = None,
    cancelled: Optional[Callable[[], bool]] = None,
    rng: Optional[JitterRng] = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> EnterResult:
    """Ввод цены: enter_text со строкой цифр; verify сравнивает значение,
    разобранное грамматикой price региона, с ожидаемой ценой (OCR может
    вернуть отформатированную запись «1,500,000» — не введённый текст)."""
    return enter_text(
        str(price),
        model=model,
        region_id=region_id,
        ocr=ocr,
        screen=screen,
        hid=hid,
        cursor_pos=cursor_pos,
        activate=activate,
        window_status=window_status,
        find_anchor=find_anchor,
        run_checks=run_checks,
        config=config,
        field_element_id=field_element_id,
        search_button_id=search_button_id,
        numpad_map=numpad_map,
        submit=submit,
        recover_layout=recover_layout,
        expected_value=price,
        key_tap=key_tap,
        switch_layout=switch_layout,
        on_fail=on_fail,
        free_area_point=free_area_point,
        cancelled=cancelled,
        rng=rng,
        clock=clock,
        sleep=sleep,
    )
