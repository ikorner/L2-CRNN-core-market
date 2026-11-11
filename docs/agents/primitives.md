# Примитивы действий (задачи 5.1–5.8)

> Детальный справочник по слою `l2core/primitives/`, вынесенный из
> AGENTS.md (прогрессивная загрузка: AGENTS.md — ядро, этот файл — детали
> по слою). Правится вместе с кодом: обновление затронутых заметок
> `docs/agents/` — часть DoD ломающих задач (см. «Правила изменений» в
> AGENTS.md).

## wait_until и запрет sleep (задача 5.1, инвариант 1)

`l2core/primitives/wait.py` — единственная sanctioned-точка блокирующего
ожидания в ядре (инвариант 1: «Никаких sleep — только
wait_until(predicate, timeout)»). Сигнатура:
`wait_until(predicate, *, timeout=None, interval=None, config=None,
cancelled=None, clock=time.monotonic, sleep=time.sleep) -> WaitResult`.

- Подстановка таймингов (договорённость 4.5, защита от рассинхрона
  дефолтов): явные `timeout`/`interval` побеждают; если None — значения
  из переданного `config` (timing.wait_default_timeout /
  timing.verify_interval); литеральных дефолтов в модуле нет, оба отсутствуют
  → ValueError. Подставляет вызывающий слой.
- Семантика: первый poll немедленно, дальше poll каждые `interval`;
  сон перед poll урезан остатком до дедлайна — последний poll не позже
  timeout, дольше не ждёт. Elapsed на таймауте — ровно timeout. `cancelled`
  (точка инъекции kill switch 8.1, сам kill switch не здесь) проверяется
  до первого poll'а и после каждого неуспешного; отмена — результат с
  флагом `cancelled=True`, не исключение.
- Возврат `WaitResult(success, cancelled, polls, elapsed, last_value)` —
  факты и диагностика; WaitResult — не результат действия (нет HID-эффекта),
  в обёртку outcome.py (5.7) не адаптируется, структура не мешает.
- Часы и сон — инъекции (`clock`, `sleep`), в проде time.monotonic /
  time.sleep; тесты на фейковом времени без реального ожидания
  (инвариант 7).

Guard-тест запрета sleep — `tests/test_sleep_contract.py` (по аналогии с
R13/4.2b): AST-проход по l2core/ ищет вызовы `.sleep(...)` (time/asyncio),
`sleep(...)` (from-import) и `.wait(...)` на threading.Event/Condition;
сообщение — файл:строка. Ссылка `time.sleep` без вызова не нарушение
(инъекция sleeper'а). Allowlist явный: `adapters/hid.py` (сон у железа —
перезагрузка Arduino), `adapters/ocr_vendor/`, `adapters/table_segmentation/`
(байт-идентичный вендор), `primitives/wait.py` (sanctioned-точка ожидания).

## activate_game (задача 5.2, spec core-api-events, safety)

`l2core/primitives/activate.py`: активация окна игры перед действием —
поиск окна (GameWindowFinder, критерии из config.window), проверка фокуса
(GetForegroundWindow == hwnd через шлюз), при необходимости SetForeground,
verify через wait_until («фокус получен», тайминги timing из конфига),
retry: попыток = 1 + timing.click_retry_count; исчерпание → ActivateGameError
(reason=focus_not_acquired, диагностика) — сигнал эскалации, recover-цепочка
— 5.8. Окно не найдено/свёрнуто/без клиентской области → ActivateGameError
с reason находилки (3.2), SetForeground не вызывается (инвариант 9). Отмена
через cancelled (kill switch 8.1) → результат с cancelled=True, не исключение.
Протокол Win32Gateway расширен методами `get_foreground_window()` /
`set_foreground(hwnd)` / `get_cursor_pos()` (реализация —
PyWin32Gateway, в тестах fake).

Проверка перекрытия — честный минимум: фокус. Надёжной дешёвой проверки
перекрытия клиентской области другим окном Win32 не даёт; перекрытие
покрывается статус-детектором 3.5 (occluded_or_unstable) и перехватчиком
модалок 3.6 — вне примитива. Контракт «activate перед каждым действием»
оформляется в 5.3+ (click_element), здесь — готовый примитив + сигнал.

## click_element (задача 5.3, spec action-primitives, инварианты 1/2/3/9)

`l2core/primitives/click.py` — оркестратор клика по элементу модели; все
внешние эффекты — инъекции (HID, экран/FrameTransaction 3.6, activate,
статус, поиск якоря, оценка проверок run_checks, rng, часы):
пролог → цикл попыток (1 + timing.click_retry_count) → on_fail[] → сигнал
recover.

- **Пролог (binding ruling ревью 5.2): клик допустим только при
  activate_game OK (5.2) И статусе окна found (3.5).** Иначе — исключение
  (ActivateGameError проброшена / ClickPreconditionError), HID не трогается
  — слепых кликов нет (инвариант 9).
- Verify (инвариант 2) через обязательную типизированную VerifySpec
  (проверки перехода модели, VerifySpec.from_transition) на свежем кадре;
  клики без verify запрещены — параметр обязателен, guard-тест
  `tests/test_click.py::TestVerifyContract` следит за сигнатурой.
- Hover: наведение курсора на click_point (варианта/центра rect, HUD — только
  через инъекцию hud_locator, без него клик невозможен) → hover-пауза
  (timing.hover_pause) → re-verify присутствия элемента (hover мог изменить
  вид/перекрыть) → только потом click.
- Hover-пауза без sleep (инвариант 1): фиксированная задержка через
  wait_until по инъецированным часам с предикатом «прошло время»
  (интервал — timing.verify_interval); результат игнорируется — задержка
  гарантирована механизмом дедлайна.
- Тултип: детекции из кадра нет (модель элемент-тултип не содержит) —
  честная механика «сдвиг курсора на TOOLTIP_SHIFT_PX (24) + пауза +
  повтор в пределах retry», не детекция.
- on_fail[]: «press_esc», «click_free_area» (точка — free_area_point или
  центр config.resolution, честное допущение); перед on_fail пролог
  повторяется (activate + found): иначе click_free_area был бы слепым
  кликом при деградировавшем окне (инвариант 9) — при провале on_fail
  пропускается, сразу needs_recover=True с причиной в failures;
  неизвестный шаг — ошибка модели. Исчерпание retry + on_fail →
  ClickResult(needs_recover=True) — исполнение recover — задача 5.8.
- Джиттер/hold_ms из behavior-профиля через инъецируемый rng (детерминизм
  с seed, инвариант 7) — вендорный random HidDevice.click не используется.
- **Якорь потерян перед verify** (клик закрыл/сменил окно, анимация):
  verify-набор оценивается без якоря (`run_checks(checks, frame, None)` —
  StateDetector оценивает HUD-проверки по search_area клиентской области,
  якорно-зависимые проверки честно падают; пустой verify без якоря ничего
  не подтверждает). Сошёлся — успех; иначе UNKNOWN (5.7), повтор
  необратимого — за стражом. Для ПОТЕНЦИАЛЬНО НЕОБРАТИМЫХ действий
  (start_sell) вызывающая сторона передаёт `allow_anchorless_verify=False`:
  окно обязано остаться, потеря якоря — UNKNOWN со стражем (живой кейс
  2026-11-12: закрытие «Обзора лавок» уходило в UNKNOWN, хотя окно
  закрывалось). Тесты: tests/test_click.py::TestAnchorlessVerifyAfterClose,
  tests/test_states.py::TestEvaluateWithoutAnchor.

## enter_text / enter_price (задача 5.4, spec action-primitives)

`l2core/primitives/enter.py` — ввод текста/цены: пролог (как в 5.3:
activate OK + статус found, иначе EnterPreconditionError/ActivateGameError,
HID не трогается) → выбор способа ввода по профилю → фокус → набор →
OCR verify → восстановление раскладки → submit → retry (1 +
timing.click_retry_count) → on_fail[] → сигнал recover (EnterResult;
трёхзначная обёртка OK/FAILED/UNKNOWN — адаптер outcome.py, 5.7).

- Способ ввода (клавиатура/экранный нумпад) — по behavior.numpad_fraction
  через rng (детерминизм с seed). Клавиатура: символ → HID keycode (usage
  table: цифры 0x1E..0x27, a-z → 0x04.., пробел 0x2C; константа LEFT_CTRL
  0xE0 задана локально — в вендорном hid.py её нет), заглавные — с SHIFT
  (key_down/press/up), пауза key_delay_min/max_ms (wait_until по
  инъецированным часам — инвариант 1). Нумпад: клики по элементам модели
  через click_element (маппинг символов — параметр numpad_map; семантика
  экранной клавиатуры сырьём не подтверждена — KNOWN_GAPS, Q25). Не-ASCII
  (кириллица) вне 1-байтного протокола вендора — fail fast (инъекция key_tap
  для расширения).
- Фокус поля — клик по элементу модели (field_element_id) через
  click_element с присутствие-верификацией; без него фокус-клика нет (поле
  сфокусировано вызывающим слоем).
- Verify содержимого: read_stable (4.5) по OCR-региону поля (rect от якоря
  на свежем кадре) до submit; enter_price сравнивает значение, разобранное
  грамматикой price, с ожидаемой ценой (OCR «1,500,000» == введённые
  «1500000»).
- Восстановление раскладки (spec): несовпадение OCR → backspace ×
  len(наблюденного OCR текста) + switch_layout + повтор ввода + повторная
  OCR-проверка (раз на попытку). Та же семантика «стереть-видимое-и-набрать»
  у retry: перед повторным набором попытки стирают содержимое, видимое OCR
  в конце предыдущей (печать поверх нестёртого делала бы verify
  принципиально несходимым). switch_layout — инъекция, дефолт Ctrl+Shift
  (key_down/press/up), сочетание подтверждается у владельца (Q25); чего OCR
  не видит (placeholder/invalid), стирание не достаёт (очистка поля/
  select-on-focus — Q8/Q25, досъёмка модели).
- Submit: Enter (KEY_ENTER) или кнопка «Поиск» (элемент модели, клик через
  click_element) — выбор по behavior.enter_fraction через rng; submit=False
  — ввод без отправки. Verify submit: присутствие кнопки после клика; у
  Enter наблюдаемого признака в модели нет (список обновился — Q25, задача
  модели/стратегии). OCR verify идёт до submit (после Enter поле может
  очиститься).
- Ограничения (R4, честно паркуется — Q25): семантика нумпада, сочетание
  раскладки, признак результата поиска, кириллица.

## scroll_list / scroll_position (задача 5.5, spec market-scenarios)

`l2core/primitives/scroll.py` — скролл списка и чтение позиции скролла.

- **Механика**: wheel есть в протоколе вендора (`HidDevice.scroll`,
  CMD_SCROLL) — fallback на клики по стрелкам не нужен; единица n = один
  щелчок колеса = HID-команда `scroll(1, ±1, delay)` (вниз +1, вверх −1).
  Подтверждённая владельцем механика игры (Q9): щелчок колеса = 1 строка,
  клик по стрелке = 1 строка; колесо работает над любой частью окна
  с товарами, не только над списком.
- **Пролог** — как в 5.3/5.4: activate OK + статус found, иначе
  ScrollPreconditionError/ActivateGameError, HID не трогается (инвариант 9).
- **Область списка**: элемент-область в модели отсутствует (KNOWN_GAPS) —
  курсор наводится на центр таблицы по table_geometry (середина
  col_seps[0]..[-1] × row_y0..row_y0+max_rows·pitch от якоря).
- **Verify каждой итерации** (инвариант 2): wait_until по свежим кадрам
  (refresh транзакции 3.6) до смены хэша нижней видимой строки ИЛИ признака
  края по направлению; таймаут/интервал — timing.wait_default_timeout /
  verify_interval; sleep нет (инвариант 1). retry = 1 + click_retry_count,
  затем повторный пролог → on_fail[] (те же шаги, что в 5.3) → needs_recover.
- **Признак конца** (Q9, ответ владельца + живой кадр на конце списка):
  стрелки «вверх»/«вниз» границы НЕ дискриминируют — активны всегда, кроме
  случая «нечего листать» (нет предметов или меньше экрана; это и есть
  варианты `prokrutka_vverh/vniz_not_active`, кропы scr_153722, вариант
  записи «active»). Поэтому край определяется ТОЛЬКО сравнением строк
  (bottom_row_hash не меняется между кадрами) — основной механизм, не
  fallback (зафиксировано в KNOWN_GAPS п.8); not_active-стрелки — признак
  пустого/короткого списка, оценка template-проверок через инъекцию
  run_checks (matcher/assets — слой детекции).
- **Позиция**: `scroll_position(frame, ...)` → ScrollPosition{at_start,
  at_end, bottom_row_hash}; хэш — sha1 по обрезанным ячейкам последней
  непустой строки нарезки TableSegmentation (4.3) на константах модели,
  байтовое сравнение между кадрами, не OCR; None — строк нет/окно вне кадра.
- **Результат**: ScrollResult(requested/scrolled/reached_end/position/
  wheels/cancelled/unconfirmed/on_fail_executed/failures), success = всё n
  проскроллено ИЛИ край достигнут; трёхзначная обёртка — outcome.py (5.7).

## drag_item (задача 5.6, spec action-primitives, инварианты 1/2/9)

`l2core/primitives/drag.py` — drag-перенос предметов (в лавку и обратно);
оркестратор по паттерну 5.3-5.5, общие хелперы — из click.py
(CheckRunner/JitterRng/VerifySpec/_fixed_pause/_move_to/_run_on_fail).

- **Граница ответственности**: drag_item принимает ГОТОВЫЕ экранные точки
  `source_point`/`target_point` — координаты ячеек сеток инвентаря/лавки
  генерируются из геометрии сетки вызывающим слоем (контракт design.md:
  «slot: координаты ячейки генерируются из геометрии сетки лавки, не
  размечаются вручную» — 6.2), якорь вызывающий берёт свежий непосредственно
  перед вызовом; drag точки внутри попытки пересчитать не может (курсор уже
  тащит предмет) — сдвиг окна между расчётом точек и drop ловит verify,
  retry со свежими точками — решение вызывающего слоя. Verify — обязательная
  типизированная VerifySpec (guard-тест TestVerifyContract следит за
  сигнатурой), оценка через инъекцию run_checks.
- **HID-drag собран из низкоуровневых примитивов вендора** (готовой команды
  drag в протоколе нет, вендорить нельзя): move_rel к source →
  mouse_down (BTN_LEFT) → удержание → пакетная траектория traj
  ((dx, dy, delay_ms) за точку, исполнение на плате, один пакет CMD_TRAJ)
  → пауза на цели → mouse_up. Механика между press и release — под
  try/finally: кнопка отпускается при любом выходе (отмена, транспортный
  HidError, исключение cancelled()). HidError механики — попытка потрачена
  с причиной в failures (drag обратим, retry безопасен), HidError самого
  release глушится и фиксируется в failures (состояние кнопки неизвестно →
  needs_recover → release_all в recover 5.8); mouse_down намеренно вне try
  (press не подтверждён — release не вызываем, очистка у 5.8).
- **Траектория (behavior-профиль)**: source/target + джиттер
  click_jitter_px; 1..3 промежуточные точки (drag_midpoints_min/max) с
  джиттером move_jitter_px; последний шаг — ровно в target; пауза шага —
  move_step_delay_min/max_ms через rng (детерминизм с seed; порядок
  rng-розыгрышей зафиксирован пинами tests/test_drag.py::planned).
  Число точек сверх ёмкости пакета HID-протокола (TRAJ_MAX_POINTS) —
  DragElementError до HID (fail fast).
- **Пролог** — как 5.3-5.5: activate OK + статус found, иначе
  DragPreconditionError/ActivateGameError, HID не трогается (инвариант 9);
  перед press каждой попытки — свежий якорь, якоря нет — попытка потрачена
  без press.
- **Паузы без sleep (инвариант 1)**: удержание после press (press_delay
  профиля, мс→сек) и пауза на цели перед release (timing.hover_pause) —
  _fixed_pause по инъецированным часам.
- **Обратимость/re-verify (3.6)**: перенос обратим (предмет можно
  перетащить обратно) — re-verify «перед необратимым» применяется к
  start_sell (6.2), НЕ к drag; drag обязан verify ПОСЛЕ release и не мешает
  reverify сценария (транзакция кадра не вложенная: drag открывает свою
  FrameTransaction, reverify start_sell — вне примитива).
- on_fail[] — общая механика 5.3 (press_esc / click_free_area) с повторным
  прологом перед шагами; при деградации окна on_fail пропускается →
  needs_recover=True. Результат DragResult; трёхзначная обёртка —
  outcome.py (5.7).

## Трёхзначный результат и страж необратимых (задача 5.7, spec
action-primitives, инвариант 4)

`l2core/primitives/outcome.py` — адаптер ПОВЕРХ результатов примитивов
5.1–5.6 (сами примитивы не переписывались): `ActionOutcome {OK, FAILED,
UNKNOWN}` (str-Enum, в лог диагностикой) + `PrimitiveResult` (общий
контракт success/cancelled/unconfirmed) + `to_outcome(result)` + страж
`IrreversibleGuard`. CANCELLED в enum не входит: отмена — домен kill
switch (8.1, README-замечание 1.4), не итог действия; `to_outcome`
отменённый результат не пропускает (ValueError, fail fast) — стратегия
ветвит `cancelled` до адаптации.

- **UNKNOWN vs FAILED** (честное различие): FAILED — verify выполнялся и
  провалился либо HID-эффект не применён вовсе (якорь потерян до клика,
  hover re-verify не сошёлся): эффект отсутствует/опровергнут. UNKNOWN —
  HID-эффект, скорее всего, применён (клик ушёл / текст набран / wheel
  послан / drag отпущен / HID-сбой оставил состояние неизвестным), но
  подтвердить невозможно (якорь потерян на свежем кадре ПОСЛЕ действия,
  OCR-чтение поля невозможно после набора). Различие несёт поле
  `unconfirmed: bool = False`, добавленное в ClickResult/EnterResult/
  ScrollResult/DragResult: True выставляется только в точках «эффект ушёл,
  подтверждение невозможно» (клик→verify без якоря; drag: verify без
  якоря, HID-сбой механики/release; enter: OcrReadError/_TypeFailed после
  набора; scroll: wheel ушёл, якоря не было ни на одном poll'ах verify —
  якорь был, но строка не менялась = опровергнутый эффект, FAILED).
  Маппинг: success→OK; unconfirmed→UNKNOWN; иначе→FAILED; cancelled→
  ValueError. Пины — tests/test_outcome.py (27 тестов, включая реальные
  прогоны примитивов на UNKNOWN/FAILED-ветках).
- **Страж необратимых**: спека запрещает ядру повторять необратимое при
  UNKNOWN, «решение за стратегией» = выбор момента повтора и ответственность
  за него. `IrreversibleGuard` делает запрет исполняемым: стратегия
  помечает необратимое идентификатором и обязует себя before/after вокруг
  исполнения (`wrap()` — обёртка с before/after, защита от пропуска).
  `after(id, UNKNOWN, detail)` блокирует повтор; `before(id)` при блоке —
  `IrreversibleRepeatError` (действие не доходит до исполнителя); снятие —
  только явное `confirm_after_unknown(id, detail)` (записывает
  GuardResolution в журнал resolutions; снятие без блока — ValueError).
  OK/FAILED страж не задействуют; `after` принимает только ActionOutcome
  (результат примитива/отмена — TypeError: домены не смешиваются). Страж
  не исполняет действия и не решает за стратегию — только не даёт молча
  повторить то, что могло уже свершиться.

## Общий recover-цепочка (задача 5.8, spec action-primitives, инвариант 9)

`l2core/primitives/recover.py` — оркестратор-исполнитель сигнала
needs_recover примитивов 5.1–5.6 поверх существующих механик (пассивные
шаги — общей механикой on_fail click.py; примитивы не переписываются):

    ESC → клик в свободную область → закрыть окно (если открыто не то) →
    переоткрыть через иконку HUD (1 клик + wait verify открытия) →
    перекалибровка якоря → эскалация (событие + пауза)

- **Не ритуал, а путь к цели**: цель — инъекция `goal_reached(frame)`
  (wiring на детекторах 3.4/3.5); проверяется на входе и после каждого
  шага (wait_until по свежим кадрам) — достигнута, цепочка завершается
  досрочно. Шаги, неприменимые к ситуации (окон нет, открыто целевое,
  HUD не локализован), — пропуск с диагностикой; каждый шаг однократен
  (повторы — сама цепочка). Диагностика по шагам — `steps:
  Tuple[RecoverStep]` (имя/итог/detail).
- **Пассивные шаги первыми**; автоперетаскивание окон запрещено (клик по
  заголовку для подъёма — только по решению стратегии, recover не
  исполняет). Пролог — одна активация окна игры на входе и повторная
  перед каждым активным шагом: без окна игры HID-шаги слепы (инвариант
  9) → фиксация + эскалация. Пролог 5.3–5.5 (статус found целевого
  окна) сознательно не применяется: recover вызывается при деградации,
  целевое окно законно отсутствует.
- **Закрыть окно**: открытое окно — инъекция `current_window(frame)`;
  точку кнопки закрытия даёт `close_point(window_id, frame)` (wiring
  считает от якоря той лавки по модели: market — main_market_close_button,
  shop_sell — close_sell_button); точки нет (shop_buy) — документированный
  fallback: ESC как закрытие. Verify: окно закрылось устойчиво (не то
  окно не видно подряд stability_frames свежих кадров).
- **Переоткрыть рецептом навигации** (menu-quickslot-navigation):
  окно открывает инъекция recipe_runner(open_<target>) — исполнитель
  рецептов (сам верифицирует открытие), затем wait verify цели.
  Неоткрытие → штатный FAILED (фиксация состояния: статус 3.5 +
  capture_state — снимок/модалка) + событие escalation, БЕЗ повторных
  попыток открытия («Лавку нельзя создать в текущей зоне»: предусловия
  за таймаут не меняются).
- **Перекалибровка якоря**: кэша позиции нет (инвариант 3) — шаг =
  подтверждение стабильности: find_anchor целевого окна подряд
  stability_frames свежих кадров, затем финальная проверка цели.
- **Эскалация**: фиксация состояния → событие escalation
  (callback-инъекция on_escalation, push-форма; диспетчер событий — 7.3,
  здесь точка входа) → пауза wait_until по инъекции `resume()`
  (pause_timeout=None — бесконечно до явного продолжения стратегией).
- **release_all**: вызывающий слой передаёт результат примитива-причины
  (`cause`); при needs_recover recover делает best-effort
  hid.release_all() ПЕРВЫМ HID-действием (до пассивных шагов — залипшая
  кнопка из 5.6 исказила бы любой клик); HidError глушится с фиксацией
  в failures.
- **Отмена**: cancelled (kill switch 8.1) проверяется на входе, перед
  шагами и во всех wait_until: результат cancelled=True, событие
  escalation НЕ отправляется. RecoverResult совместим с
  PrimitiveResult (to_outcome: success→OK, иначе→FAILED; unconfirmed
  всегда False — recover сам необратимого эффекта не производит).
  Исполнение сценариями — 6.4; YAGNI: никакого event dispatcher/API/
  kill switch.

## press_key (5.9) и исполнитель рецептов навигации (menu-quickslot-navigation)

press_key (`l2core/primitives/key.py`) — нажатие клавиши с verify и
трёхзначным исходом: пролог активации (без пролога статуса/якоря —
клавиша X открывает «Меню» при закрытом окне торговли), key_press ->
wait_until verify на свежих кадрах, retry из конфига -> on_fail ->
FAILED; HID-сбой -> UNKNOWN (unconfirmed). Verify оценивает инъекция
`evaluate(checks, frame) -> CheckReport` БЕЗ якоря окна: навигационные
признаки — объекты клиентской области. Имена клавиш разрешаются по
словарю `KEYCODES` (`l2core/adapters/keycodes.py` — вне вендоренного
hid.py); неизвестное имя — PressKeyError (fail fast).

Исполнитель рецептов (`l2core/primitives/navigation.py`, run_recipe)
исполняет рецепты открытия окон из навигационной модели
(`ui_model/navigation.yaml`, формат — FORMAT.md): шаги key (делегируются
press_key) и click (паттерн открытия через HUD: точку даёт hud_locator
или якорь «Меню» + rect элемента; иконка не локализована -> попытка
потрачена БЕЗ клика; клик ушёл, verify не сошёлся -> retry; HID-сбой ->
UNKNOWN и прерыв рецепта). Условные пометки шагов: `when`/`skip_if`
оцениваются на текущем кадре до шага — нормализация «Меню» (X — toggle:
open_market не должен закрыть открытое меню, лавки закрывают меню до
клика по quick-slot). Проверки поддерживают отрицание `expect: absent`.
Итог: все шаги успешны -> OK; провалы -> on_fail рецепта -> alternate
(рекурсивно) -> FAILED (needs_recover). Координатных действий в рецептах
нет (инвариант 3). RecipeResult совместим с to_outcome (5.7) по полям
success/cancelled/unconfirmed.

Оценщик проверок навигации — `l2core/detector/navchecks.py`
(build_nav_evaluator): HUD-элементы — template matching по search_area,
элементы меню — от якоря menu_anchor (1:1, инвариант 3), pixel_probe —
точки от якоря меню (цвет модели RGB, кадр BGR, допуск на канал).
search_area задана в клиентской области игры (окно не fullscreen): при
живом клиенте оценщик смещает область на начало клиента
(`client_origin` callable; None — абсолютные координаты, тесты).
То же смещение применяет `hud_locator` сборки wiring и StateDetector
(HUD-проверки сигнатур/verify). Тесты: tests/test_press_key.py,
tests/test_recipes.py, tests/test_navchecks.py.
